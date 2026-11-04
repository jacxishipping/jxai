#include "pilot_io.h"
#include <algorithm>
#include <cstdlib>

namespace moe {

PilotIOEngine::~PilotIOEngine() {
    shutdown();
}

void PilotIOEngine::shutdown() {
#ifdef _WIN32
    for (auto& d : drives_) {
        if (d.overlapped_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(d.overlapped_handle);
            d.overlapped_handle = INVALID_HANDLE_VALUE;
        }
    }
    if (iocp_) {
        CloseHandle(iocp_);
        iocp_ = nullptr;
    }
#else
    for (auto& d : drives_) {
        if (d.fd >= 0) {
            ::close(d.fd);
            d.fd = -1;
        }
    }
#endif
    drives_.clear();
    weighted_drive_table_.clear();
}

bool PilotIOEngine::init(const std::vector<std::string>& model_paths, std::string& err_msg) {
    shutdown();
    if (model_paths.empty()) {
        err_msg = "No model paths provided to PilotIOEngine";
        return false;
    }

    // Also check MODEL_MIRROR environment variable for comma/semicolon-separated mirror paths
    std::vector<std::string> all_paths = model_paths;
    const char* env_mirror = std::getenv("MODEL_MIRROR");
    if (env_mirror && env_mirror[0] != '\0') {
        std::string mirror_str(env_mirror);
        std::stringstream ss(mirror_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty() && std::find(all_paths.begin(), all_paths.end(), item) == all_paths.end()) {
                all_paths.push_back(item);
            }
        }
    }

#ifdef _WIN32
    iocp_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (!iocp_) {
        err_msg = "Failed to create Win32 I/O Completion Port (IOCP)";
        return false;
    }
#endif

    for (size_t idx = 0; idx < all_paths.size(); ++idx) {
        DriveStripeInfo info;
        info.path = all_paths[idx];

#ifdef _WIN32
        info.overlapped_handle = CreateFileA(
            info.path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr
        );
        if (info.overlapped_handle == INVALID_HANDLE_VALUE) {
            if (idx == 0) {
                err_msg = "Failed to open primary model file for IOCP: " + info.path;
                shutdown();
                return false;
            }
            continue;
        }
        if (!CreateIoCompletionPort(info.overlapped_handle, iocp_, static_cast<ULONG_PTR>(idx), 0)) {
            err_msg = "Failed to bind file handle to IOCP: " + info.path;
            shutdown();
            return false;
        }
        bypass_io_supported_ = true;
        direct_dma_active_ = true;
#else
        info.fd = ::open(info.path.c_str(), O_RDONLY);
        if (info.fd < 0) {
            if (idx == 0) {
                err_msg = "Failed to open primary model file for async I/O: " + info.path;
                shutdown();
                return false;
            }
            continue;
        }
#endif
        drives_.push_back(info);
    }

    // Benchmark startup throughput of each drive using a real read probe
    std::vector<uint8_t> probe_buf(64 * 1024);
    double min_bw = 1e18;
    for (size_t i = 0; i < drives_.size(); ++i) {
        double t0 = now_ms();
        for (int iter = 0; iter < 4; ++iter) {
            read_at_offset(0, probe_buf.data(), 4096, static_cast<uint32_t>(i));
        }
        double elapsed_ms = std::max(0.005, now_ms() - t0);
        double bytes_read = 4.0 * 4096.0;
        drives_[i].bandwidth_mb_s = (bytes_read / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
        min_bw = std::min(min_bw, drives_[i].bandwidth_mb_s);
    }

    weighted_drive_table_.clear();
    for (size_t i = 0; i < drives_.size(); ++i) {
        uint32_t ratio = static_cast<uint32_t>(std::clamp(std::round(drives_[i].bandwidth_mb_s / std::max(1.0, min_bw)), 1.0, 8.0));
        drives_[i].weight_ratio = ratio;
        for (uint32_t r = 0; r < ratio; ++r) {
            weighted_drive_table_.push_back(i);
        }
    }

    // Reset counters after startup probe
    total_bytes_streamed_ = 0;
    total_io_ops_ = 0;
    return true;
}

size_t PilotIOEngine::select_drive_index(uint32_t stripe_hint) const {
    if (weighted_drive_table_.empty()) return 0;
    return weighted_drive_table_[stripe_hint % weighted_drive_table_.size()];
}

bool PilotIOEngine::read_at_offset(uint64_t offset, uint8_t* dst, size_t size, uint32_t stripe_hint) {
    if (size == 0 || drives_.empty()) return true;
    size_t d_idx = select_drive_index(stripe_hint);
    const auto& drive = drives_[d_idx];

#ifdef _WIN32
    std::lock_guard<std::mutex> lk(iocp_mtx_);
    OVERLAPPED ov{};
    ov.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    ov.OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFFULL);

    DWORD bytes_read = 0;
    BOOL ok = ReadFile(
        drive.overlapped_handle,
        dst,
        static_cast<DWORD>(size),
        nullptr,
        &ov
    );

    if (!ok && GetLastError() != ERROR_IO_PENDING) {
        return false;
    }

    ULONG_PTR comp_key = 0;
    LPOVERLAPPED completed_ov = nullptr;
    if (!GetQueuedCompletionStatus(iocp_, &bytes_read, &comp_key, &completed_ov, 10000)) {
        return false;
    }
    total_bytes_streamed_ += bytes_read;
    total_io_ops_++;
    return bytes_read == size;
#else
    ssize_t r = ::pread(drive.fd, dst, size, static_cast<off_t>(offset));
    if (r < 0) return false;
    total_bytes_streamed_ += static_cast<uint64_t>(r);
    total_io_ops_++;
    return static_cast<size_t>(r) == size;
#endif
}

StreamedExpertWeights PilotIOEngine::read_expert_sync(
    uint32_t layer,
    uint32_t expert,
    const ExpertDiskSlice& slice
) {
    double t0 = now_ms();
    StreamedExpertWeights out;
    out.layer = layer;
    out.expert = expert;
    out.gate.resize(static_cast<size_t>(slice.gate_bytes));
    out.up.resize(static_cast<size_t>(slice.up_bytes));
    out.down.resize(static_cast<size_t>(slice.down_bytes));

    uint32_t hint = layer * 257u + expert;
    if (slice.gate_bytes > 0) read_at_offset(slice.gate_offset, out.gate.data(), out.gate.size(), hint);
    if (slice.up_bytes > 0)   read_at_offset(slice.up_offset,   out.up.data(),   out.up.size(),   hint + 1);
    if (slice.down_bytes > 0) read_at_offset(slice.down_offset, out.down.data(), out.down.size(), hint + 2);

    out.io_ms = now_ms() - t0;
    return out;
}

std::future<StreamedExpertWeights> PilotIOEngine::prefetch_expert_async(
    uint32_t layer,
    uint32_t expert,
    const ExpertDiskSlice& slice
) {
    return std::async(std::launch::async, [this, layer, expert, slice]() {
        return read_expert_sync(layer, expert, slice);
    });
}

} // namespace moe

