#pragma once

#include "common.h"
#include "tier.h"
#include <future>
#include <queue>
#include <condition_variable>
#include <thread>

namespace moe {

struct ExpertDiskSlice {
    uint64_t gate_offset = 0;
    uint64_t gate_bytes  = 0;
    uint64_t up_offset   = 0;
    uint64_t up_bytes    = 0;
    uint64_t down_offset = 0;
    uint64_t down_bytes  = 0;
};

struct StreamedExpertWeights {
    uint32_t layer = 0;
    uint32_t expert = 0;
    std::vector<uint8_t> gate;
    std::vector<uint8_t> up;
    std::vector<uint8_t> down;
    double io_ms = 0.0;
};

struct DriveStripeInfo {
    std::string path;
    double bandwidth_mb_s = 0.0;
    uint32_t weight_ratio = 1;
#ifdef _WIN32
    HANDLE overlapped_handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

class PilotIOEngine {
public:
    PilotIOEngine() = default;
    ~PilotIOEngine();

    PilotIOEngine(const PilotIOEngine&) = delete;
    PilotIOEngine& operator=(const PilotIOEngine&) = delete;

    // Initialize async I/O handles (Win32 IOCP / POSIX pread) and benchmark MODEL_MIRROR drives
    bool init(const std::vector<std::string>& model_paths, std::string& err_msg);
    void shutdown();

    // Perform raw overlapped disk read at exact byte offset using IOCP (Windows) or pread (POSIX)
    bool read_at_offset(uint64_t offset, uint8_t* dst, size_t size, uint32_t stripe_hint = 0);

    // Synchronously read an expert's 3 weight matrices (gate, up, down) from NVMe disk
    StreamedExpertWeights read_expert_sync(
        uint32_t layer,
        uint32_t expert,
        const ExpertDiskSlice& slice
    );

    // PILOT Prefetch: dispatch asynchronous read for Layer L+1 expert on background I/O worker
    std::future<StreamedExpertWeights> prefetch_expert_async(
        uint32_t layer,
        uint32_t expert,
        const ExpertDiskSlice& slice
    );

    const std::vector<DriveStripeInfo>& drives() const { return drives_; }
    uint64_t total_bytes_streamed() const { return total_bytes_streamed_.load(); }
    uint64_t total_io_ops() const { return total_io_ops_.load(); }
    bool bypass_io_supported() const { return bypass_io_supported_; }
    bool direct_dma_active() const { return direct_dma_active_; }

private:
    size_t select_drive_index(uint32_t stripe_hint) const;

    bool bypass_io_supported_ = false;
    bool direct_dma_active_ = false;

    std::vector<DriveStripeInfo> drives_;
    std::vector<size_t> weighted_drive_table_;
#ifdef _WIN32
    HANDLE iocp_ = nullptr;
    std::mutex iocp_mtx_;
#endif

    std::atomic<uint64_t> total_bytes_streamed_{0};
    std::atomic<uint64_t> total_io_ops_{0};
};

} // namespace moe
