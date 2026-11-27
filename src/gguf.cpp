#include "gguf.h"
#include "quant.h"
#include <algorithm>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#ifndef _WIN32
#include <sys/mman.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace moe {

ModelFile::~ModelFile() {
    close();
}

void ModelFile::close() {
#ifdef _WIN32
    if (mapped_base_) {
        UnmapViewOfFile(mapped_base_);
        mapped_base_ = nullptr;
    }
    if (map_handle_) {
        CloseHandle(map_handle_);
        map_handle_ = nullptr;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
    for (auto& s : extra_shards_) {
        if (s.base) UnmapViewOfFile(s.base);
        if (s.map_handle) CloseHandle(s.map_handle);
        if (s.file_handle != INVALID_HANDLE_VALUE) CloseHandle(s.file_handle);
    }
#else
    if (mapped_base_ && file_size_ > 0) {
        munmap(const_cast<uint8_t*>(mapped_base_), file_size_);
        mapped_base_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    for (auto& s : extra_shards_) {
        if (s.base && s.size > 0) munmap(const_cast<uint8_t*>(s.base), s.size);
        if (s.fd >= 0) ::close(s.fd);
    }
#endif
    extra_shards_.clear();
    packed_buffers_.clear();
    metadata_.clear();
    tensors_.clear();
    tensor_order_.clear();
    alias_.clear();
    config_loaded_ = false;
    detected_arch_.clear();
    config_path_.clear();
    tokenizer_path_.clear();
    st_moe_unsupported_ = false;
    st_untranslated_.clear();
    st_unknown_dtypes_.clear();
    st_shard_count_ = 0;
    kv_end_offset_ = 0;
    file_size_ = 0;
}

namespace {

// Parse "<prefix>-NNNNN-of-MMMMM.safetensors" (e.g.
// "model-00001-of-00002.safetensors"). Returns false for non-shard names.
bool parse_st_shard_name(const std::string& filename, std::string& prefix,
                         uint32_t& index, uint32_t& total) {
    std::string low = filename;
    std::transform(low.begin(), low.end(), low.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const std::string suffix = ".safetensors";
    if (low.size() <= suffix.size() ||
        low.compare(low.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
    std::string stem = filename.substr(0, filename.size() - suffix.size());
    size_t of = stem.rfind("-of-");
    if (of == std::string::npos) return false;
    std::string total_s = stem.substr(of + 4);
    if (total_s.empty() ||
        !std::all_of(total_s.begin(), total_s.end(), ::isdigit)) return false;
    size_t dash = stem.rfind('-', of - 1);
    if (dash == std::string::npos || dash + 1 >= of) return false;
    std::string index_s = stem.substr(dash + 1, of - dash - 1);
    if (index_s.empty() ||
        !std::all_of(index_s.begin(), index_s.end(), ::isdigit)) return false;
    try {
        total = static_cast<uint32_t>(std::stoul(total_s));
        index = static_cast<uint32_t>(std::stoul(index_s));
    } catch (...) {
        return false;
    }
    if (total < 2 || index < 1 || index > total) return false;
    prefix = stem.substr(0, dash);
    return true;
}

// Collect every "<prefix>-NNNNN-of-<total>.safetensors" file in dir, in
// shard order. False with err_msg when the group is incomplete.
bool find_st_shard_group(const std::filesystem::path& dir, const std::string& prefix,
                         uint32_t total, std::vector<std::string>& out,
                         std::string& err_msg) {
    namespace fs = std::filesystem;
    out.clear();
    std::map<uint32_t, std::string> found;
    std::error_code ec;
    if (!fs::is_directory(dir, ec) || ec) {
        err_msg = "Cannot list directory for SafeTensors shards: " + dir.string();
        return false;
    }
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        std::string fn = entry.path().filename().string();
        std::string p; uint32_t idx = 0, tot = 0;
        if (parse_st_shard_name(fn, p, idx, tot) && p == prefix && tot == total) {
            found[idx] = entry.path().string();
        }
    }
    if (found.empty()) {
        err_msg = "SafeTensors shard group '" + prefix + "' not found in " + dir.string();
        return false;
    }
    for (uint32_t i = 1; i <= total; ++i) {
        auto it = found.find(i);
        if (it == found.end()) {
            err_msg = "Incomplete SafeTensors shard group '" + prefix + "': missing part " +
                      std::to_string(i) + " of " + std::to_string(total) + " in " + dir.string();
            return false;
        }
        out.push_back(it->second);
    }
    return true;
}

// A directory's shard group, if any: the model-00001-of-*.safetensors group
// (any total). Returns true with an empty `out` when there is no group
// (caller falls back to a lone model.safetensors); false only on an
// incomplete group.
bool find_st_shard_group_in_dir(const std::string& dir, std::vector<std::string>& out,
                                std::string& err_msg) {
    namespace fs = std::filesystem;
    out.clear();
    std::error_code ec;
    std::string prefix; uint32_t total = 0;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        std::string fn = entry.path().filename().string();
        std::string p; uint32_t idx = 0, tot = 0;
        if (parse_st_shard_name(fn, p, idx, tot) && idx == 1) {
            prefix = p; total = tot; break;
        }
    }
    if (prefix.empty()) return true; // no shard group in this directory
    return find_st_shard_group(dir, prefix, total, out, err_msg);
}

} // namespace

bool ModelFile::open(const std::string& path, std::string& err_msg) {
    close();

    // Multi-file SafeTensors: a directory, or one shard of a
    // model-0000N-of-0000M.safetensors group, loads every shard.
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::is_directory(path, ec) && !ec) {
        std::vector<std::string> st_group;
        if (!find_st_shard_group_in_dir(path, st_group, err_msg)) return false;
        if (!st_group.empty()) return open_safetensors_group(st_group, err_msg);
        fs::path single = fs::path(path) / "model.safetensors";
        if (fs::exists(single, ec) && !ec) return open(single.string(), err_msg);
        err_msg = "No model file found in directory: " + path +
                  " (expected model.safetensors or model-00001-of-*.safetensors)";
        return false;
    }
    {
        std::string prefix; uint32_t sidx = 0, stotal = 0;
        std::string fname = fs::path(path).filename().string();
        if (parse_st_shard_name(fname, prefix, sidx, stotal)) {
            fs::path dir = fs::path(path).parent_path();
            if (dir.empty()) dir = ".";
            std::vector<std::string> st_group;
            if (!find_st_shard_group(dir, prefix, stotal, st_group, err_msg)) return false;
            return open_safetensors_group(st_group, err_msg);
        }
    }

    path_ = path;

#ifdef _WIN32
    file_handle_ = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        err_msg = "Failed to open model file: " + path;
        return false;
    }

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file_handle_, &sz) || sz.QuadPart < 16) {
        err_msg = "Invalid or empty model file: " + path;
        close();
        return false;
    }
    file_size_ = static_cast<uint64_t>(sz.QuadPart);

    map_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map_handle_) {
        err_msg = "CreateFileMapping failed for: " + path;
        close();
        return false;
    }

    mapped_base_ = static_cast<const uint8_t*>(MapViewOfFile(map_handle_, FILE_MAP_READ, 0, 0, 0));
    if (!mapped_base_) {
        err_msg = "MapViewOfFile failed for: " + path;
        close();
        return false;
    }
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        err_msg = "Failed to open model file: " + path;
        return false;
    }
    struct stat st;
    if (fstat(fd_, &st) != 0 || st.st_size < 16) {
        err_msg = "Invalid or empty model file: " + path;
        close();
        return false;
    }
    file_size_ = static_cast<uint64_t>(st.st_size);
    void* ptr = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr == MAP_FAILED) {
        err_msg = "mmap failed for: " + path;
        close();
        return false;
    }
    mapped_base_ = static_cast<const uint8_t*>(ptr);
#endif

    // Warm the mapping so the first chat tokens do not stall on a burst of
    // major page faults (smooth TTFT). Best-effort on every platform.
    if (mapped_base_ && file_size_ > 0) {
#ifdef _WIN32
        // PrefetchVirtualMemory (Win8+) — resolved dynamically so we do not
        // hard-depend on a newer SDK. Ignore failure.
        {
            struct WinMemRange { void* VirtualAddress; SIZE_T NumberOfBytes; };
            using PrefetchFn = BOOL (WINAPI*)(HANDLE, ULONG_PTR, WinMemRange*, ULONG);
            HMODULE k32 = ::GetModuleHandleA("kernel32.dll");
            auto prefetch = k32
                ? reinterpret_cast<PrefetchFn>(::GetProcAddress(k32, "PrefetchVirtualMemory"))
                : nullptr;
            if (prefetch) {
                WinMemRange range{};
                range.VirtualAddress = const_cast<uint8_t*>(mapped_base_);
                range.NumberOfBytes = static_cast<SIZE_T>(file_size_);
                (void)prefetch(::GetCurrentProcess(), 1, &range, 0);
            }
        }
#else
        (void)madvise(const_cast<uint8_t*>(mapped_base_), static_cast<size_t>(file_size_), MADV_WILLNEED);
#ifdef MADV_HUGEPAGE
        (void)madvise(const_cast<uint8_t*>(mapped_base_), static_cast<size_t>(file_size_), MADV_HUGEPAGE);
#endif
#endif
        // Async prefetch still leaves large GGUFs (≥512 MiB) stuttering on
        // first decode. Synchronously touch every 16th page (64 KiB stride)
        // so weights are resident before the first forward.
        constexpr uint64_t kPrefaultMinBytes = 512ull << 20;
        if (file_size_ >= kPrefaultMinBytes) {
            constexpr size_t kPage = 4096;
            constexpr size_t kStride = kPage * 16;
            const size_t n = static_cast<size_t>(file_size_);
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (long long off = 0; off < static_cast<long long>(n); off += static_cast<long long>(kStride)) {
                volatile uint8_t c = mapped_base_[static_cast<size_t>(off)];
                (void)c;
            }
        }
    }

    // Check magic bytes for GGUF ("GGUF" = 0x46554747)
    if (std::memcmp(mapped_base_, "GGUF", 4) == 0) {
        format_ = "GGUFv3";
        return parse_gguf(err_msg);
    } else {
        format_ = "SafeTensors";
        uint64_t data_off = 0;
        if (!parse_safetensors_shard(mapped_base_, file_size_, data_off, err_msg)) return false;
        data_offset_ = data_off;
        st_shard_count_ = 1;
        // Best-effort only: a missing/unparseable config.json must not fail
        // the open — the engine then falls back to its previous defaults
        // (and the caller can check config_loaded() / detected_arch()).
        load_safetensors_config();
        pack_safetensors_experts();
        return true;
    }
}

// Open every shard of a multi-file SafeTensors model: mmap + parse each
// shard's header into the one shared tensor table, then load the sibling
// config.json and pack per-expert tensors (both need all shards present).
bool ModelFile::open_safetensors_group(const std::vector<std::string>& shard_paths,
                                       std::string& err_msg) {
    close();
    if (shard_paths.empty()) {
        err_msg = "No SafeTensors shards given";
        return false;
    }
    path_ = shard_paths.front();
    format_ = "SafeTensors";
    st_shard_count_ = static_cast<uint32_t>(shard_paths.size());

    for (size_t i = 0; i < shard_paths.size(); ++i) {
        const std::string& sp = shard_paths[i];
#ifdef _WIN32
        HANDLE fh = CreateFileA(sp.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fh == INVALID_HANDLE_VALUE) {
            err_msg = "Failed to open SafeTensors shard: " + sp;
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(fh, &sz) || sz.QuadPart < 16) {
            CloseHandle(fh);
            err_msg = "Invalid or empty SafeTensors shard: " + sp;
            return false;
        }
        HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mh) {
            CloseHandle(fh);
            err_msg = "CreateFileMapping failed for SafeTensors shard: " + sp;
            return false;
        }
        const uint8_t* base = static_cast<const uint8_t*>(MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0));
        if (!base) {
            CloseHandle(mh);
            CloseHandle(fh);
            err_msg = "MapViewOfFile failed for SafeTensors shard: " + sp;
            return false;
        }
        uint64_t size = static_cast<uint64_t>(sz.QuadPart);
        if (i == 0) {
            file_handle_ = fh; map_handle_ = mh;
            mapped_base_ = base; file_size_ = size;
        } else {
            ShardMapping m;
            m.file_handle = fh; m.map_handle = mh; m.base = base; m.size = size;
            extra_shards_.push_back(m);
        }
#else
        int fd = ::open(sp.c_str(), O_RDONLY);
        if (fd < 0) {
            err_msg = "Failed to open SafeTensors shard: " + sp;
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 16) {
            ::close(fd);
            err_msg = "Invalid or empty SafeTensors shard: " + sp;
            return false;
        }
        uint64_t size = static_cast<uint64_t>(st.st_size);
        void* ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) {
            ::close(fd);
            err_msg = "mmap failed for SafeTensors shard: " + sp;
            return false;
        }
        const uint8_t* base = static_cast<const uint8_t*>(ptr);
        if (i == 0) {
            fd_ = fd; mapped_base_ = base; file_size_ = size;
        } else {
            ShardMapping m;
            m.fd = fd; m.base = base; m.size = size;
            extra_shards_.push_back(m);
        }
#endif
        if (std::memcmp(base, "GGUF", 4) == 0) {
            err_msg = "Shard looks like GGUF, not SafeTensors: " + sp +
                      " — merge GGUF shards with `moe_engine convert --merge` instead";
            return false;
        }
        uint64_t data_off = 0;
        if (!parse_safetensors_shard(base, size, data_off, err_msg)) return false;
        if (i == 0) data_offset_ = data_off;
    }

    load_safetensors_config();
    pack_safetensors_experts();
    std::cout << "[SAFETENSORS] Loaded " << shard_paths.size() << " shards ("
              << tensors_.size() << " tensors total).\n";
    return true;
}

namespace {

struct BinaryCursor {
    const uint8_t* base;
    uint64_t size;
    uint64_t pos = 0;

    bool can_read(uint64_t n) const { return pos + n <= size; }

    template <typename T>
    bool read(T& out) {
        if (!can_read(sizeof(T))) return false;
        std::memcpy(&out, base + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }

    bool read_string(std::string& out) {
        uint64_t len = 0;
        if (!read(len)) return false;
        if (!can_read(len)) return false;
        out.assign(reinterpret_cast<const char*>(base + pos), static_cast<size_t>(len));
        pos += len;
        return true;
    }
};

bool is_expert_tensor_name(const std::string& name) {
    return name.find("ffn_gate_exps") != std::string::npos ||
           name.find("ffn_up_exps")   != std::string::npos ||
           name.find("ffn_down_exps") != std::string::npos ||
           name.find(".experts.")     != std::string::npos;
}

// --- Malformed-file defenses (workstream C, 2026-11-21) -----------------
// GGUF/SafeTensors headers are attacker-influenced bytes: every count,
// dimension, offset and length below is validated with checked arithmetic
// before it is used, so a corrupt file fails with a clear error instead of
// crashing (allocation bomb, overflowed offset) or silently producing
// wrong output (unknown tensor type read as F32).

// True when `raw` is a GGMLType value this engine knows. Unknown values
// (e.g. 999, or the unassigned ids 4/5/32-35) are rejected at parse time:
// tensor_byte_size() would otherwise silently treat them as F32-sized
// (blck 1, 4 bytes) and the dequantizer would return zeros.
bool is_known_ggml_type(uint32_t raw) {
    switch (static_cast<GGMLType>(raw)) {
        case GGMLType::F32: case GGMLType::F16:
        case GGMLType::Q4_0: case GGMLType::Q4_1:
        case GGMLType::Q5_0: case GGMLType::Q5_1:
        case GGMLType::Q8_0: case GGMLType::Q8_1:
        case GGMLType::Q2_K: case GGMLType::Q3_K:
        case GGMLType::Q4_K: case GGMLType::Q5_K:
        case GGMLType::Q6_K: case GGMLType::Q8_K:
        case GGMLType::IQ2_XXS: case GGMLType::IQ2_XS:
        case GGMLType::IQ3_XXS: case GGMLType::IQ1_S:
        case GGMLType::IQ4_NL: case GGMLType::IQ3_S:
        case GGMLType::IQ2_S: case GGMLType::IQ4_XS:
        case GGMLType::I8: case GGMLType::I16:
        case GGMLType::I32: case GGMLType::I64:
        case GGMLType::F64: case GGMLType::IQ1_M:
        case GGMLType::BF16: case GGMLType::FP8_E4M3:
        case GGMLType::IQ2_M:
            return true;
        default:
            return false;
    }
}

// Strict unsigned decimal parse: no exceptions, no sign, no trailing junk,
// no overflow. (std::stoull throws out_of_range on huge inputs and wraps
// a leading '-' — both crash paths in the old SafeTensors parser.)
bool parse_u64_strict(const std::string& s, uint64_t& out) {
    if (s.empty()) return false;
    const char* b = s.data();
    const char* e = b + s.size();
    uint64_t v = 0;
    auto res = std::from_chars(b, e, v);
    if (res.ec != std::errc() || res.ptr != e) return false;
    out = v;
    return true;
}

// Hard caps for header counts. A real model has < 10k tensors and < 10k
// KV entries; the caps only stop an allocation bomb from a corrupt header
// (reserve() on a 2^60 count would throw / OOM before any byte is read).
constexpr uint64_t kMaxGgufTensors = 1000000ULL;
constexpr uint64_t kMaxGgufKv = 1000000ULL;
constexpr uint64_t kMaxStHeaderBytes = 256ULL * 1024 * 1024;
constexpr uint32_t kMaxGgufAlignment = 1u << 20;

// ---------------------------------------------------------------------------
// SafeTensors helpers: minimal flat-JSON field extraction (no JSON library),
// HF architecture mapping, and HF -> GGUF tensor-name translation.
// ---------------------------------------------------------------------------

// First string value for a flat JSON key, e.g. "torch_dtype": "bfloat16".
bool cfg_get_string(const std::string& json, const std::string& key, std::string& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t colon = json.find(':', k + pat.size());
    if (colon == std::string::npos) return false;
    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return false;
    size_t i = q1 + 1;
    out.clear();
    while (i < json.size() && json[i] != '"') {
        if (json[i] == '\\' && i + 1 < json.size()) ++i;
        out.push_back(json[i++]);
    }
    return i < json.size();
}

bool cfg_get_number(const std::string& json, const std::string& key, double& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t colon = json.find(':', k + pat.size());
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) ++i;
    size_t start = i;
    while (i < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[i])) || json[i] == '-' ||
            json[i] == '+' || json[i] == '.' || json[i] == 'e' || json[i] == 'E')) ++i;
    if (i == start) return false;
    try { out = std::stod(json.substr(start, i - start)); } catch (...) { return false; }
    return true;
}

bool cfg_get_bool(const std::string& json, const std::string& key, bool& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t colon = json.find(':', k + pat.size());
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) ++i;
    std::string rest = json.substr(i, 5);
    if (rest.rfind("true", 0) == 0) { out = true; return true; }
    if (rest.rfind("false", 0) == 0) { out = false; return true; }
    return false;
}

// First string element of an array value, e.g. "architectures": ["Qwen2..."].
bool cfg_get_first_array_string(const std::string& json, const std::string& key, std::string& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t br = json.find('[', k + pat.size());
    if (br == std::string::npos) return false;
    size_t q1 = json.find('"', br + 1);
    if (q1 == std::string::npos) return false;
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    out = json.substr(q1 + 1, q2 - q1 - 1);
    return true;
}

// Extract the balanced {...} object for a top-level JSON key, e.g.
// "text_config": {...}. Used for multimodal configs (Gemma3) where the
// language-model hparams live in a nested object. Returns false when the
// key is absent or the braces do not balance.
bool cfg_subobject(const std::string& json, const std::string& key, std::string& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t colon = json.find(':', k + pat.size());
    if (colon == std::string::npos) return false;
    size_t open = json.find('{', colon + 1);
    if (open == std::string::npos) return false;
    int depth = 0;
    bool in_str = false;
    for (size_t i = open; i < json.size(); ++i) {
        char c = json[i];
        if (in_str) {
            if (c == '\\') { ++i; continue; }
            if (c == '"') in_str = false;
        } else if (c == '"') {
            in_str = true;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}') {
            if (--depth == 0) {
                out = json.substr(open, i - open + 1);
                return true;
            }
        }
    }
    return false;
}

// HF transformers class name -> GGUF architecture string used by the engine
// (general.architecture, and the "<arch>.*" metadata prefix).
std::string hf_arch_to_gguf(const std::string& hf_class) {
    static const std::pair<const char*, const char*> kArch[] = {
        {"Qwen2ForCausalLM", "qwen2"},   {"Qwen2MoeForCausalLM", "qwen2moe"},
        {"Qwen3ForCausalLM", "qwen3"},   {"Qwen3MoeForCausalLM", "qwen3moe"},
        {"LlamaForCausalLM", "llama"},   {"MistralForCausalLM", "mistral"},
        {"MixtralForCausalLM", "mixtral"},
        {"GemmaForCausalLM", "gemma"},   {"Gemma2ForCausalLM", "gemma2"},
        {"Gemma3ForCausalLM", "gemma3"},
        // Multimodal Gemma3 checkpoints use this class in config.json.
        {"Gemma3ForConditionalGeneration", "gemma3"},
        {"PhiForCausalLM", "phi"},       {"Phi3ForCausalLM", "phi3"},
        {"Phi3SmallForCausalLM", "phi3"},
        {"DeepseekForCausalLM", "deepseek"},
        {"DeepseekV2ForCausalLM", "deepseek2"}, {"DeepseekV3ForCausalLM", "deepseek2"},
        {"FalconForCausalLM", "falcon"}, {"GPTNeoXForCausalLM", "gptneox"},
        {"GPTJForCausalLM", "gptj"},     {"OlmoForCausalLM", "olmo"},
        {"OlmoeForCausalLM", "olmoe"},   {"ChatGLMForCausalLM", "glm"},
        {"GLMForCausalLM", "glm"},       {"InternLM2ForCausalLM", "internlm2"},
        {"StableLmForCausalLM", "stablelm"},
        {"CohereForCausalLM", "cohere"}, {"DbrxForCausalLM", "dbrx"},
    };
    for (const auto& [hf, gguf] : kArch) {
        if (hf_class == hf) return gguf;
    }
    return "";
}

bool is_known_gguf_arch(const std::string& arch) {
    static const char* kKnown[] = {
        "qwen2", "qwen2moe", "qwen3", "qwen3moe", "llama", "mistral", "mixtral",
        "gemma", "gemma2", "gemma3", "gemma4", "phi", "phi3", "deepseek",
        "deepseek2", "falcon", "gptneox", "gptj", "olmo", "olmoe", "glm",
        "internlm2", "stablelm", "cohere", "dbrx",
    };
    for (const char* k : kKnown) if (arch == k) return true;
    return false;
}

std::vector<std::string> split_dots(const std::string& s) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        size_t dot = s.find('.', start);
        if (dot == std::string::npos) { parts.push_back(s.substr(start)); break; }
        parts.push_back(s.substr(start, dot - start));
        start = dot + 1;
    }
    return parts;
}

// Parse a per-expert SafeTensors tensor name into (layer, expert, kind)
// with kind 0 = gate, 1 = up, 2 = down. Matches:
//   <...>.layers.<L>.mlp.experts.<E>.gate_proj|up_proj|down_proj.weight
//   <...>.layers.<L>.block_sparse_moe.experts.<E>.w1|w3|w2.weight
// (Mixtral: w1 = gate, w3 = up, w2 = down.) Returns false otherwise.
bool parse_expert_tensor_name(const std::string& name, uint32_t& layer,
                              uint32_t& expert, int& kind) {
    auto all_digits = [](const std::string& s) {
        return !s.empty() &&
               std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
    };
    std::vector<std::string> p = split_dots(name);
    for (size_t i = 0; i + 6 < p.size(); ++i) {
        if (p[i] != "layers") continue;
        if (!all_digits(p[i + 1]) || !all_digits(p[i + 4])) continue;
        if (p[i + 6] != "weight") continue;
        uint32_t l = 0, e = 0;
        try {
            l = static_cast<uint32_t>(std::stoul(p[i + 1]));
            e = static_cast<uint32_t>(std::stoul(p[i + 4]));
        } catch (...) {
            continue;
        }
        int k = -1;
        if (p[i + 2] == "mlp" && p[i + 3] == "experts") {
            if (p[i + 5] == "gate_proj") k = 0;
            else if (p[i + 5] == "up_proj") k = 1;
            else if (p[i + 5] == "down_proj") k = 2;
            else continue;
        } else if (p[i + 2] == "block_sparse_moe" && p[i + 3] == "experts") {
            if (p[i + 5] == "w1") k = 0;
            else if (p[i + 5] == "w3") k = 1;
            else if (p[i + 5] == "w2") k = 2;
            else continue;
        } else {
            continue;
        }
        layer = l; expert = e; kind = k;
        return true;
    }
    return false;
}

// Parse an already-packed 3D MoE expert tensor name into (layer, which):
//   <...>.layers.<L>.mlp.experts.gate_up_proj[.weight]       -> which 0
//   <...>.layers.<L>.mlp.experts.down_proj[.weight]           -> which 1
//   <...>.layers.<L>.mlp.experts.gate_proj[.weight] (3D)      -> which 2
//   <...>.layers.<L>.mlp.experts.up_proj[.weight] (3D)        -> which 3
// (same under block_sparse_moe). This is the layout transformers 5.x
// writes for MoE checkpoints (e.g. the hf-internal-testing tiny-random
// MoE models): gate_up_proj is [E, 2*I, H] (gate rows first, then up),
// down_proj is [E, H, I]. Per-expert 2D tensors do NOT match (their 5th
// segment is a digit), so the two formats never collide.
bool parse_packed_expert_name(const std::string& name, uint32_t& layer, int& which) {
    auto all_digits = [](const std::string& s) {
        return !s.empty() &&
               std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
    };
    std::vector<std::string> p = split_dots(name);
    for (size_t i = 0; i + 4 < p.size(); ++i) {
        if (p[i] != "layers") continue;
        if (!all_digits(p[i + 1])) continue;
        if (p[i + 2] != "mlp" && p[i + 2] != "block_sparse_moe") continue;
        if (p[i + 3] != "experts") continue;
        if (p[i + 4] == "gate_up_proj" &&
            (p.size() == i + 5 || (p.size() == i + 6 && p[i + 5] == "weight"))) {
            which = 0;
        } else if (p[i + 4] == "down_proj" &&
                   (p.size() == i + 5 || (p.size() == i + 6 && p[i + 5] == "weight"))) {
            which = 1;
        } else if (p[i + 4] == "gate_proj" &&
                   (p.size() == i + 5 || (p.size() == i + 6 && p[i + 5] == "weight"))) {
            which = 2;
        } else if (p[i + 4] == "up_proj" &&
                   (p.size() == i + 5 || (p.size() == i + 6 && p[i + 5] == "weight"))) {
            which = 3;
        } else {
            continue;
        }
        try {
            layer = static_cast<uint32_t>(std::stoul(p[i + 1]));
        } catch (...) {
            continue;
        }
        return true;
    }
    return false;
}

// Translate one HF (transformers) tensor name to its GGUF equivalent.
// Returns "" when there is no translation (unknown pattern, MoE experts,
// fused projections). The caller records those in st_untranslated_.
std::string translate_hf_tensor_name(const std::string& hf_name) {
    std::vector<std::string> p = split_dots(hf_name);
    if (!p.empty() && p[0] == "model") p.erase(p.begin());
    if (p.size() == 2 && p[0] == "embed_tokens" && p[1] == "weight") return "token_embd.weight";
    if (p.size() == 2 && p[0] == "norm" && p[1] == "weight") return "output_norm.weight";
    if (p.size() == 2 && p[0] == "lm_head" && p[1] == "weight") return "output.weight";
    if (p.size() < 4 || p[0] != "layers") return "";
    const std::string& layer = p[1];
    const std::string blk = "blk." + layer + ".";
    if (p[2] == "input_layernorm" && p[3] == "weight") return blk + "attn_norm.weight";
    if (p[2] == "post_attention_layernorm" && p[3] == "weight") return blk + "ffn_norm.weight";
    if (p[2] == "self_attn" && p.size() >= 4) {
        std::string suffix = (p.size() >= 5) ? p[4] : "weight"; // "weight" or "bias"
        if (suffix != "weight" && suffix != "bias") return "";
        if (p[3] == "q_proj") return blk + "attn_q." + suffix;
        if (p[3] == "k_proj") return blk + "attn_k." + suffix;
        if (p[3] == "v_proj") return blk + "attn_v." + suffix;
        if (p[3] == "o_proj") return blk + "attn_output." + suffix;
        if (p[3] == "q_norm") return blk + "attn_q_norm.weight";
        if (p[3] == "k_norm") return blk + "attn_k_norm.weight";
        // Fused QKV (Phi3 qkv_proj): aliased as one tensor and split on
        // demand by ModelFile::split_fused_qkv() — never faked as Q, K or V.
        if (p[3] == "qkv_proj") return blk + "attn_qkv." + suffix;
        return ""; // anything else: no translation
    }
    if (p[2] == "mlp" && p.size() >= 4) {
        if (p[3] == "gate_proj" && p.size() >= 5 && p[4] == "weight") return blk + "ffn_gate.weight";
        if (p[3] == "up_proj" && p.size() >= 5 && p[4] == "weight") return blk + "ffn_up.weight";
        if (p[3] == "down_proj" && p.size() >= 5 && p[4] == "weight") return blk + "ffn_down.weight";
        // Fused gate_up_proj (Phi-3 dense MLP): a single [2*intermediate,
        // hidden] tensor whose rows [0:intermediate] are the gate and
        // [intermediate:2*intermediate] are the up projection. Mapped to
        // ffn_up.weight; bind_tensors() splits it into the two halves
        // (see the fused gate_up branch there). Without this translation
        // the tensor stays untranslated, no dense MLP is bound, the layer
        // keeps the default n_experts=8 with no backing tensors, and the
        // FFN contributes zero (model generates only <unk>).
        // NOTE: this is the DENSE mlp.gate_up_proj only (p[3] ==
        // "gate_up_proj"); MoE experts use mlp.experts.<i>.gate_up_proj
        // (p[3] == "experts"), which is packed separately and never
        // reaches this branch.
        if (p[3] == "gate_up_proj" && p.size() >= 5 && p[4] == "weight") return blk + "ffn_up.weight";
        // MoE router (Qwen2/3-MoE, DeepSeek): without it the engine routes
        // every token to expert 0 (see the router_gate fallback in
        // forward). Translated like any dense tensor.
        if (p[3] == "gate" && p.size() >= 5 && p[4] == "weight") return blk + "ffn_gate_inp.weight";
        // Qwen2.5-MoE / DeepSeek shared expert (dense tensors, translatable).
        if (p[3] == "shared_expert" && p.size() >= 6 && p[5] == "weight") {
            if (p[4] == "gate_proj") return blk + "ffn_gate_shexp.weight";
            if (p[4] == "up_proj") return blk + "ffn_up_shexp.weight";
            if (p[4] == "down_proj") return blk + "ffn_down_shexp.weight";
        }
        // Shared-expert gate (Qwen2MoE shared_expert_gate.weight, [1, dim]):
        // the scalar gate for the shared expert (ffn_gate_inp_shexp).
        if (p[3] == "shared_expert_gate" && p.size() >= 5 && p[4] == "weight") {
            return blk + "ffn_gate_inp_shexp.weight";
        }
        return ""; // routed experts ("mlp.experts.<i>.*") and the rest
    }
    if (p[2] == "block_sparse_moe" && p.size() >= 4) {
        // Mixtral router. Per-expert w1/w2/w3 have no per-tensor translation
        // (packed later by pack_safetensors_experts()).
        if (p[3] == "gate" && p.size() >= 5 && p[4] == "weight") return blk + "ffn_gate_inp.weight";
        return "";
    }
    return "";
}

bool read_gguf_value(BinaryCursor& cur, GGUFValueType vtype, GGUFValue& val) {
    val.type = vtype;
    switch (vtype) {
        case GGUFValueType::UINT8: {
            uint8_t v; if (!cur.read(v)) return false;
            val.u64 = v; val.i64 = v; return true;
        }
        case GGUFValueType::INT8: {
            int8_t v; if (!cur.read(v)) return false;
            val.i64 = v; val.u64 = static_cast<uint64_t>(v); return true;
        }
        case GGUFValueType::UINT16: {
            uint16_t v; if (!cur.read(v)) return false;
            val.u64 = v; val.i64 = v; return true;
        }
        case GGUFValueType::INT16: {
            int16_t v; if (!cur.read(v)) return false;
            val.i64 = v; val.u64 = static_cast<uint64_t>(v); return true;
        }
        case GGUFValueType::UINT32: {
            uint32_t v; if (!cur.read(v)) return false;
            val.u64 = v; val.i64 = v; return true;
        }
        case GGUFValueType::INT32: {
            int32_t v; if (!cur.read(v)) return false;
            val.i64 = v; val.u64 = static_cast<uint64_t>(v); return true;
        }
        case GGUFValueType::FLOAT32: {
            float v; if (!cur.read(v)) return false;
            val.f64 = v; return true;
        }
        case GGUFValueType::BOOL: {
            uint8_t v; if (!cur.read(v)) return false;
            val.bval = (v != 0); val.u64 = val.bval ? 1 : 0; return true;
        }
        case GGUFValueType::STRING: {
            return cur.read_string(val.str);
        }
        case GGUFValueType::UINT64: {
            uint64_t v; if (!cur.read(v)) return false;
            val.u64 = v; val.i64 = static_cast<int64_t>(v); return true;
        }
        case GGUFValueType::INT64: {
            int64_t v; if (!cur.read(v)) return false;
            val.i64 = v; val.u64 = static_cast<uint64_t>(v); return true;
        }
        case GGUFValueType::FLOAT64: {
            double v; if (!cur.read(v)) return false;
            val.f64 = v; return true;
        }
        case GGUFValueType::ARRAY: {
            uint32_t elem_type_raw = 0;
            uint64_t arr_len = 0;
            if (!cur.read(elem_type_raw) || !cur.read(arr_len)) return false;
            if (arr_len > 10000000ULL) return false;
            // Fixed-size element types must fit in the remaining bytes;
            // otherwise the per-element reads below would fail one by one
            // after a huge resize (10M strings ~= 320 MB) — fail fast.
            {
                size_t elem_sz = 0;
                switch (static_cast<GGUFValueType>(elem_type_raw)) {
                    case GGUFValueType::UINT8: case GGUFValueType::INT8:
                    case GGUFValueType::BOOL: elem_sz = 1; break;
                    case GGUFValueType::UINT16: case GGUFValueType::INT16:
                        elem_sz = 2; break;
                    case GGUFValueType::UINT32: case GGUFValueType::INT32:
                    case GGUFValueType::FLOAT32: elem_sz = 4; break;
                    case GGUFValueType::UINT64: case GGUFValueType::INT64:
                    case GGUFValueType::FLOAT64: elem_sz = 8; break;
                    default: elem_sz = 0; break; // STRING / nested: checked per element
                }
                if (elem_sz > 0) {
                    uint64_t remain = (cur.size > cur.pos) ? (cur.size - cur.pos) : 0;
                    if (arr_len > remain / elem_sz) return false;
                }
            }
            auto etype = static_cast<GGUFValueType>(elem_type_raw);
            if (etype == GGUFValueType::STRING) {
                val.str_arr.resize(static_cast<size_t>(arr_len));
                for (uint64_t i = 0; i < arr_len; ++i) {
                    if (!cur.read_string(val.str_arr[static_cast<size_t>(i)])) return false;
                }
            } else if (etype == GGUFValueType::FLOAT32) {
                val.f32_arr.resize(static_cast<size_t>(arr_len));
                for (uint64_t i = 0; i < arr_len; ++i) {
                    if (!cur.read(val.f32_arr[static_cast<size_t>(i)])) return false;
                }
            } else if (etype == GGUFValueType::INT32 || etype == GGUFValueType::UINT32) {
                val.i32_arr.resize(static_cast<size_t>(arr_len));
                for (uint64_t i = 0; i < arr_len; ++i) {
                    if (!cur.read(val.i32_arr[static_cast<size_t>(i)])) return false;
                }
            } else if (etype == GGUFValueType::BOOL || etype == GGUFValueType::UINT8 || etype == GGUFValueType::INT8) {
                val.i32_arr.resize(static_cast<size_t>(arr_len));
                for (uint64_t i = 0; i < arr_len; ++i) {
                    uint8_t b = 0;
                    if (!cur.read(b)) return false;
                    val.i32_arr[static_cast<size_t>(i)] = static_cast<int32_t>(b);
                }
            } else {
                for (uint64_t i = 0; i < arr_len; ++i) {
                    GGUFValue dummy;
                    if (!read_gguf_value(cur, etype, dummy)) return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

} // namespace

// Checked version of tensor_byte_size(): false on overflow instead of
// wrapping. (tensor_byte_size itself is in common.h and cannot check.)
// Moved out of the anonymous namespace in workstream C so the transformer
// layer can validate bound tensors with the same arithmetic the parsers
// use; declared in gguf.h.
bool checked_tensor_bytes(GGMLType t, uint64_t num_elements, uint64_t& out) {
    uint64_t blck = static_cast<uint64_t>(ggml_blck_size(t));
    uint64_t tsize = static_cast<uint64_t>(ggml_type_size(t));
    if (blck == 0 || tsize == 0) return false;
    if (num_elements > UINT64_MAX - (blck - 1)) return false;
    uint64_t blocks = (num_elements + blck - 1) / blck;
    if (blocks > UINT64_MAX / tsize) return false;
    out = blocks * tsize;
    return true;
}

bool ModelFile::parse_gguf(std::string& err_msg) {
    BinaryCursor cur{mapped_base_, file_size_, 4}; // skip "GGUF"
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;

    if (!cur.read(version) || (version != 2 && version != 3)) {
        err_msg = "Unsupported GGUF version: " + std::to_string(version);
        return false;
    }
    if (!cur.read(n_tensors) || !cur.read(n_kv)) {
        err_msg = "Truncated GGUF header";
        return false;
    }
    // Corrupt-header guard: a huge count would make reserve() throw (or
    // OOM) before a single byte is validated. Real models have < 10k
    // tensors/KV entries; each tensor info is >= 24 bytes on disk.
    if (n_tensors > kMaxGgufTensors || n_kv > kMaxGgufKv) {
        err_msg = "GGUF header counts out of range (tensors: " + std::to_string(n_tensors) +
                  ", kv: " + std::to_string(n_kv) + ")";
        return false;
    }
    if (n_tensors > (file_size_ - cur.pos) / 24) {
        err_msg = "GGUF tensor count exceeds file size";
        return false;
    }

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t vtype_raw = 0;
        if (!cur.read_string(key) || !cur.read(vtype_raw)) {
            err_msg = "Failed reading GGUF KV entry " + std::to_string(i);
            return false;
        }
        GGUFValue val;
        if (!read_gguf_value(cur, static_cast<GGUFValueType>(vtype_raw), val)) {
            err_msg = "Failed reading GGUF KV value for key: " + key;
            return false;
        }
        if (key == "general.alignment") {
            if (val.u64 == 0 || val.u64 > kMaxGgufAlignment) {
                err_msg = "Invalid general.alignment in GGUF metadata: " +
                          std::to_string(val.u64);
                return false;
            }
            alignment_ = static_cast<uint32_t>(val.u64);
        }
        metadata_[key] = std::move(val);
    }
    kv_end_offset_ = cur.pos; // end of the KV section == start of tensor infos

    std::vector<TensorDescriptor> temp_tensors;
    temp_tensors.reserve(static_cast<size_t>(n_tensors));

    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorDescriptor td;
        if (!cur.read_string(td.name) || !cur.read(td.n_dims) || td.n_dims < 1 || td.n_dims > 4) {
            err_msg = "Invalid GGUF tensor header at index " + std::to_string(i);
            return false;
        }
        // Checked element count: a dim of 0 is malformed, and the product
        // must not overflow uint64 (2^32 x 2^32 dims would wrap to a
        // small byte_size and pass the bounds check below).
        uint64_t ne = 1;
        for (uint32_t d = 0; d < td.n_dims; ++d) {
            if (!cur.read(td.dims[d])) return false;
            if (td.dims[d] == 0) {
                err_msg = "GGUF tensor '" + td.name + "' has a zero dimension";
                return false;
            }
            if (ne > UINT64_MAX / td.dims[d]) {
                err_msg = "GGUF tensor '" + td.name + "' dimensions overflow";
                return false;
            }
            ne *= td.dims[d];
        }
        uint32_t type_raw = 0;
        if (!cur.read(type_raw) || !cur.read(td.offset)) {
            err_msg = "Truncated GGUF tensor descriptor: " + td.name;
            return false;
        }
        if (!is_known_ggml_type(type_raw)) {
            err_msg = "GGUF tensor '" + td.name + "' has unknown type id " +
                      std::to_string(type_raw) + " (not a GGML type)";
            return false;
        }
        td.type = static_cast<GGMLType>(type_raw);
        if (!checked_tensor_bytes(td.type, ne, td.byte_size)) {
            err_msg = "GGUF tensor '" + td.name + "' byte size overflows";
            return false;
        }
        td.is_sparse_expert = is_expert_tensor_name(td.name);
        temp_tensors.push_back(std::move(td));
    }

    // Align cursor position to general.alignment (default 32 bytes)
    uint64_t rem = cur.pos % alignment_;
    data_offset_ = (rem == 0) ? cur.pos : (cur.pos + (alignment_ - rem));
    if (data_offset_ > file_size_) {
        err_msg = "GGUF data section starts past end of file (alignment " +
                  std::to_string(alignment_) + ")";
        return false;
    }

    for (auto& td : temp_tensors) {
        // Checked absolute offset: td.offset is file-controlled and can be
        // up to 2^64-1, so data_offset_ + offset must not wrap.
        if (td.offset > file_size_ - data_offset_) {
            err_msg = "Tensor offset out of file bounds: " + td.name +
                      " (offset: " + std::to_string(td.offset) +
                      ", file_size: " + std::to_string(file_size_) + ")";
            return false;
        }
        td.abs_file_offset = data_offset_ + td.offset;
        if (td.byte_size > file_size_ - td.abs_file_offset) {
            err_msg = "Tensor exceeds file bounds: " + td.name +
                      " (abs_offset: " + std::to_string(td.abs_file_offset) +
                      ", byte_size: " + std::to_string(td.byte_size) +
                      ", type: " + std::to_string(static_cast<int>(td.type)) +
                      ", file_size: " + std::to_string(file_size_) + ")";
            return false;
        }
        td.mapped_ptr = mapped_base_ + td.abs_file_offset;
        if (tensors_.count(td.name) != 0) {
            err_msg = "Duplicate tensor name in GGUF file: " + td.name;
            return false;
        }
        tensor_order_.push_back(td.name);
        tensors_[td.name] = std::move(td);
    }

    return true;
}

// Native SafeTensors binary + JSON header parser for ONE shard. `base`/`size`
// are that shard's mmap; tensors are merged into the shared tensors_ table
// (a duplicate tensor name across shards is an error). data_offset_out
// receives this shard's data-section start (8 + header length).
bool ModelFile::parse_safetensors_shard(const uint8_t* base, uint64_t size,
                                        uint64_t& data_offset_out, std::string& err_msg) {
    if (size < 8) {
        err_msg = "SafeTensors file too small";
        return false;
    }
    uint64_t header_len = 0;
    std::memcpy(&header_len, base, sizeof(uint64_t));
    // Checked: `8 + header_len` overflows for header_len near 2^64, which
    // used to pass the bounds check and then throw length_error in the
    // std::string construction below (uncaught -> terminate).
    if (header_len == 0 || header_len > kMaxStHeaderBytes ||
        header_len > size - 8) {
        err_msg = "Invalid SafeTensors JSON header size";
        return false;
    }

    data_offset_out = 8 + header_len;
    std::string json(reinterpret_cast<const char*>(base + 8), static_cast<size_t>(header_len));

    // Parse top-level tensor objects in SafeTensors JSON header:
    // { "tensor_name": { "dtype": "F16", "shape": [r, c], "data_offsets": [start, end] }, "__metadata__": {...} }
    // Any structural failure below is a hard error now: the old code
    // `break` out of the loop on malformed input and returned success
    // with whatever tensors parsed so far (partial model, silent).
    size_t pos = 0;
    auto skip_ws = [&]() {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == ',')) pos++;
    };
    auto parse_str = [&](std::string& s) -> bool {
        skip_ws();
        if (pos >= json.size() || json[pos] != '"') return false;
        pos++;
        s.clear();
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) pos++;
            s.push_back(json[pos++]);
        }
        if (pos >= json.size()) return false;
        pos++; // closing quote
        return true;
    };

    skip_ws();
    if (pos >= json.size() || json[pos] != '{') {
        err_msg = "SafeTensors header does not start with '{'";
        return false;
    }
    pos++;

    bool header_closed = false;
    while (pos < json.size()) {
        skip_ws();
        if (pos >= json.size()) {
            err_msg = "SafeTensors header truncated (no closing '}')";
            return false;
        }
        if (json[pos] == '}') { pos++; header_closed = true; break; }
        std::string key;
        if (!parse_str(key)) {
            err_msg = "SafeTensors header: cannot parse tensor name";
            return false;
        }
        skip_ws();
        if (pos >= json.size() || json[pos] != ':') {
            err_msg = "SafeTensors header: expected ':' after tensor name '" + key + "'";
            return false;
        }
        pos++;
        skip_ws();
        if (pos >= json.size() || json[pos] != '{') {
            err_msg = "SafeTensors header: expected '{' for tensor '" + key + "'";
            return false;
        }

        // Find matching '}' for inner object, string-aware: the old
        // counter counted braces inside string values (a tensor name or
        // metadata value containing '{'/' }' broke the split).
        size_t obj_start = pos;
        int depth = 0;
        bool in_str = false;
        bool obj_closed = false;
        while (pos < json.size()) {
            char c = json[pos];
            if (in_str) {
                if (c == '\\' && pos + 1 < json.size()) { pos += 2; continue; }
                if (c == '"') in_str = false;
            } else if (c == '"') {
                in_str = true;
            } else if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) { pos++; obj_closed = true; break; }
            }
            pos++;
        }
        if (!obj_closed) {
            err_msg = "SafeTensors header: unterminated object for tensor '" + key + "'";
            return false;
        }
        std::string obj = json.substr(obj_start, pos - obj_start);

        if (key == "__metadata__") {
            // Previously ignored entirely. Store string pairs as metadata
            // (usually just {"format": "pt"}); the engine does not read them,
            // but they are now visible via metadata() instead of dropped.
            size_t scan = 0;
            while (scan < obj.size()) {
                size_t k1 = obj.find('"', scan);
                if (k1 == std::string::npos) break;
                size_t k2 = obj.find('"', k1 + 1);
                if (k2 == std::string::npos) break;
                std::string mk = obj.substr(k1 + 1, k2 - k1 - 1);
                size_t colon = obj.find(':', k2 + 1);
                if (colon == std::string::npos) break;
                size_t v1 = obj.find('"', colon + 1);
                if (v1 == std::string::npos) break;
                size_t v2 = obj.find('"', v1 + 1);
                if (v2 == std::string::npos) break;
                GGUFValue mv;
                mv.type = GGUFValueType::STRING;
                mv.str = obj.substr(v1 + 1, v2 - v1 - 1);
                metadata_["safetensors.metadata." + mk] = std::move(mv);
                scan = v2 + 1;
            }
            continue;
        }

        TensorDescriptor td;
        td.name = key;
        td.is_sparse_expert = is_expert_tensor_name(td.name);

        // Extract dtype. An unknown dtype is a hard error now: the old
        // code warned once and treated the bytes as F32, so a typo'd or
        // unsupported dtype silently misread every weight of the tensor.
        size_t dt_pos = obj.find("\"dtype\"");
        if (dt_pos == std::string::npos) {
            err_msg = "SafeTensors tensor '" + key + "' has no dtype";
            return false;
        }
        {
            size_t q1 = obj.find('"', dt_pos + 7);
            size_t q2 = (q1 != std::string::npos) ? obj.find('"', q1 + 1) : std::string::npos;
            if (q1 == std::string::npos || q2 == std::string::npos) {
                err_msg = "SafeTensors tensor '" + key + "' has a malformed dtype";
                return false;
            }
            std::string dt = obj.substr(q1 + 1, q2 - q1 - 1);
                static const std::pair<const char*, GGMLType> kDtypes[] = {
                    {"F32", GGMLType::F32},     {"F16", GGMLType::F16},
                    {"BF16", GGMLType::BF16},   {"F64", GGMLType::F64},
                    {"I8", GGMLType::I8},       {"I16", GGMLType::I16},
                    {"I32", GGMLType::I32},     {"I64", GGMLType::I64},
                    {"F8_E4M3", GGMLType::FP8_E4M3},
                    {"Q4_0", GGMLType::Q4_0},   {"Q8_0", GGMLType::Q8_0},
                    {"Q4_K", GGMLType::Q4_K},   {"Q5_K", GGMLType::Q5_K},
                    {"Q6_K", GGMLType::Q6_K},   {"Q8_K", GGMLType::Q8_K},
                    {"IQ4_NL", GGMLType::IQ4_NL},
                };
                bool known = false;
                for (const auto& [name, type] : kDtypes) {
                    if (dt == name) { td.type = type; known = true; break; }
                }
                if (!known) {
                    err_msg = "SafeTensors tensor '" + key + "' has unsupported dtype '" + dt +
                              "' (supported: F32 F16 BF16 F64 I8 I16 I32 I64 F8_E4M3 Q4_0 Q8_0 "
                              "Q4_K Q5_K Q6_K Q8_K IQ4_NL)";
                    return false;
                }
            }

        // Extract shape. SafeTensors shapes are PyTorch order [out, in],
        // but this engine's GGUF convention is dims = [in, out]
        // (dims[0] = input cols, dims[1] = output rows — see bind_tensors(),
        // which reads q_dim from dims[1] and dim/vocab from the embed
        // tensor's dims[0]/dims[1], and model_forge.cpp, which writes
        // token_embd as {dim, vocab_size}). The ST bytes themselves are
        // already row-major [out][in], which is exactly the layout
        // matmul_quant_f32 expects (one output row per row of data), so
        // only the dims are reversed — never the data.
        // A missing/malformed shape, a zero dimension, > 4 dims, or an
        // overflowing element count is a hard error (the old code left
        // defaults in place and std::stoull could throw on huge values).
        size_t sh_pos = obj.find("\"shape\"");
        if (sh_pos == std::string::npos) {
            err_msg = "SafeTensors tensor '" + key + "' has no shape";
            return false;
        }
        uint64_t numel = 1;
        {
            size_t b1 = obj.find('[', sh_pos);
            size_t b2 = (b1 != std::string::npos) ? obj.find(']', b1) : std::string::npos;
            if (b1 == std::string::npos || b2 == std::string::npos) {
                err_msg = "SafeTensors tensor '" + key + "' has a malformed shape";
                return false;
            }
            std::string arr = obj.substr(b1 + 1, b2 - b1 - 1);
            std::stringstream ss(arr);
            std::string item;
            uint32_t d = 0;
            while (std::getline(ss, item, ',')) {
                if (item.empty()) continue;
                // Trim spaces (std::from_chars rejects them).
                size_t ib = item.find_first_not_of(" \t");
                size_t ie = item.find_last_not_of(" \t");
                if (ib == std::string::npos) continue;
                uint64_t dim_v = 0;
                if (!parse_u64_strict(item.substr(ib, ie - ib + 1), dim_v)) {
                    err_msg = "SafeTensors tensor '" + key + "' has a non-numeric shape entry";
                    return false;
                }
                if (d >= 4) {
                    err_msg = "SafeTensors tensor '" + key + "' has more than 4 dimensions";
                    return false;
                }
                if (dim_v == 0) {
                    err_msg = "SafeTensors tensor '" + key + "' has a zero dimension";
                    return false;
                }
                if (numel > UINT64_MAX / dim_v) {
                    err_msg = "SafeTensors tensor '" + key + "' dimensions overflow";
                    return false;
                }
                numel *= dim_v;
                td.dims[d++] = dim_v;
            }
            td.n_dims = d;
            if (td.n_dims == 2) std::swap(td.dims[0], td.dims[1]);
        }

        // Extract data_offsets. start <= end, end within this shard's data
        // section, and the byte span must EXACTLY match dtype*shape: a
        // shorter span used to leave mapped_ptr null while the tensor
        // stayed in the table (later readers dereference null or read
        // short), and a longer span silently over-read into the next
        // tensor. end < start used to underflow to a huge byte_size.
        size_t do_pos = obj.find("\"data_offsets\"");
        if (do_pos == std::string::npos) {
            err_msg = "SafeTensors tensor '" + key + "' has no data_offsets";
            return false;
        }
        {
            size_t b1 = obj.find('[', do_pos);
            size_t b2 = (b1 != std::string::npos) ? obj.find(']', b1) : std::string::npos;
            if (b1 == std::string::npos || b2 == std::string::npos) {
                err_msg = "SafeTensors tensor '" + key + "' has malformed data_offsets";
                return false;
            }
            std::string arr = obj.substr(b1 + 1, b2 - b1 - 1);
            size_t comma = arr.find(',');
            if (comma == std::string::npos) {
                err_msg = "SafeTensors tensor '" + key + "' data_offsets need [start, end]";
                return false;
            }
            auto trim = [](const std::string& s) {
                size_t b = s.find_first_not_of(" \t");
                size_t e = s.find_last_not_of(" \t");
                return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
            };
            uint64_t start_off = 0, end_off = 0;
            if (!parse_u64_strict(trim(arr.substr(0, comma)), start_off) ||
                !parse_u64_strict(trim(arr.substr(comma + 1)), end_off)) {
                err_msg = "SafeTensors tensor '" + key + "' has non-numeric data_offsets";
                return false;
            }
            uint64_t data_size = size - data_offset_out;
            if (end_off < start_off || end_off > data_size) {
                err_msg = "SafeTensors tensor '" + key + "' data_offsets out of range [" +
                          std::to_string(start_off) + ", " + std::to_string(end_off) +
                          "] (data section: " + std::to_string(data_size) + " bytes)";
                return false;
            }
            uint64_t expected_bytes = 0;
            if (!checked_tensor_bytes(td.type, numel, expected_bytes)) {
                err_msg = "SafeTensors tensor '" + key + "' byte size overflows";
                return false;
            }
            if (end_off - start_off != expected_bytes) {
                err_msg = "SafeTensors tensor '" + key + "' byte size mismatch: header says " +
                          std::to_string(end_off - start_off) + " bytes but dtype " +
                          ggml_type_name(td.type) + " x " + std::to_string(numel) +
                          " elements needs " + std::to_string(expected_bytes);
                return false;
            }
            td.offset = start_off;
            td.abs_file_offset = data_offset_out + start_off;
            td.byte_size = end_off - start_off;
            td.mapped_ptr = base + td.abs_file_offset;
        }

        // HF -> GGUF name translation. Routed MoE expert tensors have no
        // per-tensor translation (they are packed later by
        // pack_safetensors_experts()); fused qkv_proj IS translated to
        // blk.<l>.attn_qkv.weight and split on demand by split_fused_qkv().
        // Anything else without a translation stays under its original name
        // and is flagged.
        std::string gguf_name = translate_hf_tensor_name(td.name);
        if (!gguf_name.empty() && gguf_name != td.name) {
            alias_[gguf_name] = td.name;
        } else if (td.name.find(".experts.") != std::string::npos) {
            st_moe_unsupported_ = true;
            st_untranslated_.push_back(td.name);
        } else {
            st_untranslated_.push_back(td.name);
        }

        if (tensors_.count(td.name) != 0) {
            err_msg = "Duplicate tensor name across SafeTensors shards: " + td.name;
            return false;
        }
        tensor_order_.push_back(td.name);
        tensors_[td.name] = std::move(td);
    }

    // The loop only exits via the '}' above (every other exit returns
    // false). Anything but whitespace after the closing brace is trailing
    // garbage — reject it instead of ignoring it.
    if (!header_closed) {
        err_msg = "SafeTensors header truncated (no closing '}')";
        return false;
    }
    skip_ws();
    if (pos < json.size()) {
        err_msg = "SafeTensors header: trailing data after closing '}'";
        return false;
    }
    if (tensors_.empty()) {
        err_msg = "SafeTensors header contains no tensors";
        return false;
    }
    return true;
}

// Load a sibling config.json (and note a sibling tokenizer.json) for a
// SafeTensors model. Best-effort: returns false when no usable config was
// found, and open() treats that as "load without config", not an error.
bool ModelFile::load_safetensors_config() {
    namespace fs = std::filesystem;
    fs::path dir = fs::path(path_).parent_path();
    std::error_code ec;

    fs::path tok_path = dir / "tokenizer.json";
    if (fs::exists(tok_path, ec) && !ec) {
        tokenizer_path_ = tok_path.string();
        std::cout << "[SAFETENSORS] tokenizer.json found: " << tokenizer_path_
                  << " (parsing it belongs to the tokenizer layer).\n";
    }

    fs::path cfg_path = dir / "config.json";
    if (!fs::exists(cfg_path, ec) || ec) return false;
    std::ifstream in(cfg_path);
    if (!in) return false;
    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (json.empty()) return false;

    std::string arch_class, model_type;
    cfg_get_first_array_string(json, "architectures", arch_class);
    cfg_get_string(json, "model_type", model_type);

    std::string arch = hf_arch_to_gguf(arch_class);
    if (arch.empty() && is_known_gguf_arch(model_type)) arch = model_type;
    if (arch.empty()) {
        std::cout << "[SAFETENSORS] WARNING: config.json architecture '" << arch_class
                  << "' (model_type '" << model_type << "') is not mapped — "
                  << "no hparams were loaded; the engine will use its defaults.\n";
        return false;
    }

    // Multimodal configs (Gemma3) nest the language-model hparams under
    // "text_config" while "vision_config" holds the vision tower's (e.g.
    // Gemma3-1b: text 26 layers, vision 27). The flat lookups below take
    // the FIRST match in the file, which would resolve to the vision
    // tower's values, so prefer text_config whenever the section exists.
    // (architectures / model_type above deliberately stay top-level:
    // text_config's own model_type is "gemma3_text".)
    std::string text_json;
    bool have_text = cfg_subobject(json, "text_config", text_json);
    auto cfg_num = [&](const std::string& key, double& out) -> bool {
        if (have_text && cfg_get_number(text_json, key, out)) return true;
        return cfg_get_number(json, key, out);
    };
    auto cfg_bool = [&](const std::string& key, bool& out) -> bool {
        if (have_text && cfg_get_bool(text_json, key, out)) return true;
        return cfg_get_bool(json, key, out);
    };
    auto cfg_str = [&](const std::string& key, std::string& out) -> bool {
        if (have_text && cfg_get_string(text_json, key, out)) return true;
        return cfg_get_string(json, key, out);
    };

    double hidden = 0, intermediate = 0, layers = 0, heads = 0, kv_heads = 0;
    double ctx = 0, vocab = 0, eps = 0, rope_theta = 0, v = 0;
    bool have_hidden = cfg_num("hidden_size", hidden);
    bool have_layers = cfg_num("num_hidden_layers", layers);
    if (!have_hidden || hidden <= 0 || !have_layers || layers <= 0) {
        std::cout << "[SAFETENSORS] WARNING: config.json for '" << arch
                  << "' has no usable hidden_size/num_hidden_layers — no hparams loaded.\n";
        return false;
    }

    auto set_u32 = [&](const std::string& k, uint32_t v) {
        GGUFValue val; val.type = GGUFValueType::UINT32; val.u64 = v; val.i64 = v;
        metadata_[k] = std::move(val);
    };
    auto set_f32 = [&](const std::string& k, float v) {
        GGUFValue val; val.type = GGUFValueType::FLOAT32; val.f64 = v;
        metadata_[k] = std::move(val);
    };
    auto set_str = [&](const std::string& k, const std::string& v) {
        GGUFValue val; val.type = GGUFValueType::STRING; val.str = v;
        metadata_[k] = std::move(val);
    };
    auto set_bool = [&](const std::string& k, bool v) {
        GGUFValue val; val.type = GGUFValueType::BOOL; val.bval = v; val.u64 = v ? 1 : 0;
        metadata_[k] = std::move(val);
    };

    set_str("general.architecture", arch);
    set_str("general.name", arch_class.empty() ? model_type : arch_class);
    set_u32(arch + ".embedding_length", static_cast<uint32_t>(hidden));
    if (cfg_num("intermediate_size", intermediate) && intermediate > 0) {
        set_u32(arch + ".feed_forward_length", static_cast<uint32_t>(intermediate));
    }
    if (cfg_get_number(json, "moe_intermediate_size", v) && v > 0) {
        // Qwen2/3-MoE routed-expert dim (distinct from intermediate_size,
        // which is the shared-expert dim for those models).
        set_u32(arch + ".expert_feed_forward_length", static_cast<uint32_t>(v));
    }
    set_u32(arch + ".block_count", static_cast<uint32_t>(layers));
    if (cfg_num("num_attention_heads", heads) && heads > 0) {
        set_u32(arch + ".attention.head_count", static_cast<uint32_t>(heads));
    }
    if (cfg_num("num_key_value_heads", kv_heads) && kv_heads > 0) {
        set_u32(arch + ".attention.head_count_kv", static_cast<uint32_t>(kv_heads));
    } else if (heads > 0) {
        set_u32(arch + ".attention.head_count_kv", static_cast<uint32_t>(heads));
    }
    if (cfg_num("max_position_embeddings", ctx) && ctx > 0) {
        set_u32(arch + ".context_length", static_cast<uint32_t>(ctx));
    }
    if (cfg_num("vocab_size", vocab) && vocab > 0) {
        set_u32(arch + ".vocab_size", static_cast<uint32_t>(vocab));
    }
    if (cfg_num("rms_norm_eps", eps) && eps > 0) {
        set_f32(arch + ".attention.layer_norm_rms_epsilon", static_cast<float>(eps));
    }
    if (cfg_num("rope_theta", rope_theta) && rope_theta > 0) {
        set_f32(arch + ".rope.freq_base", static_cast<float>(rope_theta));
    }
    if (cfg_num("head_dim", v) && v > 0) {
        set_u32(arch + ".attention.key_length", static_cast<uint32_t>(v));
    }
    if (cfg_num("sliding_window", v) && v > 0) {
        set_u32(arch + ".attention.sliding_window", static_cast<uint32_t>(v));
    }
    if (cfg_num("num_experts", v) && v > 0) {
        set_u32(arch + ".expert_count", static_cast<uint32_t>(v));
    } else if (cfg_get_number(json, "num_local_experts", v) && v > 0) {
        // Mixtral calls the expert count num_local_experts.
        set_u32(arch + ".expert_count", static_cast<uint32_t>(v));
    }
    if (cfg_num("num_experts_per_tok", v) && v > 0) {
        set_u32(arch + ".expert_used_count", static_cast<uint32_t>(v));
    }
    bool b = false;
    if (cfg_bool("tie_word_embeddings", b)) {
        set_bool(arch + ".tie_word_embeddings", b);
    }
    std::string dtype;
    if (cfg_str("torch_dtype", dtype)) {
        set_str("general.torch_dtype", dtype);
    }

    detected_arch_ = arch;
    config_path_ = cfg_path.string();
    config_loaded_ = true;
    std::cout << "[SAFETENSORS] config.json loaded: arch=" << arch
              << " layers=" << static_cast<uint32_t>(layers)
              << " hidden=" << static_cast<uint32_t>(hidden)
              << " heads=" << static_cast<uint32_t>(heads)
              << " kv_heads=" << static_cast<uint32_t>(kv_heads > 0 ? kv_heads : heads)
              << " (" << config_path_ << ")\n";
    return true;
}

bool ModelFile::pack_safetensors_experts() {
    struct ExpertEntry {
        uint32_t idx;
        std::string name;
        const TensorDescriptor* td;
    };
    // Per-expert 2D tensors (production checkpoints) and already-packed 3D
    // tensors (transformers 5.x), grouped by layer.
    std::map<uint32_t, std::map<int, std::vector<ExpertEntry>>> by_layer;
    std::map<uint32_t, std::map<int, const TensorDescriptor*>> packed3d;
    std::map<std::pair<uint32_t, int>, std::string> packed3d_names;
    std::set<std::string> unpacked;
    for (const auto& name : tensor_order_) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) continue;
        uint32_t layer = 0, expert = 0; int kind = -1;
        if (parse_expert_tensor_name(name, layer, expert, kind)) {
            by_layer[layer][kind].push_back({expert, name, &it->second});
            unpacked.insert(name);
            continue;
        }
        int which = -1;
        if (parse_packed_expert_name(name, layer, which)) {
            packed3d[layer][which] = &it->second;
            packed3d_names[{layer, which}] = name;
            unpacked.insert(name);
        }
    }
    if (by_layer.empty() && packed3d.empty()) return true; // dense model

    static const char* kKindName[3] = {"gate", "up", "down"};
    // Gather `chunks` (source pointer + length, in expert order) into one
    // owned buffer and publish the packed blk.<l>.ffn_<kind>_exps.weight
    // descriptor for it. False on a null source pointer or OOM.
    auto publish_gather = [&](uint32_t layer, const char* kind_name, GGMLType type,
                              uint64_t dim, uint64_t expert_dim, uint32_t n_exp,
                              const std::vector<std::pair<const uint8_t*, uint64_t>>& chunks) -> bool {
        uint64_t total = 0;
        for (const auto& c : chunks) { if (!c.first) return false; total += c.second; }
        std::vector<uint8_t> buf;
        try { buf.resize(static_cast<size_t>(total)); } catch (...) { return false; }
        size_t off = 0;
        for (const auto& c : chunks) {
            std::memcpy(buf.data() + off, c.first, static_cast<size_t>(c.second));
            off += static_cast<size_t>(c.second);
        }
        // deque element addresses never move, so the descriptor's mapped_ptr
        // stays valid while this ModelFile is open.
        packed_buffers_.push_back(std::move(buf));
        TensorDescriptor pd;
        pd.name = "blk." + std::to_string(layer) + ".ffn_" + kind_name + "_exps.weight";
        pd.n_dims = 3;
        pd.dims[0] = dim; pd.dims[1] = expert_dim; pd.dims[2] = n_exp;
        pd.type = type;
        pd.offset = 0;
        pd.abs_file_offset = 0; // synthetic: no file offset (forward uses mapped_ptr)
        pd.byte_size = total;
        pd.is_sparse_expert = true;
        pd.mapped_ptr = packed_buffers_.back().data();
        tensors_[pd.name] = pd;
        tensor_order_.push_back(pd.name);
        return true;
    };
    // Remove a layer's published packed descriptors again (a publish that
    // failed halfway must not leave a half-packed layer behind).
    auto unpublish_layer = [&](uint32_t layer) {
        for (int k = 0; k < 3; ++k) {
            std::string nm = "blk." + std::to_string(layer) + ".ffn_" + kKindName[k] + "_exps.weight";
            tensors_.erase(nm);
            tensor_order_.erase(std::remove(tensor_order_.begin(), tensor_order_.end(), nm),
                                tensor_order_.end());
        }
    };

    std::set<uint32_t> layers;
    for (const auto& kv : by_layer) layers.insert(kv.first);
    for (const auto& kv : packed3d) layers.insert(kv.first);

    uint32_t packed_layers = 0;
    for (uint32_t layer : layers) {
        // ---- Path B: already-packed 3D tensors (transformers 5.x) ----
        auto p3 = packed3d.find(layer);
        if (p3 != packed3d.end()) {
            const auto& m = p3->second;
            const TensorDescriptor* gate_up = m.count(0) ? m.at(0) : nullptr;
            const TensorDescriptor* down = m.count(1) ? m.at(1) : nullptr;
            const TensorDescriptor* gate3 = m.count(2) ? m.at(2) : nullptr;
            const TensorDescriptor* up3 = m.count(3) ? m.at(3) : nullptr;
            std::string why;
            bool ok3 = true;
            uint32_t n_exp = 0;
            uint64_t dim = 0, expert_dim = 0, gate_chunk = 0, down_chunk = 0, gate_up_stride = 0;
            if (!down) { ok3 = false; why = "packed 3D experts without down_proj"; }
            if (ok3 && !gate_up && !(gate3 && up3)) {
                ok3 = false; why = "packed 3D experts without gate_up_proj (or gate_proj+up_proj)";
            }
            if (ok3) {
                // gate_up [E, 2I, H]; down [E, H, I]; gate3/up3 [E, I, H].
                // (3D SafeTensors dims are stored as-is, not reversed.)
                const TensorDescriptor* ref = gate_up ? gate_up : gate3;
                if (ref->n_dims != 3) { ok3 = false; why = "packed expert tensor is not 3D"; }
                else {
                    n_exp = static_cast<uint32_t>(ref->dims[0]);
                    dim = ref->dims[2];
                    expert_dim = gate_up ? ref->dims[1] / 2 : ref->dims[1];
                    if (gate_up && ref->dims[1] % 2 != 0) {
                        ok3 = false; why = "gate_up_proj middle dim is not even";
                    }
                    if (down->n_dims != 3 || down->dims[0] != n_exp ||
                        down->dims[1] != dim || down->dims[2] != expert_dim) {
                        ok3 = false; why = "down_proj dims do not match gate_up_proj (want [E, H, I])";
                    }
                }
            }
            if (ok3) {
                GGMLType gt = gate_up ? gate_up->type : gate3->type;
                gate_chunk = tensor_byte_size(gt, expert_dim * dim);
                down_chunk = tensor_byte_size(down->type, expert_dim * dim);
                if (gate_up) {
                    gate_up_stride = tensor_byte_size(gt, 2 * expert_dim * dim);
                    if (gate_up->byte_size != static_cast<uint64_t>(n_exp) * gate_up_stride) {
                        ok3 = false; why = "gate_up_proj byte size does not match its dims";
                    }
                } else if (gate3->byte_size != static_cast<uint64_t>(n_exp) * gate_chunk ||
                           up3->byte_size != static_cast<uint64_t>(n_exp) * gate_chunk) {
                    ok3 = false; why = "gate_proj/up_proj byte size does not match dims";
                }
                if (ok3 && down->byte_size != static_cast<uint64_t>(n_exp) * down_chunk) {
                    ok3 = false; why = "down_proj byte size does not match its dims";
                }
            }
            if (ok3) {
                const uint8_t* gu = gate_up ? static_cast<const uint8_t*>(gate_up->mapped_ptr) : nullptr;
                const uint8_t* g3 = gate3 ? static_cast<const uint8_t*>(gate3->mapped_ptr) : nullptr;
                const uint8_t* u3 = up3 ? static_cast<const uint8_t*>(up3->mapped_ptr) : nullptr;
                const uint8_t* dn = static_cast<const uint8_t*>(down->mapped_ptr);
                std::vector<std::pair<const uint8_t*, uint64_t>> gate_chunks, up_chunks, down_chunks;
                if (!dn || (gate_up && !gu) || (gate3 && (!g3 || !u3))) {
                    ok3 = false; why = "packed expert tensor has no data pointer";
                } else {
                    for (uint32_t e = 0; e < n_exp; ++e) {
                        if (gate_up) {
                            gate_chunks.push_back({gu + e * gate_up_stride, gate_chunk});
                            up_chunks.push_back({gu + e * gate_up_stride + gate_chunk, gate_chunk});
                        } else {
                            gate_chunks.push_back({g3 + e * gate_chunk, gate_chunk});
                            up_chunks.push_back({u3 + e * gate_chunk, gate_chunk});
                        }
                        down_chunks.push_back({dn + e * down_chunk, down_chunk});
                    }
                    GGMLType gt = gate_up ? gate_up->type : gate3->type;
                    bool pub = publish_gather(layer, "gate", gt, dim, expert_dim, n_exp, gate_chunks) &&
                               publish_gather(layer, "up", gt, dim, expert_dim, n_exp, up_chunks) &&
                               publish_gather(layer, "down", down->type, dim, expert_dim, n_exp, down_chunks);
                    if (!pub) {
                        ok3 = false; why = "out of memory while packing";
                        unpublish_layer(layer);
                    }
                }
            }
            if (ok3) {
                ++packed_layers;
                for (const auto& kv : m) unpacked.erase(packed3d_names[{layer, kv.first}]);
                continue; // packed-3D done for this layer
            }
            std::cerr << "[SAFETENSORS] WARNING: layer " << layer << " packed 3D experts not usable ("
                      << why << ") — trying per-expert tensors.\n";
            // Fall through to the per-expert path below.
        }

        // ---- Path A: per-expert 2D tensors (production checkpoints) ----
        auto bit = by_layer.find(layer);
        const std::map<int, std::vector<ExpertEntry>>* kinds_ptr =
            (bit == by_layer.end()) ? nullptr : &bit->second;
        static const std::map<int, std::vector<ExpertEntry>> kEmptyKinds;
        const auto& kinds = kinds_ptr ? *kinds_ptr : kEmptyKinds;
        std::string why;
        bool ok = kinds.count(0) && kinds.count(1) && kinds.count(2);
        if (!ok) why = "missing gate/up/down tensors for some experts";
        uint32_t n_exp = 0;
        if (ok) {
            n_exp = static_cast<uint32_t>(kinds.at(0).size());
            if (kinds.at(1).size() != n_exp || kinds.at(2).size() != n_exp) {
                ok = false; why = "unequal expert counts per projection";
            }
        }
        // Within each kind: contiguous indices 0..N-1, one type, one shape.
        // Entries are sorted by expert index once here; the sorted copies
        // are used for validation AND publishing, so chunk order is always
        // expert order regardless of file order.
        const TensorDescriptor* kind_ref[3] = {nullptr, nullptr, nullptr};
        std::vector<ExpertEntry> sorted_kinds[3];
        if (ok) {
            for (int k = 0; k < 3 && ok; ++k) {
                sorted_kinds[k] = kinds.at(k);
                std::sort(sorted_kinds[k].begin(), sorted_kinds[k].end(),
                          [](const ExpertEntry& a, const ExpertEntry& b) { return a.idx < b.idx; });
                for (uint32_t e = 0; e < sorted_kinds[k].size(); ++e) {
                    if (sorted_kinds[k][e].idx != e) { ok = false; why = "expert indices are not contiguous 0..N-1"; break; }
                }
                if (!ok) break;
                for (const auto& ent : sorted_kinds[k]) {
                    const TensorDescriptor* td = ent.td;
                    if (td->n_dims != 2) { ok = false; why = "expert tensor is not 2D"; break; }
                    if (!kind_ref[k]) kind_ref[k] = td;
                    else if (td->type != kind_ref[k]->type || td->byte_size != kind_ref[k]->byte_size ||
                             td->dims[0] != kind_ref[k]->dims[0] || td->dims[1] != kind_ref[k]->dims[1]) {
                        ok = false; why = "expert tensors differ in type/shape/size within one projection"; break;
                    }
                }
            }
        }
        // Cross-kind sanity: gate and up share [dim, expert_dim]; down is
        // the transpose [expert_dim, dim].
        if (ok) {
            if (kind_ref[0]->dims[0] != kind_ref[1]->dims[0] ||
                kind_ref[0]->dims[1] != kind_ref[1]->dims[1] ||
                kind_ref[2]->dims[0] != kind_ref[0]->dims[1] ||
                kind_ref[2]->dims[1] != kind_ref[0]->dims[0]) {
                ok = false; why = "gate/up/down shapes are inconsistent (down must transpose gate/up)";
            }
        }
        // Cross-check against config.json's expert count when available.
        if (ok && !detected_arch_.empty()) {
            uint32_t cfg_exp = get_u32(detected_arch_ + ".expert_count", 0);
            if (cfg_exp > 0 && cfg_exp != n_exp) {
                std::cerr << "[SAFETENSORS] WARNING: layer " << layer << " has " << n_exp
                          << " experts on disk but config.json says " << cfg_exp
                          << " — packing the " << n_exp << " found.\n";
            }
        }
        if (!ok) {
            std::cerr << "[SAFETENSORS] WARNING: layer " << layer << " experts not packed ("
                      << why << ") — those experts will not run.\n";
            continue;
        }

        uint64_t dim = kind_ref[0]->dims[0];
        uint64_t expert_dim = kind_ref[0]->dims[1];
        bool layer_ok = true;
        for (int k = 0; k < 3 && layer_ok; ++k) {
            std::vector<std::pair<const uint8_t*, uint64_t>> chunks;
            for (const auto& ent : sorted_kinds[k]) {
                if (!ent.td->mapped_ptr) { layer_ok = false; break; }
                chunks.push_back({static_cast<const uint8_t*>(ent.td->mapped_ptr), ent.td->byte_size});
            }
            if (!layer_ok) break;
            if (!publish_gather(layer, kKindName[k], kind_ref[k]->type, dim, expert_dim, n_exp, chunks)) {
                std::cerr << "[SAFETENSORS] WARNING: out of memory packing layer " << layer
                          << " " << kKindName[k] << " experts — layer skipped.\n";
                layer_ok = false;
                unpublish_layer(layer);
                break;
            }
            for (const auto& ent : sorted_kinds[k]) unpacked.erase(ent.name);
        }
        if (layer_ok) ++packed_layers;
    }

    if (!unpacked.empty()) {
        std::cerr << "[SAFETENSORS] WARNING: " << unpacked.size()
                  << " per-expert tensor(s) could not be packed into ffn_*_exps.\n";
        st_moe_unsupported_ = true;
    } else {
        st_moe_unsupported_ = false;
        std::cout << "[SAFETENSORS] Packed MoE experts for " << packed_layers
                  << " layer(s) into ffn_*_exps tensors.\n";
    }
    // Packed experts are translated now: drop them from the untranslated list.
    std::vector<std::string> kept;
    for (const auto& n : st_untranslated_) {
        uint32_t l = 0, e = 0; int k = -1, w = -1;
        bool is_expert = parse_expert_tensor_name(n, l, e, k) ||
                         parse_packed_expert_name(n, l, w);
        if (is_expert && unpacked.count(n) == 0) continue;
        kept.push_back(n);
    }
    st_untranslated_ = std::move(kept);
    return true;
}

bool ModelFile::split_fused_qkv(uint32_t layer, uint32_t q_dim, uint32_t kv_dim,
                                std::string& err_msg) {
    std::string bp = "blk." + std::to_string(layer) + ".";
    if (find_tensor(bp + "attn_q.weight") && find_tensor(bp + "attn_k.weight") &&
        find_tensor(bp + "attn_v.weight")) return true; // already split
    const TensorDescriptor* fused = find_tensor(bp + "attn_qkv.weight");
    if (!fused) return false; // no fused tensor: caller keeps previous behaviour
    if (!fused->mapped_ptr) {
        err_msg = "attn_qkv.weight for layer " + std::to_string(layer) + " has no data pointer";
        return false;
    }
    if (fused->n_dims != 2) {
        err_msg = "attn_qkv.weight for layer " + std::to_string(layer) + " is not 2D";
        return false;
    }
    uint64_t dim = fused->dims[0];
    uint64_t total_out = fused->dims[1];
    if (total_out != static_cast<uint64_t>(q_dim) + 2ull * static_cast<uint64_t>(kv_dim)) {
        err_msg = "attn_qkv.weight for layer " + std::to_string(layer) + " has " +
                  std::to_string(total_out) + " output rows, expected q_dim + 2*kv_dim = " +
                  std::to_string(static_cast<uint64_t>(q_dim) + 2ull * static_cast<uint64_t>(kv_dim));
        return false;
    }
    // One output row = `dim` elements of the fused type. The fused matrix
    // is row-major [q_rows | k_rows | v_rows] (SafeTensors [out, in] order,
    // the layout matmul_quant_f32 expects), so each split is a contiguous
    // byte range of the fused tensor — views, no copy.
    // Checked arithmetic: row_bytes * total_out used to be a plain uint64
    // multiply, so a crafted dims[1] could wrap it into a false match
    // with byte_size and produce views reading past the tensor.
    uint64_t row_bytes = 0;
    if (!checked_tensor_bytes(fused->type, dim, row_bytes) || row_bytes == 0) {
        err_msg = "attn_qkv.weight for layer " + std::to_string(layer) +
                  " has an inconsistent row size for its type/dims";
        return false;
    }
    if (total_out > UINT64_MAX / row_bytes || fused->byte_size != row_bytes * total_out) {
        err_msg = "attn_qkv.weight for layer " + std::to_string(layer) +
                  " has an inconsistent byte size for its type/dims";
        return false;
    }
    auto make_view = [&](const std::string& name, uint64_t row_off, uint64_t rows) {
        TensorDescriptor td;
        td.name = name;
        td.n_dims = 2;
        td.dims[0] = dim;
        td.dims[1] = rows;
        td.type = fused->type;
        td.offset = fused->offset + row_off * row_bytes;
        td.abs_file_offset = fused->abs_file_offset + row_off * row_bytes;
        td.byte_size = row_bytes * rows;
        td.mapped_ptr = fused->mapped_ptr
            ? static_cast<const uint8_t*>(fused->mapped_ptr) + row_off * row_bytes
            : nullptr;
        tensors_[name] = td;
        tensor_order_.push_back(name);
    };
    make_view(bp + "attn_q.weight", 0, q_dim);
    make_view(bp + "attn_k.weight", q_dim, kv_dim);
    make_view(bp + "attn_v.weight", q_dim + kv_dim, kv_dim);
    // Fused bias (1D): same row split by element counts. Any plain element
    // type works — the views keep the fused bias's dtype (F32, F16, BF16).
    // Note: forward currently only applies F32 biases, so a BF16/F16 fused
    // bias is split but ignored — exactly the same as an unfused BF16/F16
    // bias is ignored. The split is still done so the tensors exist under
    // their canonical names.
    if (const TensorDescriptor* fb = find_tensor(bp + "attn_qkv.bias")) {
        uint64_t elem = ggml_type_size(fb->type);
        if (fb->n_dims == 1 && fb->dims[0] == total_out &&
            fb->byte_size == total_out * elem && fb->mapped_ptr) {
            auto make_bias = [&](const std::string& name, uint64_t elem_off, uint64_t n) {
                TensorDescriptor td;
                td.name = name; td.n_dims = 1; td.dims[0] = n;
                td.type = fb->type;
                td.offset = fb->offset + elem_off * elem;
                td.abs_file_offset = fb->abs_file_offset + elem_off * elem;
                td.byte_size = n * elem;
                td.mapped_ptr = static_cast<const uint8_t*>(fb->mapped_ptr) + elem_off * elem;
                tensors_[name] = td;
                tensor_order_.push_back(name);
            };
            make_bias(bp + "attn_q.bias", 0, q_dim);
            make_bias(bp + "attn_k.bias", q_dim, kv_dim);
            make_bias(bp + "attn_v.bias", q_dim + kv_dim, kv_dim);
        } else {
            std::cerr << "[SAFETENSORS] WARNING: layer " << layer
                      << " attn_qkv.bias has unexpected dims/byte size — bias not split (attention runs without bias).\n";
        }
    }
    return true;
}

const TensorDescriptor* ModelFile::find_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) return &it->second;
    // SafeTensors: try the HF -> GGUF alias table (e.g. "blk.0.attn_q.weight"
    // resolves "model.layers.0.self_attn.q_proj.weight"). The alias map is
    // empty for GGUF files, so their lookup behaviour is unchanged.
    auto alias = alias_.find(name);
    if (alias != alias_.end()) {
        auto jt = tensors_.find(alias->second);
        if (jt != tensors_.end()) return &jt->second;
    }
    return nullptr;
}

uint32_t ModelFile::get_u32(const std::string& key, uint32_t def) const {
    auto it = metadata_.find(key);
    return (it != metadata_.end()) ? static_cast<uint32_t>(it->second.u64) : def;
}

float ModelFile::get_f32(const std::string& key, float def) const {
    auto it = metadata_.find(key);
    return (it != metadata_.end()) ? static_cast<float>(it->second.f64) : def;
}

std::string ModelFile::get_str(const std::string& key, const std::string& def) const {
    auto it = metadata_.find(key);
    return (it != metadata_.end()) ? it->second.str : def;
}

namespace {

uint64_t align_up_u64(uint64_t x, uint64_t a) {
    if (a == 0) a = 32;
    return ((x + a - 1) / a) * a;
}

void write_u32_le(std::ofstream& out, uint32_t v) {
    char b[4];
    std::memcpy(b, &v, 4);
    out.write(b, 4);
}

void write_u64_le(std::ofstream& out, uint64_t v) {
    char b[8];
    std::memcpy(b, &v, 8);
    out.write(b, 8);
}

// Build a merged file's KV section from a shard's: copy every KV entry
// verbatim EXCEPT "split.*" keys — a merged file is not a shard, so
// split.no / split.count / split.tensors.count must not survive the merge
// (llama.cpp's gguf-split --merge drops them too). Entries are copied
// byte-for-byte, never re-serialized, so exotic value types are preserved.
// Returns false when the KV section cannot be walked (caller falls back
// to a verbatim copy).
bool filter_split_kv(const uint8_t* base, uint64_t kv_end, uint64_t n_kv,
                     std::vector<uint8_t>& out_kv, uint64_t& out_n_kv) {
    out_kv.clear();
    out_n_kv = 0;
    BinaryCursor cur{base, kv_end, 24};
    for (uint64_t i = 0; i < n_kv; ++i) {
        uint64_t entry_start = cur.pos;
        std::string key;
        uint32_t vtype_raw = 0;
        if (!cur.read_string(key) || !cur.read(vtype_raw)) return false;
        GGUFValue dummy;
        if (!read_gguf_value(cur, static_cast<GGUFValueType>(vtype_raw), dummy)) return false;
        if (key.rfind("split.", 0) == 0) continue; // drop split.* entries
        out_kv.insert(out_kv.end(), base + entry_start, base + cur.pos);
        ++out_n_kv;
    }
    if (cur.pos != kv_end) return false;
    return true;
}

} // namespace

bool merge_gguf_files(const std::vector<std::string>& shard_paths,
                      const std::string& out_path, std::string& err_msg) {
    namespace fs = std::filesystem;
    if (shard_paths.empty()) {
        err_msg = "No GGUF shards given to merge";
        return false;
    }
    std::error_code ec;
    for (const auto& s : shard_paths) {
        if (fs::equivalent(fs::path(s), fs::path(out_path), ec) && !ec) {
            err_msg = "Output path must differ from the input shards: " + out_path;
            return false;
        }
    }

    struct ShardInfo {
        std::unique_ptr<ModelFile> file;
        uint32_t version = 0;
        uint64_t n_tensors = 0;
        uint64_t n_kv = 0;
    };
    std::vector<ShardInfo> shards;
    std::string arch;
    uint32_t alignment = 32;
    uint64_t total_tensors = 0;
    std::unordered_set<std::string> seen_names;

    for (size_t si = 0; si < shard_paths.size(); ++si) {
        const std::string& sp = shard_paths[si];
        // Header first (exact version / counts, before the ModelFile parse).
        std::ifstream in(sp, std::ios::binary);
        if (!in) {
            err_msg = "Cannot open GGUF shard: " + sp;
            return false;
        }
        char hdr[24] = {0};
        in.read(hdr, 24);
        if (in.gcount() != 24 || std::memcmp(hdr, "GGUF", 4) != 0) {
            err_msg = "Not a GGUF file (bad magic): " + sp;
            return false;
        }
        uint32_t version = 0;
        uint64_t n_tensors = 0, n_kv = 0;
        std::memcpy(&version, hdr + 4, 4);
        std::memcpy(&n_tensors, hdr + 8, 8);
        std::memcpy(&n_kv, hdr + 16, 8);
        if (version != 2 && version != 3) {
            err_msg = "Unsupported GGUF version " + std::to_string(version) + " in shard: " + sp;
            return false;
        }
        in.close();

        ShardInfo info;
        info.file = std::make_unique<ModelFile>();
        std::string open_err;
        if (!info.file->open(sp, open_err)) {
            err_msg = "Cannot parse GGUF shard " + sp + ": " + open_err;
            return false;
        }
        info.version = version;
        info.n_tensors = n_tensors;
        info.n_kv = n_kv;
        if (si == 0) {
            arch = info.file->get_str("general.architecture", "");
            alignment = info.file->get_u32("general.alignment", 32);
            if (alignment == 0) alignment = 32;
        } else {
            if (version != shards.front().version) {
                err_msg = "GGUF version mismatch between shards (" +
                          std::to_string(shards.front().version) + " vs " +
                          std::to_string(version) + ")";
                return false;
            }
            // A shard only declares general.architecture when it carries the
            // full metadata (in practice only shard 1 does — later shards
            // from gguf-split carry just the split.* keys). A missing key
            // is not a mismatch; a present key must agree with the first
            // non-empty architecture seen.
            std::string shard_arch = info.file->get_str("general.architecture", "");
            if (!shard_arch.empty()) {
                if (arch.empty()) {
                    arch = shard_arch;
                } else if (shard_arch != arch) {
                    err_msg = "Architecture mismatch between shards in " + sp;
                    return false;
                }
            }
        }
        if (info.file->tensor_names().size() != n_tensors) {
            err_msg = "Shard header tensor count disagrees with parsed tensors in " + sp;
            return false;
        }
        for (const auto& name : info.file->tensor_names()) {
            if (!seen_names.insert(name).second) {
                err_msg = "Duplicate tensor name across shards: " + name;
                return false;
            }
        }
        total_tensors += n_tensors;
        shards.push_back(std::move(info));
    }

    const ModelFile& first = *shards.front().file;
    uint64_t kv_end = first.kv_section_end();
    if (kv_end < 24 || kv_end > first.file_size()) {
        err_msg = "First shard has an invalid KV section end";
        return false;
    }
    // KV section for the merged file: shard 1's entries minus split.* keys.
    // Falls back to a verbatim copy if the section cannot be walked.
    std::vector<uint8_t> kv_bytes;
    uint64_t n_kv_out = shards.front().n_kv;
    {
        uint64_t filtered_n = 0;
        std::vector<uint8_t> filtered;
        if (filter_split_kv(first.mapped_base(), kv_end, shards.front().n_kv,
                            filtered, filtered_n)) {
            kv_bytes = std::move(filtered);
            n_kv_out = filtered_n;
            if (n_kv_out != shards.front().n_kv) {
                std::cout << "[MERGE] Dropped " << (shards.front().n_kv - n_kv_out)
                          << " split.* metadata entr"
                          << ((shards.front().n_kv - n_kv_out) == 1 ? "y" : "ies")
                          << " (a merged file is not a shard).\n";
            }
        } else {
            std::cerr << "[MERGE] WARNING: could not walk the KV section — copying it verbatim (split.* keys may survive).\n";
            kv_bytes.assign(first.mapped_base() + 24, first.mapped_base() + kv_end);
            n_kv_out = shards.front().n_kv;
        }
    }
    uint64_t out_kv_end = 24 + kv_bytes.size();

    // Tensor list in shard order, with source pointers for the data copy.
    struct OutTensor {
        std::string name;
        uint32_t n_dims = 0;
        uint64_t dims[4] = {1, 1, 1, 1};
        GGMLType type = GGMLType::F32;
        uint64_t byte_size = 0;
        const uint8_t* src = nullptr;
        uint64_t new_offset = 0;
    };
    std::vector<OutTensor> out_tensors;
    out_tensors.reserve(static_cast<size_t>(total_tensors));
    for (const auto& s : shards) {
        for (const auto& name : s.file->tensor_names()) {
            const TensorDescriptor* td = s.file->find_tensor(name);
            OutTensor ot;
            ot.name = name;
            ot.n_dims = td->n_dims;
            for (uint32_t d = 0; d < td->n_dims; ++d) ot.dims[d] = td->dims[d];
            ot.type = td->type;
            ot.byte_size = td->byte_size;
            ot.src = static_cast<const uint8_t*>(td->mapped_ptr);
            if (!ot.src && ot.byte_size > 0) {
                err_msg = "Tensor has no data pointer in shard: " + name;
                return false;
            }
            out_tensors.push_back(std::move(ot));
        }
    }

    // Layout: header (24) + KV bytes + tensor infos, then the data section
    // (aligned), with every tensor's data aligned to `alignment`.
    uint64_t infos_bytes = 0;
    for (const auto& ot : out_tensors) {
        infos_bytes += 8 + ot.name.size() + 4 + 8ull * ot.n_dims + 4 + 8;
    }
    uint64_t data_start = align_up_u64(out_kv_end + infos_bytes, alignment);
    uint64_t cursor = 0;
    for (auto& ot : out_tensors) {
        ot.new_offset = align_up_u64(cursor, alignment);
        cursor = ot.new_offset + ot.byte_size;
    }

    fs::path out_p(out_path);
    if (out_p.has_parent_path()) {
        fs::create_directories(out_p.parent_path(), ec);
        if (ec) {
            err_msg = "Cannot create output directory: " + out_p.parent_path().string();
            return false;
        }
    }
    {
        std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            err_msg = "Cannot open output file for writing: " + out_path;
            return false;
        }
        out.write("GGUF", 4);
        write_u32_le(out, shards.front().version);
        write_u64_le(out, total_tensors);
        write_u64_le(out, n_kv_out);
        // KV section (shard 1's entries, minus split.* keys).
        if (!kv_bytes.empty()) {
            out.write(reinterpret_cast<const char*>(kv_bytes.data()),
                      static_cast<std::streamsize>(kv_bytes.size()));
        }
        // Tensor infos.
        for (const auto& ot : out_tensors) {
            write_u64_le(out, ot.name.size());
            out.write(ot.name.data(), static_cast<std::streamsize>(ot.name.size()));
            write_u32_le(out, ot.n_dims);
            for (uint32_t d = 0; d < ot.n_dims; ++d) write_u64_le(out, ot.dims[d]);
            write_u32_le(out, static_cast<uint32_t>(ot.type));
            write_u64_le(out, ot.new_offset);
        }
        // Pad to the data section start.
        uint64_t pos = out_kv_end + infos_bytes;
        std::vector<char> zeros(4096, 0);
        while (pos < data_start) {
            uint64_t chunk = std::min<uint64_t>(data_start - pos, zeros.size());
            out.write(zeros.data(), static_cast<std::streamsize>(chunk));
            pos += chunk;
        }
        // Tensor data in order, aligned.
        for (const auto& ot : out_tensors) {
            uint64_t want = data_start + ot.new_offset;
            while (pos < want) {
                uint64_t chunk = std::min<uint64_t>(want - pos, zeros.size());
                out.write(zeros.data(), static_cast<std::streamsize>(chunk));
                pos += chunk;
            }
            if (ot.byte_size > 0) {
                out.write(reinterpret_cast<const char*>(ot.src),
                          static_cast<std::streamsize>(ot.byte_size));
                pos += ot.byte_size;
            }
        }
        out.flush();
        if (!out) {
            err_msg = "Write failed while merging to " + out_path;
            out.close();
            fs::remove(out_path, ec);
            return false;
        }
    }

    // Verify: re-open the merged file and spot-check bytes.
    ModelFile merged;
    std::string verr;
    if (!merged.open(out_path, verr)) {
        err_msg = "Merged file failed to re-open: " + verr;
        fs::remove(out_path, ec);
        return false;
    }
    if (merged.tensor_names().size() != total_tensors) {
        err_msg = "Merged file tensor count mismatch (" +
                  std::to_string(merged.tensor_names().size()) + " vs " +
                  std::to_string(total_tensors) + ")";
        fs::remove(out_path, ec);
        return false;
    }
    auto check_tensor = [&](const OutTensor& ot) {
        const TensorDescriptor* td = merged.find_tensor(ot.name);
        if (!td || td->byte_size != ot.byte_size) return false;
        if (ot.byte_size == 0) return true;
        return std::memcmp(td->mapped_ptr, ot.src, static_cast<size_t>(ot.byte_size)) == 0;
    };
    if (!out_tensors.empty()) {
        // First, middle, and last tensor byte-checked against the shards.
        if (!check_tensor(out_tensors.front()) ||
            !check_tensor(out_tensors[out_tensors.size() / 2]) ||
            !check_tensor(out_tensors.back())) {
            err_msg = "Merged file byte verification failed (tensor data mismatch)";
            fs::remove(out_path, ec);
            return false;
        }
    }
    uint64_t merged_size = merged.file_size();
    std::cout << "[MERGE] Merged " << shard_paths.size() << " shard(s) -> " << out_path << "\n"
              << "  tensors: " << total_tensors << ", arch: " << (arch.empty() ? "?" : arch)
              << ", size: " << (merged_size / (1024 * 1024)) << " MB, verified OK.\n";
    return true;
}

// ---------------------------------------------------------------------------
// quantize_gguf_file (workstream G, 2026-11-17)
// ---------------------------------------------------------------------------

bool parse_quant_type_name(const std::string& name, GGMLType& out) {
    // Only types with a quantizer in quant.cpp (verified bit-exact vs ggml
    // for the wave-1 set; Q4_0/Q8_0/Q4_K/Q5_K/Q6_K pre-date it). IQ types
    // have no quantizer here and F32 is not a quantization.
    static const std::pair<const char*, GGMLType> kTargets[] = {
        {"Q4_0", GGMLType::Q4_0}, {"Q8_0", GGMLType::Q8_0},
        {"Q4_K", GGMLType::Q4_K}, {"Q5_K", GGMLType::Q5_K},
        {"Q6_K", GGMLType::Q6_K}, {"Q4_1", GGMLType::Q4_1},
        {"Q5_0", GGMLType::Q5_0}, {"Q5_1", GGMLType::Q5_1},
        {"Q8_1", GGMLType::Q8_1}, {"Q2_K", GGMLType::Q2_K},
        {"Q3_K", GGMLType::Q3_K}, {"Q8_K", GGMLType::Q8_K},
    };
    for (const auto& kv : kTargets) {
        if (name == kv.first) { out = kv.second; return true; }
    }
    return false;
}

namespace {

// Source types dequantize_row() can actually decode (its switch, minus the
// default branch that warns and returns zeros). A tensor in any other type
// is copied byte-for-byte, never dequantized.
bool has_dequantizer(GGMLType t) {
    switch (t) {
        case GGMLType::F32: case GGMLType::F16: case GGMLType::BF16:
        case GGMLType::FP8_E4M3:
        case GGMLType::Q4_0: case GGMLType::Q4_1: case GGMLType::Q5_0:
        case GGMLType::Q5_1: case GGMLType::Q8_0: case GGMLType::Q8_1:
        case GGMLType::IQ4_NL:
        case GGMLType::Q2_K: case GGMLType::Q3_K: case GGMLType::Q4_K:
        case GGMLType::Q5_K: case GGMLType::Q6_K: case GGMLType::Q8_K:
        case GGMLType::IQ2_S: case GGMLType::IQ3_S: case GGMLType::IQ4_XS:
        case GGMLType::IQ2_XXS: case GGMLType::IQ2_XS: case GGMLType::IQ3_XXS:
        case GGMLType::IQ1_S: case GGMLType::IQ1_M:
            return true;
        default:
            return false;
    }
}

using QuantizeRowFn = void (*)(const float*, void*, int64_t);

QuantizeRowFn quantizer_for(GGMLType t) {
    switch (t) {
        case GGMLType::Q4_0: return quantize_row_q4_0;
        case GGMLType::Q8_0: return quantize_row_q8_0;
        case GGMLType::Q4_K: return quantize_row_q4_K;
        case GGMLType::Q5_K: return quantize_row_q5_K;
        case GGMLType::Q6_K: return quantize_row_q6_K;
        case GGMLType::Q4_1: return quantize_row_q4_1;
        case GGMLType::Q5_0: return quantize_row_q5_0;
        case GGMLType::Q5_1: return quantize_row_q5_1;
        case GGMLType::Q8_1: return quantize_row_q8_1;
        case GGMLType::Q2_K: return quantize_row_q2_K;
        case GGMLType::Q3_K: return quantize_row_q3_K;
        case GGMLType::Q8_K: return quantize_row_q8_K;
        default: return nullptr;
    }
}

} // namespace

bool quantize_gguf_file(const std::string& in_path, GGMLType target_type,
                        const std::string& out_path, std::string& err_msg) {
    namespace fs = std::filesystem;
    QuantizeRowFn quant_fn = quantizer_for(target_type);
    if (quant_fn == nullptr) {
        err_msg = "No quantizer in quant.cpp for target type " +
                  std::string(ggml_type_name(target_type)) +
                  " (supported: Q4_0 Q8_0 Q4_K Q5_K Q6_K Q4_1 Q5_0 Q5_1 Q8_1 "
                  "Q2_K Q3_K Q8_K)";
        return false;
    }
    std::error_code ec;
    if (fs::equivalent(fs::path(in_path), fs::path(out_path), ec) && !ec) {
        err_msg = "Output path must differ from the input file: " + out_path;
        return false;
    }

    // Header first (exact version / counts).
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    {
        std::ifstream in(in_path, std::ios::binary);
        if (!in) {
            err_msg = "Cannot open GGUF file: " + in_path;
            return false;
        }
        char hdr[24] = {0};
        in.read(hdr, 24);
        if (in.gcount() != 24 || std::memcmp(hdr, "GGUF", 4) != 0) {
            err_msg = "Not a GGUF file (bad magic): " + in_path;
            return false;
        }
        std::memcpy(&version, hdr + 4, 4);
        std::memcpy(&n_tensors, hdr + 8, 8);
        std::memcpy(&n_kv, hdr + 16, 8);
        if (version != 2 && version != 3) {
            err_msg = "Unsupported GGUF version " + std::to_string(version) + ": " + in_path;
            return false;
        }
    }

    ModelFile input;
    std::string open_err;
    if (!input.open(in_path, open_err)) {
        err_msg = "Cannot parse GGUF file " + in_path + ": " + open_err;
        return false;
    }
    if (input.is_safetensors()) {
        err_msg = "SafeTensors input is not supported by convert --quantize "
                  "(load it directly with the engine, or convert it to GGUF first)";
        return false;
    }
    if (input.tensor_names().size() != n_tensors) {
        err_msg = "Header tensor count disagrees with parsed tensors in " + in_path;
        return false;
    }
    uint32_t alignment = input.get_u32("general.alignment", 32);
    if (alignment == 0) alignment = 32;

    const size_t tgt_block = ggml_blck_size(target_type);

    struct OutTensor {
        std::string name;
        uint32_t n_dims = 0;
        uint64_t dims[4] = {1, 1, 1, 1};
        GGMLType type = GGMLType::F32;
        uint64_t byte_size = 0;
        uint64_t new_offset = 0;
        std::vector<uint8_t> data; // owned when re-quantized; empty = copy from input
        const uint8_t* src = nullptr; // input bytes when copied verbatim
    };
    std::vector<OutTensor> out_tensors;
    out_tensors.reserve(input.tensor_names().size());

    size_t n_quantized = 0;
    std::map<std::string, size_t> kept_reasons;
    for (const auto& name : input.tensor_names()) {
        const TensorDescriptor* td = input.find_tensor(name);
        OutTensor ot;
        ot.name = name;
        ot.n_dims = td->n_dims;
        for (uint32_t d = 0; d < td->n_dims; ++d) ot.dims[d] = td->dims[d];
        ot.type = td->type;
        ot.byte_size = td->byte_size;
        ot.src = static_cast<const uint8_t*>(td->mapped_ptr);

        uint64_t num_elems = 1;
        for (uint32_t d = 0; d < td->n_dims; ++d) num_elems *= td->dims[d];
        const uint64_t ne0 = (td->n_dims >= 1) ? td->dims[0] : 0;

        std::string keep_reason;
        if (td->n_dims <= 1) {
            keep_reason = "1-D tensor (norm/bias) kept as " + std::string(ggml_type_name(td->type));
        } else if (num_elems == 0) {
            keep_reason = "empty tensor kept as " + std::string(ggml_type_name(td->type));
        } else if (td->type == target_type) {
            keep_reason = "already " + std::string(ggml_type_name(target_type));
        } else if (!has_dequantizer(td->type)) {
            keep_reason = "no dequantizer for source type " + std::string(ggml_type_name(td->type));
        } else if (ne0 % ggml_blck_size(td->type) != 0) {
            keep_reason = "source row length " + std::to_string(ne0) + " is not a multiple of source block " +
                          std::to_string(ggml_blck_size(td->type)) + " (" +
                          std::string(ggml_type_name(td->type)) + ")";
        } else if (ne0 % tgt_block != 0) {
            keep_reason = "row length " + std::to_string(ne0) + " is not a multiple of " +
                          std::string(ggml_type_name(target_type)) + " block " +
                          std::to_string(tgt_block);
        }
        if (!keep_reason.empty()) {
            kept_reasons[keep_reason]++;
            if (ot.byte_size > 0 && !ot.src) {
                err_msg = "Tensor has no data pointer in input: " + name;
                return false;
            }
            out_tensors.push_back(std::move(ot));
            continue;
        }

        // Re-quantize row by row (rows = num_elems / ne0, so 3-D MoE
        // expert tensors work too: each ne0-length row is one block row).
        const uint64_t rows = num_elems / ne0;
        const uint64_t src_row_bytes = tensor_byte_size(td->type, ne0);
        const uint64_t dst_row_bytes = tensor_byte_size(target_type, ne0);
        std::vector<uint8_t> dst(static_cast<size_t>(dst_row_bytes * rows));
        std::vector<float> f32row(static_cast<size_t>(ne0));
        const uint8_t* src_bytes = ot.src;
        if (src_bytes == nullptr) {
            err_msg = "Tensor has no data pointer in input: " + name;
            return false;
        }
        for (uint64_t r = 0; r < rows; ++r) {
            dequantize_row(td->type, src_bytes + r * src_row_bytes,
                           f32row.data(), static_cast<int64_t>(ne0));
            quant_fn(f32row.data(), dst.data() + r * dst_row_bytes,
                     static_cast<int64_t>(ne0));
        }
        ot.type = target_type;
        ot.byte_size = dst.size();
        ot.data = std::move(dst);
        ot.src = nullptr;
        out_tensors.push_back(std::move(ot));
        n_quantized++;
    }

    // KV section: input's entries verbatim, minus split.* keys.
    uint64_t kv_end = input.kv_section_end();
    if (kv_end < 24 || kv_end > input.file_size()) {
        err_msg = "Input has an invalid KV section end";
        return false;
    }
    std::vector<uint8_t> kv_bytes;
    uint64_t n_kv_out = n_kv;
    {
        uint64_t filtered_n = 0;
        std::vector<uint8_t> filtered;
        if (filter_split_kv(input.mapped_base(), kv_end, n_kv, filtered, filtered_n)) {
            kv_bytes = std::move(filtered);
            n_kv_out = filtered_n;
        } else {
            std::cerr << "[QUANTIZE] WARNING: could not walk the KV section — copying it verbatim.\n";
            kv_bytes.assign(input.mapped_base() + 24, input.mapped_base() + kv_end);
            n_kv_out = n_kv;
        }
    }
    uint64_t out_kv_end = 24 + kv_bytes.size();

    uint64_t infos_bytes = 0;
    for (const auto& ot : out_tensors) {
        infos_bytes += 8 + ot.name.size() + 4 + 8ull * ot.n_dims + 4 + 8;
    }
    uint64_t data_start = align_up_u64(out_kv_end + infos_bytes, alignment);
    uint64_t cursor = 0;
    for (auto& ot : out_tensors) {
        ot.new_offset = align_up_u64(cursor, alignment);
        cursor = ot.new_offset + ot.byte_size;
    }

    fs::path out_p(out_path);
    if (out_p.has_parent_path()) {
        fs::create_directories(out_p.parent_path(), ec);
        if (ec) {
            err_msg = "Cannot create output directory: " + out_p.parent_path().string();
            return false;
        }
    }
    {
        std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            err_msg = "Cannot open output file for writing: " + out_path;
            return false;
        }
        out.write("GGUF", 4);
        write_u32_le(out, version);
        write_u64_le(out, n_tensors);
        write_u64_le(out, n_kv_out);
        if (!kv_bytes.empty()) {
            out.write(reinterpret_cast<const char*>(kv_bytes.data()),
                      static_cast<std::streamsize>(kv_bytes.size()));
        }
        for (const auto& ot : out_tensors) {
            write_u64_le(out, ot.name.size());
            out.write(ot.name.data(), static_cast<std::streamsize>(ot.name.size()));
            write_u32_le(out, ot.n_dims);
            for (uint32_t d = 0; d < ot.n_dims; ++d) write_u64_le(out, ot.dims[d]);
            write_u32_le(out, static_cast<uint32_t>(ot.type));
            write_u64_le(out, ot.new_offset);
        }
        uint64_t pos = out_kv_end + infos_bytes;
        std::vector<char> zeros(4096, 0);
        while (pos < data_start) {
            uint64_t chunk = std::min<uint64_t>(data_start - pos, zeros.size());
            out.write(zeros.data(), static_cast<std::streamsize>(chunk));
            pos += chunk;
        }
        for (const auto& ot : out_tensors) {
            uint64_t want = data_start + ot.new_offset;
            while (pos < want) {
                uint64_t chunk = std::min<uint64_t>(want - pos, zeros.size());
                out.write(zeros.data(), static_cast<std::streamsize>(chunk));
                pos += chunk;
            }
            if (ot.byte_size > 0) {
                const uint8_t* src = ot.data.empty() ? ot.src : ot.data.data();
                out.write(reinterpret_cast<const char*>(src),
                          static_cast<std::streamsize>(ot.byte_size));
                pos += ot.byte_size;
            }
        }
        out.flush();
        if (!out) {
            err_msg = "Write failed while quantizing to " + out_path;
            out.close();
            fs::remove(out_path, ec);
            return false;
        }
    }

    // Verify: re-open and check tensor count, types, and byte sizes.
    ModelFile check;
    std::string verr;
    if (!check.open(out_path, verr)) {
        err_msg = "Quantized file failed to re-open: " + verr;
        fs::remove(out_path, ec);
        return false;
    }
    if (check.tensor_names().size() != n_tensors) {
        err_msg = "Quantized file tensor count mismatch";
        fs::remove(out_path, ec);
        return false;
    }
    for (const auto& ot : out_tensors) {
        const TensorDescriptor* td = check.find_tensor(ot.name);
        if (!td || td->type != ot.type || td->byte_size != ot.byte_size) {
            err_msg = "Quantized file verification failed for tensor: " + ot.name;
            fs::remove(out_path, ec);
            return false;
        }
    }
    uint64_t out_size = check.file_size();
    std::cout << "[QUANTIZE] " << in_path << " -> " << out_path << " ("
              << ggml_type_name(target_type) << ")\n"
              << "  tensors re-quantized: " << n_quantized << " / " << n_tensors
              << ", size: " << (out_size / (1024 * 1024)) << " MB, verified OK.\n";
    for (const auto& kv : kept_reasons) {
        std::cout << "  kept " << kv.second << " tensor(s): " << kv.first << "\n";
    }
    return true;
}

} // namespace moe

