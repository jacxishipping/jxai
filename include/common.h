#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace moe {

// High-resolution timer helper
inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline uint64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

// Tensor Data Types (GGUF v3 + SafeTensors standard enum values)
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ4_NL  = 20,
    BF16    = 30,
    FP8_E4M3 = 31,
};

inline const char* ggml_type_name(GGMLType t) {
    switch (t) {
        case GGMLType::F32:      return "F32";
        case GGMLType::F16:      return "F16";
        case GGMLType::BF16:     return "BF16";
        case GGMLType::Q4_0:     return "Q4_0";
        case GGMLType::Q4_1:     return "Q4_1";
        case GGMLType::Q5_0:     return "Q5_0";
        case GGMLType::Q5_1:     return "Q5_1";
        case GGMLType::Q8_0:     return "Q8_0";
        case GGMLType::Q8_1:     return "Q8_1";
        case GGMLType::Q4_K:     return "Q4_K";
        case GGMLType::Q6_K:     return "Q6_K";
        case GGMLType::IQ4_NL:   return "IQ4_NL";
        case GGMLType::FP8_E4M3: return "FP8_E4M3";
        default:                 return "UNKNOWN";
    }
}

// Returns block size (number of elements per quantization block)
inline size_t ggml_blck_size(GGMLType t) {
    switch (t) {
        case GGMLType::F32:
        case GGMLType::F16:
        case GGMLType::BF16:
        case GGMLType::FP8_E4M3:
            return 1;
        case GGMLType::Q4_0:
        case GGMLType::Q4_1:
        case GGMLType::Q5_0:
        case GGMLType::Q5_1:
        case GGMLType::Q8_0:
        case GGMLType::IQ4_NL:
            return 32;
        case GGMLType::Q4_K:
        case GGMLType::Q6_K:
            return 256;
        default:
            return 1;
    }
}

// Returns byte size of a single quantization block
inline size_t ggml_type_size(GGMLType t) {
    switch (t) {
        case GGMLType::F32:      return 4;
        case GGMLType::F16:      return 2;
        case GGMLType::BF16:     return 2;
        case GGMLType::FP8_E4M3: return 1;
        case GGMLType::Q4_0:     return 18;  // 2B fp16 d + 16B nibbles (32 weights)
        case GGMLType::Q4_1:     return 20;  // 2B d + 2B m + 16B nibbles
        case GGMLType::Q8_0:     return 34;  // 2B fp16 d + 32B int8 (32 weights)
        case GGMLType::IQ4_NL:   return 18;  // 2B fp16 d + 16B non-linear indices
        case GGMLType::Q4_K:     return 144; // 2B d + 2B dmin + 12B scales + 128B qs (256 weights)
        case GGMLType::Q6_K:     return 210;
        default:                 return 4;
    }
}

inline size_t tensor_byte_size(GGMLType t, uint64_t num_elements) {
    size_t blck = ggml_blck_size(t);
    size_t tsize = ggml_type_size(t);
    return ((num_elements + blck - 1) / blck) * tsize;
}

// Hardware & System Telemetry
struct HardwareSpecs {
    std::string cpu_brand;
    uint32_t logical_cores = 1;
    bool has_avx2 = false;
    bool has_avx512 = false;
    bool has_neon = false;
    std::string simd_backend;
    std::string io_backend;
    uint64_t total_ram_bytes = 0;
    uint64_t avail_ram_bytes = 0;
    uint64_t vram_budget_bytes = 0;
};

HardwareSpecs query_hardware_specs();

// Rolling Latency Snapshot for /profile
struct ProfileSample {
    uint64_t timestamp_ms = 0;
    uint32_t token_id = 0;
    double wall_ms = 0.0;
    double io_wait_ms = 0.0;
    double matmul_ms = 0.0;
    double attention_ms = 0.0;
    double lm_head_ms = 0.0;
    double tok_per_sec = 0.0;
    uint32_t cache_hits = 0;
    uint32_t disk_reads = 0;
    uint32_t cache_routed = 0;
};

} // namespace moe

