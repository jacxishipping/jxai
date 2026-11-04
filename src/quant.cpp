#include "quant.h"
#include <algorithm>
#include <thread>

#if defined(__AVX2__) || defined(_M_AVX2) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace moe {

const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

HardwareSpecs query_hardware_specs() {
    HardwareSpecs specs;
    specs.logical_cores = std::max(1u, std::thread::hardware_concurrency());
    specs.cpu_brand = "x86_64 Multi-Core Processor";

#if defined(__x86_64__) || defined(_M_X64)
    int cpuInfo[4] = {0};
#ifdef _WIN32
    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = cpuInfo[0];
    char brand[0x40] = {0};
    if (nExIds >= 0x80000004) {
        __cpuid(reinterpret_cast<int*>(brand), 0x80000002);
        __cpuid(reinterpret_cast<int*>(brand + 16), 0x80000003);
        __cpuid(reinterpret_cast<int*>(brand + 32), 0x80000004);
        specs.cpu_brand = brand;
        // trim leading spaces
        size_t first = specs.cpu_brand.find_first_not_of(' ');
        if (first != std::string::npos) specs.cpu_brand = specs.cpu_brand.substr(first);
    }
    __cpuidex(cpuInfo, 7, 0);
    specs.has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
    specs.has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        specs.has_avx2 = (ebx & (1 << 5)) != 0;
        specs.has_avx512 = (ebx & (1 << 16)) != 0;
    }
#endif
#endif

    if (specs.has_avx512) {
        specs.simd_backend = "AVX-512 + AVX2 FMA";
    } else if (specs.has_avx2) {
        specs.simd_backend = "AVX2 + FMA3 Vectorized";
    } else {
        specs.simd_backend = "Scalar IEEE-754";
    }

#ifdef _WIN32
    specs.io_backend = "Win32 IOCP Overlapped Async I/O";
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        specs.total_ram_bytes = memInfo.ullTotalPhys;
        specs.avail_ram_bytes = memInfo.ullAvailPhys;
    }
#else
    specs.io_backend = "Linux io_uring + mmap Direct I/O";
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        specs.total_ram_bytes = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
        specs.avail_ram_bytes = specs.total_ram_bytes / 2;
    }
#endif
    specs.vram_budget_bytes = std::min<uint64_t>(specs.total_ram_bytes / 4, 8ULL * 1024 * 1024 * 1024);
    return specs;
}

// IEEE 754 FP16 <-> FP32 conversion
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t fbits = 0;

    if (exp == 0) {
        if (mant == 0) {
            fbits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            fbits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        fbits = sign | 0x7F800000 | (mant << 13);
    } else {
        fbits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &fbits, sizeof(float));
    return out;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t fbits;
    std::memcpy(&fbits, &f, sizeof(float));
    uint16_t sign = static_cast<uint16_t>((fbits >> 16) & 0x8000);
    int32_t exp = ((fbits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = fbits & 0x007FFFFF;

    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= 0x00800000;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        return sign | static_cast<uint16_t>(mant >> shift);
    } else if (exp >= 31) {
        return sign | 0x7C00;
    }
    return sign | static_cast<uint16_t>(exp << 10) | static_cast<uint16_t>(mant >> 13);
}

float bf16_to_fp32(uint16_t b) {
    uint32_t fbits = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &fbits, sizeof(float));
    return out;
}

uint16_t fp32_to_bf16(float f) {
    uint32_t fbits;
    std::memcpy(&fbits, &f, sizeof(float));
    uint32_t rounding_bias = ((fbits >> 16) & 1) + 0x7FFF;
    return static_cast<uint16_t>((fbits + rounding_bias) >> 16);
}

// FP8 E4M3 <-> FP32
float fp8_e4m3_to_fp32(uint8_t v) {
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp  = (v >> 3) & 0x0F;
    uint32_t mant = v & 0x07;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float val;
    if (exp == 0) {
        val = std::ldexp(static_cast<float>(mant) / 8.0f, -6);
    } else {
        val = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, static_cast<int>(exp) - 7);
    }
    return sign ? -val : val;
}

uint8_t fp32_to_fp8_e4m3(float f) {
    if (f == 0.0f || std::isnan(f)) return 0;
    uint8_t sign = (f < 0.0f) ? 0x80 : 0x00;
    float af = std::min(std::fabs(f), 448.0f);
    int exp = 0;
    float norm = std::frexp(af, &exp); // af = norm * 2^exp, norm in [0.5, 1)
    norm *= 2.0f;
    exp -= 1;
    int biased_exp = exp + 7;
    if (biased_exp <= 0) {
        int mant = static_cast<int>(std::round(af * std::ldexp(1.0f, 6) * 8.0f));
        mant = std::clamp(mant, 0, 7);
        return sign | static_cast<uint8_t>(mant);
    }
    if (biased_exp >= 15) {
        return sign | 0x7E;
    }
    int mant = static_cast<int>(std::round((norm - 1.0f) * 8.0f));
    if (mant >= 8) {
        mant = 0;
        biased_exp++;
        if (biased_exp >= 15) return sign | 0x7E;
    }
    return sign | static_cast<uint8_t>((biased_exp << 3) | (mant & 0x07));
}

// ============================================================================
// Quantization routines
// ============================================================================

void quantize_row_q4_0(const float* x, void* vy, int64_t k) {
    const int64_t nb = k / 32;
    auto* y = static_cast<block_q4_0*>(vy);

    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        float maxv = 0.0f;
        for (int j = 0; j < 32; ++j) {
            float v = x[i * 32 + j];
            if (std::fabs(v) > amax) {
                amax = std::fabs(v);
                maxv = v;
            }
        }
        const float d = maxv / -8.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < 16; ++j) {
            const float x0 = x[i * 32 + j] * id;
            const float x1 = x[i * 32 + 16 + j] * id;
            const uint8_t xi0 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(x0 + 8.0f)), 0, 15));
            const uint8_t xi1 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(x1 + 8.0f)), 0, 15));
            y[i].qs[j] = xi0 | (xi1 << 4);
        }
    }
}

void dequantize_row_q4_0(const void* vx, float* y, int64_t k) {
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_q4_0*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 16; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;
            y[i * 32 + j]      = x0 * d;
            y[i * 32 + j + 16] = x1 * d;
        }
    }
}

void quantize_row_q8_0(const float* x, void* vy, int64_t k) {
    const int64_t nb = k / 32;
    auto* y = static_cast<block_q8_0*>(vy);

    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            amax = std::max(amax, std::fabs(x[i * 32 + j]));
        }
        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < 32; ++j) {
            y[i].qs[j] = static_cast<int8_t>(std::clamp(static_cast<int>(std::round(x[i * 32 + j] * id)), -127, 127));
        }
    }
}

void dequantize_row_q8_0(const void* vx, float* y, int64_t k) {
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_q8_0*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 32; ++j) {
            y[i * 32 + j] = x[i].qs[j] * d;
        }
    }
}

static uint8_t nearest_iq4nl_index(float val) {
    uint8_t best = 0;
    float best_err = std::fabs(val - static_cast<float>(kvalues_iq4nl[0]));
    for (uint8_t idx = 1; idx < 16; ++idx) {
        float err = std::fabs(val - static_cast<float>(kvalues_iq4nl[idx]));
        if (err < best_err) {
            best_err = err;
            best = idx;
        }
    }
    return best;
}

void quantize_row_iq4_nl(const float* x, void* vy, int64_t k) {
    const int64_t nb = k / 32;
    auto* y = static_cast<block_iq4_nl*>(vy);

    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            amax = std::max(amax, std::fabs(x[i * 32 + j]));
        }
        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < 16; ++j) {
            uint8_t i0 = nearest_iq4nl_index(x[i * 32 + j] * id);
            uint8_t i1 = nearest_iq4nl_index(x[i * 32 + 16 + j] * id);
            y[i].qs[j] = i0 | (i1 << 4);
        }
    }
}

void dequantize_row_iq4_nl(const void* vx, float* y, int64_t k) {
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_iq4_nl*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 16; ++j) {
            const uint8_t i0 = x[i].qs[j] & 0x0F;
            const uint8_t i1 = x[i].qs[j] >> 4;
            y[i * 32 + j]      = d * static_cast<float>(kvalues_iq4nl[i0]);
            y[i * 32 + 16 + j] = d * static_cast<float>(kvalues_iq4nl[i1]);
        }
    }
}

void quantize_row_q4_K(const float* x, void* vy, int64_t k) {
    const int64_t nb = k / 256;
    auto* y = static_cast<block_q4_K*>(vy);

    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * 256;
        float amax = 0.0f;
        for (int j = 0; j < 256; ++j) {
            amax = std::max(amax, std::fabs(xb[j]));
        }
        float d = amax / 7.5f;
        y[i].d = fp32_to_fp16(d);
        y[i].dmin = fp32_to_fp16(d);
        std::memset(y[i].scales, 1, 12); // unit sub-block scale=1, min=1 -> offset by 8
        const float actual_d = fp16_to_fp32(y[i].d);
        const float id = (actual_d != 0.0f) ? (1.0f / actual_d) : 0.0f;

        for (int blk = 0; blk < 4; ++blk) {
            for (int j = 0; j < 32; ++j) {
                float v0 = xb[blk * 64 + j] * id + 8.0f;
                float v1 = xb[blk * 64 + 32 + j] * id + 8.0f;
                uint8_t q0 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(v0)), 0, 15));
                uint8_t q1 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(v1)), 0, 15));
                y[i].qs[blk * 32 + j] = q0 | (q1 << 4);
            }
        }
    }
}

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize_row_q4_K(const void* vx, float* y, int64_t k) {
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q4_K*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        float* yb = y + i * 256;

        for (int blk = 0; blk < 4; ++blk) {
            const uint8_t sc0 = x[i].scales[blk * 2] & 0x3F;
            const uint8_t sc1 = x[i].scales[blk * 2 + 1] & 0x3F;
            const float dl0 = d * static_cast<float>(sc0 == 0 ? 1 : sc0);
            const float ml0 = dmin * 8.0f * static_cast<float>(sc0 == 0 ? 1 : sc0);
            const float dl1 = d * static_cast<float>(sc1 == 0 ? 1 : sc1);
            const float ml1 = dmin * 8.0f * static_cast<float>(sc1 == 0 ? 1 : sc1);

            for (int j = 0; j < 32; ++j) {
                uint8_t q = x[i].qs[blk * 32 + j];
                yb[blk * 64 + j]      = dl0 * static_cast<float>(q & 0x0F) - ml0;
                yb[blk * 64 + 32 + j] = dl1 * static_cast<float>(q >> 4) - ml1;
            }
        }
    }
}

void dequantize_row_q5_K(const void* vx, float* y, int64_t k) {
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q5_K*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        float* yb = y + i * 256;

        for (int sb = 0; sb < 8; ++sb) {
            uint8_t sc = 0, m = 0;
            get_scale_min_k4(sb, x[i].scales, &sc, &m);
            const float dl = d * static_cast<float>(sc);
            const float ml = dmin * static_cast<float>(m);

            const int qh_offset = (sb * 32) / 8;
            const int qs_offset = sb * 16;
            for (int l = 0; l < 32; ++l) {
                uint8_t q_low = (x[i].qs[qs_offset + l / 2] >> (4 * (l % 2))) & 0x0F;
                uint8_t qh_bit = (x[i].qh[qh_offset + l / 8] >> (l % 8)) & 1;
                uint8_t q5 = q_low | (qh_bit << 4);
                yb[sb * 32 + l] = dl * static_cast<float>(q5) - ml;
            }
        }
    }
}

void dequantize_row_q6_K(const void* vx, float* y, int64_t k) {
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q6_K*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        float* yb = y + i * 256;

        for (int sb = 0; sb < 16; ++sb) {
            const float sc = d * static_cast<float>(x[i].scales[sb]);
            for (int l = 0; l < 16; ++l) {
                int idx = sb * 16 + l;
                uint8_t ql = (x[i].ql[idx / 2] >> (4 * (idx % 2))) & 0x0F;
                uint8_t qh = (x[i].qh[idx / 4] >> (2 * (idx % 4))) & 0x03;
                int8_t q6 = static_cast<int8_t>((ql | (qh << 4))) - 32;
                yb[idx] = sc * static_cast<float>(q6);
            }
        }
    }
}

void dequantize_row_iq3_s(const void* vx, float* y, int64_t k) {
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_iq3_s*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        float* yb = y + i * 256;

        // Unpack 256 3-bit values from 96 bytes (8 weights per 3 bytes)
        for (int g = 0; g < 32; ++g) {
            uint32_t b0 = x[i].qs[g * 3 + 0];
            uint32_t b1 = x[i].qs[g * 3 + 1];
            uint32_t b2 = x[i].qs[g * 3 + 2];
            uint32_t tri = b0 | (b1 << 8) | (b2 << 16);

            for (int w = 0; w < 8; ++w) {
                uint8_t q = (tri >> (w * 3)) & 0x07;
                int global_idx = g * 8 + w;
                uint8_t sign_byte = x[i].signs[global_idx / 64];
                float s = ((sign_byte >> (global_idx % 8)) & 1) ? -1.0f : 1.0f;
                yb[global_idx] = d * (static_cast<float>(q) - 3.5f) * s;
            }
        }
    }
}

void dequantize_row_iq2_s(const void* vx, float* y, int64_t k) {
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_iq2_s*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        float* yb = y + i * 256;

        // Unpack 256 2-bit values from 64 bytes (4 weights per byte)
        for (int b = 0; b < 64; ++b) {
            uint8_t byte_val = x[i].qs[b];
            for (int w = 0; w < 4; ++w) {
                uint8_t q = (byte_val >> (w * 2)) & 0x03;
                int global_idx = b * 4 + w;
                uint8_t sc_val = x[i].scales[global_idx / 32];
                float sc = (sc_val == 0) ? 1.0f : static_cast<float>(sc_val & 0x0F);
                yb[global_idx] = d * sc * (static_cast<float>(q) - 1.5f);
            }
        }
    }
}

void quantize_row_f16(const float* x, void* vy, int64_t k) {
    auto* y = static_cast<uint16_t*>(vy);
    for (int64_t i = 0; i < k; ++i) y[i] = fp32_to_fp16(x[i]);
}

void dequantize_row_f16(const void* vx, float* y, int64_t k) {
    const auto* x = static_cast<const uint16_t*>(vx);
    for (int64_t i = 0; i < k; ++i) y[i] = fp16_to_fp32(x[i]);
}

void quantize_row_bf16(const float* x, void* vy, int64_t k) {
    auto* y = static_cast<uint16_t*>(vy);
    for (int64_t i = 0; i < k; ++i) y[i] = fp32_to_bf16(x[i]);
}

void dequantize_row_bf16(const void* vx, float* y, int64_t k) {
    const auto* x = static_cast<const uint16_t*>(vx);
    for (int64_t i = 0; i < k; ++i) y[i] = bf16_to_fp32(x[i]);
}

void quantize_row_fp8_e4m3(const float* x, void* vy, int64_t k) {
    auto* y = static_cast<uint8_t*>(vy);
    for (int64_t i = 0; i < k; ++i) y[i] = fp32_to_fp8_e4m3(x[i]);
}

void dequantize_row_fp8_e4m3(const void* vx, float* y, int64_t k) {
    const auto* x = static_cast<const uint8_t*>(vx);
    for (int64_t i = 0; i < k; ++i) y[i] = fp8_e4m3_to_fp32(x[i]);
}

void dequantize_row(GGMLType type, const void* vx, float* y, int64_t k) {
    switch (type) {
        case GGMLType::F32:      std::memcpy(y, vx, k * sizeof(float)); break;
        case GGMLType::F16:      dequantize_row_f16(vx, y, k); break;
        case GGMLType::BF16:     dequantize_row_bf16(vx, y, k); break;
        case GGMLType::FP8_E4M3: dequantize_row_fp8_e4m3(vx, y, k); break;
        case GGMLType::Q4_0:     dequantize_row_q4_0(vx, y, k); break;
        case GGMLType::Q8_0:     dequantize_row_q8_0(vx, y, k); break;
        case GGMLType::IQ4_NL:   dequantize_row_iq4_nl(vx, y, k); break;
        case GGMLType::Q4_K:     dequantize_row_q4_K(vx, y, k); break;
        case GGMLType::Q5_K:     dequantize_row_q5_K(vx, y, k); break;
        case GGMLType::Q6_K:     dequantize_row_q6_K(vx, y, k); break;
        case GGMLType::IQ2_S:    dequantize_row_iq2_s(vx, y, k); break;
        case GGMLType::IQ3_S:    dequantize_row_iq3_s(vx, y, k); break;
        default:                 std::memset(y, 0, k * sizeof(float)); break;
    }
}

// ============================================================================
// Vectorized SIMD Dot Products (AVX2 + FMA / Scalar fallback)
// ============================================================================

float vec_dot_f32(const float* a, const float* b, int64_t k) {
    int64_t i = 0;
    float sum = 0.0f;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= k; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) sum += tmp[j];
#endif
    for (; i < k; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float vec_dot_q4_0_f32(const void* vx, const float* y, int64_t k) {
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_q4_0*>(vx);
    float total = 0.0f;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    const __m128i mask_low = _mm_set1_epi8(0x0F);
    const __m256i offset_8 = _mm256_set1_epi32(8);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const __m256 vd = _mm256_set1_ps(d);
        const __m128i qbytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[i].qs));

        const __m128i q_lo = _mm_and_si128(qbytes, mask_low);
        const __m128i q_hi = _mm_and_si128(_mm_srli_epi16(qbytes, 4), mask_low);

        // First 8 of lo
        __m256i i_lo0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q_lo), offset_8);
        __m256i i_lo1 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q_lo, 8)), offset_8);
        __m256i i_hi0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q_hi), offset_8);
        __m256i i_hi1 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q_hi, 8)), offset_8);

        const float* yb = y + i * 32;
        __m256 bsum = _mm256_setzero_ps();
        bsum = _mm256_add_ps(bsum, _mm256_mul_ps(_mm256_cvtepi32_ps(i_lo0), _mm256_loadu_ps(yb + 0)));
        bsum = _mm256_add_ps(bsum, _mm256_mul_ps(_mm256_cvtepi32_ps(i_lo1), _mm256_loadu_ps(yb + 8)));
        bsum = _mm256_add_ps(bsum, _mm256_mul_ps(_mm256_cvtepi32_ps(i_hi0), _mm256_loadu_ps(yb + 16)));
        bsum = _mm256_add_ps(bsum, _mm256_mul_ps(_mm256_cvtepi32_ps(i_hi1), _mm256_loadu_ps(yb + 24)));

        acc = _mm256_add_ps(acc, _mm256_mul_ps(vd, bsum));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) total += tmp[j];
#else
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        float bsum = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;
            bsum += x0 * y[i * 32 + j] + x1 * y[i * 32 + 16 + j];
        }
        total += d * bsum;
    }
#endif
    return total;
}

float vec_dot_q8_0_f32(const void* vx, const float* y, int64_t k) {
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_q8_0*>(vx);
    float total = 0.0f;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const __m256 vd = _mm256_set1_ps(d);
        const float* yb = y + i * 32;
        __m256 bsum = _mm256_setzero_ps();
        for (int chunk = 0; chunk < 4; ++chunk) {
            __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(x[i].qs + chunk * 8));
            __m256i q32 = _mm256_cvtepi8_epi32(q8);
            bsum = _mm256_add_ps(bsum, _mm256_mul_ps(_mm256_cvtepi32_ps(q32), _mm256_loadu_ps(yb + chunk * 8)));
        }
        acc = _mm256_add_ps(acc, _mm256_mul_ps(vd, bsum));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) total += tmp[j];
#else
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        float bsum = 0.0f;
        for (int j = 0; j < 32; ++j) {
            bsum += x[i].qs[j] * y[i * 32 + j];
        }
        total += d * bsum;
    }
#endif
    return total;
}

float vec_dot_iq4_nl_f32(const void* vx, const float* y, int64_t k) {
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_iq4_nl*>(vx);
    float total = 0.0f;

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        float bsum = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const uint8_t i0 = x[i].qs[j] & 0x0F;
            const uint8_t i1 = x[i].qs[j] >> 4;
            bsum += static_cast<float>(kvalues_iq4nl[i0]) * y[i * 32 + j]
                  + static_cast<float>(kvalues_iq4nl[i1]) * y[i * 32 + 16 + j];
        }
        total += d * bsum;
    }
    return total;
}

float vec_dot_q4_K_f32(const void* vx, const float* y, int64_t k) {
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q4_K*>(vx);
    float total = 0.0f;

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        const float* yb = y + i * 256;

        for (int blk = 0; blk < 4; ++blk) {
            const uint8_t sc0 = x[i].scales[blk * 2] & 0x3F;
            const uint8_t sc1 = x[i].scales[blk * 2 + 1] & 0x3F;
            const float dl0 = d * static_cast<float>(sc0 == 0 ? 1 : sc0);
            const float ml0 = dmin * 8.0f * static_cast<float>(sc0 == 0 ? 1 : sc0);
            const float dl1 = d * static_cast<float>(sc1 == 0 ? 1 : sc1);
            const float ml1 = dmin * 8.0f * static_cast<float>(sc1 == 0 ? 1 : sc1);

            for (int j = 0; j < 32; ++j) {
                uint8_t q = x[i].qs[blk * 32 + j];
                float w0 = dl0 * static_cast<float>(q & 0x0F) - ml0;
                float w1 = dl1 * static_cast<float>(q >> 4) - ml1;
                total += w0 * yb[blk * 64 + j] + w1 * yb[blk * 64 + 32 + j];
            }
        }
    }
    return total;
}

float vec_dot_f16_f32(const void* vx, const float* y, int64_t k) {
    const auto* x = static_cast<const uint16_t*>(vx);
    float total = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        total += fp16_to_fp32(x[i]) * y[i];
    }
    return total;
}

float vec_dot_bf16_f32(const void* vx, const float* y, int64_t k) {
    const auto* x = static_cast<const uint16_t*>(vx);
    float total = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        total += bf16_to_fp32(x[i]) * y[i];
    }
    return total;
}

float vec_dot_fp8_f32(const void* vx, const float* y, int64_t k) {
    const auto* x = static_cast<const uint8_t*>(vx);
    float total = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        total += fp8_e4m3_to_fp32(x[i]) * y[i];
    }
    return total;
}

float vec_dot_q5_K_f32(const void* vx, const float* y, int64_t k) {
    alignas(32) float deq[256];
    const int64_t nb = k / 256;
    const auto* blocks = static_cast<const block_q5_K*>(vx);
    float sum = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        dequantize_row_q5_K(&blocks[i], deq, 256);
        sum += vec_dot_f32(deq, y + i * 256, 256);
    }
    return sum;
}

float vec_dot_q6_K_f32(const void* vx, const float* y, int64_t k) {
    alignas(32) float deq[256];
    const int64_t nb = k / 256;
    const auto* blocks = static_cast<const block_q6_K*>(vx);
    float sum = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        dequantize_row_q6_K(&blocks[i], deq, 256);
        sum += vec_dot_f32(deq, y + i * 256, 256);
    }
    return sum;
}

float vec_dot_iq2_s_f32(const void* vx, const float* y, int64_t k) {
    alignas(32) float deq[256];
    const int64_t nb = k / 256;
    const auto* blocks = static_cast<const block_iq2_s*>(vx);
    float sum = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        dequantize_row_iq2_s(&blocks[i], deq, 256);
        sum += vec_dot_f32(deq, y + i * 256, 256);
    }
    return sum;
}

float vec_dot_iq3_s_f32(const void* vx, const float* y, int64_t k) {
    alignas(32) float deq[256];
    const int64_t nb = k / 256;
    const auto* blocks = static_cast<const block_iq3_s*>(vx);
    float sum = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        dequantize_row_iq3_s(&blocks[i], deq, 256);
        sum += vec_dot_f32(deq, y + i * 256, 256);
    }
    return sum;
}

float vec_dot_quant_f32(GGMLType type, const void* vx, const float* y, int64_t k) {
    switch (type) {
        case GGMLType::F32:      return vec_dot_f32(static_cast<const float*>(vx), y, k);
        case GGMLType::F16:      return vec_dot_f16_f32(vx, y, k);
        case GGMLType::BF16:     return vec_dot_bf16_f32(vx, y, k);
        case GGMLType::FP8_E4M3: return vec_dot_fp8_f32(vx, y, k);
        case GGMLType::Q4_0:     return vec_dot_q4_0_f32(vx, y, k);
        case GGMLType::Q8_0:     return vec_dot_q8_0_f32(vx, y, k);
        case GGMLType::IQ4_NL:   return vec_dot_iq4_nl_f32(vx, y, k);
        case GGMLType::Q4_K:     return vec_dot_q4_K_f32(vx, y, k);
        case GGMLType::Q5_K:     return vec_dot_q5_K_f32(vx, y, k);
        case GGMLType::Q6_K:     return vec_dot_q6_K_f32(vx, y, k);
        case GGMLType::IQ2_S:    return vec_dot_iq2_s_f32(vx, y, k);
        case GGMLType::IQ3_S:    return vec_dot_iq3_s_f32(vx, y, k);
        default:                 return 0.0f;
    }
}

void matmul_quant_f32(
    GGMLType type,
    const void* W,
    const float* x,
    float* out,
    int64_t rows,
    int64_t cols
) {
    const size_t row_bytes = tensor_byte_size(type, static_cast<uint64_t>(cols));
    const auto* base = static_cast<const uint8_t*>(W);

#if defined(_OPENMP)
    #pragma omp parallel for if(rows >= 64) schedule(static)
#endif
    for (int64_t r = 0; r < rows; ++r) {
        out[r] = vec_dot_quant_f32(type, base + r * row_bytes, x, cols);
    }
}

} // namespace moe

