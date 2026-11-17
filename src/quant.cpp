#include "quant.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>
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
#if defined(__F16C__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
#else
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
#endif
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

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}

static inline void pack_scale_min_k4(const uint8_t sc[8], const uint8_t m[8], uint8_t q[12]) {
    for (int j = 0; j < 4; ++j) {
        q[j]     = static_cast<uint8_t>((sc[j] & 63) | ((sc[j + 4] >> 4) << 6));
        q[j + 4] = static_cast<uint8_t>((m[j]  & 63) | ((m[j + 4]  >> 4) << 6));
        q[j + 8] = static_cast<uint8_t>((sc[j + 4] & 0x0F) | ((m[j + 4] & 0x0F) << 4));
    }
}

void quantize_row_q4_K(const float* x, void* vy, int64_t k) {
    const int64_t nb = k / 256;
    auto* y = static_cast<block_q4_K*>(vy);

    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * 256;
        float scales_f[8];
        float mins_f[8];
        float max_scale = 0.0f;
        float max_min = 0.0f;

        for (int sb = 0; sb < 8; ++sb) {
            float min_v = 0.0f;
            float max_v = 0.0f;
            for (int l = 0; l < 32; ++l) {
                float v = xb[sb * 32 + l];
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }
            scales_f[sb] = (max_v - min_v) / 15.0f;
            mins_f[sb]   = -min_v;
            if (scales_f[sb] > max_scale) max_scale = scales_f[sb];
            if (mins_f[sb] > max_min)     max_min   = mins_f[sb];
        }

        const float d    = max_scale / 63.0f;
        const float dmin = max_min / 63.0f;
        y[i].d    = fp32_to_fp16(d);
        y[i].dmin = fp32_to_fp16(dmin);

        const float actual_d    = fp16_to_fp32(y[i].d);
        const float actual_dmin = fp16_to_fp32(y[i].dmin);
        const float inv_d    = (actual_d > 0.0f)    ? (1.0f / actual_d)    : 0.0f;
        const float inv_dmin = (actual_dmin > 0.0f) ? (1.0f / actual_dmin) : 0.0f;

        uint8_t sc[8];
        uint8_t m[8];
        for (int sb = 0; sb < 8; ++sb) {
            sc[sb] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(scales_f[sb] * inv_d)), 0, 63));
            m[sb]  = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(mins_f[sb] * inv_dmin)), 0, 63));
        }
        pack_scale_min_k4(sc, m, y[i].scales);

        for (int blk = 0; blk < 4; ++blk) {
            const float dl0 = actual_d * static_cast<float>(sc[blk * 2 + 0]);
            const float ml0 = actual_dmin * static_cast<float>(m[blk * 2 + 0]);
            const float dl1 = actual_d * static_cast<float>(sc[blk * 2 + 1]);
            const float ml1 = actual_dmin * static_cast<float>(m[blk * 2 + 1]);
            const float idl0 = (dl0 > 0.0f) ? (1.0f / dl0) : 0.0f;
            const float idl1 = (dl1 > 0.0f) ? (1.0f / dl1) : 0.0f;

            for (int l = 0; l < 32; ++l) {
                float v0 = (xb[blk * 64 + l] + ml0) * idl0;
                float v1 = (xb[blk * 64 + 32 + l] + ml1) * idl1;
                uint8_t q0 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(v0)), 0, 15));
                uint8_t q1 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(v1)), 0, 15));
                y[i].qs[blk * 32 + l] = q0 | (q1 << 4);
            }
        }
    }
}

void dequantize_row_q4_K(const void* vx, float* y, int64_t k) {
    // Matches llama.cpp ggml block_q4_K (ggml-quants.c): 8 sub-blocks of 32
    // weights with 6-bit scales and mins unpacked via get_scale_min_k4.
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q4_K*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q = x[i].qs;
        const float d    = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        float* yb = y + i * 256;

        int is = 0;
        uint8_t sc = 0, m = 0;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = dmin * static_cast<float>(m);
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = dmin * static_cast<float>(m);

            for (int l = 0; l < 32; ++l) {
                yb[j + l]      = d1 * static_cast<float>(q[l] & 0x0F) - m1;
                yb[j + 32 + l] = d2 * static_cast<float>(q[l] >> 4)   - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

void quantize_row_q5_K(const float* x, void* vy, int64_t k) {
    const int64_t nb = k / 256;
    auto* y = static_cast<block_q5_K*>(vy);

    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * 256;
        float amax = 0.0f;
        for (int j = 0; j < 256; ++j) {
            amax = std::max(amax, std::fabs(xb[j]));
        }
        float d = amax / 15.5f;
        y[i].d = fp32_to_fp16(d);
        y[i].dmin = fp32_to_fp16(d * 16.0f);
        const uint8_t unit_sc[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        const uint8_t unit_m[8]  = {1, 1, 1, 1, 1, 1, 1, 1};
        pack_scale_min_k4(unit_sc, unit_m, y[i].scales);
        std::memset(y[i].qh, 0, 32);

        const float actual_d = fp16_to_fp32(y[i].d);
        const float id = (actual_d != 0.0f) ? (1.0f / actual_d) : 0.0f;

        uint8_t u1 = 1, u2 = 2;
        uint8_t* ql = y[i].qs;
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; ++l) {
                float v0 = xb[j + l]      * id + 16.0f;
                float v1 = xb[j + 32 + l] * id + 16.0f;
                uint8_t q0 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(v0)), 0, 31));
                uint8_t q1 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(v1)), 0, 31));
                ql[l] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
                if (q0 & 0x10) y[i].qh[l] |= u1;
                if (q1 & 0x10) y[i].qh[l] |= u2;
            }
            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void quantize_row_q6_K(const float* x, void* vy, int64_t k) {
    const int64_t nb = k / 256;
    auto* y = static_cast<block_q6_K*>(vy);

    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * 256;
        float amax = 0.0f;
        for (int j = 0; j < 256; ++j) {
            amax = std::max(amax, std::fabs(xb[j]));
        }
        float d = amax / 31.5f;
        y[i].d = fp32_to_fp16(d);
        for (int sb = 0; sb < 16; ++sb) {
            y[i].scales[sb] = 1;
        }
        std::memset(y[i].ql, 0, 128);
        std::memset(y[i].qh, 0, 64);

        const float actual_d = fp16_to_fp32(y[i].d);
        const float id = (actual_d != 0.0f) ? (1.0f / actual_d) : 0.0f;

        // Write bits in the llama.cpp ggml layout (inverse of
        // dequantize_row_q6_K above) so quantize/dequantize round-trips
        // against real GGUF files, not just against itself.
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q0 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(xb[n + l +  0] * id)) + 32, 0, 63));
                const uint8_t q1 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(xb[n + l + 32] * id)) + 32, 0, 63));
                const uint8_t q2 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(xb[n + l + 64] * id)) + 32, 0, 63));
                const uint8_t q3 = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(xb[n + l + 96] * id)) + 32, 0, 63));
                uint8_t* ql = y[i].ql + (n / 2);
                uint8_t* qh = y[i].qh + (n / 4);
                ql[l]      |= (q0 & 0x0F);
                ql[l + 32] |= (q1 & 0x0F);
                ql[l]      |= static_cast<uint8_t>((q2 & 0x0F) << 4);
                ql[l + 32] |= static_cast<uint8_t>((q3 & 0x0F) << 4);
                qh[l] = static_cast<uint8_t>((q0 >> 4) | ((q1 >> 4) << 2) | ((q2 >> 4) << 4) | ((q3 >> 4) << 6));
            }
        }
    }
}

void dequantize_row_q5_K(const void* vx, float* y, int64_t k) {
    // Matches llama.cpp ggml block_q5_K (ggml-quants.c): 64-element strided chunks
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q5_K*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d    = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        float* yb = y + i * 256;

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        uint8_t sc = 0, m = 0;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = dmin * static_cast<float>(m);
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = dmin * static_cast<float>(m);

            for (int l = 0; l < 32; ++l) {
                uint8_t q0 = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                uint8_t q1 = (ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
                yb[j + l]      = d1 * static_cast<float>(q0) - m1;
                yb[j + 32 + l] = d2 * static_cast<float>(q1) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void dequantize_row_q6_K(const void* vx, float* y, int64_t k) {
    // Layout matches llama.cpp ggml block_q6_K (ggml-quants.c): each
    // 128-element half stores four strided elements per (ql[l], qh[l]) pair.
    // The previous version read ql/qh in straight element order, which
    // permutes the weights and scrambles any real Q6_K tensor (e.g. the
    // output.weight LM head of a llama.cpp Q4_0 file).
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q6_K*>(vx);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        float* yb = y + i * 256;

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = static_cast<int8_t>(((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4))) - 32;
                const int8_t q2 = static_cast<int8_t>(((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4))) - 32;
                const int8_t q3 = static_cast<int8_t>(((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4))) - 32;
                const int8_t q4 = static_cast<int8_t>(((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4))) - 32;
                yb[n + l +  0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
                yb[n + l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
                yb[n + l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
                yb[n + l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
            }
            ql += 64;
            qh += 32;
            sc += 8;
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

// ============================================================================
// New quant types: Q4_1 / Q5_0 / Q5_1 / Q8_1 / Q2_K / Q3_K / Q8_K / IQ4_XS
// Layouts and dequantization are ported from llama.cpp ggml-quants.c
// (Apache-2.0, https://github.com/ggml-org/llama.cpp) — the *_ref
// quantizers and dequantize_row_* functions. Where ggml's production
// quantizer runs an RMSE optimiser over the scales (make_qkx2_quants /
// make_qkx3_quants / make_q3_quants), the quantizers here use a simpler
// per-group min/max or abs-max scale selection: the block layout and the
// dequantization path are exact, only the encoder's scale choice is
// simpler (same trade-off as the repo's existing quantize_row_q5_K,
// which uses unit scales). Dequantization of real GGUF files does not
// depend on the encoder at all.
// ============================================================================

// ggml-quants.c nearest_int: round-to-nearest via the magic-number trick.
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

void quantize_row_q4_1(const float* x, void* vy, int64_t k) {
    // ggml-quants.c quantize_row_q4_1_ref, verbatim.
    const int64_t nb = k / 32;
    auto* y = static_cast<block_q4_1*>(vy);
    for (int64_t i = 0; i < nb; ++i) {
        float mn = std::numeric_limits<float>::max();
        float mx = -std::numeric_limits<float>::max();
        for (int j = 0; j < 32; ++j) {
            const float v = x[i * 32 + j];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        const float d  = (mx - mn) / 15.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        y[i].m = fp32_to_fp16(mn);
        for (int j = 0; j < 16; ++j) {
            const float x0 = (x[i * 32 + 0  + j] - mn) * id;
            const float x1 = (x[i * 32 + 16 + j] - mn) * id;
            const uint8_t xi0 = static_cast<uint8_t>(std::min(15, static_cast<int>(x0 + 0.5f)));
            const uint8_t xi1 = static_cast<uint8_t>(std::min(15, static_cast<int>(x1 + 0.5f)));
            y[i].qs[j] = xi0 | (xi1 << 4);
        }
    }
}

void dequantize_row_q4_1(const void* vx, float* y, int64_t k) {
    // ggml-quants.c dequantize_row_q4_1, verbatim.
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_q4_1*>(vx);
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const float m = fp16_to_fp32(x[i].m);
        for (int j = 0; j < 16; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >> 4);
            y[i * 32 + j]      = x0 * d + m;
            y[i * 32 + j + 16] = x1 * d + m;
        }
    }
}

void quantize_row_q5_0(const float* x, void* vy, int64_t k) {
    // ggml-quants.c quantize_row_q5_0_ref, verbatim.
    const int64_t nb = k / 32;
    auto* y = static_cast<block_q5_0*>(vy);
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        float maxv = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const float v = x[i * 32 + j];
            if (std::fabs(v) > amax) { amax = std::fabs(v); maxv = v; }
        }
        const float d  = maxv / -16.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        uint32_t qh = 0;
        for (int j = 0; j < 16; ++j) {
            const float x0 = x[i * 32 + 0  + j] * id;
            const float x1 = x[i * 32 + 16 + j] * id;
            const uint8_t xi0 = static_cast<uint8_t>(std::min(31, static_cast<int>(x0 + 16.5f)));
            const uint8_t xi1 = static_cast<uint8_t>(std::min(31, static_cast<int>(x1 + 16.5f)));
            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + 16);
        }
        std::memcpy(y[i].qh, &qh, sizeof(qh));
    }
}

void dequantize_row_q5_0(const void* vx, float* y, int64_t k) {
    // ggml-quants.c dequantize_row_q5_0, verbatim.
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_q5_0*>(vx);
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        uint32_t qh;
        std::memcpy(&qh, &x[i].qh, sizeof(qh));
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;
            y[i * 32 + j]      = x0 * d;
            y[i * 32 + j + 16] = x1 * d;
        }
    }
}

void quantize_row_q5_1(const float* x, void* vy, int64_t k) {
    // ggml-quants.c quantize_row_q5_1_ref, verbatim.
    const int64_t nb = k / 32;
    auto* y = static_cast<block_q5_1*>(vy);
    for (int64_t i = 0; i < nb; ++i) {
        float mn = std::numeric_limits<float>::max();
        float mx = -std::numeric_limits<float>::max();
        for (int j = 0; j < 32; ++j) {
            const float v = x[i * 32 + j];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        const float d  = (mx - mn) / 31.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        y[i].m = fp32_to_fp16(mn);
        uint32_t qh = 0;
        for (int j = 0; j < 16; ++j) {
            const float x0 = (x[i * 32 + 0  + j] - mn) * id;
            const float x1 = (x[i * 32 + 16 + j] - mn) * id;
            const uint8_t xi0 = static_cast<uint8_t>(x0 + 0.5f);
            const uint8_t xi1 = static_cast<uint8_t>(x1 + 0.5f);
            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + 16);
        }
        std::memcpy(y[i].qh, &qh, sizeof(qh));
    }
}

void dequantize_row_q5_1(const void* vx, float* y, int64_t k) {
    // ggml-quants.c dequantize_row_q5_1, verbatim.
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_q5_1*>(vx);
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const float m = fp16_to_fp32(x[i].m);
        uint32_t qh;
        std::memcpy(&qh, &x[i].qh, sizeof(qh));
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >>   4) | xh_1;
            y[i * 32 + j]      = x0 * d + m;
            y[i * 32 + j + 16] = x1 * d + m;
        }
    }
}

void quantize_row_q8_1(const float* x, void* vy, int64_t k) {
    // ggml-quants.c quantize_row_q8_1_ref, verbatim.
    const int64_t nb = k / 32;
    auto* y = static_cast<block_q8_1*>(vy);
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            amax = std::max(amax, std::fabs(x[i * 32 + j]));
        }
        const float d  = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        int sum = 0;
        for (int j = 0; j < 16; ++j) {
            const float v0 = x[i * 32 + 0  + j] * id;
            const float v1 = x[i * 32 + 16 + j] * id;
            y[i].qs[j]      = static_cast<int8_t>(std::round(v0));
            y[i].qs[j + 16] = static_cast<int8_t>(std::round(v1));
            sum += y[i].qs[j];
            sum += y[i].qs[j + 16];
        }
        y[i].s = fp32_to_fp16(static_cast<float>(sum) * d);
    }
}

void dequantize_row_q8_1(const void* vx, float* y, int64_t k) {
    // No dequantize_row_q8_1 in current ggml (commented out in
    // ggml-quants.h); the per-element value is qs * d, exactly what
    // quantize_row_q8_1_ref above writes. s (block sum) is auxiliary
    // metadata for ggml dot kernels and is not added per element —
    // adding it would break the round-trip (s is ~sum(qs)*d, not 0).
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_q8_1*>(vx);
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 32; ++j) {
            y[i * 32 + j] = x[i].qs[j] * d;
        }
    }
}

void quantize_row_q2_K(const float* x, void* vy, int64_t k) {
    // Scale selection: simple per-16-group min/max (ggml's production
    // quantizer optimises the same scales with make_qkx2_quants).
    // Packing and the final assignment are verbatim from ggml-quants.c
    // quantize_row_q2_K_ref. Note the format cannot represent a nonzero
    // constant 16-group (scale 0, d 0): such a group quantizes to 0,
    // the same limitation as ggml's ref (its L is left uninitialised
    // there); L is zero-initialised here so the result is deterministic.
    const int64_t nb = k / 256;
    auto* y = static_cast<block_q2_K*>(vy);
    uint8_t L[256];
    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * 256;
        float scales_f[16];
        float mins_f[16];
        float max_scale = 0.0f;
        float max_min = 0.0f;
        for (int j = 0; j < 16; ++j) {
            float mn = xb[j * 16];
            float mx = xb[j * 16];
            for (int l = 1; l < 16; ++l) {
                mn = std::min(mn, xb[j * 16 + l]);
                mx = std::max(mx, xb[j * 16 + l]);
            }
            // Dequantized values are dl*q - ml with dl, ml >= 0, so the
            // representable interval is [-ml, 3*dl - ml]. A group that
            // straddles 0 uses ml = -mn and dl = (mx-mn)/3; an all-positive
            // group must use ml = 0 and dl = mx/3 (a range-based dl would
            // clamp most of the group to q = 3). mins_f is therefore never
            // negative, and both 4-bit indices are clamped to [0, 15]:
            // a negative min index would wrap through the uint8_t cast
            // into a large ml and shift the whole group down.
            if (mn < 0.0f) {
                scales_f[j] = (mx - mn) / 3.0f;
                mins_f[j]   = -mn;
            } else {
                scales_f[j] = mx / 3.0f;
                mins_f[j]   = 0.0f;
            }
            max_scale = std::max(max_scale, scales_f[j]);
            max_min   = std::max(max_min, mins_f[j]);
        }
        std::memset(y[i].scales, 0, 16);
        if (max_scale > 0.0f) {
            const float iscale = 15.0f / max_scale;
            for (int j = 0; j < 16; ++j) {
                y[i].scales[j] = static_cast<uint8_t>(std::clamp(nearest_int(iscale * scales_f[j]), 0, 15));
            }
            y[i].d = fp32_to_fp16(max_scale / 15.0f);
        } else {
            y[i].d = fp32_to_fp16(0.0f);
        }
        if (max_min > 0.0f) {
            const float iscale = 15.0f / max_min;
            for (int j = 0; j < 16; ++j) {
                y[i].scales[j] |= static_cast<uint8_t>(std::clamp(nearest_int(iscale * mins_f[j]), 0, 15) << 4);
            }
            y[i].dmin = fp32_to_fp16(max_min / 15.0f);
        } else {
            y[i].dmin = fp32_to_fp16(0.0f);
        }
        std::memset(L, 0, sizeof(L));
        for (int j = 0; j < 16; ++j) {
            const float d = fp16_to_fp32(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = fp16_to_fp32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((xb[16 * j + ii] + dm) / d);
                l = std::clamp(l, 0, 3);
                L[16 * j + ii] = static_cast<uint8_t>(l);
            }
        }
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }
    }
}

void dequantize_row_q2_K(const void* vx, float* y, int64_t k) {
    // ggml-quants.c dequantize_row_q2_K, verbatim.
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q2_K*>(vx);
    for (int64_t i = 0; i < nb; ++i) {
        const float d   = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        const uint8_t* q = x[i].qs;
        float* yp = y + i * 256;
        int is = 0;
        float dl, ml;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *yp++ = dl * static_cast<float>(static_cast<int8_t>((q[l] >> shift) & 3)) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *yp++ = dl * static_cast<float>(static_cast<int8_t>((q[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

void quantize_row_q3_K(const float* x, void* vy, int64_t k) {
    // Scale selection: simple per-16-group abs-max/3 (positive scales;
    // ggml's production quantizer uses make_q3_quants, which may pick
    // negative scales). Packing, assignment, hmask and qs layout are
    // verbatim from ggml-quants.c quantize_row_q3_K_ref.
    const int64_t nb = k / 256;
    auto* y = static_cast<block_q3_K*>(vy);
    int8_t L[256];
    float scales_f[16];
    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * 256;
        float max_scale = 0.0f;
        for (int j = 0; j < 16; ++j) {
            float a = 0.0f;
            for (int l = 0; l < 16; ++l) a = std::max(a, std::fabs(xb[j * 16 + l]));
            scales_f[j] = a / 3.0f;
            max_scale = std::max(max_scale, scales_f[j]);
        }
        std::memset(y[i].scales, 0, 12);
        if (max_scale > 0.0f) {
            const float iscale = -32.0f / max_scale;
            for (int j = 0; j < 16; ++j) {
                int8_t l = static_cast<int8_t>(nearest_int(iscale * scales_f[j]));
                l = std::clamp<int8_t>(l, -32, 31);
                l = static_cast<int8_t>(l + 32);
                if (j < 8) {
                    y[i].scales[j] = l & 0xF;
                } else {
                    y[i].scales[j - 8] |= ((l & 0xF) << 4);
                }
                l = static_cast<int8_t>(l >> 4);
                y[i].scales[j % 4 + 8] |= (l << (2 * (j / 4)));
            }
            y[i].d = fp32_to_fp16(1.0f / iscale);
        } else {
            y[i].d = fp32_to_fp16(0.0f);
        }
        std::memset(L, 0, sizeof(L));
        int8_t sc;
        for (int j = 0; j < 16; ++j) {
            sc = j < 8 ? (y[i].scales[j] & 0xF) : (y[i].scales[j - 8] >> 4);
            sc = static_cast<int8_t>((sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32);
            const float d = fp16_to_fp32(y[i].d) * sc;
            if (!d) continue;
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(xb[16 * j + ii] / d);
                l = std::clamp(l, -4, 3);
                L[16 * j + ii] = static_cast<int8_t>(l + 4);
            }
        }
        std::memset(y[i].hmask, 0, 32);
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < 256; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == 32) {
                m = 0;
                hm <<= 1;
            }
        }
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }
    }
}

void dequantize_row_q3_K(const void* vx, float* y, int64_t k) {
    // ggml-quants.c dequantize_row_q3_K, verbatim.
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q3_K*>(vx);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
    for (int64_t i = 0; i < nb; ++i) {
        const float d_all = fp16_to_fp32(x[i].d);
        const uint8_t* q = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        uint8_t m = 1;
        std::memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int is = 0;
        float dl;
        float* yp = y + i * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *yp++ = dl * static_cast<float>(static_cast<int8_t>((q[l +  0] >> shift) & 3) - ((hm[l +  0] & m) ? 0 : 4));
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *yp++ = dl * static_cast<float>(static_cast<int8_t>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

void quantize_row_q8_K(const float* x, void* vy, int64_t k) {
    // ggml-quants.c quantize_row_q8_K_ref, verbatim. Note d is FP32.
    const int64_t nb = k / 256;
    auto* y = static_cast<block_q8_K*>(vy);
    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * 256;
        float maxv = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < 256; ++j) {
            const float ax = std::fabs(xb[j]);
            if (ax > amax) { amax = ax; maxv = xb[j]; }
        }
        if (!amax) {
            y[i].d = 0.0f;
            std::memset(y[i].qs, 0, 256);
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }
        const float iscale = -127.0f / maxv;
        for (int j = 0; j < 256; ++j) {
            int v = nearest_int(iscale * xb[j]);
            y[i].qs[j] = static_cast<int8_t>(std::clamp(v, -127, 127));
        }
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = static_cast<int16_t>(sum);
        }
        y[i].d = 1.0f / iscale;
    }
}

void dequantize_row_q8_K(const void* vx, float* y, int64_t k) {
    // ggml-quants.c dequantize_row_q8_K, verbatim (d is FP32).
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q8_K*>(vx);
    for (int64_t i = 0; i < nb; ++i) {
        for (int j = 0; j < 256; ++j) {
            y[i * 256 + j] = x[i].d * static_cast<float>(x[i].qs[j]);
        }
    }
}

void dequantize_row_iq4_xs(const void* vx, float* y, int64_t k) {
    // ggml-quants.c dequantize_row_iq4_xs, verbatim. This is the one IQ
    // type that needs no grid table — only kvalues_iq4nl, which this
    // repo already has. (No quantizer is provided: ggml's
    // quantize_row_iq4_xs_ref needs the full IQ4_XS encoder; the
    // dequantizer is verified in moe_engine test with a hand-built
    // block instead. Real IQ4_XS files dequantize through this path.)
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_iq4_xs*>(vx);
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* qs = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        float* yb = y + i * 256;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((x[i].scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                yb[j +  0] = dl * static_cast<float>(kvalues_iq4nl[qs[j] & 0xf]);
                yb[j + 16] = dl * static_cast<float>(kvalues_iq4nl[qs[j] >>  4]);
            }
            yb += 32;
            qs += 16;
        }
    }
}

void quantize_row_f16(const float* x, void* vy, int64_t k) {
    auto* y = static_cast<uint16_t*>(vy);
    for (int64_t i = 0; i < k; ++i) y[i] = fp32_to_fp16(x[i]);
}

void dequantize_row_f16(const void* vx, float* y, int64_t k) {
    const auto* x = static_cast<const uint16_t*>(vx);
    int64_t i = 0;
#if defined(__F16C__) && defined(__AVX2__)
    for (; i + 8 <= k; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
        _mm256_storeu_ps(y + i, _mm256_cvtph_ps(h));
    }
#endif
    for (; i < k; ++i) y[i] = fp16_to_fp32(x[i]);
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

// Warn once per unsupported type instead of silently producing zeros:
// a model file containing such a type would otherwise generate garbage
// with no indication of why. (Called only from the default branches.)
static void warn_unsupported_quant_once(GGMLType type) {
    static std::atomic<bool> warned[64];
    const uint32_t idx = static_cast<uint32_t>(type);
    if (idx < 64) {
        bool expected = false;
        if (warned[idx].compare_exchange_strong(expected, true)) {
            std::fprintf(stderr,
                "[jxai quant] WARNING: GGML type %s has no dequantizer — "
                "returning zeros. IQ2_XXS/IQ2_XS/IQ3_XXS/IQ1_S/IQ1_M/IQ2_M "
                "need llama.cpp's IQ grid tables (not ported).\n",
                ggml_type_name(type));
        }
    }
}

void dequantize_row(GGMLType type, const void* vx, float* y, int64_t k) {
    switch (type) {
        case GGMLType::F32:      std::memcpy(y, vx, k * sizeof(float)); break;
        case GGMLType::F16:      dequantize_row_f16(vx, y, k); break;
        case GGMLType::BF16:     dequantize_row_bf16(vx, y, k); break;
        case GGMLType::FP8_E4M3: dequantize_row_fp8_e4m3(vx, y, k); break;
        case GGMLType::Q4_0:     dequantize_row_q4_0(vx, y, k); break;
        case GGMLType::Q4_1:     dequantize_row_q4_1(vx, y, k); break;
        case GGMLType::Q5_0:     dequantize_row_q5_0(vx, y, k); break;
        case GGMLType::Q5_1:     dequantize_row_q5_1(vx, y, k); break;
        case GGMLType::Q8_0:     dequantize_row_q8_0(vx, y, k); break;
        case GGMLType::Q8_1:     dequantize_row_q8_1(vx, y, k); break;
        case GGMLType::IQ4_NL:   dequantize_row_iq4_nl(vx, y, k); break;
        case GGMLType::Q2_K:     dequantize_row_q2_K(vx, y, k); break;
        case GGMLType::Q3_K:     dequantize_row_q3_K(vx, y, k); break;
        case GGMLType::Q4_K:     dequantize_row_q4_K(vx, y, k); break;
        case GGMLType::Q5_K:     dequantize_row_q5_K(vx, y, k); break;
        case GGMLType::Q6_K:     dequantize_row_q6_K(vx, y, k); break;
        case GGMLType::Q8_K:     dequantize_row_q8_K(vx, y, k); break;
        case GGMLType::IQ2_S:    dequantize_row_iq2_s(vx, y, k); break;
        case GGMLType::IQ3_S:    dequantize_row_iq3_s(vx, y, k); break;
        case GGMLType::IQ4_XS:   dequantize_row_iq4_xs(vx, y, k); break;
        // Still unsupported (documented, not silent): IQ2_XXS, IQ2_XS,
        // IQ3_XXS, IQ1_S, IQ1_M, IQ2_M — their dequantizers index
        // llama.cpp's IQ grid tables (iq2xxs_grid, iq2xs_grid,
        // iq3xxs_grid, iq1s_grid), which are generated by ggml's
        // iq2xs/iq3xs init code and were not available to port exactly.
        // Also unsupported (pre-existing): I8/I16/I32/I64/F64.
        default:
            warn_unsupported_quant_once(type);
            std::memset(y, 0, k * sizeof(float));
            break;
    }
}

// ============================================================================
// Vectorized SIMD Dot Products (AVX2 + FMA / Scalar fallback)
// ============================================================================

float vec_dot_f32(const float* a, const float* b, int64_t k) {
    int64_t i = 0;
    float sum = 0.0f;
#if defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= k; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= k; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0);
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

        __m256i i_lo0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q_lo), offset_8);
        __m256i i_lo1 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q_lo, 8)), offset_8);
        __m256i i_hi0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q_hi), offset_8);
        __m256i i_hi1 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q_hi, 8)), offset_8);

        const float* yb = y + i * 32;
        __m256 bsum = _mm256_mul_ps(_mm256_cvtepi32_ps(i_lo0), _mm256_loadu_ps(yb + 0));
        bsum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i_lo1), _mm256_loadu_ps(yb + 8),  bsum);
        bsum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i_hi0), _mm256_loadu_ps(yb + 16), bsum);
        bsum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i_hi1), _mm256_loadu_ps(yb + 24), bsum);

        acc = _mm256_fmadd_ps(vd, bsum, acc);
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
            bsum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q32), _mm256_loadu_ps(yb + chunk * 8), bsum);
        }
        acc = _mm256_fmadd_ps(vd, bsum, acc);
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

#if defined(__AVX2__)
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask_low = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const __m256 vd = _mm256_set1_ps(d);
        const __m128i qbytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[i].qs));
        const __m128i q_lo = _mm_and_si128(qbytes, mask_low);
        const __m128i q_hi = _mm_and_si128(_mm_srli_epi16(qbytes, 4), mask_low);

        const __m128i v_lo = _mm_shuffle_epi8(lut, q_lo);
        const __m128i v_hi = _mm_shuffle_epi8(lut, q_hi);

        const float* yb = y + i * 32;
        __m256 bsum = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo)), _mm256_loadu_ps(yb + 0));
        bsum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8))), _mm256_loadu_ps(yb + 8),  bsum);
        bsum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi)),                    _mm256_loadu_ps(yb + 16), bsum);
        bsum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8))), _mm256_loadu_ps(yb + 24), bsum);
        acc = _mm256_fmadd_ps(vd, bsum, acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) total += tmp[j];
#else
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
#endif
    return total;
}

float vec_dot_q4_K_f32(const void* vx, const float* y, int64_t k) {
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q4_K*>(vx);
    float total = 0.0f;

#if defined(__AVX2__)
    const __m128i mask_low = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q = x[i].qs;
        const float d    = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        const float* yb  = y + i * 256;

        int is = 0;
        uint8_t sc = 0, m = 0;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = dmin * static_cast<float>(m);
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = dmin * static_cast<float>(m);

            __m256 qsum1 = _mm256_setzero_ps();
            __m256 qsum2 = _mm256_setzero_ps();
            __m256 ysum1 = _mm256_setzero_ps();
            __m256 ysum2 = _mm256_setzero_ps();

            for (int h = 0; h < 2; ++h) {
                const __m128i qbytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + h * 16));
                const __m128i q_lo = _mm_and_si128(qbytes, mask_low);
                const __m128i q_hi = _mm_and_si128(_mm_srli_epi16(qbytes, 4), mask_low);

                const float* y1 = yb + j + h * 16;
                const float* y2 = yb + j + 32 + h * 16;

                __m256 y1_0 = _mm256_loadu_ps(y1 + 0);
                __m256 y1_1 = _mm256_loadu_ps(y1 + 8);
                __m256 y2_0 = _mm256_loadu_ps(y2 + 0);
                __m256 y2_1 = _mm256_loadu_ps(y2 + 8);

                ysum1 = _mm256_add_ps(ysum1, _mm256_add_ps(y1_0, y1_1));
                ysum2 = _mm256_add_ps(ysum2, _mm256_add_ps(y2_0, y2_1));

                __m256 f_lo0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_lo));
                __m256 f_lo1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(q_lo, 8)));
                __m256 f_hi0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_hi));
                __m256 f_hi1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(q_hi, 8)));

                qsum1 = _mm256_fmadd_ps(f_lo0, y1_0, qsum1);
                qsum1 = _mm256_fmadd_ps(f_lo1, y1_1, qsum1);
                qsum2 = _mm256_fmadd_ps(f_hi0, y2_0, qsum2);
                qsum2 = _mm256_fmadd_ps(f_hi1, y2_1, qsum2);
            }

            __m256 pos_term = _mm256_fmadd_ps(_mm256_set1_ps(d1), qsum1, _mm256_mul_ps(_mm256_set1_ps(d2), qsum2));
            __m256 neg_term = _mm256_fmadd_ps(_mm256_set1_ps(m1), ysum1, _mm256_mul_ps(_mm256_set1_ps(m2), ysum2));
            acc = _mm256_add_ps(acc, _mm256_sub_ps(pos_term, neg_term));

            q += 32;
            is += 2;
        }
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int t = 0; t < 8; ++t) total += tmp[t];
#else
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q = x[i].qs;
        const float d    = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        const float* yb  = y + i * 256;

        int is = 0;
        uint8_t sc = 0, m = 0;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = dmin * static_cast<float>(m);
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = dmin * static_cast<float>(m);

            for (int l = 0; l < 32; ++l) {
                float w0 = d1 * static_cast<float>(q[l] & 0x0F) - m1;
                float w1 = d2 * static_cast<float>(q[l] >> 4)   - m2;
                total += w0 * yb[j + l] + w1 * yb[j + 32 + l];
            }
            q += 32;
            is += 2;
        }
    }
#endif
    return total;
}

float vec_dot_f16_f32(const void* vx, const float* y, int64_t k) {
    const auto* x = static_cast<const uint16_t*>(vx);
    float total = 0.0f;
    int64_t i = 0;
#if defined(__F16C__) && defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= k; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
        __m256 vx_f32 = _mm256_cvtph_ps(h);
        __m256 vy_f32 = _mm256_loadu_ps(y + i);
        acc = _mm256_fmadd_ps(vx_f32, vy_f32, acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) total += tmp[j];
#endif
    for (; i < k; ++i) {
        total += fp16_to_fp32(x[i]) * y[i];
    }
    return total;
}

float vec_dot_bf16_f32(const void* vx, const float* y, int64_t k) {
    const auto* x = static_cast<const uint16_t*>(vx);
    float total = 0.0f;
    int64_t i = 0;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= k; i += 8) {
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
        __m256i fbits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b), 16);
        __m256 vx_f32 = _mm256_castsi256_ps(fbits);
        __m256 vy_f32 = _mm256_loadu_ps(y + i);
        acc = _mm256_fmadd_ps(vx_f32, vy_f32, acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) total += tmp[j];
#endif
    for (; i < k; ++i) {
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
    const int64_t nb = k / 256;
    const auto* blocks = static_cast<const block_q6_K*>(vx);
    float sum = 0.0f;

#if defined(__AVX2__)
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i mask_03 = _mm_set1_epi8(0x03);
    const __m128i offset_32 = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        const uint8_t* ql = blocks[i].ql;
        const uint8_t* qh = blocks[i].qh;
        const int8_t* sc  = blocks[i].scales;
        const float* yb   = y + i * 256;

        for (int n = 0; n < 256; n += 128) {
            for (int is = 0; is < 2; ++is) {
                const int l_off = is * 16;
                const __m128i ql0  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql + l_off));
                const __m128i ql1  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql + l_off + 32));
                const __m128i qh_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh + l_off));

                const __m128i q1 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(ql0, mask_0f), _mm_slli_epi16(_mm_and_si128(qh_v, mask_03), 4)),
                    offset_32
                );
                const __m128i q2 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(ql1, mask_0f), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 2), mask_03), 4)),
                    offset_32
                );
                const __m128i q3 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), mask_0f), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 4), mask_03), 4)),
                    offset_32
                );
                const __m128i q4 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), mask_0f), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 6), mask_03), 4)),
                    offset_32
                );

                auto dot16 = [](__m128i qv, const float* yp) -> __m256 {
                    __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qv));
                    __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qv, 8)));
                    __m256 s0 = _mm256_mul_ps(f0, _mm256_loadu_ps(yp + 0));
                    return _mm256_fmadd_ps(f1, _mm256_loadu_ps(yp + 8), s0);
                };

                const float* y_base = yb + n + l_off;
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * static_cast<float>(sc[is + 0])), dot16(q1, y_base +  0), acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * static_cast<float>(sc[is + 2])), dot16(q2, y_base + 32), acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * static_cast<float>(sc[is + 4])), dot16(q3, y_base + 64), acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * static_cast<float>(sc[is + 6])), dot16(q4, y_base + 96), acc);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) sum += tmp[j];
#else
    alignas(32) float deq[256];
    for (int64_t i = 0; i < nb; ++i) {
        dequantize_row_q6_K(&blocks[i], deq, 256);
        sum += vec_dot_f32(deq, y + i * 256, 256);
    }
#endif
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

// New vec_dot wrappers: dequantize one block at a time, then dot — the
// same pattern as vec_dot_iq2_s_f32 / vec_dot_q5_K_f32 above. Scalar
// dequantization is correct first; the Q8-activation fast paths in
// vec_dot_row_with_q8() do not apply to these types and fall through to
// vec_dot_quant_f32() (never to a silent 0).
template <typename BlockT, int BLCK, void (*DeqFn)(const void*, float*, int64_t)>
static float vec_dot_deq_f32(const void* vx, const float* y, int64_t k) {
    alignas(32) float deq[BLCK];
    const int64_t nb = k / BLCK;
    const auto* blocks = static_cast<const BlockT*>(vx);
    float sum = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        DeqFn(&blocks[i], deq, BLCK);
        sum += vec_dot_f32(deq, y + i * BLCK, BLCK);
    }
    return sum;
}

float vec_dot_q4_1_f32(const void* vx, const float* y, int64_t k) {
    return vec_dot_deq_f32<block_q4_1, 32, dequantize_row_q4_1>(vx, y, k);
}
float vec_dot_q5_0_f32(const void* vx, const float* y, int64_t k) {
    return vec_dot_deq_f32<block_q5_0, 32, dequantize_row_q5_0>(vx, y, k);
}
float vec_dot_q5_1_f32(const void* vx, const float* y, int64_t k) {
    return vec_dot_deq_f32<block_q5_1, 32, dequantize_row_q5_1>(vx, y, k);
}
float vec_dot_q8_1_f32(const void* vx, const float* y, int64_t k) {
    return vec_dot_deq_f32<block_q8_1, 32, dequantize_row_q8_1>(vx, y, k);
}
float vec_dot_q2_K_f32(const void* vx, const float* y, int64_t k) {
    return vec_dot_deq_f32<block_q2_K, 256, dequantize_row_q2_K>(vx, y, k);
}
float vec_dot_q3_K_f32(const void* vx, const float* y, int64_t k) {
    return vec_dot_deq_f32<block_q3_K, 256, dequantize_row_q3_K>(vx, y, k);
}
float vec_dot_q8_K_f32(const void* vx, const float* y, int64_t k) {
    return vec_dot_deq_f32<block_q8_K, 256, dequantize_row_q8_K>(vx, y, k);
}
float vec_dot_iq4_xs_f32(const void* vx, const float* y, int64_t k) {
    return vec_dot_deq_f32<block_iq4_xs, 256, dequantize_row_iq4_xs>(vx, y, k);
}

float vec_dot_quant_f32(GGMLType type, const void* vx, const float* y, int64_t k) {
    switch (type) {
        case GGMLType::F32:      return vec_dot_f32(static_cast<const float*>(vx), y, k);
        case GGMLType::F16:      return vec_dot_f16_f32(vx, y, k);
        case GGMLType::BF16:     return vec_dot_bf16_f32(vx, y, k);
        case GGMLType::FP8_E4M3: return vec_dot_fp8_f32(vx, y, k);
        case GGMLType::Q4_0:     return vec_dot_q4_0_f32(vx, y, k);
        case GGMLType::Q4_1:     return vec_dot_q4_1_f32(vx, y, k);
        case GGMLType::Q5_0:     return vec_dot_q5_0_f32(vx, y, k);
        case GGMLType::Q5_1:     return vec_dot_q5_1_f32(vx, y, k);
        case GGMLType::Q8_0:     return vec_dot_q8_0_f32(vx, y, k);
        case GGMLType::Q8_1:     return vec_dot_q8_1_f32(vx, y, k);
        case GGMLType::IQ4_NL:   return vec_dot_iq4_nl_f32(vx, y, k);
        case GGMLType::Q2_K:     return vec_dot_q2_K_f32(vx, y, k);
        case GGMLType::Q3_K:     return vec_dot_q3_K_f32(vx, y, k);
        case GGMLType::Q4_K:     return vec_dot_q4_K_f32(vx, y, k);
        case GGMLType::Q5_K:     return vec_dot_q5_K_f32(vx, y, k);
        case GGMLType::Q6_K:     return vec_dot_q6_K_f32(vx, y, k);
        case GGMLType::Q8_K:     return vec_dot_q8_K_f32(vx, y, k);
        case GGMLType::IQ2_S:    return vec_dot_iq2_s_f32(vx, y, k);
        case GGMLType::IQ3_S:    return vec_dot_iq3_s_f32(vx, y, k);
        case GGMLType::IQ4_XS:   return vec_dot_iq4_xs_f32(vx, y, k);
        // Still unsupported (documented, warns once via
        // warn_unsupported_quant_once): IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S,
        // IQ1_M, IQ2_M — need llama.cpp IQ grid tables. Also I8/I16/I32/
        // I64/F64 (pre-existing). vec_dot_row_with_q8() falls back here,
        // so these types can no longer return 0 silently from matmul.
        default:
            warn_unsupported_quant_once(type);
            return 0.0f;
    }
}

float vec_dot_q4_0_q8_0(const void* vx, const block_q8_0* y, int64_t k) {
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_q4_0*>(vx);
    float total = 0.0f;

#if defined(__AVX2__)
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m256i offset_8 = _mm256_set1_epi8(8);
    const __m256i ones_16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        const __m128i qbytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[i].qs));
        const __m128i q_lo = _mm_and_si128(qbytes, mask_0f);
        const __m128i q_hi = _mm_and_si128(_mm_srli_epi16(qbytes, 4), mask_0f);
        const __m256i q = _mm256_sub_epi8(_mm256_set_m128i(q_hi, q_lo), offset_8);
        const __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[i].qs));

        const __m256i ax = _mm256_sign_epi8(q, q);
        const __m256i sy = _mm256_sign_epi8(yv, q);
        const __m256i dot16 = _mm256_maddubs_epi16(ax, sy);
        const __m256i dot32 = _mm256_madd_epi16(ones_16, dot16);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(dot32), acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) total += tmp[j];
#else
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        int32_t bsum = 0;
        for (int j = 0; j < 16; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;
            bsum += x0 * y[i].qs[j] + x1 * y[i].qs[j + 16];
        }
        total += d * static_cast<float>(bsum);
    }
#endif
    return total;
}

float vec_dot_q8_0_q8_0(const void* vx, const block_q8_0* y, int64_t k) {
    const int64_t nb = k / 32;
    const auto* x = static_cast<const block_q8_0*>(vx);
    float total = 0.0f;

#if defined(__AVX2__)
    const __m256i ones_16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i].qs));
        const __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[i].qs));

        const __m256i ax = _mm256_sign_epi8(xv, xv);
        const __m256i sy = _mm256_sign_epi8(yv, xv);
        const __m256i dot16 = _mm256_maddubs_epi16(ax, sy);
        const __m256i dot32 = _mm256_madd_epi16(ones_16, dot16);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(dot32), acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) total += tmp[j];
#else
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        int32_t bsum = 0;
        for (int j = 0; j < 32; ++j) {
            bsum += static_cast<int32_t>(x[i].qs[j]) * static_cast<int32_t>(y[i].qs[j]);
        }
        total += d * static_cast<float>(bsum);
    }
#endif
    return total;
}

float vec_dot_q4_K_q8_0(const void* vx, const block_q8_0* y, int64_t k) {
    const int64_t nb = k / 256;
    const auto* x = static_cast<const block_q4_K*>(vx);
    float total = 0.0f;

#if defined(__AVX2__)
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i ones_u8 = _mm256_set1_epi8(1);
    const __m256i ones_16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q = x[i].qs;
        const float d    = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        const block_q8_0* yb = y + i * 8;

        int is = 0;
        uint8_t sc = 0, m = 0;
        for (int j = 0; j < 4; ++j) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = dmin * static_cast<float>(m);
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = dmin * static_cast<float>(m);

            const __m256i qbytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            const __m256i q_lo   = _mm256_and_si256(qbytes, mask_0f);
            const __m256i q_hi   = _mm256_and_si256(_mm256_srli_epi16(qbytes, 4), mask_0f);

            const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb[j * 2 + 0].qs));
            const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb[j * 2 + 1].qs));

            const __m256i dot0 = _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(q_lo, y0));
            const __m256i sum0 = _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(ones_u8, y0));
            const __m256i dot1 = _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(q_hi, y1));
            const __m256i sum1 = _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(ones_u8, y1));

            const float dy0 = fp16_to_fp32(yb[j * 2 + 0].d);
            const float dy1 = fp16_to_fp32(yb[j * 2 + 1].d);

            acc = _mm256_fmadd_ps(_mm256_set1_ps(d1 * dy0), _mm256_cvtepi32_ps(dot0), acc);
            acc = _mm256_fnmadd_ps(_mm256_set1_ps(m1 * dy0), _mm256_cvtepi32_ps(sum0), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d2 * dy1), _mm256_cvtepi32_ps(dot1), acc);
            acc = _mm256_fnmadd_ps(_mm256_set1_ps(m2 * dy1), _mm256_cvtepi32_ps(sum1), acc);

            q += 32;
            is += 2;
        }
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int t = 0; t < 8; ++t) total += tmp[t];
#else
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q = x[i].qs;
        const float d    = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        const block_q8_0* yb = y + i * 8;
        int is = 0;
        uint8_t sc = 0, m = 0;
        for (int j = 0; j < 4; ++j) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = dmin * static_cast<float>(m);
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = dmin * static_cast<float>(m);
            const float dy0 = fp16_to_fp32(yb[j * 2 + 0].d);
            const float dy1 = fp16_to_fp32(yb[j * 2 + 1].d);
            int32_t s0 = 0, sy0 = 0, s1 = 0, sy1 = 0;
            for (int l = 0; l < 32; ++l) {
                s0  += (q[l] & 0x0F) * yb[j * 2 + 0].qs[l];
                sy0 += yb[j * 2 + 0].qs[l];
                s1  += (q[l] >> 4)   * yb[j * 2 + 1].qs[l];
                sy1 += yb[j * 2 + 1].qs[l];
            }
            total += (d1 * s0 - m1 * sy0) * dy0 + (d2 * s1 - m2 * sy1) * dy1;
            q += 32;
            is += 2;
        }
    }
#endif
    return total;
}

float vec_dot_q5_K_q8_0(const void* vx, const block_q8_0* y, int64_t k) {
    const int64_t nb = k / 256;
    const auto* blocks = static_cast<const block_q5_K*>(vx);
    float total = 0.0f;

#if defined(__AVX2__)
    const __m256i ones_u8 = _mm256_set1_epi8(1);
    const __m256i ones_16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    alignas(32) uint8_t q5_0[32];
    alignas(32) uint8_t q5_1[32];

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* ql = blocks[i].qs;
        const uint8_t* qh = blocks[i].qh;
        const float d    = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);
        const block_q8_0* yb = y + i * 8;

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        uint8_t sc = 0, m = 0;
        for (int j = 0; j < 4; ++j) {
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = dmin * static_cast<float>(m);
            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = dmin * static_cast<float>(m);

            for (int l = 0; l < 32; ++l) {
                q5_0[l] = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                q5_1[l] = (ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
            }

            const __m256i qv0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(q5_0));
            const __m256i qv1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(q5_1));
            const __m256i y0  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb[j * 2 + 0].qs));
            const __m256i y1  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb[j * 2 + 1].qs));

            const __m256i dot0 = _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(qv0, y0));
            const __m256i sum0 = _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(ones_u8, y0));
            const __m256i dot1 = _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(qv1, y1));
            const __m256i sum1 = _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(ones_u8, y1));

            const float dy0 = fp16_to_fp32(yb[j * 2 + 0].d);
            const float dy1 = fp16_to_fp32(yb[j * 2 + 1].d);

            acc = _mm256_fmadd_ps(_mm256_set1_ps(d1 * dy0), _mm256_cvtepi32_ps(dot0), acc);
            acc = _mm256_fnmadd_ps(_mm256_set1_ps(m1 * dy0), _mm256_cvtepi32_ps(sum0), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d2 * dy1), _mm256_cvtepi32_ps(dot1), acc);
            acc = _mm256_fnmadd_ps(_mm256_set1_ps(m2 * dy1), _mm256_cvtepi32_ps(sum1), acc);

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int t = 0; t < 8; ++t) total += tmp[t];
#else
    (void)blocks; (void)y; (void)nb;
#endif
    return total;
}

float vec_dot_q6_K_q8_0(const void* vx, const block_q8_0* y, int64_t k) {
    const int64_t nb = k / 256;
    const auto* blocks = static_cast<const block_q6_K*>(vx);
    float sum = 0.0f;

#if defined(__AVX2__)
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i mask_03 = _mm_set1_epi8(0x03);
    const __m128i offset_32 = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();

    auto dot16_q8 = [](__m128i qv, __m128i yv, int8_t sc_val) -> __m128i {
        __m128i ax = _mm_sign_epi8(qv, qv);
        __m128i sy = _mm_sign_epi8(yv, qv);
        __m128i dot16 = _mm_maddubs_epi16(ax, sy);
        return _mm_madd_epi16(dot16, _mm_set1_epi16(static_cast<int16_t>(sc_val)));
    };

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        const uint8_t* ql = blocks[i].ql;
        const uint8_t* qh = blocks[i].qh;
        const int8_t* sc  = blocks[i].scales;
        const block_q8_0* yb = y + i * 8;

        for (int n = 0; n < 256; n += 128) {
            const block_q8_0* yh = yb + (n / 32);
            __m128i s0 = _mm_setzero_si128();
            __m128i s1 = _mm_setzero_si128();
            __m128i s2 = _mm_setzero_si128();
            __m128i s3 = _mm_setzero_si128();

            for (int is = 0; is < 2; ++is) {
                const int l_off = is * 16;
                const __m128i ql0  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql + l_off));
                const __m128i ql1  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql + l_off + 32));
                const __m128i qh_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh + l_off));

                const __m128i q1 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(ql0, mask_0f), _mm_slli_epi16(_mm_and_si128(qh_v, mask_03), 4)),
                    offset_32
                );
                const __m128i q2 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(ql1, mask_0f), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 2), mask_03), 4)),
                    offset_32
                );
                const __m128i q3 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), mask_0f), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 4), mask_03), 4)),
                    offset_32
                );
                const __m128i q4 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), mask_0f), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 6), mask_03), 4)),
                    offset_32
                );

                const __m128i y0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(yh[0].qs + l_off));
                const __m128i y1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(yh[1].qs + l_off));
                const __m128i y2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(yh[2].qs + l_off));
                const __m128i y3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(yh[3].qs + l_off));

                s0 = _mm_add_epi32(s0, dot16_q8(q1, y0, sc[is + 0]));
                s1 = _mm_add_epi32(s1, dot16_q8(q2, y1, sc[is + 2]));
                s2 = _mm_add_epi32(s2, dot16_q8(q3, y2, sc[is + 4]));
                s3 = _mm_add_epi32(s3, dot16_q8(q4, y3, sc[is + 6]));
            }

            const float dy0 = d * fp16_to_fp32(yh[0].d);
            const float dy1 = d * fp16_to_fp32(yh[1].d);
            const float dy2 = d * fp16_to_fp32(yh[2].d);
            const float dy3 = d * fp16_to_fp32(yh[3].d);

            __m256 scale01 = _mm256_set_m128(_mm_set1_ps(dy1), _mm_set1_ps(dy0));
            __m256 scale23 = _mm256_set_m128(_mm_set1_ps(dy3), _mm_set1_ps(dy2));
            acc = _mm256_fmadd_ps(scale01, _mm256_cvtepi32_ps(_mm256_set_m128i(s1, s0)), acc);
            acc = _mm256_fmadd_ps(scale23, _mm256_cvtepi32_ps(_mm256_set_m128i(s3, s2)), acc);

            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) sum += tmp[j];
#else
    (void)vx; (void)y; (void)k;
#endif
    return sum;
}

static inline float vec_dot_row_with_q8(
    GGMLType type,
    const void* row_ptr,
    const float* x,
    const block_q8_0* x_q8,
    int64_t cols
) {
#if defined(__AVX2__)
    if (x_q8) {
        if (type == GGMLType::Q4_0 && (cols % 32 == 0)) {
            return vec_dot_q4_0_q8_0(row_ptr, x_q8, cols);
        }
        if (type == GGMLType::Q8_0 && (cols % 32 == 0)) {
            return vec_dot_q8_0_q8_0(row_ptr, x_q8, cols);
        }
        if (type == GGMLType::Q4_K && (cols % 256 == 0)) {
            return vec_dot_q4_K_q8_0(row_ptr, x_q8, cols);
        }
        if (type == GGMLType::Q5_K && (cols % 256 == 0)) {
            return vec_dot_q5_K_q8_0(row_ptr, x_q8, cols);
        }
        if (type == GGMLType::Q6_K && (cols % 256 == 0)) {
            return vec_dot_q6_K_q8_0(row_ptr, x_q8, cols);
        }
    }
#else
    (void)x_q8;
#endif
    return vec_dot_quant_f32(type, row_ptr, x, cols);
}

static inline bool supports_q8_activation(GGMLType type, int64_t cols) {
    switch (type) {
        case GGMLType::Q4_0:
        case GGMLType::Q8_0:
            return (cols % 32) == 0;
        case GGMLType::Q4_K:
        case GGMLType::Q5_K:
        case GGMLType::Q6_K:
            return (cols % 256) == 0;
        default:
            return false;
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

    std::vector<block_q8_0> x_q8;
    const block_q8_0* x_q8_ptr = nullptr;
#if defined(__AVX2__)
    if (rows >= 8 && supports_q8_activation(type, cols)) {
        x_q8.resize(static_cast<size_t>(cols / 32));
        quantize_row_q8_0(x, x_q8.data(), cols);
        x_q8_ptr = x_q8.data();
    }
#endif

#if defined(_OPENMP)
    #pragma omp parallel for if(rows >= 64) schedule(static)
#endif
    for (int64_t r = 0; r < rows; ++r) {
        out[r] = vec_dot_row_with_q8(type, base + r * row_bytes, x, x_q8_ptr, cols);
    }
}

void matmul_qkv_quant_f32(
    GGMLType wq_type, const void* Wq, int64_t q_rows, float* q_out,
    GGMLType wk_type, const void* Wk, int64_t k_rows, float* k_out,
    GGMLType wv_type, const void* Wv, int64_t v_rows, float* v_out,
    const float* x, int64_t cols
) {
    const size_t q_row_bytes = Wq ? tensor_byte_size(wq_type, static_cast<uint64_t>(cols)) : 0;
    const size_t k_row_bytes = Wk ? tensor_byte_size(wk_type, static_cast<uint64_t>(cols)) : 0;
    const size_t v_row_bytes = Wv ? tensor_byte_size(wv_type, static_cast<uint64_t>(cols)) : 0;

    const auto* q_base = static_cast<const uint8_t*>(Wq);
    const auto* k_base = static_cast<const uint8_t*>(Wk);
    const auto* v_base = static_cast<const uint8_t*>(Wv);

    const int64_t total_rows = (Wq ? q_rows : 0) + (Wk ? k_rows : 0) + (Wv ? v_rows : 0);
    if (total_rows <= 0) return;

    std::vector<block_q8_0> x_q8;
    const block_q8_0* x_q8_ptr = nullptr;
#if defined(__AVX2__)
    if (total_rows >= 8 && (cols % 32 == 0)) {
        x_q8.resize(static_cast<size_t>(cols / 32));
        quantize_row_q8_0(x, x_q8.data(), cols);
        x_q8_ptr = x_q8.data();
    }
#endif

    const int64_t q_end = Wq ? q_rows : 0;
    const int64_t k_end = q_end + (Wk ? k_rows : 0);

#if defined(_OPENMP)
    #pragma omp parallel for if(total_rows >= 64) schedule(static)
#endif
    for (int64_t idx = 0; idx < total_rows; ++idx) {
        if (idx < q_end) {
            q_out[idx] = vec_dot_row_with_q8(wq_type, q_base + idx * q_row_bytes, x, x_q8_ptr, cols);
        } else if (idx < k_end) {
            int64_t kr = idx - q_end;
            k_out[kr] = vec_dot_row_with_q8(wk_type, k_base + kr * k_row_bytes, x, x_q8_ptr, cols);
        } else {
            int64_t vr = idx - k_end;
            v_out[vr] = vec_dot_row_with_q8(wv_type, v_base + vr * v_row_bytes, x, x_q8_ptr, cols);
        }
    }
}

void matmul_swiglu_quant_f32(
    GGMLType gate_type, const void* W_gate,
    GGMLType up_type,   const void* W_up,
    const float* x,
    float* swiglu_out,
    int64_t exp_dim,
    int64_t cols,
    bool use_gelu
) {
    const size_t gate_row_bytes = tensor_byte_size(gate_type, static_cast<uint64_t>(cols));
    const size_t up_row_bytes   = tensor_byte_size(up_type,   static_cast<uint64_t>(cols));
    const auto* gate_base = static_cast<const uint8_t*>(W_gate);
    const auto* up_base   = static_cast<const uint8_t*>(W_up);

    std::vector<block_q8_0> x_q8;
    const block_q8_0* x_q8_ptr = nullptr;
#if defined(__AVX2__)
    if (exp_dim >= 8 && (cols % 32 == 0)) {
        x_q8.resize(static_cast<size_t>(cols / 32));
        quantize_row_q8_0(x, x_q8.data(), cols);
        x_q8_ptr = x_q8.data();
    }
#endif

#if defined(_OPENMP)
    #pragma omp parallel for if(exp_dim >= 64) schedule(static)
#endif
    for (int64_t r = 0; r < exp_dim; ++r) {
        float g = vec_dot_row_with_q8(gate_type, gate_base + r * gate_row_bytes, x, x_q8_ptr, cols);
        float u = vec_dot_row_with_q8(up_type,   up_base   + r * up_row_bytes,   x, x_q8_ptr, cols);
        if (use_gelu) {
            constexpr float kSqrt2OverPi = 0.7978845608028654f;
            float inner = kSqrt2OverPi * (g + 0.044715f * g * g * g);
            float gelu_g = 0.5f * g * (1.0f + std::tanh(inner));
            swiglu_out[r] = gelu_g * u;
        } else {
            swiglu_out[r] = (g / (1.0f + std::exp(-g))) * u;
        }
    }
}

} // namespace moe

