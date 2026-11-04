#pragma once

#include "common.h"

namespace moe {

#pragma pack(push, 1)

// Q4_0: 32 weights per block (18 bytes)
struct block_q4_0 {
    uint16_t d;       // FP16 delta scale
    uint8_t  qs[16];  // 32 x 4-bit unsigned nibbles (offset by -8)
};

// Q8_0: 32 weights per block (34 bytes)
struct block_q8_0 {
    uint16_t d;       // FP16 delta scale
    int8_t   qs[32];  // 32 x 8-bit signed ints
};

// IQ4_NL: 32 weights per block (18 bytes) with non-linear kvalues_iq4nl LUT
struct block_iq4_nl {
    uint16_t d;       // FP16 scale
    uint8_t  qs[16];  // 32 x 4-bit indices into kvalues_iq4nl[16]
};

// Q4_K: 256 weights per super-block (144 bytes)
struct block_q4_K {
    uint16_t d;           // super-block scale for quantized scales (FP16)
    uint16_t dmin;        // super-block scale for quantized mins (FP16)
    uint8_t  scales[12];  // 8 sub-block scales & mins packed into 6-bit ints
    uint8_t  qs[128];     // 256 x 4-bit weights
};

// Q5_K: 256 weights per super-block (176 bytes)
struct block_q5_K {
    uint16_t d;           // super-block scale (FP16)
    uint16_t dmin;        // super-block min scale (FP16)
    uint8_t  scales[12];  // packed 6-bit scales and mins
    uint8_t  qh[32];      // high 1-bit of 5-bit values (256 bits)
    uint8_t  qs[128];     // low 4-bits (256 nibbles)
};

// Q6_K: 256 weights per super-block (210 bytes)
struct block_q6_K {
    uint8_t ql[128];      // low 4-bits
    uint8_t qh[64];       // high 2-bits
    int8_t  scales[16];   // 16 sub-block scales
    uint16_t d;           // super-block scale (FP16)
};

// IQ2_S: 256 weights per super-block (82 bytes)
struct block_iq2_s {
    uint16_t d;           // super-block scale (FP16)
    uint8_t  qs[64];      // 256 x 2-bit weights
    uint8_t  qh[8];       // high bits / grids
    uint8_t  scales[8];   // sub-block scales
};

// IQ3_S: 256 weights per super-block (110 bytes)
struct block_iq3_s {
    uint16_t d;           // super-block scale (FP16)
    uint8_t  qs[96];      // 256 x 3-bit weights
    uint8_t  qh[8];       // grid indices
    uint8_t  signs[4];    // sign bits
};

#pragma pack(pop)

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");
static_assert(sizeof(block_iq4_nl) == 18, "block_iq4_nl must be 18 bytes");
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");
static_assert(sizeof(block_iq2_s) == 82, "block_iq2_s must be 82 bytes");
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s must be 110 bytes");

// Non-linear 4-bit lookup table (standard llama.cpp / ggml IQ4_NL table)
extern const int8_t kvalues_iq4nl[16];

// Scalar conversions
float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
float bf16_to_fp32(uint16_t b);
uint16_t fp32_to_bf16(float f);
float fp8_e4m3_to_fp32(uint8_t v);
uint8_t fp32_to_fp8_e4m3(float f);

// Quantization functions (FP32 -> Quantized blocks)
void quantize_row_q4_0(const float* x, void* vy, int64_t k);
void quantize_row_q8_0(const float* x, void* vy, int64_t k);
void quantize_row_iq4_nl(const float* x, void* vy, int64_t k);
void quantize_row_q4_K(const float* x, void* vy, int64_t k);
void quantize_row_q5_K(const float* x, void* vy, int64_t k);
void quantize_row_q6_K(const float* x, void* vy, int64_t k);
void quantize_row_f16(const float* x, void* vy, int64_t k);
void quantize_row_bf16(const float* x, void* vy, int64_t k);
void quantize_row_fp8_e4m3(const float* x, void* vy, int64_t k);

// Dequantization functions (Quantized blocks -> FP32)
void dequantize_row_q4_0(const void* vx, float* y, int64_t k);
void dequantize_row_q8_0(const void* vx, float* y, int64_t k);
void dequantize_row_iq4_nl(const void* vx, float* y, int64_t k);
void dequantize_row_q4_K(const void* vx, float* y, int64_t k);
void dequantize_row_q5_K(const void* vx, float* y, int64_t k);
void dequantize_row_q6_K(const void* vx, float* y, int64_t k);
void dequantize_row_iq2_s(const void* vx, float* y, int64_t k);
void dequantize_row_iq3_s(const void* vx, float* y, int64_t k);
void dequantize_row_f16(const void* vx, float* y, int64_t k);
void dequantize_row_bf16(const void* vx, float* y, int64_t k);
void dequantize_row_fp8_e4m3(const void* vx, float* y, int64_t k);
void dequantize_row(GGMLType type, const void* vx, float* y, int64_t k);

// Dequantize-on-the-fly SIMD Dot Products: dot(weight_row[k], activation_f32[k])
float vec_dot_f32(const float* a, const float* b, int64_t k);
float vec_dot_q4_0_f32(const void* vx, const float* y, int64_t k);
float vec_dot_q8_0_f32(const void* vx, const float* y, int64_t k);
float vec_dot_iq4_nl_f32(const void* vx, const float* y, int64_t k);
float vec_dot_q4_K_f32(const void* vx, const float* y, int64_t k);
float vec_dot_q5_K_f32(const void* vx, const float* y, int64_t k);
float vec_dot_q6_K_f32(const void* vx, const float* y, int64_t k);
float vec_dot_iq2_s_f32(const void* vx, const float* y, int64_t k);
float vec_dot_iq3_s_f32(const void* vx, const float* y, int64_t k);
float vec_dot_f16_f32(const void* vx, const float* y, int64_t k);
float vec_dot_bf16_f32(const void* vx, const float* y, int64_t k);
float vec_dot_fp8_f32(const void* vx, const float* y, int64_t k);

float vec_dot_quant_f32(GGMLType type, const void* vx, const float* y, int64_t k);

// Matrix-Vector Multiplication: out[rows] = W[rows, cols] * x[cols]
// Supports any GGMLType without full-matrix dequantization allocation
void matmul_quant_f32(
    GGMLType type,
    const void* W,
    const float* x,
    float* out,
    int64_t rows,
    int64_t cols
);

} // namespace moe

