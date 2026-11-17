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

// Q4_1: 32 weights per block (20 bytes) — ggml block_q4_1
struct block_q4_1 {
    uint16_t d;       // FP16 delta scale
    uint16_t m;       // FP16 min offset
    uint8_t  qs[16];  // 32 x 4-bit unsigned values (offset by m, not -8)
};

// Q5_0: 32 weights per block (22 bytes) — ggml block_q5_0
struct block_q5_0 {
    uint16_t d;       // FP16 delta scale
    uint8_t  qh[4];   // high 1-bit of each 5-bit value (bit j -> element j,
                      // bit j+16 -> element j+16), as a 32-bit little-endian
                      // word (ggml memcpy's it into a uint32_t)
    uint8_t  qs[16];  // low 4-bits (32 nibbles, offset by -16)
};

// Q5_1: 32 weights per block (24 bytes) — ggml block_q5_1
struct block_q5_1 {
    uint16_t d;       // FP16 delta scale
    uint16_t m;       // FP16 min offset
    uint8_t  qh[4];   // high 1-bit of each 5-bit value (as in Q5_0)
    uint8_t  qs[16];  // low 4-bits
};

// Q8_1: 32 weights per block (36 bytes) — ggml block_q8_1.
// d/s are FP16. s is the block sum (sum(qs) * d), used by ggml's
// dot-product kernels; it is NOT added to each dequantized value.
// Current llama.cpp/ggml ships no dequantize_row_q8_1 at all (it is
// commented out in ggml-quants.h) — Q8_1 is an activation format
// upstream. The per-element value is qs * d, which is what
// quantize_row_q8_1_ref in ggml-quants.c writes, so that is what we
// dequantize to.
struct block_q8_1 {
    uint16_t d;       // FP16 delta scale
    uint16_t s;       // FP16 block sum (auxiliary, see above)
    int8_t   qs[32];  // 32 x 8-bit signed ints
};

// Q2_K: 256 weights per super-block (84 bytes) — ggml block_q2_K
struct block_q2_K {
    uint16_t d;           // super-block scale (FP16)
    uint16_t dmin;        // super-block min scale (FP16)
    uint8_t  scales[16];  // 16 x (4-bit scale | 4-bit min << 4)
    uint8_t  qs[64];      // 256 x 2-bit values, strided (see dequantize)
};

// Q3_K: 256 weights per super-block (110 bytes) — ggml block_q3_K
struct block_q3_K {
    uint16_t d;           // super-block scale (FP16)
    uint8_t  qs[64];      // 256 x 2-bit low parts, strided
    uint8_t  hmask[32];   // high-bit mask: bit (elem/32) of byte (elem%32)
    uint8_t  scales[12];  // 16 x 6-bit scales, packed (see dequantize)
};

// Q8_K: 256 weights per super-block (292 bytes) — ggml block_q8_K.
// NOTE: d is FP32 here, unlike every other K-quant (see ggml-common.h:
// sizeof(block_q8_K) == sizeof(float) + 256 + 16*sizeof(int16_t)) and
// ggml-quants.c quantize_row_q8_K_ref (y[i].d = 1/iscale, no FP16
// conversion) / dequantize_row_q8_K (x[i].d * qs, d used as float).
struct block_q8_K {
    float    d;           // FP32 super-block scale
    int8_t   qs[256];     // 256 x 8-bit signed ints
    int16_t  bsums[16];   // sub-block sums (unused by dequantization)
};

// IQ4_XS: 256 weights per super-block (136 bytes) — ggml block_iq4_xs.
// Dequantization needs only kvalues_iq4nl (already in this repo), so
// unlike the other IQ types it is exactly portable without ggml's IQ
// grid tables.
struct block_iq4_xs {
    uint16_t d;           // FP16 super-block scale
    uint16_t scales_h;    // high 2 bits of each 6-bit sub-block scale
    uint8_t  scales_l[4]; // low 4 bits of each 6-bit sub-block scale
    uint8_t  qs[128];     // 256 x 4-bit indices into kvalues_iq4nl
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
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 must be 20 bytes");
static_assert(sizeof(block_q5_0) == 22, "block_q5_0 must be 22 bytes");
static_assert(sizeof(block_q5_1) == 24, "block_q5_1 must be 24 bytes");
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");
static_assert(sizeof(block_q2_K) == 84, "block_q2_K must be 84 bytes");
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");
static_assert(sizeof(block_q8_K) == 292, "block_q8_K must be 292 bytes");
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs must be 136 bytes");

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
// New quantizers (layouts mirror llama.cpp ggml-quants.c *_ref versions;
// Q2_K/Q3_K use a simplified per-group scale selection instead of ggml's
// make_qkx*_quants optimiser — see src/quant.cpp)
void quantize_row_q4_1(const float* x, void* vy, int64_t k);
void quantize_row_q5_0(const float* x, void* vy, int64_t k);
void quantize_row_q5_1(const float* x, void* vy, int64_t k);
void quantize_row_q8_1(const float* x, void* vy, int64_t k);
void quantize_row_q2_K(const float* x, void* vy, int64_t k);
void quantize_row_q3_K(const float* x, void* vy, int64_t k);
void quantize_row_q8_K(const float* x, void* vy, int64_t k);
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
// New dequantizers (scalar, exact ggml layouts — see src/quant.cpp)
void dequantize_row_q4_1(const void* vx, float* y, int64_t k);
void dequantize_row_q5_0(const void* vx, float* y, int64_t k);
void dequantize_row_q5_1(const void* vx, float* y, int64_t k);
void dequantize_row_q8_1(const void* vx, float* y, int64_t k);
void dequantize_row_q2_K(const void* vx, float* y, int64_t k);
void dequantize_row_q3_K(const void* vx, float* y, int64_t k);
void dequantize_row_q8_K(const void* vx, float* y, int64_t k);
void dequantize_row_iq4_xs(const void* vx, float* y, int64_t k);
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
float vec_dot_q4_1_f32(const void* vx, const float* y, int64_t k);
float vec_dot_q5_0_f32(const void* vx, const float* y, int64_t k);
float vec_dot_q5_1_f32(const void* vx, const float* y, int64_t k);
float vec_dot_q8_1_f32(const void* vx, const float* y, int64_t k);
float vec_dot_q2_K_f32(const void* vx, const float* y, int64_t k);
float vec_dot_q3_K_f32(const void* vx, const float* y, int64_t k);
float vec_dot_q8_K_f32(const void* vx, const float* y, int64_t k);
float vec_dot_iq4_xs_f32(const void* vx, const float* y, int64_t k);
float vec_dot_f16_f32(const void* vx, const float* y, int64_t k);
float vec_dot_bf16_f32(const void* vx, const float* y, int64_t k);
float vec_dot_fp8_f32(const void* vx, const float* y, int64_t k);

float vec_dot_quant_f32(GGMLType type, const void* vx, const float* y, int64_t k);

// Integer SIMD Dot Products with pre-quantized Q8_0 activations: dot(weight_row[k], x_q8[k])
float vec_dot_q4_0_q8_0(const void* vx, const block_q8_0* y, int64_t k);
float vec_dot_q8_0_q8_0(const void* vx, const block_q8_0* y, int64_t k);
float vec_dot_q4_K_q8_0(const void* vx, const block_q8_0* y, int64_t k);
float vec_dot_q5_K_q8_0(const void* vx, const block_q8_0* y, int64_t k);
float vec_dot_q6_K_q8_0(const void* vx, const block_q8_0* y, int64_t k);

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

// Fused QKV Projection: computes q_out, k_out, v_out in a single OpenMP region with shared Q8_0 activation
void matmul_qkv_quant_f32(
    GGMLType wq_type, const void* Wq, int64_t q_rows, float* q_out,
    GGMLType wk_type, const void* Wk, int64_t k_rows, float* k_out,
    GGMLType wv_type, const void* Wv, int64_t v_rows, float* v_out,
    const float* x, int64_t cols
);

// Fused SwiGLU / GeGLU Gate+Up Projection: out[r] = Act((W_gate * x)[r]) * (W_up * x)[r]
void matmul_swiglu_quant_f32(
    GGMLType gate_type, const void* W_gate,
    GGMLType up_type,   const void* W_up,
    const float* x,
    float* swiglu_out,
    int64_t exp_dim,
    int64_t cols,
    bool use_gelu = false
);

} // namespace moe

