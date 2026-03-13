#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>

#define QK_K 256

// TQ1_0: base-3 ternary packing, 256 weights per block
// qs packs 240 values (5 per byte in base-3), qh packs remaining 16 (4 per byte)
// Total: 48 + 4 + 2 = 54 bytes per 256-element block (1.6875 bpw)
typedef struct {
    uint8_t  qs[(QK_K - 4 * QK_K / 64) / 5];  // 48 bytes: (256-16)/5
    uint8_t  qh[QK_K / 64];                    // 4 bytes
    uint16_t d;                                  // FP16 scale
} BlockTQ1;

// TQ2_0: 2-bit ternary packing, 256 weights per block
// 64 bytes qs (4 weights per byte), 2 bytes scale per block
typedef struct {
    uint8_t  qs[QK_K / 4];  // 64 bytes
    uint16_t d;              // FP16 scale
} BlockTQ2;

// I2_S: Microsoft BitNet 2-bit ternary, no per-block scale
// Interleaved byte layout: each byte packs 4 values from 4 sub-rows of 32
// Single per-tensor scale stored at offset nelements/4 in the data
// Encoding: 0=-1, 1=0, 2=+1

// Ternary weight tensor descriptor (zero-copy into GGUF buffer)
typedef struct {
    const void *data;   // packed weight data
    int type;           // GGUF_TENSOR_TQ1_0, TQ2_0, or I2_S
    int rows, cols;
    float scale;        // per-tensor scale (from .scale tensor or embedded in data)
} QWeight;

float    fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
void     dequant_tq1_block(const BlockTQ1 *block, float *out);
void     dequant_tq2_block(const BlockTQ2 *block, float *out);
void     dequant_i2s_row(const uint8_t *data, float *out, int n, float scale);
void     ternary_matvec(float *out, const QWeight *W, const float *x);

// Batch matvec: run multiple independent matvecs with a single OMP fork/join
typedef struct {
    float *out;
    const QWeight *W;
} MatvecTask;

void ternary_matvec_batch(const MatvecTask *tasks, int n_tasks,
                           const float *x, int8_t *x_q_buf);

// Inner batch matvec: must be called from within #pragma omp parallel.
// x_scale_out must point to a shared float (written by one thread, read by all).
// Includes an omp barrier at the end to ensure outputs are complete.
void ternary_matvec_batch_inner(const MatvecTask *tasks, int n_tasks,
                                 const float *x, int8_t *x_q_buf,
                                 float *x_scale_out);

// Low-level SDOT primitives for building merged parallel regions.
// quantize_x_to_i8: quantize float vector to int8, returns scale = amax/127.
// i2s_matvec_sdot: I2_S matvec using SDOT, must be inside #pragma omp parallel.
//                  Uses #pragma omp for nowait internally.
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
float quantize_x_to_i8(const float *x, int8_t *x_q, int n);
void  i2s_matvec_sdot(float *out, const QWeight *W,
                       const int8_t *x_q, float x_scale);
#endif

#endif // QUANT_H
