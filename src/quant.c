#include "quant.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifdef __ARM_NEON
#include <arm_neon.h>

// Widen 16 int8 ternary values (-1/0/+1) to float, FMA with 16 floats from x,
// accumulate into 4 float32x4 accumulators.
static inline void neon_acc_i8x16_f32(int8x16_t t, const float *x,
    float32x4_t *a0, float32x4_t *a1, float32x4_t *a2, float32x4_t *a3) {
    int16x8_t lo16 = vmovl_s8(vget_low_s8(t));
    int16x8_t hi16 = vmovl_s8(vget_high_s8(t));
    *a0 = vmlaq_f32(*a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))),  vld1q_f32(x + 0));
    *a1 = vmlaq_f32(*a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vld1q_f32(x + 4));
    *a2 = vmlaq_f32(*a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))),  vld1q_f32(x + 8));
    *a3 = vmlaq_f32(*a3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vld1q_f32(x + 12));
}

// Reduce 4 float32x4 accumulators to a single scalar sum (ARMv7-compatible).
static inline float neon_reduce4(float32x4_t a, float32x4_t b,
                                  float32x4_t c, float32x4_t d) {
    float32x4_t s = vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d));
    float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}
#endif // __ARM_NEON

// --- FP16 <-> FP32 conversion ---

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // ±0
        } else {
            // #37: Subnormal: normalize by shifting mantissa left until hidden bit appears.
            // At most 10 shifts (10 mantissa bits), so exp goes from 1 down to at most -9.
            // Result: exp+112 ranges from 103 to 113, always valid for float32 exponent.
            exp = 1;
            while (!(mant & 0x0400)) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            f = sign | ((uint32_t)(exp + 112) << 23) | ((uint32_t)mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | ((uint32_t)mant << 13);  // Inf/NaN
    } else {
        f = sign | ((uint32_t)(exp + 112) << 23) | ((uint32_t)mant << 13);
    }

    float result;
    memcpy(&result, &f, 4);
    return result;
}

uint16_t fp32_to_fp16(float val) {
    uint32_t f;
    memcpy(&f, &val, 4);

    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = ((f >> 23) & 0xFF) - 127;
    uint32_t mant = f & 0x007FFFFF;

    if (exp > 15) {
        return (uint16_t)(sign | 0x7C00);  // Inf
    } else if (exp < -14) {
        return (uint16_t)sign;  // Zero (flush subnormals)
    } else {
        return (uint16_t)(sign | ((uint32_t)(exp + 15) << 10) | (mant >> 13));
    }
}

// --- TQ2_0 dequantization ---
// 2-bit packing: 4 values per byte, map {0,1,2} -> {-1,0,+1}

void dequant_tq2_block(const BlockTQ2 *block, float *out) {
    float d = fp16_to_fp32(block->d);
    int idx = 0;

    // Two groups of 32 bytes
    for (int j = 0; j < 64; j += 32) {
        for (int l = 0; l < 4; l++) {
            for (int m = 0; m < 32; m++) {
                int8_t q = (block->qs[j + m] >> (l * 2)) & 3;
                out[idx++] = (float)(q - 1) * d;
            }
        }
    }
}

// --- TQ1_0 dequantization ---
// Base-3 packing: 5 values per byte in qs (240 values), 4 values per byte in qh (16 values)

void dequant_tq1_block(const BlockTQ1 *block, float *out) {
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    float d = fp16_to_fp32(block->d);
    int idx = 0;

    // Process qs: 48 bytes, in two chunks (32 + 16)
    // First chunk: bytes 0..31, 5 trits each → 160 values
    for (int n = 0; n < 5; n++) {
        for (int m = 0; m < 32; m++) {
            uint8_t q = block->qs[m] * pow3[n];  // uint8 overflow is intentional
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    // Second chunk: bytes 32..47, 5 trits each → 80 values
    for (int n = 0; n < 5; n++) {
        for (int m = 0; m < 16; m++) {
            uint8_t q = block->qs[32 + m] * pow3[n];
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    // Process qh: 4 bytes, 4 trits each → 16 values
    for (int n = 0; n < 4; n++) {
        for (int m = 0; m < 4; m++) {
            uint8_t q = block->qh[m] * pow3[n];
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    // #35: Assert we produced exactly QK_K values (160 + 80 + 16 = 256)
    assert(idx == QK_K);
}

// --- I2_S dequantization (Microsoft BitNet format) ---
// 2-bit ternary, interleaved byte layout, single per-tensor scale
// Each byte: bits 7-6=subrow0, 5-4=subrow1, 3-2=subrow2, 1-0=subrow3
// Processes 128 elements (4 × 32) per 32-byte chunk

// #36: I2_S uses an interleaved byte layout where each byte contains 2-bit values
// from 4 sub-rows of 32 elements. This means each 128-element chunk always uses
// exactly 32 bytes. Model dimensions are always multiples of 128 in practice.
void dequant_i2s_row(const uint8_t *data, float *out, int n, float scale) {
    static const float map2bit[4] = { -1.0f, 0.0f, +1.0f, 0.0f };
    int done = 0;

    while (done < n) {
        int blk_e = (n - done >= 128) ? 128 : (n - done);
        int cols0 = blk_e >= 32  ? 32 : blk_e;
        int cols1 = blk_e >= 64  ? 32 : (blk_e > 32  ? blk_e - 32  : 0);
        int cols2 = blk_e >= 96  ? 32 : (blk_e > 64  ? blk_e - 64  : 0);
        int cols3 = blk_e >= 128 ? 32 : (blk_e > 96  ? blk_e - 96  : 0);

        for (int gp = 0; gp < 32; gp++) {
            uint8_t b = data[gp];
            uint8_t c0 = (b >> 6) & 0x3;
            uint8_t c1 = (b >> 4) & 0x3;
            uint8_t c2 = (b >> 2) & 0x3;
            uint8_t c3 = (b >> 0) & 0x3;

            if (gp < cols0) out[done + 0*32 + gp] = scale * map2bit[c0];
            if (gp < cols1) out[done + 1*32 + gp] = scale * map2bit[c1];
            if (gp < cols2) out[done + 2*32 + gp] = scale * map2bit[c2];
            if (gp < cols3) out[done + 3*32 + gp] = scale * map2bit[c3];
        }

        data += 32;
        done += blk_e;
    }
}

// --- SDOT int8 accumulation for I2_S (Apple Silicon / ARMv8.2+ dotprod) ---

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

// Quantize float vector x[n] to int8, returning scale = amax/127.
// x_q[n] = round(x[i] / scale), clamped to [-127, 127].
static float quantize_x_to_i8(const float *x, int8_t *x_q, int n) {
    // Find absolute max via NEON
    float32x4_t vmax = vdupq_n_f32(0);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i)));
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i + 4)));
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i + 8)));
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i + 12)));
    }
    for (; i + 3 < n; i += 4)
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i)));
    float amax = vmaxvq_f32(vmax);
    for (; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }

    if (amax == 0.0f) {
        memset(x_q, 0, n);
        return 0.0f;
    }

    float scale = amax / 127.0f;
    float inv_scale = 127.0f / amax;
    float32x4_t vinv = vdupq_n_f32(inv_scale);

    i = 0;
    for (; i + 15 < n; i += 16) {
        // Convert 16 floats to 16 int8s
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i),      vinv));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4),  vinv));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 8),  vinv));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 12), vinv));
        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int16x4_t s2 = vqmovn_s32(i2);
        int16x4_t s3 = vqmovn_s32(i3);
        int8x8_t  b0 = vqmovn_s16(vcombine_s16(s0, s1));
        int8x8_t  b1 = vqmovn_s16(vcombine_s16(s2, s3));
        vst1_s8(x_q + i,     b0);
        vst1_s8(x_q + i + 8, b1);
    }
    for (; i < n; i++) {
        int v = (int)roundf(x[i] * inv_scale);
        x_q[i] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
    }
    return scale;
}

// SDOT I2_S matvec inner loop — called from within an OMP parallel region.
// Accumulates int32, converts at the end. Uses omp for nowait.
static void i2s_matvec_sdot(float *out, const QWeight *W,
                             const int8_t *x_q, float x_scale) {
    int row_bytes = W->cols / 4;
    const uint8_t *base = (const uint8_t *)W->data;
    float combined_scale = W->scale * x_scale;
    int cols = W->cols;

    #ifdef _OPENMP
    #pragma omp for nowait
    #endif
    for (int row = 0; row < W->rows; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;

        int32x4_t iaccA = vdupq_n_s32(0), iaccB = vdupq_n_s32(0);
        int32x4_t iaccC = vdupq_n_s32(0), iaccD = vdupq_n_s32(0);
        const uint8x16_t neon_zero = vdupq_n_u8(0);
        const uint8x16_t neon_two  = vdupq_n_u8(2);

        while (done < cols) {
            __builtin_prefetch(rd + 128, 0, 0);
            for (int h = 0; h < 2; h++) {
                uint8x16_t raw = vld1q_u8(rd + h * 16);
                const int8_t *xp = x_q + done + h * 16;

                uint8x16_t v0 = vshrq_n_u8(raw, 6);
                uint8x16_t v1 = vandq_u8(vshrq_n_u8(raw, 4), vdupq_n_u8(3));
                uint8x16_t v2 = vandq_u8(vshrq_n_u8(raw, 2), vdupq_n_u8(3));
                uint8x16_t v3 = vandq_u8(raw, vdupq_n_u8(3));

                // Decode ternary: 0→-1, 1→0, 2→+1
                int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(v0, neon_zero)),
                                         vreinterpretq_s8_u8(vceqq_u8(v0, neon_two)));
                int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(v1, neon_zero)),
                                         vreinterpretq_s8_u8(vceqq_u8(v1, neon_two)));
                int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(v2, neon_zero)),
                                         vreinterpretq_s8_u8(vceqq_u8(v2, neon_two)));
                int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(v3, neon_zero)),
                                         vreinterpretq_s8_u8(vceqq_u8(v3, neon_two)));

                // SDOT: 16 int8×int8 multiply-adds per instruction
                int8x16_t xv0 = vld1q_s8(xp + 0*32);
                int8x16_t xv1 = vld1q_s8(xp + 1*32);
                int8x16_t xv2 = vld1q_s8(xp + 2*32);
                int8x16_t xv3 = vld1q_s8(xp + 3*32);

                iaccA = vdotq_s32(iaccA, t0, xv0);
                iaccB = vdotq_s32(iaccB, t1, xv1);
                iaccC = vdotq_s32(iaccC, t2, xv2);
                iaccD = vdotq_s32(iaccD, t3, xv3);
            }
            rd += 32;
            done += 128;
        }

        // Reduce int32 accumulators to scalar
        int32x4_t sum4 = vaddq_s32(vaddq_s32(iaccA, iaccB), vaddq_s32(iaccC, iaccD));
        int32_t total = vaddvq_s32(sum4);
        out[row] = (float)total * combined_scale;
    }
}

#endif // __ARM_NEON && __ARM_FEATURE_DOTPROD

// --- Ternary matrix-vector multiply ---
// out[rows] = W[rows × cols] @ x[cols]

void ternary_matvec(float *out, const QWeight *W, const float *x) {

    if (W->type == 36) {  // I2_S
        int cols = W->cols;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        // SDOT path: quantize x once, then use integer dot products
        int8_t *x_q = (int8_t *)malloc(cols * sizeof(int8_t));
        float x_scale = quantize_x_to_i8(x, x_q, cols);

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            i2s_matvec_sdot(out, W, x_q, x_scale);
        }
        free(x_q);
#elif defined(__ARM_NEON)
        // Float NEON path (no dotprod)
        {
        int row_bytes = cols / 4;
        const uint8_t *base = (const uint8_t *)W->data;
        float scale = W->scale;
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int row = 0; row < W->rows; row++) {
            const uint8_t *rd = base + (size_t)row * row_bytes;
            int done = 0;
            float32x4_t accA0 = vdupq_n_f32(0), accA1 = vdupq_n_f32(0);
            float32x4_t accA2 = vdupq_n_f32(0), accA3 = vdupq_n_f32(0);
            float32x4_t accB0 = vdupq_n_f32(0), accB1 = vdupq_n_f32(0);
            float32x4_t accB2 = vdupq_n_f32(0), accB3 = vdupq_n_f32(0);
            const uint8x16_t neon_zero = vdupq_n_u8(0);
            const uint8x16_t neon_two  = vdupq_n_u8(2);
            while (done < cols) {
                __builtin_prefetch(rd + 128, 0, 0);
                __builtin_prefetch(rd + 192, 0, 0);
                for (int h = 0; h < 2; h++) {
                    uint8x16_t raw = vld1q_u8(rd + h * 16);
                    const float *xp = x + done + h * 16;
                    uint8x16_t v0 = vshrq_n_u8(raw, 6);
                    uint8x16_t v1 = vandq_u8(vshrq_n_u8(raw, 4), vdupq_n_u8(3));
                    uint8x16_t v2 = vandq_u8(vshrq_n_u8(raw, 2), vdupq_n_u8(3));
                    uint8x16_t v3 = vandq_u8(raw, vdupq_n_u8(3));
                    int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(v0, neon_zero)),
                                             vreinterpretq_s8_u8(vceqq_u8(v0, neon_two)));
                    int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(v1, neon_zero)),
                                             vreinterpretq_s8_u8(vceqq_u8(v1, neon_two)));
                    int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(v2, neon_zero)),
                                             vreinterpretq_s8_u8(vceqq_u8(v2, neon_two)));
                    int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(v3, neon_zero)),
                                             vreinterpretq_s8_u8(vceqq_u8(v3, neon_two)));
                    neon_acc_i8x16_f32(t0, xp + 0*32, &accA0, &accA1, &accA2, &accA3);
                    neon_acc_i8x16_f32(t1, xp + 1*32, &accB0, &accB1, &accB2, &accB3);
                    neon_acc_i8x16_f32(t2, xp + 2*32, &accA0, &accA1, &accA2, &accA3);
                    neon_acc_i8x16_f32(t3, xp + 3*32, &accB0, &accB1, &accB2, &accB3);
                }
                rd += 32;
                done += 128;
            }
            out[row] = (neon_reduce4(accA0, accA1, accA2, accA3) +
                        neon_reduce4(accB0, accB1, accB2, accB3)) * scale;
        }
        }
#else
        // Scalar fallback
        {
        int row_bytes = cols / 4;
        const uint8_t *base = (const uint8_t *)W->data;
        float scale = W->scale;
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int row = 0; row < W->rows; row++) {
            const uint8_t *rd = base + (size_t)row * row_bytes;
            int done = 0;
            const int8_t imap[4] = {-1, 0, 1, 0};
            float sum = 0.0f;
            while (done < cols) {
                for (int gp = 0; gp < 32; gp++) {
                    uint8_t b = rd[gp];
                    sum += imap[(b >> 6) & 3] * x[done + 0*32 + gp];
                    sum += imap[(b >> 4) & 3] * x[done + 1*32 + gp];
                    sum += imap[(b >> 2) & 3] * x[done + 2*32 + gp];
                    sum += imap[(b >> 0) & 3] * x[done + 3*32 + gp];
                }
                rd += 32;
                done += 128;
            }
            out[row] = sum * scale;
        }
        }
#endif
        return;
    }

    if (W->type == 35) {  // TQ2_0: fused dequant + dot product
        const BlockTQ2 *blocks = (const BlockTQ2 *)W->data;
        int n_blocks_per_row = W->cols / QK_K;
        float tensor_scale = W->scale;

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int row = 0; row < W->rows; row++) {
            float row_sum = 0.0f;
            for (int b = 0; b < n_blocks_per_row; b++) {
                const BlockTQ2 *blk = &blocks[row * n_blocks_per_row + b];
                __builtin_prefetch(blk + 2, 0, 0);
                float d = fp16_to_fp32(blk->d);
                const float *xb = x + b * QK_K;
#ifdef __ARM_NEON
                float32x4_t accA0 = vdupq_n_f32(0), accA1 = vdupq_n_f32(0);
                float32x4_t accA2 = vdupq_n_f32(0), accA3 = vdupq_n_f32(0);
                float32x4_t accB0 = vdupq_n_f32(0), accB1 = vdupq_n_f32(0);
                float32x4_t accB2 = vdupq_n_f32(0), accB3 = vdupq_n_f32(0);
                const uint8x16_t mask3 = vdupq_n_u8(3);
                const int8x16_t one_s8 = vdupq_n_s8(1);
                for (int half = 0; half < 2; half++) {
                    const uint8_t *qs = blk->qs + half * 32;
                    const float *xh = xb + half * 128;
                    for (int i = 0; i < 2; i++) {
                        uint8x16_t raw = vld1q_u8(qs + i * 16);
                        const float *xp = xh + i * 16;
                        int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask3)), one_s8);
                        int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask3)), one_s8);
                        int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask3)), one_s8);
                        int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 6), mask3)), one_s8);
                        neon_acc_i8x16_f32(t0, xp + 0*32, &accA0, &accA1, &accA2, &accA3);
                        neon_acc_i8x16_f32(t1, xp + 1*32, &accB0, &accB1, &accB2, &accB3);
                        neon_acc_i8x16_f32(t2, xp + 2*32, &accA0, &accA1, &accA2, &accA3);
                        neon_acc_i8x16_f32(t3, xp + 3*32, &accB0, &accB1, &accB2, &accB3);
                    }
                }
                row_sum += (neon_reduce4(accA0, accA1, accA2, accA3) +
                            neon_reduce4(accB0, accB1, accB2, accB3)) * d;
#else
                float block_sum = 0.0f;
                for (int half = 0; half < 2; half++) {
                    const uint8_t *qs = blk->qs + half * 32;
                    const float *xh = xb + half * 128;
                    for (int m = 0; m < 32; m++) {
                        uint8_t byte = qs[m];
                        int8_t q0 = (int8_t)((byte >> 0) & 3) - 1;
                        int8_t q1 = (int8_t)((byte >> 2) & 3) - 1;
                        int8_t q2 = (int8_t)((byte >> 4) & 3) - 1;
                        int8_t q3 = (int8_t)((byte >> 6) & 3) - 1;
                        block_sum += q0 * xh[0*32 + m];
                        block_sum += q1 * xh[1*32 + m];
                        block_sum += q2 * xh[2*32 + m];
                        block_sum += q3 * xh[3*32 + m];
                    }
                }
                row_sum += block_sum * d;
#endif
            }
            out[row] = row_sum * tensor_scale;
        }
        return;
    }

    // TQ1_0 (type == 34): fused dequant + dot product
    {
        const BlockTQ1 *blocks = (const BlockTQ1 *)W->data;
        int n_blocks_per_row = W->cols / QK_K;
        float tensor_scale = W->scale;
        static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int row = 0; row < W->rows; row++) {
            float row_sum = 0.0f;
            for (int b = 0; b < n_blocks_per_row; b++) {
                const BlockTQ1 *blk = &blocks[row * n_blocks_per_row + b];
                __builtin_prefetch(blk + 2, 0, 0);
                float d = fp16_to_fp32(blk->d);
                float block_sum = 0.0f;
                const float *xb = x + b * QK_K;
#ifdef __ARM_NEON
                float32x4_t accA0 = vdupq_n_f32(0), accA1 = vdupq_n_f32(0);
                float32x4_t accA2 = vdupq_n_f32(0), accA3 = vdupq_n_f32(0);
                float32x4_t accB0 = vdupq_n_f32(0), accB1 = vdupq_n_f32(0);
                float32x4_t accB2 = vdupq_n_f32(0), accB3 = vdupq_n_f32(0);
                const int8x16_t one_s8 = vdupq_n_s8(1);
                const uint8x8_t three_u8 = vdup_n_u8(3);
                int acc_flip = 0;

                // Section 1: qs[0..31], 5 trits/byte -> 160 values
                for (int n = 0; n < 5; n++) {
                    uint8x16_t pow3_vec = vdupq_n_u8(pow3[n]);
                    for (int i = 0; i < 2; i++) {
                        uint8x16_t raw = vld1q_u8(blk->qs + i * 16);
                        uint8x16_t q = vmulq_u8(raw, pow3_vec);
                        uint8x8_t xi_lo = vshrn_n_u16(vmull_u8(vget_low_u8(q), three_u8), 8);
                        uint8x8_t xi_hi = vshrn_n_u16(vmull_u8(vget_high_u8(q), three_u8), 8);
                        int8x16_t ternary = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(xi_lo, xi_hi)), one_s8);
                        if (acc_flip++ & 1)
                            neon_acc_i8x16_f32(ternary, xb + n*32 + i*16, &accB0, &accB1, &accB2, &accB3);
                        else
                            neon_acc_i8x16_f32(ternary, xb + n*32 + i*16, &accA0, &accA1, &accA2, &accA3);
                    }
                }

                // Section 2: qs[32..47], 5 trits/byte -> 80 values
                for (int n = 0; n < 5; n++) {
                    uint8x16_t raw = vld1q_u8(blk->qs + 32);
                    uint8x16_t q = vmulq_u8(raw, vdupq_n_u8(pow3[n]));
                    uint8x8_t xi_lo = vshrn_n_u16(vmull_u8(vget_low_u8(q), three_u8), 8);
                    uint8x8_t xi_hi = vshrn_n_u16(vmull_u8(vget_high_u8(q), three_u8), 8);
                    int8x16_t ternary = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(xi_lo, xi_hi)), one_s8);
                    if (acc_flip++ & 1)
                        neon_acc_i8x16_f32(ternary, xb + 160 + n*16, &accB0, &accB1, &accB2, &accB3);
                    else
                        neon_acc_i8x16_f32(ternary, xb + 160 + n*16, &accA0, &accA1, &accA2, &accA3);
                }

                block_sum = neon_reduce4(accA0, accA1, accA2, accA3) +
                            neon_reduce4(accB0, accB1, accB2, accB3);

                // Section 3: qh[0..3], 4 trits/byte -> 16 values (scalar)
                for (int n = 0; n < 4; n++) {
                    for (int m = 0; m < 4; m++) {
                        uint8_t q = blk->qh[m] * pow3[n];
                        int16_t xi = ((uint16_t)q * 3) >> 8;
                        block_sum += (xi - 1) * xb[240 + n*4 + m];
                    }
                }
#else
                // Section 1: qs[0..31], 5 trits/byte -> 160 values
                for (int n = 0; n < 5; n++) {
                    for (int m = 0; m < 32; m++) {
                        uint8_t q = blk->qs[m] * pow3[n];
                        int16_t xi = ((uint16_t)q * 3) >> 8;
                        block_sum += (xi - 1) * xb[n*32 + m];
                    }
                }

                // Section 2: qs[32..47], 5 trits/byte -> 80 values
                for (int n = 0; n < 5; n++) {
                    for (int m = 0; m < 16; m++) {
                        uint8_t q = blk->qs[32 + m] * pow3[n];
                        int16_t xi = ((uint16_t)q * 3) >> 8;
                        block_sum += (xi - 1) * xb[160 + n*16 + m];
                    }
                }

                // Section 3: qh[0..3], 4 trits/byte -> 16 values
                for (int n = 0; n < 4; n++) {
                    for (int m = 0; m < 4; m++) {
                        uint8_t q = blk->qh[m] * pow3[n];
                        int16_t xi = ((uint16_t)q * 3) >> 8;
                        block_sum += (xi - 1) * xb[240 + n*4 + m];
                    }
                }
#endif
                row_sum += block_sum * d;
            }
            out[row] = row_sum * tensor_scale;
        }
    }
}

// --- Batch ternary matvec ---
// Runs multiple independent matvecs with a single OMP parallel region.

void ternary_matvec_batch(const MatvecTask *tasks, int n_tasks,
                           const float *x, int8_t *x_q_buf) {
    if (n_tasks <= 0) return;

    // Check if all tasks are I2_S (for SDOT fast path)
    int cols = tasks[0].W->cols;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int all_i2s = 1;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].W->type != 36) { all_i2s = 0; break; }
    }

    if (all_i2s) {
        // Quantize x to int8 once, shared across all tasks
        float x_scale = quantize_x_to_i8(x, x_q_buf, cols);

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            for (int t = 0; t < n_tasks; t++) {
                i2s_matvec_sdot(tasks[t].out, tasks[t].W, x_q_buf, x_scale);
            }
        }
        return;
    }
#else
    (void)x_q_buf;  // unused on non-DOTPROD platforms
#endif

    // Fallback: use existing per-task matvec (each forks its own OMP region)
    for (int t = 0; t < n_tasks; t++) {
        ternary_matvec(tasks[t].out, tasks[t].W, x);
    }
}
