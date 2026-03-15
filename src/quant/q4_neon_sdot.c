#include "quant_internal.h"
#include <arm_neon.h>

void bn_quant_q4_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int8x16_t eight = vdupq_n_s8(8);
    const int32x4_t zero = vdupq_n_s32(0);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        int base = row * n_blocks_per_row;
        int b = 0;

        // Process 4 blocks at a time for better ILP
        for (; b + 3 < n_blocks_per_row; b += 4) {
            const BnBlockQ4_0 *b0 = &blocks[base + b];
            const BnBlockQ4_0 *b1 = &blocks[base + b + 1];
            const BnBlockQ4_0 *b2 = &blocks[base + b + 2];
            const BnBlockQ4_0 *b3 = &blocks[base + b + 3];
            __builtin_prefetch(b0 + 6, 0, 0);

            float d0 = bn_fp16_to_fp32(b0->d);
            float d1 = bn_fp16_to_fp32(b1->d);
            float d2 = bn_fp16_to_fp32(b2->d);
            float d3 = bn_fp16_to_fp32(b3->d);

            float dx0 = x_scales[b];
            float dx1 = x_scales[b + 1];
            float dx2 = x_scales[b + 2];
            float dx3 = x_scales[b + 3];

            const int8_t *xb0 = x_q + (b)     * 32;
            const int8_t *xb1 = x_q + (b + 1) * 32;
            const int8_t *xb2 = x_q + (b + 2) * 32;
            const int8_t *xb3 = x_q + (b + 3) * 32;

            // Block 0
            uint8x16_t raw0 = vld1q_u8(b0->qs);
            int8x16_t lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw0, mask_lo)), eight);
            int8x16_t hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4)), eight);
            int32x4_t a0 = vdotq_s32(zero, lo0, vld1q_s8(xb0));
            a0 = vdotq_s32(a0, hi0, vld1q_s8(xb0 + 16));

            // Block 1
            uint8x16_t raw1 = vld1q_u8(b1->qs);
            int8x16_t lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw1, mask_lo)), eight);
            int8x16_t hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4)), eight);
            int32x4_t a1 = vdotq_s32(zero, lo1, vld1q_s8(xb1));
            a1 = vdotq_s32(a1, hi1, vld1q_s8(xb1 + 16));

            // Block 2
            uint8x16_t raw2 = vld1q_u8(b2->qs);
            int8x16_t lo2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw2, mask_lo)), eight);
            int8x16_t hi2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw2, 4)), eight);
            int32x4_t a2 = vdotq_s32(zero, lo2, vld1q_s8(xb2));
            a2 = vdotq_s32(a2, hi2, vld1q_s8(xb2 + 16));

            // Block 3
            uint8x16_t raw3 = vld1q_u8(b3->qs);
            int8x16_t lo3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw3, mask_lo)), eight);
            int8x16_t hi3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw3, 4)), eight);
            int32x4_t a3 = vdotq_s32(zero, lo3, vld1q_s8(xb3));
            a3 = vdotq_s32(a3, hi3, vld1q_s8(xb3 + 16));

            // Deferred reduction
            row_sum += d0 * dx0 * (float)vaddvq_s32(a0)
                     + d1 * dx1 * (float)vaddvq_s32(a1)
                     + d2 * dx2 * (float)vaddvq_s32(a2)
                     + d3 * dx3 * (float)vaddvq_s32(a3);
        }

        // Tail: remaining blocks
        for (; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[base + b];
            float d_q4 = bn_fp16_to_fp32(blk->d);
            float d_q8 = x_scales[b];

            uint8x16_t raw = vld1q_u8(blk->qs);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);

            const int8_t *xb = x_q + b * 32;
            int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
            acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));

            row_sum += d_q4 * d_q8 * (float)vaddvq_s32(acc);
        }

        c->out[row] = row_sum;
    }
}

void bn_quant_q4_repacked_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const float *rp_scales = c->W->rp_scales;
    const uint8_t *rp_qs = c->W->rp_qs;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int8x16_t eight = vdupq_n_s8(8);
    const int32x4_t zero = vdupq_n_s32(0);

    int row = row_start;

    // Pre-tail: rows before first 4-aligned boundary
    for (; row < row_end && (row & 3); row++) {
        int group = row >> 2;
        int offset = row & 3;
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            size_t gb = (size_t)group * n_blocks_per_row + b;
            float d = rp_scales[gb * 4 + offset];
            float dx = x_scales[b];
            const int8_t *xb = x_q + b * 32;

            // Extract this row's bytes from nibble-transposed layout
            const uint8_t *base = rp_qs + gb * 64;
            uint8_t row_qs[16];
            for (int ng = 0; ng < 4; ng++)
                memcpy(row_qs + ng * 4, base + ng * 16 + offset * 4, 4);

            uint8x16_t raw = vld1q_u8(row_qs);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
            int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
            acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));
            row_sum += d * dx * (float)vaddvq_s32(acc);
        }
        c->out[row] = row_sum;
    }

    // Main loop: 4 rows at a time (nibble-transposed layout + vdotq_laneq_s32)
    for (; row + 3 < row_end; row += 4) {
        int group = row >> 2;
        float32x4_t row_sums = vdupq_n_f32(0.0f);

        for (int b = 0; b < n_blocks_per_row; b++) {
            // Load activation ONCE per block
            int8x16_t a0 = vld1q_s8(x_q + b * 32);
            int8x16_t a1 = vld1q_s8(x_q + b * 32 + 16);
            float dx = x_scales[b];

            size_t gb = (size_t)group * n_blocks_per_row + b;
            const uint8_t *qbase = rp_qs + gb * 64;

            // Load 4 × 16 nibble-transposed packed bytes
            uint8x16_t raw0 = vld1q_u8(qbase);
            uint8x16_t raw1 = vld1q_u8(qbase + 16);
            uint8x16_t raw2 = vld1q_u8(qbase + 32);
            uint8x16_t raw3 = vld1q_u8(qbase + 48);

            // Unpack low nibbles (elements 0-15)
            int8x16_t lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw0, mask_lo)), eight);
            int8x16_t lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw1, mask_lo)), eight);
            int8x16_t lo2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw2, mask_lo)), eight);
            int8x16_t lo3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw3, mask_lo)), eight);

            // Unpack high nibbles (elements 16-31)
            int8x16_t hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4)), eight);
            int8x16_t hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4)), eight);
            int8x16_t hi2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw2, 4)), eight);
            int8x16_t hi3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw3, 4)), eight);

            // 8 SDOT with lane selection — 4 independent accumulators to avoid
            // serial dependency chain (each pair is 2-deep, all 4 run in parallel)
            int32x4_t acc02 = vdotq_laneq_s32(zero, lo0, a0, 0);
            int32x4_t acc13 = vdotq_laneq_s32(zero, lo1, a0, 1);
            acc02 = vdotq_laneq_s32(acc02, lo2, a0, 2);
            acc13 = vdotq_laneq_s32(acc13, lo3, a0, 3);
            int32x4_t acc46 = vdotq_laneq_s32(zero, hi0, a1, 0);
            int32x4_t acc57 = vdotq_laneq_s32(zero, hi1, a1, 1);
            acc46 = vdotq_laneq_s32(acc46, hi2, a1, 2);
            acc57 = vdotq_laneq_s32(acc57, hi3, a1, 3);
            int32x4_t acc = vaddq_s32(vaddq_s32(acc02, acc13),
                                      vaddq_s32(acc46, acc57));

            // Vector scale multiply + accumulate (no per-row reduction!)
            float32x4_t f = vcvtq_f32_s32(acc);
            float32x4_t d4 = vld1q_f32(rp_scales + gb * 4);
            row_sums = vfmaq_f32(row_sums, f, vmulq_n_f32(d4, dx));
        }

        vst1q_f32(&c->out[row], row_sums);
    }

    // Post-tail: remaining 0-3 rows
    for (; row < row_end; row++) {
        int group = row >> 2;
        int offset = row & 3;
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            size_t gb = (size_t)group * n_blocks_per_row + b;
            float d = rp_scales[gb * 4 + offset];
            float dx = x_scales[b];
            const int8_t *xb = x_q + b * 32;

            // Extract this row's bytes from nibble-transposed layout
            const uint8_t *base = rp_qs + gb * 64;
            uint8_t row_qs[16];
            for (int ng = 0; ng < 4; ng++)
                memcpy(row_qs + ng * 4, base + ng * 16 + offset * 4, 4);

            uint8x16_t raw = vld1q_u8(row_qs);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
            int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
            acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));
            row_sum += d * dx * (float)vaddvq_s32(acc);
        }
        c->out[row] = row_sum;
    }
}
