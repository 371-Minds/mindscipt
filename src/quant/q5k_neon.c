#include "quant_internal.h"
#include "quant_neon_helpers.h"
#include <arm_neon.h>

// Extract high bits for 16 consecutive l-positions from qh, testing bit `bit_pos`.
// qh[l] stores high bits for position l: bits are interleaved as lo/hi pairs per group.
// bit_pos = group*2 (0,2,4,6) for first half, group*2+1 (1,3,5,7) for second half.
// l_offset = starting l position (0 or 16).
// Returns 16 bytes, each 0x00 or 0x10.
static inline uint8x16_t q5k_extract_hb_neon(const uint8_t *qh, int l_offset, int bit_pos) {
    // Load 16 consecutive qh bytes starting at l_offset
    uint8x16_t qh_vec = vld1q_u8(qh + l_offset);
    // Test the target bit using a mask
    uint8x16_t mask = vdupq_n_u8(1 << bit_pos);
    uint8x16_t tested = vandq_u8(qh_vec, mask);
    uint8x16_t nonzero = vcgtq_u8(tested, vdupq_n_u8(0));
    return vandq_u8(nonzero, vdupq_n_u8(0x10));
}

void bn_quant_q5k_neon_range(void *ctx, int row_start, int row_end) {
    BnQ5KCtx *c = (BnQ5KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const float *x = c->x;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ5K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const uint8_t *qh = blk->qh;
            const float *xb = x + b * BN_QK_K;

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;
                int group = j / 64;  // 0..3
                int bit_lo = group * 2;      // bits 0,2,4,6 for first 32
                int bit_hi = group * 2 + 1;  // bits 1,3,5,7 for second 32

                uint8x16_t raw0 = vld1q_u8(qs);
                uint8x16_t raw1 = vld1q_u8(qs + 16);

                uint8x16_t hb0 = q5k_extract_hb_neon(qh, 0,  bit_lo);  // l=0..15, first half
                uint8x16_t hb1 = q5k_extract_hb_neon(qh, 16, bit_lo);  // l=16..31, first half
                uint8x16_t hb2 = q5k_extract_hb_neon(qh, 0,  bit_hi);  // l=0..15, second half
                uint8x16_t hb3 = q5k_extract_hb_neon(qh, 16, bit_hi);  // l=16..31, second half

                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                float ds = d * sc;
                float dm = dmin * m;
                int8x16_t w0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(raw0, mask_lo), hb0));
                int8x16_t w1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(raw1, mask_lo), hb1));
                BN_QK_ACC_SCALED_16(w0, xb + j, ds, dm);
                BN_QK_ACC_SCALED_16(w1, xb + j + 16, ds, dm);

                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                ds = d * sc;
                dm = dmin * m;
                w0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(raw0, 4), hb2));
                w1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(raw1, 4), hb3));
                BN_QK_ACC_SCALED_16(w0, xb + j + 32, ds, dm);
                BN_QK_ACC_SCALED_16(w1, xb + j + 48, ds, dm);

                qs += 32;
            }

            float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
            row_sum += vget_lane_f32(vpadd_f32(r, r), 0);
        }
        c->out[row] = row_sum;
    }
}
