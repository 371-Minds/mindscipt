#include "quant_internal.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_q4_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ4Ctx *c = (BnQ4Ctx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            v128_t raw = wasm_v128_load(blk->qs);
            v128_t lo = wasm_i8x16_sub(wasm_v128_and(raw, wasm_i8x16_splat(0xF)), wasm_i8x16_splat(8));
            v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), wasm_i8x16_splat(8));
            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);
            {
                v128_t lo16 = wasm_i16x8_extend_low_i8x16(lo);
                v128_t hi16 = wasm_i16x8_extend_high_i8x16(lo);
#ifdef __wasm_relaxed_simd__
                acc0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)),  wasm_v128_load(xb),      acc0);
                acc1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xb + 4),  acc1);
                acc2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)),  wasm_v128_load(xb + 8),  acc2);
                acc3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xb + 12), acc3);
#else
                acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), wasm_v128_load(xb)));
                acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xb + 4)));
                acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), wasm_v128_load(xb + 8)));
                acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xb + 12)));
#endif
            }
            {
                v128_t lo16 = wasm_i16x8_extend_low_i8x16(hi);
                v128_t hi16 = wasm_i16x8_extend_high_i8x16(hi);
#ifdef __wasm_relaxed_simd__
                acc0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)),  wasm_v128_load(xb + 16), acc0);
                acc1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xb + 20), acc1);
                acc2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)),  wasm_v128_load(xb + 24), acc2);
                acc3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xb + 28), acc3);
#else
                acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), wasm_v128_load(xb + 16)));
                acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xb + 20)));
                acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), wasm_v128_load(xb + 24)));
                acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xb + 28)));
#endif
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum) * d;
        }
        c->out[row] = row_sum;
    }
}

// Relaxed SIMD SDOT path — mirrors NEON SDOT: integer dot product on Q4×Q8 blocks
#ifdef __wasm_relaxed_simd__
void bn_quant_q4_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const v128_t mask_lo = wasm_i8x16_splat(0xF);
    const v128_t eight = wasm_i8x16_splat(8);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            float d_q4 = bn_fp16_to_fp32(blk->d);
            float d_q8 = x_scales[b];

            v128_t raw = wasm_v128_load(blk->qs);
            v128_t lo = wasm_i8x16_sub(wasm_v128_and(raw, mask_lo), eight);
            v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), eight);

            const int8_t *xb = x_q + b * 32;
            v128_t acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(lo, wasm_v128_load(xb), wasm_i32x4_splat(0));
            acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(hi, wasm_v128_load(xb + 16), acc);

            // Horizontal sum
            v128_t shuf = wasm_i32x4_shuffle(acc, acc, 2, 3, 0, 1);
            acc = wasm_i32x4_add(acc, shuf);
            shuf = wasm_i32x4_shuffle(acc, acc, 1, 0, 3, 2);
            acc = wasm_i32x4_add(acc, shuf);

            row_sum += d_q4 * d_q8 * (float)wasm_i32x4_extract_lane(acc, 0);
        }
        c->out[row] = row_sum;
    }
}
#else
void bn_quant_q4_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    (void)ctx; (void)row_start; (void)row_end;
}
#endif
