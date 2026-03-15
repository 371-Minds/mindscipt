#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q8_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 2), _MM_HINT_T0);
            float d_w = bn_fp16_to_fp32(blk->d);
            float d_x = x_scales[b];

            __m256i w256 = _mm256_loadu_si256((const __m256i *)blk->qs);
            __m256i xq256 = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));

            __m256i acc = bn_avx2_dpbusd(_mm256_setzero_si256(), w256, xq256);
            row_sum += d_w * d_x * (float)bn_avx2_hsum_epi32(acc);
        }
        c->out[row] = row_sum;
    }
}
