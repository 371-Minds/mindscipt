#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

/* Q4_0 AVX2 SDOT matvec with float-domain accumulation.
 * Replaces per-block integer hsum with cvtepi32_ps + FMA pipelining.
 * One float hsum per row instead of n_blocks integer hsums.
 */

void bn_quant_q4_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_bpr = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0xF);
    const __m128i bias = _mm_set1_epi8(8);

    for (int row = row_start; row < row_end; row++) {
        __m256 facc = _mm256_setzero_ps();

        for (int b = 0; b < n_bpr; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_bpr + b];
            _mm_prefetch((const char *)(blk + 4), _MM_HINT_T0);

            __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
            __m128i lo = _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias);
            __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias);
            __m256i w256 = _mm256_set_m128i(hi, lo);
            __m256i xq = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));

            __m256i dot = bn_avx2_dpbusd(_mm256_setzero_si256(), w256, xq);
            __m256 scale = _mm256_set1_ps(bn_fp16_to_fp32(blk->d) * x_scales[b]);
            facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), scale, facc);
        }

        c->out[row] = bn_avx2_hsum_ps(facc);
    }
}
