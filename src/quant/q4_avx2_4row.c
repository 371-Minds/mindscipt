#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

/* 4-row Q4_0 matvec with float-domain accumulation.
 *
 * Instead of: DPBUSD → integer hsum → scalar multiply (serial per block)
 * We do:      DPBUSD → cvtepi32_ps → FMA with scale (pipelined)
 * Then ONE float hsum at the end of all blocks per row.
 *
 * This eliminates n_blocks integer hsums (5 cycles each, serial) and
 * replaces them with n_blocks cvt+FMA (pipelined, ~1 cycle throughput)
 * plus 1 float hsum per row.
 */

void bn_quant_q4_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_bpr = c->W->cols / 32;
    int rows = c->W->rows;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0xF);
    const __m128i bias = _mm_set1_epi8(8);

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        /* Float accumulators — 8 lanes per row, hsum at the end */
        __m256 facc[4];
        for (int r = 0; r < nrows; r++) facc[r] = _mm256_setzero_ps();

        for (int b = 0; b < n_bpr; b++) {
            __m256i xq256 = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));
            __m256 d_q8 = _mm256_set1_ps(x_scales[b]);

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ4_0 *blk = &blocks[(row0 + r) * n_bpr + b];

                __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
                __m128i lo = _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias);
                __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias);
                __m256i w256 = _mm256_set_m128i(hi, lo);

                __m256i dot = bn_avx2_dpbusd(_mm256_setzero_si256(), w256, xq256);
                __m256 scale = _mm256_mul_ps(d_q8, _mm256_set1_ps(bn_fp16_to_fp32(blk->d)));
                facc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), scale, facc[r]);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = bn_avx2_hsum_ps(facc[r]);
    }
}
