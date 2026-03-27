#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

/* Q4_0 AVX2 tiled matmul with float-domain accumulation.
 *
 * Tiles token dimension (TILE_T=8) for L1 cache residency.
 * Uses cvtepi32_ps + FMA instead of integer hsum per block — eliminates
 * n_bpr serial integer hsums per (row, token) pair and replaces with
 * pipelined float FMAs. ONE float hsum at the end of all blocks.
 */

#define Q4_MATMUL_TILE_T 8

typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_scales;
    int n_tokens;
    int cols;
} BnQ4MatmulCtx;

void bn_quant_q4_avx2_matmul_range(void *ctx, int row_start, int row_end) {
    BnQ4MatmulCtx *c = (BnQ4MatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / 32;
    int n_tokens = c->n_tokens;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0xF);
    const __m128i bias = _mm_set1_epi8(8);

    for (int t0 = 0; t0 < n_tokens; t0 += Q4_MATMUL_TILE_T) {
        int t_end = t0 + Q4_MATMUL_TILE_T;
        if (t_end > n_tokens) t_end = n_tokens;
        int tile_n = t_end - t0;

        for (int row = row_start; row < row_end; row++) {
            /* Per-token float accumulators for this row */
            __m256 facc[Q4_MATMUL_TILE_T];
            for (int i = 0; i < tile_n; i++) facc[i] = _mm256_setzero_ps();

            for (int b = 0; b < n_bpr; b++) {
                /* Load and unpack weight block ONCE for all tokens */
                const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_bpr + b];
                __m256 d_q4 = _mm256_set1_ps(bn_fp16_to_fp32(blk->d));

                __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
                __m128i lo = _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias);
                __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias);
                __m256i w256 = _mm256_set_m128i(hi, lo);

                for (int ti = 0; ti < tile_n; ti++) {
                    int t = t0 + ti;
                    __m256i xq = _mm256_loadu_si256((const __m256i *)(x_q + (size_t)t * cols + b * 32));
                    __m256i dot = bn_avx2_dpbusd(_mm256_setzero_si256(), w256, xq);
                    __m256 scale = _mm256_mul_ps(d_q4, _mm256_set1_ps(x_scales[(size_t)t * n_bpr + b]));
                    facc[ti] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), scale, facc[ti]);
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += bn_avx2_hsum_ps(facc[ti]);
        }
    }
}
