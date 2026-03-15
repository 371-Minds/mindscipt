#include "transformer_internal.h"

#ifdef __AVX2__

void bn_transformer_rmsnorm_avx2(float *out, const float *x, const float *w, int size, float eps) {
    __m256 sum_sq = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        sum_sq = _mm256_fmadd_ps(xv, xv, sum_sq);
    }
    float ss = bn_avx2_hsum_ps(sum_sq);
    for (; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    __m256 ss_v = _mm256_set1_ps(ss);
    for (i = 0; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 wv = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(xv, ss_v), wv));
    }
    for (; i < size; i++) out[i] = x[i] * ss * w[i];
}

#endif // __AVX2__
