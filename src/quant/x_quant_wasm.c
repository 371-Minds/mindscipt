#include "quant_internal.h"
#include <wasm_simd128.h>
#include <math.h>

#ifdef __wasm_relaxed_simd__

// Quantize float vector x[n] to int8, returning scale = amax/127.
float bn_quant_x_to_i8(const float *x, int8_t *x_q, int n) {
    // Find absolute max
    v128_t vmax = wasm_f32x4_splat(0);
    v128_t sign_mask = wasm_i32x4_splat(0x7FFFFFFF);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(x + i), sign_mask));
        vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(x + i + 4), sign_mask));
        vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(x + i + 8), sign_mask));
        vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(x + i + 12), sign_mask));
    }
    for (; i + 3 < n; i += 4)
        vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(x + i), sign_mask));

    // Horizontal max
    v128_t shuf = wasm_i32x4_shuffle(vmax, vmax, 2, 3, 0, 1);
    vmax = wasm_f32x4_max(vmax, shuf);
    shuf = wasm_i32x4_shuffle(vmax, vmax, 1, 0, 3, 2);
    vmax = wasm_f32x4_max(vmax, shuf);
    float amax = wasm_f32x4_extract_lane(vmax, 0);

    for (; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }

    if (amax == 0.0f) {
        memset(x_q, 0, n);
        return 0.0f;
    }

    float scale = amax / (float)BN_I8_MAX;
    float inv_scale = (float)BN_I8_MAX / amax;
    v128_t vinv = wasm_f32x4_splat(inv_scale);

    i = 0;
    for (; i + 15 < n; i += 16) {
        // Convert float to int32 (round to nearest)
        v128_t f0 = wasm_f32x4_mul(wasm_v128_load(x + i), vinv);
        v128_t f1 = wasm_f32x4_mul(wasm_v128_load(x + i + 4), vinv);
        v128_t f2 = wasm_f32x4_mul(wasm_v128_load(x + i + 8), vinv);
        v128_t f3 = wasm_f32x4_mul(wasm_v128_load(x + i + 12), vinv);
        v128_t i0 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f0));
        v128_t i1 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f1));
        v128_t i2 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f2));
        v128_t i3 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f3));
        // Narrow i32 -> i16 -> i8 with saturation
        v128_t s01 = wasm_i16x8_narrow_i32x4(i0, i1);
        v128_t s23 = wasm_i16x8_narrow_i32x4(i2, i3);
        v128_t bytes = wasm_i8x16_narrow_i16x8(s01, s23);
        wasm_v128_store(x_q + i, bytes);
    }
    for (; i < n; i++) {
        int v = (int)roundf(x[i] * inv_scale);
        x_q[i] = (int8_t)(v < -BN_I8_MAX ? -BN_I8_MAX : (v > BN_I8_MAX ? BN_I8_MAX : v));
    }
    return scale;
}

// Per-block Q8_0 quantization for Q4_0 integer dot product path.
void bn_quant_x_to_q8_blocks(const float *x, int8_t *x_q, float *x_scales, int n) {
    int n_blocks = n / 32;
    v128_t sign_mask = wasm_i32x4_splat(0x7FFFFFFF);
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + b * 32;
        int8_t *qb = x_q + b * 32;

        v128_t v0 = wasm_v128_and(wasm_v128_load(xb), sign_mask);
        v128_t v1 = wasm_v128_and(wasm_v128_load(xb + 4), sign_mask);
        v128_t v2 = wasm_v128_and(wasm_v128_load(xb + 8), sign_mask);
        v128_t v3 = wasm_v128_and(wasm_v128_load(xb + 12), sign_mask);
        v128_t v4 = wasm_v128_and(wasm_v128_load(xb + 16), sign_mask);
        v128_t v5 = wasm_v128_and(wasm_v128_load(xb + 20), sign_mask);
        v128_t v6 = wasm_v128_and(wasm_v128_load(xb + 24), sign_mask);
        v128_t v7 = wasm_v128_and(wasm_v128_load(xb + 28), sign_mask);
        v128_t vmax = wasm_f32x4_max(wasm_f32x4_max(wasm_f32x4_max(v0, v1), wasm_f32x4_max(v2, v3)),
                                      wasm_f32x4_max(wasm_f32x4_max(v4, v5), wasm_f32x4_max(v6, v7)));
        v128_t shuf = wasm_i32x4_shuffle(vmax, vmax, 2, 3, 0, 1);
        vmax = wasm_f32x4_max(vmax, shuf);
        shuf = wasm_i32x4_shuffle(vmax, vmax, 1, 0, 3, 2);
        vmax = wasm_f32x4_max(vmax, shuf);
        float amax = wasm_f32x4_extract_lane(vmax, 0);

        if (amax == 0.0f) {
            memset(qb, 0, 32);
            x_scales[b] = 0.0f;
            continue;
        }

        float inv_scale = 127.0f / amax;
        x_scales[b] = amax / 127.0f;
        v128_t vinv = wasm_f32x4_splat(inv_scale);

        v128_t i0 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb), vinv)));
        v128_t i1 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 4), vinv)));
        v128_t i2 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 8), vinv)));
        v128_t i3 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 12), vinv)));
        v128_t i4 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 16), vinv)));
        v128_t i5 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 20), vinv)));
        v128_t i6 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 24), vinv)));
        v128_t i7 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 28), vinv)));

        v128_t s01 = wasm_i16x8_narrow_i32x4(i0, i1);
        v128_t s23 = wasm_i16x8_narrow_i32x4(i2, i3);
        v128_t s45 = wasm_i16x8_narrow_i32x4(i4, i5);
        v128_t s67 = wasm_i16x8_narrow_i32x4(i6, i7);
        wasm_v128_store(qb, wasm_i8x16_narrow_i16x8(s01, s23));
        wasm_v128_store(qb + 16, wasm_i8x16_narrow_i16x8(s45, s67));
    }
}

#endif // __wasm_relaxed_simd__
