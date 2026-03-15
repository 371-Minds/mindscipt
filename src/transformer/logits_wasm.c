#include "transformer_internal.h"

#ifdef __wasm_simd128__

void bn_transformer_logits_f16_wasm_range(void *ctx, int v_start, int v_end) {
    BnLogitsCtx *lc = (BnLogitsCtx *)ctx;
    const uint16_t *emb = (const uint16_t *)lc->emb;
    const float *x = lc->x;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const uint16_t *row = emb + (size_t)v * dim;
        v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
        for (int d = 0; d < dim; d += 8) {
            // Software F16->F32 conversion (no hardware F16C on WASM)
            float f0[4], f1[4];
            for (int k = 0; k < 4; k++) f0[k] = bn_fp16_to_fp32(row[d + k]);
            for (int k = 0; k < 4; k++) f1[k] = bn_fp16_to_fp32(row[d + 4 + k]);
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(f0), wasm_v128_load(x + d)));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(f1), wasm_v128_load(x + d + 4)));
        }
        lc->logits[v] = bn_wasm_hsum_f32x4(wasm_f32x4_add(acc0, acc1));
    }
}

#endif // __wasm_simd128__
