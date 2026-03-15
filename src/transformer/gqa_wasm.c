#include "transformer_internal.h"

#ifdef __wasm_simd128__

void bn_transformer_gqa_wasm_range(void *ctx, int h_start, int h_end) {
    BnGQACtx *g = (BnGQACtx *)ctx;
    const BnConfig *c = g->c;
    BnRunState *s = g->s;
    int head_size = g->head_size;
    int kv_dim = g->kv_dim;
    int kv_mul = g->kv_mul;
    int n_kv = g->n_kv;
    int seq_len = g->seq_len;
    int start = g->pos - n_kv + 1;
    size_t loff = g->loff;
    int kv_f16 = c->kv_f16;

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        float *att = s->att + h * seq_len;
        int kv_h = h / kv_mul;
        float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);

        for (int i = 0; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            float k_buf[head_size];
            const float *k_t;
            if (kv_f16) {
                const uint16_t *k_f16 = (const uint16_t *)s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                for (int d = 0; d < head_size; d++) k_buf[d] = bn_fp16_to_fp32(k_f16[d]);
                k_t = k_buf;
            } else {
                k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
            }
            v128_t wa0 = wasm_f32x4_splat(0), wa1 = wasm_f32x4_splat(0);
            for (int d = 0; d < head_size; d += 8) {
#ifdef __wasm_relaxed_simd__
                wa0 = wasm_f32x4_relaxed_madd(wasm_v128_load(q_h + d),     wasm_v128_load(k_t + d),     wa0);
                wa1 = wasm_f32x4_relaxed_madd(wasm_v128_load(q_h + d + 4), wasm_v128_load(k_t + d + 4), wa1);
#else
                wa0 = wasm_f32x4_add(wa0, wasm_f32x4_mul(wasm_v128_load(q_h + d),     wasm_v128_load(k_t + d)));
                wa1 = wasm_f32x4_add(wa1, wasm_f32x4_mul(wasm_v128_load(q_h + d + 4), wasm_v128_load(k_t + d + 4)));
#endif
            }
            att[i] = bn_wasm_hsum_f32x4(wasm_f32x4_add(wa0, wa1)) * inv_sqrt_hs;
        }

        bn_transformer_softmax(att, n_kv);

        float *xb_h = s->xb + h * head_size;
        memset(xb_h, 0, head_size * sizeof(float));
        for (int i = 0; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            float v_buf[head_size];
            const float *v_t;
            if (kv_f16) {
                const uint16_t *v_f16 = (const uint16_t *)s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                for (int d = 0; d < head_size; d++) v_buf[d] = bn_fp16_to_fp32(v_f16[d]);
                v_t = v_buf;
            } else {
                v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
            }
            float a = att[i];
            v128_t wav = wasm_f32x4_splat(a);
            for (int d = 0; d < head_size; d += 4) {
                v128_t cur = wasm_v128_load(xb_h + d);
#ifdef __wasm_relaxed_simd__
                wasm_v128_store(xb_h + d, wasm_f32x4_relaxed_madd(wav, wasm_v128_load(v_t + d), cur));
#else
                wasm_v128_store(xb_h + d, wasm_f32x4_add(cur, wasm_f32x4_mul(wav, wasm_v128_load(v_t + d))));
#endif
            }
        }
    }
}

#endif // __wasm_simd128__
