#include "transformer_internal.h"
#include "sh_log.h"
#include <stdio.h>

// Max elements for stack VLAs (head_size, dim). Prevents stack overflow
// from malicious model configs. 8192 = 32KB of floats, well within stack.
#define BN_MAX_VLA_ELEMS 8192

// Backend-selected rmsnorm
#ifdef __ARM_NEON
#define rmsnorm bn_transformer_rmsnorm_neon
#elif defined(__AVX2__)
#define rmsnorm bn_transformer_rmsnorm_avx2
#elif defined(__wasm_simd128__)
#define rmsnorm bn_transformer_rmsnorm_wasm
#else
#define rmsnorm bn_transformer_rmsnorm_scalar
#endif

// --- Forward pass ---

// Embed + all layers (attention + FFN). Populates KV cache at `pos`.
// Leaves final activation in s->x. Returns 0 on success, -1 on error.
static int forward_layers(BnModel *m, int token, int pos) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &m->state;
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul;
    int head_size = c->head_size;

    // Guard against stack overflow from VLAs sized by model config
    if (head_size > BN_MAX_VLA_ELEMS || dim > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("Model dimensions too large for stack VLAs");
        return -1;
    }

    // #9: Validate token bounds
    if (token < 0 || token >= c->vocab_size) {
        SH_LOG_ERROR("Token out of range");
        return -1;
    }

    // #10: Validate pos bounds
    if (pos < 0) {
        SH_LOG_ERROR("Position out of range");
        return -1;
    }

    // Embed the token
    bn_model_embed_token(m, s->x, token);

    // Precompute RoPE cos/sin for this position (128 trig calls total,
    // vs 96,000 if computed per-head per-layer)
    int half_head = head_size / 2;
    float rope_cos[half_head], rope_sin[half_head];
    for (int i = 0; i < half_head; i++) {
        float angle = pos * s->rope_freq[i];
        rope_cos[i] = cosf(angle);
        rope_sin[i] = sinf(angle);
    }

    // Process each layer
    int cache_pos = pos % c->seq_len;
    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        size_t loff = (size_t)l * c->seq_len * kv_dim;
        float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_dim;
        float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_dim;

        // ---- Attention block ----

        rmsnorm(s->xb, s->x, lw->attn_norm, dim, c->norm_eps);

        if (c->kv_f16) {
            // F16 KV cache: write K/V to temp F32 buffers, apply RoPE, convert to F16
            float *k_tmp = s->hb, *v_tmp = s->hb2;  // [hidden_dim] >= kv_dim
            BnMatvecTask qkv[3] = {
                { s->q,  &lw->wq },
                { k_tmp, &lw->wk },
                { v_tmp, &lw->wv },
            };
            bn_quant_matvec_batch(qkv, 3, s->xb, s->x_q, m->pool);

            // Add attention biases (Qwen2)
            if (lw->q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->q_bias[i];
            if (lw->k_bias) for (int i = 0; i < kv_dim; i++) k_tmp[i] += lw->k_bias[i];
            if (lw->v_bias) for (int i = 0; i < kv_dim; i++) v_tmp[i] += lw->v_bias[i];

            // RoPE on Q
            for (int i = 0; i < dim; i += 2) {
                int fi = (i / 2) % half_head;
                float v0 = s->q[i], v1 = s->q[i + 1];
                s->q[i]     = v0 * rope_cos[fi] - v1 * rope_sin[fi];
                s->q[i + 1] = v0 * rope_sin[fi] + v1 * rope_cos[fi];
            }

            // RoPE on K temp buffer
            for (int i = 0; i < kv_dim; i += 2) {
                int fi = (i / 2) % half_head;
                float v0 = k_tmp[i], v1 = k_tmp[i + 1];
                k_tmp[i]     = v0 * rope_cos[fi] - v1 * rope_sin[fi];
                k_tmp[i + 1] = v0 * rope_sin[fi] + v1 * rope_cos[fi];
            }

            // Convert F32 -> F16 into cache
            uint16_t *kc = (uint16_t *)s->key_cache   + loff + (size_t)cache_pos * kv_dim;
            uint16_t *vc = (uint16_t *)s->value_cache + loff + (size_t)cache_pos * kv_dim;
#ifdef __ARM_NEON
            for (int i = 0; i < kv_dim; i += 4) {
                float32x4_t kv4 = vld1q_f32(k_tmp + i);
                float16x4_t kh4 = vcvt_f16_f32(kv4);
                vst1_u16(kc + i, vreinterpret_u16_f16(kh4));
                float32x4_t vv4 = vld1q_f32(v_tmp + i);
                float16x4_t vh4 = vcvt_f16_f32(vv4);
                vst1_u16(vc + i, vreinterpret_u16_f16(vh4));
            }
#elif defined(__AVX2__)
            for (int i = 0; i < kv_dim; i += 8) {
                _mm_storeu_si128((__m128i *)(kc + i), _mm256_cvtps_ph(_mm256_loadu_ps(k_tmp + i), _MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128((__m128i *)(vc + i), _mm256_cvtps_ph(_mm256_loadu_ps(v_tmp + i), _MM_FROUND_TO_NEAREST_INT));
            }
#else
            for (int i = 0; i < kv_dim; i++) {
                kc[i] = bn_fp32_to_fp16(k_tmp[i]);
                vc[i] = bn_fp32_to_fp16(v_tmp[i]);
            }
#endif
        } else {
            // F32 KV cache: matvec directly into cache, RoPE in-place
            BnMatvecTask qkv[3] = {
                { s->q,            &lw->wq },
                { key_cache_row,   &lw->wk },
                { value_cache_row, &lw->wv },
            };
            bn_quant_matvec_batch(qkv, 3, s->xb, s->x_q, m->pool);

            // Add attention biases (Qwen2)
            if (lw->q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->q_bias[i];
            if (lw->k_bias) for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->k_bias[i];
            if (lw->v_bias) for (int i = 0; i < kv_dim; i++) value_cache_row[i] += lw->v_bias[i];

            // RoPE using precomputed cos/sin (no trig calls here)
            for (int i = 0; i < dim; i += 2) {
                int fi = (i / 2) % half_head;
                float v0 = s->q[i], v1 = s->q[i + 1];
                s->q[i]     = v0 * rope_cos[fi] - v1 * rope_sin[fi];
                s->q[i + 1] = v0 * rope_sin[fi] + v1 * rope_cos[fi];
            }
            for (int i = 0; i < kv_dim; i += 2) {
                int fi = (i / 2) % half_head;
                float v0 = key_cache_row[i], v1 = key_cache_row[i + 1];
                key_cache_row[i]     = v0 * rope_cos[fi] - v1 * rope_sin[fi];
                key_cache_row[i + 1] = v0 * rope_sin[fi] + v1 * rope_cos[fi];
            }
        }

        // GQA attention
        {
            int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
            BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_dim, c->seq_len };
#ifdef __ARM_NEON
            bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_neon_range : bn_transformer_gqa_neon_range;
#elif defined(__AVX2__)
            bn_tp_fn attn_fn = bn_transformer_gqa_avx2_range;
#elif defined(__wasm_simd128__)
            bn_tp_fn attn_fn = bn_transformer_gqa_wasm_range;
#else
            bn_tp_fn attn_fn = bn_transformer_gqa_scalar_range;
#endif
            BnTPTask gqa = { attn_fn, &gctx, c->n_heads };
            bn_tp_dispatch(m->pool, &gqa, 1);
        }

        // Attention sub-norm + wo projection
        if (lw->attn_sub_norm)
            rmsnorm(s->xb, s->xb, lw->attn_sub_norm, dim, c->norm_eps);

        {
            BnMatvecTask wo[1] = {{ s->xb2, &lw->wo }};
            bn_quant_matvec_batch(wo, 1, s->xb, s->x_q, m->pool);
        }

        // Residual connection
#ifdef __ARM_NEON
        for (int i = 0; i < dim; i += 4)
            vst1q_f32(s->x + i, vaddq_f32(vld1q_f32(s->x + i), vld1q_f32(s->xb2 + i)));
#elif defined(__AVX2__)
        for (int i = 0; i < dim; i += 8)
            _mm256_storeu_ps(s->x + i, _mm256_add_ps(_mm256_loadu_ps(s->x + i), _mm256_loadu_ps(s->xb2 + i)));
#elif defined(__wasm_simd128__)
        for (int i = 0; i < dim; i += 4)
            wasm_v128_store(s->x + i, wasm_f32x4_add(wasm_v128_load(s->x + i), wasm_v128_load(s->xb2 + i)));
#else
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
#endif

        // ---- FFN block ----

        rmsnorm(s->xb, s->x, lw->ffn_norm, dim, c->norm_eps);

        if (c->has_ffn_gate) {
            // Gate + Up in one dispatch
            {
                BnMatvecTask ffn[2] = {
                    { s->hb,  &lw->ffn_gate },
                    { s->hb2, &lw->ffn_up   },
                };
                bn_quant_matvec_batch(ffn, 2, s->xb, s->x_q, m->pool);
            }

            // Activation
            if (c->act_type == 1) {
#ifdef __ARM_NEON
                float32x4_t zero = vdupq_n_f32(0);
                for (int i = 0; i < hidden_dim; i += 4) {
                    float32x4_t g = vmaxq_f32(vld1q_f32(s->hb + i), zero);
                    vst1q_f32(s->hb + i, vmulq_f32(vmulq_f32(g, g), vld1q_f32(s->hb2 + i)));
                }
#elif defined(__AVX2__)
                __m256 zero = _mm256_setzero_ps();
                for (int i = 0; i < hidden_dim; i += 8) {
                    __m256 g = _mm256_max_ps(_mm256_loadu_ps(s->hb + i), zero);
                    _mm256_storeu_ps(s->hb + i, _mm256_mul_ps(_mm256_mul_ps(g, g), _mm256_loadu_ps(s->hb2 + i)));
                }
#elif defined(__wasm_simd128__)
                v128_t zero = wasm_f32x4_splat(0);
                for (int i = 0; i < hidden_dim; i += 4) {
                    v128_t g = wasm_f32x4_max(wasm_v128_load(s->hb + i), zero);
                    wasm_v128_store(s->hb + i, wasm_f32x4_mul(wasm_f32x4_mul(g, g), wasm_v128_load(s->hb2 + i)));
                }
#else
                for (int i = 0; i < hidden_dim; i++) {
                    float g = s->hb[i] > 0 ? s->hb[i] : 0;
                    s->hb[i] = g * g * s->hb2[i];
                }
#endif
            } else {
                for (int i = 0; i < hidden_dim; i++) {
                    float g = s->hb[i];
                    s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
                }
            }
        } else {
            {
                BnMatvecTask ffn[1] = {{ s->hb, &lw->ffn_up }};
                bn_quant_matvec_batch(ffn, 1, s->xb, s->x_q, m->pool);
            }

            if (c->act_type == 1) {
                for (int i = 0; i < hidden_dim; i++) {
                    float v = s->hb[i] > 0 ? s->hb[i] : 0;
                    s->hb[i] = v * v;
                }
            } else {
                for (int i = 0; i < hidden_dim; i++) {
                    float v = s->hb[i];
                    s->hb[i] = v / (1.0f + expf(-v));
                }
            }
        }

        // FFN sub-norm + down projection
        if (lw->ffn_sub_norm)
            rmsnorm(s->hb, s->hb, lw->ffn_sub_norm, hidden_dim, c->norm_eps);

        {
            BnMatvecTask down[1] = {{ s->xb, &lw->ffn_down }};
            bn_quant_matvec_batch(down, 1, s->hb, s->x_q, m->pool);
        }

        // Residual connection
#ifdef __ARM_NEON
        for (int i = 0; i < dim; i += 4)
            vst1q_f32(s->x + i, vaddq_f32(vld1q_f32(s->x + i), vld1q_f32(s->xb + i)));
#elif defined(__AVX2__)
        for (int i = 0; i < dim; i += 8)
            _mm256_storeu_ps(s->x + i, _mm256_add_ps(_mm256_loadu_ps(s->x + i), _mm256_loadu_ps(s->xb + i)));
#elif defined(__wasm_simd128__)
        for (int i = 0; i < dim; i += 4)
            wasm_v128_store(s->x + i, wasm_f32x4_add(wasm_v128_load(s->x + i), wasm_v128_load(s->xb + i)));
#else
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
#endif

        if (l == 0 && pos == 0) {
            char v0[16], v1[16], v2[16], v3[16];
            snprintf(v0, sizeof(v0), "%.6f", s->x[0]);
            snprintf(v1, sizeof(v1), "%.6f", s->x[1]);
            snprintf(v2, sizeof(v2), "%.6f", s->x[2]);
            snprintf(v3, sizeof(v3), "%.6f", s->x[3]);
            SH_LOG_DEBUG("Layer 0 pos 0", "x0", v0, "x1", v1, "x2", v2, "x3", v3);
        }
#ifdef DEBUG
        if (pos == 0 && (l == 0 || l == c->n_layers - 1)) {
            fprintf(stderr, "DBG layer=%d x[0..3]= %.6f %.6f %.6f %.6f\n",
                    l, s->x[0], s->x[1], s->x[2], s->x[3]);
        }
#endif
    }

    return 0;
}

// Final RMSNorm + logits computation. Reads s->x, writes s->logits.
// Returns s->logits.
static float *forward_logits(BnModel *m) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &m->state;
    int dim = c->dim;

    if (dim > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("Model dim too large for stack VLAs");
        return NULL;
    }

    // Final RMSNorm
    rmsnorm(s->x, s->x, w->output_norm, dim, c->norm_eps);

    // Untied output weight: logits = output_weight @ x
    if (w->output_weight.data && w->output_weight.type == BN_GGUF_TENSOR_F16) {
        int n_rows = w->output_weight.rows;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (w->emb_out_i8) {
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { bn_transformer_logits_i8_neon_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
        } else
#elif defined(__AVX2__)
        if (w->emb_out_i8) {
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { bn_transformer_logits_i8_avx2_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
        } else
#endif
        {
            const uint16_t *emb = (const uint16_t *)w->output_weight.data;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            uint16_t x_f16[dim];
            for (int d = 0; d < dim; d += 8) {
                float16x4_t lo = vcvt_f16_f32(vld1q_f32(s->x + d));
                float16x4_t hi = vcvt_f16_f32(vld1q_f32(s->x + d + 4));
                vst1q_u16(x_f16 + d, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
            }
            BnLogitsCtx lctx = { s->logits, (const float *)(void *)x_f16, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_native_neon_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__ARM_NEON)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_neon_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__AVX2__)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_avx2_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__wasm_simd128__)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_wasm_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#else
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_scalar_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#endif
        }
    }
    else if (w->output_weight.data) {
        bn_quant_matvec(s->logits, &w->output_weight, s->x, s->x_q, m->pool);
    }
    // Tied Q4_0/Q8_0/Q6_K embeddings: use quant matvec
    else if (w->emb_type == BN_GGUF_TENSOR_Q4_0 || w->emb_type == BN_GGUF_TENSOR_Q8_0 ||
             w->emb_type == BN_GGUF_TENSOR_Q6_K) {
        BnQWeight tied = { w->token_embedding, w->emb_type, c->vocab_size, dim, 1.0f };
        bn_quant_matvec(s->logits, &tied, s->x, s->x_q, m->pool);
    }
    // Tied F16 embeddings: logits = token_embedding^T @ x
    else if (w->emb_type == BN_GGUF_TENSOR_F16) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (w->emb_out_i8) {
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { bn_transformer_logits_i8_neon_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
        } else
#elif defined(__AVX2__)
        if (w->emb_out_i8) {
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { bn_transformer_logits_i8_avx2_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
        } else
#endif
        {
            const uint16_t *emb = (const uint16_t *)w->token_embedding;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            uint16_t x_f16[dim];
            for (int d = 0; d < dim; d += 8) {
                float16x4_t lo = vcvt_f16_f32(vld1q_f32(s->x + d));
                float16x4_t hi = vcvt_f16_f32(vld1q_f32(s->x + d + 4));
                vst1q_u16(x_f16 + d, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
            }
            BnLogitsCtx lctx = { s->logits, (const float *)(void *)x_f16, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_native_neon_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__ARM_NEON)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_neon_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__AVX2__)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_avx2_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__wasm_simd128__)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_wasm_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#else
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_scalar_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#endif
        }
    } else {
        // F32 embeddings
        const float *emb = (const float *)w->token_embedding;
        BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
        BnTPTask logits_task = { bn_transformer_logits_f32_range, &lctx, c->vocab_size };
        bn_tp_dispatch(m->pool, &logits_task, 1);
    }

    return s->logits;
}

float *bn_transformer_forward(BnModel *m, int token, int pos) {
    if (forward_layers(m, token, pos) != 0) return NULL;
    return forward_logits(m);
}

float *bn_transformer_prefill(BnModel *m, const int *tokens, int n_tokens, int pos0) {
    if (n_tokens <= 0) return NULL;
    for (int i = 0; i < n_tokens; i++) {
        if (forward_layers(m, tokens[i], pos0 + i) != 0) return NULL;
    }
    return forward_logits(m);
}
