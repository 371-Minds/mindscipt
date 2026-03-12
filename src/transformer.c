#include "transformer.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef __ARM_NEON
#include <arm_neon.h>

static inline float neon_hsum_f32(float32x4_t v) {
    float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}
#endif

// --- Helper functions ---

static void rmsnorm(float *out, const float *x, const float *w, int size, float eps) {
#ifdef __ARM_NEON
    float32x4_t sum_sq = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        sum_sq = vmlaq_f32(sum_sq, xv, xv);
    }
    float ss = neon_hsum_f32(sum_sq);
    for (; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    float32x4_t ss_v = vdupq_n_f32(ss);
    for (i = 0; i + 3 < size; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(xv, ss_v), wv));
    }
    for (; i < size; i++) out[i] = x[i] * ss * w[i];
#else
    float ss = 0.0f;
    #ifdef _OPENMP
    #pragma omp simd reduction(+:ss)
    #endif
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    #ifdef _OPENMP
    #pragma omp simd
    #endif
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
#endif
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    #ifdef _OPENMP
    #pragma omp simd reduction(max:max_val)
    #endif
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    #ifdef _OPENMP
    #pragma omp simd
    #endif
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void rope(float *vec, int dim, int head_size, int pos, float theta) {
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(theta, (float)head_dim / (float)head_size);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * cos_a - v1 * sin_a;
        vec[i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

// --- Forward pass ---

float *transformer_forward(Model *m, int token, int pos) {
    Config *c = &m->config;
    Weights *w = &m->weights;
    RunState *s = &m->state;
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul;
    int head_size = c->head_size;

    // #9: Validate token bounds
    if (token < 0 || token >= c->vocab_size) {
        fprintf(stderr, "transformer: token %d out of range [0, %d)\n", token, c->vocab_size);
        return NULL;
    }

    // #10: Validate pos bounds to prevent KV-cache OOB write
    if (pos < 0 || pos >= c->seq_len) {
        fprintf(stderr, "transformer: pos %d out of range [0, %d)\n", pos, c->seq_len);
        return NULL;
    }

    // Embed the token
    model_embed_token(m, s->x, token);

    // Process each layer
    for (int l = 0; l < c->n_layers; l++) {
        LayerWeights *lw = &w->layers[l];

        // ---- Attention block ----

        // RMSNorm before attention
        rmsnorm(s->xb, s->x, lw->attn_norm, dim, c->norm_eps);

        // QKV projections (ternary matmul)
        ternary_matvec(s->q, &lw->wq, s->xb);    // q = Wq @ xb
        // KV go directly into cache
        size_t loff = (size_t)l * c->seq_len * kv_dim;
        float *key_cache_row   = s->key_cache   + loff + (size_t)pos * kv_dim;
        float *value_cache_row = s->value_cache + loff + (size_t)pos * kv_dim;

        ternary_matvec(key_cache_row,   &lw->wk, s->xb);  // k = Wk @ xb
        ternary_matvec(value_cache_row, &lw->wv, s->xb);  // v = Wv @ xb

        // RoPE on q and k
        rope(s->q, dim, head_size, pos, c->rope_theta);
        rope(key_cache_row, kv_dim, head_size, pos, c->rope_theta);

        // Grouped Query Attention (GQA)
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int h = 0; h < c->n_heads; h++) {
            float *q_h = s->q + h * head_size;
            float *att = s->att + h * c->seq_len;
            int kv_h = h / kv_mul;  // which KV head this query head attends to

            // Attention scores: q · k for all positions up to pos
            for (int t = 0; t <= pos; t++) {
                float *k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
#ifdef __ARM_NEON
                float32x4_t acc = vdupq_n_f32(0);
                for (int d = 0; d < head_size; d += 4) {
                    acc = vmlaq_f32(acc, vld1q_f32(q_h + d), vld1q_f32(k_t + d));
                }
                att[t] = neon_hsum_f32(acc) / sqrtf((float)head_size);
#else
                float score = 0.0f;
                #ifdef _OPENMP
                #pragma omp simd reduction(+:score)
                #endif
                for (int d = 0; d < head_size; d++) {
                    score += q_h[d] * k_t[d];
                }
                att[t] = score / sqrtf((float)head_size);
#endif
            }

            // Softmax over attention scores
            softmax(att, pos + 1);

            // Weighted sum of values
            float *xb_h = s->xb + h * head_size;
            memset(xb_h, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                float a = att[t];
#ifdef __ARM_NEON
                float32x4_t a_v = vdupq_n_f32(a);
                for (int d = 0; d < head_size; d += 4) {
                    vst1q_f32(xb_h + d, vmlaq_f32(vld1q_f32(xb_h + d), a_v, vld1q_f32(v_t + d)));
                }
#else
                #ifdef _OPENMP
                #pragma omp simd
                #endif
                for (int d = 0; d < head_size; d++) {
                    xb_h[d] += a * v_t[d];
                }
#endif
            }
        }

        // Attention sub-norm (BitNet-specific)
        if (lw->attn_sub_norm) {
            rmsnorm(s->xb, s->xb, lw->attn_sub_norm, dim, c->norm_eps);
        }

        // Output projection: Wo @ xb → xb2
        ternary_matvec(s->xb2, &lw->wo, s->xb);

        // Residual connection
#ifdef __ARM_NEON
        for (int i = 0; i < dim; i += 4)
            vst1q_f32(s->x + i, vaddq_f32(vld1q_f32(s->x + i), vld1q_f32(s->xb2 + i)));
#else
        #ifdef _OPENMP
        #pragma omp simd
        #endif
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
#endif

        // ---- FFN block ----

        // RMSNorm before FFN
        rmsnorm(s->xb, s->x, lw->ffn_norm, dim, c->norm_eps);

        if (c->has_ffn_gate) {
            // SwiGLU / Gated: gate * activation(up)
            ternary_matvec(s->hb,  &lw->ffn_gate, s->xb);  // gate
            ternary_matvec(s->hb2, &lw->ffn_up,   s->xb);  // up

            if (c->act_type == 1) {
                // BitNet b1.58 with ReLU²: relu²(gate) * up
#ifdef __ARM_NEON
                {
                    float32x4_t zero = vdupq_n_f32(0);
                    for (int i = 0; i < hidden_dim; i += 4) {
                        float32x4_t g = vmaxq_f32(vld1q_f32(s->hb + i), zero);
                        vst1q_f32(s->hb + i, vmulq_f32(vmulq_f32(g, g), vld1q_f32(s->hb2 + i)));
                    }
                }
#else
                #ifdef _OPENMP
                #pragma omp simd
                #endif
                for (int i = 0; i < hidden_dim; i++) {
                    float g = s->hb[i] > 0 ? s->hb[i] : 0;  // ReLU
                    s->hb[i] = g * g * s->hb2[i];            // ReLU² * up
                }
#endif
            } else {
                // SiLU (SwiGLU): silu(gate) * up
                #ifdef _OPENMP
                #pragma omp simd
                #endif
                for (int i = 0; i < hidden_dim; i++) {
                    float g = s->hb[i];
                    s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
                }
            }
        } else {
            // No gate: just up + activation
            ternary_matvec(s->hb, &lw->ffn_up, s->xb);
            if (c->act_type == 1) {
                #ifdef _OPENMP
                #pragma omp simd
                #endif
                for (int i = 0; i < hidden_dim; i++) {
                    float v = s->hb[i] > 0 ? s->hb[i] : 0;
                    s->hb[i] = v * v;
                }
            } else {
                #ifdef _OPENMP
                #pragma omp simd
                #endif
                for (int i = 0; i < hidden_dim; i++) {
                    float v = s->hb[i];
                    s->hb[i] = v / (1.0f + expf(-v));
                }
            }
        }

        // FFN sub-norm (BitNet-specific)
        if (lw->ffn_sub_norm) {
            rmsnorm(s->hb, s->hb, lw->ffn_sub_norm, hidden_dim, c->norm_eps);
        }

        // Down projection
        ternary_matvec(s->xb, &lw->ffn_down, s->hb);

        // Residual connection
#ifdef __ARM_NEON
        for (int i = 0; i < dim; i += 4)
            vst1q_f32(s->x + i, vaddq_f32(vld1q_f32(s->x + i), vld1q_f32(s->xb + i)));
#else
        #ifdef _OPENMP
        #pragma omp simd
        #endif
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
#endif

        #ifdef DEBUG
        if (l == 0 && pos == 0) {
            fprintf(stderr, "debug: layer 0 pos 0 x[0..3] = %.6f %.6f %.6f %.6f\n",
                    s->x[0], s->x[1], s->x[2], s->x[3]);
        }
        #endif
    }

    // Final RMSNorm
    rmsnorm(s->x, s->x, w->output_norm, dim, c->norm_eps);

    // Tied embeddings: logits = token_embedding^T @ x
    // Compute logits as dot product of each embedding row with x
    if (m->weights.emb_type == GGUF_TENSOR_F16) {
        const uint16_t *emb = (const uint16_t *)w->token_embedding;
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int v = 0; v < c->vocab_size; v++) {
            const uint16_t *row = emb + (size_t)v * dim;
            float sum = 0.0f;
#ifdef __ARM_NEON
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            for (int d = 0; d < dim; d += 8) {
                float16x8_t f16 = vreinterpretq_f16_u16(vld1q_u16(row + d));
                acc0 = vmlaq_f32(acc0, vcvt_f32_f16(vget_low_f16(f16)),  vld1q_f32(s->x + d));
                acc1 = vmlaq_f32(acc1, vcvt_f32_f16(vget_high_f16(f16)), vld1q_f32(s->x + d + 4));
            }
            sum = neon_hsum_f32(vaddq_f32(acc0, acc1));
#else
            for (int d = 0; d < dim; d++) {
                sum += fp16_to_fp32(row[d]) * s->x[d];
            }
#endif
            s->logits[v] = sum;
        }
    } else {
        // F32 embeddings
        const float *emb = (const float *)w->token_embedding;
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int v = 0; v < c->vocab_size; v++) {
            const float *row = emb + (size_t)v * dim;
            float sum = 0.0f;
            #ifdef _OPENMP
            #pragma omp simd reduction(+:sum)
            #endif
            for (int d = 0; d < dim; d++) {
                sum += row[d] * s->x[d];
            }
            s->logits[v] = sum;
        }
    }

    return s->logits;
}
