#include "transformer_internal.h"
#include "quant_internal.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdlib.h>

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

// Inline helper: add residual xb (or xb2) into x
static inline void residual_add(float *x, const float *r, int dim) {
#ifdef __ARM_NEON
    for (int i = 0; i < dim; i += 4)
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(r + i)));
#elif defined(__AVX2__)
    for (int i = 0; i < dim; i += 8)
        _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(r + i)));
#elif defined(__wasm_simd128__)
    for (int i = 0; i < dim; i += 4)
        wasm_v128_store(x + i, wasm_f32x4_add(wasm_v128_load(x + i), wasm_v128_load(r + i)));
#else
    for (int i = 0; i < dim; i++) x[i] += r[i];
#endif
}

// Apply per-head RoPE using precomputed cos/sin.
// rope_dims = number of dims per head to rotate (rest pass through).
static inline void apply_rope_heads(float *buf, int n_heads, int head_size,
                                    int rope_dims, const float *rc, const float *rs) {
    for (int h = 0; h < n_heads; h++) {
        float *hd = buf + h * head_size;
        for (int i = 0; i < rope_dims; i += 2) {
            int fi = i / 2;
            float v0 = hd[i], v1 = hd[i + 1];
            hd[i]     = v0 * rc[fi] - v1 * rs[fi];
            hd[i + 1] = v0 * rs[fi] + v1 * rc[fi];
        }
    }
}

// SSM block: Gated DeltaNet recurrence. Reads s->x, writes s->xb (result for residual).
static void forward_ssm_block(BnModel *m, BnLayerWeights *lw, int l) {
    BnConfig *c = &m->config;
    BnRunState *s = &m->state;
    int dim = c->dim;
    int num_k_heads = c->ssm_group_count;           // 16
    int head_k_dim  = c->ssm_state_size;            // 128
    int num_v_heads = c->ssm_time_step_rank;         // 32
    int head_v_dim  = c->ssm_inner_size / num_v_heads; // 128
    int key_dim     = num_k_heads * head_k_dim;     // 2048
    int value_dim   = c->ssm_inner_size;            // 4096
    int qkv_dim     = key_dim * 2 + value_dim;      // 8192
    int kern        = c->ssm_conv_kernel;           // 4
    (void)0; // kv_ratio not needed: GGUF uses tiled V-head order (hk = hv % num_k_heads)

    // SSM layer index (contiguous among SSM layers)
    int ssm_idx = l - (l + 1) / c->full_attn_interval;
    size_t state_per_layer = (size_t)num_v_heads * head_k_dim * head_v_dim;
    float *state = s->ssm_state + (size_t)ssm_idx * state_per_layer;
    size_t conv_per_layer = (size_t)(kern - 1) * qkv_dim;
    float *conv_state = s->ssm_conv_state + (size_t)ssm_idx * conv_per_layer;

    // 1. Norm input
    rmsnorm(s->xb, s->x, lw->attn_norm, dim, c->norm_eps);

#ifdef DEBUG
    if (l == 0) {
        fprintf(stderr, "SSM L%d norm: %.6f %.6f %.6f %.6f\n", l, s->xb[0], s->xb[1], s->xb[2], s->xb[3]);
        // Dump normed vector, embedding, and norm weights for Python verification
        FILE *df = fopen("/tmp/bitnet_debug_norm.bin", "wb");
        if (df) {
            fwrite(s->x, sizeof(float), dim, df);     // embedding
            fwrite(s->xb, sizeof(float), dim, df);    // normed
            fwrite(lw->attn_norm, sizeof(float), dim, df); // norm weights
            fclose(df);
        }
    }
#endif

    // 2. QKV + Z gate projections (both read from s->xb)
    float *qkv = s->hb;   // [hidden_dim] >= qkv_dim
    float *z   = s->hb2;  // [hidden_dim] >= value_dim
    {
        BnMatvecTask tasks[2] = {
            { qkv, &lw->wqkv },
            { z,   &lw->wz   },
        };
        bn_quant_matvec_batch(tasks, 2, s->xb, s->x_q, m->pool);
    }

#ifdef DEBUG
    if (l == 0) {
        fprintf(stderr, "SSM L%d qkv: %.6f %.6f %.6f %.6f | z: %.6f %.6f\n", l, qkv[0], qkv[1], qkv[2], qkv[3], z[0], z[1]);
        // Dump first 4 bytes of Q5K raw data to verify correct GGUF read
        {
            const uint8_t *raw = (const uint8_t *)lw->wqkv.data;
            fprintf(stderr, "Q5K raw hex[0..19]:");
            for (int i = 0; i < 20; i++) fprintf(stderr, " %02x", raw[i]);
            fprintf(stderr, "\n");
        }
        // Validate Q5K NEON vs scalar for first 4 rows of wqkv
        float ref[4];
        BnQ5KCtx ref_ctx = { ref, &lw->wqkv, s->xb };
        bn_quant_q5k_scalar_range(&ref_ctx, 0, 4);
        fprintf(stderr, "Q5K validate: neon=[%.6f %.6f %.6f %.6f] scalar=[%.6f %.6f %.6f %.6f]\n",
            qkv[0], qkv[1], qkv[2], qkv[3], ref[0], ref[1], ref[2], ref[3]);
        float max_diff = 0;
        for (int i = 0; i < 4; i++) {
            float d = fabsf(qkv[i] - ref[i]);
            if (d > max_diff) max_diff = d;
        }
        fprintf(stderr, "Q5K max_diff=%.2e\n", max_diff);
        // Validate Q4K NEON vs scalar for ALL rows of wz (4096 rows)
        {
            int n = lw->wz.rows;
            float *zref_all = (float *)malloc(n * sizeof(float));
            if (zref_all) {
                BnQ4KCtx zref_ctx = { zref_all, &lw->wz, s->xb };
                bn_quant_q4k_scalar_range(&zref_ctx, 0, n);
                float q4k_max = 0;
                int q4k_worst_row = 0;
                for (int i = 0; i < n; i++) {
                    float d = fabsf(z[i] - zref_all[i]);
                    if (d > q4k_max) { q4k_max = d; q4k_worst_row = i; }
                }
                fprintf(stderr, "Q4K full validate (wz, %d rows): max_diff=%.2e at row %d (neon=%.6f scalar=%.6f)\n",
                    n, q4k_max, q4k_worst_row, z[q4k_worst_row], zref_all[q4k_worst_row]);
                free(zref_all);
            }
        }
    }
#endif

    // 3. Causal conv1d (depthwise, kernel=4)
    // Conv state holds (kern-1) previous QKV projections per channel.
    // conv1d weight layout: GGUF dims [kern, qkv_dim] → channel-major: weight(ch,k) = data[ch * kern + k]
    for (int ch = 0; ch < qkv_dim; ch++) {
        float sum = 0;
        for (int k = 0; k < kern - 1; k++)
            sum += conv_state[(size_t)k * qkv_dim + ch] *
                   lw->ssm_conv1d[(size_t)ch * kern + k];
        float cur = qkv[ch];
        sum += cur * lw->ssm_conv1d[(size_t)ch * kern + (kern - 1)];
        // Shift conv_state for this channel
        for (int k = 0; k < kern - 2; k++)
            conv_state[(size_t)k * qkv_dim + ch] =
                conv_state[(size_t)(k + 1) * qkv_dim + ch];
        conv_state[(size_t)(kern - 2) * qkv_dim + ch] = cur;
        qkv[ch] = sum;
    }

    // 4. SiLU activation on conv output
    for (int i = 0; i < qkv_dim; i++) {
        float v = qkv[i];
        qkv[i] = v / (1.0f + expf(-v));
    }

    // 5. Split xBC: [Q(key_dim), K(key_dim), V(value_dim)]
    //    GGUF attn_qkv stores [Q_all, K_all, V_all] (confirmed by llama.cpp conversion)
    float *q_raw = qkv;
    float *k_raw = qkv + key_dim;
    float *v_raw = qkv + 2 * key_dim;

#ifdef DEBUG
    if (l == 0) fprintf(stderr, "SSM L%d post-conv-silu Q: %.6f %.6f K: %.6f %.6f V: %.6f %.6f\n",
        l, q_raw[0], q_raw[1], k_raw[0], k_raw[1], v_raw[0], v_raw[1]);
#endif

    // 6. L2 normalize Q and K per head
    for (int h = 0; h < num_k_heads; h++) {
        float *qh = q_raw + h * head_k_dim;
        float *kh = k_raw + h * head_k_dim;
        float qn = 0, kn = 0;
        for (int d = 0; d < head_k_dim; d++) {
            qn += qh[d] * qh[d];
            kn += kh[d] * kh[d];
        }
        qn = 1.0f / (sqrtf(qn) + 1e-6f);
        kn = 1.0f / (sqrtf(kn) + 1e-6f);
        for (int d = 0; d < head_k_dim; d++) {
            qh[d] *= qn;
            kh[d] *= kn;
        }
    }

    // 7. Alpha (decay) and Beta (update rate) from normalized input
    float alpha_arr[num_v_heads], beta_arr[num_v_heads];
    {
        BnMatvecTask ab[2] = {
            { alpha_arr, &lw->ssm_alpha },
            { beta_arr,  &lw->ssm_beta  },
        };
        bn_quant_matvec_batch(ab, 2, s->xb, s->x_q, m->pool);
    }
    for (int h = 0; h < num_v_heads; h++) {
        // GGUF stores ssm_a = -exp(A_log) (pre-transformed, always negative)
        // decay = exp(softplus(alpha_proj + dt_bias) * ssm_a)
        float dt = alpha_arr[h] + lw->ssm_dt_bias[h];
        float dt_sp = (dt > 20.0f) ? dt : logf(1.0f + expf(dt)); // softplus
        alpha_arr[h] = expf(dt_sp * lw->ssm_a[h]);  // decay ∈ (0, 1]
        // beta: sigmoid
        beta_arr[h] = 1.0f / (1.0f + expf(-beta_arr[h]));
    }

#ifdef DEBUG
    if (l == 0) {
        fprintf(stderr, "SSM L%d alpha: %.6f %.6f %.6f beta: %.6f %.6f %.6f | A_log[0]=%.4f dt_bias[0]=%.4f\n",
            l, alpha_arr[0], alpha_arr[1], alpha_arr[2], beta_arr[0], beta_arr[1], beta_arr[2],
            lw->ssm_a[0], lw->ssm_dt_bias[0]);
        // Dump all alpha/beta for Python verification
        FILE *df2 = fopen("/tmp/bitnet_debug_ab.bin", "wb");
        if (df2) {
            fwrite(alpha_arr, sizeof(float), num_v_heads, df2);  // decay values
            fwrite(beta_arr, sizeof(float), num_v_heads, df2);   // beta values
            fclose(df2);
        }
    }
#endif

    // 8. Delta rule recurrence (per V-head)
    // Scale Q by 1/sqrt(head_k_dim) (matches llama.cpp delta net readout scaling)
    float q_scale = 1.0f / sqrtf((float)head_k_dim);
    for (int i = 0; i < key_dim; i++)
        q_raw[i] *= q_scale;

    float *out = s->xb2;  // [dim] >= value_dim

    for (int hv = 0; hv < num_v_heads; hv++) {
        int hk = hv % num_k_heads;  // tiled order: GGUF reorders V-heads for broadcast
        const float *qh = q_raw + hk * head_k_dim;
        const float *kh = k_raw + hk * head_k_dim;
        float *vh = v_raw + hv * head_v_dim;
        float *S = state + (size_t)hv * head_k_dim * head_v_dim;
        float decay = alpha_arr[hv];
        float beta = beta_arr[hv];

        // Decay state
        for (int i = 0; i < head_k_dim * head_v_dim; i++)
            S[i] *= decay;

        // sk = S @ k  (prediction: what state expects for this key)
        float sk[head_v_dim];
        for (int v = 0; v < head_v_dim; v++) {
            float sum = 0;
            for (int k = 0; k < head_k_dim; k++)
                sum += S[(size_t)k * head_v_dim + v] * kh[k];
            sk[v] = sum;
        }

        // State update: S += k ⊗ (beta * (v - sk))
        for (int k = 0; k < head_k_dim; k++) {
            float kk = kh[k];
            for (int v = 0; v < head_v_dim; v++)
                S[(size_t)k * head_v_dim + v] += kk * beta * (vh[v] - sk[v]);
        }

        // Read output: o = S^T @ q * scale
        float *oh = out + hv * head_v_dim;
        for (int v = 0; v < head_v_dim; v++) {
            float sum = 0;
            for (int k = 0; k < head_k_dim; k++)
                sum += S[(size_t)k * head_v_dim + v] * qh[k];
            oh[v] = sum;
        }
    }

#ifdef DEBUG
    if (l == 0) {
        // Compute L2 norm and max of full recurrence output
        float l2 = 0, mx = 0;
        for (int i = 0; i < value_dim; i++) {
            l2 += out[i] * out[i];
            float a = fabsf(out[i]);
            if (a > mx) mx = a;
        }
        l2 = sqrtf(l2);
        // Also compute L2 norm of state for head 0
        float *S0 = state;
        float s_l2 = 0;
        for (int i = 0; i < head_k_dim * head_v_dim; i++)
            s_l2 += S0[i] * S0[i];
        s_l2 = sqrtf(s_l2);
        fprintf(stderr, "SSM L%d recurrence: out_l2=%.6f out_max=%.6f state_h0_l2=%.6f\n", l, l2, mx, s_l2);
        fprintf(stderr, "SSM L%d out[0..7]: %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n",
            l, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
        // Print V values (first 8)
        float *v0 = qkv + key_dim * 2;
        fprintf(stderr, "SSM L%d V[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
            l, v0[0], v0[1], v0[2], v0[3], v0[4], v0[5], v0[6], v0[7]);
        // Print z gate values (first 8)
        fprintf(stderr, "SSM L%d z[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
            l, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
    }
#endif

    // 9. Per-head RMSNorm + SiLU-gated output with z
    for (int hv = 0; hv < num_v_heads; hv++) {
        float *oh = out + hv * head_v_dim;
        float *zh = z + hv * head_v_dim;
        rmsnorm(oh, oh, lw->ssm_norm, head_v_dim, c->norm_eps);
        for (int d = 0; d < head_v_dim; d++) {
            float g = zh[d];
            oh[d] *= g / (1.0f + expf(-g));  // SiLU(z)
        }
    }

    // 10. Output projection: out[value_dim] → xb[dim]
    {
        BnMatvecTask proj[1] = {{ s->xb, &lw->ssm_out }};
        bn_quant_matvec_batch(proj, 1, out, s->x_q, m->pool);
    }

#ifdef DEBUG
    if (l == 0) {
        fprintf(stderr, "SSM L%d final xb: %.6f %.6f %.6f %.6f\n", l, s->xb[0], s->xb[1], s->xb[2], s->xb[3]);
        // Dump recurrence output (pre-norm) and final projection output
        FILE *df3 = fopen("/tmp/bitnet_debug_ssm_out.bin", "wb");
        if (df3) {
            fwrite(out, sizeof(float), value_dim, df3);   // recurrence output
            fwrite(s->xb, sizeof(float), dim, df3);       // final projected output
            fclose(df3);
        }
    }
#endif
    // (debug zero-SSM removed)
}

// FFN block: shared by both attention and SSM layers.
// Reads s->x, uses s->xb/hb/hb2/x_q as scratch. Adds result to s->x.
static int ffn_call_count = 0;
static void forward_ffn_block(BnModel *m, BnLayerWeights *lw) {
    BnConfig *c = &m->config;
    BnRunState *s = &m->state;
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;

    rmsnorm(s->xb, s->x, lw->ffn_norm, dim, c->norm_eps);

#ifdef DEBUG
    if (ffn_call_count == 0) {
        fprintf(stderr, "FFN L0: x_pre[0..3]= %.6f %.6f %.6f %.6f normed[0..3]= %.6f %.6f %.6f %.6f\n",
            s->x[0], s->x[1], s->x[2], s->x[3],
            s->xb[0], s->xb[1], s->xb[2], s->xb[3]);
    }
#endif

    if (c->has_ffn_gate) {
        {
            BnMatvecTask ffn[2] = {
                { s->hb,  &lw->ffn_gate },
                { s->hb2, &lw->ffn_up   },
            };
            bn_quant_matvec_batch(ffn, 2, s->xb, s->x_q, m->pool);
        }

#ifdef DEBUG
        if (ffn_call_count == 0) {
            fprintf(stderr, "FFN L0: gate[0..3]= %.6f %.6f %.6f %.6f up[0..3]= %.6f %.6f %.6f %.6f\n",
                s->hb[0], s->hb[1], s->hb[2], s->hb[3],
                s->hb2[0], s->hb2[1], s->hb2[2], s->hb2[3]);
            // Validate Q4K for ffn_gate (first 4 rows)
            float gref[4];
            BnQ4KCtx gref_ctx = { gref, &lw->ffn_gate, s->xb };
            bn_quant_q4k_scalar_range(&gref_ctx, 0, 4);
            float q4k_ffn_diff = 0;
            for (int i = 0; i < 4; i++) {
                float d = fabsf(s->hb[i] - gref[i]);
                if (d > q4k_ffn_diff) q4k_ffn_diff = d;
            }
            fprintf(stderr, "FFN L0 Q4K gate validate: scalar=[%.6f %.6f %.6f %.6f] max_diff=%.2e\n",
                gref[0], gref[1], gref[2], gref[3], q4k_ffn_diff);
        }
#endif

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

    if (lw->ffn_sub_norm)
        rmsnorm(s->hb, s->hb, lw->ffn_sub_norm, hidden_dim, c->norm_eps);

    {
        BnMatvecTask down[1] = {{ s->xb, &lw->ffn_down }};
        bn_quant_matvec_batch(down, 1, s->hb, s->x_q, m->pool);
    }

#ifdef DEBUG
    if (ffn_call_count == 0) {
        fprintf(stderr, "FFN L0: down[0..3]= %.6f %.6f %.6f %.6f\n",
            s->xb[0], s->xb[1], s->xb[2], s->xb[3]);
    }
    ffn_call_count++;
#endif

    residual_add(s->x, s->xb, dim);
}

// Embed + all layers (attention + FFN). Populates KV cache at `pos`.
// Leaves final activation in s->x. Returns 0 on success, -1 on error.
static int forward_layers(BnModel *m, int token, int pos) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &m->state;
    int dim = c->dim;
    int kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul;
    int head_size = c->head_size;
    int n_heads = c->n_heads;

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

#ifdef DEBUG
    if (pos <= 1) {
        float emb_l2 = 0;
        for (int d = 0; d < dim; d++) emb_l2 += s->x[d] * s->x[d];
        emb_l2 = sqrtf(emb_l2);
        fprintf(stderr, "DBG embed tok=%d pos=%d x[0..3]= %.6f %.6f %.6f %.6f |x|=%.3f\n",
            token, pos, s->x[0], s->x[1], s->x[2], s->x[3], emb_l2);
    }
    // Diagnostic: test embedding→logits directly (skip all layers)
    if (pos == 0) {
        float x_copy[dim];
        memcpy(x_copy, s->x, dim * sizeof(float));
        rmsnorm(s->x, s->x, w->output_norm, dim, c->norm_eps);
        // Quick logits for first 10 vocab entries
        fprintf(stderr, "DBG embed→logits (skip layers): x_normed[0..3]= %.6f %.6f %.6f %.6f\n",
            s->x[0], s->x[1], s->x[2], s->x[3]);
        // Restore
        memcpy(s->x, x_copy, dim * sizeof(float));
    }
#endif

    // Precompute RoPE cos/sin for this position
    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : head_size;
    int half_rope = rope_dims / 2;
    float rope_cos[half_rope], rope_sin[half_rope];
    for (int i = 0; i < half_rope; i++) {
        float angle = pos * s->rope_freq[i];
        rope_cos[i] = cosf(angle);
        rope_sin[i] = sinf(angle);
    }

    // Process each layer
    int cache_pos = pos % c->seq_len;
    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];

        int is_attn = (c->full_attn_interval == 0) ||
                      ((l + 1) % c->full_attn_interval == 0);

        if (is_attn) {
            // ---- Attention block ----

            // KV cache offset: contiguous among attention layers only
            int attn_idx = (c->full_attn_interval > 0)
                ? (l + 1) / c->full_attn_interval - 1 : l;
            size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;

            // Gated Q: Q weight produces 2x dim (interleaved [Q,gate] per head)
            int q_gated = lw->wq.data && (lw->wq.rows > dim);

            rmsnorm(s->xb, s->x, lw->attn_norm, dim, c->norm_eps);

#ifdef DEBUG
            if (pos == 0 && l == 3)
                fprintf(stderr, "ATN L%d q_gated=%d wq.rows=%d wq.cols=%d dim=%d head_size=%d n_heads=%d\n",
                    l, q_gated, lw->wq.rows, lw->wq.cols, dim, head_size, n_heads);
#endif

            if (q_gated) {
                // --- Gated Q path (Qwen3.5 attention) ---
                // Q matvec to hb (>= 2*dim), K/V to hb2 (temp) or cache
                // q_full layout: [Q_h0(hs), gate_h0(hs), Q_h1(hs), gate_h1(hs), ...]
                float *q_full = s->hb;  // [2*dim]

                if (c->kv_f16) {
                    float *k_tmp = s->hb2;
                    float *v_tmp = s->hb2 + kv_dim;
                    BnMatvecTask q_task[1] = {{ q_full, &lw->wq }};
                    bn_quant_matvec_batch(q_task, 1, s->xb, s->x_q, m->pool);
                    BnMatvecTask kv[2] = {
                        { k_tmp, &lw->wk },
                        { v_tmp, &lw->wv },
                    };
                    bn_quant_matvec_batch(kv, 2, s->xb, s->x_q, m->pool);

                    // De-interleave Q from per-head [Q_h, gate_h] layout
                    for (int h = 0; h < n_heads; h++)
                        memcpy(s->q + h * head_size,
                               q_full + h * 2 * head_size,
                               head_size * sizeof(float));

                    // Q/K RMSNorm
                    if (lw->q_norm)
                        for (int h = 0; h < n_heads; h++)
                            rmsnorm(s->q + h*head_size, s->q + h*head_size,
                                    lw->q_norm, head_size, c->norm_eps);
                    if (lw->k_norm)
                        for (int h = 0; h < c->n_kv_heads; h++)
                            rmsnorm(k_tmp + h*head_size, k_tmp + h*head_size,
                                    lw->k_norm, head_size, c->norm_eps);

                    // Partial RoPE
                    apply_rope_heads(s->q, n_heads, head_size,
                                     rope_dims, rope_cos, rope_sin);
                    apply_rope_heads(k_tmp, c->n_kv_heads, head_size,
                                     rope_dims, rope_cos, rope_sin);

                    // F32→F16 into cache
                    uint16_t *kc = (uint16_t *)s->key_cache   + loff + (size_t)cache_pos * kv_dim;
                    uint16_t *vc = (uint16_t *)s->value_cache + loff + (size_t)cache_pos * kv_dim;
#ifdef __ARM_NEON
                    for (int i = 0; i < kv_dim; i += 4) {
                        vst1_u16(kc + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(k_tmp + i))));
                        vst1_u16(vc + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(v_tmp + i))));
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
                    // F32 cache
                    float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_dim;
                    float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_dim;
                    BnMatvecTask q_task[1] = {{ q_full, &lw->wq }};
                    bn_quant_matvec_batch(q_task, 1, s->xb, s->x_q, m->pool);
                    BnMatvecTask kv[2] = {
                        { key_cache_row,   &lw->wk },
                        { value_cache_row, &lw->wv },
                    };
                    bn_quant_matvec_batch(kv, 2, s->xb, s->x_q, m->pool);

                    // De-interleave Q from per-head [Q_h, gate_h] layout
                    for (int h = 0; h < n_heads; h++)
                        memcpy(s->q + h * head_size,
                               q_full + h * 2 * head_size,
                               head_size * sizeof(float));

#ifdef DEBUG
                    if (l == 3 && pos == 0) {
                        fprintf(stderr, "ATN L3 Q_h0[0..3]= %.6f %.6f %.6f %.6f Q_h1[0..3]= %.6f %.6f %.6f %.6f\n",
                            q_full[0], q_full[1], q_full[2], q_full[3],
                            q_full[head_size], q_full[head_size+1], q_full[head_size+2], q_full[head_size+3]);
                        fprintf(stderr, "ATN L3 gate_h0[0..3]= %.6f %.6f %.6f %.6f (at q_full+dim=%d)\n",
                            q_full[dim], q_full[dim+1], q_full[dim+2], q_full[dim+3], dim);
                        fprintf(stderr, "ATN L3 K[0..3]= %.6f %.6f %.6f %.6f V[0..3]= %.6f %.6f %.6f %.6f\n",
                            key_cache_row[0], key_cache_row[1], key_cache_row[2], key_cache_row[3],
                            value_cache_row[0], value_cache_row[1], value_cache_row[2], value_cache_row[3]);
                        // Dump all ATN L3 intermediates for Python verification
                        FILE *df = fopen("/tmp/bitnet_debug_attn3.bin", "wb");
                        if (df) {
                            fwrite(s->xb, sizeof(float), dim, df);        // normed input
                            fwrite(q_full, sizeof(float), 2 * dim, df);   // Q + gate interleaved
                            fwrite(key_cache_row, sizeof(float), kv_dim, df);  // K
                            fwrite(value_cache_row, sizeof(float), kv_dim, df); // V
                            fclose(df);
                        }
                    }
#endif

                    // Q/K RMSNorm
                    if (lw->q_norm)
                        for (int h = 0; h < n_heads; h++)
                            rmsnorm(s->q + h*head_size, s->q + h*head_size,
                                    lw->q_norm, head_size, c->norm_eps);
                    if (lw->k_norm)
                        for (int h = 0; h < c->n_kv_heads; h++)
                            rmsnorm(key_cache_row + h*head_size,
                                    key_cache_row + h*head_size,
                                    lw->k_norm, head_size, c->norm_eps);

#ifdef DEBUG
                    if (l == 3 && pos == 0)
                        fprintf(stderr, "ATN L3 post-norm Q[0..3]= %.6f %.6f %.6f %.6f K[0..3]= %.6f %.6f %.6f %.6f\n",
                            s->q[0], s->q[1], s->q[2], s->q[3],
                            key_cache_row[0], key_cache_row[1], key_cache_row[2], key_cache_row[3]);
#endif

                    // Partial RoPE
                    apply_rope_heads(s->q, n_heads, head_size,
                                     rope_dims, rope_cos, rope_sin);
                    apply_rope_heads(key_cache_row, c->n_kv_heads, head_size,
                                     rope_dims, rope_cos, rope_sin);

#ifdef DEBUG
                    if (l == 3 && pos == 0)
                        fprintf(stderr, "ATN L3 post-rope Q[0..3]= %.6f %.6f %.6f %.6f K[0..3]= %.6f %.6f %.6f %.6f\n",
                            s->q[0], s->q[1], s->q[2], s->q[3],
                            key_cache_row[0], key_cache_row[1], key_cache_row[2], key_cache_row[3]);
#endif
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
                    BnTPTask gqa = { attn_fn, &gctx, n_heads };
                    bn_tp_dispatch(m->pool, &gqa, 1);
                }

                // Sigmoid gate: xb *= sigmoid(gate)
                // Gate is at q_full[h*2*hs + hs] for each head (interleaved layout)
#ifdef DEBUG
                if (l == 3 && pos == 0) {
                    fprintf(stderr, "ATN L3 xb_h0[0..3] (pre-gate)= %.6f %.6f %.6f %.6f\n",
                        s->xb[0], s->xb[1], s->xb[2], s->xb[3]);
                }
#endif
                for (int h = 0; h < n_heads; h++) {
                    float *gate_h = q_full + h * 2 * head_size + head_size;
                    float *xb_h = s->xb + h * head_size;
                    for (int d = 0; d < head_size; d++)
                        xb_h[d] *= 1.0f / (1.0f + expf(-gate_h[d]));
                }

                // wo projection + residual
                if (lw->attn_sub_norm)
                    rmsnorm(s->xb, s->xb, lw->attn_sub_norm, dim, c->norm_eps);
#ifdef DEBUG
                if (l == 3 && pos == 0) {
                    fprintf(stderr, "ATN L3 post-gate xb[0..3]= %.6f %.6f %.6f %.6f\n",
                        s->xb[0], s->xb[1], s->xb[2], s->xb[3]);
                    // Dump post-gate xb for verification
                    FILE *df = fopen("/tmp/bitnet_debug_attn3_xb.bin", "wb");
                    if (df) { fwrite(s->xb, sizeof(float), dim, df); fclose(df); }
                }
#endif
                {
                    BnMatvecTask wo[1] = {{ s->xb2, &lw->wo }};
                    bn_quant_matvec_batch(wo, 1, s->xb, s->x_q, m->pool);
                }
#ifdef DEBUG
                if (l == 3 && pos == 0) {
                    fprintf(stderr, "ATN L3 wo_out[0..3]= %.6f %.6f %.6f %.6f\n",
                        s->xb2[0], s->xb2[1], s->xb2[2], s->xb2[3]);
                    FILE *df = fopen("/tmp/bitnet_debug_attn3_wo.bin", "wb");
                    if (df) { fwrite(s->xb2, sizeof(float), dim, df); fclose(df); }
                }
#endif
                residual_add(s->x, s->xb2, dim);

            } else {
                // --- Classic attention path (existing) ---
                float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_dim;
                float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_dim;

                if (c->kv_f16) {
                    float *k_tmp = s->hb, *v_tmp = s->hb2;
                    BnMatvecTask qkv[3] = {
                        { s->q,  &lw->wq },
                        { k_tmp, &lw->wk },
                        { v_tmp, &lw->wv },
                    };
                    bn_quant_matvec_batch(qkv, 3, s->xb, s->x_q, m->pool);

                    if (lw->q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->q_bias[i];
                    if (lw->k_bias) for (int i = 0; i < kv_dim; i++) k_tmp[i] += lw->k_bias[i];
                    if (lw->v_bias) for (int i = 0; i < kv_dim; i++) v_tmp[i] += lw->v_bias[i];

                    apply_rope_heads(s->q, n_heads, head_size,
                                     rope_dims, rope_cos, rope_sin);
                    apply_rope_heads(k_tmp, c->n_kv_heads, head_size,
                                     rope_dims, rope_cos, rope_sin);

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
                    BnMatvecTask qkv[3] = {
                        { s->q,            &lw->wq },
                        { key_cache_row,   &lw->wk },
                        { value_cache_row, &lw->wv },
                    };
                    bn_quant_matvec_batch(qkv, 3, s->xb, s->x_q, m->pool);

                    if (lw->q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->q_bias[i];
                    if (lw->k_bias) for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->k_bias[i];
                    if (lw->v_bias) for (int i = 0; i < kv_dim; i++) value_cache_row[i] += lw->v_bias[i];

                    apply_rope_heads(s->q, n_heads, head_size,
                                     rope_dims, rope_cos, rope_sin);
                    apply_rope_heads(key_cache_row, c->n_kv_heads, head_size,
                                     rope_dims, rope_cos, rope_sin);
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
                    BnTPTask gqa = { attn_fn, &gctx, n_heads };
                    bn_tp_dispatch(m->pool, &gqa, 1);
                }

                // Attention sub-norm + wo projection + residual
                if (lw->attn_sub_norm)
                    rmsnorm(s->xb, s->xb, lw->attn_sub_norm, dim, c->norm_eps);
                {
                    BnMatvecTask wo[1] = {{ s->xb2, &lw->wo }};
                    bn_quant_matvec_batch(wo, 1, s->xb, s->x_q, m->pool);
                }
                residual_add(s->x, s->xb2, dim);
            }

        } else {
            // ---- SSM block ----
            forward_ssm_block(m, lw, l);
            residual_add(s->x, s->xb, dim);
        }

        // ---- FFN block ---- (shared by both layer types)
        forward_ffn_block(m, lw);

        if (l == 0 && pos == 0) {
            char v0[16], v1[16], v2[16], v3[16];
            snprintf(v0, sizeof(v0), "%.6f", s->x[0]);
            snprintf(v1, sizeof(v1), "%.6f", s->x[1]);
            snprintf(v2, sizeof(v2), "%.6f", s->x[2]);
            snprintf(v3, sizeof(v3), "%.6f", s->x[3]);
            SH_LOG_DEBUG("Layer 0 pos 0", "x0", v0, "x1", v1, "x2", v2, "x3", v3);
        }
#ifdef DEBUG
        if (pos == 0) {
            float l2 = 0;
            for (int d = 0; d < dim; d++) l2 += s->x[d] * s->x[d];
            l2 = sqrtf(l2);
            fprintf(stderr, "DBG layer=%d %s x[0..3]= %.6f %.6f %.6f %.6f |x|=%.3f\n",
                    l, is_attn ? "ATN" : "SSM",
                    s->x[0], s->x[1], s->x[2], s->x[3], l2);
            // Dump hidden state at key layers for Python verification
            if (l == 2) {  // Before layer 3 (first ATN) = after layer 2 (SSM)
                FILE *df = fopen("/tmp/bitnet_debug_layer2.bin", "wb");
                if (df) { fwrite(s->x, sizeof(float), dim, df); fclose(df); }
            }
            if (l == 3) {  // After layer 3 (first ATN + FFN)
                FILE *df = fopen("/tmp/bitnet_debug_layer3.bin", "wb");
                if (df) { fwrite(s->x, sizeof(float), dim, df); fclose(df); }
            }
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
#ifdef DEBUG
    {
        // Dump pre-norm hidden state for Python verification
        FILE *df = fopen("/tmp/bitnet_debug_hidden.bin", "wb");
        if (df) {
            fwrite(s->x, sizeof(float), dim, df);
            fwrite(w->output_norm, sizeof(float), dim, df);
            fclose(df);
        }
        float xl2 = 0;
        for (int d = 0; d < dim; d++) xl2 += s->x[d] * s->x[d];
        fprintf(stderr, "LOGITS pre-norm |x|=%.3f x[0..7]= %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
            sqrtf(xl2), s->x[0], s->x[1], s->x[2], s->x[3], s->x[4], s->x[5], s->x[6], s->x[7]);
    }
#endif
    rmsnorm(s->x, s->x, w->output_norm, dim, c->norm_eps);
#ifdef DEBUG
    {
        // Dump post-norm hidden state
        FILE *df = fopen("/tmp/bitnet_debug_hidden_normed.bin", "wb");
        if (df) {
            fwrite(s->x, sizeof(float), dim, df);
            fclose(df);
        }
        float xl2 = 0;
        for (int d = 0; d < dim; d++) xl2 += s->x[d] * s->x[d];
        fprintf(stderr, "LOGITS post-norm |x|=%.3f x[0..7]= %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
            sqrtf(xl2), s->x[0], s->x[1], s->x[2], s->x[3], s->x[4], s->x[5], s->x[6], s->x[7]);
    }
#endif

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
        BnQWeight tied = { w->token_embedding, w->emb_type, c->vocab_size, dim, 1.0f, NULL, NULL };
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


#ifdef DEBUG
    // Q6K NEON vs scalar validation for output weight
    if (w->output_weight.data && w->output_weight.type == BN_GGUF_TENSOR_Q6_K) {
        int test_toks[] = {169479, 69759, 11471, 0, 1, 2};
        int n_test = sizeof(test_toks) / sizeof(test_toks[0]);
        float *scalar_buf = (float *)calloc(c->vocab_size, sizeof(float));
        if (scalar_buf) {
            BnQ6KCtx ref_ctx = { scalar_buf, &w->output_weight, s->x };
            for (int t = 0; t < n_test; t++) {
                int tok = test_toks[t];
                bn_quant_q6k_scalar_range(&ref_ctx, tok, tok + 1);
                float diff = fabsf(scalar_buf[tok] - s->logits[tok]);
                fprintf(stderr, "Q6K validate token %d: scalar=%.4f neon=%.4f diff=%.2e\n",
                    tok, scalar_buf[tok], s->logits[tok], diff);
            }
            free(scalar_buf);
        }
    }
    // Print top-5 logits
    {
        int top5[5] = {0};
        for (int i = 1; i < c->vocab_size; i++) {
            for (int j = 0; j < 5; j++) {
                if (s->logits[i] > s->logits[top5[j]]) {
                    for (int k = 4; k > j; k--) top5[k] = top5[k-1];
                    top5[j] = i;
                    break;
                }
            }
        }
        fprintf(stderr, "LOGITS top5: [%d]=%.4f [%d]=%.4f [%d]=%.4f [%d]=%.4f [%d]=%.4f\n",
            top5[0], s->logits[top5[0]], top5[1], s->logits[top5[1]],
            top5[2], s->logits[top5[2]], top5[3], s->logits[top5[3]],
            top5[4], s->logits[top5[4]]);
        // Print x before final norm
        fprintf(stderr, "LOGITS x_prenorm l2=%.3f x[0..3]=%.6f %.6f %.6f %.6f\n",
            0.0f, s->logits[0], s->logits[1], s->logits[2], s->logits[3]);
    }
#endif

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
