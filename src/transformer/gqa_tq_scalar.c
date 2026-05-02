#include "transformer_internal.h"
#include "turboquant.h"

// TurboQuant GQA: attention directly from packed compressed keys/values.
// Uses precomputed QJL signs to avoid redundant per-key projection.

void bn_transformer_gqa_tq_scalar_range(void *ctx, int h_start, int h_end) {
    BnGQATQCtx *g = (BnGQATQCtx *)ctx;
    BnRunState *s = g->s;
    const BnTQState *tq = g->tq;
    int head_size = g->head_size;
    int kv_mul = g->kv_mul;
    int n_kv = g->n_kv;
    int seq_len = g->seq_len;
    int start = g->pos - n_kv + 1; // always >= 0: n_kv = min(pos+1, seq_len)
    int key_bytes = g->key_bytes;
    int val_bytes = g->val_bytes;
    int n_kv_heads = g->n_kv_heads;
    float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);
    if (head_size > BN_MAX_VLA_ELEMS) return;

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        float *att = s->att + h * seq_len;
        int kv_h = h / kv_mul;
        // Reuse the session scratch buffer so the TQ path avoids per-head stack VLAs
        // in the decode hot path and matches the lifetime of other attention scratch data.
        float *q_rot = s->q_rotated + h * head_size;

        // Step 1: Rotate query for this head
        bn_tq_rotate_query_head(tq, kv_h, q_h, q_rot);

        // Step 2: Precompute QJL signs ONCE per head
        uint8_t q_signs[head_size / 8];
        bn_tq_qjl_precompute_head(tq, kv_h, q_rot, q_signs);

        // Step 3: Score all keys using precomputed QJL
        for (int i = 0; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            const uint8_t *pk = g->tq_keys + (size_t)t * n_kv_heads * key_bytes + kv_h * key_bytes;
            att[i] = bn_tq_score_key_precomputed_head(tq, kv_h, q_rot, q_signs, pk) * inv_sqrt_hs;
        }

        // Step 4: Softmax
        bn_transformer_softmax(att, n_kv);

        // Step 5: Fused weighted combine over contiguous ring-buffer segments
        float *xb_h = s->xb + h * head_size;
        if (bn_tq_get_flags(tq) & BN_TQ_FLAG_FUSED_ATTENTION) {
            int first_t = start % seq_len;
            int first_count = n_kv;
            if (first_t + first_count > seq_len)
                first_count = seq_len - first_t;

            const uint8_t *pv = g->tq_values + (size_t)first_t * g->val_stride + (size_t)kv_h * val_bytes;
            bn_tq_attention_combine_head(tq, kv_h, pv, first_count, g->val_stride, att, xb_h, 0);
            if (first_count < n_kv) {
                const uint8_t *pv_wrap = g->tq_values + (size_t)kv_h * val_bytes;
                bn_tq_attention_combine_head(tq, kv_h, pv_wrap, n_kv - first_count,
                                             g->val_stride, att + first_count, xb_h, 1);
            }
        } else {
            memset(xb_h, 0, head_size * sizeof(float));
            for (int i = 0; i < n_kv; i++) {
                int t = (start + i) % seq_len;
                float w = att[i];
                if (w == 0.0f) continue;
                const uint8_t *pv = g->tq_values + (size_t)t * n_kv_heads * val_bytes + kv_h * val_bytes;
                bn_tq_attention_combine_head(tq, kv_h, pv, 1, val_bytes, &w, xb_h, 1);
            }
        }
    }
}

// NEON stub: falls back to scalar for now (NEON optimized version in gqa_tq_neon.c)
#ifndef __ARM_NEON
void bn_transformer_gqa_tq_neon_range(void *ctx, int start, int end) {
    bn_transformer_gqa_tq_scalar_range(ctx, start, end);
}
#endif
