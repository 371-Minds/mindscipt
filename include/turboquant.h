#ifndef BN_TURBOQUANT_H
#define BN_TURBOQUANT_H

#include <stdint.h>
#include <stddef.h>

#ifndef BN_MAX_VLA_ELEMS
#define BN_MAX_VLA_ELEMS 8192
#endif

// TurboQuant KV cache compression (ICLR 2026, arXiv 2504.19874).
// Compresses KV cache to 3-bit via random rotation + Lloyd-Max scalar
// quantization + 1-bit QJL residual correction for keys.

enum {
    BN_TQ_FORMAT_VERSION = 2,
    BN_TQ_FLAG_ADAPTIVE = 1u << 0,
    BN_TQ_FLAG_FUSED_ATTENTION = 1u << 1,
};

typedef enum {
    BN_TQ_STRATEGY_BASELINE = 0,
    BN_TQ_STRATEGY_CALIBRATED = 1,
    BN_TQ_STRATEGY_OUTLIER = 2,
    BN_TQ_STRATEGY_CONSERVATIVE = 3,
    BN_TQ_STRATEGY_COUNT = 4
} BnTQStrategy;

typedef struct {
    int n_centroids;    // 2^bits
    float *centroids;   // [2^b] Lloyd-Max centroids (scaled by 1/sqrt(d))
    float *boundaries;  // [2^b - 1] decision boundaries
} BnTQTables;

typedef struct {
    int samples;
    float variance_ratio_sum;
    float outlier_ratio_sum;
    float residual_norm_sum;
    float score_error_sum;
    float top1_agreement_sum;
} BnTQHeadCalibration;

typedef struct {
    uint8_t strategy;
    float clip_threshold;
    float qjl_weight;
} BnTQHeadRuntime;

typedef struct {
    int head_dim;       // d (e.g. 128)
    int bits;           // quantization bits (2, 3, or 4)
    uint8_t format_version;
    uint32_t flags;
    BnTQTables tables;
    // RHT state: y = (1/sqrt(d)) * H * D * x
    float *rht_signs;   // [d] random ±1 diagonal for rotation
    float *qjl_signs;   // [d] random ±1 diagonal for QJL projection
    float rht_scale;    // 1/sqrt(d) normalization
    int n_heads;        // number of configured KV heads
    BnTQHeadCalibration *head_calibration;
    BnTQHeadRuntime *head_runtime;
} BnTQState;

// Initialize TurboQuant state for a given head dimension and bit width.
// seed: deterministic seed for rotation + QJL matrices.
// Returns 0 on success, -1 on failure.
int  bn_tq_init(BnTQState *state, int head_dim, int bits, uint64_t seed);

// Free all allocated memory.
void bn_tq_free(BnTQState *state);

// Configure the number of KV heads tracked by the runtime metadata.
// When not called explicitly, head 0 is used as the default runtime.
int bn_tq_configure_heads(BnTQState *state, int n_heads);

// Enable or disable runtime flags such as adaptive routing / fused attention.
void bn_tq_set_flags(BnTQState *state, uint32_t flags);
uint32_t bn_tq_get_flags(const BnTQState *state);
uint8_t bn_tq_format_version(const BnTQState *state);

// Query or override the runtime strategy for a specific KV head.
BnTQStrategy bn_tq_get_head_strategy(const BnTQState *state, int head_idx);
int bn_tq_set_head_strategy(BnTQState *state, int head_idx, BnTQStrategy strategy);

// Deterministic synthetic calibration for one KV head. The samples use flat
// arrays with the provided stride (0 = head_dim).
int bn_tq_calibrate_head(BnTQState *state, int head_idx,
                         const float *queries, const float *keys,
                         const float *values, int n_samples, int stride);
const BnTQHeadCalibration *bn_tq_get_head_calibration(const BnTQState *state, int head_idx);

// Quantize a key vector [head_dim] into packed format.
// out must be bn_tq_key_bytes(st) bytes.
void bn_tq_quantize_key(const BnTQState *st, const float *key, uint8_t *out);
void bn_tq_quantize_key_head(const BnTQState *st, int head_idx, const float *key, uint8_t *out);

// Quantize a value vector [head_dim] into packed format.
// out must be bn_tq_value_bytes(st) bytes.
void bn_tq_quantize_value(const BnTQState *st, const float *val, uint8_t *out);
void bn_tq_quantize_value_head(const BnTQState *st, int head_idx, const float *val, uint8_t *out);

// Rotate query: q_out = Pi * q_in. Both [head_dim].
void bn_tq_rotate_query(const BnTQState *st, const float *q_in, float *q_out);
void bn_tq_rotate_query_head(const BnTQState *st, int head_idx, const float *q_in, float *q_out);

// Compute attention scores for one rotated query against n_keys packed keys.
// rotated_q: [head_dim] (already rotated via bn_tq_rotate_query).
// packed_keys: contiguous packed keys, each key_stride bytes apart.
// scores_out: [n_keys] raw dot-product scores (unscaled).
void bn_tq_attention_scores(const BnTQState *st, const float *rotated_q,
                             const uint8_t *packed_keys, int n_keys,
                             int key_stride, float *scores_out);

// Weighted sum of dequantized values: out = sum(weights[i] * dequant(packed_values[i])).
// packed_values: contiguous packed values, each val_stride bytes apart.
// weights: [n_keys] attention weights (after softmax).
// out: [head_dim] result.
void bn_tq_attention_combine(const BnTQState *st, const uint8_t *packed_values,
                               int n_keys, int val_stride,
                               const float *weights, float *out);
void bn_tq_attention_combine_head(const BnTQState *st, int head_idx,
                                  const uint8_t *packed_values, int n_keys, int val_stride,
                                  const float *weights, float *out, int accumulate);

// Precompute QJL sign projection for a rotated query (once per head).
// q_signs_out must be head_dim/8 bytes.
void bn_tq_qjl_precompute(const BnTQState *st, const float *rotated_q,
                            uint8_t *q_signs_out);
void bn_tq_qjl_precompute_head(const BnTQState *st, int head_idx,
                               const float *rotated_q, uint8_t *q_signs_out);

// Score one packed key using precomputed QJL signs (avoids redundant projection).
float bn_tq_score_key_precomputed(const BnTQState *st, const float *rotated_q,
                                    const uint8_t *q_signs, const uint8_t *packed_key);
float bn_tq_score_key_precomputed_head(const BnTQState *st, int head_idx,
                                       const float *rotated_q, const uint8_t *q_signs,
                                       const uint8_t *packed_key);

// Packed byte sizes per vector.
int bn_tq_key_bytes(const BnTQState *st);
int bn_tq_value_bytes(const BnTQState *st);

#endif // BN_TURBOQUANT_H
