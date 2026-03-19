#ifndef BN_MODEL_H
#define BN_MODEL_H

#include "platform.h"
#include "gguf.h"
#include "quant.h"
#include "threadpool.h"
#include "sh_arena.h"

#define BN_DEFAULT_ROPE_THETA  10000.0f
#define BN_DEFAULT_NORM_EPS    1e-5f

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads;
    int vocab_size, seq_len;
    float rope_theta, norm_eps;
    int head_size, kv_dim, kv_mul;  // derived
    int has_ffn_gate, act_type;     // 0=SiLU, 1=ReLU²
    int flash_attn;                 // use flash attention (online softmax)
    int kv_f16;                     // store KV cache in FP16 (halves attention DRAM bandwidth)
    // Hybrid SSM + Attention (all zero = pure attention, backward compatible)
    int rope_dim_count;             // partial RoPE dim count (0 = full head_size)
    int full_attn_interval;         // 0 = all attention, N = every Nth layer is attention
    int ssm_state_size;             // head_k_dim (128)
    int ssm_conv_kernel;            // conv kernel size (4)
    int ssm_inner_size;             // value_dim = num_v_heads * head_v_dim (4096)
    int ssm_time_step_rank;         // num_v_heads (32)
    int ssm_group_count;            // num_k_heads (16)
} BnConfig;

typedef struct {
    float *attn_norm, *attn_sub_norm;       // RMSNorm weights [dim]
    BnQWeight wq, wk, wv, wo;                 // attention projection weights (NULL for SSM layers)
    float *q_bias, *k_bias, *v_bias;        // attention biases (NULL if not present)
    float *q_norm, *k_norm;                 // per-head Q/K RMSNorm (NULL if absent)
    float *ffn_norm, *ffn_sub_norm;         // RMSNorm weights
    BnQWeight ffn_gate, ffn_up, ffn_down;     // FFN weights
    // SSM-specific (NULL/zero for attention layers)
    BnQWeight wqkv;                         // fused QKV [dim, qkv_dim]
    BnQWeight wz;                           // Z gate projection [dim, value_dim]
    float *ssm_a;                           // [num_v_heads] F32 — A_log
    BnQWeight ssm_alpha;                    // [dim, num_v_heads] — decay projection
    BnQWeight ssm_beta;                     // [dim, num_v_heads] — update rate projection
    float *ssm_conv1d;                      // [conv_kernel, conv_dim] F32
    float *ssm_dt_bias;                     // [num_v_heads] F32
    float *ssm_norm;                        // [head_v_dim] F32
    BnQWeight ssm_out;                      // [value_dim, dim]
} BnLayerWeights;

typedef struct {
    const void *token_embedding;  // raw embedding data (F16, Q4_0, Q8_0, etc.)
    int emb_type;                 // tensor type (F16, Q4_0, Q8_0, etc.)
    int8_t *emb_out_i8;          // [vocab_size * dim] INT8 copy for logits (NULL if unused)
    float  *emb_out_scales;      // [vocab_size] per-row scales (NULL if unused)
    BnQWeight output_weight;      // untied output projection (data=NULL if tied)
    float *output_norm;           // [dim]
    BnLayerWeights *layers;         // [n_layers]
} BnWeights;

typedef struct {
    float *x, *xb, *xb2;         // [dim] activation buffers
    float *hb, *hb2;             // [hidden_dim]
    float *q;                     // [dim] query buffer
    float *att;                   // [n_heads * seq_len] attention scores
    float *logits;                // [vocab_size]
    float *key_cache;             // [n_attn_layers * seq_len * kv_dim]
    float *value_cache;           // [n_attn_layers * seq_len * kv_dim]
    int8_t *x_q;                  // [max(dim, hidden_dim)] scratch for int8 quantized x
    float *rope_freq;             // [head_size/2] precomputed RoPE frequencies
    // SSM state (NULL if no SSM layers)
    float *ssm_state;             // [n_ssm * num_v_heads * head_k_dim * head_v_dim]
    float *ssm_conv_state;        // [n_ssm * (conv_kernel-1) * conv_dim]
} BnRunState;

typedef struct {
    BnConfig config;
    BnWeights weights;
    BnRunState state;
    BnMappedFile file;  // keeps mmap/buffer alive
    BnThreadPool *pool; // thread pool for parallel dispatch
    SHArena *arena;     // arena for all RunState buffers
} BnModel;

int  bn_model_load(BnModel *m, BnGGUFFile *f, int max_seq_len, int kv_f16);
void bn_model_free(BnModel *m);
void bn_model_reset_state(BnModel *m);
void bn_model_embed_token(const BnModel *m, float *out, int token);

#endif // BN_MODEL_H
