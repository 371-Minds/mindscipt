#include <metal_stdlib>
using namespace metal;

// Fused flash attention: Q·K scores + online softmax + weighted V combine
// Zero barriers, zero shared memory — uses simd_sum for dot product reduction.
//
// 8 heads per threadgroup (1 simdgroup = 32 threads per head).
// Each thread owns head_size/32 output dimensions and loops over all KV positions.
// simd_sum across 32 lanes gives the full Q·K dot product (0 barriers).
// Online softmax per thread (all lanes see the same score via simd_sum).
//
// Dispatch: (ceil(n_heads/8), 1, 1)
//
// p0 = n_heads, p1 = head_size, p2 = n_kv, p3 = kv_mul
// p4 = kv_dim, p5 = seq_len, p6 = loff, p7 = inv_sqrt_hs (bitcast)

// Max dims per thread: head_size=256 / 32 threads = 8
constant uint MAX_DPT = 8;

kernel void flash_attn(device const float *q           [[buffer(0)]],
                       device const float *key_cache   [[buffer(1)]],
                       device const float *value_cache [[buffer(2)]],
                       device float       *xb          [[buffer(3)]],
                       constant uint      *p           [[buffer(4)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    uint n_heads = p[0], head_size = p[1], n_kv = p[2], kv_mul = p[3];
    uint kv_dim = p[4], loff = p[6];
    float scale = as_type<float>(p[7]);

    uint simd_id = lid.x >> 5;       // 0-7: which simdgroup
    uint lane = lid.x & 31;          // 0-31: lane within simdgroup
    uint h = wid.x * 8 + simd_id;   // global head index

    if (h >= n_heads) return;

    uint kv_h = h / kv_mul;
    uint dpt = (head_size + 31) >> 5;  // dims per thread = ceil(head_size/32)
    uint d_base = lane * dpt;

    // Load Q for this head into registers
    float q_reg[MAX_DPT];
    for (uint i = 0; i < dpt && d_base + i < head_size; i++)
        q_reg[i] = q[h * head_size + d_base + i];

    // Online softmax state
    float my_max = -3.402823e+38f;
    float my_sum = 0.0f;
    float out_reg[MAX_DPT];
    for (uint i = 0; i < dpt; i++)
        out_reg[i] = 0.0f;

    for (uint t = 0; t < n_kv; t++) {
        uint kv_base = loff + t * kv_dim + kv_h * head_size;

        // Partial Q·K dot product (dpt elements per thread)
        float partial = 0.0f;
        for (uint i = 0; i < dpt && d_base + i < head_size; i++)
            partial += q_reg[i] * key_cache[kv_base + d_base + i];

        // Full dot product via simd_sum (0 barriers)
        float score = simd_sum(partial) * scale;

        // Online softmax update (all 32 lanes compute the same values)
        float prev_max = my_max;
        my_max = max(my_max, score);
        float exp_diff = exp(prev_max - my_max);
        float exp_score = exp(score - my_max);
        my_sum = my_sum * exp_diff + exp_score;

        // Weighted V accumulation (each thread handles its own dimensions)
        for (uint i = 0; i < dpt && d_base + i < head_size; i++)
            out_reg[i] = out_reg[i] * exp_diff + exp_score * value_cache[kv_base + d_base + i];
    }

    // Write output (each thread writes its dimensions independently)
    float inv_sum = 1.0f / my_sum;
    uint out_base = h * head_size;
    for (uint i = 0; i < dpt && d_base + i < head_size; i++)
        xb[out_base + d_base + i] = out_reg[i] * inv_sum;
}
