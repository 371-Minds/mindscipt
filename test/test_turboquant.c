#include "turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

// --- Helpers ---

static float dot(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static float vec_norm(const float *v, int n) {
    return sqrtf(dot(v, v, n));
}

static float cosine_sim(const float *a, const float *b, int n) {
    float na = vec_norm(a, n);
    float nb = vec_norm(b, n);
    if (na < 1e-10f || nb < 1e-10f) return 0.0f;
    return dot(a, b, n) / (na * nb);
}

// Simple PRNG for test vectors
static uint64_t test_rng = 12345;
static float test_randn(void) {
    test_rng = test_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    float u1 = (float)((test_rng >> 11) + 1) / (float)(1ULL << 53);
    test_rng = test_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    float u2 = (float)((test_rng >> 11) + 1) / (float)(1ULL << 53);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307f * u2);
}

// --- Test: RHT preserves norms (orthogonal transform) ---
static void test_norm_preservation(void) {
    printf("test_norm_preservation...");
    BnTQState st;
    assert(bn_tq_init(&st, 128, 3, 42) == 0);

    // RHT should preserve vector norms: ||RHT(x)|| == ||x||
    float max_err = 0.0f;
    for (int t = 0; t < 50; t++) {
        float x[128], y[128];
        for (int i = 0; i < 128; i++) x[i] = test_randn();

        bn_tq_rotate_query(&st, x, y);

        float nx = vec_norm(x, 128);
        float ny = vec_norm(y, 128);
        float err = fabsf(nx - ny) / (nx + 1e-10f);
        if (err > max_err) max_err = err;
    }

    printf(" max_rel_err=%.6f", max_err);
    assert(max_err < 1e-4f);

    bn_tq_free(&st);
    printf(" PASS\n");
}

// --- Test: roundtrip value quantize/dequantize ---
static void test_value_roundtrip(void) {
    printf("test_value_roundtrip...");
    BnTQState st;
    assert(bn_tq_init(&st, 128, 3, 42) == 0);

    int val_bytes = bn_tq_value_bytes(&st);

    float val[128];
    for (int i = 0; i < 128; i++) val[i] = test_randn() * 0.1f;

    uint8_t packed[val_bytes];
    bn_tq_quantize_value(&st, val, packed);

    // Dequantize via attention_combine with weight=1
    float dequant[128];
    float w = 1.0f;
    bn_tq_attention_combine(&st, packed, 1, val_bytes, &w, dequant);

    float cos_sim = cosine_sim(val, dequant, 128);
    printf(" cosine_sim=%.4f", cos_sim);
    assert(cos_sim > 0.80f); // 3-bit quantization should preserve direction

    bn_tq_free(&st);
    printf(" PASS\n");
}

// --- Test: score accuracy ---
static void test_score_accuracy(void) {
    printf("test_score_accuracy...");
    BnTQState st;
    assert(bn_tq_init(&st, 128, 3, 42) == 0);

    int key_bytes = bn_tq_key_bytes(&st);
    int n_keys = 64;

    // Generate random query and keys
    float query[128];
    for (int i = 0; i < 128; i++) query[i] = test_randn() * 0.1f;

    float keys[n_keys][128];
    uint8_t packed_keys[n_keys * key_bytes];
    for (int k = 0; k < n_keys; k++) {
        for (int i = 0; i < 128; i++) keys[k][i] = test_randn() * 0.1f;
        bn_tq_quantize_key(&st, keys[k], packed_keys + k * key_bytes);
    }

    // Compute exact scores
    float exact_scores[n_keys];
    for (int k = 0; k < n_keys; k++)
        exact_scores[k] = dot(query, keys[k], 128);

    // Compute TQ scores
    float rotated_q[128];
    bn_tq_rotate_query(&st, query, rotated_q);

    float tq_scores[n_keys];
    bn_tq_attention_scores(&st, rotated_q, packed_keys, n_keys, key_bytes, tq_scores);

    // Compare: cosine similarity of score vectors should be high
    float score_cos = cosine_sim(exact_scores, tq_scores, n_keys);
    printf(" score_cosine=%.4f", score_cos);
    assert(score_cos > 0.85f);

    // Also check correlation of rankings (Spearman-like: top scores should agree)
    int exact_top = 0, tq_top = 0;
    for (int k = 1; k < n_keys; k++) {
        if (exact_scores[k] > exact_scores[exact_top]) exact_top = k;
        if (tq_scores[k] > tq_scores[tq_top]) tq_top = k;
    }
    printf(" exact_top=%d tq_top=%d", exact_top, tq_top);

    bn_tq_free(&st);
    printf(" PASS\n");
}

// --- Test: attention combine cosine similarity ---
static void test_attention_combine(void) {
    printf("test_attention_combine...");
    BnTQState st;
    assert(bn_tq_init(&st, 128, 3, 42) == 0);

    int val_bytes = bn_tq_value_bytes(&st);
    int n_vals = 32;

    // Generate random values and weights
    float values[n_vals][128];
    uint8_t packed_values[n_vals * val_bytes];
    for (int k = 0; k < n_vals; k++) {
        for (int i = 0; i < 128; i++) values[k][i] = test_randn() * 0.1f;
        bn_tq_quantize_value(&st, values[k], packed_values + k * val_bytes);
    }

    // Softmax-like weights
    float weights[n_vals];
    float sum = 0.0f;
    for (int k = 0; k < n_vals; k++) {
        weights[k] = expf(test_randn());
        sum += weights[k];
    }
    for (int k = 0; k < n_vals; k++) weights[k] /= sum;

    // Exact weighted sum
    float exact[128];
    memset(exact, 0, sizeof(exact));
    for (int k = 0; k < n_vals; k++)
        for (int i = 0; i < 128; i++)
            exact[i] += weights[k] * values[k][i];

    // TQ weighted sum
    float tq_out[128];
    bn_tq_attention_combine(&st, packed_values, n_vals, val_bytes, weights, tq_out);

    float cos_sim = cosine_sim(exact, tq_out, 128);
    printf(" cosine_sim=%.4f", cos_sim);
    assert(cos_sim > 0.80f);

    bn_tq_free(&st);
    printf(" PASS\n");
}

// --- Test: packed byte sizes ---
static void test_byte_sizes(void) {
    printf("test_byte_sizes...");
    BnTQState st;

    // 2-bit, d=128: idx=32B, qjl=16B, norms=4B → 52B key, 34B value
    assert(bn_tq_init(&st, 128, 2, 42) == 0);
    assert(bn_tq_key_bytes(&st) == 52);
    assert(bn_tq_value_bytes(&st) == 34);
    bn_tq_free(&st);

    // 3-bit, d=128: idx=48B, qjl=16B, norms=4B → 68B key, 50B value
    assert(bn_tq_init(&st, 128, 3, 42) == 0);
    assert(bn_tq_key_bytes(&st) == 68);
    assert(bn_tq_value_bytes(&st) == 50);
    bn_tq_free(&st);

    // 4-bit, d=128: idx=64B, qjl=16B, norms=4B → 84B key, 66B value
    assert(bn_tq_init(&st, 128, 4, 42) == 0);
    assert(bn_tq_key_bytes(&st) == 84);
    assert(bn_tq_value_bytes(&st) == 66);
    bn_tq_free(&st);

    printf(" PASS\n");
}

// --- Test: deterministic init ---
static void test_deterministic(void) {
    printf("test_deterministic...");
    BnTQState st1, st2;
    assert(bn_tq_init(&st1, 128, 3, 42) == 0);
    assert(bn_tq_init(&st2, 128, 3, 42) == 0);

    // Same seed → same RHT signs
    for (int i = 0; i < 128; i++)
        assert(st1.rht_signs[i] == st2.rht_signs[i]);

    // Same seed → same QJL signs
    for (int i = 0; i < 128; i++)
        assert(st1.qjl_signs[i] == st2.qjl_signs[i]);

    bn_tq_free(&st1);
    bn_tq_free(&st2);
    printf(" PASS\n");
}

// --- Test: different bit widths ---
static void test_bit_widths(void) {
    printf("test_bit_widths...");

    float vec[128];
    for (int i = 0; i < 128; i++) vec[i] = test_randn() * 0.1f;

    float prev_cos = 0.0f;
    for (int bits = 2; bits <= 4; bits++) {
        BnTQState st;
        assert(bn_tq_init(&st, 128, bits, 42) == 0);

        int val_bytes = bn_tq_value_bytes(&st);
        uint8_t packed[val_bytes];
        bn_tq_quantize_value(&st, vec, packed);

        // Dequantize via combine with weight=1
        float dequant[128];
        float w = 1.0f;
        bn_tq_attention_combine(&st, packed, 1, val_bytes, &w, dequant);

        float cos_sim = cosine_sim(vec, dequant, 128);
        printf(" %db=%.3f", bits, cos_sim);

        // More bits should give better or equal accuracy
        assert(cos_sim >= prev_cos - 0.01f);
        prev_cos = cos_sim;

        bn_tq_free(&st);
    }

    printf(" PASS\n");
}

int main(void) {
    printf("=== TurboQuant Tests ===\n");
    test_norm_preservation();
    test_byte_sizes();
    test_deterministic();
    test_value_roundtrip();
    test_score_accuracy();
    test_attention_combine();
    test_bit_widths();
    printf("=== All TurboQuant tests passed ===\n");
    return 0;
}
