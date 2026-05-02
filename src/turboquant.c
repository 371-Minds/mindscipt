#include "turboquant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TQ_HEADER_BYTES 2
#define TQ_OUTLIER_CLIP_FACTOR 2.0f
#define TQ_CONSERVATIVE_CLIP_FACTOR 1.5f
#define TQ_CALIBRATED_QJL_WEIGHT 0.85f
#define TQ_CONSERVATIVE_ERROR_THRESHOLD 0.35f
#define TQ_OUTLIER_RATIO_THRESHOLD 0.12f
#define TQ_OUTLIER_VARIANCE_THRESHOLD 3.5f
#define TQ_CALIBRATED_ERROR_THRESHOLD 0.15f
#define TQ_EPSILON 1e-6f

#ifdef __ARM_NEON
#include <arm_neon.h>
static inline float tq_neon_hsum(float32x4_t v) {
    float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}
#endif

static const BnTQHeadRuntime tq_default_runtime = {
    .strategy = BN_TQ_STRATEGY_BASELINE,
    .clip_threshold = 0.0f,
    .qjl_weight = 1.0f,
};

// --- xoshiro256** PRNG (deterministic, seeded, thread-safe via local state) ---

static uint64_t tq_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static void tq_rng_seed(uint64_t rng[4], uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng[i] = z ^ (z >> 31);
    }
}

static uint64_t tq_rng_next(uint64_t *s) {
    uint64_t result = tq_rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = tq_rotl(s[3], 45);
    return result;
}

// --- Lloyd-Max centroids for N(0,1), hardcoded ---
// These are the optimal scalar quantization centroids for a standard Gaussian.
// Scaled by 1/sqrt(d) at init time.

static const float lloyd_max_2bit[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};
static const float lloyd_max_2bit_bounds[3] = {
    -0.9816f, 0.0f, 0.9816f
};

static const float lloyd_max_3bit[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f
};
static const float lloyd_max_3bit_bounds[7] = {
    -1.7480f, -1.0500f, -0.5006f, 0.0f, 0.5006f, 1.0500f, 1.7480f
};

static const float lloyd_max_4bit[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9424f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9424f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};
static const float lloyd_max_4bit_bounds[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f,
    -0.7996f, -0.5224f, -0.2582f, 0.0f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,
     1.4371f,  1.8435f,  2.4008f
};

// --- Fast Walsh-Hadamard Transform (in-place, unnormalized) ---
// d must be a power of 2, >= 8. Enforced by bn_tq_init. O(d log d) add/sub operations.

#ifdef __ARM_NEON
static void fwht_inplace(float *x, int d) {
    // Stage 0: len=1 butterflies (stride-1, deinterleave)
    for (int i = 0; i < d; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4x2_t deint = vuzpq_f32(v0, v1);
        float32x4_t sum = vaddq_f32(deint.val[0], deint.val[1]);
        float32x4_t dif = vsubq_f32(deint.val[0], deint.val[1]);
        float32x4x2_t reint = vzipq_f32(sum, dif);
        vst1q_f32(x + i, reint.val[0]);
        vst1q_f32(x + i + 4, reint.val[1]);
    }
    // Stage 1: len=2 butterflies (stride-2, lo/hi split)
    for (int i = 0; i < d; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x2_t lo = vget_low_f32(v);
        float32x2_t hi = vget_high_f32(v);
        vst1q_f32(x + i, vcombine_f32(vadd_f32(lo, hi), vsub_f32(lo, hi)));
    }
    // Stages 2+: len=4,8,16,... (contiguous NEON pairs)
    for (int len = 4; len < d; len <<= 1) {
        for (int i = 0; i < d; i += len << 1) {
            for (int j = i; j < i + len; j += 4) {
                float32x4_t a = vld1q_f32(x + j);
                float32x4_t b = vld1q_f32(x + j + len);
                vst1q_f32(x + j, vaddq_f32(a, b));
                vst1q_f32(x + j + len, vsubq_f32(a, b));
            }
        }
    }
}
#else
static void fwht_inplace(float *x, int d) {
    for (int len = 1; len < d; len <<= 1) {
        for (int i = 0; i < d; i += len << 1) {
            for (int j = i; j < i + len; j++) {
                float a = x[j], b = x[j + len];
                x[j] = a + b;
                x[j + len] = a - b;
            }
        }
    }
}
#endif

// --- NEON helpers for sign application ---
// Invariant: d >= 8 and d is a power of 2 (enforced by bn_tq_init), so d % 4 == 0.
#ifdef __ARM_NEON
static inline void tq_apply_signs_neon(const float *signs, const float *in, float *out, int d) {
    uint32x4_t sign_mask = vdupq_n_u32(0x80000000);
    for (int i = 0; i < d; i += 4) {
        uint32x4_t v = vreinterpretq_u32_f32(vld1q_f32(in + i));
        uint32x4_t s = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(signs + i)), sign_mask);
        vst1q_f32(out + i, vreinterpretq_f32_u32(veorq_u32(v, s)));
    }
}
static inline void tq_scale_neon(float *x, float scale, int d) {
    float32x4_t sv = vdupq_n_f32(scale);
    for (int i = 0; i < d; i += 4)
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), sv));
}
#endif

// RHT forward: out = scale * H * D * in
static void rht_forward(const BnTQState *st, const float *in, float *out, int d) {
#ifdef __ARM_NEON
    tq_apply_signs_neon(st->rht_signs, in, out, d);
    fwht_inplace(out, d);
    tq_scale_neon(out, st->rht_scale, d);
#else
    for (int i = 0; i < d; i++)
        out[i] = st->rht_signs[i] * in[i];
    fwht_inplace(out, d);
    float s = st->rht_scale;
    for (int i = 0; i < d; i++)
        out[i] *= s;
#endif
}

// RHT inverse: out = D * H * (scale * in)
// Inverse is: D * H * scale * in (H and D are self-inverse, scale cancels).
static void rht_inverse(const BnTQState *st, const float *in, float *out, int d) {
#ifdef __ARM_NEON
    {
        float32x4_t sv = vdupq_n_f32(st->rht_scale);
        for (int i = 0; i < d; i += 4)
            vst1q_f32(out + i, vmulq_f32(vld1q_f32(in + i), sv));
    }
    fwht_inplace(out, d);
    tq_apply_signs_neon(st->rht_signs, out, out, d);
#else
    float s = st->rht_scale;
    for (int i = 0; i < d; i++)
        out[i] = s * in[i];
    fwht_inplace(out, d);
    for (int i = 0; i < d; i++)
        out[i] *= st->rht_signs[i];
#endif
}

// QJL sign projection via RHT: out_signs = sign(H * D_qjl * in)
// Produces d/8 bytes of packed sign bits. O(d log d).
// Precondition: d % 8 == 0 (enforced by bn_tq_init).
static void qjl_project_signs(const BnTQState *st, const float *in, uint8_t *out_signs, int d) {
    float tmp[d];
#ifdef __ARM_NEON
    tq_apply_signs_neon(st->qjl_signs, in, tmp, d);
#else
    for (int i = 0; i < d; i++)
        tmp[i] = st->qjl_signs[i] * in[i];
#endif
    fwht_inplace(tmp, d);
    memset(out_signs, 0, (size_t)d / 8);
    for (int i = 0; i < d; i++)
        if (tmp[i] >= 0.0f)
            out_signs[i / 8] |= (1 << (i % 8));
}

// --- Quantize a scalar to nearest centroid index ---

static inline int quantize_scalar(float x, const float *boundaries, int n_centroids) {
    // Binary search on boundaries
    int lo = 0, hi = n_centroids - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (mid < n_centroids - 1 && x >= boundaries[mid])
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// --- Pack indices ---
// Precondition: d must be a multiple of 8 (enforced by bn_tq_init power-of-2 >= 8).

// Pack 2-bit indices: 4 per byte, d indices → d/4 bytes
static void pack_2bit(const int *indices, int d, uint8_t *out) {
    for (int i = 0; i < d; i += 4) {
        out[i / 4] = (uint8_t)((indices[i] & 3) |
                                ((indices[i+1] & 3) << 2) |
                                ((indices[i+2] & 3) << 4) |
                                ((indices[i+3] & 3) << 6));
    }
}

// Unpack 2-bit indices
static void unpack_2bit(const uint8_t *packed, int d, int *indices) {
    for (int i = 0; i < d; i += 4) {
        uint8_t b = packed[i / 4];
        indices[i]   = b & 3;
        indices[i+1] = (b >> 2) & 3;
        indices[i+2] = (b >> 4) & 3;
        indices[i+3] = (b >> 6) & 3;
    }
}

// Pack 3-bit indices: 8 indices per 3 bytes, d indices → d*3/8 bytes
static void pack_3bit(const int *indices, int d, uint8_t *out) {
    for (int i = 0; i < d; i += 8) {
        // Pack 8 3-bit values into 3 bytes (24 bits)
        uint32_t packed = 0;
        for (int j = 0; j < 8; j++)
            packed |= ((uint32_t)(indices[i + j] & 7)) << (j * 3);
        out[(i / 8) * 3]     = (uint8_t)(packed & 0xFF);
        out[(i / 8) * 3 + 1] = (uint8_t)((packed >> 8) & 0xFF);
        out[(i / 8) * 3 + 2] = (uint8_t)((packed >> 16) & 0xFF);
    }
}

// Unpack 3-bit indices
static void unpack_3bit(const uint8_t *packed, int d, int *indices) {
    for (int i = 0; i < d; i += 8) {
        uint32_t v = (uint32_t)packed[(i / 8) * 3] |
                     ((uint32_t)packed[(i / 8) * 3 + 1] << 8) |
                     ((uint32_t)packed[(i / 8) * 3 + 2] << 16);
        for (int j = 0; j < 8; j++)
            indices[i + j] = (v >> (j * 3)) & 7;
    }
}

// Pack 4-bit indices: 2 per byte, d indices → d/2 bytes
static void pack_4bit(const int *indices, int d, uint8_t *out) {
    for (int i = 0; i < d; i += 2)
        out[i / 2] = (uint8_t)((indices[i] & 0xF) | ((indices[i+1] & 0xF) << 4));
}

// Unpack 4-bit indices
static void unpack_4bit(const uint8_t *packed, int d, int *indices) {
    for (int i = 0; i < d; i += 2) {
        uint8_t b = packed[i / 2];
        indices[i]   = b & 0xF;
        indices[i+1] = (b >> 4) & 0xF;
    }
}

// --- Dispatch pack/unpack by bits ---

static void pack_indices(const int *indices, int d, int bits, uint8_t *out) {
    if (bits == 2) pack_2bit(indices, d, out);
    else if (bits == 3) pack_3bit(indices, d, out);
    else pack_4bit(indices, d, out);
}

static void unpack_indices(const uint8_t *packed, int d, int bits, int *indices) {
    if (bits == 2) unpack_2bit(packed, d, indices);
    else if (bits == 3) unpack_3bit(packed, d, indices);
    else unpack_4bit(packed, d, indices);
}

static int index_bytes(int d, int bits) {
    if (bits == 2) return d / 4;
    if (bits == 3) return d * 3 / 8;
    return d / 2; // bits == 4
}

static inline const BnTQHeadRuntime *tq_runtime_head(const BnTQState *st, int head_idx) {
    if (!st || !st->head_runtime || st->n_heads <= 0)
        return &tq_default_runtime;
    if (head_idx < 0) head_idx = 0;
    if (head_idx >= st->n_heads) head_idx = st->n_heads - 1;
    return &st->head_runtime[head_idx];
}

static inline BnTQHeadRuntime *tq_runtime_head_mut(BnTQState *st, int head_idx) {
    if (!st || !st->head_runtime || st->n_heads <= 0)
        return NULL;
    if (head_idx < 0) head_idx = 0;
    if (head_idx >= st->n_heads) head_idx = st->n_heads - 1;
    return &st->head_runtime[head_idx];
}

static inline BnTQHeadCalibration *tq_calibration_head_mut(BnTQState *st, int head_idx) {
    if (!st || !st->head_calibration || st->n_heads <= 0)
        return NULL;
    if (head_idx < 0) head_idx = 0;
    if (head_idx >= st->n_heads) head_idx = st->n_heads - 1;
    return &st->head_calibration[head_idx];
}

static inline const BnTQHeadCalibration *tq_calibration_head(const BnTQState *st, int head_idx) {
    if (!st || !st->head_calibration || st->n_heads <= 0)
        return NULL;
    if (head_idx < 0) head_idx = 0;
    if (head_idx >= st->n_heads) head_idx = st->n_heads - 1;
    return &st->head_calibration[head_idx];
}

static void tq_apply_runtime_clip(float *x, int d, float clip_threshold) {
    if (clip_threshold <= 0.0f) return;
    for (int i = 0; i < d; i++) {
        if (x[i] > clip_threshold) x[i] = clip_threshold;
        else if (x[i] < -clip_threshold) x[i] = -clip_threshold;
    }
}

static inline int tq_respects_qjl(uint8_t strategy) {
    return strategy != BN_TQ_STRATEGY_CONSERVATIVE;
}

static inline int tq_strategy_valid(uint8_t strategy) {
    return strategy < BN_TQ_STRATEGY_COUNT;
}

static inline float tq_qjl_weight_for_strategy(uint8_t strategy) {
    switch (strategy) {
        case BN_TQ_STRATEGY_CALIBRATED:
            return TQ_CALIBRATED_QJL_WEIGHT;
        case BN_TQ_STRATEGY_CONSERVATIVE:
            return 0.0f;
        case BN_TQ_STRATEGY_BASELINE:
        case BN_TQ_STRATEGY_OUTLIER:
            return 1.0f;
        default:
            return 0.0f;
    }
}

static void tq_write_header(const BnTQState *st, uint8_t strategy, uint8_t *out) {
    out[0] = st ? st->format_version : BN_TQ_FORMAT_VERSION;
    out[1] = strategy;
}

static uint8_t tq_read_version(const uint8_t *packed) {
    return packed ? packed[0] : 0;
}

static uint8_t tq_read_strategy(const uint8_t *packed) {
    return packed ? packed[1] : BN_TQ_STRATEGY_BASELINE;
}

// --- FP16 helpers (IEEE 754 half-precision) ---

// Accepted: denorms flush to zero. Smallest representable is ~6.1e-5.
// Residual norms below this threshold lose QJL correction (negligible in practice).
static inline uint16_t tq_fp32_to_fp16(float f) {
    union { float f; uint32_t u; } fi = { .f = f };
    uint32_t sign = (fi.u >> 16) & 0x8000;
    int exp = ((fi.u >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (fi.u >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign; // flush denorms to zero
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | frac);
}

static inline float tq_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) { union { uint32_t u; float f; } r = { .u = sign }; return r.f; }
        // Denorm
        exp = 1;
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
    } else if (exp == 31) {
        union { uint32_t u; float f; } r = { .u = sign | 0x7F800000 | (frac << 13) };
        return r.f;
    }
    uint32_t bits = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    union { uint32_t u; float f; } r = { .u = bits };
    return r.f;
}

// --- Key packed layout (d=128, b=3): ---
// [version:1B][strategy:1B][idx_bytes][qjl_signs][residual_norm:fp16][vec_norm:fp16]

// --- Value packed layout (d=128, b=3): ---
// [version:1B][strategy:1B][idx_bytes][vec_norm: fp16=2B]

int bn_tq_key_bytes(const BnTQState *st) {
    return TQ_HEADER_BYTES + index_bytes(st->head_dim, st->bits) + st->head_dim / 8 + 4;
}

int bn_tq_value_bytes(const BnTQState *st) {
    return TQ_HEADER_BYTES + index_bytes(st->head_dim, st->bits) + 2;
}

// --- Init / Free ---

int bn_tq_init(BnTQState *state, int head_dim, int bits, uint64_t seed) {
    if (!state) return -1;
    if (bits < 2 || bits > 4) return -1;
    if (head_dim <= 0 || (head_dim & (head_dim - 1)) != 0) return -1; // must be power of 2
    if (head_dim < 8) return -1; // minimum for NEON (8-wide FWHT stage 0)

    memset(state, 0, sizeof(BnTQState));
    state->head_dim = head_dim;
    state->bits = bits;
    state->format_version = BN_TQ_FORMAT_VERSION;
    state->tables.n_centroids = 1 << bits;
    state->n_heads = 1;

    int d = head_dim;
    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    // Allocate centroids + boundaries
    state->tables.centroids = (float *)malloc(state->tables.n_centroids * sizeof(float));
    state->tables.boundaries = (float *)malloc((state->tables.n_centroids - 1) * sizeof(float));
    if (!state->tables.centroids || !state->tables.boundaries) { bn_tq_free(state); return -1; }

    // Copy and scale Lloyd-Max centroids
    const float *src_c, *src_b;
    if (bits == 2) { src_c = lloyd_max_2bit; src_b = lloyd_max_2bit_bounds; }
    else if (bits == 3) { src_c = lloyd_max_3bit; src_b = lloyd_max_3bit_bounds; }
    else { src_c = lloyd_max_4bit; src_b = lloyd_max_4bit_bounds; }

    for (int i = 0; i < state->tables.n_centroids; i++)
        state->tables.centroids[i] = src_c[i] * inv_sqrt_d;
    for (int i = 0; i < state->tables.n_centroids - 1; i++)
        state->tables.boundaries[i] = src_b[i] * inv_sqrt_d;

    // Generate RHT diagonal signs: random ±1 from seeded PRNG (local state, thread-safe)
    state->rht_signs = (float *)malloc((size_t)d * sizeof(float));
    if (!state->rht_signs) { bn_tq_free(state); return -1; }

    uint64_t rng[4];
    tq_rng_seed(rng, seed);
    for (int i = 0; i < d; i++)
        state->rht_signs[i] = (tq_rng_next(rng) & 1) ? 1.0f : -1.0f;
    state->rht_scale = inv_sqrt_d;

    // Generate QJL diagonal signs (second RHT with independent signs)
    state->qjl_signs = (float *)malloc((size_t)d * sizeof(float));
    if (!state->qjl_signs) { bn_tq_free(state); return -1; }
    for (int i = 0; i < d; i++)
        state->qjl_signs[i] = (tq_rng_next(rng) & 1) ? 1.0f : -1.0f;

    if (bn_tq_configure_heads(state, 1) != 0) {
        bn_tq_free(state);
        return -1;
    }
    return 0;
}

void bn_tq_free(BnTQState *state) {
    if (!state) return;
    free(state->rht_signs);
    free(state->qjl_signs);
    free(state->tables.centroids);
    free(state->tables.boundaries);
    free(state->head_calibration);
    free(state->head_runtime);
    memset(state, 0, sizeof(BnTQState));
}

int bn_tq_configure_heads(BnTQState *state, int n_heads) {
    if (!state || n_heads <= 0) return -1;
    if (state->n_heads == n_heads && state->head_calibration && state->head_runtime)
        return 0;

    BnTQHeadCalibration *cal = (BnTQHeadCalibration *)calloc((size_t)n_heads, sizeof(BnTQHeadCalibration));
    BnTQHeadRuntime *run = (BnTQHeadRuntime *)calloc((size_t)n_heads, sizeof(BnTQHeadRuntime));
    if (!cal || !run) {
        free(cal);
        free(run);
        return -1;
    }

    for (int i = 0; i < n_heads; i++) {
        run[i] = tq_default_runtime;
    }

    free(state->head_calibration);
    free(state->head_runtime);
    state->head_calibration = cal;
    state->head_runtime = run;
    state->n_heads = n_heads;
    return 0;
}

void bn_tq_set_flags(BnTQState *state, uint32_t flags) {
    if (!state) return;
    state->flags = flags;
}

uint32_t bn_tq_get_flags(const BnTQState *state) {
    return state ? state->flags : 0;
}

uint8_t bn_tq_format_version(const BnTQState *state) {
    return state ? state->format_version : 0;
}

BnTQStrategy bn_tq_get_head_strategy(const BnTQState *state, int head_idx) {
    const BnTQHeadRuntime *rt = tq_runtime_head(state, head_idx);
    return (BnTQStrategy)rt->strategy;
}

int bn_tq_set_head_strategy(BnTQState *state, int head_idx, BnTQStrategy strategy) {
    BnTQHeadRuntime *rt;
    if (!state || strategy < 0 || strategy >= BN_TQ_STRATEGY_COUNT)
        return -1;
    rt = tq_runtime_head_mut(state, head_idx);
    if (!rt) return -1;
    rt->strategy = (uint8_t)strategy;
    rt->qjl_weight = tq_qjl_weight_for_strategy((uint8_t)strategy);
    switch (strategy) {
        case BN_TQ_STRATEGY_BASELINE:
            rt->clip_threshold = 0.0f;
            break;
        case BN_TQ_STRATEGY_CALIBRATED:
            rt->clip_threshold = 0.0f;
            break;
        case BN_TQ_STRATEGY_OUTLIER:
            if (rt->clip_threshold <= 0.0f)
                rt->clip_threshold = TQ_OUTLIER_CLIP_FACTOR * state->rht_scale;
            break;
        case BN_TQ_STRATEGY_CONSERVATIVE:
            if (rt->clip_threshold <= 0.0f)
                rt->clip_threshold = TQ_CONSERVATIVE_CLIP_FACTOR * state->rht_scale;
            break;
        default:
            return -1;
    }
    return 0;
}

const BnTQHeadCalibration *bn_tq_get_head_calibration(const BnTQState *state, int head_idx) {
    return tq_calibration_head(state, head_idx);
}

// --- Quantize key/value + query rotation ---

void bn_tq_quantize_key_head(const BnTQState *st, int head_idx, const float *key, uint8_t *out) {
    const BnTQHeadRuntime *rt;
    if (!st || !out) return;
    rt = tq_runtime_head(st, head_idx);
    int d = st->head_dim;
    if (d <= 0 || d > BN_MAX_VLA_ELEMS) return;
    int idx_sz = index_bytes(d, st->bits);
    int qjl_sz = d / 8;
    uint8_t strategy = rt->strategy;
    float norm_sq = 0.0f;
    for (int i = 0; i < d; i++) norm_sq += key[i] * key[i];
    float vec_norm = sqrtf(norm_sq);

    float inv_norm = (vec_norm > 1e-10f) ? 1.0f / vec_norm : 0.0f;
    float normalized[d];
    for (int i = 0; i < d; i++) normalized[i] = key[i] * inv_norm;
    float rotated[d];
    rht_forward(st, normalized, rotated, d);
    tq_apply_runtime_clip(rotated, d, rt->clip_threshold);

    int indices[d];
    for (int i = 0; i < d; i++)
        indices[i] = quantize_scalar(rotated[i], st->tables.boundaries, st->tables.n_centroids);

    float residual[d];
    float res_norm_sq = 0.0f;
    for (int i = 0; i < d; i++) {
        residual[i] = rotated[i] - st->tables.centroids[indices[i]];
        res_norm_sq += residual[i] * residual[i];
    }
    float res_norm = sqrtf(res_norm_sq);

    uint8_t qjl_signs[qjl_sz];
    if (tq_respects_qjl(strategy)) {
        qjl_project_signs(st, residual, qjl_signs, d);
    } else {
        memset(qjl_signs, 0, (size_t)qjl_sz);
    }

    tq_write_header(st, strategy, out);
    pack_indices(indices, d, st->bits, out + TQ_HEADER_BYTES);
    memcpy(out + TQ_HEADER_BYTES + idx_sz, qjl_signs, (size_t)qjl_sz);
    uint16_t res_norm_fp16 = tq_fp32_to_fp16(res_norm);
    uint16_t vec_norm_fp16 = tq_fp32_to_fp16(vec_norm);
    memcpy(out + TQ_HEADER_BYTES + idx_sz + qjl_sz, &res_norm_fp16, 2);
    memcpy(out + TQ_HEADER_BYTES + idx_sz + qjl_sz + 2, &vec_norm_fp16, 2);
}

void bn_tq_quantize_key(const BnTQState *st, const float *key, uint8_t *out) {
    bn_tq_quantize_key_head(st, 0, key, out);
}

void bn_tq_quantize_value_head(const BnTQState *st, int head_idx, const float *val, uint8_t *out) {
    const BnTQHeadRuntime *rt;
    if (!st || !out) return;
    rt = tq_runtime_head(st, head_idx);
    int d = st->head_dim;
    if (d <= 0 || d > BN_MAX_VLA_ELEMS) return;
    int idx_sz = index_bytes(d, st->bits);

    float norm_sq = 0.0f;
    for (int i = 0; i < d; i++) norm_sq += val[i] * val[i];
    float vec_norm = sqrtf(norm_sq);

    float inv_norm = (vec_norm > 1e-10f) ? 1.0f / vec_norm : 0.0f;
    float normalized[d];
    for (int i = 0; i < d; i++) normalized[i] = val[i] * inv_norm;
    float rotated[d];
    rht_forward(st, normalized, rotated, d);
    tq_apply_runtime_clip(rotated, d, rt->clip_threshold);

    int indices[d];
    for (int i = 0; i < d; i++)
        indices[i] = quantize_scalar(rotated[i], st->tables.boundaries, st->tables.n_centroids);

    tq_write_header(st, rt->strategy, out);
    pack_indices(indices, d, st->bits, out + TQ_HEADER_BYTES);
    uint16_t vec_norm_fp16 = tq_fp32_to_fp16(vec_norm);
    memcpy(out + TQ_HEADER_BYTES + idx_sz, &vec_norm_fp16, 2);
}

void bn_tq_quantize_value(const BnTQState *st, const float *val, uint8_t *out) {
    bn_tq_quantize_value_head(st, 0, val, out);
}

void bn_tq_rotate_query_head(const BnTQState *st, int head_idx, const float *q_in, float *q_out) {
    const BnTQHeadRuntime *rt;
    if (!st || !q_in || !q_out) return;
    rt = tq_runtime_head(st, head_idx);
    rht_forward(st, q_in, q_out, st->head_dim);
    tq_apply_runtime_clip(q_out, st->head_dim, rt->clip_threshold);
}

void bn_tq_rotate_query(const BnTQState *st, const float *q_in, float *q_out) {
    bn_tq_rotate_query_head(st, 0, q_in, q_out);
}

// --- Attention scores (the win: read 52B/key instead of 512B) ---
// Score = <q_rot, centroids[idx]> * vec_norm
//       + residual_norm * sqrt(pi/2)/d * <sign(S*q_rot), qjl_signs>

void bn_tq_attention_scores(const BnTQState *st, const float *rotated_q,
                             const uint8_t *packed_keys, int n_keys,
                             int key_stride, float *scores_out) {
    if (!st) return;
    int d = st->head_dim;
    if (d <= 0 || d > BN_MAX_VLA_ELEMS) return;
    int idx_sz = index_bytes(d, st->bits);
    int qjl_sz = d / 8;

    // Precompute sign(H * D_qjl * rotated_q) for QJL correction
    uint8_t q_signs[qjl_sz];
    qjl_project_signs(st, rotated_q, q_signs, d);

    float qjl_scale = sqrtf(3.14159265f / 2.0f) / (float)d;
    int indices[d];  // moved outside loop

    for (int k = 0; k < n_keys; k++) {
        const uint8_t *pk = packed_keys + (size_t)k * key_stride;
        if (tq_read_version(pk) != BN_TQ_FORMAT_VERSION) {
            scores_out[k] = 0.0f;
            continue;
        }
        uint8_t strategy = tq_read_strategy(pk);
        if (!tq_strategy_valid(strategy)) {
            scores_out[k] = 0.0f;
            continue;
        }

        // Unpack indices
        unpack_indices(pk + TQ_HEADER_BYTES, d, st->bits, indices);

        // Read norms
        uint16_t res_norm_fp16, vec_norm_fp16;
        memcpy(&res_norm_fp16, pk + TQ_HEADER_BYTES + idx_sz + qjl_sz, 2);
        memcpy(&vec_norm_fp16, pk + TQ_HEADER_BYTES + idx_sz + qjl_sz + 2, 2);
        float res_norm = tq_fp16_to_fp32(res_norm_fp16);
        float vec_norm = tq_fp16_to_fp32(vec_norm_fp16);

        // Centroid dot product: sum(q_rot[i] * centroids[idx[i]])
#ifdef __ARM_NEON
        float c_tmp[4];
        float32x4_t acc = vdupq_n_f32(0);
        for (int i = 0; i < d; i += 4) {
            c_tmp[0] = st->tables.centroids[indices[i]];
            c_tmp[1] = st->tables.centroids[indices[i+1]];
            c_tmp[2] = st->tables.centroids[indices[i+2]];
            c_tmp[3] = st->tables.centroids[indices[i+3]];
            acc = vfmaq_f32(acc, vld1q_f32(rotated_q + i), vld1q_f32(c_tmp));
        }
        float centroid_dot = tq_neon_hsum(acc);
#else
        float centroid_dot = 0.0f;
        for (int i = 0; i < d; i++)
            centroid_dot += rotated_q[i] * st->tables.centroids[indices[i]];
#endif

        // QJL correction: XNOR popcount between q_signs and key qjl_signs
        const uint8_t *key_signs = pk + TQ_HEADER_BYTES + idx_sz;
        int agree = 0;
        float qjl_weight = tq_qjl_weight_for_strategy(strategy);
#ifdef __ARM_NEON
        if (qjl_weight != 0.0f) {
            int b = 0;
            for (; b + 16 <= qjl_sz; b += 16) {
                uint8x16_t xnor = vmvnq_u8(veorq_u8(vld1q_u8(q_signs + b),
                                                       vld1q_u8(key_signs + b)));
                uint8x16_t cnt = vcntq_u8(xnor);
                uint16x8_t p1 = vpaddlq_u8(cnt);
                uint32x4_t p2 = vpaddlq_u16(p1);
                uint64x2_t p3 = vpaddlq_u32(p2);
                agree += (int)(vgetq_lane_u64(p3, 0) + vgetq_lane_u64(p3, 1));
            }
            for (; b < qjl_sz; b++) {
                uint8_t xnor = ~(q_signs[b] ^ key_signs[b]);
                uint8_t v = xnor;
                v = (v & 0x55) + ((v >> 1) & 0x55);
                v = (v & 0x33) + ((v >> 2) & 0x33);
                v = (v & 0x0F) + ((v >> 4) & 0x0F);
                agree += v;
            }
        }
#else
        if (qjl_weight != 0.0f) {
            for (int b = 0; b < qjl_sz; b++) {
                uint8_t xnor = ~(q_signs[b] ^ key_signs[b]);
                uint8_t v = xnor;
                v = (v & 0x55) + ((v >> 1) & 0x55);
                v = (v & 0x33) + ((v >> 2) & 0x33);
                v = (v & 0x0F) + ((v >> 4) & 0x0F);
                agree += v;
            }
        }
#endif
        // QJL estimator: (2*agree - d) * res_norm * scale
        float qjl_correction = qjl_weight * (float)(2 * agree - d) * res_norm * qjl_scale;

        scores_out[k] = vec_norm * (centroid_dot + qjl_correction);
    }
}

// --- Attention combine (dequantize values, weighted sum) ---

void bn_tq_attention_combine_head(const BnTQState *st, int head_idx,
                                  const uint8_t *packed_values,
                                  int n_keys, int val_stride,
                                  const float *weights, float *out, int accumulate) {
    const BnTQHeadRuntime *rt;
    if (!st) return;
    rt = tq_runtime_head(st, head_idx);
    int d = st->head_dim;
    if (d <= 0 || d > BN_MAX_VLA_ELEMS) return;
    int idx_sz = index_bytes(d, st->bits);

    if (!accumulate)
        memset(out, 0, d * sizeof(float));

    float rotated[d], dequant[d];
    int indices[d];  // moved outside loop to avoid per-iteration VLA allocation
    for (int k = 0; k < n_keys; k++) {
        const uint8_t *pv = packed_values + (size_t)k * val_stride;
        float w = weights[k];
        if (w == 0.0f) continue;

        // Unpack indices
        if (tq_read_version(pv) != BN_TQ_FORMAT_VERSION) continue;
        unpack_indices(pv + TQ_HEADER_BYTES, d, st->bits, indices);

        // Read vec_norm
        uint16_t vec_norm_fp16;
        size_t norm_off = (size_t)TQ_HEADER_BYTES + (size_t)idx_sz;
        memcpy(&vec_norm_fp16, pv + norm_off, 2);
        float val_norm = tq_fp16_to_fp32(vec_norm_fp16);

        // Build rotated vector from centroids, then inverse-rotate
        for (int i = 0; i < d; i++)
            rotated[i] = st->tables.centroids[indices[i]];
        tq_apply_runtime_clip(rotated, d, rt->clip_threshold);
        rht_inverse(st, rotated, dequant, d);

        // Accumulate: out += w * val_norm * dequant
        float wn = w * val_norm;
#ifdef __ARM_NEON
        {
            float32x4_t wv = vdupq_n_f32(wn);
            for (int i = 0; i < d; i += 4)
                vst1q_f32(out + i, vfmaq_f32(vld1q_f32(out + i), wv, vld1q_f32(dequant + i)));
        }
#else
        for (int i = 0; i < d; i++)
            out[i] += wn * dequant[i];
#endif
    }
}

void bn_tq_attention_combine(const BnTQState *st, const uint8_t *packed_values,
                               int n_keys, int val_stride,
                               const float *weights, float *out) {
    bn_tq_attention_combine_head(st, 0, packed_values, n_keys, val_stride, weights, out, 0);
}

// --- Precomputed QJL API (avoids redundant per-key projection) ---

void bn_tq_qjl_precompute(const BnTQState *st, const float *rotated_q,
                            uint8_t *q_signs_out) {
    bn_tq_qjl_precompute_head(st, 0, rotated_q, q_signs_out);
}

void bn_tq_qjl_precompute_head(const BnTQState *st, int head_idx,
                               const float *rotated_q, uint8_t *q_signs_out) {
    if (!st || !rotated_q || !q_signs_out) return;
    (void)head_idx;
    qjl_project_signs(st, rotated_q, q_signs_out, st->head_dim);
}

float bn_tq_score_key_precomputed_head(const BnTQState *st, int head_idx, const float *rotated_q,
                                       const uint8_t *q_signs, const uint8_t *packed_key) {
    if (!st) return 0.0f;
    (void)head_idx;
    int d = st->head_dim;
    if (d <= 0 || d > BN_MAX_VLA_ELEMS) return 0.0f;
    int idx_sz = index_bytes(d, st->bits);
    int qjl_sz = d / 8;
    if (tq_read_version(packed_key) != BN_TQ_FORMAT_VERSION) return 0.0f;
    uint8_t strategy = tq_read_strategy(packed_key);
    if (!tq_strategy_valid(strategy)) return 0.0f;
    float qjl_weight = tq_qjl_weight_for_strategy(strategy);

    int indices[d];
    unpack_indices(packed_key + TQ_HEADER_BYTES, d, st->bits, indices);

    uint16_t res_norm_fp16, vec_norm_fp16;
    memcpy(&res_norm_fp16, packed_key + TQ_HEADER_BYTES + idx_sz + qjl_sz, 2);
    memcpy(&vec_norm_fp16, packed_key + TQ_HEADER_BYTES + idx_sz + qjl_sz + 2, 2);
    float res_norm = tq_fp16_to_fp32(res_norm_fp16);
    float vec_norm = tq_fp16_to_fp32(vec_norm_fp16);

    // Centroid dot product
#ifdef __ARM_NEON
    float c_tmp[4];
    float32x4_t acc = vdupq_n_f32(0);
    for (int i = 0; i < d; i += 4) {
        c_tmp[0] = st->tables.centroids[indices[i]];
        c_tmp[1] = st->tables.centroids[indices[i+1]];
        c_tmp[2] = st->tables.centroids[indices[i+2]];
        c_tmp[3] = st->tables.centroids[indices[i+3]];
        acc = vfmaq_f32(acc, vld1q_f32(rotated_q + i), vld1q_f32(c_tmp));
    }
    float centroid_dot = tq_neon_hsum(acc);
#else
    float centroid_dot = 0.0f;
    for (int i = 0; i < d; i++)
        centroid_dot += rotated_q[i] * st->tables.centroids[indices[i]];
#endif

    // XNOR popcount
    const uint8_t *key_signs = packed_key + TQ_HEADER_BYTES + idx_sz;
    int agree = 0;
#ifdef __ARM_NEON
    if (qjl_weight != 0.0f) {
        int b = 0;
        for (; b + 16 <= qjl_sz; b += 16) {
            uint8x16_t xnor = vmvnq_u8(veorq_u8(vld1q_u8(q_signs + b),
                                                   vld1q_u8(key_signs + b)));
            uint8x16_t cnt = vcntq_u8(xnor);
            uint16x8_t p1 = vpaddlq_u8(cnt);
            uint32x4_t p2 = vpaddlq_u16(p1);
            uint64x2_t p3 = vpaddlq_u32(p2);
            agree += (int)(vgetq_lane_u64(p3, 0) + vgetq_lane_u64(p3, 1));
        }
        for (; b < qjl_sz; b++) {
            uint8_t xnor = ~(q_signs[b] ^ key_signs[b]);
            uint8_t v = xnor;
            v = (v & 0x55) + ((v >> 1) & 0x55);
            v = (v & 0x33) + ((v >> 2) & 0x33);
            v = (v & 0x0F) + ((v >> 4) & 0x0F);
            agree += v;
        }
    }
#else
    if (qjl_weight != 0.0f) {
        for (int b = 0; b < qjl_sz; b++) {
            uint8_t xnor = ~(q_signs[b] ^ key_signs[b]);
            uint8_t v = xnor;
            v = (v & 0x55) + ((v >> 1) & 0x55);
            v = (v & 0x33) + ((v >> 2) & 0x33);
            v = (v & 0x0F) + ((v >> 4) & 0x0F);
            agree += v;
        }
    }
#endif

    float qjl_scale = sqrtf(3.14159265f / 2.0f) / (float)d;
    float qjl_correction = qjl_weight * (float)(2 * agree - d) * res_norm * qjl_scale;
    return vec_norm * (centroid_dot + qjl_correction);
}

float bn_tq_score_key_precomputed(const BnTQState *st, const float *rotated_q,
                                    const uint8_t *q_signs, const uint8_t *packed_key) {
    return bn_tq_score_key_precomputed_head(st, 0, rotated_q, q_signs, packed_key);
}

int bn_tq_calibrate_head(BnTQState *state, int head_idx,
                         const float *queries, const float *keys,
                         const float *values, int n_samples, int stride) {
    BnTQHeadCalibration *cal;
    BnTQHeadRuntime *rt;
    if (!state || !queries || !keys || !values || n_samples <= 0) return -1;
    if (stride <= 0) stride = state->head_dim;
    if (bn_tq_configure_heads(state, state->n_heads > 0 ? state->n_heads : 1) != 0) return -1;
    cal = tq_calibration_head_mut(state, head_idx);
    rt = tq_runtime_head_mut(state, head_idx);
    if (!cal || !rt) return -1;
    memset(cal, 0, sizeof(*cal));

    int d = state->head_dim;
    int key_bytes = bn_tq_key_bytes(state);
    float threshold = TQ_OUTLIER_CLIP_FACTOR * state->rht_scale;
    float q_rot[d];
    float k_rot[d];
    float *exact_scores = (float *)malloc((size_t)n_samples * sizeof(float));
    float *approx_scores = (float *)malloc((size_t)n_samples * sizeof(float));
    uint8_t *packed_keys = (uint8_t *)malloc((size_t)n_samples * (size_t)key_bytes);
    if (!exact_scores || !approx_scores || !packed_keys) {
        free(exact_scores);
        free(approx_scores);
        free(packed_keys);
        return -1;
    }
    bn_tq_rotate_query_head(state, head_idx, queries, q_rot);

    for (int i = 0; i < n_samples; i++) {
        const float *k = keys + (size_t)i * stride;
        const float *v = values + (size_t)i * stride;
        float kn = 0.0f, vn = 0.0f;
        bn_tq_quantize_key_head(state, head_idx, k, packed_keys + (size_t)i * key_bytes);
        for (int j = 0; j < d; j++) {
            kn += k[j] * k[j];
            vn += v[j] * v[j];
        }
        bn_tq_rotate_query_head(state, head_idx, k, k_rot);

        float max_abs = 0.0f;
        float mean_abs = 0.0f;
        float outliers = 0.0f;
        for (int j = 0; j < d; j++) {
            float a = fabsf(k_rot[j]);
            if (a > max_abs) max_abs = a;
            mean_abs += a;
            if (a > threshold) outliers += 1.0f;
        }
        cal->variance_ratio_sum += max_abs / (mean_abs / (float)d + TQ_EPSILON);
        cal->outlier_ratio_sum += outliers / (float)d;
        cal->residual_norm_sum += fabsf(sqrtf(kn) - sqrtf(vn));
        exact_scores[i] = 0.0f;
        for (int j = 0; j < d; j++)
            exact_scores[i] += queries[j] * k[j];
    }

    bn_tq_attention_scores(state, q_rot, packed_keys, n_samples, key_bytes, approx_scores);
    int best_exact = 0, best_approx = 0;
    for (int i = 0; i < n_samples; i++) {
        float denom = fabsf(exact_scores[i]) + TQ_EPSILON;
        cal->score_error_sum += fabsf(exact_scores[i] - approx_scores[i]) / denom;
        if (exact_scores[i] > exact_scores[best_exact]) best_exact = i;
        if (approx_scores[i] > approx_scores[best_approx]) best_approx = i;
    }
    cal->top1_agreement_sum = (best_exact == best_approx) ? 1.0f : 0.0f;
    cal->samples = n_samples;

    float variance_ratio = cal->variance_ratio_sum / (float)n_samples;
    float outlier_ratio = cal->outlier_ratio_sum / (float)n_samples;
    float score_error = cal->score_error_sum / (float)n_samples;
    if (score_error > TQ_CONSERVATIVE_ERROR_THRESHOLD) {
        rt->strategy = BN_TQ_STRATEGY_CONSERVATIVE;
        rt->clip_threshold = TQ_CONSERVATIVE_CLIP_FACTOR * state->rht_scale;
        rt->qjl_weight = 0.0f;
    } else if (outlier_ratio > TQ_OUTLIER_RATIO_THRESHOLD || variance_ratio > TQ_OUTLIER_VARIANCE_THRESHOLD) {
        rt->strategy = BN_TQ_STRATEGY_OUTLIER;
        rt->clip_threshold = TQ_OUTLIER_CLIP_FACTOR * state->rht_scale;
        rt->qjl_weight = 1.0f;
    } else if (score_error > TQ_CALIBRATED_ERROR_THRESHOLD) {
        rt->strategy = BN_TQ_STRATEGY_CALIBRATED;
        rt->clip_threshold = 0.0f;
        rt->qjl_weight = TQ_CALIBRATED_QJL_WEIGHT;
    } else {
        rt->strategy = BN_TQ_STRATEGY_BASELINE;
        rt->clip_threshold = 0.0f;
        rt->qjl_weight = 1.0f;
    }
    free(exact_scores);
    free(approx_scores);
    free(packed_keys);
    return 0;
}
