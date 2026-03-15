#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static void run_i2s_matvec(float *out, const BnQWeight *W, const float *x,
                            int8_t *x_q) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float x_scale = bn_quant_x_to_i8(x, x_q, W->cols);
    BnI2SCtx ctx = { out, W, x_q, W->scale * x_scale };
    bn_quant_i2s_neon_sdot_range(&ctx, 0, W->rows);
#elif defined(__ARM_NEON)
    (void)x_q;
    BnI2SFloatCtx ctx = { out, W, x };
    bn_quant_i2s_neon_range(&ctx, 0, W->rows);
#elif defined(__AVX2__)
    float x_scale = bn_quant_x_to_i8(x, x_q, W->cols);
    BnI2SCtx ctx = { out, W, x_q, W->scale * x_scale };
    bn_quant_i2s_avx2_range(&ctx, 0, W->rows);
#else
    (void)x_q;
    BnI2SFloatCtx ctx = { out, W, x };
    bn_quant_i2s_scalar_range(&ctx, 0, W->rows);
#endif
}

static void test_i2s_matvec(void) {
    printf("test_i2s_matvec... ");

    // Create I2_S weight matrix: 4 rows x 256 cols
    int rows = 4, cols = 256;
    int row_bytes = cols / 4;
    size_t data_size = (size_t)rows * row_bytes + sizeof(float);
    uint8_t *data = (uint8_t *)calloc(data_size, 1);

    // Fill: alternating -1, +1, 0, 0
    // byte = (0<<6) | (2<<4) | (1<<2) | (1<<0) = 0x25
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < row_bytes; b++) {
            data[r * row_bytes + b] = 0x25;
        }
    }

    float tensor_scale = 0.5f;
    memcpy(data + (size_t)rows * row_bytes, &tensor_scale, sizeof(float));

    BnQWeight W = { data, BN_GGUF_TENSOR_I2_S, rows, cols, tensor_scale };

    float x[256];
    for (int i = 0; i < cols; i++) x[i] = 0.1f * (i % 17) - 0.8f;

    float out[4];
    int8_t x_q[256];
    run_i2s_matvec(out, &W, x, x_q);

    // Compute expected values via dequantization
    float weights[4][256];
    for (int r = 0; r < rows; r++) {
        bn_quant_dequant_i2s(data + (size_t)r * row_bytes, weights[r], cols, tensor_scale);
    }
    for (int r = 0; r < rows; r++) {
        float expected = 0.0f;
        for (int c = 0; c < cols; c++) expected += weights[r][c] * x[c];
        float err = fabsf(out[r] - expected);
        float mag = fabsf(expected) + 1e-6f;
        assert(err / mag < 0.05f);
    }

    free(data);
    printf("PASSED\n");
}

static void test_i2s_matvec_batch(void) {
    printf("test_i2s_matvec_batch... ");

    // 4 rows x 256 cols, split into 2 sub-matrices of 2 rows each
    int rows = 4, cols = 256;
    int row_bytes = cols / 4;
    size_t data_size = (size_t)rows * row_bytes + sizeof(float);
    uint8_t *data = (uint8_t *)calloc(data_size, 1);

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < row_bytes; b++) {
            data[r * row_bytes + b] = 0x25;
        }
    }

    float tensor_scale = 0.5f;
    memcpy(data + (size_t)rows * row_bytes, &tensor_scale, sizeof(float));

    float x[256];
    for (int i = 0; i < cols; i++) x[i] = 0.1f * (i % 17) - 0.8f;

    // Full call
    BnQWeight W_full = { data, BN_GGUF_TENSOR_I2_S, rows, cols, tensor_scale };
    float ref[4];
    int8_t x_q[256];
    run_i2s_matvec(ref, &W_full, x, x_q);

    // Split: 2 sub-matrices
    size_t half_data = (size_t)2 * row_bytes;
    BnQWeight W1 = { data, BN_GGUF_TENSOR_I2_S, 2, cols, tensor_scale };
    BnQWeight W2 = { data + half_data, BN_GGUF_TENSOR_I2_S, 2, cols, tensor_scale };

    float out1[2], out2[2];
    run_i2s_matvec(out1, &W1, x, x_q);
    run_i2s_matvec(out2, &W2, x, x_q);

    for (int i = 0; i < 2; i++) {
        float err = fabsf(out1[i] - ref[i]);
        float mag = fabsf(ref[i]) + 1e-6f;
        assert(err / mag < 0.02f);
    }
    for (int i = 0; i < 2; i++) {
        float err = fabsf(out2[i] - ref[2 + i]);
        float mag = fabsf(ref[2 + i]) + 1e-6f;
        assert(err / mag < 0.02f);
    }

    free(data);
    printf("PASSED\n");
}

int main(void) {
    printf("=== I2_S Tests ===\n");
    test_i2s_matvec();
    test_i2s_matvec_batch();
    printf("All I2_S tests passed!\n");
    return 0;
}
