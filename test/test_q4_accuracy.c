#include "quant.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Compare Q4_0 matvec with dequantized float reference
// This catches any bugs in the NEON/AVX2/WASM SIMD kernel
static void test_q4_accuracy(void) {
    printf("test_q4_accuracy (random data, multi-block)... ");

    int rows = 16;
    int cols = 2048;  // matches Qwen 2.5 dim
    int n_blocks_per_row = cols / 32;
    int total_blocks = rows * n_blocks_per_row;

    BnBlockQ4_0 *blocks = (BnBlockQ4_0 *)calloc(total_blocks, sizeof(BnBlockQ4_0));
    float *x = (float *)calloc(cols, sizeof(float));

    // Fill with pseudo-random data
    unsigned int seed = 12345;
    for (int i = 0; i < total_blocks; i++) {
        // Random scale in [-2, 2]
        seed = seed * 1103515245 + 12345;
        float scale = ((float)(seed % 10000) / 2500.0f) - 2.0f;
        blocks[i].d = bn_fp32_to_fp16(scale);
        for (int j = 0; j < 16; j++) {
            seed = seed * 1103515245 + 12345;
            blocks[i].qs[j] = (uint8_t)(seed & 0xFF);
        }
    }
    for (int i = 0; i < cols; i++) {
        seed = seed * 1103515245 + 12345;
        x[i] = ((float)(seed % 10000) / 5000.0f) - 1.0f;
    }

    // Reference: dequantize and float dot product
    float *deq = (float *)calloc(rows * cols, sizeof(float));
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q4_0(&blocks[r * n_blocks_per_row + b],
                                   deq + r * cols + b * 32);
        }
    }

    float *ref = (float *)calloc(rows, sizeof(float));
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += deq[r * cols + c] * x[c];
        }
        ref[r] = sum;
    }

    // Kernel output
    float *out = (float *)calloc(rows, sizeof(float));
    int8_t *x_q = (int8_t *)calloc(cols, sizeof(int8_t));
    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.0f };
    bn_quant_matvec(out, &W, x, x_q, NULL);

    // Compare
    float max_err = 0.0f;
    for (int r = 0; r < rows; r++) {
        float err = fabsf(out[r] - ref[r]);
        float mag = fabsf(ref[r]) + 1e-6f;
        float rel_err = err / mag;
        if (rel_err > max_err) max_err = rel_err;
        if (rel_err > 0.001f) {
            printf("\nFAILED: row %d: kernel=%.6f ref=%.6f err=%.6f rel=%.4f%%\n",
                   r, out[r], ref[r], err, rel_err * 100.0f);
            assert(0);
        }
    }

    printf("PASSED (max_rel_err=%.6f%%)\n", max_err * 100.0f);

    free(blocks);
    free(x);
    free(deq);
    free(ref);
    free(out);
    free(x_q);
}

// Test Q6_K matvec accuracy against dequantized reference
static void test_q6k_accuracy(void) {
    printf("test_q6k_accuracy (random data, multi-block)... ");

    int rows = 8;
    int cols = 2048;
    int n_blocks_per_row = cols / 256;
    int total_blocks = rows * n_blocks_per_row;

    BnBlockQ6K *blocks = (BnBlockQ6K *)calloc(total_blocks, sizeof(BnBlockQ6K));
    float *x = (float *)calloc(cols, sizeof(float));

    unsigned int seed = 67890;
    for (int i = 0; i < total_blocks; i++) {
        seed = seed * 1103515245 + 12345;
        float scale = ((float)(seed % 10000) / 5000.0f) - 1.0f;
        blocks[i].d = bn_fp32_to_fp16(scale);
        for (int j = 0; j < 16; j++) {
            seed = seed * 1103515245 + 12345;
            blocks[i].scales[j] = (int8_t)(seed % 127);
        }
        for (int j = 0; j < 128; j++) {
            seed = seed * 1103515245 + 12345;
            blocks[i].ql[j] = (uint8_t)(seed & 0xFF);
        }
        for (int j = 0; j < 64; j++) {
            seed = seed * 1103515245 + 12345;
            blocks[i].qh[j] = (uint8_t)(seed & 0xFF);
        }
    }
    for (int i = 0; i < cols; i++) {
        seed = seed * 1103515245 + 12345;
        x[i] = ((float)(seed % 10000) / 5000.0f) - 1.0f;
    }

    // Reference: dequantize and float dot product
    float *deq = (float *)calloc(rows * cols, sizeof(float));
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q6k(&blocks[r * n_blocks_per_row + b],
                                  deq + r * cols + b * 256);
        }
    }

    float *ref = (float *)calloc(rows, sizeof(float));
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += deq[r * cols + c] * x[c];
        }
        ref[r] = sum;
    }

    // Kernel output
    float *out = (float *)calloc(rows, sizeof(float));
    int8_t *x_q = (int8_t *)calloc(cols, sizeof(int8_t));
    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q6_K, rows, cols, 1.0f };
    bn_quant_matvec(out, &W, x, x_q, NULL);

    float max_err = 0.0f;
    for (int r = 0; r < rows; r++) {
        float err = fabsf(out[r] - ref[r]);
        float mag = fabsf(ref[r]) + 1e-6f;
        float rel_err = err / mag;
        if (rel_err > max_err) max_err = rel_err;
        if (rel_err > 0.01f) {
            printf("\nFAILED: row %d: kernel=%.6f ref=%.6f err=%.6f rel=%.4f%%\n",
                   r, out[r], ref[r], err, rel_err * 100.0f);
            assert(0);
        }
    }

    printf("PASSED (max_rel_err=%.6f%%)\n", max_err * 100.0f);

    free(blocks);
    free(x);
    free(deq);
    free(ref);
    free(out);
    free(x_q);
}

int main(void) {
    printf("=== Q4/Q6K Accuracy Tests ===\n");
    test_q4_accuracy();
    test_q6k_accuracy();
    printf("All accuracy tests passed!\n");
    return 0;
}
