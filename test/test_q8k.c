#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __ARM_NEON
#define Q8K_RANGE bn_quant_q8k_neon_range
#elif defined(__AVX2__)
#define Q8K_RANGE bn_quant_q8k_avx2_range
#else
#define Q8K_RANGE bn_quant_q8k_scalar_range
#endif

static void test_q8k_dequant(void) {
    printf("test_q8k_dequant... ");

    BnBlockQ8K block;
    memset(&block, 0, sizeof(block));
    block.d = 0.5f;

    block.qs[0] = 10;
    block.qs[1] = -20;
    block.qs[127] = 127;
    block.qs[255] = -128;

    float out[256];
    bn_quant_dequant_q8k(&block, out);

    assert(fabsf(out[0] - (0.5f * 10)) < 1e-4f);
    assert(fabsf(out[1] - (0.5f * -20)) < 1e-4f);
    assert(fabsf(out[127] - (0.5f * 127)) < 1e-4f);
    assert(fabsf(out[255] - (0.5f * -128)) < 1e-4f);
    assert(fabsf(out[100]) < 1e-6f);

    printf("PASSED\n");
}

static void test_q8k_matvec(void) {
    printf("test_q8k_matvec... ");

    int rows = 2, cols = 256;
    BnBlockQ8K *blocks = (BnBlockQ8K *)calloc(rows, sizeof(BnBlockQ8K));

    blocks[0].d = 1.0f;
    for (int i = 0; i < 256; i++) blocks[0].qs[i] = 1;

    blocks[1].d = 0.5f;
    for (int i = 0; i < 256; i++) blocks[1].qs[i] = (i % 2 == 0) ? 2 : -2;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q8_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    BnQ8KCtx ctx = { out, &W, x };
    Q8K_RANGE(&ctx, 0, rows);

    assert(fabsf(out[0] - 256.0f) < 0.1f);
    assert(fabsf(out[1] - 0.0f) < 0.1f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Q8_K Tests ===\n");
    test_q8k_dequant();
    test_q8k_matvec();
    printf("All Q8_K tests passed!\n");
    return 0;
}
