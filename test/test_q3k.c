#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __ARM_NEON
#define Q3K_RANGE bn_quant_q3k_neon_range
#elif defined(__AVX2__)
#define Q3K_RANGE bn_quant_q3k_avx2_range
#else
#define Q3K_RANGE bn_quant_q3k_scalar_range
#endif

static void test_q3k_dequant(void) {
    printf("test_q3k_dequant... ");

    BnBlockQ3K block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;

    float out[256];
    bn_quant_dequant_q3k(&block, out);

    assert(fabsf(out[0] - 128.0f) < 0.01f);
    assert(fabsf(out[100] - 128.0f) < 0.01f);
    assert(fabsf(out[255] - 128.0f) < 0.01f);

    // Test with hmask bit set
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;
    block.hmask[0] = 0xFF;
    bn_quant_dequant_q3k(&block, out);
    assert(fabsf(out[0]) < 0.01f);

    printf("PASSED\n");
}

static void test_q3k_matvec(void) {
    printf("test_q3k_matvec... ");

    int rows = 2, cols = 256;
    BnBlockQ3K *blocks = (BnBlockQ3K *)calloc(rows, sizeof(BnBlockQ3K));

    // Row 0: hmask all set → q3 = 0 → val = 0
    blocks[0].d = 0x3C00;
    memset(blocks[0].hmask, 0xFF, 32);

    // Row 1: hmask=0 → q3 = -4, scale=-32 → val = 64, dot = 16384
    blocks[1].d = 0x3800;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q3_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    BnQ3KCtx ctx = { out, &W, x };
    Q3K_RANGE(&ctx, 0, rows);

    assert(fabsf(out[0]) < 1.0f);
    assert(fabsf(out[1] - 16384.0f) < 10.0f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Q3_K Tests ===\n");
    test_q3k_dequant();
    test_q3k_matvec();
    printf("All Q3_K tests passed!\n");
    return 0;
}
