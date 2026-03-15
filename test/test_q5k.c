#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __ARM_NEON
#define Q5K_RANGE bn_quant_q5k_neon_range
#elif defined(__AVX2__)
#define Q5K_RANGE bn_quant_q5k_avx2_range
#else
#define Q5K_RANGE bn_quant_q5k_scalar_range
#endif

static void test_q5k_dequant(void) {
    printf("test_q5k_dequant... ");

    BnBlockQ5K block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;
    block.dmin = 0x3800;

    block.scales[0] = 1;
    block.scales[4] = 0;

    block.qs[0] = 0x07;
    block.qh[0] = 0x01;

    float out[256];
    bn_quant_dequant_q5k(&block, out);

    assert(fabsf(out[0] - 23.0f) < 0.01f);
    assert(fabsf(out[1]) < 0.01f);

    printf("PASSED\n");
}

static void test_q5k_matvec(void) {
    printf("test_q5k_matvec... ");

    int rows = 2, cols = 256;
    BnBlockQ5K *blocks = (BnBlockQ5K *)calloc(rows, sizeof(BnBlockQ5K));

    blocks[0].d = 0x3C00;
    blocks[0].dmin = 0x0000;
    for (int j = 0; j < 4; j++) blocks[0].scales[j] = 1;

    blocks[1].d = 0x3800;
    blocks[1].dmin = 0x0000;
    for (int j = 0; j < 4; j++) blocks[1].scales[j] = 2;
    for (int i = 0; i < 128; i++) blocks[1].qs[i] = 0x33;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    BnQ5KCtx ctx = { out, &W, x };
    Q5K_RANGE(&ctx, 0, rows);

    assert(fabsf(out[0]) < 1.0f);
    assert(fabsf(out[1] - 384.0f) < 1.0f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Q5_K Tests ===\n");
    test_q5k_dequant();
    test_q5k_matvec();
    printf("All Q5_K tests passed!\n");
    return 0;
}
