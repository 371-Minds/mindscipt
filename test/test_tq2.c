#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __ARM_NEON
#define TQ2_RANGE bn_quant_tq2_neon_range
#else
#define TQ2_RANGE bn_quant_tq2_scalar_range
#endif

static void test_tq2_dequant(void) {
    printf("test_tq2_dequant... ");

    BnBlockTQ2 block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;
    block.qs[0] = 0x92;
    for (int i = 1; i < 64; i++) block.qs[i] = 0x55;

    float out[256];
    bn_quant_dequant_tq2(&block, out);

    assert(fabsf(out[0] - 1.0f) < 1e-6f);
    assert(fabsf(out[1] - 0.0f) < 1e-6f);
    assert(fabsf(out[32] - (-1.0f)) < 1e-6f);
    assert(fabsf(out[64] - 0.0f) < 1e-6f);
    assert(fabsf(out[96] - 1.0f) < 1e-6f);

    printf("PASSED\n");
}

static void test_tq2_matvec(void) {
    printf("test_tq2_matvec... ");

    int n_blocks = 2;
    BnBlockTQ2 *blocks = (BnBlockTQ2 *)calloc(n_blocks, sizeof(BnBlockTQ2));

    // Row 0: all +1
    for (int i = 0; i < 64; i++) blocks[0].qs[i] = 0xAA;
    blocks[0].d = 0x3C00;

    // Row 1: all 0
    for (int i = 0; i < 64; i++) blocks[1].qs[i] = 0x55;
    blocks[1].d = 0x3C00;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_TQ2_0, 2, 256, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    BnTQ2Ctx ctx = { out, &W, x };
    TQ2_RANGE(&ctx, 0, 2);

    assert(fabsf(out[0] - 256.0f) < 1e-3f);
    assert(fabsf(out[1] - 0.0f) < 1e-3f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== TQ2 Tests ===\n");
    test_tq2_dequant();
    test_tq2_matvec();
    printf("All TQ2 tests passed!\n");
    return 0;
}
