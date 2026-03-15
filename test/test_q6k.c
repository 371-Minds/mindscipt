#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __ARM_NEON
#define Q6K_RANGE bn_quant_q6k_neon_range
#elif defined(__AVX2__)
#define Q6K_RANGE bn_quant_q6k_avx2_range
#else
#define Q6K_RANGE bn_quant_q6k_scalar_range
#endif

static void test_q6k_dequant(void) {
    printf("test_q6k_dequant... ");

    BnBlockQ6K block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;

    for (int i = 0; i < 16; i++) block.scales[i] = 1;

    float out[256];
    bn_quant_dequant_q6k(&block, out);
    for (int i = 0; i < 256; i++) {
        assert(fabsf(out[i] - (-32.0f)) < 0.01f);
    }

    // Test with known values
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;
    for (int i = 0; i < 16; i++) block.scales[i] = 2;
    block.ql[0] = 0x35;
    block.ql[32] = 0x72;
    block.qh[0] = 0xC9;

    bn_quant_dequant_q6k(&block, out);

    assert(fabsf(out[0] - (-22.0f)) < 0.01f);
    assert(fabsf(out[32] - 4.0f) < 0.01f);
    assert(fabsf(out[64] - (-58.0f)) < 0.01f);
    assert(fabsf(out[96] - 46.0f) < 0.01f);

    printf("PASSED\n");
}

static void test_q6k_matvec(void) {
    printf("test_q6k_matvec... ");

    int rows = 2, cols = 256;
    BnBlockQ6K *blocks = (BnBlockQ6K *)calloc(rows, sizeof(BnBlockQ6K));

    blocks[0].d = 0x3C00;
    for (int i = 0; i < 16; i++) blocks[0].scales[i] = 1;

    blocks[1].d = 0x3800;
    for (int i = 0; i < 16; i++) blocks[1].scales[i] = 2;
    for (int i = 0; i < 128; i++) blocks[1].ql[i] = 0x88;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q6_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    BnQ6KCtx ctx = { out, &W, x };
    Q6K_RANGE(&ctx, 0, rows);

    assert(fabsf(out[0] - (-8192.0f)) < 1.0f);
    assert(fabsf(out[1] - (-6144.0f)) < 1.0f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Q6_K Tests ===\n");
    test_q6k_dequant();
    test_q6k_matvec();
    printf("All Q6_K tests passed!\n");
    return 0;
}
