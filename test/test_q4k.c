#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __ARM_NEON
#define Q4K_RANGE bn_quant_q4k_neon_range
#elif defined(__AVX2__)
#define Q4K_RANGE bn_quant_q4k_avx2_range
#else
#define Q4K_RANGE bn_quant_q4k_scalar_range
#endif

static void test_q4k_dequant(void) {
    printf("test_q4k_dequant... ");

    BnBlockQ4K block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;
    block.dmin = 0x3800;

    block.scales[0] = 2;
    block.scales[4] = 1;

    block.qs[0] = 0x53;

    float out[256];
    bn_quant_dequant_q4k(&block, out);

    assert(fabsf(out[0] - 5.5f) < 0.01f);
    assert(fabsf(out[32]) < 0.01f);

    printf("PASSED\n");
}

static void test_q4k_matvec(void) {
    printf("test_q4k_matvec... ");

    int rows = 2, cols = 256;
    BnBlockQ4K *blocks = (BnBlockQ4K *)calloc(rows, sizeof(BnBlockQ4K));

    blocks[0].d = 0x3C00;
    blocks[0].dmin = 0x0000;
    for (int j = 0; j < 8; j++) {
        if (j < 4) {
            blocks[0].scales[j] = 1;
            blocks[0].scales[j + 4] = 0;
        } else {
            blocks[0].scales[j + 4] = 1;
        }
    }
    for (int i = 0; i < 128; i++) blocks[0].qs[i] = 0x22;

    blocks[1].d = 0x3800;
    blocks[1].dmin = 0x3400;
    for (int j = 0; j < 4; j++) {
        blocks[1].scales[j] = 1;
        blocks[1].scales[j + 4] = 1;
    }

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q4_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    BnQ4KCtx ctx = { out, &W, x };
    Q4K_RANGE(&ctx, 0, rows);

    assert(fabsf(out[0] - 512.0f) < 1.0f);
    assert(fabsf(out[1] - (-32.0f)) < 1.0f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Q4_K Tests ===\n");
    test_q4k_dequant();
    test_q4k_matvec();
    printf("All Q4_K tests passed!\n");
    return 0;
}
