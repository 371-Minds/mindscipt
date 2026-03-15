#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __ARM_NEON
#define Q8_RANGE bn_quant_q8_neon_range
#elif defined(__AVX2__)
#define Q8_RANGE bn_quant_q8_avx2_range
#else
#define Q8_RANGE bn_quant_q8_scalar_range
#endif

static void test_q8_matvec(void) {
    printf("test_q8_matvec... ");

    int rows = 2, cols = 32;
    BnBlockQ8_0 *blocks = (BnBlockQ8_0 *)calloc(rows, sizeof(BnBlockQ8_0));

    blocks[0].d = 0x3C00;
    for (int i = 0; i < 32; i++) blocks[0].qs[i] = 1;

    blocks[1].d = 0x3800;
    for (int i = 0; i < 32; i++) blocks[1].qs[i] = (i % 2 == 0) ? 2 : -2;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };

    float x[32];
    for (int i = 0; i < 32; i++) x[i] = 1.0f;

    float out[2];
    BnQ8Ctx ctx = { out, &W, x };
    Q8_RANGE(&ctx, 0, rows);

    assert(fabsf(out[0] - 32.0f) < 0.1f);
    assert(fabsf(out[1] - 0.0f) < 0.1f);

    free(blocks);
    printf("PASSED\n");
}

static void test_q8_matvec_multiblock(void) {
    printf("test_q8_matvec_multiblock... ");

    int rows = 2, cols = 64;
    int n_blocks = rows * 2;
    BnBlockQ8_0 *blocks = (BnBlockQ8_0 *)calloc(n_blocks, sizeof(BnBlockQ8_0));

    blocks[0].d = 0x3C00;
    for (int i = 0; i < 32; i++) blocks[0].qs[i] = 3;

    blocks[1].d = 0x4000;
    for (int i = 0; i < 32; i++) blocks[1].qs[i] = 1;

    blocks[2].d = 0x3C00;
    for (int i = 0; i < 32; i++) blocks[2].qs[i] = -1;

    blocks[3].d = 0x3C00;
    for (int i = 0; i < 32; i++) blocks[3].qs[i] = 2;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };

    float x[64];
    for (int i = 0; i < 64; i++) x[i] = 1.0f;

    float out[2];
    BnQ8Ctx ctx = { out, &W, x };
    Q8_RANGE(&ctx, 0, rows);

    assert(fabsf(out[0] - 160.0f) < 0.1f);
    assert(fabsf(out[1] - 32.0f) < 0.1f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Q8_0 Tests ===\n");
    test_q8_matvec();
    test_q8_matvec_multiblock();
    printf("All Q8_0 tests passed!\n");
    return 0;
}
