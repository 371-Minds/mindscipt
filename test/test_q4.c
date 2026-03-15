#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static void run_q4_matvec(float *out, const BnQWeight *W, const float *x,
                           int8_t *x_q) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int n_blocks = W->cols / 32;
    float x_scales[n_blocks];
    bn_quant_x_to_q8_blocks(x, x_q, x_scales, W->cols);
    BnQ4SdotCtx ctx = { out, W, x_q, x_scales };
    bn_quant_q4_neon_sdot_range(&ctx, 0, W->rows);
#elif defined(__AVX2__)
    int n_blocks = W->cols / 32;
    float x_scales[n_blocks];
    bn_quant_x_to_q8_blocks(x, x_q, x_scales, W->cols);
    BnQ4SdotCtx ctx = { out, W, x_q, x_scales };
    bn_quant_q4_avx2_range(&ctx, 0, W->rows);
#else
    (void)x_q;
    BnQ4Ctx ctx = { out, W, x };
    bn_quant_q4_scalar_range(&ctx, 0, W->rows);
#endif
}

static void test_q4_matvec(void) {
    printf("test_q4_matvec... ");

    int rows = 2, cols = 32;
    BnBlockQ4_0 *blocks = (BnBlockQ4_0 *)calloc(rows, sizeof(BnBlockQ4_0));

    // Row 0: all nibbles = 10 → dequant = 10-8 = +2
    blocks[0].d = 0x3C00;
    for (int i = 0; i < 16; i++) blocks[0].qs[i] = 0xAA;

    // Row 1: lo=12(+4), hi=4(-4)
    blocks[1].d = 0x3800;
    for (int i = 0; i < 16; i++) blocks[1].qs[i] = 0x4C;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.0f };

    float x[32];
    for (int i = 0; i < 32; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[32];
    run_q4_matvec(out, &W, x, x_q);

    assert(fabsf(out[0] - 64.0f) < 0.1f);
    assert(fabsf(out[1] - 0.0f) < 0.1f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Q4_0 Tests ===\n");
    test_q4_matvec();
    printf("All Q4_0 tests passed!\n");
    return 0;
}
