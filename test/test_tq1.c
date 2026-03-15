#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static void run_tq1_matvec(float *out, const BnQWeight *W, const float *x,
                            int8_t *x_q) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float x_scale = bn_quant_x_to_i8(x, x_q, W->cols);
    BnTQ1SdotCtx ctx = { out, W, x_q, W->scale * x_scale };
    bn_quant_tq1_neon_sdot_range(&ctx, 0, W->rows);
#elif defined(__ARM_NEON)
    (void)x_q;
    BnTQ1Ctx ctx = { out, W, x };
    bn_quant_tq1_neon_range(&ctx, 0, W->rows);
#else
    (void)x_q;
    BnTQ1Ctx ctx = { out, W, x };
    bn_quant_tq1_scalar_range(&ctx, 0, W->rows);
#endif
}

static void test_tq1_dequant(void) {
    printf("test_tq1_dequant... ");

    BnBlockTQ1 block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;
    block.qs[0] = 0;
    block.qs[1] = 127;

    float out[256];
    bn_quant_dequant_tq1(&block, out);

    assert(fabsf(out[0] - (-1.0f)) < 1e-6f);

    printf("PASSED\n");
}

static void test_tq1_matvec(void) {
    printf("test_tq1_matvec... ");

    // 2 rows x 256 cols
    int n_blocks = 2;
    BnBlockTQ1 *blocks = (BnBlockTQ1 *)calloc(n_blocks, sizeof(BnBlockTQ1));

    // Row 0: all qs=0, all qh=0 → all values = -1, scale = 1.0
    blocks[0].d = 0x3C00;

    // Row 1: varied pattern
    for (int i = 0; i < 48; i++) blocks[1].qs[i] = (uint8_t)(i * 5 + 3);
    for (int i = 0; i < 4; i++) blocks[1].qh[i] = (uint8_t)(i * 17);
    blocks[1].d = 0x3800;  // scale = 0.5

    BnQWeight W = { blocks, BN_GGUF_TENSOR_TQ1_0, 2, 256, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 0.1f * (i % 11) - 0.5f;

    // Compute expected via dequantization
    float expected[2] = {0};
    for (int r = 0; r < 2; r++) {
        float dequant[256];
        bn_quant_dequant_tq1(&blocks[r], dequant);
        float d = bn_fp16_to_fp32(blocks[r].d);
        // The range function applies tensor_scale (W.scale) at the end
        // but dequant already includes block d, so expected = sum(dequant[i]*x[i]) * tensor_scale / d * d
        // Actually: range function computes sum(decoded_weight * x) * block_d * tensor_scale
        // dequant gives: decoded_weight * block_d
        // So expected = sum(dequant[i] * x[i]) * tensor_scale
        for (int c = 0; c < 256; c++) expected[r] += dequant[c] * x[c];
        expected[r] *= W.scale;
        (void)d;
    }

    float out[2];
    int8_t x_q[256];
    run_tq1_matvec(out, &W, x, x_q);

    for (int r = 0; r < 2; r++) {
        float err = fabsf(out[r] - expected[r]);
        float mag = fabsf(expected[r]) + 1e-6f;
        assert(err / mag < 0.05f);
    }

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== TQ1 Tests ===\n");
    test_tq1_dequant();
    test_tq1_matvec();
    printf("All TQ1 tests passed!\n");
    return 0;
}
