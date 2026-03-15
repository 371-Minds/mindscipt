#include "quant.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

static void test_fp16_conversion(void) {
    printf("test_fp16_conversion... ");

    // Test zero
    assert(bn_fp16_to_fp32(0x0000) == 0.0f);

    // Test 1.0 (FP16: 0x3C00)
    float one = bn_fp16_to_fp32(0x3C00);
    assert(fabsf(one - 1.0f) < 1e-6f);

    // Test -1.0 (FP16: 0xBC00)
    float neg_one = bn_fp16_to_fp32(0xBC00);
    assert(fabsf(neg_one - (-1.0f)) < 1e-6f);

    // Test 0.5 (FP16: 0x3800)
    float half = bn_fp16_to_fp32(0x3800);
    assert(fabsf(half - 0.5f) < 1e-6f);

    // Round-trip test
    float test_vals[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 0.001f};
    for (int i = 0; i < 7; i++) {
        uint16_t h = bn_fp32_to_fp16(test_vals[i]);
        float back = bn_fp16_to_fp32(h);
        float err = fabsf(back - test_vals[i]);
        // FP16 has limited precision, allow some error for small values
        assert(err < 0.01f || (test_vals[i] != 0 && err / fabsf(test_vals[i]) < 0.01f));
    }

    printf("PASSED\n");
}

int main(void) {
    printf("=== FP16 Tests ===\n");
    test_fp16_conversion();
    printf("All FP16 tests passed!\n");
    return 0;
}
