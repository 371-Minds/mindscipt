#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building WASM benchmark..."

emcc \
    "$PROJECT_DIR/bench/bench_kernels.c" \
    "$PROJECT_DIR/src/gguf.c" \
    "$PROJECT_DIR/src/quant/fp16.c" \
    "$PROJECT_DIR/src/quant/dequant.c" \
    "$PROJECT_DIR/src/quant/dispatch.c" \
    "$PROJECT_DIR/src/quant/x_quant_wasm.c" \
    "$PROJECT_DIR/src/quant/i2s_wasm.c" \
    "$PROJECT_DIR/src/quant/i2s_scalar.c" \
    "$PROJECT_DIR/src/quant/tq2_scalar.c" \
    "$PROJECT_DIR/src/quant/tq1_scalar.c" \
    "$PROJECT_DIR/src/quant/q8_wasm.c" \
    "$PROJECT_DIR/src/quant/q8_scalar.c" \
    "$PROJECT_DIR/src/quant/q4_wasm.c" \
    "$PROJECT_DIR/src/quant/q4_scalar.c" \
    "$PROJECT_DIR/src/quant/q6k_wasm.c" \
    "$PROJECT_DIR/src/quant/q6k_scalar.c" \
    "$PROJECT_DIR/src/quant/q8k_wasm.c" \
    "$PROJECT_DIR/src/quant/q8k_scalar.c" \
    "$PROJECT_DIR/src/quant/q4k_wasm.c" \
    "$PROJECT_DIR/src/quant/q4k_scalar.c" \
    "$PROJECT_DIR/src/quant/q5k_wasm.c" \
    "$PROJECT_DIR/src/quant/q5k_scalar.c" \
    "$PROJECT_DIR/src/quant/q3k_wasm.c" \
    "$PROJECT_DIR/src/quant/q3k_scalar.c" \
    "$PROJECT_DIR/src/quant/q2k_wasm.c" \
    "$PROJECT_DIR/src/quant/q2k_scalar.c" \
    "$PROJECT_DIR/src/quant/q4_1_wasm.c" \
    "$PROJECT_DIR/src/quant/q4_1_scalar.c" \
    "$PROJECT_DIR/src/quant/bf16_wasm.c" \
    "$PROJECT_DIR/src/quant/bf16_scalar.c" \
    "$PROJECT_DIR/src/quant/iq4nl_wasm.c" \
    "$PROJECT_DIR/src/quant/iq4nl_scalar.c" \
    "$PROJECT_DIR/src/quant/iq4xs_wasm.c" \
    "$PROJECT_DIR/src/quant/iq4xs_scalar.c" \
    "$PROJECT_DIR/src/quant/iq3xxs_wasm.c" \
    "$PROJECT_DIR/src/quant/iq3xxs_scalar.c" \
    "$PROJECT_DIR/src/quant/iq3s_wasm.c" \
    "$PROJECT_DIR/src/quant/iq3s_scalar.c" \
    "$PROJECT_DIR/src/quant/iq2xxs_wasm.c" \
    "$PROJECT_DIR/src/quant/iq2xxs_scalar.c" \
    "$PROJECT_DIR/src/quant/iq2xs_wasm.c" \
    "$PROJECT_DIR/src/quant/iq2xs_scalar.c" \
    "$PROJECT_DIR/src/quant/iq2s_wasm.c" \
    "$PROJECT_DIR/src/quant/iq2s_scalar.c" \
    "$PROJECT_DIR/src/model.c" \
    "$PROJECT_DIR/src/transformer.c" \
    "$PROJECT_DIR/src/transformer/rmsnorm_wasm.c" \
    "$PROJECT_DIR/src/transformer/rmsnorm_scalar.c" \
    "$PROJECT_DIR/src/transformer/gqa_wasm.c" \
    "$PROJECT_DIR/src/transformer/gqa_scalar.c" \
    "$PROJECT_DIR/src/transformer/logits_wasm.c" \
    "$PROJECT_DIR/src/transformer/logits_scalar.c" \
    "$PROJECT_DIR/src/tokenizer.c" \
    "$PROJECT_DIR/src/sampler.c" \
    "$PROJECT_DIR/src/platform.c" \
    "$PROJECT_DIR/src/threadpool.c" \
    "$PROJECT_DIR/src/sh_arena.c" \
    "$PROJECT_DIR/src/sh_log.c" \
    -I"$PROJECT_DIR/include" \
    -std=c11 -D_GNU_SOURCE -O3 -flto -msimd128 -mrelaxed-simd \
    -sALLOW_MEMORY_GROWTH=1 \
    -sMAXIMUM_MEMORY=4294967296 \
    -sNODERAWFS=1 \
    -sENVIRONMENT=node \
    -sEXIT_RUNTIME=1 \
    -o "$SCRIPT_DIR/bench_wasm.js"

echo "WASM benchmark built: bench/bench_wasm.js"
echo "Usage: node --experimental-wasm-relaxed-simd bench/bench_wasm.js model.gguf [--iters N]"
