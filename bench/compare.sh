#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 model.gguf [iters]"
    exit 1
fi

MODEL="$1"
ITERS="${2:-100}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Native benchmark ==="
"$PROJECT_DIR/bench_kernels" "$MODEL" --iters "$ITERS" | tee /tmp/bench_native.txt

echo ""
echo "=== WASM benchmark ==="
node "$SCRIPT_DIR/bench_wasm.js" "$MODEL" --iters "$ITERS" | tee /tmp/bench_wasm.txt

echo ""
echo "=== Side-by-side comparison ==="
echo "NATIVE                                                    | WASM"
echo "----------------------------------------------------------|----------------------------------------------------------"
paste -d '|' /tmp/bench_native.txt /tmp/bench_wasm.txt
