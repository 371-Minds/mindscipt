#!/bin/bash
# bench_suite.sh — Standardized benchmark suite for regression tracking.
# Runs tok/s measurement on available models with fixed prompts.
# Output: machine-parseable TSV on stdout, human summary on stderr.
#
# Usage:
#   make bench              # build + run (default models)
#   ./bench/bench_suite.sh  # run with pre-built binary
#   ./bench/bench_suite.sh models/specific.gguf  # single model
#
# Environment:
#   BENCH_TOKENS=50    tokens to generate per run (default: 50)
#   BENCH_RUNS=3       runs per model (default: 3, takes median)
#   BENCH_THREADS=0    thread count (default: 0 = auto)
#   BENCH_TQ=0         TQ bits (default: 0 = disabled)

set -uo pipefail

TOKENS="${BENCH_TOKENS:-50}"
RUNS="${BENCH_RUNS:-3}"
THREADS="${BENCH_THREADS:-0}"
TQ="${BENCH_TQ:-0}"
PROMPT="The meaning of life is to"
BINARY="./bitnet"

if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY not found. Run 'make' first." >&2
    exit 1
fi

# Collect models: either from args or auto-detect small models (<6 GB)
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=()
    for f in models/*.gguf; do
        [ -f "$f" ] || continue
        size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
        # Skip models > 6 GB for quick regression suite
        if [ "$size" -lt 6442450944 ]; then
            MODELS+=("$f")
        fi
    done
    if [ ${#MODELS[@]} -eq 0 ]; then
        echo "ERROR: No models found in models/ (< 6 GB). Provide a path." >&2
        exit 1
    fi
fi

# Build thread flag
THREAD_FLAG=""
if [ "$THREADS" -gt 0 ]; then
    THREAD_FLAG="-t $THREADS"
fi

# Build TQ flag
TQ_FLAG=""
if [ "$TQ" -gt 0 ]; then
    TQ_FLAG="--kv-tq $TQ"
fi

# Header
echo -e "model\ttok_s\ttokens\tthreads\ttq_bits\trun" >&1
echo "=== Benchmark Suite ===" >&2
echo "Tokens: $TOKENS | Runs: $RUNS | Threads: ${THREADS:-auto} | TQ: $TQ" >&2
echo "" >&2

for model in "${MODELS[@]}"; do
    fname=$(basename "$model" .gguf)
    echo "--- $fname ---" >&2

    results=()
    for run in $(seq 1 "$RUNS"); do
        # Run inference with timeout, capture tok/s from structured log
        timeout_sec=$(( TOKENS * 5 + 30 ))  # generous: 5s/tok + 30s warmup
        output=$(timeout "$timeout_sec" "$BINARY" "$model" -p "$PROMPT" -n "$TOKENS" $THREAD_FLAG $TQ_FLAG 2>&1 || true)

        # Parse tok/s from "tok/s=XX.XX" in log output
        toks=$(echo "$output" | grep -oE 'tok/s=[0-9.]+' | head -1 | cut -d= -f2)
        if [ -z "$toks" ]; then
            toks="0"
            echo "  run $run: FAILED" >&2
            continue
        fi

        # Parse thread count from log
        threads_used=$(echo "$output" | grep -oE 'threads=[0-9]+' | head -1 | cut -d= -f2)
        [ -z "$threads_used" ] && threads_used="?"

        results+=("$toks")
        echo -e "${fname}\t${toks}\t${TOKENS}\t${threads_used}\t${TQ}\t${run}"
        echo "  run $run: ${toks} tok/s" >&2
    done

    # Compute median (sort numerically, pick middle)
    if [ ${#results[@]} -gt 0 ]; then
        sorted=($(printf '%s\n' "${results[@]}" | sort -n))
        mid=$(( ${#sorted[@]} / 2 ))
        median="${sorted[$mid]}"
        echo "  median: ${median} tok/s" >&2
    else
        echo "  median: FAILED (all runs crashed)" >&2
    fi
    echo "" >&2
done

echo "=== Done ===" >&2
