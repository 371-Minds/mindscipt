#!/bin/bash
MODEL="models/bitnet-b1.58-2B-4T.gguf"
URL="https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/bitnet-b1.58-2B-4T.gguf"

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    echo "Download it with:"
    echo "  mkdir -p models && wget -O $MODEL $URL"
    exit 1
fi

./bitnet "$MODEL" --chat "$@"
