#!/usr/bin/env python3
"""Compare tokenization and logits between llama-cpp-python and bitnet.c"""
import sys
import ctypes
import numpy as np
from llama_cpp import Llama

MODEL = "models/qwen2.5-3b-instruct-q4_0.gguf"
PROMPT = "The capital of France is"

print(f"Loading model: {MODEL}")
llm = Llama(model_path=MODEL, n_ctx=512, verbose=False, logits_all=False)

# Tokenize
tokens = llm.tokenize(PROMPT.encode(), add_bos=True)
print(f"Tokens (add_bos=True): {tokens}")
tokens_no_bos = llm.tokenize(PROMPT.encode(), add_bos=False)
print(f"Tokens (add_bos=False): {tokens_no_bos}")
print(f"Decoded tokens: {[llm.detokenize([t]).decode('utf-8', errors='replace') for t in tokens_no_bos]}")

# Use the high-level API to generate
print(f"\n=== Text completion (greedy, temp=0) ===")
output = llm(PROMPT, max_tokens=20, temperature=0, echo=False)
print(f"Output: {output['choices'][0]['text']}")

# Also try with BOS added manually
print(f"\n=== Text completion with add_bos=True ===")
# llama-cpp-python auto-handles BOS based on model metadata
# Let's check what it does

# Use low-level API for token-by-token comparison
print(f"\n=== Low-level token-by-token forward pass ===")
llm.reset()

# Feed prompt tokens
use_tokens = tokens_no_bos  # Qwen2 says add_bos=false
llm.eval(use_tokens)

# Get logits from the context
n_vocab = llm.n_vocab()
logits_ptr = llm._ctx.get_logits()
logits = np.array([logits_ptr[i] for i in range(n_vocab)])
top_idx = np.argsort(logits)[-10:][::-1]

print(f"\nAfter prefill ({len(use_tokens)} tokens), top 10 logits:")
for idx in top_idx:
    piece = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  token {idx:6d} ({piece:20s}): {logits[idx]:.4f}")

# Generate 10 tokens greedily
print(f"\n=== Greedy generation ===")
next_token = int(top_idx[0])
for i in range(10):
    piece = llm.detokenize([next_token]).decode('utf-8', errors='replace')
    print(f"gen {i}: token {next_token:6d} = '{piece}'")

    llm.eval([next_token])
    logits_ptr = llm._ctx.get_logits()
    logits = np.array([logits_ptr[j] for j in range(n_vocab)])
    top5 = np.argsort(logits)[-5:][::-1]
    for t in top5:
        p = llm.detokenize([t]).decode('utf-8', errors='replace')
        print(f"    {t:6d} ({p:15s}): {logits[t]:.4f}")
    next_token = int(top5[0])
