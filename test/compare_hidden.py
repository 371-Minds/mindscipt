#!/usr/bin/env python3
"""Extract hidden state values from llama-cpp-python for comparison"""
import ctypes
import numpy as np
from llama_cpp import Llama
import llama_cpp

MODEL = "models/qwen2.5-3b-instruct-q4_0.gguf"

print(f"Loading model: {MODEL}")
llm = Llama(model_path=MODEL, n_ctx=512, verbose=False)

tokens = llm.tokenize(b"The capital of France is", add_bos=False)
print(f"Tokens: {tokens}")

# Prefill
llm.reset()
llm.eval(tokens)

# Get logits (which are computed from the final hidden state)
n_vocab = llm.n_vocab()
logits_ptr = llm._ctx.get_logits()
logits = np.array([logits_ptr[i] for i in range(n_vocab)])

# Print top logits after prefill
top_idx = np.argsort(logits)[-5:][::-1]
print(f"\nAfter prefill, top 5:")
for idx in top_idx:
    piece = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  token {idx:6d} ({piece:20s}): {logits[idx]:.4f}")

# Generate " Paris" (12095) and get logits after
print(f"\nEval token 12095 (' Paris')...")
llm.eval([12095])
logits_ptr = llm._ctx.get_logits()
logits = np.array([logits_ptr[i] for i in range(n_vocab)])
top_idx = np.argsort(logits)[-5:][::-1]
print(f"After ' Paris', top 5:")
for idx in top_idx:
    piece = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  token {idx:6d} ({piece:20s}): {logits[idx]:.4f}")

# Generate "." (13) and compare
print(f"\nEval token 13 ('.')...")
llm.eval([13])
logits_ptr = llm._ctx.get_logits()
logits = np.array([logits_ptr[i] for i in range(n_vocab)])
top_idx = np.argsort(logits)[-10:][::-1]
print(f"After '.', top 10:")
for idx in top_idx:
    piece = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  token {idx:6d} ({piece:20s}): {logits[idx]:.4f}")

# Compare specific tokens
print(f"\nSpecific logits comparison:")
for tid in [12095, 576, 15920, 3555, 1084]:
    piece = llm.detokenize([tid]).decode('utf-8', errors='replace')
    print(f"  token {tid:6d} ({piece:20s}): {logits[tid]:.6f}")
