#!/usr/bin/env python3
"""Compare Q6_K embedding dequantization between Python and bitnet.c"""
import struct
import numpy as np

MODEL = "models/qwen2.5-3b-instruct-q4_0.gguf"

# Parse GGUF to find token_embd.weight tensor
with open(MODEL, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    # Skip KV pairs
    for _ in range(n_kv):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8')
        vtype = struct.unpack('<I', f.read(4))[0]

        if vtype == 8:  # STRING
            val_len = struct.unpack('<Q', f.read(8))[0]
            f.read(val_len)
        elif vtype == 4:  # UINT32
            f.read(4)
        elif vtype == 5:  # INT32
            f.read(4)
        elif vtype == 6:  # FLOAT32
            f.read(4)
        elif vtype == 7:  # BOOL
            f.read(1)
        elif vtype == 9:  # ARRAY
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_n = struct.unpack('<Q', f.read(8))[0]
            if arr_type == 8:  # array of strings
                for _ in range(arr_n):
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
            elif arr_type == 6:  # array of float32
                f.read(arr_n * 4)
            elif arr_type == 5:  # array of int32
                f.read(arr_n * 4)
            elif arr_type == 4:  # array of uint32
                f.read(arr_n * 4)
            else:
                print(f"Unknown array type {arr_type} for key {key}")
                break
        elif vtype == 10:  # UINT64
            f.read(8)
        elif vtype == 12:  # FLOAT64
            f.read(8)
        else:
            print(f"Unknown type {vtype} for key {key}")
            break

    # Read tensor infos
    tensors = {}
    for _ in range(n_tensors):
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensors[name] = {'dims': dims, 'type': ttype, 'offset': offset}

    # Find data start (aligned to 32 bytes)
    pos = f.tell()
    alignment = 32
    data_start = (pos + alignment - 1) & ~(alignment - 1)

    # Now use llama-cpp-python for proper dequantization comparison
    print(f"token_embd.weight: dims={tensors['token_embd.weight']['dims']}, type={tensors['token_embd.weight']['type']}")

# Use llama-cpp-python to get the proper embedding
from llama_cpp import Llama
llm = Llama(model_path=MODEL, n_ctx=32, verbose=False, embedding=True)

# Get embedding for token 13 (".") by evaluating it
llm.reset()
llm.eval([13])

# Get logits (these are computed from the final hidden state, not the embedding)
# We need a different approach - compute the raw embedding
n_vocab = llm.n_vocab()

# Compare specific logits between implementations
# For token 13 ("."), after the first prompt token "The" (785)
llm.reset()
llm.eval([785])
logits_ptr = llm._ctx.get_logits()
logits = np.array([logits_ptr[i] for i in range(n_vocab)])
top_idx = np.argsort(logits)[-5:][::-1]
print(f"\nAfter token 785 ('The'), top 5 logits:")
for idx in top_idx:
    piece = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  token {idx:6d} ({piece:20s}): {logits[idx]:.6f}")

# Now do 5-token prefill
llm.reset()
llm.eval([785, 6722, 315, 9625, 374])
logits_ptr = llm._ctx.get_logits()
logits = np.array([logits_ptr[i] for i in range(n_vocab)])
top_idx = np.argsort(logits)[-5:][::-1]
print(f"\nAfter 5-token prefill, top 5 logits:")
for idx in top_idx:
    piece = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  token {idx:6d} ({piece:20s}): {logits[idx]:.6f}")

# Feed tokens one-by-one (NOT prefill) and check logits after each
print(f"\n=== Token-by-token logits comparison ===")
llm.reset()
tokens = [785, 6722, 315, 9625, 374]
for i, tok in enumerate(tokens):
    llm.eval([tok])
    logits_ptr = llm._ctx.get_logits()
    logits = np.array([logits_ptr[j] for j in range(n_vocab)])
    top3 = np.argsort(logits)[-3:][::-1]
    print(f"After token {tok} (pos {i}): top3 = {[(int(t), round(logits[t], 4)) for t in top3]}")
