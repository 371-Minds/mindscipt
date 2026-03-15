#!/usr/bin/env python3
"""Compare outputs for various prompts"""
from llama_cpp import Llama

MODEL = "models/qwen2.5-3b-instruct-q4_0.gguf"
llm = Llama(model_path=MODEL, n_ctx=512, verbose=False)

prompts = [
    "Hello",
    "The capital of France is",
    "Question: What is 2+2? Answer:",
    "2+2=",
]

for prompt in prompts:
    tokens = llm.tokenize(prompt.encode(), add_bos=True)
    output = llm(prompt, max_tokens=20, temperature=0, echo=False)
    text = output['choices'][0]['text']
    print(f"Prompt: {prompt!r}")
    print(f"  Tokens: {tokens}")
    print(f"  Output: {text!r}")
    print()
