#!/usr/bin/env python3
"""
Compare Q4_0 x float vs Q4_0 x Q8_0 matvec precision.

bitnet.c uses Q4_0 x float (dequantize weights, multiply with float activations).
llama.cpp uses Q4_0 x Q8_0 (quantize activations to Q8_0, integer dot products).

This script loads a real GGUF model, extracts one Q4_0 weight tensor and the
corresponding embedding + norm weights, then computes the same matvec both ways
to measure the numerical difference.
"""

import struct
import sys
import os
import math
import numpy as np


# ---------------------------------------------------------------------------
# FP16 conversion
# ---------------------------------------------------------------------------

def fp16_to_fp32(h):
    """Convert a uint16 IEEE-754 half-precision value to float32."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        if mant == 0:
            return (-1.0 if sign else 1.0) * 0.0
        # Subnormal
        val = mant / 1024.0 * (2.0 ** -14)
        return -val if sign else val
    elif exp == 31:
        if mant == 0:
            return float('-inf') if sign else float('inf')
        return float('nan')
    else:
        val = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))
        return -val if sign else val


# Vectorised version using numpy (much faster for large arrays)
def fp16_array_to_fp32(arr_u16):
    """Convert a numpy uint16 array of FP16 values to float32 via np.frombuffer."""
    return np.frombuffer(arr_u16.astype('<u2').tobytes(), dtype=np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# GGUF parser (minimal, read-only, little-endian)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

GGUF_TENSOR_F32   = 0
GGUF_TENSOR_F16   = 1
GGUF_TENSOR_Q4_0  = 2
GGUF_TENSOR_Q8_0  = 8
GGUF_TENSOR_Q6_K  = 14


TYPE_SIZES = {
    GGUF_TYPE_UINT8:   1,
    GGUF_TYPE_INT8:    1,
    GGUF_TYPE_UINT16:  2,
    GGUF_TYPE_INT16:   2,
    GGUF_TYPE_UINT32:  4,
    GGUF_TYPE_INT32:   4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL:    1,
    GGUF_TYPE_UINT64:  8,
    GGUF_TYPE_INT64:   8,
    GGUF_TYPE_FLOAT64: 8,
}


class GGUFReader:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.mm = None
        try:
            import mmap
            self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
            self.buf = self.mm
        except Exception:
            self.buf = self.f.read()
        self.pos = 0
        self.kvs = {}
        self.tensors = {}
        self._parse()

    def close(self):
        if self.mm:
            self.mm.close()
        self.f.close()

    def _read(self, n):
        data = self.buf[self.pos:self.pos + n]
        self.pos += n
        return data

    def _u8(self):
        return struct.unpack('<B', self._read(1))[0]

    def _u16(self):
        return struct.unpack('<H', self._read(2))[0]

    def _u32(self):
        return struct.unpack('<I', self._read(4))[0]

    def _u64(self):
        return struct.unpack('<Q', self._read(8))[0]

    def _i8(self):
        return struct.unpack('<b', self._read(1))[0]

    def _i16(self):
        return struct.unpack('<h', self._read(2))[0]

    def _i32(self):
        return struct.unpack('<i', self._read(4))[0]

    def _i64(self):
        return struct.unpack('<q', self._read(8))[0]

    def _f32(self):
        return struct.unpack('<f', self._read(4))[0]

    def _f64(self):
        return struct.unpack('<d', self._read(8))[0]

    def _string(self):
        length = self._u64()
        return self._read(length).decode('utf-8', errors='replace')

    def _read_value(self, vtype):
        if vtype == GGUF_TYPE_UINT8:   return self._u8()
        if vtype == GGUF_TYPE_INT8:    return self._i8()
        if vtype == GGUF_TYPE_UINT16:  return self._u16()
        if vtype == GGUF_TYPE_INT16:   return self._i16()
        if vtype == GGUF_TYPE_UINT32:  return self._u32()
        if vtype == GGUF_TYPE_INT32:   return self._i32()
        if vtype == GGUF_TYPE_FLOAT32: return self._f32()
        if vtype == GGUF_TYPE_BOOL:    return bool(self._u8())
        if vtype == GGUF_TYPE_STRING:  return self._string()
        if vtype == GGUF_TYPE_UINT64:  return self._u64()
        if vtype == GGUF_TYPE_INT64:   return self._i64()
        if vtype == GGUF_TYPE_FLOAT64: return self._f64()
        if vtype == GGUF_TYPE_ARRAY:
            elem_type = self._u32()
            count = self._u64()
            if elem_type == GGUF_TYPE_STRING:
                return [self._string() for _ in range(count)]
            else:
                sz = TYPE_SIZES.get(elem_type, 0)
                if sz > 0:
                    data = self._read(count * sz)
                    return data  # raw bytes
                return None
        return None

    def _parse(self):
        magic = self._u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"Bad GGUF magic: 0x{magic:08x}")

        version = self._u32()
        n_tensors = self._u64()
        n_kv = self._u64()

        alignment = 32  # default

        # Read KV pairs
        for _ in range(n_kv):
            key = self._string()
            vtype = self._u32()
            value = self._read_value(vtype)
            self.kvs[key] = value
            if key == 'general.alignment' and isinstance(value, int):
                alignment = value

        self.alignment = alignment

        # Read tensor infos
        tensor_infos = []
        for _ in range(n_tensors):
            name = self._string()
            n_dims = self._u32()
            dims = [self._u64() for _ in range(n_dims)]
            ttype = self._u32()
            offset = self._u64()
            tensor_infos.append({
                'name': name,
                'n_dims': n_dims,
                'dims': dims,
                'type': ttype,
                'offset': offset,
            })

        # Compute data offset (aligned)
        header_end = self.pos
        data_offset = header_end + (alignment - (header_end % alignment)) % alignment
        self.data_offset = data_offset

        for info in tensor_infos:
            info['abs_offset'] = data_offset + info['offset']
            self.tensors[info['name']] = info

    def tensor_data(self, name):
        """Return raw bytes for a tensor's data region."""
        info = self.tensors[name]
        offset = info['abs_offset']
        nelements = 1
        for d in info['dims']:
            nelements *= d
        ttype = info['type']
        if ttype == GGUF_TENSOR_F32:
            nbytes = nelements * 4
        elif ttype == GGUF_TENSOR_F16:
            nbytes = nelements * 2
        elif ttype == GGUF_TENSOR_Q4_0:
            nbytes = (nelements // 32) * 18
        elif ttype == GGUF_TENSOR_Q8_0:
            nbytes = (nelements // 32) * 34
        elif ttype == GGUF_TENSOR_Q6_K:
            nbytes = (nelements // 256) * 210
        else:
            raise ValueError(f"Unknown tensor type {ttype} for {name}")
        return bytes(self.buf[offset:offset + nbytes])


# ---------------------------------------------------------------------------
# Q4_0 dequantization
# ---------------------------------------------------------------------------

def dequant_q4_0_block(block_bytes):
    """Dequantize one Q4_0 block (18 bytes -> 32 floats).

    Layout: 2 bytes FP16 scale (d) + 16 bytes packed nibbles (qs).
    Low nibble = elements 0-15, high nibble = elements 16-31.
    Value = (nibble - 8) * d.
    """
    d_u16 = struct.unpack('<H', block_bytes[0:2])[0]
    d = fp16_to_fp32(d_u16)
    out = np.zeros(32, dtype=np.float32)
    for i in range(16):
        b = block_bytes[2 + i]
        out[i]      = ((b & 0xF) - 8) * d
        out[i + 16] = ((b >> 4)  - 8) * d
    return out


def dequant_q4_0_tensor(raw_bytes, rows, cols):
    """Dequantize an entire Q4_0 tensor (rows x cols) to float32."""
    n_blocks_per_row = cols // 32
    block_size = 18
    result = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for b in range(n_blocks_per_row):
            offset = (row * n_blocks_per_row + b) * block_size
            block = raw_bytes[offset:offset + block_size]
            result[row, b * 32:(b + 1) * 32] = dequant_q4_0_block(block)
    return result


# ---------------------------------------------------------------------------
# Q6_K dequantization
# ---------------------------------------------------------------------------

def dequant_q6k_block(block_bytes):
    """Dequantize one Q6_K block (210 bytes -> 256 floats).

    Layout: ql[128] + qh[64] + scales[16] + d[2 bytes FP16] = 210 bytes.
    Process in two 128-element chunks. Each chunk uses 64 ql bytes, 32 qh bytes,
    8 scale bytes.
    """
    ql = np.frombuffer(block_bytes[0:128], dtype=np.uint8)
    qh = np.frombuffer(block_bytes[128:192], dtype=np.uint8)
    scales = np.frombuffer(block_bytes[192:208], dtype=np.int8)
    d_u16 = struct.unpack('<H', block_bytes[208:210])[0]
    d = fp16_to_fp32(d_u16)

    out = np.zeros(256, dtype=np.float32)

    for chunk in range(2):  # two 128-element chunks
        ql_off = chunk * 64
        qh_off = chunk * 32
        sc_off = chunk * 8
        out_off = chunk * 128

        for l in range(32):
            is_ = l // 16
            q1 = int((ql[ql_off + l]      & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) - 32
            q2 = int((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32
            q3 = int((ql[ql_off + l]      >> 4)  | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32
            q4 = int((ql[ql_off + l + 32] >> 4)  | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32

            out[out_off + l +  0] = d * int(scales[sc_off + is_ + 0]) * q1
            out[out_off + l + 32] = d * int(scales[sc_off + is_ + 2]) * q2
            out[out_off + l + 64] = d * int(scales[sc_off + is_ + 4]) * q3
            out[out_off + l + 96] = d * int(scales[sc_off + is_ + 6]) * q4

    return out


def dequant_q6k_row(raw_bytes, cols):
    """Dequantize one row of a Q6_K tensor."""
    n_blocks = cols // 256
    block_size = 210
    out = np.zeros(cols, dtype=np.float32)
    for b in range(n_blocks):
        offset = b * block_size
        block = raw_bytes[offset:offset + block_size]
        out[b * 256:(b + 1) * 256] = dequant_q6k_block(block)
    return out


# ---------------------------------------------------------------------------
# Q8_0 quantization (what llama.cpp does to the activation vector)
# ---------------------------------------------------------------------------

def quantize_to_q8_0(x):
    """Quantize a float32 vector to Q8_0 format.

    Block size = 32. Per-block: scale d = max(|x|) / 127.
    q[i] = round(x[i] / d), clamped to [-127, 127].

    Returns (q_int8, scales) where q_int8 is the quantized array and
    scales is per-block scales.
    """
    n = len(x)
    assert n % 32 == 0, f"Vector length {n} not a multiple of 32"
    n_blocks = n // 32
    q = np.zeros(n, dtype=np.int8)
    scales = np.zeros(n_blocks, dtype=np.float32)

    for b in range(n_blocks):
        block = x[b * 32:(b + 1) * 32]
        amax = np.max(np.abs(block))
        if amax == 0.0:
            scales[b] = 0.0
            continue
        d = amax / 127.0
        scales[b] = d
        inv_d = 127.0 / amax
        for i in range(32):
            v = int(round(block[i] * inv_d))
            v = max(-127, min(127, v))
            q[b * 32 + i] = v

    return q, scales


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

def rmsnorm(x, w, eps=1e-6):
    """Apply RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]."""
    ss = np.sum(x * x) / len(x)
    inv_rms = 1.0 / math.sqrt(ss + eps)
    return x * inv_rms * w


# ---------------------------------------------------------------------------
# Matvec methods
# ---------------------------------------------------------------------------

def matvec_q4_float(raw_bytes, rows, cols, x_float):
    """Q4_0 x float: dequantize each Q4_0 block, multiply by float x, accumulate.

    This is what bitnet.c does.
    """
    n_blocks_per_row = cols // 32
    block_size = 18
    out = np.zeros(rows, dtype=np.float32)

    for row in range(rows):
        row_sum = np.float32(0.0)
        for b in range(n_blocks_per_row):
            offset = (row * n_blocks_per_row + b) * block_size
            block = raw_bytes[offset:offset + block_size]

            d_u16 = struct.unpack('<H', block[0:2])[0]
            d = np.float32(fp16_to_fp32(d_u16))

            xb = x_float[b * 32:(b + 1) * 32]

            block_sum = np.float32(0.0)
            for i in range(16):
                byte = block[2 + i]
                w_lo = np.float32((byte & 0xF) - 8) * d
                w_hi = np.float32((byte >> 4) - 8) * d
                block_sum += w_lo * xb[i]
                block_sum += w_hi * xb[i + 16]

            row_sum += block_sum
        out[row] = row_sum

    return out


def matvec_q4_q8(raw_bytes, rows, cols, x_float):
    """Q4_0 x Q8_0: quantize x to Q8_0, use integer dot products.

    This is what llama.cpp does for Q4_0 matvec (ggml_vec_dot_q4_0_q8_0).
    For each Q4_0 block and corresponding Q8_0 block:
        sum += d_q4 * d_q8 * sum_i(q4[i] * q8[i])
    where q4[i] = nibble - 8, q8[i] = quantized activation.
    """
    n_blocks_per_row = cols // 32
    block_size = 18

    # Quantize the activation vector to Q8_0
    x_q8, x_scales = quantize_to_q8_0(x_float)

    out = np.zeros(rows, dtype=np.float32)

    for row in range(rows):
        row_sum = np.float32(0.0)
        for b in range(n_blocks_per_row):
            offset = (row * n_blocks_per_row + b) * block_size
            block = raw_bytes[offset:offset + block_size]

            d_q4_u16 = struct.unpack('<H', block[0:2])[0]
            d_q4 = np.float32(fp16_to_fp32(d_q4_u16))
            d_q8 = np.float32(x_scales[b])

            # Integer dot product between Q4_0 weights and Q8_0 activations
            isum = 0
            xq = x_q8[b * 32:(b + 1) * 32]
            for i in range(16):
                byte = block[2 + i]
                w_lo = (byte & 0xF) - 8   # int
                w_hi = (byte >> 4) - 8     # int
                isum += w_lo * int(xq[i])
                isum += w_hi * int(xq[i + 16])

            row_sum += d_q4 * d_q8 * np.float32(isum)
        out[row] = row_sum

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'models', 'qwen2.5-3b-instruct-q4_0.gguf')

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    print(f"Loading GGUF: {model_path}")
    gguf = GGUFReader(model_path)

    # Print model info
    arch = gguf.kvs.get('general.architecture', '?')
    print(f"Architecture: {arch}")

    dim_key = f"{arch}.embedding_length"
    dim = gguf.kvs.get(dim_key, 0)
    print(f"Embedding dim: {dim}")

    eps_key = f"{arch}.attention.layer_norm_rms_epsilon"
    eps = gguf.kvs.get(eps_key, 1e-6)
    print(f"RMSNorm eps: {eps}")

    # -----------------------------------------------------------------------
    # 1) Find the token embedding tensor and get embedding for token 785
    # -----------------------------------------------------------------------
    emb_name = 'token_embd.weight'
    emb_info = gguf.tensors[emb_name]
    emb_type = emb_info['type']
    emb_dims = emb_info['dims']  # [dim, vocab_size] in GGUF (col-major dims)
    print(f"\nEmbedding tensor: {emb_name}")
    print(f"  type: {emb_type}, dims: {emb_dims}")

    token_id = 785
    print(f"\nExtracting embedding for token {token_id}...")

    if emb_type == GGUF_TENSOR_Q6_K:
        # Q6_K: 210 bytes per 256-element block
        block_size = 210
        n_blocks_per_row = dim // 256
        row_bytes = n_blocks_per_row * block_size
        emb_raw = gguf.tensor_data(emb_name)
        row_start = token_id * row_bytes
        row_data = emb_raw[row_start:row_start + row_bytes]
        embedding = dequant_q6k_row(row_data, dim)
        print(f"  Dequantized Q6_K embedding, shape: {embedding.shape}")
    elif emb_type == GGUF_TENSOR_F16:
        emb_raw = gguf.tensor_data(emb_name)
        row_start = token_id * dim * 2
        row_u16 = np.frombuffer(emb_raw[row_start:row_start + dim * 2], dtype=np.uint16)
        embedding = fp16_array_to_fp32(row_u16)
        print(f"  Dequantized F16 embedding, shape: {embedding.shape}")
    elif emb_type == GGUF_TENSOR_Q4_0:
        block_size = 18
        n_blocks_per_row = dim // 32
        row_bytes = n_blocks_per_row * block_size
        emb_raw = gguf.tensor_data(emb_name)
        row_start = token_id * row_bytes
        row_data = emb_raw[row_start:row_start + row_bytes]
        embedding = np.zeros(dim, dtype=np.float32)
        for b in range(n_blocks_per_row):
            offset = b * block_size
            embedding[b * 32:(b + 1) * 32] = dequant_q4_0_block(row_data[offset:offset + block_size])
        print(f"  Dequantized Q4_0 embedding, shape: {embedding.shape}")
    else:
        print(f"  ERROR: Unsupported embedding type {emb_type}")
        gguf.close()
        sys.exit(1)

    print(f"  Embedding stats: min={embedding.min():.6f}, max={embedding.max():.6f}, "
          f"mean={embedding.mean():.6f}, std={embedding.std():.6f}")

    # -----------------------------------------------------------------------
    # 2) Load attn_norm.weight for blk.0 and apply RMSNorm
    # -----------------------------------------------------------------------
    norm_name = 'blk.0.attn_norm.weight'
    norm_info = gguf.tensors[norm_name]
    print(f"\nNorm tensor: {norm_name}")
    print(f"  type: {norm_info['type']}, dims: {norm_info['dims']}")

    norm_raw = gguf.tensor_data(norm_name)
    if norm_info['type'] == GGUF_TENSOR_F32:
        norm_weights = np.frombuffer(norm_raw, dtype=np.float32).copy()
    else:
        print(f"  ERROR: Expected F32 norm weights, got type {norm_info['type']}")
        gguf.close()
        sys.exit(1)

    x_normed = rmsnorm(embedding, norm_weights, eps=eps)
    print(f"  After RMSNorm: min={x_normed.min():.6f}, max={x_normed.max():.6f}, "
          f"mean={x_normed.mean():.6f}, std={x_normed.std():.6f}")

    # -----------------------------------------------------------------------
    # 3) Load Q4_0 weight tensor (blk.0.attn_q.weight)
    # -----------------------------------------------------------------------
    weight_name = 'blk.0.attn_q.weight'
    w_info = gguf.tensors[weight_name]
    w_type = w_info['type']
    w_dims = w_info['dims']
    w_cols = w_dims[0]
    w_rows = w_dims[1]
    print(f"\nWeight tensor: {weight_name}")
    print(f"  type: {w_type}, dims: {w_dims} (rows={w_rows}, cols={w_cols})")

    if w_type != GGUF_TENSOR_Q4_0:
        print(f"  ERROR: Expected Q4_0 (type 2), got type {w_type}")
        gguf.close()
        sys.exit(1)

    w_raw = gguf.tensor_data(weight_name)
    print(f"  Raw data size: {len(w_raw)} bytes")
    expected_size = (w_rows * w_cols // 32) * 18
    print(f"  Expected size: {expected_size} bytes")
    assert len(w_raw) == expected_size, "Size mismatch!"

    # -----------------------------------------------------------------------
    # 4) Show Q8_0 quantization of the activation vector
    # -----------------------------------------------------------------------
    print(f"\n--- Q8_0 quantization of x_normed ---")
    x_q8, x_scales = quantize_to_q8_0(x_normed)
    n_blocks = len(x_normed) // 32
    print(f"  {n_blocks} blocks of 32 elements")
    print(f"  Scale range: [{x_scales.min():.8f}, {x_scales.max():.8f}]")
    print(f"  Mean scale: {x_scales.mean():.8f}")

    # Show quantization error
    x_reconstructed = np.zeros_like(x_normed)
    for b in range(n_blocks):
        x_reconstructed[b*32:(b+1)*32] = x_q8[b*32:(b+1)*32].astype(np.float32) * x_scales[b]

    q8_err = x_normed - x_reconstructed
    print(f"  Q8_0 reconstruction error: max={np.max(np.abs(q8_err)):.8f}, "
          f"mean={np.mean(np.abs(q8_err)):.8f}, "
          f"rms={np.sqrt(np.mean(q8_err**2)):.8f}")
    print(f"  Relative RMS error: {np.sqrt(np.mean(q8_err**2)) / np.sqrt(np.mean(x_normed**2)):.6f}")

    # -----------------------------------------------------------------------
    # 5) Compute matvec both ways (first N rows for speed)
    # -----------------------------------------------------------------------
    # Full matrix is large, compute a subset for detailed comparison
    n_test_rows = min(w_rows, 256)
    print(f"\n{'='*70}")
    print(f"Computing matvec for first {n_test_rows} rows (of {w_rows})")
    print(f"{'='*70}")

    # Subset raw bytes: only first n_test_rows
    n_blocks_per_row = w_cols // 32
    row_bytes = n_blocks_per_row * 18
    w_raw_subset = w_raw[:n_test_rows * row_bytes]

    print("\nMethod A: Q4_0 x float (bitnet.c approach)")
    print("  Dequantize Q4_0 weights to float, multiply with float activations")
    result_float = matvec_q4_float(w_raw_subset, n_test_rows, w_cols, x_normed)
    print(f"  Result stats: min={result_float.min():.6f}, max={result_float.max():.6f}, "
          f"mean={result_float.mean():.6f}")

    print("\nMethod B: Q4_0 x Q8_0 (llama.cpp approach)")
    print("  Quantize activations to Q8_0, integer dot product with Q4_0")
    result_q8 = matvec_q4_q8(w_raw_subset, n_test_rows, w_cols, x_normed)
    print(f"  Result stats: min={result_q8.min():.6f}, max={result_q8.max():.6f}, "
          f"mean={result_q8.mean():.6f}")

    # -----------------------------------------------------------------------
    # 6) Compare results
    # -----------------------------------------------------------------------
    diff = result_float - result_q8
    abs_diff = np.abs(diff)
    rel_diff = np.where(np.abs(result_float) > 1e-10,
                        abs_diff / np.abs(result_float),
                        0.0)

    print(f"\n{'='*70}")
    print(f"COMPARISON: Q4_0 x float vs Q4_0 x Q8_0  ({n_test_rows} rows)")
    print(f"{'='*70}")
    print(f"  Max absolute difference:  {abs_diff.max():.8f}")
    print(f"  Mean absolute difference: {abs_diff.mean():.8f}")
    print(f"  RMS difference:           {np.sqrt(np.mean(diff**2)):.8f}")
    print(f"  Max relative difference:  {rel_diff.max():.6f} ({rel_diff.max()*100:.4f}%)")
    print(f"  Mean relative difference: {rel_diff.mean():.6f} ({rel_diff.mean()*100:.4f}%)")

    # Correlation
    if np.std(result_float) > 0 and np.std(result_q8) > 0:
        corr = np.corrcoef(result_float, result_q8)[0, 1]
        print(f"  Pearson correlation:      {corr:.10f}")
    else:
        print(f"  Pearson correlation:      N/A (zero variance)")

    # Cosine similarity
    norm_a = np.linalg.norm(result_float)
    norm_b = np.linalg.norm(result_q8)
    if norm_a > 0 and norm_b > 0:
        cosine = np.dot(result_float, result_q8) / (norm_a * norm_b)
        print(f"  Cosine similarity:        {cosine:.10f}")

    # Signal-to-noise ratio
    signal_power = np.mean(result_float ** 2)
    noise_power = np.mean(diff ** 2)
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
        print(f"  SNR (dB):                 {snr_db:.2f}")

    # Show worst-case elements
    print(f"\n  Top 5 worst-case elements:")
    worst_idx = np.argsort(abs_diff)[-5:][::-1]
    for idx in worst_idx:
        print(f"    row {idx:4d}: float={result_float[idx]:+12.6f}  "
              f"q8={result_q8[idx]:+12.6f}  "
              f"diff={diff[idx]:+10.8f}  "
              f"rel={rel_diff[idx]*100:.4f}%")

    # Histogram of absolute differences
    print(f"\n  Absolute difference distribution:")
    thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for t in thresholds:
        count = np.sum(abs_diff < t)
        pct = count / len(abs_diff) * 100
        print(f"    < {t:.0e}: {count:5d}/{len(abs_diff)} ({pct:6.2f}%)")

    # -----------------------------------------------------------------------
    # Summary interpretation
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print(f"The Q8_0 quantization of the activation vector introduces a per-block")
    print(f"quantization error. For each block of 32 elements, the activation is")
    print(f"rounded to int8 with scale = max(|x|)/127. This means values within")
    print(f"each block share the same scale, losing about 1 bit of precision.")
    print(f"")
    print(f"This error propagates through the dot product, causing the Q4_0 x Q8_0")
    print(f"result to differ from Q4_0 x float. The difference is:")
    print(f"  - Systematic (not random noise): each block's rounding error is fixed")
    print(f"  - Proportional to activation magnitude and weight magnitude")
    print(f"  - Typically small relative to the output values")

    gguf.close()
    print(f"\nDone.")


if __name__ == '__main__':
    main()
