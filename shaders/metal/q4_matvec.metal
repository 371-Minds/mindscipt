#include <metal_stdlib>
using namespace metal;

// Q4_0 REPACKED matvec — 32 rows/tile, 8 threads/row, simd reduction
//
// GPU buffer layout: [f32 scales: n_blocks][nibble u32s: n_blocks * 4]
//
// 8 simdgroups × 4 rows/simdgroup = 32 rows per tile.
// 8 threads per row stride across blocks (each handles blocks_per_row/8 blocks).
// Each thread computes the full 32-element dot product per block.
// Reduction: simd_shuffle_xor for 8-thread partial sum (0 barriers, 0 shared memory).
// x vector naturally shared across 4 rows within each simdgroup via L1 cache.
//
// Dispatch: (ceil(rows/32), n_tokens, 1)

kernel void q4_matvec(device const uint  *weights [[buffer(0)]],
                      device const float *x       [[buffer(1)]],
                      device float       *out     [[buffer(2)]],
                      constant uint      *p       [[buffer(3)]],
                      uint3 wid [[threadgroup_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3], bias_offset = p[4];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * 32 : wid.x * 32;
    uint token = (extra > 0) ? 0 : wid.y;

    uint row_lane = lid.x & 7;        // 0-7: position within row
    uint local_row = lid.x >> 3;       // 0-31: row within tile
    uint global_row = tile_start + local_row;

    uint blocks_per_row = cols >> 5;   // cols / 32
    uint total_blocks = rows * blocks_per_row;
    uint x_base = token * cols;

    float acc = 0.0f;

    if (global_row < rows) {
        uint row_block_base = global_row * blocks_per_row;

        for (uint b = row_lane; b < blocks_per_row; b += 8) {
            uint block_idx = row_block_base + b;
            float s = as_type<float>(weights[block_idx]);

            uint nib_base = total_blocks + block_idx * 4;
            uint w0 = weights[nib_base];
            uint w1 = weights[nib_base + 1];
            uint w2 = weights[nib_base + 2];
            uint w3 = weights[nib_base + 3];

            uint elem = x_base + b * 32;

            // Word 0: elements 0-7
            acc += s * float(int((w0      ) & 0xF) - 8) * x[elem +  0];
            acc += s * float(int((w0 >>  4) & 0xF) - 8) * x[elem +  1];
            acc += s * float(int((w0 >>  8) & 0xF) - 8) * x[elem +  2];
            acc += s * float(int((w0 >> 12) & 0xF) - 8) * x[elem +  3];
            acc += s * float(int((w0 >> 16) & 0xF) - 8) * x[elem +  4];
            acc += s * float(int((w0 >> 20) & 0xF) - 8) * x[elem +  5];
            acc += s * float(int((w0 >> 24) & 0xF) - 8) * x[elem +  6];
            acc += s * float(int((w0 >> 28)      ) - 8) * x[elem +  7];

            // Word 1: elements 8-15
            acc += s * float(int((w1      ) & 0xF) - 8) * x[elem +  8];
            acc += s * float(int((w1 >>  4) & 0xF) - 8) * x[elem +  9];
            acc += s * float(int((w1 >>  8) & 0xF) - 8) * x[elem + 10];
            acc += s * float(int((w1 >> 12) & 0xF) - 8) * x[elem + 11];
            acc += s * float(int((w1 >> 16) & 0xF) - 8) * x[elem + 12];
            acc += s * float(int((w1 >> 20) & 0xF) - 8) * x[elem + 13];
            acc += s * float(int((w1 >> 24) & 0xF) - 8) * x[elem + 14];
            acc += s * float(int((w1 >> 28)      ) - 8) * x[elem + 15];

            // Word 2: elements 16-23
            acc += s * float(int((w2      ) & 0xF) - 8) * x[elem + 16];
            acc += s * float(int((w2 >>  4) & 0xF) - 8) * x[elem + 17];
            acc += s * float(int((w2 >>  8) & 0xF) - 8) * x[elem + 18];
            acc += s * float(int((w2 >> 12) & 0xF) - 8) * x[elem + 19];
            acc += s * float(int((w2 >> 16) & 0xF) - 8) * x[elem + 20];
            acc += s * float(int((w2 >> 20) & 0xF) - 8) * x[elem + 21];
            acc += s * float(int((w2 >> 24) & 0xF) - 8) * x[elem + 22];
            acc += s * float(int((w2 >> 28)      ) - 8) * x[elem + 23];

            // Word 3: elements 24-31
            acc += s * float(int((w3      ) & 0xF) - 8) * x[elem + 24];
            acc += s * float(int((w3 >>  4) & 0xF) - 8) * x[elem + 25];
            acc += s * float(int((w3 >>  8) & 0xF) - 8) * x[elem + 26];
            acc += s * float(int((w3 >> 12) & 0xF) - 8) * x[elem + 27];
            acc += s * float(int((w3 >> 16) & 0xF) - 8) * x[elem + 28];
            acc += s * float(int((w3 >> 20) & 0xF) - 8) * x[elem + 29];
            acc += s * float(int((w3 >> 24) & 0xF) - 8) * x[elem + 30];
            acc += s * float(int((w3 >> 28)      ) - 8) * x[elem + 31];
        }
    }

    // 8-way simd reduction (within simdgroup, 0 barriers, 0 shared memory)
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows) {
        float result = acc;
        if (bias_offset > 0)
            result += as_type<float>(weights[bias_offset + global_row]);
        out[token * rows + global_row] = result;
    }
}
