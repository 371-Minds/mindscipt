// Q4_0 REPACKED matvec — 32 rows/tile, 8 threads/row
//
// GPU buffer layout: [f32 scales: n_blocks][nibble u32s: n_blocks * 4]
//
// TILE_ROWS=32, THREADS_PER_ROW=8 (matches Metal kernel).
// 8 threads per row stride across blocks, each computes full 32-element dot per block.
// Reduction: 3-step workgroup barrier (s=4,2,1) within groups of 8.
//
// Dispatch: (ceil(rows/32), n_tokens, 1)

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;

struct Uniforms {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    extra: u32,
    bias_offset: u32,  // u32 offset into weights[] for fused bias, 0 = no bias
    _pad5: u32,
    _pad6: u32,
    _pad7: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

var<workgroup> reduce_buf: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_start = select(wid.x * TILE_ROWS, (wid.x + wid.y * uniforms.extra) * TILE_ROWS, uniforms.extra > 0u);
    let token = select(wid.y, 0u, uniforms.extra > 0u);
    let tid = lid.x;

    let row_lane = tid & 7u;         // 0-7: position within row
    let local_row = tid >> 3u;        // 0-31: row within tile
    let global_row = tile_start + local_row;

    let cols = uniforms.cols;
    let blocks_per_row = cols >> 5u;  // cols / 32
    let total_blocks = uniforms.rows * blocks_per_row;
    let x_base = token * cols;

    var acc: f32 = 0.0;

    if (global_row < uniforms.rows) {
        let row_block_base = global_row * blocks_per_row;

        for (var b = row_lane; b < blocks_per_row; b += 8u) {
            let block_idx = row_block_base + b;
            let s = bitcast<f32>(weights[block_idx]);

            let nib_base = total_blocks + block_idx * 4u;
            let w0 = weights[nib_base];
            let w1 = weights[nib_base + 1u];
            let w2 = weights[nib_base + 2u];
            let w3 = weights[nib_base + 3u];

            let elem = x_base + b * 32u;

            // Word 0: elements 0-7
            acc += s * f32(i32(w0 & 0xFu) - 8) * x[elem + 0u];
            acc += s * f32(i32((w0 >> 4u) & 0xFu) - 8) * x[elem + 1u];
            acc += s * f32(i32((w0 >> 8u) & 0xFu) - 8) * x[elem + 2u];
            acc += s * f32(i32((w0 >> 12u) & 0xFu) - 8) * x[elem + 3u];
            acc += s * f32(i32((w0 >> 16u) & 0xFu) - 8) * x[elem + 4u];
            acc += s * f32(i32((w0 >> 20u) & 0xFu) - 8) * x[elem + 5u];
            acc += s * f32(i32((w0 >> 24u) & 0xFu) - 8) * x[elem + 6u];
            acc += s * f32(i32(w0 >> 28u) - 8) * x[elem + 7u];

            // Word 1: elements 8-15
            acc += s * f32(i32(w1 & 0xFu) - 8) * x[elem + 8u];
            acc += s * f32(i32((w1 >> 4u) & 0xFu) - 8) * x[elem + 9u];
            acc += s * f32(i32((w1 >> 8u) & 0xFu) - 8) * x[elem + 10u];
            acc += s * f32(i32((w1 >> 12u) & 0xFu) - 8) * x[elem + 11u];
            acc += s * f32(i32((w1 >> 16u) & 0xFu) - 8) * x[elem + 12u];
            acc += s * f32(i32((w1 >> 20u) & 0xFu) - 8) * x[elem + 13u];
            acc += s * f32(i32((w1 >> 24u) & 0xFu) - 8) * x[elem + 14u];
            acc += s * f32(i32(w1 >> 28u) - 8) * x[elem + 15u];

            // Word 2: elements 16-23
            acc += s * f32(i32(w2 & 0xFu) - 8) * x[elem + 16u];
            acc += s * f32(i32((w2 >> 4u) & 0xFu) - 8) * x[elem + 17u];
            acc += s * f32(i32((w2 >> 8u) & 0xFu) - 8) * x[elem + 18u];
            acc += s * f32(i32((w2 >> 12u) & 0xFu) - 8) * x[elem + 19u];
            acc += s * f32(i32((w2 >> 16u) & 0xFu) - 8) * x[elem + 20u];
            acc += s * f32(i32((w2 >> 20u) & 0xFu) - 8) * x[elem + 21u];
            acc += s * f32(i32((w2 >> 24u) & 0xFu) - 8) * x[elem + 22u];
            acc += s * f32(i32(w2 >> 28u) - 8) * x[elem + 23u];

            // Word 3: elements 24-31
            acc += s * f32(i32(w3 & 0xFu) - 8) * x[elem + 24u];
            acc += s * f32(i32((w3 >> 4u) & 0xFu) - 8) * x[elem + 25u];
            acc += s * f32(i32((w3 >> 8u) & 0xFu) - 8) * x[elem + 26u];
            acc += s * f32(i32((w3 >> 12u) & 0xFu) - 8) * x[elem + 27u];
            acc += s * f32(i32((w3 >> 16u) & 0xFu) - 8) * x[elem + 28u];
            acc += s * f32(i32((w3 >> 20u) & 0xFu) - 8) * x[elem + 29u];
            acc += s * f32(i32((w3 >> 24u) & 0xFu) - 8) * x[elem + 30u];
            acc += s * f32(i32(w3 >> 28u) - 8) * x[elem + 31u];
        }
    }

    // 8-way reduction within groups of 8 threads (3 barriers)
    reduce_buf[tid] = acc;
    workgroupBarrier();

    let row_base = local_row * THREADS_PER_ROW;
    if (row_lane < 4u) {
        reduce_buf[row_base + row_lane] += reduce_buf[row_base + row_lane + 4u];
    }
    workgroupBarrier();
    if (row_lane < 2u) {
        reduce_buf[row_base + row_lane] += reduce_buf[row_base + row_lane + 2u];
    }
    workgroupBarrier();
    if (row_lane < 1u) {
        reduce_buf[row_base + row_lane] += reduce_buf[row_base + row_lane + 1u];
    }
    workgroupBarrier();

    if (row_lane == 0u && global_row < uniforms.rows) {
        var result = reduce_buf[row_base];
        if (uniforms.bias_offset > 0u) {
            result += bitcast<f32>(weights[uniforms.bias_offset + global_row]);
        }
        out[token * uniforms.rows + global_row] = result;
    }
}
