// Q4_0 TILED matvec — optimized for GPU bandwidth utilization
//
// Strategy: process TILE_ROWS output rows per workgroup.
// All threads iterate over column blocks SYNCHRONOUSLY.
// Per block: cooperatively load 32 x values into shared memory,
// then each thread decodes its row's weight block and dots with cached x.
// This amortizes x vector reads across TILE_ROWS rows.
//
// With TILE_ROWS=32 and dim=2048: 64 workgroups instead of 2048.
// x vector bandwidth: 32x reduction.
//
// Dispatch: (ceil(rows / TILE_ROWS), n_tokens, 1)

const TILE_ROWS: u32 = 32u;
const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 8u;
const ELEMS_PER_THREAD: u32 = 32u / THREADS_PER_ROW;  // WG_SIZE / TILE_ROWS = 256/32 = 8

struct Uniforms {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    extra: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// Shared memory: cached x values for current block (32 floats)
var<workgroup> x_cache: array<f32, 32>;
// Shared memory: per-thread partial sums for reduction
var<workgroup> reduce_buf: array<f32, 256>;

fn fp16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    if (exp == 0u && mant == 0u) {
        return select(0.0, -0.0, sign == 1u);
    }
    let f_bits = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    return bitcast<f32>(f_bits);
}

fn read_byte(addr: u32) -> u32 {
    return (weights[addr >> 2u] >> ((addr & 3u) * 8u)) & 0xFFu;
}

fn read_u16_at(addr: u32) -> u32 {
    let lo = read_byte(addr);
    let hi = read_byte(addr + 1u);
    return lo | (hi << 8u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_start = select(wid.x * TILE_ROWS, (wid.x + wid.y * uniforms.extra) * TILE_ROWS, uniforms.extra > 0u);
    let token = select(wid.y, 0u, uniforms.extra > 0u);
    let tid = lid.x;

    // Thread's assigned row and sub-element index within each block
    let local_row = tid / THREADS_PER_ROW;    // 0..31 (which row in the tile)
    let local_elem = tid % THREADS_PER_ROW;   // 0..7  (which element group)
    let global_row = tile_start + local_row;

    let cols = uniforms.cols;
    let blocks_per_row = cols / 32u;
    let x_base = token * cols;

    // Per-row byte offset for weight data
    let row_byte_base = global_row * blocks_per_row * 18u;

    // Each thread accumulates partial sum for its row
    var acc: f32 = 0.0;

    // Synchronous iteration: ALL threads process the same block at the same time
    for (var b = 0u; b < blocks_per_row; b++) {
        // Step 1: Cooperatively load x values for block b into shared memory.
        // 32 values to load, 256 threads available. First 32 threads load.
        if (tid < 32u) {
            let col = b * 32u + tid;
            x_cache[tid] = select(0.0, x[x_base + col], col < cols);
        }
        workgroupBarrier();

        // Step 2: Each thread processes its portion of the block for its row.
        // 8 threads per row, each handles 4 elements (32 / 8 = 4).
        if (global_row < uniforms.rows) {
            let block_byte = row_byte_base + b * 18u;
            let scale = fp16_to_f32(read_u16_at(block_byte));

            // This thread handles elements [local_elem*4 .. local_elem*4+3]
            // within the 32-element block.
            // Q4_0 layout: low nibbles = elements 0..15, high nibbles = 16..31
            let my_start = local_elem * ELEMS_PER_THREAD;

            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                let elem = my_start + i;
                // Determine which byte and nibble this element is in
                var val: f32;
                if (elem < 16u) {
                    // Low nibble of qs[elem]
                    let byte_val = read_byte(block_byte + 2u + elem);
                    val = f32(i32(byte_val & 0xFu) - 8);
                } else {
                    // High nibble of qs[elem - 16]
                    let byte_val = read_byte(block_byte + 2u + elem - 16u);
                    val = f32(i32((byte_val >> 4u) & 0xFu) - 8);
                }
                acc += scale * val * x_cache[elem];
            }
        }
        workgroupBarrier();  // Ensure all threads done before next x_cache load
    }

    // Step 3: Reduce THREADS_PER_ROW (8) partial sums per row → 1 value
    reduce_buf[tid] = acc;
    workgroupBarrier();

    // Generic tree reduction within each row's thread lane
    let row_base = local_row * THREADS_PER_ROW;
    for (var s = THREADS_PER_ROW / 2u; s > 0u; s >>= 1u) {
        if (local_elem < s) {
            reduce_buf[row_base + local_elem] += reduce_buf[row_base + local_elem + s];
        }
        workgroupBarrier();
    }

    // Step 4: First thread of each row writes the result
    if (local_elem == 0u && global_row < uniforms.rows) {
        out[token * uniforms.rows + global_row] = reduce_buf[row_base];
    }
}
