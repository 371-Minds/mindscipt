// Per-head RMSNorm + SiLU gate (one workgroup per V-head):
//   out[d] = rmsnorm(out[d], norm_w[d]) * silu(z[d])
// Dispatch: (num_v_heads, 1, 1)

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;     // [num_v_heads * hv]
@group(0) @binding(1) var<storage, read> z: array<f32>;             // [num_v_heads * hv]
@group(0) @binding(2) var<storage, read> norm_w: array<f32>;        // [hv]
@group(0) @binding(3) var<uniform> u: Uniforms;

// p0 = head_v_dim, p1 = eps (bitcast to f32)

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let hv_idx = wid.x;
    let tid = lid.x;
    let hv = u.p0;
    let eps = bitcast<f32>(u.p1);
    let base = hv_idx * hv;

    // Accumulate squared sum for RMSNorm
    var ss: f32 = 0.0;
    var d = tid;
    while (d < hv) {
        let val = out[base + d];
        ss += val * val;
        d += 256u;
    }
    shared_sum[tid] = ss;
    workgroupBarrier();

    // Reduction
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let inv_rms = 1.0 / sqrt(shared_sum[0] / f32(hv) + eps);
    workgroupBarrier();

    // Apply RMSNorm + SiLU gate
    d = tid;
    while (d < hv) {
        let normed = out[base + d] * inv_rms * norm_w[d];
        let g = z[base + d];
        out[base + d] = normed * (g / (1.0 + exp(-g)));
        d += 256u;
    }
}
