// Buffer-to-buffer copy as a compute shader.
// Stays in the same compute pass — no pass boundary transition.
//
// p0 = src_offset (float index), p1 = dst_offset (float index), p2 = count (floats)
//
// Dispatch: (ceil(count/256), 1, 1)

struct Uniforms {
    src_off: u32,
    dst_off: u32,
    count: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
    _pad7: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = wid.x * 256u + lid.x;
    if (i < uniforms.count) {
        dst[uniforms.dst_off + i] = src[uniforms.src_off + i];
    }
}
