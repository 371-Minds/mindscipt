// Conv1d (kernel=4) + SiLU activation + conv_state shift.
// One thread per channel. Processes all channels in parallel.
// Dispatch: (ceil(qkv_dim/256), 1, 1)

struct Uniforms {
    p0: u32, p1: u32, p2: u32, p3: u32,
    p4: u32, p5: u32, p6: u32, p7: u32,
}

@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(2) var<storage, read> conv1d_w: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

// p0 = qkv_dim, p1 = kern (typically 4), p2 = conv_state_offset (floats), p3 = conv_state_layer_size (floats)

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let ch = wid.x * 256u + lid.x;
    let qkv_dim = u.p0;
    let kern = u.p1;
    let cs_off = u.p2;  // per-layer conv_state float offset

    if (ch >= qkv_dim) {
        return;
    }

    // Conv1d: sum over conv_state history + current input
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < kern - 1u; k++) {
        sum += conv_state[cs_off + k * qkv_dim + ch] * conv1d_w[ch * kern + k];
    }
    let cur = qkv[ch];
    sum += cur * conv1d_w[ch * kern + kern - 1u];

    // Shift conv_state for this channel
    for (var k: u32 = 0u; k < kern - 2u; k++) {
        conv_state[cs_off + k * qkv_dim + ch] = conv_state[cs_off + (k + 1u) * qkv_dim + ch];
    }
    conv_state[cs_off + (kern - 2u) * qkv_dim + ch] = cur;

    // SiLU activation
    qkv[ch] = sum / (1.0 + exp(-sum));
}
