#include <metal_stdlib>
using namespace metal;

// Buffer-to-buffer copy as a compute shader.
// Stays in the same compute encoder — no blit encoder transition.
//
// p0 = src_offset (float index), p1 = dst_offset (float index), p2 = count (floats)
//
// Dispatch: (ceil(count/256), 1, 1)

kernel void buf_copy(device const float *src [[buffer(0)]],
                     device float       *dst [[buffer(1)]],
                     constant uint      *p   [[buffer(2)]],
                     uint3 wid [[threadgroup_position_in_grid]],
                     uint3 lid [[thread_position_in_threadgroup]]) {
    uint src_off = p[0];
    uint dst_off = p[1];
    uint count   = p[2];

    uint i = wid.x * 256 + lid.x;
    if (i < count) {
        dst[dst_off + i] = src[src_off + i];
    }
}
