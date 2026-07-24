// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

/// ABI-compatible with `layers::ops::silu_mul`.
kernel void moe_silu_mul(
    device const bfloat *gate [[buffer(0)]],
    device const bfloat *up [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    const float g = float(gate[gid]);
    output[gid] = bfloat((g / (1.0f + exp(-g))) * float(up[gid]));
}
