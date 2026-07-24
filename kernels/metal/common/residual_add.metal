// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void bf16_residual_add(
    device bfloat *residual [[buffer(0)]],
    device const bfloat *src [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) residual[gid] = bfloat(float(residual[gid]) + float(src[gid]));
}

kernel void sigmoid_gate_mul(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) output[gid] = bfloat(float(input[gid]) / (1.0f + exp(-float(gate[gid]))));
}

kernel void sigmoid_gate_mul_head_broadcast(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &heads [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant uint &total [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total) return;
    const uint head = (gid / head_dim) % heads;
    const uint token = gid / (heads * head_dim);
    const float g = float(gate[token * heads + head]);
    output[gid] = bfloat(float(input[gid]) / (1.0f + exp(-g)));
}

kernel void softplus_gate_mul_head_broadcast(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &heads [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant uint &total [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total) return;
    const uint head = (gid / head_dim) % heads;
    const uint token = gid / (heads * head_dim);
    const float g = float(gate[token * heads + head]);
    const float softplus = g > 20.0f ? g : log(1.0f + exp(g));
    output[gid] = bfloat(float(input[gid]) * softplus);
}
