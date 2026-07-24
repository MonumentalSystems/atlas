// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void dense_gemm_bf16(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= m || gid.x >= n) return;
    float sum = 0.0f;
    for (uint col = 0; col < k; ++col) {
        sum += float(input[ulong(gid.y) * k + col])
            * float(weight[ulong(gid.x) * k + col]);
    }
    output[ulong(gid.y) * n + gid.x] = bfloat(sum);
}
