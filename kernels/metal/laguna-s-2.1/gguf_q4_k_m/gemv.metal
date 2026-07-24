// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void dense_gemv_bf16(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[32];
    if (row >= n) return;
    float sum = 0.0f;
    for (uint col = tid; col < k; col += 256u) {
        sum += float(input[col]) * float(weight[ulong(row) * k + col]);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float v = tid < 8u ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0u) output[row] = bfloat(v);
    }
}
