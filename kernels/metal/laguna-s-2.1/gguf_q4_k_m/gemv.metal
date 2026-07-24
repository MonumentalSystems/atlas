// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void dense_gemv_bf16(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &k [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    // The shared dense_gemv host ABI launches ceil(n / 4) threadgroups.
    // Assign one 64-thread pair of SIMD groups to each of those four rows.
    const uint row_in_group = tid / 64u;
    const uint row_tid = tid % 64u;
    const uint row = group * 4u + row_in_group;
    const uint row_simd = simd % 2u;
    const bool active = row < n;
    threadgroup float partial[4][2];
    float sum = 0.0f;
    if (active) {
        for (uint col = row_tid; col < k; col += 64u) {
            sum += float(input[col]) * float(weight[ulong(row) * k + col]);
        }
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[row_in_group][row_simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && row_simd == 0u && lane == 0u) {
        output[row] = bfloat(partial[row_in_group][0] + partial[row_in_group][1]);
    }
}
