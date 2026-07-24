// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void argmax_bf16(
    device const bfloat *logits [[buffer(0)]],
    device uint *result [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial_values[32];
    threadgroup uint partial_indices[32];
    float best = -INFINITY;
    uint best_index = 0;
    for (uint index = tid; index < count; index += threads) {
        const float value = float(logits[index]);
        if (value > best || (value == best && index < best_index)) {
            best = value;
            best_index = index;
        }
    }
    for (uint offset = 16; offset != 0; offset >>= 1) {
        const float other = simd_shuffle_xor(best, offset);
        const uint other_index = simd_shuffle_xor(best_index, offset);
        if (other > best || (other == best && other_index < best_index)) {
            best = other;
            best_index = other_index;
        }
    }
    if (lane == 0) {
        partial_values[simd] = best;
        partial_indices[simd] = best_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const uint groups = (threads + 31u) / 32u;
        best = tid < groups ? partial_values[tid] : -INFINITY;
        best_index = tid < groups ? partial_indices[tid] : 0u;
        for (uint offset = 16; offset != 0; offset >>= 1) {
            const float other = simd_shuffle_xor(best, offset);
            const uint other_index = simd_shuffle_xor(best_index, offset);
            if (other > best || (other == best && other_index < best_index)) {
                best = other;
                best_index = other_index;
            }
        }
        if (tid == 0) result[0] = best_index;
    }
}
