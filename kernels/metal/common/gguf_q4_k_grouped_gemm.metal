// SPDX-License-Identifier: AGPL-3.0-only

// Grouped projection over one contiguous GGUF Q4_K expert stack. Each input
// slot selects an expert by integer id; negative ids produce a zero row.

#include "gguf_q4_k.h"

constant uint ROWS_PER_TG = 4u;

kernel void gguf_q4_k_grouped_gemm(
    device const bfloat *input [[buffer(0)]],
    device const uchar *expert_weights [[buffer(1)]],
    device const int *expert_ids [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant uint &total_slots [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant uint &k [[buffer(6)]],
    constant uint &num_experts [[buffer(7)]],
    constant ulong &expert_stride_bytes [[buffer(8)]],
    uint2 tg_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_idx.x;
    const uint row = tg_idx.y * ROWS_PER_TG + simd_idx;
    if (slot >= total_slots || row >= n) {
        return;
    }
    const int expert = expert_ids[slot];
    if (expert < 0 || uint(expert) >= num_experts) {
        if (lane == 0u) {
            output[ulong(slot) * ulong(n) + row] = bfloat(0.0f);
        }
        return;
    }
    const uint blocks_per_row = k / GGUF_Q4_K_VALUES;
    device const uchar *row_weights = expert_weights
        + ulong(expert) * expert_stride_bytes
        + ulong(row) * ulong(blocks_per_row) * ulong(GGUF_Q4_K_BYTES);
    device const bfloat *input_row = input + ulong(slot) * ulong(k);
    float sum = 0.0f;
    for (uint block = 0u; block < blocks_per_row; ++block) {
        sum += gguf_q4_k_dot_block(
            row_weights + ulong(block) * ulong(GGUF_Q4_K_BYTES),
            input_row + block * GGUF_Q4_K_VALUES,
            lane);
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        output[ulong(slot) * ulong(n) + row] = bfloat(sum);
    }
}
