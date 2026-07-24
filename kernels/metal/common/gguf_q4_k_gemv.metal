// SPDX-License-Identifier: AGPL-3.0-only

// Decode matrix-vector product over native GGUF block_q4_K weights.

#include "gguf_q4_k.h"

constant uint ROWS_PER_TG = 4u;

kernel void gguf_q4_k_gemv(
    device const bfloat *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &k [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG + simd_idx;
    if (row >= n) {
        return;
    }
    const uint blocks_per_row = k / GGUF_Q4_K_VALUES;
    device const uchar *row_weights =
        weights + ulong(row) * ulong(blocks_per_row) * ulong(GGUF_Q4_K_BYTES);
    float sum = 0.0f;
    for (uint block = 0u; block < blocks_per_row; ++block) {
        sum += gguf_q4_k_dot_block(
            row_weights + ulong(block) * ulong(GGUF_Q4_K_BYTES),
            input + block * GGUF_Q4_K_VALUES,
            lane);
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        output[row] = bfloat(sum);
    }
}
