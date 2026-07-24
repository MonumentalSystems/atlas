// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void paged_decode_attn(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *k_cache [[buffer(1)]],
    device const bfloat *v_cache [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    device const int *block_tables [[buffer(4)]],
    device const int *seq_lens [[buffer(5)]],
    constant uint &max_blocks_per_seq [[buffer(6)]],
    constant uint &num_q_heads [[buffer(7)]],
    constant uint &num_kv_heads [[buffer(8)]],
    constant uint &head_dim [[buffer(9)]],
    constant uint &block_size [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    constant uint &q_stride [[buffer(12)]],
    constant uint &sliding_window [[buffer(13)]],
    constant uint &physical_blocks [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]])
{
    const uint tid = tid2.x;
    if (tid >= 32 || group.x >= num_q_heads || physical_blocks == 0) {
        return;
    }
    const uint sequence = group.y;
    const uint seq_len = uint(max(seq_lens[sequence], 0));
    if (seq_len == 0) {
        return;
    }
    const uint kv_head = group.x / (num_q_heads / num_kv_heads);
    const uint begin = sliding_window != 0 && seq_len > sliding_window
        ? seq_len - sliding_window : 0;
    const ulong q_base = ulong(sequence) * q_stride + ulong(group.x) * head_dim;
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint owned = (head_dim + 31u) / 32u;

    for (uint position = begin; position < seq_len; ++position) {
        const uint logical = position / block_size;
        const uint block = uint(block_tables[ulong(sequence) * max_blocks_per_seq + logical])
            % physical_blocks;
        const uint offset = position % block_size;
        const ulong cache_base = (ulong(block) * block_size + offset)
            * num_kv_heads * head_dim + ulong(kv_head) * head_dim;
        float partial = 0.0f;
        for (uint item = 0; item < owned; ++item) {
            const uint dimension = tid + item * 32u;
            if (dimension < head_dim) {
                partial += float(q[q_base + dimension])
                    * float(k_cache[cache_base + dimension]);
            }
        }
        const float score = simd_sum(partial) * scale;
        const float next_max = max(row_max, score);
        const float old_factor = exp(row_max - next_max);
        const float new_factor = exp(score - next_max);
        row_sum = row_sum * old_factor + new_factor;
        for (uint item = 0; item < owned; ++item) {
            const uint dimension = tid + item * 32u;
            if (dimension < head_dim) {
                accumulator[item] = accumulator[item] * old_factor
                    + new_factor * float(v_cache[cache_base + dimension]);
            }
        }
        row_max = next_max;
    }
    for (uint item = 0; item < owned; ++item) {
        const uint dimension = tid + item * 32u;
        if (dimension < head_dim) {
            output[(ulong(sequence) * num_q_heads + group.x) * head_dim + dimension]
                = bfloat(accumulator[item] / row_sum);
        }
    }
}
