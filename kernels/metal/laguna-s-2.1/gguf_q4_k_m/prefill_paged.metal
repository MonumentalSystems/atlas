// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

inline void paged_prefill(
    device const bfloat *q,
    device const bfloat *k_cache,
    device const bfloat *v_cache,
    device bfloat *output,
    device const int *block_table,
    uint q_len,
    uint kv_len,
    uint q_offset,
    uint num_q_heads,
    uint num_kv_heads,
    uint head_dim,
    uint block_size,
    uint sliding_window,
    uint causal,
    float scale,
    uint physical_blocks,
    uint rows_per_group,
    uint2 group,
    uint tid)
{
    if (tid >= 32 || group.x >= num_q_heads || physical_blocks == 0) return;
    const uint q_start = group.y * rows_per_group;
    const uint kv_head = group.x / (num_q_heads / num_kv_heads);
    const uint owned = (head_dim + 31u) / 32u;
    for (uint query = q_start; query < min(q_start + rows_per_group, q_len); ++query) {
        const uint absolute_query = q_offset + query;
        const uint end = causal != 0 ? min(absolute_query + 1u, kv_len) : kv_len;
        const uint begin = sliding_window != 0 && end > sliding_window
            ? end - sliding_window : 0;
        const ulong q_base = (ulong(query) * num_q_heads + group.x) * head_dim;
        float row_max = -INFINITY;
        float row_sum = 0.0f;
        float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint position = begin; position < end; ++position) {
            const uint logical = position / block_size;
            const uint block = uint(block_table[logical]) % physical_blocks;
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
                output[q_base + dimension] = bfloat(accumulator[item] / row_sum);
            }
        }
    }
}

#define DEFINE_PAGED_PREFILL(NAME, ROWS) \
kernel void NAME( \
    device const bfloat *q [[buffer(0)]], device const bfloat *k [[buffer(1)]], \
    device const bfloat *v [[buffer(2)]], device bfloat *o [[buffer(3)]], \
    device const int *table [[buffer(4)]], constant uint &ql [[buffer(5)]], \
    constant uint &kl [[buffer(6)]], constant uint &qo [[buffer(7)]], \
    constant uint &nq [[buffer(8)]], constant uint &nkv [[buffer(9)]], \
    constant uint &hd [[buffer(10)]], constant uint &bs [[buffer(11)]], \
    constant uint &sw [[buffer(12)]], constant uint &causal [[buffer(13)]], \
    constant float &scale [[buffer(14)]], constant uint &physical [[buffer(15)]], \
    uint2 group [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]]) \
{ paged_prefill(q, k, v, o, table, ql, kl, qo, nq, nkv, hd, bs, sw, causal, \
    scale, physical, ROWS, group, tid2.x); }

DEFINE_PAGED_PREFILL(inferspark_prefill_paged, 32u)
DEFINE_PAGED_PREFILL(inferspark_prefill_paged_64, 64u)
