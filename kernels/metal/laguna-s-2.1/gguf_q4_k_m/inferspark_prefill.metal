// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

inline void contiguous_prefill(
    device const bfloat *q, device const bfloat *k, device const bfloat *v,
    device bfloat *output, uint seq_len, uint num_q_heads, uint num_kv_heads,
    uint head_dim, float scale, uint causal, uint sliding_window,
    uint rows_per_group, uint3 group, uint tid)
{
    if (tid >= 32 || group.x >= num_q_heads) return;
    const uint q_start = group.y * rows_per_group;
    const uint kv_head = group.x / (num_q_heads / num_kv_heads);
    const uint owned = (head_dim + 31u) / 32u;
    const ulong q_batch = ulong(group.z) * seq_len * num_q_heads * head_dim;
    const ulong kv_batch = ulong(group.z) * seq_len * num_kv_heads * head_dim;
    for (uint query = q_start; query < min(q_start + rows_per_group, seq_len); ++query) {
        const uint end = causal != 0 ? query + 1u : seq_len;
        const uint begin = sliding_window != 0 && end > sliding_window
            ? end - sliding_window : 0;
        const ulong q_base = q_batch + (ulong(query) * num_q_heads + group.x) * head_dim;
        float row_max = -INFINITY;
        float row_sum = 0.0f;
        float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint position = begin; position < end; ++position) {
            const ulong kv_base = kv_batch
                + (ulong(position) * num_kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (uint item = 0; item < owned; ++item) {
                const uint dimension = tid + item * 32u;
                if (dimension < head_dim) {
                    partial += float(q[q_base + dimension]) * float(k[kv_base + dimension]);
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
                        + new_factor * float(v[kv_base + dimension]);
                }
            }
            row_max = next_max;
        }
        for (uint item = 0; item < owned; ++item) {
            const uint dimension = tid + item * 32u;
            if (dimension < head_dim) output[q_base + dimension] = bfloat(accumulator[item] / row_sum);
        }
    }
}

#define DEFINE_CONTIGUOUS_PREFILL(NAME, ROWS) \
kernel void NAME( \
    device const bfloat *q [[buffer(0)]], device const bfloat *k [[buffer(1)]], \
    device const bfloat *v [[buffer(2)]], device bfloat *o [[buffer(3)]], \
    constant uint &sl [[buffer(4)]], constant uint &nq [[buffer(5)]], \
    constant uint &nkv [[buffer(6)]], constant uint &hd [[buffer(7)]], \
    constant float &scale [[buffer(8)]], constant uint &causal [[buffer(9)]], \
    constant uint &sw [[buffer(10)]], uint3 group [[threadgroup_position_in_grid]], \
    uint3 tid3 [[thread_position_in_threadgroup]]) \
{ contiguous_prefill(q, k, v, o, sl, nq, nkv, hd, scale, causal, sw, ROWS, group, tid3.x); }

DEFINE_CONTIGUOUS_PREFILL(inferspark_prefill, 32u)
DEFINE_CONTIGUOUS_PREFILL(inferspark_prefill_64, 64u)
