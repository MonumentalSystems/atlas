// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

inline void rotate_neox(device bfloat *values, uint pair, uint half_dim,
                        float cosine, float sine)
{
    const float first = float(values[pair]);
    const float second = float(values[pair + half_dim]);
    values[pair] = bfloat(first * cosine - second * sine);
    values[pair + half_dim] = bfloat(second * cosine + first * sine);
}

kernel void rope_forward(
    device bfloat *q [[buffer(0)]], device bfloat *k [[buffer(1)]],
    device const uint *positions [[buffer(2)]], constant uint &seq_len [[buffer(3)]],
    constant uint &num_q_heads [[buffer(4)]], constant uint &num_kv_heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]], constant uint &rotary_dim [[buffer(7)]],
    constant float &theta [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]])
{
    const uint tid = tid2.x;
    const uint half_dim = rotary_dim / 2u;
    const uint per_group = max(128u / half_dim, 1u);
    const uint local = tid / half_dim;
    const uint pair = tid % half_dim;
    const uint token = group.y * per_group + local;
    if (token >= seq_len || pair >= half_dim) return;
    const bool is_q = group.x < num_q_heads;
    const uint head = is_q ? group.x : group.x - num_q_heads;
    const uint heads = is_q ? num_q_heads : num_kv_heads;
    if (head >= heads) return;
    device bfloat *base = (is_q ? q : k) + (ulong(token) * heads + head) * head_dim;
    const float frequency = pow(theta, -2.0f * float(pair) / float(rotary_dim));
    const float angle = float(positions[token]) * frequency;
    rotate_neox(base, pair, half_dim, cos(angle), sin(angle));
}

kernel void rope_forward_yarn_scaled(
    device bfloat *q [[buffer(0)]], device bfloat *k [[buffer(1)]],
    device const uint *positions [[buffer(2)]], constant uint &seq_len [[buffer(3)]],
    constant uint &num_q_heads [[buffer(4)]], constant uint &num_kv_heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]], constant uint &rotary_dim [[buffer(7)]],
    device const float *inv_freq [[buffer(8)]], constant float &factor [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]])
{
    const uint tid = tid2.x;
    const uint half_dim = rotary_dim / 2u;
    const uint per_group = max(128u / half_dim, 1u);
    const uint local = tid / half_dim;
    const uint pair = tid % half_dim;
    const uint token = group.y * per_group + local;
    if (token >= seq_len || pair >= half_dim) return;
    const bool is_q = group.x < num_q_heads;
    const uint head = is_q ? group.x : group.x - num_q_heads;
    const uint heads = is_q ? num_q_heads : num_kv_heads;
    if (head >= heads) return;
    device bfloat *base = (is_q ? q : k) + (ulong(token) * heads + head) * head_dim;
    const float angle = float(positions[token]) * inv_freq[pair];
    rotate_neox(base, pair, half_dim, cos(angle) * factor, sin(angle) * factor);
}
