// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32u;

inline float rms_reduce(
    device const bfloat *input,
    uint hidden,
    uint tid,
    uint threads,
    uint lane,
    uint simd,
    threadgroup float *partial)
{
    float sum = 0.0f;
    for (uint col = tid; col < hidden; col += threads) {
        const float x = float(input[col]);
        sum += x * x;
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float v = tid < (threads + 31u) / 32u ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0u) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

kernel void rms_norm_vanilla(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS];
    const ulong base = ulong(token) * hidden;
    const float total = rms_reduce(input + base, hidden, tid, threads, lane, simd, partial);
    const float inv = rsqrt(total / float(hidden) + eps);
    for (uint col = tid; col < hidden; col += threads) {
        output[base + col] = bfloat(float(input[base + col]) * float(weight[col]) * inv);
    }
}

kernel void rms_norm_residual_vanilla(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    device bfloat *residual [[buffer(3)]],
    constant uint &hidden [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS];
    const ulong base = ulong(token) * hidden;
    const float total = rms_reduce(input + base, hidden, tid, threads, lane, simd, partial);
    const float inv = rsqrt(total / float(hidden) + eps);
    for (uint col = tid; col < hidden; col += threads) {
        const bfloat x = input[base + col];
        residual[base + col] = x;
        output[base + col] = bfloat(float(x) * float(weight[col]) * inv);
    }
}

kernel void residual_add_rms_norm_vanilla(
    device bfloat *hidden [[buffer(0)]],
    device const bfloat *src [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    device bfloat *residual [[buffer(4)]],
    constant uint &hidden_size [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS];
    const ulong base = ulong(token) * hidden_size;
    for (uint col = tid; col < hidden_size; col += threads) {
        hidden[base + col] = bfloat(float(hidden[base + col]) + float(src[base + col]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = rms_reduce(
        hidden + base, hidden_size, tid, threads, lane, simd, partial);
    const float inv = rsqrt(total / float(hidden_size) + eps);
    for (uint col = tid; col < hidden_size; col += threads) {
        const bfloat value = hidden[base + col];
        residual[base + col] = value;
        output[base + col] = bfloat(float(value) * float(weight[col]) * inv);
    }
}
