// SPDX-License-Identifier: AGPL-3.0-only

// Decode matrix-vector product over native GGUF block_q8_0 weights.

#include <metal_stdlib>
using namespace metal;

constant uint QK = 32u;
constant uint BLOCK_BYTES = 34u;
constant uint ROWS_PER_TG = 4u;

kernel void gguf_q8_0_gemv(
    device const bfloat *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device bfloat *y [[buffer(2)]],
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

    const uint blocks_per_row = k / QK;
    float sum = 0.0f;
    for (uint block = lane; block < blocks_per_row; block += 32u) {
        device const uchar *p = weights + (row * blocks_per_row + block) * BLOCK_BYTES;
        const float d = float(*reinterpret_cast<device const half *>(p));
        device const char *q = reinterpret_cast<device const char *>(p + 2u);
        const uint col = block * QK;
        for (uint i = 0; i < QK; ++i) {
            sum += d * float(q[i]) * float(x[col + i]);
        }
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        y[row] = bfloat(sum);
    }
}
