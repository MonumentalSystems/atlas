// SPDX-License-Identifier: AGPL-3.0-only

// Prefill matrix multiply over native GGUF block_q8_0 weights.

#include <metal_stdlib>
using namespace metal;

constant uint QK = 32u;
constant uint BLOCK_BYTES = 34u;

kernel void gguf_q8_0_gemm(
    device const bfloat *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device bfloat *y [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.y;
    const uint out_col = gid.x;
    if (row >= m || out_col >= n) {
        return;
    }

    const uint blocks_per_row = k / QK;
    float sum = 0.0f;
    for (uint block = 0; block < blocks_per_row; ++block) {
        device const uchar *p = weights + (out_col * blocks_per_row + block) * BLOCK_BYTES;
        const float d = float(*reinterpret_cast<device const half *>(p));
        device const char *q = reinterpret_cast<device const char *>(p + 2u);
        const uint col = block * QK;
        for (uint i = 0; i < QK; ++i) {
            sum += float(x[row * k + col + i]) * d * float(q[i]);
        }
    }
    y[row * n + out_col] = bfloat(sum);
}
