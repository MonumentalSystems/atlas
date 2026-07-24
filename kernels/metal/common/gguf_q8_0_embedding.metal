// SPDX-License-Identifier: AGPL-3.0-only

// Token embedding gather directly from native GGUF block_q8_0 rows.

#include <metal_stdlib>
using namespace metal;

constant uint QK = 32u;
constant uint BLOCK_BYTES = 34u;

kernel void gguf_q8_0_embedding(
    device const uint *token_ids [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &num_tokens [[buffer(3)]],
    constant uint &hidden_size [[buffer(4)]],
    constant uint &vocab_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint token_idx = gid.y;
    const uint hidden_idx = gid.x;
    if (token_idx >= num_tokens || hidden_idx >= hidden_size) {
        return;
    }
    const uint token = token_ids[token_idx];
    if (token >= vocab_size) {
        out[token_idx * hidden_size + hidden_idx] = bfloat(0.0f);
        return;
    }

    const uint blocks_per_row = hidden_size / QK;
    const uint block = hidden_idx / QK;
    device const uchar *p = weights + (token * blocks_per_row + block) * BLOCK_BYTES;
    const float d = float(*reinterpret_cast<device const half *>(p));
    device const char *q = reinterpret_cast<device const char *>(p + 2u);
    out[token_idx * hidden_size + hidden_idx] = bfloat(d * float(q[hidden_idx % QK]));
}
