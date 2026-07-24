// SPDX-License-Identifier: AGPL-3.0-only

#ifndef ATLAS_GGUF_Q4_K_H
#define ATLAS_GGUF_Q4_K_H

#include <metal_stdlib>
using namespace metal;

constant uint GGUF_Q4_K_VALUES = 256u;
constant uint GGUF_Q4_K_BYTES = 144u;

inline void gguf_q4_k_scale_min(
    uint group,
    device const uchar *packed,
    thread uchar &scale,
    thread uchar &minimum)
{
    if (group < 4u) {
        scale = packed[group] & 63u;
        minimum = packed[group + 4u] & 63u;
    } else {
        scale = (packed[group + 4u] & 15u) | ((packed[group - 4u] >> 6u) << 4u);
        minimum = (packed[group + 4u] >> 4u) | ((packed[group] >> 6u) << 4u);
    }
}

// One SIMD lane owns one value from each of the eight 32-value groups.
// The caller reduces the returned partial sum with simd_sum().
inline float gguf_q4_k_dot_block(
    device const uchar *block,
    device const bfloat *input,
    uint lane)
{
    const float d = float(*reinterpret_cast<device const half *>(block));
    const float dmin = float(*reinterpret_cast<device const half *>(block + 2u));
    device const uchar *packed_scales = block + 4u;
    device const uchar *quants = block + 16u;
    float sum = 0.0f;
    for (uint group = 0u; group < 8u; ++group) {
        uchar scale;
        uchar minimum;
        gguf_q4_k_scale_min(group, packed_scales, scale, minimum);
        const uchar byte = quants[(group >> 1u) * 32u + lane];
        const uchar quant = (group & 1u) != 0u ? byte >> 4u : byte & 15u;
        const float weight = d * float(scale) * float(quant) - dmin * float(minimum);
        sum += float(input[group * 32u + lane]) * weight;
    }
    return sum;
}

#endif
