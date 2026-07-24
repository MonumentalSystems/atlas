// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// Correction-biased sigmoid top-k.  Selection uses sigmoid(logit) + bias,
// while the emitted router weight is the unbiased sigmoid score, matching the
// CUDA and llama.cpp Laguna definitions exactly. One threadgroup handles one
// token; lane zero performs the small 256-expert selection deterministically.
kernel void moe_topk_sigmoid_batched(
    device const bfloat *gate_logits [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device uint *expert_indices [[buffer(2)]],
    device float *expert_weights [[buffer(3)]],
    constant uint &num_experts [[buffer(4)]],
    constant uint &top_k [[buffer(5)]],
    constant uint &normalize [[buffer(6)]],
    constant float &scaling_factor [[buffer(7)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0) {
        return;
    }

    const ulong input_base = ulong(token) * num_experts;
    const ulong output_base = ulong(token) * top_k;
    float sum = 0.0f;
    for (uint rank = 0; rank < top_k; ++rank) {
        float best_selection = -INFINITY;
        float best_score = 0.0f;
        uint best_expert = 0;
        for (uint expert = 0; expert < num_experts; ++expert) {
            bool already_selected = false;
            for (uint previous = 0; previous < rank; ++previous) {
                already_selected |= expert_indices[output_base + previous] == expert;
            }
            if (already_selected) {
                continue;
            }
            const float score = 1.0f
                / (1.0f + exp(-float(gate_logits[input_base + expert])));
            const float selection = score + bias[expert];
            if (selection > best_selection) {
                best_selection = selection;
                best_score = score;
                best_expert = expert;
            }
        }
        expert_indices[output_base + rank] = best_expert;
        expert_weights[output_base + rank] = best_score;
        sum += best_score;
    }

    const float multiplier = normalize != 0 && sum > 1.0e-20f
        ? scaling_factor / sum
        : scaling_factor;
    for (uint rank = 0; rank < top_k; ++rank) {
        expert_weights[output_base + rank] *= multiplier;
    }
}
