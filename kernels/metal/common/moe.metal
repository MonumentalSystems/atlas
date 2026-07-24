// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void moe_permute_tokens(
    device const bfloat *hidden_states [[buffer(0)]],
    device bfloat *permuted [[buffer(1)]],
    device const int *sorted_token_ids [[buffer(2)]],
    constant uint &hidden [[buffer(3)]],
    constant uint &total_expanded [[buffer(4)]],
    uint slot [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (slot >= total_expanded) {
        return;
    }
    const int token = sorted_token_ids[slot];
    for (uint col = tid; col < hidden; col += threads) {
        permuted[ulong(slot) * hidden + col] = token < 0
            ? bfloat(0.0f)
            : hidden_states[ulong(token) * hidden + col];
    }
}

// Correctness-first stable counting sort for routed-token slots.  Laguna's
// first Metal release is batch-1/eager, so one lane doing the small
// (num_tokens * top-k) control-plane sort is preferable to device pointer
// tables or a CPU synchronization point.
kernel void moe_sort_by_expert(
    device const uint *topk_ids [[buffer(0)]],
    device int *sorted_token_ids [[buffer(1)]],
    device int *sorted_expert_ids [[buffer(2)]],
    device int *expert_offsets [[buffer(3)]],
    device int *token_to_perm [[buffer(4)]],
    constant uint &total_expanded [[buffer(5)]],
    constant uint &num_experts [[buffer(6)]],
    constant uint &topk [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0) {
        return;
    }

    expert_offsets[0] = 0;
    for (uint expert = 0; expert < num_experts; ++expert) {
        uint count = 0;
        for (uint slot = 0; slot < total_expanded; ++slot) {
            count += topk_ids[slot] == expert;
        }
        expert_offsets[expert + 1] = expert_offsets[expert] + int(count);
    }

    for (uint expert = 0; expert < num_experts; ++expert) {
        uint position = uint(expert_offsets[expert]);
        for (uint slot = 0; slot < total_expanded; ++slot) {
            if (topk_ids[slot] == expert) {
                sorted_token_ids[position] = int(slot / topk);
                sorted_expert_ids[position] = int(expert);
                token_to_perm[slot] = int(position);
                ++position;
            }
        }
    }
}

kernel void moe_unpermute_reduce_indexed(
    device const bfloat *expert_output [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    device const int *token_to_perm [[buffer(2)]],
    device const float *topk_weights [[buffer(3)]],
    constant uint &hidden_size [[buffer(4)]],
    constant uint &num_tokens [[buffer(5)]],
    constant uint &topk [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (token >= num_tokens) {
        return;
    }
    for (uint column = tid; column < hidden_size; column += threads) {
        float value = 0.0f;
        for (uint k = 0; k < topk; ++k) {
            const uint slot = token * topk + k;
            const int row = token_to_perm[slot];
            value += topk_weights[slot]
                * float(expert_output[ulong(row) * hidden_size + column]);
        }
        output[ulong(token) * hidden_size + column] = bfloat(value);
    }
}

kernel void moe_batched_blend(
    device bfloat *output [[buffer(0)]],
    device const bfloat *shared_out [[buffer(1)]],
    device const bfloat *normed [[buffer(2)]],
    device const bfloat *gate_weight [[buffer(3)]],
    constant uint &hidden_size [[buffer(4)]],
    constant uint &num_tokens [[buffer(5)]],
    constant uint &has_gate [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (token >= num_tokens) {
        return;
    }

    threadgroup float gate;
    if (tid == 0) {
        if (has_gate == 0) {
            gate = 1.0f;
        } else {
            float dot = 0.0f;
            for (uint column = 0; column < hidden_size; ++column) {
                dot += float(normed[ulong(token) * hidden_size + column])
                    * float(gate_weight[column]);
            }
            gate = 1.0f / (1.0f + exp(-dot));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint column = tid; column < hidden_size; column += threads) {
        const ulong index = ulong(token) * hidden_size + column;
        output[index] = bfloat(float(output[index]) + gate * float(shared_out[index]));
    }
}
