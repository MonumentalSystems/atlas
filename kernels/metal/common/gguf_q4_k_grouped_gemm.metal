// SPDX-License-Identifier: AGPL-3.0-only

// Grouped projection over one contiguous GGUF Q4_K expert stack. Prefill
// consumes 16 sorted routed-token rows per threadgroup and uses 8x8 SIMD-group
// matrix operations. Decode retains the one-slot SIMD dot-product path.

#include "gguf_q4_k.h"
#include <metal_simdgroup_matrix>

constant uint DECODE_ROWS_PER_TG = 4u;
constant uint PREFILL_TILE_M = 16u;
constant uint PREFILL_TILE_N = 16u;
constant uint PREFILL_TILE_K = 32u;
constant uint MAX_SLOTS_PER_TG = 16u;

inline float gguf_q4_k_value(device const uchar *row_weights, uint col)
{
    const uint block_idx = col / GGUF_Q4_K_VALUES;
    const uint in_block = col % GGUF_Q4_K_VALUES;
    const uint group = in_block / 32u;
    const uint value_lane = in_block % 32u;
    device const uchar *block = row_weights
        + ulong(block_idx) * ulong(GGUF_Q4_K_BYTES);
    const float d = float(*reinterpret_cast<device const half *>(block));
    const float dmin = float(*reinterpret_cast<device const half *>(block + 2u));
    device const uchar *packed_scales = block + 4u;
    device const uchar *quants = block + 16u;
    uchar scale;
    uchar minimum;
    gguf_q4_k_scale_min(group, packed_scales, scale, minimum);
    const uchar byte = quants[(group >> 1u) * 32u + value_lane];
    const uchar quant = (group & 1u) != 0u ? byte >> 4u : byte & 15u;
    return d * float(scale) * float(quant) - dmin * float(minimum);
}

kernel void gguf_q4_k_grouped_gemm(
    device const bfloat *input [[buffer(0)]],
    device const uchar *expert_weights [[buffer(1)]],
    device const int *expert_ids [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant uint &total_slots [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant uint &k [[buffer(6)]],
    constant uint &num_experts [[buffer(7)]],
    constant ulong &expert_stride_bytes [[buffer(8)]],
    constant uint &slots_per_tg [[buffer(9)]],
    uint2 tg_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]])
{
    const uint slot_base = tg_idx.x * slots_per_tg;
    if (slot_base >= total_slots) {
        return;
    }
    const uint tile_slots = min(slots_per_tg, total_slots - slot_base);
    const int first_expert = expert_ids[slot_base];
    bool one_expert = slots_per_tg > 1u
        && first_expert >= 0
        && uint(first_expert) < num_experts;
    for (uint local = 1u; local < tile_slots; ++local) {
        one_expert = one_expert && expert_ids[slot_base + local] == first_expert;
    }

    const uint blocks_per_row = k / GGUF_Q4_K_VALUES;
    if (one_expert) {
        const uint out_base = tg_idx.y * PREFILL_TILE_N;
        threadgroup half a_tile[PREFILL_TILE_M][PREFILL_TILE_K];
        threadgroup half b_tile[PREFILL_TILE_K][PREFILL_TILE_N];

        const short qid = short(lane / 4u);
        const short frag_row = (qid & 4) + short((lane / 2u) % 4u);
        const short frag_col = (qid & 2) * 2 + short(lane % 2u) * 2;
        const uint simd_m = (simd_idx / 2u) * 8u;
        const uint simd_n = (simd_idx % 2u) * 8u;
        simdgroup_matrix<float, 8, 8> accum;
        accum.thread_elements()[0] = 0.0f;
        accum.thread_elements()[1] = 0.0f;

        for (uint k_base = 0u; k_base < k; k_base += PREFILL_TILE_K) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < PREFILL_TILE_M * PREFILL_TILE_K; idx += 128u) {
                const uint local_m = idx / PREFILL_TILE_K;
                const uint local_k = idx % PREFILL_TILE_K;
                a_tile[local_m][local_k] = local_m < tile_slots
                    ? half(input[ulong(slot_base + local_m) * ulong(k) + k_base + local_k])
                    : half(0.0f);
            }
            for (uint idx = tid; idx < PREFILL_TILE_K * PREFILL_TILE_N; idx += 128u) {
                const uint local_k = idx / PREFILL_TILE_N;
                const uint local_n = idx % PREFILL_TILE_N;
                const uint out_row = out_base + local_n;
                half value = half(0.0f);
                if (out_row < n) {
                    device const uchar *row_weights = expert_weights
                        + ulong(first_expert) * expert_stride_bytes
                        + ulong(out_row) * ulong(blocks_per_row) * ulong(GGUF_Q4_K_BYTES);
                    value = half(gguf_q4_k_value(row_weights, k_base + local_k));
                }
                b_tile[local_k][local_n] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_matrix<half, 8, 8> b_frag;
            for (uint kk = 0u; kk < PREFILL_TILE_K; kk += 8u) {
                a_frag.thread_elements()[0] =
                    a_tile[simd_m + uint(frag_row)][kk + uint(frag_col)];
                a_frag.thread_elements()[1] =
                    a_tile[simd_m + uint(frag_row)][kk + uint(frag_col) + 1u];
                b_frag.thread_elements()[0] =
                    b_tile[kk + uint(frag_row)][simd_n + uint(frag_col)];
                b_frag.thread_elements()[1] =
                    b_tile[kk + uint(frag_row)][simd_n + uint(frag_col) + 1u];
                simdgroup_multiply_accumulate(accum, a_frag, b_frag, accum);
            }
        }

        const uint local_m = simd_m + uint(frag_row);
        const uint local_n = simd_n + uint(frag_col);
        if (local_m < tile_slots && out_base + local_n < n) {
            output[ulong(slot_base + local_m) * ulong(n) + out_base + local_n]
                = bfloat(accum.thread_elements()[0]);
        }
        if (local_m < tile_slots && out_base + local_n + 1u < n) {
            output[ulong(slot_base + local_m) * ulong(n) + out_base + local_n + 1u]
                = bfloat(accum.thread_elements()[1]);
        }
        return;
    }

    // Decode and the occasional prefill tile crossing an expert-run boundary.
    // Four SIMD groups cover four rows at a time; mixed prefill tiles loop over
    // the remaining rows while preserving the exact scalar accumulation path.
    const uint rows_per_tg = slots_per_tg == 1u
        ? DECODE_ROWS_PER_TG
        : PREFILL_TILE_N;
    const uint row_base = tg_idx.y * rows_per_tg;
    for (uint local_row = simd_idx; local_row < rows_per_tg; local_row += 4u) {
        const uint row = row_base + local_row;
        if (row >= n) {
            continue;
        }
        for (uint local = 0u; local < tile_slots; ++local) {
            const uint slot = slot_base + local;
            const int expert = expert_ids[slot];
            float sum = 0.0f;
            if (expert >= 0 && uint(expert) < num_experts) {
                device const uchar *row_weights = expert_weights
                    + ulong(expert) * expert_stride_bytes
                    + ulong(row) * ulong(blocks_per_row) * ulong(GGUF_Q4_K_BYTES);
                device const bfloat *input_row = input + ulong(slot) * ulong(k);
                for (uint block_idx = 0u; block_idx < blocks_per_row; ++block_idx) {
                    sum += gguf_q4_k_dot_block(
                        row_weights + ulong(block_idx) * ulong(GGUF_Q4_K_BYTES),
                        input_row + block_idx * GGUF_Q4_K_VALUES,
                        lane);
                }
                sum = simd_sum(sum);
            }
            if (lane == 0u) {
                output[ulong(slot) * ulong(n) + row] = bfloat(sum);
            }
        }
    }
}
