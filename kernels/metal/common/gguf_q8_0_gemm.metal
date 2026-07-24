// SPDX-License-Identifier: AGPL-3.0-only

// Prefill matrix multiply over native GGUF block_q8_0 weights. Four SIMD
// groups cooperatively produce one 16x16 output tile using 8x8 matrix MMA.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint QK = 32u;
constant uint BLOCK_BYTES = 34u;
constant uint TILE_M = 16u;
constant uint TILE_N = 16u;
constant uint TILE_K = 32u;

inline float q8_0_value(device const uchar *row_weights, uint col)
{
    device const uchar *block = row_weights
        + ulong(col / QK) * ulong(BLOCK_BYTES);
    const float scale = float(*reinterpret_cast<device const half *>(block));
    device const char *quants = reinterpret_cast<device const char *>(block + 2u);
    return scale * float(quants[col % QK]);
}

kernel void gguf_q8_0_gemm(
    device const bfloat *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device bfloat *y [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    uint2 tg_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]])
{
    const uint m_base = tg_idx.y * TILE_M;
    const uint n_base = tg_idx.x * TILE_N;
    if (m_base >= m || n_base >= n) {
        return;
    }

    threadgroup half a_tile[TILE_M][TILE_K];
    threadgroup half b_tile[TILE_K][TILE_N];
    const short qid = short(lane / 4u);
    const short frag_row = (qid & 4) + short((lane / 2u) % 4u);
    const short frag_col = (qid & 2) * 2 + short(lane % 2u) * 2;
    const uint simd_m = (simd_idx / 2u) * 8u;
    const uint simd_n = (simd_idx % 2u) * 8u;
    const uint blocks_per_row = k / QK;
    simdgroup_matrix<float, 8, 8> accum;
    accum.thread_elements()[0] = 0.0f;
    accum.thread_elements()[1] = 0.0f;

    for (uint k_base = 0u; k_base < k; k_base += TILE_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < TILE_M * TILE_K; idx += 128u) {
            const uint local_m = idx / TILE_K;
            const uint local_k = idx % TILE_K;
            a_tile[local_m][local_k] = m_base + local_m < m
                ? half(x[ulong(m_base + local_m) * ulong(k) + k_base + local_k])
                : half(0.0f);
        }
        for (uint idx = tid; idx < TILE_K * TILE_N; idx += 128u) {
            const uint local_k = idx / TILE_N;
            const uint local_n = idx % TILE_N;
            const uint out_row = n_base + local_n;
            half value = half(0.0f);
            if (out_row < n) {
                device const uchar *row_weights = weights
                    + ulong(out_row) * ulong(blocks_per_row) * ulong(BLOCK_BYTES);
                value = half(q8_0_value(row_weights, k_base + local_k));
            }
            b_tile[local_k][local_n] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<half, 8, 8> a_frag;
        simdgroup_matrix<half, 8, 8> b_frag;
        for (uint kk = 0u; kk < TILE_K; kk += 8u) {
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
    if (m_base + local_m < m && n_base + local_n < n) {
        y[ulong(m_base + local_m) * ulong(n) + n_base + local_n]
            = bfloat(accum.thread_elements()[0]);
    }
    if (m_base + local_m < m && n_base + local_n + 1u < n) {
        y[ulong(m_base + local_m) * ulong(n) + n_base + local_n + 1u]
            = bfloat(accum.thread_elements()[1]);
    }
}
