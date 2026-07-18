// SPDX-License-Identifier: AGPL-3.0-only

// Paged Decode Attention — decode attention reading K/V from paged cache via block table.
//
// Compatible with vLLM's paged attention interface.
// Uses NHD cache layout: [num_blocks, block_size, num_kv_heads, head_dim] BF16.
//
// Key design:
//   - One CTA per (q_head, seq) pair
//   - 8 warps split the KV sequence, each thread covers head_dim/32 = 8 BF16 elements
//   - Block table lookup to find physical blocks for each logical position
//   - Within-block positions are contiguous → good memory coalescing
//   - Batched KV loading (BC=4) within blocks, single-load at block boundaries
//   - Online softmax with tree-based inter-warp reduction
//
// Grid: (num_q_heads, num_seqs, 1)   [or with split-K: (num_q_heads, num_splits, num_seqs)]
// Block: (256, 1, 1)

#include <cuda_bf16.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define BC 4            // KV positions batched per loop iteration

__device__ __forceinline__ void unpack2_pd(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// Helper: compute pointer to K or V for a given position in paged cache
__device__ __forceinline__ const __nv_bfloat16* paged_kv_ptr(
    const __nv_bfloat16* __restrict__ cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_table,       // [max_blocks_per_seq]
    unsigned int pos,
    unsigned int block_size,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int kv_head
) {
    unsigned int logical_block = pos / block_size;
    unsigned int block_offset = pos % block_size;
    unsigned int physical_block = (unsigned int)block_table[logical_block];
    unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
    return cache + (unsigned long long)physical_block * page_stride
                 + (unsigned long long)block_offset * num_kv_heads * head_dim
                 + (unsigned long long)kv_head * head_dim;
}

extern "C" __global__ void paged_decode_attn(
    const __nv_bfloat16* __restrict__ Q,          // [num_seqs, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ K_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ O,                // [num_seqs, num_q_heads, head_dim]
    const int* __restrict__ block_tables,         // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,             // [num_seqs]
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,              // query.stride(0) in elements
    const unsigned int sliding_window         // 0 = full attention; >0 = only attend to last `sliding_window` KV positions (Gemma-4 hybrid attn)
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Sliding-window start position. For Gemma-4 sliding layers with
    // window=1024, we mask out KV positions older than seq_len - 1024.
    // When sliding_window == 0 (full attention) or seq_len fits inside
    // the window, window_start = 0 (no masking).
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    // Block table for this sequence
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q into registers (strided: Q may be a non-contiguous QKV split view)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_pd(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    // Each warp handles a chunk of the KV sequence. Split across the
    // ATTENDED range [window_start, seq_len) rather than the raw [0, seq_len)
    // so warps aren't wasted on positions masked out by the sliding window.
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    // Online softmax state
    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    // === Main loop: process positions with batched KV loading ===
    // We can batch BC=4 positions when they're in the same physical block.
    // At block boundaries, fall back to single-position processing.
    unsigned int pos = my_start;

    while (pos < my_end) {
        // Check how many consecutive positions share the same physical block
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        // Get physical block pointer base
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
        unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
        const __nv_bfloat16* k_block_base = K_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;
        const __nv_bfloat16* v_block_base = V_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;

        // Process in batches of BC within this physical block
        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        // Batched path: BC=4 positions at a time (contiguous in memory)
        for (; processed < aligned_count; processed += BC) {
            // Load BC K vectors
            unsigned int k_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    k_packed[b][i] = k32[i];
            }

            // Compute BC dot products
            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float k0, k1;
                    unpack2_pd(k_packed[b][i], k0, k1);
                    dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot * inv_sqrt_d;
            }

            // Prefetch V
            unsigned int v_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    v_packed[b][i] = v32[i];
            }

            // Batched softmax
            float m_new = m;
            #pragma unroll
            for (int b = 0; b < BC; b++)
                m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m - m_new);
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] *= exp_old;
            l *= exp_old;

            float exp_factors[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l += exp_factors[b];
            }
            m = m_new;

            // V accumulate
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float ef = exp_factors[b];
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float v0, v1;
                    unpack2_pd(v_packed[b][i], v0, v1);
                    o_reg[2*i]   += ef * v0;
                    o_reg[2*i+1] += ef * v1;
                }
            }
        }

        // Remainder: single positions
        for (; processed < batch_count; processed++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float k0, k1;
                unpack2_pd(k32[i], k0, k1);
                dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * inv_sqrt_d;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float v0, v1;
                unpack2_pd(v32[i], v0, v1);
                o_reg[2*i]   = o_reg[2*i]   * exp_old + exp_new * v0;
                o_reg[2*i+1] = o_reg[2*i+1] * exp_old + exp_new * v1;
            }
            m = m_new;
        }

        pos += batch_count;
    }

    // === Tree-based inter-warp reduction ===
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset + i] = o_reg[i];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset + i] =
                        smem_o[warp_id][vec_offset + i] * scale_me +
                        smem_o[other][vec_offset + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    // Warp 0 writes final output
    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = smem_o[0][vec_offset + 2*i]     * inv_l;
            float v1 = smem_o[0][vec_offset + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// ============================================================================
// Split-K variant for long sequences with few heads (low SM utilization)
// Grid: (num_q_heads, num_splits, num_seqs)
// ============================================================================

extern "C" __global__ void paged_decode_attn_splitk(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    float* __restrict__ workspace,               // [num_seqs, num_q_heads, num_splits, head_dim+2]
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride               // query.stride(0) in elements
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Compute this split's KV range
    unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (strided: Q may be a non-contiguous QKV split view)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_pd(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    // Each warp handles a chunk of this split's range
    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
    unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;

    // Process positions
    for (unsigned int pos = my_start; pos < my_end; pos++) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned int* k32 = (const unsigned int*)(K_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float k0, k1;
            unpack2_pd(k32[i], k0, k1);
            dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
        }
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, offset);

        float score = dot * inv_sqrt_d;
        float m_new = fmaxf(m, score);
        float exp_old = __expf(m - m_new);
        float exp_new = __expf(score - m_new);
        l = l * exp_old + exp_new;

        const unsigned int* v32 = (const unsigned int*)(V_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset);

        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0, v1;
            unpack2_pd(v32[i], v0, v1);
            o_reg[2*i]   = o_reg[2*i]   * exp_old + exp_new * v0;
            o_reg[2*i+1] = o_reg[2*i+1] * exp_old + exp_new * v1;
        }
        m = m_new;
    }

    // Tree merge within CTA
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset + i] = o_reg[i];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset + i] =
                        smem_o[warp_id][vec_offset + i] * scale_me +
                        smem_o[other][vec_offset + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    // Write partial to workspace
    unsigned int ws_stride = (head_dim + 2);
    float* ws_base = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
                   + split_id * ws_stride;

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            ws_base[vec_offset + i] = smem_o[0][vec_offset + i];
        }
        if (lane_id == 0) {
            ws_base[head_dim] = smem_m[0];
            ws_base[head_dim + 1] = smem_l[0];
        }
    }
}

// Reduction kernel — identical to inferspark_decode_reduce (reuse is fine).
// `seq_lens` guard mirrors the fp8/nvfp4 reduce twins: a zero-length (padded /
// empty) sequence slot must be skipped so stale workspace is never normalized
// into O.
extern "C" __global__ void paged_decode_attn_reduce(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ seq_lens,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % WARP_SIZE;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    if (q_head >= num_q_heads) return;
    if (seq_lens[seq_idx] == 0) return;

    unsigned int ws_stride = (head_dim + 2);
    const float* ws_head = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride;

    float m = ws_head[head_dim];
    float l = ws_head[head_dim + 1];
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        o_reg[i] = ws_head[vec_offset + i];
    }

    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws_s = ws_head + s * ws_stride;
        float ms = ws_s[head_dim];
        float ls = ws_s[head_dim + 1];
        if (ls > 0.0f) {
            float m_new = fmaxf(m, ms);
            float scale_me = __expf(m - m_new);
            float scale_s = __expf(ms - m_new);
            l = l * scale_me + ls * scale_s;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) {
                o_reg[i] = o_reg[i] * scale_me + ws_s[vec_offset + i] * scale_s;
            }
            m = m_new;
        }
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                          + (unsigned long long)q_head * head_dim + vec_offset);
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        float v0 = o_reg[2*i]     * inv_l;
        float v1 = o_reg[2*i + 1] * inv_l;
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}

// ============================================================================
// GQA-group-packed MMA flash-decode — Increment 1 (non-split, writes O direct).
//
// One CTA owns ONE kv-head for ONE sequence and computes attention for all
// `group = num_q_heads / num_kv_heads` q-heads of that kv-head as the M-rows of
// a tensor-core MMA (mma.sync.m16n8k16.bf16). This kills the `group`-fold
// redundant KV read of the scalar per-q-head `paged_decode_attn`.
//
//   Grid:  (num_kv_heads, 1, num_seqs)   Block: (128,1,1) = 4 warps
//   head_dim == 256 only (compile-guarded below); group must be <= 8.
//
// Warp partition: the attended KV range [window_start, seq_len) is split into
// GQA_MMA_WARPS contiguous stripes (one per warp, scalar-style). Each warp
// computes the FULL attention (all 8 group rows, all 256 head_dim) over its
// stripe, holding a partial online-softmax state (m, l, O) in registers; the
// 4 partials are tree-merged per-row across warps in the epilogue.
//
// Numerics track the scalar `paged_decode_attn` golden reference exactly where
// possible: f32 S accumulation, f32 scale-after-dot (inv_sqrt_d), __expf, m
// init = -1e30f, l>0 guards, single bf16 output round, identical window clamps.
// Two accepted, bounded divergences: (a) MMA reorders the head-dim / position
// sums (still f32, with --fmad=false), (b) P is rounded to bf16 before the P·V
// MMA. Bit-exactness is NOT expected; the microtest gates on max-abs error and
// argmax-flip count, not equality.
// ============================================================================
#if HDIM == 256

#define GQA_MMA_WARPS 4
#define GQA_MMA_STEP  16   // KV positions per MMA step per warp (2 n8-tiles) — Inc1 kernel
// CTA-cooperative shared-tile params (splitk kernel, Increment 3.5). ONE 32-position
// K tile + one 32-position V tile is loaded per CTA step by all 128 threads; the 4
// warps partition the 32 positions (8 each) and merge partials in the epilogue.
#define GQA_TILE_KV   32                              // positions per CTA shared tile
#define GQA_MMA_WPOS  (GQA_TILE_KV / GQA_MMA_WARPS)   // 8 positions per warp per tile

// Manual m16n8k16 fragment MMA — same idiom as w8a16_gemm.cu:66 (g=lane>>2,
// t=lane&3). D[16][8] += A[16][16] * B[16][8], f32 accumulate, bf16 operands.
__device__ __forceinline__ void gqa_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3)
    );
}

__device__ __forceinline__ unsigned int gqa_pack_bf16x2_f(float a, float b) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(a));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(b));
    return lo | (hi << 16);
}

__device__ __forceinline__ unsigned int gqa_pack_bf16x2_raw(__nv_bfloat16 a, __nv_bfloat16 b) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(a);
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(b);
    return lo | (hi << 16);
}

extern "C" __global__ void paged_decode_attn_gqa_mma(
    const __nv_bfloat16* __restrict__ Q,          // [num_seqs, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ K_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ O,                // [num_seqs, num_q_heads, head_dim]
    const int* __restrict__ block_tables,         // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,             // [num_seqs]
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,                  // query.stride(0) in elements
    const unsigned int sliding_window             // dispatch only calls with 0 (full attn)
) {
    static_assert(HDIM == 256, "paged_decode_attn_gqa_mma is specialized for head_dim=256");

    const unsigned int kv_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tidx    = threadIdx.x;
    const unsigned int warp_id = tidx / WARP_SIZE;
    const unsigned int lane_id = tidx % WARP_SIZE;
    const unsigned int g = lane_id >> 2;   // 0..7 -> q-row within group (frag row)
    const unsigned int t = lane_id & 3;    // 0..3 -> frag col quad

    if (kv_head >= num_kv_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int group = num_q_heads / num_kv_heads;   // 8 for qwen3.6-35b
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    // sQ: 8 fragment rows (rows >= group zeroed). Padded to cut bank conflicts.
    const int PADQ = 8;
    __shared__ __nv_bfloat16 sQ[8][HDIM + PADQ];
    // Per-row (m,l) and per-row full O for the 4-warp epilogue tree-merge.
    __shared__ float smem_m[GQA_MMA_WARPS][8];
    __shared__ float smem_l[GQA_MMA_WARPS][8];
    __shared__ float smem_o[GQA_MMA_WARPS][8][HDIM];

    // Cooperative Q load: rows [0,group) from global (honoring q_stride), rows
    // [group,8) zeroed so dead fragment rows contribute nothing.
    for (unsigned int idx = tidx; idx < 8u * (HDIM / 2); idx += blockDim.x) {
        unsigned int r = idx / (HDIM / 2);
        unsigned int c = idx % (HDIM / 2);   // u32 column; dim = c*2
        unsigned int val = 0u;
        if (r < group) {
            const unsigned int* qsrc = (const unsigned int*)(Q
                + (unsigned long long)seq_idx * q_stride
                + (unsigned long long)(kv_head * group + r) * head_dim);
            val = qsrc[c];
        }
        *(unsigned int*)&sQ[r][c * 2] = val;
    }
    __syncthreads();

    const int* my_block_table = block_tables + (unsigned long long)seq_idx * max_blocks_per_seq;

    // Warp KV stripe over the ATTENDED range (scalar-style contiguous split).
    const unsigned int attended = seq_len - window_start;
    const unsigned int chunk = (attended + GQA_MMA_WARPS - 1) / GQA_MMA_WARPS;
    unsigned int my_start = window_start + warp_id * chunk;
    unsigned int my_end = my_start + chunk;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    // Online-softmax state for this warp's row g (m_run/l_run redundant across t).
    float m_run = -1e30f;
    float l_run = 0.0f;
    const unsigned int NT = HDIM / 8;   // 32 head_dim n8-tiles
    // Only the 2 real C regs per n8-tile are live (row g). The MMA writes 4 C
    // regs, but the g+8 rows are structurally zero (A rows g+8 are 0), so their
    // outputs are parked in two reused throwaway regs that stay 0 — halving the
    // O accumulator footprint from 128 to 64 registers.
    float O_accum[HDIM / 8][2];
    #pragma unroll
    for (unsigned int nt = 0; nt < NT; nt++) {
        O_accum[nt][0] = 0.0f; O_accum[nt][1] = 0.0f;
    }
    float dead2 = 0.0f, dead3 = 0.0f;   // g+8 rows; remain 0 across all MMAs

    for (unsigned int p0 = my_start; p0 < my_end; p0 += GQA_MMA_STEP) {
        const unsigned int valid = (my_end - p0) < GQA_MMA_STEP ? (my_end - p0) : GQA_MMA_STEP;

        // --- (1) S = Q · K^T for 16 positions -> two n8-tiles (lo: 0..7, hi: 8..15)
        // B(K) position provided by THIS lane's fragment column = n8*8 + g.
        const __nv_bfloat16* kp_lo = (0u * 8 + g) < valid
            ? paged_kv_ptr(K_cache, my_block_table, p0 + (0u * 8 + g), block_size, num_kv_heads, head_dim, kv_head)
            : nullptr;
        const __nv_bfloat16* kp_hi = (1u * 8 + g) < valid
            ? paged_kv_ptr(K_cache, my_block_table, p0 + (1u * 8 + g), block_size, num_kv_heads, head_dim, kv_head)
            : nullptr;

        float S_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float S_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (unsigned int kk = 0; kk < HDIM / 16; kk++) {
            const unsigned int a0 = *(const unsigned int*)&sQ[g][kk * 16 + t * 2];
            const unsigned int a2 = *(const unsigned int*)&sQ[g][kk * 16 + t * 2 + 8];
            // a1/a3 = dead fragment rows (g+8); literal 0 lets nvcc fold them.
            unsigned int b0_lo = 0, b1_lo = 0, b0_hi = 0, b1_hi = 0;
            if (kp_lo) {
                b0_lo = *(const unsigned int*)(kp_lo + kk * 16 + t * 2);
                b1_lo = *(const unsigned int*)(kp_lo + kk * 16 + t * 2 + 8);
            }
            if (kp_hi) {
                b0_hi = *(const unsigned int*)(kp_hi + kk * 16 + t * 2);
                b1_hi = *(const unsigned int*)(kp_hi + kk * 16 + t * 2 + 8);
            }
            gqa_mma_m16n8k16(S_lo[0], S_lo[1], S_lo[2], S_lo[3], a0, 0u, a2, 0u, b0_lo, b1_lo);
            gqa_mma_m16n8k16(S_hi[0], S_hi[1], S_hi[2], S_hi[3], a0, 0u, a2, 0u, b0_hi, b1_hi);
        }

        // f32 scale AFTER the dot (scalar order). Frag col c0/c1 = local pos t*2/t*2+1.
        float s0 = S_lo[0] * inv_sqrt_d;   // local pos t*2
        float s1 = S_lo[1] * inv_sqrt_d;   // local pos t*2+1
        float s2 = S_hi[0] * inv_sqrt_d;   // local pos 8+t*2
        float s3 = S_hi[1] * inv_sqrt_d;   // local pos 8+t*2+1
        if ((t * 2)         >= valid) s0 = -1e30f;
        if ((t * 2 + 1)     >= valid) s1 = -1e30f;
        if ((8 + t * 2)     >= valid) s2 = -1e30f;
        if ((8 + t * 2 + 1) >= valid) s3 = -1e30f;

        // --- (2) online softmax (per row g; quad-reduce across the 4 t-lanes)
        float tmax = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, 1));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, 2));
        float m_new = fmaxf(m_run, tmax);
        float scale = __expf(m_run - m_new);
        #pragma unroll
        for (unsigned int nt = 0; nt < NT; nt++) {
            O_accum[nt][0] *= scale;
            O_accum[nt][1] *= scale;
        }
        l_run *= scale;
        float p0v = __expf(s0 - m_new);
        float p1v = __expf(s1 - m_new);
        float p2v = __expf(s2 - m_new);
        float p3v = __expf(s3 - m_new);
        float psum = p0v + p1v + p2v + p3v;
        psum += __shfl_xor_sync(0xffffffff, psum, 1);
        psum += __shfl_xor_sync(0xffffffff, psum, 2);
        l_run += psum;
        m_run = m_new;

        // --- (3) P · V  (P->bf16, register-chained C->A; N = head_dim in n8 tiles)
        const unsigned int a0m = gqa_pack_bf16x2_f(p0v, p1v);   // K-cols t*2, t*2+1
        const unsigned int a2m = gqa_pack_bf16x2_f(p2v, p3v);   // K-cols t*2+8, t*2+9
        // V base ptrs for the 4 positions this lane contracts over (dim added below).
        const __nv_bfloat16* vlo0 = (t * 2)         < valid
            ? paged_kv_ptr(V_cache, my_block_table, p0 + (t * 2),         block_size, num_kv_heads, head_dim, kv_head) : nullptr;
        const __nv_bfloat16* vlo1 = (t * 2 + 1)     < valid
            ? paged_kv_ptr(V_cache, my_block_table, p0 + (t * 2 + 1),     block_size, num_kv_heads, head_dim, kv_head) : nullptr;
        const __nv_bfloat16* vhi0 = (8 + t * 2)     < valid
            ? paged_kv_ptr(V_cache, my_block_table, p0 + (8 + t * 2),     block_size, num_kv_heads, head_dim, kv_head) : nullptr;
        const __nv_bfloat16* vhi1 = (8 + t * 2 + 1) < valid
            ? paged_kv_ptr(V_cache, my_block_table, p0 + (8 + t * 2 + 1), block_size, num_kv_heads, head_dim, kv_head) : nullptr;
        #pragma unroll
        for (unsigned int nt = 0; nt < NT; nt++) {
            const unsigned int ncol = nt * 8 + g;   // head_dim column
            __nv_bfloat16 zero = __float2bfloat16(0.0f);
            __nv_bfloat16 v_lo0 = vlo0 ? vlo0[ncol] : zero;
            __nv_bfloat16 v_lo1 = vlo1 ? vlo1[ncol] : zero;
            __nv_bfloat16 v_hi0 = vhi0 ? vhi0[ncol] : zero;
            __nv_bfloat16 v_hi1 = vhi1 ? vhi1[ncol] : zero;
            const unsigned int b0 = gqa_pack_bf16x2_raw(v_lo0, v_lo1);   // pos t*2, t*2+1
            const unsigned int b1 = gqa_pack_bf16x2_raw(v_hi0, v_hi1);   // pos t*2+8, t*2+9
            gqa_mma_m16n8k16(O_accum[nt][0], O_accum[nt][1], dead2, dead3,
                             a0m, 0u, a2m, 0u, b0, b1);
        }
    }

    // === Epilogue: stage per-warp (m,l,O) then 2-round per-row tree-merge ===
    smem_m[warp_id][g] = m_run;   // written by all t; identical value across the quad
    smem_l[warp_id][g] = l_run;
    #pragma unroll
    for (unsigned int nt = 0; nt < NT; nt++) {
        smem_o[warp_id][g][nt * 8 + t * 2]     = O_accum[nt][0];
        smem_o[warp_id][g][nt * 8 + t * 2 + 1] = O_accum[nt][1];
    }
    __syncthreads();

    #pragma unroll
    for (unsigned int stride = GQA_MMA_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < stride) {
            const unsigned int other = warp_id + stride;
            float lw = smem_l[other][g];
            if (lw > 0.0f) {
                float mw = smem_m[other][g];
                float my_m = smem_m[warp_id][g];
                float my_l = smem_l[warp_id][g];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                #pragma unroll
                for (unsigned int nt = 0; nt < NT; nt++) {
                    unsigned int c0 = nt * 8 + t * 2;
                    unsigned int c1 = nt * 8 + t * 2 + 1;
                    smem_o[warp_id][g][c0] = smem_o[warp_id][g][c0] * scale_me + smem_o[other][g][c0] * scale_w;
                    smem_o[warp_id][g][c1] = smem_o[warp_id][g][c1] * scale_me + smem_o[other][g][c1] * scale_w;
                }
                if (t == 0) {
                    smem_l[warp_id][g] = my_l * scale_me + lw * scale_w;
                    smem_m[warp_id][g] = m_new;
                }
            }
        }
        __syncthreads();
    }

    // Warp 0 normalizes (O*(1/l)), single bf16 round, writes O[seq, kv_head*group+g, :].
    if (warp_id == 0 && g < group) {
        float final_l = smem_l[0][g];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        const unsigned int q_head = kv_head * group + g;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim);
        #pragma unroll
        for (unsigned int nt = 0; nt < NT; nt++) {
            unsigned int dim = nt * 8 + t * 2;   // even -> adjacent pair packs to one u32
            float v0 = smem_o[0][g][dim]     * inv_l;
            float v1 = smem_o[0][g][dim + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[dim / 2] = lo | (hi << 16);
        }
    }
}

// ============================================================================
// GQA-group-packed MMA flash-decode — CTA-cooperative K+V shared-tile (Increment
// 3.5). One CTA owns ONE kv-head, for ONE KV-split, for ONE seq.
//
//   Grid:  (num_kv_heads, num_splits, num_seqs)   Block: (128,1,1) = 4 warps
//
// Core idea (fixing the warp-stripe / V-only-staged Inc1-3 design that was NOT
// faster than scalar because K was read direct from global with no latency
// hiding): the WHOLE CTA (128 threads) cooperatively loads ONE 32-position K
// tile AND one 32-position V tile into shared memory per step, via a PIPE_DEPTH=2
// `cp.async.cg` double-buffer — BOTH K and V are staged (K is the key change) so
// the NEXT tile's K+V gather overlaps the current tile's Q·K^T + softmax + P·V.
// At ~2K ctx, C=8 attention is gather-latency-bound (KV is L2-resident), so
// hiding the strided page gathers is the whole win.
//
// Warp/position partition: the 4 warps partition the 32-position shared tile,
// GQA_MMA_WPOS=8 positions each (warp w owns tile-local positions [w*8, w*8+8)).
// Each warp computes ONE n8-tile Q·K^T (8 positions x the 8 group q-rows = MMA
// M-rows) reading sK/sV from the SHARED smem tile, keeps a per-warp online-
// softmax partial (m,l,O) in registers, and the 4 partials are combined by the
// epilogue's per-row 2-round tree-merge — a valid flash-attention CTA-level
// online softmax (the merge is the cross-warp softmax combine). Positions are
// partitioned disjointly across warps (no double-count), and each shared tile is
// loaded exactly once by all 128 threads (cooperative, fully coalesced).
//
// Per-warp P·V uses the k16 MMA with only k=0..7 live (a2m==0): the 8 real
// positions map to K-contraction rows 0..7; rows 8..15 do not exist for the warp
// so their A-operand is a literal 0 and their B-operand is set 0 — half the MMA
// is structurally idle but MMA is free here (latency-bound). O_accum stays 64
// f32 regs (32 head_dim n8-tiles x 2 live C regs; the g+8 dead rows park in two
// throwaway regs).
//
// smem (dynamic, 72,064 B < 99 KB — unchanged from the prior layout, so the Rust
// `.shared_mem(72064)` wiring is untouched):
//   sQ    : [8][HDIM+PADQ] bf16                     = 4,224 B
//   smem_m: [4*8] f32 / smem_l: [4*8] f32           =   256 B
//   pool  : max( sK+sV double-buffer 67,584 , smem_o epilogue 32,768 ) = 67,584 B
//     sK  : [2 stage][32 pos][HDIM+PAD] bf16   = 33,792 B
//     sV  : [2 stage][32 pos][HDIM+PAD] bf16   = 33,792 B
//     smem_o (epilogue, reuses pool): [4*8][HDIM] f32 = 32,768 B
// The +8 bf16 (16 B) row pad on sK removes the K^T-column read bank conflict: the
// row stride is 264 bf16 = 132 u32 ≡ 4 (mod 32), so the 32 lanes of a warp read
// K at u32-bank (4*g + t) (mod 32) — a bijection over 0..31 = zero-conflict (no
// XOR swizzle needed). The pad also keeps every cp.async dest 16 B aligned
// ((256+8)*2 = 528, 528%16==0).
//
//   * num_splits==1  -> normalize + write O directly (no workspace / reduce).
//   * num_splits>1   -> write UN-normalized (O_partial[hd], m, l) f32 to
//     `workspace[seq, kv_head*group+g, split, hd+2]` — the EXACT slot layout the
//     reduce reads (`((seq*nq + q_head)*num_splits + split)*(hd+2)`).
//
// Numerics track the scalar golden within tolerance: f32 S accum, f32 scale-
// after-dot, __expf, m init -1e30f, l>0 guards, single bf16 output round. The
// interleaved (vs scalar contiguous) position partition only reorders the f32
// online-softmax merge (order-robust); P is rounded to bf16 for the P·V MMA
// exactly as before. Masked (>= valid) positions carry p==0 so their finite,
// clamped staged K/V never contribute (0*finite==0). Split partials combine
// through the reduce's log-sum-exp, tolerance-equal to the non-split result.
// (Still inside the `#if HDIM == 256` block opened above for the Inc1 kernel.)
// ============================================================================

// cp.async helpers (SM80+). File-local; the `paged_decode` module has no others.
__device__ __forceinline__ void gqa_cp_async_16(void* dst_smem, const void* src_gmem) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src_gmem));
}
__device__ __forceinline__ void gqa_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void gqa_cp_async_wait1() {
    asm volatile("cp.async.wait_group 1;");
}
__device__ __forceinline__ void gqa_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

// Cooperatively stage ONE GQA_TILE_KV=32-position K tile AND V tile (the CTA's KV
// tile for the current step) into shared memory, via cp.async.cg. All 128 threads
// participate: 32 positions x (HDIM/8 = 32) 16 B chunks = 1024 chunks per tensor,
// i.e. 8 chunks/thread for K + 8 for V = 16 cp.async ops/thread/stage. Masked
// positions (local index >= valid) are clamped to a real (finite) source row so
// every load is in-bounds; their softmax weight is later forced to 0. `sK`/`sV`
// are the stage bases (each [GQA_TILE_KV][row_stride]); one commit_group covers
// both K and V of this tile.
__device__ __forceinline__ void gqa_stage_kv_tile(
    __nv_bfloat16* sK, __nv_bfloat16* sV,          // stage bases, each [GQA_TILE_KV][row_stride]
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    const int* __restrict__ my_block_table,
    unsigned int p0, unsigned int valid,
    unsigned int block_size, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int kv_head,
    unsigned int tid, unsigned int row_stride       // row_stride = HDIM + PAD
) {
    const unsigned int CPR = HDIM / 8u;             // 16 B chunks per 256-elem row = 32
    #pragma unroll
    for (unsigned int i = 0; i < (GQA_TILE_KV * (HDIM / 8u)) / 128u; i++) {
        const unsigned int chunk = tid + 128u * i;  // 0..1023
        const unsigned int lp = chunk / CPR;        // local pos 0..31
        const unsigned int dim = (chunk % CPR) * 8u;// bf16 dim, 16 B step
        const unsigned int src_pos = p0 + (lp < valid ? lp : (valid - 1u));
        const __nv_bfloat16* kptr = paged_kv_ptr(
            K_cache, my_block_table, src_pos, block_size, num_kv_heads, head_dim, kv_head);
        const __nv_bfloat16* vptr = paged_kv_ptr(
            V_cache, my_block_table, src_pos, block_size, num_kv_heads, head_dim, kv_head);
        gqa_cp_async_16(&sK[lp * row_stride + dim], kptr + dim);
        gqa_cp_async_16(&sV[lp * row_stride + dim], vptr + dim);
    }
}

extern "C" __global__ void paged_decode_attn_gqa_mma_splitk(
    const __nv_bfloat16* __restrict__ Q,          // [num_seqs, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ K_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ O,                // [num_seqs, num_q_heads, head_dim] (used iff num_splits==1)
    float* __restrict__ workspace,                // [num_seqs, num_q_heads, num_splits, head_dim+2] (used iff >1)
    const int* __restrict__ block_tables,         // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,             // [num_seqs]
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,                  // query.stride(0) in elements
    const unsigned int sliding_window             // dispatch only calls with 0 (full attn)
) {
    static_assert(HDIM == 256, "paged_decode_attn_gqa_mma_splitk is specialized for head_dim=256");

    const unsigned int kv_head  = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx  = blockIdx.z;
    const unsigned int tidx     = threadIdx.x;
    const unsigned int warp_id  = tidx / WARP_SIZE;
    const unsigned int lane_id  = tidx % WARP_SIZE;
    const unsigned int g = lane_id >> 2;   // 0..7 -> q-row within group (frag row)
    const unsigned int t = lane_id & 3;    // 0..3 -> frag col quad

    if (kv_head >= num_kv_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int group = num_q_heads / num_kv_heads;   // 8 for qwen3.6-35b
    (void)sliding_window;   // dispatch guarantees full attention (window_start == 0)

    // Split range over [0, seq_len) — scalar-splitk-identical clamps.
    const unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;   // empty split

    // ── Dynamic smem layout (unioned; total 72,064 B < 99 KB SM cap) ──
    //   sQ    : [8][HDIM+PADQ] bf16                    = 4,224 B
    //   smem_m: [4*8] f32 / smem_l: [4*8] f32          =   256 B
    //   pool  : max( sK+sV double-buffer 67,584 , smem_o merge 32,768 ) = 67,584 B
    //     sK  : [2 stage][32 pos][HDIM+PAD] bf16 (loop)  = 33,792 B
    //     sV  : [2 stage][32 pos][HDIM+PAD] bf16 (loop)  = 33,792 B
    //     smem_o: [4 warp * 8 row][HDIM] f32              (epilogue, reuses pool)
    const int PADQ = 8;
    const int PAD  = 8;   // sK/sV row pad: makes the K^T-column read bank-conflict-free
    const unsigned int SQ_BYTES = 8u * (HDIM + PADQ) * 2u;   // 4224
    extern __shared__ unsigned char gqa_smem[];
    __nv_bfloat16* sQ = (__nv_bfloat16*)gqa_smem;               // [8][HDIM+PADQ]
    float* smem_m = (float*)(gqa_smem + SQ_BYTES);             // [4*8]
    float* smem_l = (float*)(gqa_smem + SQ_BYTES + 128u);      // [4*8]
    unsigned char* pool = gqa_smem + SQ_BYTES + 256u;
    const unsigned int VROW = HDIM + PAD;                      // bf16 row stride (264)
    const unsigned int STAGE_ROWS = GQA_TILE_KV * VROW;        // bf16 per K (or V) stage
    __nv_bfloat16* sKb = (__nv_bfloat16*)pool;                 // [2][32][VROW]
    __nv_bfloat16* sVb = sKb + 2u * STAGE_ROWS;                // [2][32][VROW]

    // Cooperative Q load: rows [0,group) from global (honoring q_stride), rows
    // [group,8) zeroed so dead fragment rows contribute nothing.
    for (unsigned int idx = tidx; idx < 8u * (HDIM / 2); idx += blockDim.x) {
        unsigned int r = idx / (HDIM / 2);
        unsigned int c = idx % (HDIM / 2);   // u32 column; dim = c*2
        unsigned int val = 0u;
        if (r < group) {
            const unsigned int* qsrc = (const unsigned int*)(Q
                + (unsigned long long)seq_idx * q_stride
                + (unsigned long long)(kv_head * group + r) * head_dim);
            val = qsrc[c];
        }
        *(unsigned int*)&sQ[r * (HDIM + PADQ) + c * 2] = val;
    }
    __syncthreads();

    const int* my_block_table = block_tables + (unsigned long long)seq_idx * max_blocks_per_seq;

    // This warp's tile-local base position within the shared 32-position tile.
    // Warp w owns tile positions [wbase, wbase+GQA_MMA_WPOS) = 8 positions each;
    // the 4 warps partition each shared tile disjointly.
    const unsigned int wbase = warp_id * GQA_MMA_WPOS;   // 0 / 8 / 16 / 24

    // Online-softmax state for this warp's row g (m_run/l_run redundant across t).
    float m_run = -1e30f;
    float l_run = 0.0f;
    const unsigned int NT = HDIM / 8;   // 32 head_dim n8-tiles
    float O_accum[HDIM / 8][2];
    #pragma unroll
    for (unsigned int nt = 0; nt < NT; nt++) { O_accum[nt][0] = 0.0f; O_accum[nt][1] = 0.0f; }
    float dead2 = 0.0f, dead3 = 0.0f;   // g+8 rows; remain 0 across all MMAs

    if (kv_start < kv_end) {
        // Prologue: cooperatively stage tile 0's K+V into stage 0 (all 128 threads).
        {
            const unsigned int valid0 = (kv_end - kv_start) < GQA_TILE_KV
                ? (kv_end - kv_start) : GQA_TILE_KV;
            gqa_stage_kv_tile(sKb, sVb, K_cache, V_cache, my_block_table,
                              kv_start, valid0, block_size, num_kv_heads,
                              head_dim, kv_head, tidx, VROW);
            gqa_cp_async_commit();
        }
        unsigned int cur_stage = 0;

        // CTA-cooperative loop: all 4 warps advance in lockstep over shared 32-tiles.
        for (unsigned int p0 = kv_start; p0 < kv_end; p0 += GQA_TILE_KV) {
            const unsigned int valid = (kv_end - p0) < GQA_TILE_KV ? (kv_end - p0) : GQA_TILE_KV;

            // Prefetch the NEXT tile's K+V (double-buffer). If none remains, reload
            // the current tile (finite filler) so exactly 2 groups stay in flight and
            // wait_group 1 always drains the CURRENT (older) tile.
            const unsigned int np0 = p0 + GQA_TILE_KV;
            const unsigned int nstage = cur_stage ^ 1u;
            const unsigned int fp0 = (np0 < kv_end) ? np0 : p0;
            const unsigned int fvalid = (np0 < kv_end)
                ? ((kv_end - np0) < GQA_TILE_KV ? (kv_end - np0) : GQA_TILE_KV)
                : valid;
            // Prefetch the next block-table entry one tile ahead (L2 hint).
            if (np0 < kv_end) {
                const unsigned int nlb = np0 / block_size;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(my_block_table + nlb));
            }
            gqa_stage_kv_tile(sKb + nstage * STAGE_ROWS, sVb + nstage * STAGE_ROWS,
                              K_cache, V_cache, my_block_table, fp0, fvalid,
                              block_size, num_kv_heads, head_dim, kv_head, tidx, VROW);
            gqa_cp_async_commit();
            gqa_cp_async_wait1();   // current tile's K+V ready; next stays in flight
            __syncthreads();        // shared tile: all warps must see the completed load

            // Only warps holding >=1 real position this tile compute. Guards the
            // first-tile-fully-masked case (m_run still -1e30f, where __expf(s-m_new)
            // would be a spurious 1 for masked cols). Warp-uniform => no barrier skew.
            if (wbase < valid) {
                const __nv_bfloat16* sKc = sKb + cur_stage * STAGE_ROWS;   // [32][VROW]
                const __nv_bfloat16* sVc = sVb + cur_stage * STAGE_ROWS;   // [32][VROW]

                // --- (1) S = Q · K^T over THIS warp's 8 positions -> ONE n8-tile.
                // B(K) column g = tile position (wbase+g); K row read from shared smem
                // (row stride 264 bf16 ≡ 4 mod 32 u32 => bank(4g+t) bijection, 0 conflict).
                const __nv_bfloat16* krow = sKc + (unsigned long long)(wbase + g) * VROW;
                float S[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (unsigned int kk = 0; kk < HDIM / 16; kk++) {
                    const unsigned int a0 = *(const unsigned int*)&sQ[g * (HDIM + PADQ) + kk * 16 + t * 2];
                    const unsigned int a2 = *(const unsigned int*)&sQ[g * (HDIM + PADQ) + kk * 16 + t * 2 + 8];
                    const unsigned int b0 = *(const unsigned int*)(krow + kk * 16 + t * 2);
                    const unsigned int b1 = *(const unsigned int*)(krow + kk * 16 + t * 2 + 8);
                    gqa_mma_m16n8k16(S[0], S[1], S[2], S[3], a0, 0u, a2, 0u, b0, b1);
                }

                // Frag col c0/c1 = tile position wbase+t*2 / wbase+t*2+1.
                float s0 = S[0] * inv_sqrt_d;   // tile pos wbase + t*2
                float s1 = S[1] * inv_sqrt_d;   // tile pos wbase + t*2+1
                if ((wbase + t * 2)     >= valid) s0 = -1e30f;
                if ((wbase + t * 2 + 1) >= valid) s1 = -1e30f;

                // --- (2) online softmax (per row g; quad-reduce across the 4 t-lanes)
                float tmax = fmaxf(s0, s1);
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, 1));
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, 2));
                float m_new = fmaxf(m_run, tmax);
                float scale = __expf(m_run - m_new);
                #pragma unroll
                for (unsigned int nt = 0; nt < NT; nt++) { O_accum[nt][0] *= scale; O_accum[nt][1] *= scale; }
                l_run *= scale;
                float p0v = __expf(s0 - m_new);
                float p1v = __expf(s1 - m_new);
                float psum = p0v + p1v;
                psum += __shfl_xor_sync(0xffffffff, psum, 1);
                psum += __shfl_xor_sync(0xffffffff, psum, 2);
                l_run += psum;
                m_run = m_new;

                // --- (3) P · V  (P->bf16; V from shared smem tile). Only k=0..7 live
                // (this warp's 8 positions): a2m==0 zeroes the non-existent k=8..15.
                const unsigned int a0m = gqa_pack_bf16x2_f(p0v, p1v);   // positions wbase+t*2, +1
                #pragma unroll
                for (unsigned int nt = 0; nt < NT; nt++) {
                    const unsigned int ncol = nt * 8 + g;   // head_dim column
                    __nv_bfloat16 v0 = sVc[(unsigned long long)(wbase + t * 2)     * VROW + ncol];
                    __nv_bfloat16 v1 = sVc[(unsigned long long)(wbase + t * 2 + 1) * VROW + ncol];
                    const unsigned int b0 = gqa_pack_bf16x2_raw(v0, v1);
                    gqa_mma_m16n8k16(O_accum[nt][0], O_accum[nt][1], dead2, dead3,
                                     a0m, 0u, 0u, 0u, b0, 0u);
                }
            }
            __syncthreads();        // all warps done reading stage cur before it is reused
            cur_stage = nstage;
        }
        gqa_cp_async_wait_all();   // drain the trailing prefetch before pool reuse
    }

    // === Epilogue: reuse `pool` as smem_o[4][8][HDIM] f32, 2-round per-row merge ===
    __syncthreads();
    float* smem_o = (float*)pool;   // [warp*8 + g][HDIM]
    smem_m[warp_id * 8 + g] = m_run;
    smem_l[warp_id * 8 + g] = l_run;
    #pragma unroll
    for (unsigned int nt = 0; nt < NT; nt++) {
        smem_o[(warp_id * 8 + g) * HDIM + nt * 8 + t * 2]     = O_accum[nt][0];
        smem_o[(warp_id * 8 + g) * HDIM + nt * 8 + t * 2 + 1] = O_accum[nt][1];
    }
    __syncthreads();

    #pragma unroll
    for (unsigned int stride = GQA_MMA_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < stride) {
            const unsigned int other = warp_id + stride;
            float lw = smem_l[other * 8 + g];
            if (lw > 0.0f) {
                float mw = smem_m[other * 8 + g];
                float my_m = smem_m[warp_id * 8 + g];
                float my_l = smem_l[warp_id * 8 + g];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                #pragma unroll
                for (unsigned int nt = 0; nt < NT; nt++) {
                    unsigned int c0 = (warp_id * 8 + g) * HDIM + nt * 8 + t * 2;
                    unsigned int c1 = c0 + 1;
                    unsigned int o0 = (other * 8 + g) * HDIM + nt * 8 + t * 2;
                    smem_o[c0] = smem_o[c0] * scale_me + smem_o[o0]     * scale_w;
                    smem_o[c1] = smem_o[c1] * scale_me + smem_o[o0 + 1] * scale_w;
                }
                if (t == 0) {
                    smem_l[warp_id * 8 + g] = my_l * scale_me + lw * scale_w;
                    smem_m[warp_id * 8 + g] = m_new;
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 0 && g < group) {
        const unsigned int q_head = kv_head * group + g;
        if (num_splits == 1) {
            // Normalize (O*(1/l)), single bf16 round, write O directly.
            float final_l = smem_l[g];
            float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
            unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                                  + (unsigned long long)q_head * head_dim);
            #pragma unroll
            for (unsigned int nt = 0; nt < NT; nt++) {
                unsigned int dim = nt * 8 + t * 2;
                float v0 = smem_o[g * HDIM + dim]     * inv_l;
                float v1 = smem_o[g * HDIM + dim + 1] * inv_l;
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
                o32[dim / 2] = lo | (hi << 16);
            }
        } else {
            // Write UN-normalized partial (O[hd], m, l) f32 at the reduce's slot:
            //   ws_base = workspace + ((seq*nq + q_head)*num_splits + split)*(hd+2)
            const unsigned int ws_stride = head_dim + 2;
            float* ws_base = workspace
                + ((unsigned long long)(seq_idx * num_q_heads + q_head) * num_splits + split_id) * ws_stride;
            #pragma unroll
            for (unsigned int nt = 0; nt < NT; nt++) {
                unsigned int dim = nt * 8 + t * 2;
                ws_base[dim]     = smem_o[g * HDIM + dim];
                ws_base[dim + 1] = smem_o[g * HDIM + dim + 1];
            }
            if (t == 0) {
                ws_base[head_dim]     = smem_m[g];
                ws_base[head_dim + 1] = smem_l[g];
            }
        }
    }
}

#endif  // HDIM == 256
