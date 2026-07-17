// SPDX-License-Identifier: AGPL-3.0-only

// Fixed-grid, weight-stationary routed NVFP4 MoE decode workers.
//
// The preceding `moe_build_decode_worklist_c8` produces expert-major groups
// of one to eight live rows.  A CTA owns sixteen output channels of one group:
// every lane loads a weight fragment once and applies it to every row in that
// group.  This is the small-M counterpart to a grouped GEMM: it avoids the
// padded M=64 tensor-core tiles while retaining actual routed-expert reuse.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define ATLAS_DECODE_OUTPUTS_PER_CTA 16u
#define ATLAS_DECODE_THREADS_PER_OUTPUT 16u
#define ATLAS_DECODE_MAX_GROUP_ROWS 8u

__device__ __constant__ float ATLAS_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float atlas_decode_nvfp4(
    const unsigned char* packed,
    const unsigned char* scales,
    float scale2,
    unsigned int n,
    unsigned int k,
    unsigned int k_size
) {
    const unsigned int half_k = k_size >> 1;
    const unsigned int groups = k_size >> 4;
    const unsigned char byte = packed[(unsigned long long)n * half_k + (k >> 1)];
    const unsigned int nibble = (k & 1u) ? (byte >> 4) : (byte & 0xfu);
    __nv_fp8_e4m3 scale_fp8;
    *(unsigned char*)&scale_fp8 = scales[(unsigned long long)n * groups + (k >> 4)];
    return ATLAS_E2M1_LUT[nibble] * (float)scale_fp8 * scale2;
}

__device__ __forceinline__ unsigned int atlas_decode_group_rows(unsigned int descriptor) {
    return descriptor & 0xfu;
}

__device__ __forceinline__ unsigned int atlas_decode_group_start(unsigned int descriptor) {
    return descriptor >> 4;
}

// Gate and up projections share one fixed-grid launch. `n_out` is the width
// of one projection; the logical output width is `2 * n_out`.
extern "C" __global__ void moe_decode_persistent_gate_up_c8(
    const __nv_bfloat16* __restrict__ input,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2,
    __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ up_out,
    const int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ worklist,
    const int* __restrict__ total_groups,
    unsigned int n_out,
    unsigned int k_size
) {
    const int groups = total_groups[0];
    if (groups <= 0) return;
    const unsigned int tiles_per_group = (2u * n_out) / ATLAS_DECODE_OUTPUTS_PER_CTA;
    const unsigned int tiles = static_cast<unsigned int>(groups) * tiles_per_group;
    const unsigned int local_output = threadIdx.x / ATLAS_DECODE_THREADS_PER_OUTPUT;
    const unsigned int lane = threadIdx.x % ATLAS_DECODE_THREADS_PER_OUTPUT;

    for (unsigned int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const unsigned int group = tile / tiles_per_group;
        const unsigned int output_base = (tile % tiles_per_group) * ATLAS_DECODE_OUTPUTS_PER_CTA;
        const unsigned int expert = worklist[group * 2];
        const unsigned int descriptor = worklist[group * 2 + 1];
        const unsigned int rows = atlas_decode_group_rows(descriptor);
        const unsigned int route_start = atlas_decode_group_start(descriptor);
        if (rows == 0 || rows > ATLAS_DECODE_MAX_GROUP_ROWS) return;

        const unsigned int output = output_base + local_output;
        const bool is_up = output >= n_out;
        const unsigned int n = is_up ? output - n_out : output;
        const unsigned char* packed = reinterpret_cast<const unsigned char*>(
            is_up ? up_packed_ptrs[expert] : gate_packed_ptrs[expert]);
        const unsigned char* scales = reinterpret_cast<const unsigned char*>(
            is_up ? up_scale_ptrs[expert] : gate_scale_ptrs[expert]);
        const float scale2 = is_up ? up_scale2[expert] : gate_scale2[expert];
        __nv_bfloat16* output_ptr = is_up ? up_out : gate_out;

        float acc[ATLAS_DECODE_MAX_GROUP_ROWS] = {};
        for (unsigned int k = lane; k < k_size; k += ATLAS_DECODE_THREADS_PER_OUTPUT) {
            const float weight = atlas_decode_nvfp4(packed, scales, scale2, n, k, k_size);
            #pragma unroll
            for (unsigned int row = 0; row < ATLAS_DECODE_MAX_GROUP_ROWS; ++row) {
                if (row < rows) {
                    const unsigned int route = route_start + row;
                    const int token = sorted_token_ids[route];
                    acc[row] += __bfloat162float(input[(unsigned long long)token * k_size + k]) * weight;
                }
            }
        }
        #pragma unroll
        for (unsigned int row = 0; row < ATLAS_DECODE_MAX_GROUP_ROWS; ++row) {
            for (unsigned int offset = 8; offset > 0; offset >>= 1) {
                acc[row] += __shfl_down_sync(0xffffffffu, acc[row], offset, 16);
            }
            if (row < rows && lane == 0) {
                output_ptr[(unsigned long long)(route_start + row) * n_out + n] = __float2bfloat16(acc[row]);
            }
        }
    }
}

// Down projection with a fused, BF16-rounded SiLU activation.  Rounding the
// activation before accumulation preserves the existing materialized-SiLU
// contract while removing its global scratch read/write and launch.
extern "C" __global__ void moe_decode_persistent_down_c8(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ down_packed_ptrs,
    const unsigned long long* __restrict__ down_scale_ptrs,
    const float* __restrict__ down_scale2,
    __nv_bfloat16* __restrict__ down_out,
    const unsigned int* __restrict__ worklist,
    const int* __restrict__ total_groups,
    unsigned int n_out,
    unsigned int k_size
) {
    const int groups = total_groups[0];
    if (groups <= 0) return;
    const unsigned int tiles_per_group = n_out / ATLAS_DECODE_OUTPUTS_PER_CTA;
    const unsigned int tiles = static_cast<unsigned int>(groups) * tiles_per_group;
    const unsigned int local_output = threadIdx.x / ATLAS_DECODE_THREADS_PER_OUTPUT;
    const unsigned int lane = threadIdx.x % ATLAS_DECODE_THREADS_PER_OUTPUT;

    for (unsigned int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const unsigned int group = tile / tiles_per_group;
        const unsigned int n = (tile % tiles_per_group) * ATLAS_DECODE_OUTPUTS_PER_CTA + local_output;
        const unsigned int expert = worklist[group * 2];
        const unsigned int descriptor = worklist[group * 2 + 1];
        const unsigned int rows = atlas_decode_group_rows(descriptor);
        const unsigned int route_start = atlas_decode_group_start(descriptor);
        if (rows == 0 || rows > ATLAS_DECODE_MAX_GROUP_ROWS) return;

        const unsigned char* packed = reinterpret_cast<const unsigned char*>(down_packed_ptrs[expert]);
        const unsigned char* scales = reinterpret_cast<const unsigned char*>(down_scale_ptrs[expert]);
        const float scale2 = down_scale2[expert];
        float acc[ATLAS_DECODE_MAX_GROUP_ROWS] = {};
        for (unsigned int k = lane; k < k_size; k += ATLAS_DECODE_THREADS_PER_OUTPUT) {
            const float weight = atlas_decode_nvfp4(packed, scales, scale2, n, k, k_size);
            #pragma unroll
            for (unsigned int row = 0; row < ATLAS_DECODE_MAX_GROUP_ROWS; ++row) {
                if (row < rows) {
                    const unsigned int route = route_start + row;
                    const float gate = __bfloat162float(gate_out[(unsigned long long)route * k_size + k]);
                    const float up = __bfloat162float(up_out[(unsigned long long)route * k_size + k]);
                    const __nv_bfloat16 activation = __float2bfloat16(
                        (gate / (1.0f + __expf(-gate))) * up);
                    acc[row] += __bfloat162float(activation) * weight;
                }
            }
        }
        #pragma unroll
        for (unsigned int row = 0; row < ATLAS_DECODE_MAX_GROUP_ROWS; ++row) {
            for (unsigned int offset = 8; offset > 0; offset >>= 1) {
                acc[row] += __shfl_down_sync(0xffffffffu, acc[row], offset, 16);
            }
            if (row < rows && lane == 0) {
                down_out[(unsigned long long)(route_start + row) * n_out + n] = __float2bfloat16(acc[row]);
            }
        }
    }
}
