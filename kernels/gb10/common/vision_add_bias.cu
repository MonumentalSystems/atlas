// SPDX-License-Identifier: AGPL-3.0-only
//
// Fused bias-add for the tensor-core ViT GEMM path: C[m,n] += bias[n] (BF16,
// in-place). Pairs with the common `dense_gemm_bf16_pipelined` so that
// VisionEncoder::vit_gemm_bias takes the ~40× tensor-core path instead of the
// scalar `vision_gemm_bias` fallback.
//
// Lives in `common/` (module name `vision_add_bias`, NOT overridden by any
// per-target `vision_encoder.cu`) so EVERY Qwen3.5/3.6 VL target — qwen3.6-35b-a3b
// (Holo), qwen3.5-122b-a10b, qwen3.5-397b-a17b, qwen3-vl-30b-a3b — gets it, not
// just the one target that happened to define it. Looked up via
// try_kernel(gpu, "vision_add_bias", "vision_add_bias").

#include <cuda_bf16.h>

extern "C" __global__ void vision_add_bias(
    __nv_bfloat16* __restrict__ C,          // [M, N] in-place
    const __nv_bfloat16* __restrict__ bias, // [N]
    unsigned int M, unsigned int N
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned long long)M * N) return;
    unsigned int col = (unsigned int)(idx % N);
    C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + __bfloat162float(bias[col]));
}
