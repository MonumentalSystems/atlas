// SPDX-License-Identifier: AGPL-3.0-only

// Raw-pointer AOT bridge for vLLM's Marlin MoE kernel.
//
// The included Marlin implementation is Apache-2.0 code from vLLM. This
// bridge is an Atlas adaptation: it removes the LibTorch ABI, fixes the only
// supported geometry to Qwen3.6-35B-A3B (BF16 x NVFP4), and exposes a stable
// C ABI suitable for dlopen from Rust.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>

#define MARLIN_NAMESPACE_NAME atlas_marlin_moe
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
#include "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"

namespace {

constexpr int kThreadN = 128;
constexpr int kThreadK = 128;
constexpr int kStages = 4;
int g_sms = 0;
int g_blocks = 0;
int g_shared = 0;
int g_init_status = 1;

enum class MarlinVariant { k128x128, k64x128, k128x64 };
MarlinVariant g_variant = MarlinVariant::k128x128;

// Exact cache-size calculation from vLLM's Marlin dispatcher for
// m_block_size=8, BF16 activations, NVFP4 weights, group_size=16.
constexpr int marlin_cache_bytes() {
  constexpr int tb_m = 16;
  constexpr int meta = tb_m * 16;
  constexpr int a = kStages * tb_m * kThreadK * 2;
  constexpr int b = kStages * (kThreadK * kThreadN / 8) * 4;
  constexpr int red = tb_m * (kThreadN + 8) * 2;
  constexpr int bias = kThreadN * 2;
  constexpr int tmp = b > red + bias ? b : red + bias;
  constexpr int scales = (kThreadK / 16) * kThreadN * 2 * kStages;
  return meta + a + tmp + scales;
}

template <int ThreadK, int ThreadN>
constexpr int marlin_cache_bytes_for() {
  constexpr int tb_m = 16;
  constexpr int meta = tb_m * 16;
  constexpr int a = kStages * tb_m * ThreadK * 2;
  constexpr int b = kStages * (ThreadK * ThreadN / 8) * 4;
  constexpr int red = tb_m * (ThreadN + 8) * 2;
  constexpr int bias = ThreadN * 2;
  constexpr int tmp = b > red + bias ? b : red + bias;
  constexpr int scales = (ThreadK / 16) * ThreadN * 2 * kStages;
  return meta + a + tmp + scales;
}

template <typename Kernel>
int vllm_blocks_per_sm(Kernel kernel, int threads, int cache_bytes,
                       int max_shared) {
  cudaFuncAttributes attr{};
  if (cudaFuncGetAttributes(&attr, kernel) != cudaSuccess) return 0;
  constexpr int kMaxRegistersPerSm = 255 * 1024;
  const int register_bytes = (attr.numRegs > 0 ? attr.numRegs : 1) * threads * 4;
  const int register_limit = kMaxRegistersPerSm / register_bytes;
  const int shared_limit = max_shared / (cache_bytes + 1536);
  const int allowed = register_limit < shared_limit ? register_limit : shared_limit;
  return allowed < 1 ? 1 : (allowed > 4 ? 4 : allowed);
}

template <typename Kernel>
bool set_dynamic_shared(Kernel kernel, int shared) {
  return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              shared) == cudaSuccess;
}

bool vllm_auto_config_enabled() {
  const char* value = std::getenv("ATLAS_MARLIN_VLLM_AUTO_CONFIG");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

// Build vLLM's block-aligned route metadata for M<=8, top_k=8, E=256.
// One CTA is intentional: this is a tiny, graph-captured control kernel and
// the serial prefix sum avoids an additional scan launch.
__global__ void align_routes_c8(const int32_t* topk_ids,
                                int32_t* sorted_token_ids,
                                int32_t* expert_ids,
                                int32_t* num_tokens_post_padded,
                                int num_tokens) {
  __shared__ int counts[256];
  __shared__ int offsets[257];
  __shared__ int cursors[256];
  const int tid = threadIdx.x;
  counts[tid] = 0;
  cursors[tid] = 0;
  __syncthreads();

  if (tid == 0) {
    const int routes = num_tokens * 8;
    for (int route = 0; route < routes; ++route) {
      const int expert = topk_ids[route];
      if (expert >= 0 && expert < 256) ++counts[expert];
    }

    offsets[0] = 0;
    for (int expert = 0; expert < 256; ++expert) {
      const int padded = (counts[expert] + 7) & ~7;
      offsets[expert + 1] = offsets[expert] + padded;
      for (int i = 0; i < padded; ++i) {
        sorted_token_ids[offsets[expert] + i] = routes;
      }
      for (int block = 0; block < padded / 8; ++block) {
        expert_ids[offsets[expert] / 8 + block] = expert;
      }
    }
    for (int route = 0; route < routes; ++route) {
      const int expert = topk_ids[route];
      if (expert >= 0 && expert < 256) {
        sorted_token_ids[offsets[expert] + cursors[expert]++] = route;
      }
    }
    *num_tokens_post_padded = offsets[256];
  }
}

// Repack checkpoint row-major NVFP4 [N,K/2] bytes into Marlin's 16x64 tile
// order. This is the no-permutation, 4-bit, W4A16 specialization of vLLM's
// gptq_marlin_repack_kernel, expressed directly so the AOT library has no
// Torch dependency. Repacking runs once at model load, not in decode.
__global__ void repack_nvfp4(const uint8_t* src, uint32_t* dst, int size_k,
                             int size_n) {
  constexpr int tile_k = 16;
  constexpr int tile_n = 64;
  constexpr int words_per_tile = tile_k * tile_n / 8;
  const int k_tiles = size_k / tile_k;
  const int n_tiles = size_n / tile_n;
  const int total_tiles = k_tiles * n_tiles;

  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int k_tile = tile / n_tiles;
    const int n_tile = tile % n_tiles;
    const int lane_linear = threadIdx.x;
    if (lane_linear < 128) {
      const int warp = lane_linear / 32;
      const int lane = lane_linear % 32;
      const int tc_col = lane / 4;
      const int tc_row = (lane % 4) * 2;
      const int cur_n = n_tile * tile_n + warp * 16 + tc_col;
      const int k_base = k_tile * tile_k;
      constexpr int offsets[4] = {0, 1, 8, 9};
      uint32_t values[8];

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k = k_base + tc_row + offsets[i];
        const uint8_t packed1 = src[static_cast<int64_t>(cur_n) * (size_k / 2) + k / 2];
        const uint8_t packed2 = src[static_cast<int64_t>(cur_n + 8) * (size_k / 2) + k / 2];
        values[i] = (packed1 >> ((k & 1) * 4)) & 0xf;
        values[4 + i] = (packed2 >> ((k & 1) * 4)) & 0xf;
      }

      constexpr int order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
      uint32_t packed = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) packed |= values[order[i]] << (4 * i);
      dst[tile * words_per_tile + lane * 4 + warp] = packed;
    }
  }
}

// Marlin's first MoE GEMM writes route-major rows [gate(512), up(512)].
// Atlas's generic moe_silu_mul consumes two planar buffers, so using it here
// would pair different routes once M*top_k > 1. Keep this tiny transform in
// the bridge and preserve the exact clamp used by Atlas's routed MoE path.
__global__ void silu_mul_interleaved(const __nv_bfloat16* gate_up,
                                     __nv_bfloat16* output, int routes,
                                     int intermediate) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = routes * intermediate;
  if (idx >= total) return;

  const int route = idx / intermediate;
  const int column = idx - route * intermediate;
  const int row = route * (2 * intermediate);
  float gate = __bfloat162float(gate_up[row + column]);
  float up = __bfloat162float(gate_up[row + intermediate + column]);
  constexpr float kSwiGluLimit = 10.0f;
  gate = fminf(gate, kSwiGluLimit);
  up = fminf(fmaxf(up, -kSwiGluLimit), kSwiGluLimit);
  output[idx] = __float2bfloat16(gate / (1.0f + __expf(-gate)) * up);
}

}  // namespace

extern "C" int atlas_marlin_moe_init() {
  int device = 0;
  int max_shared = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, device) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&max_shared,
                             cudaDevAttrMaxSharedMemoryPerBlockOptin, device) !=
          cudaSuccess) {
    return 1;
  }
  const auto kernel_128x128 = atlas_marlin_moe::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false>;
  const int cache_128x128 = marlin_cache_bytes_for<128, 128>();
  if (cache_128x128 > max_shared) return 2;

  g_variant = MarlinVariant::k128x128;
  int blocks_per_sm = 1;
  g_shared = cache_128x128;
  if (vllm_auto_config_enabled()) {
    const auto kernel_64x128 = atlas_marlin_moe::Marlin<
        vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
        vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 4, 1, false>;
    const auto kernel_128x64 = atlas_marlin_moe::Marlin<
        vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
        vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 4, 1, false>;

    const int blocks_128x128 = vllm_blocks_per_sm(
        kernel_128x128, 256, cache_128x128, max_shared);
    const int blocks_64x128 = vllm_blocks_per_sm(
        kernel_64x128, 128, marlin_cache_bytes_for<64, 128>(), max_shared);
    const int blocks_128x64 = vllm_blocks_per_sm(
        kernel_128x64, 128, marlin_cache_bytes_for<128, 64>(), max_shared);
    if (blocks_128x128 > blocks_per_sm) {
      blocks_per_sm = blocks_128x128;
    }
    if (blocks_64x128 > blocks_per_sm) {
      g_variant = MarlinVariant::k64x128;
      blocks_per_sm = blocks_64x128;
    }
    if (blocks_128x64 > blocks_per_sm) {
      g_variant = MarlinVariant::k128x64;
      blocks_per_sm = blocks_128x64;
    }
    // Match vLLM's dispatcher: it deliberately reserves this amount of
    // dynamic shared memory to enforce the chosen persistent CTA count.
    g_shared = blocks_per_sm > 1 ? max_shared / blocks_per_sm - 1024 : max_shared;
  }

  bool configured = false;
  switch (g_variant) {
    case MarlinVariant::k128x128:
      configured = set_dynamic_shared(kernel_128x128, g_shared);
      break;
    case MarlinVariant::k64x128:
      configured = set_dynamic_shared(
          atlas_marlin_moe::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                                   vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                                   128, 1, 8, 4, true, 4, 1, false>,
          g_shared);
      break;
    case MarlinVariant::k128x64:
      configured = set_dynamic_shared(
          atlas_marlin_moe::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                                   vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                                   128, 1, 4, 8, true, 4, 1, false>,
          g_shared);
      break;
  }
  if (!configured) {
    return 3;
  }
  g_blocks = g_sms * blocks_per_sm;
  g_init_status = 0;
  return 0;
}

extern "C" int atlas_marlin_moe_repack(const void* src_rowmajor,
                                        void* dst_marlin, int size_k,
                                        int size_n, void* stream_ptr) {
  if (!src_rowmajor || !dst_marlin || size_k <= 0 || size_n <= 0 ||
      size_k % 16 || size_n % 64) {
    return 1;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  repack_nvfp4<<<128, 128, 0, stream>>>(
      static_cast<const uint8_t*>(src_rowmajor),
      static_cast<uint32_t*>(dst_marlin), size_k, size_n);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int atlas_marlin_moe_align(const void* topk_ids,
                                       void* sorted_token_ids,
                                       void* expert_ids,
                                       void* num_tokens_post_padded,
                                       int num_tokens, void* stream_ptr) {
  if (!topk_ids || !sorted_token_ids || !expert_ids ||
      !num_tokens_post_padded || num_tokens < 1 || num_tokens > 8) {
    return 1;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  align_routes_c8<<<1, 256, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids),
      static_cast<int32_t*>(sorted_token_ids),
      static_cast<int32_t*>(expert_ids),
      static_cast<int32_t*>(num_tokens_post_padded), num_tokens);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int atlas_marlin_moe_silu_mul(const void* gate_up_bf16,
                                          void* output_bf16, int routes,
                                          int intermediate,
                                          void* stream_ptr) {
  if (!gate_up_bf16 || !output_bf16 || routes <= 0 || intermediate <= 0) {
    return 1;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const int total = routes * intermediate;
  silu_mul_interleaved<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up_bf16),
      static_cast<__nv_bfloat16*>(output_bf16), routes, intermediate);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int atlas_marlin_moe_gemm(
    const void* a_bf16, const void* weights, void* output_bf16,
    void* reduce_tmp_f32, const void* scales_e4m3,
    const void* global_scales_f32, const void* sorted_token_ids,
    const void* expert_ids, const void* num_tokens_post_padded,
    const void* topk_weights_f32, int top_k, int mul_topk_weights,
    int size_m, int size_n, int size_k, void* workspace, void* stream_ptr) {
  if (!a_bf16 || !weights || !output_bf16 || !reduce_tmp_f32 ||
      !scales_e4m3 || !global_scales_f32 || !sorted_token_ids ||
      !expert_ids || !num_tokens_post_padded || !topk_weights_f32 ||
      !workspace || size_m <= 0 || size_n <= 0 || size_k <= 0 ||
      size_n % 128 || size_k % 128 || (top_k != 1 && top_k != 8)) {
    return 1;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (g_init_status != 0 || g_blocks <= 0) return 2;
  const auto launch = [&](auto kernel, int threads) {
    kernel<<<g_blocks, threads, g_shared, stream>>>(
        static_cast<const int4*>(a_bf16), static_cast<const int4*>(weights),
        static_cast<int4*>(output_bf16), static_cast<int4*>(reduce_tmp_f32),
        nullptr, nullptr, static_cast<const int4*>(scales_e4m3),
        static_cast<const float*>(global_scales_f32), nullptr, nullptr,
        static_cast<const int32_t*>(sorted_token_ids),
        static_cast<const int32_t*>(expert_ids),
        static_cast<const int32_t*>(num_tokens_post_padded),
        static_cast<const float*>(topk_weights_f32), top_k,
        mul_topk_weights != 0, size_k / 16, size_m, size_n, size_k,
        static_cast<int*>(workspace), false, false, true);
  };
  switch (g_variant) {
    case MarlinVariant::k128x128:
      launch(atlas_marlin_moe::Marlin<vllm::kBFloat16.id(),
                                      vllm::kFE2M1f.id(),
                                      vllm::kBFloat16.id(),
                                      vllm::kFE4M3fn.id(), 256, 1, 8, 8, true,
                                      4, 1, false>,
             256);
      break;
    case MarlinVariant::k64x128:
      launch(atlas_marlin_moe::Marlin<vllm::kBFloat16.id(),
                                      vllm::kFE2M1f.id(),
                                      vllm::kBFloat16.id(),
                                      vllm::kFE4M3fn.id(), 128, 1, 8, 4, true,
                                      4, 1, false>,
             128);
      break;
    case MarlinVariant::k128x64:
      launch(atlas_marlin_moe::Marlin<vllm::kBFloat16.id(),
                                      vllm::kFE2M1f.id(),
                                      vllm::kBFloat16.id(),
                                      vllm::kFE4M3fn.id(), 128, 1, 4, 8, true,
                                      4, 1, false>,
             128);
      break;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int atlas_marlin_moe_cache_bytes() {
  return marlin_cache_bytes();
}
