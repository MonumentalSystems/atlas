// SPDX-License-Identifier: AGPL-3.0-only
// Single-launch Sm120 NVFP4 grouped GEMM for Holo MoE Phase-2.
// Replaces the per-expert dense-collective loop (atlas_cutlass_nvfp4_grouped_gate_up)
// with one GemmUniversalMode::kGrouped launch over all active experts.
//
// Style/types mirror the dense binding cutlass_nvfp4_gemm.cu: same Sm120 /
// OpClassBlockScaledTensorOp / nv_float4_t<e2m1> / float_ue4m3_t SF collective,
// same #ifdef arch guard, same extern "C" status-code convention. The only new
// machinery here is the single-launch grouped host assembly (per-group problem
// shapes / pointers / strides / SF-layouts) plus a per-group activation pack into
// the grouped SFA atom.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

// ─── element + layout aliases (IDENTICAL to dense cutlass_nvfp4_gemm.cu:23-42) ───
using ElementInput = cutlass::float_e2m1_t;
using ElementA = cutlass::nv_float4_t<ElementInput>;
using ElementB = cutlass::nv_float4_t<ElementInput>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementAccumulator = float;
using ElementCompute = float;

// pointer-to-layout  ⇒  selects GROUPED (IsGroupedGemmKernel)
using GmemLayoutA = cutlass::layout::RowMajor;
using GmemLayoutB = cutlass::layout::ColumnMajor;
using GmemLayoutC = cutlass::layout::RowMajor;  // dense path uses RowMajor C/D; keep it
using GmemLayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;  // = 128 / 4   (FP4 elems)
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // = 8
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // = 8

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape = Shape<_128, _128, _128>;  // matches dense ThreadBlockShape
using ClusterShape = Shape<_1, _1, _1>;

// ─── EPILOGUE (plain LinearCombination; beta=0; per-expert scale2 via alpha_ptr) ───
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutC*,  // trailing '*' ⇒ grouped
    AlignmentC,
    ElementD,
    GmemLayoutD*,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

// ─── MAINLOOP (pointer-to-layout ⇒ grouped; all-TMA pingpong) ───
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    GmemLayoutA*,
    AlignmentA,
    ElementB,
    GmemLayoutB*,
    AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;
// Fallback if can_implement rejects pingpong on sm_121f:
//   cutlass::gemm::collective::KernelScheduleAuto   (cooperative)

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// per-group internal types (pointer types at the Arguments boundary)
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ProblemShape = cute::Shape<int, int, int>;

static inline size_t align_up_(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

// FP4 (e2m1) quantization helper — identical to dense float_to_e2m1 (cu:95-117).
__device__ __forceinline__ unsigned char float_to_e2m1_g(float x) {
  unsigned char sign = (x < 0.0f) ? 8u : 0u;
  float ax = fabsf(x);
  unsigned char mag;
  if (ax <= 0.25f) {
    mag = 0;
  } else if (ax <= 0.75f) {
    mag = 1;
  } else if (ax <= 1.25f) {
    mag = 2;
  } else if (ax <= 1.75f) {
    mag = 3;
  } else if (ax <= 2.5f) {
    mag = 4;
  } else if (ax <= 3.5f) {
    mag = 5;
  } else if (ax <= 5.0f) {
    mag = 6;
  } else {
    mag = 7;
  }
  return sign | mag;
}

// ─── per-group activation pack into the GROUPED SFA atom ───
// Identical body to dense atlas_cutlass_pack_bf16_act_nvfp4 (cu:125-161) but the
// kernel receives the per-group layout_sfa built from THAT group's {m,n,k}.
template <class LayoutSFA_t>
__global__ void pack_act_group(
    const __nv_bfloat16* __restrict__ act_global,  // TOKEN-MAJOR base [*, k]
    const int* __restrict__ sorted_token_ids,      // null => identity (expert-contig)
    int ms,                                         // group's first sorted row
    unsigned char* __restrict__ packed,            // group packed-A region [m_e, k/2]
    unsigned char* __restrict__ scales,            // group SFA region
    int m,
    int k,
    LayoutSFA_t layout_sfa) {
  int row = blockIdx.x;
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  int groups = k / 16;
  if (row >= m || group >= groups) {
    return;
  }
  // Fused gather (lever 2): the activation row for this group's local `row` is the
  // sorted token id — read token-major A directly, no separate permute pass.
  int gid = ms + row;
  int tok = sorted_token_ids ? sorted_token_ids[gid] : gid;
  const __nv_bfloat16* arow = act_global + (unsigned long long)tok * k;
  int base = group * 16;
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float v = __bfloat162float(arow[base + i]);
    max_abs = fmaxf(max_abs, fabsf(v));
  }
  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  cutlass::float_ue4m3_t sf(scale);
  scales[layout_sfa(row, base, 0)] = *reinterpret_cast<unsigned char*>(&sf);
  float dec = static_cast<float>(sf);
  float inv = dec > 0.0f ? 1.0f / dec : 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    float v0 = __bfloat162float(arow[base + i]) * inv;
    float v1 = __bfloat162float(arow[base + i + 1]) * inv;
    packed[(unsigned long long)row * (k / 2) + base / 2 + i / 2] =
        static_cast<unsigned char>(float_to_e2m1_g(v0) | (float_to_e2m1_g(v1) << 4));
  }
}

// ─── per-{n,k} SFB swizzle pack (load-time helper) ───
// Reads Atlas-transposed E4M3 weight scale [K/16, N] (the pack_bf16_weight_to_nvfp4_t
// layout) and writes it into the grouped/dense SFB atom for one expert. SFB depends
// ONLY on N,K (not M), so a single load-time call is valid for all per-group M.
template <class LayoutSFB_t>
__global__ void pack_weight_sfb_group(
    const unsigned char* __restrict__ atlas_scales_t,  // [K/16, N] E4M3
    unsigned char* __restrict__ cutlass_scales,        // swizzled SFB out
    int n,
    int k,
    LayoutSFB_t layout_sfb) {
  int col = blockIdx.x;
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  int groups = k / 16;
  if (col >= n || group >= groups) {
    return;
  }
  unsigned char atlas_scale = atlas_scales_t[(unsigned long long)group * n + col];
  __nv_fp8_e4m3 in;
  *reinterpret_cast<unsigned char*>(&in) = atlas_scale;
  float scale = static_cast<float>(in);
  cutlass::float_ue4m3_t sf(scale);
  cutlass_scales[layout_sfb(col, group * 16, 0)] = *reinterpret_cast<unsigned char*>(&sf);
}

#endif  // arch guard for device code

// ════════════════════════════════════════════════════════════════════════════
// Load-time SFB swizzle pack — produces the grouped/dense SFB atom for one expert
// from the Atlas-transposed [K/16,N] E4M3 weight scale. SFB is M-independent, so
// this is a one-time-per-expert call (gated by FAST_MOE_MODE at the Rust layer).
// ════════════════════════════════════════════════════════════════════════════
extern "C" int atlas_cutlass_pack_weight_sfb(
    const void* scale_in,  // [K/16, N] E4M3 (Atlas transposed)
    void* scale_out,       // swizzled SFB (ue4m3)
    int n,
    int k,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (k % 16) != 0) {
    return -1;
  }
  // SFB layout depends only on N,K — M is a placeholder (use 1).
  auto layout_sfb =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(1, n, k, 1));
  dim3 block(256);
  dim3 grid(n, (k / 16 + block.x - 1) / block.x);
  pack_weight_sfb_group<<<grid, block, 0, stream>>>(
      static_cast<const unsigned char*>(scale_in),
      static_cast<unsigned char*>(scale_out),
      n,
      k,
      layout_sfb);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -static_cast<int>(err);
#else
  (void)scale_in;
  (void)scale_out;
  (void)n;
  (void)k;
  (void)stream;
  return -120;
#endif
}

// ════════════════════════════════════════════════════════════════════════════
// Core: one grouped NVFP4 projection as a single kGrouped launch (gate, up, or
// down). A_global is TOKEN-MAJOR; sorted_token_ids (null=identity) selects each
// group's rows — the gather is FUSED into the per-group A-pack (lever 2: no
// separate permute pass). Workspace carve: [ packed-A | SFA | arrays | gemm_ws ].
// ════════════════════════════════════════════════════════════════════════════
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
static int run_grouped_one(
    const __nv_bfloat16* A_global,
    const int* sorted_token_ids,
    const unsigned long long* packed_ptrs,
    const unsigned long long* sfb_ptrs,
    const float* scale2_vals,
    __nv_bfloat16* C_bf16,
    const int* expert_offsets_host,
    int num_experts,
    int n,
    int k,
    unsigned char* ws,
    size_t workspace_size,
    cudaStream_t stream,
    int tag) {
  std::vector<ProblemShape> host_shapes;
  host_shapes.reserve(num_experts);
  std::vector<const ElementA::DataType*> hA;
  std::vector<const ElementB::DataType*> hB;
  std::vector<const ElementSF*> hSFA;
  std::vector<const ElementSF*> hSFB;
  std::vector<const ElementC*> hC;
  std::vector<ElementD*> hD;
  std::vector<StrideA> sA;
  std::vector<StrideB> sB;
  std::vector<StrideC> sC;
  std::vector<StrideD> sD;
  std::vector<LayoutSFA> lSFA;
  std::vector<LayoutSFB> lSFB;
  std::vector<float> alpha_host;

  // First pass: per-group padded sizes for the packed-A / SFA staging carve.
  std::vector<size_t> a_grp_off;
  std::vector<size_t> sfa_grp_off;
  size_t a_off = 0;
  size_t sfa_off = 0;
  size_t cursor = 0;
  {
    size_t a_acc = 0;
    size_t sfa_acc = 0;
    for (int e = 0; e < num_experts; ++e) {
      int ms = expert_offsets_host[e];
      int me = expert_offsets_host[e + 1];
      int m_e = me - ms;
      if (m_e <= 0) {
        continue;
      }
      auto lsa =
          Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m_e, n, k, 1));
      a_grp_off.push_back(a_acc);
      sfa_grp_off.push_back(sfa_acc);
      a_acc += align_up_((size_t)m_e * (k / 2), 256);
      sfa_acc += align_up_((size_t)size(filter_zeros(lsa)), 256);
    }
    a_off = 0;
    sfa_off = align_up_(a_acc, 256);
    cursor = align_up_(sfa_off + sfa_acc, 256);
  }

  // Second pass: gather+pack A per group, build the device argument arrays.
  int gi = 0;
  for (int e = 0; e < num_experts; ++e) {
    int ms = expert_offsets_host[e];
    int me = expert_offsets_host[e + 1];
    int m_e = me - ms;
    if (m_e <= 0) {
      continue;
    }
    auto lsa =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m_e, n, k, 1));
    auto lsb =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m_e, n, k, 1));
    unsigned char* a_e = ws + a_off + a_grp_off[gi];
    unsigned char* sfa_e = ws + sfa_off + sfa_grp_off[gi];

    dim3 blk(256);
    dim3 grd(m_e, (k / 16 + blk.x - 1) / blk.x);
    pack_act_group<<<grd, blk, 0, stream>>>(
        A_global, sorted_token_ids, ms, a_e, sfa_e, m_e, k, lsa);

    host_shapes.push_back(ProblemShape{m_e, n, k});
    hA.push_back(reinterpret_cast<const ElementA::DataType*>(a_e));
    hB.push_back(reinterpret_cast<const ElementB::DataType*>(packed_ptrs[e]));
    hSFA.push_back(reinterpret_cast<const ElementSF*>(sfa_e));
    hSFB.push_back(reinterpret_cast<const ElementSF*>(sfb_ptrs[e]));
    hC.push_back(reinterpret_cast<const ElementC*>(C_bf16 + (size_t)ms * n));
    hD.push_back(reinterpret_cast<ElementD*>(C_bf16 + (size_t)ms * n));
    sA.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m_e, k, 1}));
    sB.push_back(cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
    sC.push_back(cutlass::make_cute_packed_stride(StrideC{}, {m_e, n, 1}));
    sD.push_back(cutlass::make_cute_packed_stride(StrideD{}, {m_e, n, 1}));
    lSFA.push_back(lsa);
    lSFB.push_back(lsb);
    alpha_host.push_back(scale2_vals[e]);
    ++gi;
  }
  int G = (int)host_shapes.size();
  if (G == 0) {
    return 0;
  }

  auto put = [&](const void* src, size_t bytes) -> void* {
    void* dst = ws + cursor;
    cursor = align_up_(cursor + bytes, 256);
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return dst;
  };
  auto* dShapes = (ProblemShape*)put(host_shapes.data(), G * sizeof(ProblemShape));
  auto* dA = (const ElementA::DataType**)put(hA.data(), G * sizeof(void*));
  auto* dB = (const ElementB::DataType**)put(hB.data(), G * sizeof(void*));
  auto* dSFA = (const ElementSF**)put(hSFA.data(), G * sizeof(void*));
  auto* dSFB = (const ElementSF**)put(hSFB.data(), G * sizeof(void*));
  auto* dC = (const ElementC**)put(hC.data(), G * sizeof(void*));
  auto* dD = (ElementD**)put(hD.data(), G * sizeof(void*));
  auto* dsA = (StrideA*)put(sA.data(), G * sizeof(StrideA));
  auto* dsB = (StrideB*)put(sB.data(), G * sizeof(StrideB));
  auto* dsC = (StrideC*)put(sC.data(), G * sizeof(StrideC));
  auto* dsD = (StrideD*)put(sD.data(), G * sizeof(StrideD));
  auto* dlSFA = (LayoutSFA*)put(lSFA.data(), G * sizeof(LayoutSFA));
  auto* dlSFB = (LayoutSFB*)put(lSFB.data(), G * sizeof(LayoutSFB));
  auto* dAlpha = (float*)put(alpha_host.data(), G * sizeof(float));

  cutlass::KernelHardwareInfo hw{};
  hw.device_id = 0;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  typename Gemm::GemmKernel::CollectiveMainloop::Arguments mainloop_args{
      dA, dsA, dB, dsB, dSFA, dlSFA, dSFB, dlSFB};

  // LinearCombination: D = alpha*acc + beta*C, alpha per-group via ptr (scale2).
  typename Gemm::GemmKernel::CollectiveEpilogue::Arguments epi_args{};
  epi_args.thread.alpha = 1.0f;
  epi_args.thread.beta = 0.0f;
  epi_args.thread.alpha_ptr = dAlpha;
  epi_args.ptr_C = dC;
  epi_args.dC = dsC;
  epi_args.ptr_D = dD;
  epi_args.dD = dsD;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {G, dShapes, host_shapes.data()},
      mainloop_args,
      epi_args,
      hw};

  Gemm gemm;
  size_t need = Gemm::get_workspace_size(args);
  if (cursor + need > workspace_size) {
    return -2;
  }
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    return tag + (-50);
  }
  cutlass::Status st = gemm.initialize(args, ws + cursor, stream);
  if (st != cutlass::Status::kSuccess) {
    return tag + static_cast<int>(st);
  }
  st = gemm.run(stream);
  return st == cutlass::Status::kSuccess ? 0 : tag + static_cast<int>(st);
}
#endif  // arch guard for run_grouped_one

// ════════════════════════════════════════════════════════════════════════════
// PUBLIC ENTRY — single-launch grouped gate_up (gather FUSED; lever 2).
//   A_bf16              : [num_tokens, K] bf16 TOKEN-MAJOR (NOT pre-permuted)
//   sorted_token_ids    : device int[M_total] — each sorted row's token index
//   *_packed_ptrs[e]    : device ptr -> [N,K/2] e2m1            (decode gate_ptrs)
//   *_sfb_ptrs[e]       : device ptr -> swizzled SFB (ue4m3)    (load-built)
//   *_scale2_vals       : HOST f32[num_experts]  (per-expert weight_scale_2)
//   C_gate/C_up         : [M_total, N] bf16, expert-contiguous (sorted) output
//   expert_offsets_host : HOST int32[num_experts+1]
// Runs gate then up as two kGrouped launches.
// ════════════════════════════════════════════════════════════════════════════
extern "C" int atlas_cutlass_nvfp4_grouped_gate_up_fused(
    const void* A_bf16,
    const int* sorted_token_ids,
    const unsigned long long* gate_packed_ptrs,
    const unsigned long long* gate_sfb_ptrs,
    const float* gate_scale2_vals,
    const unsigned long long* up_packed_ptrs,
    const unsigned long long* up_sfb_ptrs,
    const float* up_scale2_vals,
    void* C_gate_bf16,
    void* C_up_bf16,
    const int* expert_offsets_host,
    int num_experts,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (k % 16) != 0 || num_experts <= 0) {
    return -1;
  }
  unsigned char* ws = static_cast<unsigned char*>(workspace);
  int rc = run_grouped_one(
      static_cast<const __nv_bfloat16*>(A_bf16), sorted_token_ids, gate_packed_ptrs,
      gate_sfb_ptrs, gate_scale2_vals, static_cast<__nv_bfloat16*>(C_gate_bf16),
      expert_offsets_host, num_experts, n, k, ws, workspace_size, stream, 100000);
  if (rc) {
    return rc;
  }
  rc = run_grouped_one(
      static_cast<const __nv_bfloat16*>(A_bf16), sorted_token_ids, up_packed_ptrs,
      up_sfb_ptrs, up_scale2_vals, static_cast<__nv_bfloat16*>(C_up_bf16),
      expert_offsets_host, num_experts, n, k, ws, workspace_size, stream, 200000);
  return rc;
#else
  (void)A_bf16;
  (void)sorted_token_ids;
  (void)gate_packed_ptrs;
  (void)gate_sfb_ptrs;
  (void)gate_scale2_vals;
  (void)up_packed_ptrs;
  (void)up_sfb_ptrs;
  (void)up_scale2_vals;
  (void)C_gate_bf16;
  (void)C_up_bf16;
  (void)expert_offsets_host;
  (void)num_experts;
  (void)n;
  (void)k;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}

// ════════════════════════════════════════════════════════════════════════════
// PUBLIC ENTRY — single-launch grouped DOWN (lever 1). A is the post-SiLU
// intermediate [M_total, K=inter], ALREADY expert-contiguous (sorted) — no gather
// (sorted_token_ids=null). B = down_proj [N=hidden, K/2], C = [M_total, N=hidden].
// ════════════════════════════════════════════════════════════════════════════
extern "C" int atlas_cutlass_nvfp4_grouped_down(
    const void* A_bf16,
    const unsigned long long* packed_ptrs,
    const unsigned long long* sfb_ptrs,
    const float* scale2_vals,
    void* C_bf16,
    const int* expert_offsets_host,
    int num_experts,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (k % 16) != 0 || num_experts <= 0) {
    return -1;
  }
  return run_grouped_one(
      static_cast<const __nv_bfloat16*>(A_bf16), nullptr, packed_ptrs, sfb_ptrs,
      scale2_vals, static_cast<__nv_bfloat16*>(C_bf16), expert_offsets_host,
      num_experts, n, k, static_cast<unsigned char*>(workspace), workspace_size,
      stream, 300000);
#else
  (void)A_bf16;
  (void)packed_ptrs;
  (void)sfb_ptrs;
  (void)scale2_vals;
  (void)C_bf16;
  (void)expert_offsets_host;
  (void)num_experts;
  (void)n;
  (void)k;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}
