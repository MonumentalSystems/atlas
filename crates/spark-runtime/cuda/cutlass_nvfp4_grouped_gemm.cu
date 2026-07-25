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

// ─── BATCHED A-pack: one launch for ALL active expert groups ───
//
// The per-group `pack_act_group` above was launched in a host loop, once per
// active expert. With 256 experts that is ~256 launches per projection x 2
// projections x 47 MoE layers = ~11.5k launches per prefill — measured (nsys)
// as 89.3% of all prefill kernel launches for 2% of prefill GPU time, i.e.
// ~345 ms of pure host launch overhead against 32 ms of actual work. That was
// the dominant term in Laguna's C=1 TTFT.
//
// This does the identical computation with grid.z indexing the group, so the
// whole loop becomes ONE launch. Per-group scalars come in as device arrays.
// grid.x is sized to max(m_e) with an early exit for the ragged tail.
template <class LayoutSFA_t>
__global__ void pack_act_grouped_batched(
    const __nv_bfloat16* __restrict__ act_global,  // TOKEN-MAJOR base [*, k]
    const int* __restrict__ sorted_token_ids,      // null => identity
    const int* __restrict__ ms_arr,                // [G] group's first sorted row
    const int* __restrict__ m_arr,                 // [G] group row count
    unsigned char* const* __restrict__ packed_arr, // [G] group packed-A region
    unsigned char* const* __restrict__ scales_arr, // [G] group SFA region
    int k,
    LayoutSFA_t layout_sfa_dummy) {
  const int e = blockIdx.z;
  const int m_e = m_arr[e];
  int row = blockIdx.x;
  if (row >= m_e) {
    return;  // ragged tail: grid.x is max(m_e)
  }
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  const int groups = k / 16;
  if (group >= groups) {
    return;
  }
  // The SFA atom's layout depends on this group's m_e, so rebuild it here
  // rather than baking one per launch.
  auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m_e, 1, k, 1));
  (void)layout_sfa_dummy;

  unsigned char* packed = packed_arr[e];
  unsigned char* scales = scales_arr[e];
  const int ms = ms_arr[e];

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
    const unsigned char* __restrict__ atlas_scales,  // [K/16,N] (K-major) or [N,K/16] (N-major)
    unsigned char* __restrict__ cutlass_scales,      // swizzled SFB out
    int n,
    int k,
    int src_n_major,  // 0 = Atlas-transposed [K/16,N]; 1 = checkpoint-native [N,K/16]
    LayoutSFB_t layout_sfb) {
  int col = blockIdx.x;
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  int groups = k / 16;
  if (col >= n || group >= groups) {
    return;
  }
  // SFB output layout is unchanged; only the SOURCE indexing differs. N-major
  // lets a checkpoint that ships [N,K/16] scales (Laguna) build SFB without
  // materialising an Atlas-transposed copy first.
  unsigned char atlas_scale =
      src_n_major ? atlas_scales[(unsigned long long)col * groups + group]
                  : atlas_scales[(unsigned long long)group * n + col];
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
    const void* scale_in,  // [K/16,N] E4M3 (Atlas transposed) or [N,K/16] when src_n_major
    void* scale_out,       // swizzled SFB (ue4m3)
    int n,
    int k,
    int src_n_major,
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
      src_n_major,
      layout_sfb);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -static_cast<int>(err);
#else
  (void)scale_in;
  (void)scale_out;
  (void)n;
  (void)k;
  (void)src_n_major;
  (void)stream;
  return -120;
#endif
}

// ════════════════════════════════════════════════════════════════════════════
// Grouped NVFP4 MoE: A-pack (shared across projections) + per-projection GEMM.
// A_global is TOKEN-MAJOR; sorted_token_ids (null=identity) selects each group's
// rows — the gather is FUSED into the per-group A-pack (lever 2: no permute pass).
// gate_up packs A ONCE and runs gate+up against it (lever: pack-A-once); down packs
// its own (different) A. Workspace carve: [ packed-A | SFA | A-arrays | B-arrays+gemm_ws ].
// ════════════════════════════════════════════════════════════════════════════
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

// Shared per-forward A-pack result: packed-A + SFA staged in ws, A-side argument
// arrays uploaded. Reused by every projection that shares this A.
struct GroupedAPrep {
  int status = 0;                          // non-zero on failure
  int G = 0;                               // active groups
  std::vector<ProblemShape> host_shapes;   // {m_e, n, k} per group
  std::vector<int> ms;                     // sorted-row start per group (C offset)
  std::vector<int> me;                     // m_e per group
  std::vector<int> eidx;                   // expert index per group (B/scale2 lookup)
  ProblemShape* dShapes = nullptr;
  const ElementA::DataType** dA = nullptr;
  const ElementSF** dSFA = nullptr;
  StrideA* dsA = nullptr;
  LayoutSFA* dlSFA = nullptr;
  size_t cursor = 0;                       // free ws offset after the A-side arrays
};

// Gather+pack A once (per active group), upload the A-side argument arrays.
static GroupedAPrep prep_grouped_a(
    const __nv_bfloat16* A_global,
    const int* sorted_token_ids,
    const int* expert_offsets_host,
    int num_experts,
    int n,
    int k,
    unsigned char* ws,
    cudaStream_t stream) {
  GroupedAPrep p;
  std::vector<const ElementA::DataType*> hA;
  std::vector<const ElementSF*> hSFA;
  std::vector<StrideA> sA;

  // First pass: per-group padded sizes for the packed-A / SFA staging carve.
  std::vector<size_t> a_grp_off;
  std::vector<size_t> sfa_grp_off;
  size_t a_acc = 0;
  size_t sfa_acc = 0;
  for (int e = 0; e < num_experts; ++e) {
    int m_e = expert_offsets_host[e + 1] - expert_offsets_host[e];
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
  size_t a_off = 0;
  size_t sfa_off = align_up_(a_acc, 256);
  size_t cursor = align_up_(sfa_off + sfa_acc, 256);

  // Second pass: collect per-group A-pack params + build A-side host arrays.
  // The A-pack itself is issued ONCE after the loop.
  std::vector<int> h_ms, h_me;
  std::vector<unsigned char*> h_apk, h_sfa;
  int max_me = 0;
  int gi = 0;
  for (int e = 0; e < num_experts; ++e) {
    int ms = expert_offsets_host[e];
    int m_e = expert_offsets_host[e + 1] - ms;
    if (m_e <= 0) {
      continue;
    }
    auto lsa =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m_e, n, k, 1));
    unsigned char* a_e = ws + a_off + a_grp_off[gi];
    unsigned char* sfa_e = ws + sfa_off + sfa_grp_off[gi];

    dim3 blk(256);
    dim3 grd(m_e, (k / 16 + blk.x - 1) / blk.x);
    // A-pack is BATCHED after this loop (one launch for all groups) — see
    // pack_act_grouped_batched. Collect this group's scalars instead of
    // launching per group.
    h_ms.push_back(ms);
    h_me.push_back(m_e);
    h_apk.push_back(a_e);
    h_sfa.push_back(sfa_e);
    if (m_e > max_me) max_me = m_e;
    (void)lsa;

    p.host_shapes.push_back(ProblemShape{m_e, n, k});
    p.ms.push_back(ms);
    p.me.push_back(m_e);
    p.eidx.push_back(e);
    hA.push_back(reinterpret_cast<const ElementA::DataType*>(a_e));
    hSFA.push_back(reinterpret_cast<const ElementSF*>(sfa_e));
    sA.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m_e, k, 1}));
    ++gi;
  }
  // ── ONE A-pack launch for every active group ──
  // Replaces ~256 launches per call (89.3% of all prefill kernel launches,
  // ~345 ms of host launch overhead for 32 ms of GPU work).
  if (!h_me.empty()) {
    const int G = (int)h_me.size();
    size_t ms_b = align_up_((size_t)G * sizeof(int), 256);
    size_t pk_b = align_up_((size_t)G * sizeof(void*), 256);
    unsigned char* d_ms = ws + cursor;
    unsigned char* d_me = d_ms + ms_b;
    unsigned char* d_apk = d_me + ms_b;
    unsigned char* d_sfa = d_apk + pk_b;
    cursor = align_up_(cursor + 2 * ms_b + 2 * pk_b, 256);
    cudaMemcpyAsync(d_ms, h_ms.data(), G * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_me, h_me.data(), G * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_apk, h_apk.data(), G * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sfa, h_sfa.data(), G * sizeof(void*), cudaMemcpyHostToDevice, stream);
    auto lsa0 = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(max_me, n, k, 1));
    dim3 blk(256);
    dim3 grd(max_me, (k / 16 + blk.x - 1) / blk.x, G);
    pack_act_grouped_batched<<<grd, blk, 0, stream>>>(
        A_global, sorted_token_ids, (const int*)d_ms, (const int*)d_me,
        (unsigned char* const*)d_apk, (unsigned char* const*)d_sfa, k, lsa0);
  }

  p.G = (int)p.host_shapes.size();
  if (p.G == 0) {
    p.cursor = cursor;
    return p;
  }

  auto put = [&](const void* src, size_t bytes) -> void* {
    void* dst = ws + cursor;
    cursor = align_up_(cursor + bytes, 256);
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return dst;
  };
  p.dShapes = (ProblemShape*)put(p.host_shapes.data(), p.G * sizeof(ProblemShape));
  p.dA = (const ElementA::DataType**)put(hA.data(), p.G * sizeof(void*));
  p.dSFA = (const ElementSF**)put(hSFA.data(), p.G * sizeof(void*));
  p.dsA = (StrideA*)put(sA.data(), p.G * sizeof(StrideA));
  {
    std::vector<LayoutSFA> lSFA(p.G);
    for (int g = 0; g < p.G; ++g) {
      lSFA[g] =
          Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(p.me[g], n, k, 1));
    }
    p.dlSFA = (LayoutSFA*)put(lSFA.data(), p.G * sizeof(LayoutSFA));
  }
  p.cursor = cursor;
  return p;
}

// One grouped projection GEMM reusing a shared A-prep. B-side arrays + gemm_ws are
// carved starting at `cursor_start` (reused across the projections sharing `a`).
static int launch_projection(
    const GroupedAPrep& a,
    const unsigned long long* packed_ptrs,
    const unsigned long long* sfb_ptrs,
    const float* scale2_vals,
    __nv_bfloat16* C_bf16,
    int n,
    int k,
    unsigned char* ws,
    size_t cursor_start,
    size_t workspace_size,
    cudaStream_t stream,
    int tag) {
  int G = a.G;
  if (G == 0) {
    return 0;
  }
  std::vector<const ElementB::DataType*> hB(G);
  std::vector<const ElementSF*> hSFB(G);
  std::vector<const ElementC*> hC(G);
  std::vector<ElementD*> hD(G);
  std::vector<StrideB> sB(G);
  std::vector<StrideC> sC(G);
  std::vector<StrideD> sD(G);
  std::vector<LayoutSFB> lSFB(G);
  std::vector<float> alpha_host(G);
  for (int g = 0; g < G; ++g) {
    int e = a.eidx[g];
    int m_e = a.me[g];
    size_t ms = (size_t)a.ms[g];
    hB[g] = reinterpret_cast<const ElementB::DataType*>(packed_ptrs[e]);
    hSFB[g] = reinterpret_cast<const ElementSF*>(sfb_ptrs[e]);
    hC[g] = reinterpret_cast<const ElementC*>(C_bf16 + ms * n);
    hD[g] = reinterpret_cast<ElementD*>(C_bf16 + ms * n);
    sB[g] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    sC[g] = cutlass::make_cute_packed_stride(StrideC{}, {m_e, n, 1});
    sD[g] = cutlass::make_cute_packed_stride(StrideD{}, {m_e, n, 1});
    lSFB[g] =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m_e, n, k, 1));
    alpha_host[g] = scale2_vals[e];
  }

  size_t cursor = cursor_start;
  auto put = [&](const void* src, size_t bytes) -> void* {
    void* dst = ws + cursor;
    cursor = align_up_(cursor + bytes, 256);
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return dst;
  };
  auto* dB = (const ElementB::DataType**)put(hB.data(), G * sizeof(void*));
  auto* dSFB = (const ElementSF**)put(hSFB.data(), G * sizeof(void*));
  auto* dC = (const ElementC**)put(hC.data(), G * sizeof(void*));
  auto* dD = (ElementD**)put(hD.data(), G * sizeof(void*));
  auto* dsB = (StrideB*)put(sB.data(), G * sizeof(StrideB));
  auto* dsC = (StrideC*)put(sC.data(), G * sizeof(StrideC));
  auto* dsD = (StrideD*)put(sD.data(), G * sizeof(StrideD));
  auto* dlSFB = (LayoutSFB*)put(lSFB.data(), G * sizeof(LayoutSFB));
  auto* dAlpha = (float*)put(alpha_host.data(), G * sizeof(float));
  // Per-group alpha (scale2) needs alpha_ptr_array (G POINTERS, one per group →
  // &dAlpha[g]); the scalar alpha_ptr would apply dAlpha[0] to every group.
  std::vector<const float*> hAlphaPtr(G);
  for (int g = 0; g < G; ++g) {
    hAlphaPtr[g] = dAlpha + g;
  }
  auto* dAlphaPtr = (const float**)put(hAlphaPtr.data(), G * sizeof(const float*));

  cutlass::KernelHardwareInfo hw{};
  hw.device_id = 0;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  typename Gemm::GemmKernel::CollectiveMainloop::Arguments mainloop_args{
      a.dA, a.dsA, dB, dsB, a.dSFA, a.dlSFA, dSFB, dlSFB};

  typename Gemm::GemmKernel::CollectiveEpilogue::Arguments epi_args{};
  epi_args.thread.alpha = 1.0f;
  epi_args.thread.beta = 0.0f;
  epi_args.thread.alpha_ptr_array = dAlphaPtr;
  epi_args.ptr_C = dC;
  epi_args.dC = dsC;
  epi_args.ptr_D = dD;
  epi_args.dD = dsD;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {G, a.dShapes, const_cast<ProblemShape*>(a.host_shapes.data())},
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
#endif  // arch guard

// ════════════════════════════════════════════════════════════════════════════
// PUBLIC ENTRY — grouped gate_up. A_bf16 [num_tokens,K] TOKEN-MAJOR; the gather is
// fused into the A-pack via sorted_token_ids. A is packed ONCE and shared by the
// gate and up kGrouped launches. *_packed_ptrs[e]=[N,K/2] e2m1, *_sfb_ptrs[e]=
// swizzled SFB, *_scale2_vals=HOST f32[num_experts]. C_*=[M_total,N] sorted output.
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
  // Pack A ONCE (gate + up share the same activation).
  GroupedAPrep a = prep_grouped_a(static_cast<const __nv_bfloat16*>(A_bf16),
                                  sorted_token_ids, expert_offsets_host, num_experts,
                                  n, k, ws, stream);
  if (a.G == 0) {
    return 0;
  }
  // Both projections carve their B-arrays + gemm_ws from a.cursor (gate's launch
  // completes on the stream before up's overwrites the region — serialized).
  int rc = launch_projection(a, gate_packed_ptrs, gate_sfb_ptrs, gate_scale2_vals,
                             static_cast<__nv_bfloat16*>(C_gate_bf16), n, k, ws,
                             a.cursor, workspace_size, stream, 100000);
  if (rc) {
    return rc;
  }
  rc = launch_projection(a, up_packed_ptrs, up_sfb_ptrs, up_scale2_vals,
                         static_cast<__nv_bfloat16*>(C_up_bf16), n, k, ws, a.cursor,
                         workspace_size, stream, 200000);
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
// DEVICE-OFFSET grouped path (ATLAS_MOE_CUTLASS_DEVICE_OFFSETS=1) —
// CUDA-graph-capture-safe variant of the host-driven entries above.
//
// The host-driven path D2H-copies `expert_offsets` + synchronizes to build the
// per-group problem_sizes/pointer arrays on the host — illegal under CUDA graph
// capture (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, 900), which forces the whole
// multi-seq decode step eager whenever the grouped MoE fires.
//
// This variant keeps the SAME Sm120 grouped GEMM kernel but drives it entirely
// from the device:
//   • num_groups = num_experts (ALL experts, empty groups get M_k = 0 — the
//     GroupScheduler's warp-speculative group walk skips 0-tile groups; see
//     sm90_tile_scheduler_group.hpp get_work_idx_m_and_n).
//   • per-group {M_k,N,K} problem shapes + A/SFA/C pointer + stride + SF-layout
//     arrays are written by build_group_args_dev, a single-block kernel that
//     reads the device `expert_offsets` (M_k = off[k+1]-off[k]).
//   • B / SFB / alpha arrays: the device-resident per-expert u64 pointer tables
//     (`*_ptrs.packed_ptrs`, `*_sfb_cutlass`) are passed through DIRECTLY —
//     group index == expert index, so no snapshot/copy at all. alpha_ptr_array
//     entries (&scale2[e]) are written by the same prep kernel.
//   • host_problem_shapes = nullptr ⇒ the tile scheduler launches a FIXED
//     hw_info.sm_count persistent grid (sm90_tile_scheduler_group.hpp:221-222)
//     — hardware-constant, hence capture-legal.
//   • gemm.initialize()/get_workspace_size()/can_implement() are host-pure for
//     the grouped kernel (mainloop/epilogue to_underlying_arguments build the
//     template TMA descriptors from TileShape, not host shapes; all
//     initialize_workspace hooks are no-ops) — verified against the vendored
//     CUTLASS 4.6.0 headers. The launch itself is a plain `<<<>>>`.
// NO cudaMemcpy(D2H or pageable-H2D of transients), NO synchronize, NO
// allocation anywhere on this path — everything is carved from the caller's
// fixed workspace at host-arithmetic offsets (m_total is host-known: n*top_k).
// ════════════════════════════════════════════════════════════════════════════
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

static int cached_sm_count_() {
  static int c = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  return c;
}

// Exact SFA staging bytes for an [m, k] group (1-byte SF elements). Shared
// host/device formula; the host carve bound additionally relies on this being
// linear in ceil(m/128), which the public entries verify per (n,k) (cheap,
// host-pure) before trusting it.
static inline size_t sfa_bytes_host_(int m, int n, int k) {
  auto lsa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  return (size_t)cute::size(cute::filter_zeros(lsa));
}

// Single-block prep: read device expert_offsets, prefix-sum the per-group SFA
// staging sizes, and write EVERY per-group argument array the grouped GEMM
// needs. Replaces the host loop + 9 cudaMemcpyAsync uploads of the host-driven
// path. Pointer arrays are written as u64 (bit-identical to the void* arrays
// CUTLASS reads). Grid: (1,1,1) Block: (256,1,1).
__global__ void build_group_args_dev(
    const int* __restrict__ expert_offsets,  // [G+1] device prefix sum
    int G,
    int n,
    int k,
    const unsigned char* packed_a_base,      // packed-A staging region
    unsigned char* sfa_base,                 // SFA staging region
    ProblemShape* __restrict__ shapes,       // [G] {m_e, n, k}
    unsigned long long* __restrict__ ptr_a,  // [G] group packed-A ptr
    unsigned long long* __restrict__ ptr_sfa,  // [G] group SFA ptr
    LayoutSFA* __restrict__ l_sfa,           // [G]
    StrideA* __restrict__ ds_a,              // [G] (M-independent, replicated)
    StrideB* __restrict__ ds_b,              // [G]
    StrideC* __restrict__ ds_c,              // [G]
    StrideD* __restrict__ ds_d,              // [G]
    LayoutSFB* __restrict__ l_sfb,           // [G] (M-independent, replicated)
    const float* scale2_0,                   // device [G] (projection 0)
    unsigned long long* __restrict__ alpha0,  // [G] &scale2_0[e]
    const float* scale2_1,                   // nullable (projection 1)
    unsigned long long* __restrict__ alpha1,  // nullable
    unsigned long long c0_base,              // projection-0 output base
    unsigned long long* __restrict__ ptr_c0,  // [G]
    unsigned long long c1_base,              // nullable pair
    unsigned long long* __restrict__ ptr_c1) {
  __shared__ unsigned long long sfa_cum[1025];  // G <= 1024 (checked host-side)
  const int tid = threadIdx.x;
  for (int e = tid; e < G; e += blockDim.x) {
    const int m_e = expert_offsets[e + 1] - expert_offsets[e];
    unsigned long long bytes = 0;
    if (m_e > 0) {
      auto lsa =
          Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m_e, n, k, 1));
      bytes = (unsigned long long)cute::size(cute::filter_zeros(lsa));
    }
    sfa_cum[e + 1] = bytes;
  }
  __syncthreads();
  if (tid == 0) {
    sfa_cum[0] = 0;
    for (int e = 0; e < G; ++e) {
      sfa_cum[e + 1] += sfa_cum[e];
    }
  }
  __syncthreads();
  for (int e = tid; e < G; e += blockDim.x) {
    const int off = expert_offsets[e];
    const int m_e = expert_offsets[e + 1] - off;
    const int m_l = m_e > 0 ? m_e : 1;  // placeholder for never-read layouts
    shapes[e] = ProblemShape{m_e, n, k};
    ptr_a[e] =
        (unsigned long long)(packed_a_base + (unsigned long long)off * (k / 2));
    ptr_sfa[e] = (unsigned long long)(sfa_base + sfa_cum[e]);
    l_sfa[e] =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m_l, n, k, 1));
    ds_a[e] = cutlass::make_cute_packed_stride(StrideA{}, {m_l, k, 1});
    ds_b[e] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    ds_c[e] = cutlass::make_cute_packed_stride(StrideC{}, {m_l, n, 1});
    ds_d[e] = cutlass::make_cute_packed_stride(StrideD{}, {m_l, n, 1});
    l_sfb[e] =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m_l, n, k, 1));
    alpha0[e] = (unsigned long long)(scale2_0 + e);
    if (alpha1 != nullptr) {
      alpha1[e] = (unsigned long long)(scale2_1 + e);
    }
    const unsigned long long c_off = (unsigned long long)off * n * 2ull;
    ptr_c0[e] = c0_base + c_off;
    if (ptr_c1 != nullptr) {
      ptr_c1[e] = c1_base + c_off;
    }
  }
}

// A-pack for the device-offset path. Identical quantization math to
// pack_act_group/pack_act_grouped_batched, but indexed by GLOBAL sorted row
// (grid.x = m_total, host-known — no data-dependent grid): each row finds its
// expert via binary search over the device expert_offsets, writes its packed-A
// row at the globally contiguous row offset (group A ptr = base + off_e*(k/2),
// so local row lrow lands at base + row*(k/2)) and its SFA byte through the
// per-group SFA pointer written by build_group_args_dev.
__global__ void pack_act_dev_offsets(
    const __nv_bfloat16* __restrict__ act_global,  // TOKEN-MAJOR base [*, k]
    const int* __restrict__ sorted_token_ids,      // null => identity
    const int* __restrict__ expert_offsets,        // [G+1] device
    int G,
    unsigned char* __restrict__ packed_base,
    const unsigned long long* __restrict__ sfa_ptrs,  // [G] from prep
    int k) {
  const int row = blockIdx.x;  // grid.x == m_total exactly
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  const int groups = k / 16;
  if (group >= groups) {
    return;
  }
  // Last e with expert_offsets[e] <= row (duplicates from empty experts
  // resolve to the owning non-empty group: off[e] <= row < off[e+1]).
  int lo = 0;
  int hi = G;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (expert_offsets[mid] <= row) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const int e = lo;
  const int lrow = row - expert_offsets[e];
  const int m_e = expert_offsets[e + 1] - expert_offsets[e];
  // SFA layout is N-independent — same (m_e, 1, k) form as the batched packer.
  auto layout_sfa =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m_e, 1, k, 1));
  unsigned char* scales = reinterpret_cast<unsigned char*>(sfa_ptrs[e]);
  const int tok = sorted_token_ids ? sorted_token_ids[row] : row;
  const __nv_bfloat16* arow = act_global + (unsigned long long)tok * k;
  const int base = group * 16;
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float v = __bfloat162float(arow[base + i]);
    max_abs = fmaxf(max_abs, fabsf(v));
  }
  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  cutlass::float_ue4m3_t sf(scale);
  scales[layout_sfa(lrow, base, 0)] = *reinterpret_cast<unsigned char*>(&sf);
  float dec = static_cast<float>(sf);
  float inv = dec > 0.0f ? 1.0f / dec : 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    float v0 = __bfloat162float(arow[base + i]) * inv;
    float v1 = __bfloat162float(arow[base + i + 1]) * inv;
    packed_base[(unsigned long long)row * (k / 2) + base / 2 + i / 2] =
        static_cast<unsigned char>(float_to_e2m1_g(v0) | (float_to_e2m1_g(v1) << 4));
  }
}

// One grouped GEMM with fully device-resident per-group arguments and
// host_problem_shapes = nullptr (fixed sm_count grid — capture-legal). All
// host work here is pure arithmetic + the launch.
static int launch_projection_dev(
    int G,
    ProblemShape* d_shapes,
    const unsigned long long* d_a,
    StrideA* ds_a,
    const unsigned long long* d_sfa,
    LayoutSFA* dl_sfa,
    const unsigned long long* b_ptrs_dev,   // device per-expert table, as-is
    const unsigned long long* sfb_ptrs_dev,  // device per-expert table, as-is
    StrideB* ds_b,
    LayoutSFB* dl_sfb,
    const unsigned long long* d_alpha_ptr,
    const unsigned long long* d_c,
    StrideC* ds_c,
    StrideD* ds_d,
    unsigned char* ws,
    size_t cursor,
    size_t workspace_size,
    size_t* cursor_out,
    cudaStream_t stream,
    int tag) {
  cutlass::KernelHardwareInfo hw{};
  hw.device_id = 0;
  hw.sm_count = cached_sm_count_();

  typename Gemm::GemmKernel::CollectiveMainloop::Arguments mainloop_args{
      reinterpret_cast<const ElementA::DataType**>(
          const_cast<unsigned long long*>(d_a)),
      ds_a,
      reinterpret_cast<const ElementB::DataType**>(
          const_cast<unsigned long long*>(b_ptrs_dev)),
      ds_b,
      reinterpret_cast<const ElementSF**>(const_cast<unsigned long long*>(d_sfa)),
      dl_sfa,
      reinterpret_cast<const ElementSF**>(
          const_cast<unsigned long long*>(sfb_ptrs_dev)),
      dl_sfb};

  typename Gemm::GemmKernel::CollectiveEpilogue::Arguments epi_args{};
  epi_args.thread.alpha = 1.0f;
  epi_args.thread.beta = 0.0f;
  epi_args.thread.alpha_ptr_array =
      reinterpret_cast<const float**>(const_cast<unsigned long long*>(d_alpha_ptr));
  epi_args.ptr_C =
      reinterpret_cast<const ElementC**>(const_cast<unsigned long long*>(d_c));
  epi_args.dC = ds_c;
  epi_args.ptr_D =
      reinterpret_cast<ElementD**>(const_cast<unsigned long long*>(d_c));
  epi_args.dD = ds_d;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      // host_problem_shapes = nullptr ⇒ scheduler grid = hw_info.sm_count
      // (fixed) and every per-group decision moves on-device.
      {G, d_shapes, nullptr},
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
  if (cursor_out != nullptr) {
    *cursor_out = align_up_(cursor + need, 256);
  }
  return st == cutlass::Status::kSuccess ? 0 : tag + static_cast<int>(st);
}

// Per-call workspace carve for the device-offset path. All sizes are pure host
// arithmetic over (m_total, n, k, G) — m_total is n_lanes*top_k, known without
// touching the device. Returns 0 on success.
struct DevCarve {
  unsigned char* packed_a = nullptr;
  unsigned char* sfa = nullptr;
  ProblemShape* shapes = nullptr;
  unsigned long long* ptr_a = nullptr;
  unsigned long long* ptr_sfa = nullptr;
  LayoutSFA* l_sfa = nullptr;
  StrideA* ds_a = nullptr;
  StrideB* ds_b = nullptr;
  StrideC* ds_c = nullptr;
  StrideD* ds_d = nullptr;
  LayoutSFB* l_sfb = nullptr;
  unsigned long long* alpha0 = nullptr;
  unsigned long long* alpha1 = nullptr;
  unsigned long long* ptr_c0 = nullptr;
  unsigned long long* ptr_c1 = nullptr;
  size_t cursor = 0;  // first free byte after the arrays
};

static int carve_dev_(unsigned char* ws,
                      size_t ws_size,
                      int G,
                      int m_total,
                      int n,
                      int k,
                      bool dual,
                      DevCarve& d) {
  // Guard the carve's linear-in-ceil(m/128) SFA model (host-pure, cheap).
  const size_t atom = sfa_bytes_host_(128, n, k);
  if (sfa_bytes_host_(1, n, k) != atom || sfa_bytes_host_(129, n, k) != 2 * atom ||
      sfa_bytes_host_(257, n, k) != 3 * atom) {
    return -7;
  }
  size_t cur = 0;
  auto take = [&](size_t bytes) -> unsigned char* {
    unsigned char* p = ws + cur;
    cur = align_up_(cur + bytes, 256);
    return p;
  };
  d.packed_a = take((size_t)m_total * (k / 2));
  // Upper bound: sum_e ceil(m_e/128) <= ceil(m_total/128) + #nonempty, and
  // #nonempty <= min(G, m_total). The prep kernel packs groups exactly (no
  // per-group alignment padding; atom sizes are already 256B multiples).
  const size_t nonempty_max = (size_t)(G < m_total ? G : m_total);
  d.sfa = take(((size_t)(m_total + 127) / 128 + nonempty_max) * atom);
  d.shapes = (ProblemShape*)take((size_t)G * sizeof(ProblemShape));
  d.ptr_a = (unsigned long long*)take((size_t)G * 8);
  d.ptr_sfa = (unsigned long long*)take((size_t)G * 8);
  d.l_sfa = (LayoutSFA*)take((size_t)G * sizeof(LayoutSFA));
  d.ds_a = (StrideA*)take((size_t)G * sizeof(StrideA));
  d.ds_b = (StrideB*)take((size_t)G * sizeof(StrideB));
  d.ds_c = (StrideC*)take((size_t)G * sizeof(StrideC));
  d.ds_d = (StrideD*)take((size_t)G * sizeof(StrideD));
  d.l_sfb = (LayoutSFB*)take((size_t)G * sizeof(LayoutSFB));
  d.alpha0 = (unsigned long long*)take((size_t)G * 8);
  d.alpha1 = dual ? (unsigned long long*)take((size_t)G * 8) : nullptr;
  d.ptr_c0 = (unsigned long long*)take((size_t)G * 8);
  d.ptr_c1 = dual ? (unsigned long long*)take((size_t)G * 8) : nullptr;
  d.cursor = cur;
  return cur <= ws_size ? 0 : -2;
}
#endif  // arch guard

// ════════════════════════════════════════════════════════════════════════════
// PUBLIC ENTRY — DEVICE-OFFSET grouped gate_up (graph-capture-safe twin of
// atlas_cutlass_nvfp4_grouped_gate_up_fused). ALL array parameters are DEVICE
// pointers: *_ptrs_dev / *_sfb_ptrs_dev / *_scale2_dev are the per-expert
// tables as they already live on the GPU; expert_offsets is the device i32
// [num_experts+1] prefix sum from moe_sort_by_expert. m_total = total expanded
// rows (n_tokens * top_k), host-known. No D2H, no sync, no alloc.
// ════════════════════════════════════════════════════════════════════════════
extern "C" int atlas_cutlass_nvfp4_grouped_gate_up_dev(
    const void* A_bf16,
    const int* sorted_token_ids,
    const unsigned long long* gate_packed_ptrs_dev,
    const unsigned long long* gate_sfb_ptrs_dev,
    const float* gate_scale2_dev,
    const unsigned long long* up_packed_ptrs_dev,
    const unsigned long long* up_sfb_ptrs_dev,
    const float* up_scale2_dev,
    void* C_gate_bf16,
    void* C_up_bf16,
    const int* expert_offsets_dev,
    int num_experts,
    int m_total,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  // k%32: per-row packed-A bytes (k/2) stay 16B-aligned for per-group TMA
  // bases; n%8: per-group C offsets (off*n*2B) stay 16B-aligned.
  if (n <= 0 || k <= 0 || (k % 32) != 0 || (n % 8) != 0 || num_experts <= 0 ||
      num_experts > 1024) {
    return -1;
  }
  if (m_total <= 0) {
    return 0;
  }
  unsigned char* ws = static_cast<unsigned char*>(workspace);
  DevCarve d;
  int rc = carve_dev_(ws, workspace_size, num_experts, m_total, n, k, true, d);
  if (rc) {
    return rc;
  }
  build_group_args_dev<<<1, 256, 0, stream>>>(
      expert_offsets_dev, num_experts, n, k, d.packed_a, d.sfa, d.shapes, d.ptr_a,
      d.ptr_sfa, d.l_sfa, d.ds_a, d.ds_b, d.ds_c, d.ds_d, d.l_sfb, gate_scale2_dev,
      d.alpha0, up_scale2_dev, d.alpha1, (unsigned long long)C_gate_bf16, d.ptr_c0,
      (unsigned long long)C_up_bf16, d.ptr_c1);
  dim3 blk(256);
  dim3 grd(m_total, (k / 16 + blk.x - 1) / blk.x);
  pack_act_dev_offsets<<<grd, blk, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(A_bf16), sorted_token_ids,
      expert_offsets_dev, num_experts, d.packed_a, d.ptr_sfa, k);
  size_t cur = d.cursor;
  rc = launch_projection_dev(num_experts, d.shapes, d.ptr_a, d.ds_a, d.ptr_sfa,
                             d.l_sfa, gate_packed_ptrs_dev, gate_sfb_ptrs_dev,
                             d.ds_b, d.l_sfb, d.alpha0, d.ptr_c0, d.ds_c, d.ds_d,
                             ws, cur, workspace_size, &cur, stream, 400000);
  if (rc) {
    return rc;
  }
  return launch_projection_dev(num_experts, d.shapes, d.ptr_a, d.ds_a, d.ptr_sfa,
                               d.l_sfa, up_packed_ptrs_dev, up_sfb_ptrs_dev,
                               d.ds_b, d.l_sfb, d.alpha1, d.ptr_c1, d.ds_c,
                               d.ds_d, ws, cur, workspace_size, nullptr, stream,
                               500000);
#else
  (void)A_bf16;
  (void)sorted_token_ids;
  (void)gate_packed_ptrs_dev;
  (void)gate_sfb_ptrs_dev;
  (void)gate_scale2_dev;
  (void)up_packed_ptrs_dev;
  (void)up_sfb_ptrs_dev;
  (void)up_scale2_dev;
  (void)C_gate_bf16;
  (void)C_up_bf16;
  (void)expert_offsets_dev;
  (void)num_experts;
  (void)m_total;
  (void)n;
  (void)k;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}

// ════════════════════════════════════════════════════════════════════════════
// PUBLIC ENTRY — DEVICE-OFFSET grouped DOWN (graph-capture-safe twin of
// atlas_cutlass_nvfp4_grouped_down). A = post-SiLU [m_total, K=inter], already
// expert-contiguous (identity gather). All arrays are DEVICE pointers.
// ════════════════════════════════════════════════════════════════════════════
extern "C" int atlas_cutlass_nvfp4_grouped_down_dev(
    const void* A_bf16,
    const unsigned long long* packed_ptrs_dev,
    const unsigned long long* sfb_ptrs_dev,
    const float* scale2_dev,
    void* C_bf16,
    const int* expert_offsets_dev,
    int num_experts,
    int m_total,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (k % 32) != 0 || (n % 8) != 0 || num_experts <= 0 ||
      num_experts > 1024) {
    return -1;
  }
  if (m_total <= 0) {
    return 0;
  }
  unsigned char* ws = static_cast<unsigned char*>(workspace);
  DevCarve d;
  int rc = carve_dev_(ws, workspace_size, num_experts, m_total, n, k, false, d);
  if (rc) {
    return rc;
  }
  build_group_args_dev<<<1, 256, 0, stream>>>(
      expert_offsets_dev, num_experts, n, k, d.packed_a, d.sfa, d.shapes, d.ptr_a,
      d.ptr_sfa, d.l_sfa, d.ds_a, d.ds_b, d.ds_c, d.ds_d, d.l_sfb, scale2_dev,
      d.alpha0, nullptr, nullptr, (unsigned long long)C_bf16, d.ptr_c0, 0ull,
      nullptr);
  dim3 blk(256);
  dim3 grd(m_total, (k / 16 + blk.x - 1) / blk.x);
  pack_act_dev_offsets<<<grd, blk, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(A_bf16), nullptr, expert_offsets_dev,
      num_experts, d.packed_a, d.ptr_sfa, k);
  return launch_projection_dev(num_experts, d.shapes, d.ptr_a, d.ds_a, d.ptr_sfa,
                               d.l_sfa, packed_ptrs_dev, sfb_ptrs_dev, d.ds_b,
                               d.l_sfb, d.alpha0, d.ptr_c0, d.ds_c, d.ds_d, ws,
                               d.cursor, workspace_size, nullptr, stream, 600000);
#else
  (void)A_bf16;
  (void)packed_ptrs_dev;
  (void)sfb_ptrs_dev;
  (void)scale2_dev;
  (void)C_bf16;
  (void)expert_offsets_dev;
  (void)num_experts;
  (void)m_total;
  (void)n;
  (void)k;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}

// ════════════════════════════════════════════════════════════════════════════
// PUBLIC ENTRY — grouped DOWN. A = post-SiLU intermediate [M_total, K=inter],
// ALREADY expert-contiguous (sorted_token_ids=null). B = down_proj [N=hidden,K/2].
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
  unsigned char* ws = static_cast<unsigned char*>(workspace);
  GroupedAPrep a = prep_grouped_a(static_cast<const __nv_bfloat16*>(A_bf16), nullptr,
                                  expert_offsets_host, num_experts, n, k, ws, stream);
  if (a.G == 0) {
    return 0;
  }
  return launch_projection(a, packed_ptrs, sfb_ptrs, scale2_vals,
                           static_cast<__nv_bfloat16*>(C_bf16), n, k, ws, a.cursor,
                           workspace_size, stream, 300000);
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
