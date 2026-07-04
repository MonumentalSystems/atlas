// flashinfer_paged_prefill.cu
//
// Host-callable C-ABI wrapper around FlashInfer's FA2 PAGED batched-prefill
// attention (BatchPrefillWithPagedKVCacheDispatched), specialized for BF16
// Q/KV/O, head_dim = 256, GQA, causal. Built for GB10 / sm_121f, CUDA 13.
//
// Unlike the ragged wrapper (fresh contiguous K/V), this reads K/V from a
// PAGED pool via a block table, so it serves chunk-1+ single-stream prefill
// where the prefix [0, kv_len) already lives in the paged KV cache. It mirrors
// flashinfer_ragged_prefill.cu; the only structural differences are:
//   - BatchPrefillPagedParams (default_prefill_params.cuh) + a paged_kv_t
//     (page.cuh) describing the pool instead of raw k/v pointers + strides.
//   - PrefillPlan is PAGE-native: its kv_indptr is a PAGE prefix-sum and
//     page_size scales the chunk size (scheduler.cuh:608 `kv_chunk_size *=
//     page_size`). The ragged wrapper passes page_size=1 so token==page there;
//     here page_size=block_size, so we MUST pass a page-count kv_indptr, NOT a
//     token-length one. The kernel derives per-request token kv_len from the
//     paged_kv_t itself (page.cuh get_length = (npages-1)*page_size +
//     last_page_len), so no token-length array is needed anywhere.
//
// Workspace sizing is shared with the ragged wrapper
// (atlas_fi_ragged_prefill_workspace_sizes) — the PrefillPlanInfo layout is
// identical, so the same int/float/pinned budgets apply. The wrapper owns NO
// persistent state; all scratch buffers are passed in.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstddef>

#include <flashinfer/allocator.h>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/layout.cuh>   // QKVLayout
#include <flashinfer/page.cuh>     // paged_kv_t
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>

namespace flashinfer {
// Forward declaration of the dispatched paged-prefill entry point (also defined
// as a template in prefill.cuh above); mirrors csrc/batch_prefill.cu.
template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, bool enable_pdl,
                                                   cudaStream_t stream);
}  // namespace flashinfer

using flashinfer::BatchPrefillPagedParams;
using flashinfer::DefaultAttention;
using flashinfer::GetPtrFromBaseOffset;
using flashinfer::MaskMode;
using flashinfer::PosEncodingMode;
using flashinfer::PrefillPlan;
using flashinfer::PrefillPlanInfo;
using flashinfer::QKVLayout;
using flashinfer::paged_kv_t;

namespace {
constexpr uint32_t kHeadDim = 256;
constexpr PosEncodingMode kPosEnc = PosEncodingMode::kNone;
constexpr bool kUseFp16QkReduction = false;

using StandardAttention = DefaultAttention</*use_custom_mask=*/false,
                                           /*use_sliding_window=*/false,
                                           /*use_logits_soft_cap=*/false,
                                           /*use_alibi=*/false>;

using Params = BatchPrefillPagedParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;

// Populate scheduler-derived params fields from a decoded plan_info and resolve
// split-KV scratch (tmp_v/tmp_s). Byte-identical to the ragged wrapper's — the
// paged params carry the same scheduler-derived fields.
inline void apply_plan_info(Params& params, const PrefillPlanInfo& plan_info,
                            void* int_buffer_ptr, void* float_buffer_ptr,
                            __nv_bfloat16** tmp_v_out, float** tmp_s_out) {
  params.request_indices =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.request_indices_offset);
  params.qo_tile_indices =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.kv_tile_indices =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);

  __nv_bfloat16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (plan_info.split_kv) {
    params.merge_indptr =
        GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.merge_indptr_offset);
    tmp_v = GetPtrFromBaseOffset<__nv_bfloat16>(float_buffer_ptr, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask =
          GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
    }
  }
  params.padded_batch_size = plan_info.padded_batch_size;
  params.max_total_num_rows = plan_info.total_num_rows;
  if (plan_info.enable_cuda_graph) {
    params.total_num_rows =
        GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
  }
  *tmp_v_out = tmp_v;
  *tmp_s_out = tmp_s;
}

template <MaskMode MASK_MODE>
inline cudaError_t run_dispatched(Params& params, __nv_bfloat16* tmp_v, float* tmp_s,
                                  int64_t cta_tile_q, cudaStream_t stream) {
  cudaError_t status = cudaSuccess;
  DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
        CTA_TILE_Q, kHeadDim, kHeadDim, kPosEnc, kUseFp16QkReduction, MASK_MODE,
        StandardAttention, Params>(params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
  });
  return status;
}
}  // namespace

// q/o:                [total_qo_rows, num_qo_heads, 256] BF16 device (contiguous)
// k_pool/v_pool:      layer KV pool bases (paged, NHD: page = [page_size, nkv, 256])
// qo_indptr_h:        [batch+1] query-row prefix sum (HOST, plan)
// kv_page_indptr_h:   [batch+1] PAGE prefix sum (HOST, plan) — PAGE units, not tokens
// qo_indptr_d:        [batch+1] query-row prefix sum (DEVICE, params.q_indptr)
// kv_page_indptr_d:   [batch+1] PAGE prefix sum (DEVICE, paged_kv.indptr)
// kv_page_indices_d:  [nnz_pages] physical block ids (DEVICE, paged_kv.indices == block_table)
// kv_last_page_len_d: [batch]    tokens in each request's last page, in [1, page_size] (DEVICE)
extern "C" int atlas_fi_paged_prefill_bf16_hd256(
    const void* q, void* o, const void* k_pool, const void* v_pool,
    const int32_t* qo_indptr_h, const int32_t* kv_page_indptr_h,
    const int32_t* qo_indptr_d, const int32_t* kv_page_indptr_d,
    const int32_t* kv_page_indices_d, const int32_t* kv_last_page_len_d,
    uint32_t batch, uint32_t total_qo_rows,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t page_size,
    float sm_scale, int causal,
    void* float_ws, size_t float_ws_bytes,
    void* int_ws, size_t int_ws_bytes,
    void* pinned_int_ws, size_t pinned_int_ws_bytes,
    void* stream_raw) {
  if (head_dim != kHeadDim) return -2;
  if (num_kv_heads == 0 || num_qo_heads % num_kv_heads != 0) return -3;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

  // ------------------------------------------------------------------
  // Stage 1: plan. PAGE-native — kv_page_indptr_h is a PAGE prefix-sum and
  // page_size scales the chunk size internally. The plan reads the HOST indptr
  // arrays only.
  // ------------------------------------------------------------------
  PrefillPlanInfo plan_info;
  cudaError_t status = PrefillPlan<int32_t>(
      float_ws, float_ws_bytes,
      int_ws, pinned_int_ws, int_ws_bytes,
      plan_info,
      const_cast<int32_t*>(qo_indptr_h), const_cast<int32_t*>(kv_page_indptr_h),
      /*total_num_rows=*/total_qo_rows,
      /*batch_size=*/batch,
      num_qo_heads, num_kv_heads,
      /*head_dim_qk=*/head_dim, /*head_dim_vo=*/head_dim,
      /*page_size=*/page_size,
      /*enable_cuda_graph=*/false,
      /*sizeof_dtype_o=*/sizeof(__nv_bfloat16),
      /*window_left=*/-1,
      /*fixed_split_size=*/-1,
      /*disable_split_kv=*/false,
      /*num_colocated_ctas=*/0,
      stream);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  // ------------------------------------------------------------------
  // Stage 2: params + paged_kv_t. The ctor computes NHD strides that match
  // Atlas's pool layout exactly (stride_page = nkv*page_size*hd,
  // stride_n = nkv*hd, stride_h = hd — verified vs inferspark_prefill_paged.cu).
  // ctor arg order: (num_heads, page_size, head_dim, batch_size, layout, k, v,
  // indices, indptr, last_page_len).
  // ------------------------------------------------------------------
  Params params;  // zero/default-initialized by the host ctor

  paged_kv_t<__nv_bfloat16, int32_t> paged_kv(
      num_kv_heads, page_size, head_dim, batch, QKVLayout::kNHD,
      static_cast<__nv_bfloat16*>(const_cast<void*>(k_pool)),
      static_cast<__nv_bfloat16*>(const_cast<void*>(v_pool)),
      const_cast<int32_t*>(kv_page_indices_d),
      const_cast<int32_t*>(kv_page_indptr_d),
      const_cast<int32_t*>(kv_last_page_len_d));

  params.paged_kv = paged_kv;
  params.q = static_cast<__nv_bfloat16*>(const_cast<void*>(q));
  params.o = static_cast<__nv_bfloat16*>(o);
  params.lse = nullptr;

  // The kernel reads the DEVICE qo indptr copy.
  params.q_indptr = const_cast<int32_t*>(qo_indptr_d);

  params.num_qo_heads = num_qo_heads;
  params.group_size = flashinfer::uint_fastdiv(num_qo_heads / num_kv_heads);

  params.q_stride_n = num_qo_heads * head_dim;
  params.q_stride_h = head_dim;

  params.window_left = -1;
  params.logits_soft_cap = 0.0f;
  params.sm_scale = sm_scale;
  // rope_* unused (PosEncodingMode::kNone); leave at ctor defaults (0).

  __nv_bfloat16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  apply_plan_info(params, plan_info, int_ws, float_ws, &tmp_v, &tmp_s);

  // ------------------------------------------------------------------
  // Stage 3: dispatch. MaskMode is a compile-time template param selected at
  // runtime from `causal`. Causal here is bottom-right aligned: with qo_len=n
  // and per-request kv_len (from paged_kv.get_length), position i attends to
  // [0, seq_len_start + i] — exactly Atlas chunked-prefill causal.
  // ------------------------------------------------------------------
  if (causal) {
    status = run_dispatched<MaskMode::kCausal>(params, tmp_v, tmp_s, plan_info.cta_tile_q, stream);
  } else {
    status = run_dispatched<MaskMode::kNone>(params, tmp_v, tmp_s, plan_info.cta_tile_q, stream);
  }
  return status == cudaSuccess ? 0 : static_cast<int>(status);
}
