// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas <-> FlashInfer b12x fused-MoE bridge: wraps the AOT-exported dynamic MoE kernel
// (b12x_dyn_0.{h,o}) into the extern "C" symbols crates/spark-model/src/layers/ops/
// b12x_flashinfer.rs dlopens. Mirrors 3rdparty_patches/gdn_aot/gdn_shim.cpp: the shim
// OWNS a cached one-time workspace (packed-A, SFA, per-task arrays, barriers, control
// counters) sized for the baked `ATLAS_B12X_MAX_TOKENS` capacity — no per-call alloc/free.
//
// ============================ FROZEN AT P3 ============================
// Marshalling frozen against the ACTUAL generated b12x_dyn_0.h wrapper
// `cute_dsl_b12x_dyn_0_wrapper(...)` and the JIT `runtime_args` tuple at
// flashinfer moe_dispatch.py:1781-1824. The dynamic export renders every fixed-shape
// tensor arg as a DEGENERATE `{ void *data; }` struct (all shapes baked as constexpr),
// so there are NO stride/offset descriptors to pack — each such arg is one data pointer.
//   19 raw void* (passed by value)  +  16 `{void*data}` tensor structs  +
//   5 runtime int32 (num_tokens, max_rows, rows_padded, max_tasks, max_phys_tiles)  +
//   cudaStream_t. The header's static-inline wrapper builds the void*[42] arg array
//   internally, so we call it with typed args rather than hand-packing.
// =====================================================================
#include "b12x_dyn_0.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

// ── Baked geometry (Laguna-S-2.1). E=256 asserted at export. ──
// MUST match b12x_export.py's E/H/I/TOPK and the regenerated b12x_dyn_0.{h,geom.txt}.
static const int B12X_E = 256;    // state_E / num_experts
static const int B12X_H = 3072;   // hidden (k)
static const int B12X_I = 1024;   // moe_intermediate (n)
static const int B12X_TOPK = 10;
static const int B12X_TILE_M = 128;        // _LEVEL_TILE_M
static const int B12X_TILE_N = 128;        // _LEVEL_TILE_N
static const int B12X_NVFP4_BLOCK = 16;    // _NVFP4_BLOCK_SIZE
static const int B12X_SLICE_CHUNK = 1;     // _DYNAMIC_SLICE_CHUNK

static int b12x_capacity() {
  const char *e = getenv("ATLAS_B12X_MAX_TOKENS");
  int c = e ? atoi(e) : 1024; // must match the capacity b12x_export.py dumped geometry for
  return c > 0 ? c : 1024;
}

static inline int b12x_align_up(int v, int a) { return ((v + a - 1) / a) * a; }
static inline int b12x_imax(int a, int b) { return a > b ? a : b; }
static inline int b12x_imin(int a, int b) { return a < b ? a : b; }

// Mirrors moe_dispatch._dynamic_task_geometry + allocate_sm120_dynamic_workspace exactly.
struct Geom {
  int routed_rows, physical_tiles, max_tasks, rows_padded, cols_pad_k;
};
static Geom b12x_geom(int cap) {
  Geom g;
  g.routed_rows = b12x_imax(1, cap * B12X_TOPK);
  int base_m_tiles = b12x_align_up(g.routed_rows, B12X_TILE_M) / B12X_TILE_M;
  int active_ub = b12x_imin(B12X_E, g.routed_rows);
  g.physical_tiles = b12x_imax(1, base_m_tiles + active_ub - 1);
  int gate_tile_cnt = b12x_imax(1, (B12X_I + B12X_TILE_N - 1) / B12X_TILE_N);
  int slice_groups =
      b12x_imax(1, (gate_tile_cnt + B12X_SLICE_CHUNK - 1) / B12X_SLICE_CHUNK);
  g.max_tasks = g.physical_tiles * slice_groups;
  g.rows_padded = g.physical_tiles * B12X_TILE_M;
  g.cols_pad_k = b12x_align_up(B12X_H / B12X_NVFP4_BLOCK, 4);
  return g;
}

static b12x_dyn_0_Kernel_Module_t g_module;
static int g_loaded = 0;

// Cached workspace. Sized once for `b12x_capacity()` tokens. Pointers map 1:1 to the
// workspace tensors in allocate_sm120_dynamic_workspace; only packed_input +
// packed_input_scale are uninitialized (torch.empty) — everything else is zeroed once
// at alloc to match the torch.zeros() JIT allocation (the workspace is reused across
// calls, so the kernel self-manages control state between launches).
struct Ws {
  // Activation packing: packed_a_ptr AND packed_a_storage_ptr are the SAME buffer
  // (packed_a_view / packed_a_flat are views of packed_input); likewise sfa_ptr AND
  // scale_storage_ptr are the SAME buffer (packed_input_scale).
  void *packed_input = nullptr;        // [1, rows_padded, H/2] u8
  void *packed_input_scale = nullptr;  // [rows_padded, cols_pad_k] u8
  // 7 control singletons [1] i32
  void *barrier_count = nullptr, *barrier_epoch = nullptr, *pair_head = nullptr;
  void *producers_done_count = nullptr, *all_work_published = nullptr;
  void *task_head = nullptr, *task_tail = nullptr;
  // 6 task-queue arrays [max_tasks] i32
  void *task_ready = nullptr, *task_expert = nullptr, *task_m_tile = nullptr;
  void *task_slice_begin = nullptr, *task_slice_count = nullptr, *task_valid_rows = nullptr;
  void *tile_write_count = nullptr;    // [physical_tiles] i32
  // per-expert + scatter bookkeeping
  void *row_counts = nullptr;          // [E] i32
  void *expert_write_rows = nullptr;   // [E] i32
  void *expert_tile_base = nullptr;    // [E+1] i32
  void *token_map = nullptr;           // [rows_padded] i32
  void *token_weights = nullptr;       // [rows_padded] f32
  int cap = 0;
  Geom geom{};
};
static Ws g_ws;

static void *b12x_zalloc(size_t bytes) {
  void *p = nullptr;
  cudaMalloc(&p, bytes);
  cudaMemset(p, 0, bytes); // one-time zero (matches torch.zeros)
  return p;
}

static void ensure_ws() {
  int cap = b12x_capacity();
  if (g_ws.cap >= cap && g_ws.packed_input)
    return;
  Geom g = b12x_geom(cap);
  const size_t rp = (size_t)g.rows_padded;
  const size_t mt = (size_t)g.max_tasks;
  const size_t pt = (size_t)g.physical_tiles;
  const size_t i32 = sizeof(int32_t);
  // Uninitialized activation buffers (torch.empty).
  cudaMalloc(&g_ws.packed_input, rp * (B12X_H / 2));
  cudaMalloc(&g_ws.packed_input_scale, rp * (size_t)g.cols_pad_k);
  // Zeroed control singletons.
  g_ws.barrier_count = b12x_zalloc(i32);
  g_ws.barrier_epoch = b12x_zalloc(i32);
  g_ws.pair_head = b12x_zalloc(i32);
  g_ws.producers_done_count = b12x_zalloc(i32);
  g_ws.all_work_published = b12x_zalloc(i32);
  g_ws.task_head = b12x_zalloc(i32);
  g_ws.task_tail = b12x_zalloc(i32);
  // Zeroed task-queue arrays.
  g_ws.task_ready = b12x_zalloc(mt * i32);
  g_ws.task_expert = b12x_zalloc(mt * i32);
  g_ws.task_m_tile = b12x_zalloc(mt * i32);
  g_ws.task_slice_begin = b12x_zalloc(mt * i32);
  g_ws.task_slice_count = b12x_zalloc(mt * i32);
  g_ws.task_valid_rows = b12x_zalloc(mt * i32);
  g_ws.tile_write_count = b12x_zalloc(pt * i32);
  // Zeroed per-expert + scatter bookkeeping.
  g_ws.row_counts = b12x_zalloc((size_t)B12X_E * i32);
  g_ws.expert_write_rows = b12x_zalloc((size_t)B12X_E * i32);
  g_ws.expert_tile_base = b12x_zalloc((size_t)(B12X_E + 1) * i32);
  g_ws.token_map = b12x_zalloc(rp * i32);
  g_ws.token_weights = b12x_zalloc(rp * sizeof(float));
  g_ws.cap = cap;
  g_ws.geom = g;
}

extern "C" void atlas_b12x_load() {
  if (!g_loaded) {
    b12x_dyn_0_Kernel_Module_Load(&g_module);
    g_loaded = 1;
  }
  ensure_ws();
}

// Query for the Rust surface: the token capacity the workspace is sized for.
extern "C" int atlas_b12x_max_tokens() { return b12x_capacity(); }

// THE prefill entry the Rust FFI calls. Signature matches ops/b12x_flashinfer.rs
// PrefillFn (11 Atlas device pointers + num_tokens + stream). All fixed-shape tensor
// args are wrapped in `{void*data}` structs; the ptr-typed args pass through directly.
extern "C" int atlas_b12x_moe_prefill(
    void *x_bf16, void *topk_ids_i32, void *topk_w_f32, void *out_bf16,
    void *w13_fp4, void *w13_sf, void *w2_fp4, void *w2_sf,
    void *w1_alpha, void *w2_alpha, void *fc2_gs,
    int num_tokens, void *stream) {
  ensure_ws();
  if (num_tokens > g_ws.cap)
    return 2; // over capacity -> Rust falls back to grouped

  // 16 fixed-shape tensor args as `{void*data}` structs (shapes baked at export).
  b12x_dyn_0_Tensor_barrier_count_t barrier_count{g_ws.barrier_count};
  b12x_dyn_0_Tensor_barrier_epoch_t barrier_epoch{g_ws.barrier_epoch};
  b12x_dyn_0_Tensor_pair_head_t pair_head{g_ws.pair_head};
  b12x_dyn_0_Tensor_producers_done_count_t producers_done_count{g_ws.producers_done_count};
  b12x_dyn_0_Tensor_all_work_published_t all_work_published{g_ws.all_work_published};
  b12x_dyn_0_Tensor_task_head_t task_head{g_ws.task_head};
  b12x_dyn_0_Tensor_task_tail_t task_tail{g_ws.task_tail};
  b12x_dyn_0_Tensor_b_w13_t b_w13{w13_fp4};
  b12x_dyn_0_Tensor_b_down_t b_down{w2_fp4};
  b12x_dyn_0_Tensor_row_counts_t row_counts{g_ws.row_counts};
  b12x_dyn_0_Tensor_expert_write_rows_t expert_write_rows{g_ws.expert_write_rows};
  b12x_dyn_0_Tensor_expert_tile_base_t expert_tile_base{g_ws.expert_tile_base};
  // input_global_scale AND alpha both bind to w1_alpha (the kernel reuses w1_alpha as
  // the FC1 input-quant scale); down_alpha=w2_alpha; global_scale=fc2_gs.
  b12x_dyn_0_Tensor_input_global_scale_t input_global_scale{w1_alpha};
  b12x_dyn_0_Tensor_alpha_t alpha{w1_alpha};
  b12x_dyn_0_Tensor_down_alpha_t down_alpha{w2_alpha};
  b12x_dyn_0_Tensor_global_scale_t global_scale{fc2_gs};

  const int rows_padded = g_ws.geom.rows_padded; // == workspace.max_rows
  const int max_tasks = g_ws.geom.max_tasks;
  const int max_phys_tiles = g_ws.geom.physical_tiles;

  return cute_dsl_b12x_dyn_0_wrapper(
      &g_module,
      // 3 Atlas inputs + 4 activation-workspace ptrs (packed_a/sfa passed twice)
      x_bf16, topk_ids_i32, topk_w_f32,
      g_ws.packed_input, g_ws.packed_input_scale, g_ws.packed_input, g_ws.packed_input_scale,
      // 7 control singletons
      &barrier_count, &barrier_epoch, &pair_head, &producers_done_count,
      &all_work_published, &task_head, &task_tail,
      // 6 task arrays + tile_write_count
      g_ws.task_ready, g_ws.task_expert, g_ws.task_m_tile, g_ws.task_slice_begin,
      g_ws.task_slice_count, g_ws.task_valid_rows, g_ws.tile_write_count,
      // weights (fp4 + swizzled SF)
      &b_w13, w13_sf, &b_down, w2_sf,
      // per-expert bookkeeping
      &row_counts, &expert_write_rows, &expert_tile_base,
      // scales
      &input_global_scale, &alpha, &down_alpha, &global_scale,
      // output + scatter maps
      out_bf16, g_ws.token_map, g_ws.token_weights,
      // 5 runtime int32 (only num_tokens varies; rest are capacity constants)
      num_tokens, rows_padded, rows_padded, max_tasks, max_phys_tiles,
      (cudaStream_t)stream);
}

// ── Decode follow-up surface (Design B). Stubbed so the Rust ABI is final NOW: they
//    report "export absent" until the static-m1/2/3 export lands. ──
extern "C" int atlas_b12x_static_supported() { return 0; }
extern "C" int atlas_b12x_static_warmup(int) { return 1; }
extern "C" int atlas_b12x_moe_static(
    void *, void *, void *, void *, void *, void *, void *, void *,
    void *, void *, void *, int, void *) {
  return 1; // static export absent
}
