// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas <-> FlashInfer b12x fused-MoE bridge: wraps the AOT-exported dynamic MoE kernel
// (b12x_dyn_0.{h,o}) into the extern "C" symbols crates/spark-model/src/layers/ops/
// b12x_flashinfer.rs dlopens. Mirrors 3rdparty_patches/gdn_aot/gdn_shim.cpp: the shim
// OWNS a cached one-time workspace (packed-A, SFA, per-task arrays, barriers) sized for
// the baked `ATLAS_B12X_MAX_TOKENS` capacity — no per-call alloc/free/sync.
//
// ============================ FREEZE AT P3 ============================
// The dynamic export renders `make_ptr` POINTER-FAKE args (no GDN precedent — see
// moe_dispatch.py:1569-1671). The 19-ptr / 16-memref / 5-i32 marshalling below is the
// PREDICTED layout. Parent MUST regenerate b12x_dyn_0.h at P3 and freeze the exact
// wrapper prototype + descriptor packing against the ACTUAL rendering before relinking.
// Nothing here is trusted until that inspection.
// =====================================================================
#include "b12x_dyn_0.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// ── Baked geometry (Holo-3.1-35B-A3B). E=256 asserted at export. ──
static const int B12X_E = 256;
static const int B12X_H = 2048;
static const int B12X_I = 512;
static const int B12X_TOPK = 8;

static int b12x_capacity() {
  const char *e = getenv("ATLAS_B12X_MAX_TOKENS");
  int c = e ? atoi(e) : 1024; // must match the capacity b12x_export.py dumped geometry for
  return c > 0 ? c : 1024;
}

static gdn_like_module_t g_module; // renamed by the generated header; alias at P3
static int g_loaded = 0;

// Cached workspace. Sized once for `b12x_capacity()` tokens (max_rows = cap*TOPK).
struct Ws {
  void *packed_a = nullptr;   // [max_rows, H/2] fp4 storage
  void *sfa = nullptr;        // [max_rows, H/16] swizzled input scales
  void *packed_a_storage = nullptr;
  void *scale_storage = nullptr;
  void *task_arrays = nullptr; // ready/expert/m_tile/slice_begin/slice_count/valid_rows
  void *tile_write_count = nullptr;
  void *barriers = nullptr;    // barrier_count, barrier_epoch, pair_head, ... task_head/tail
  void *row_counts = nullptr;  // row_counts, expert_write_rows, expert_tile_base
  void *token_map = nullptr;   // scatter token_map + token_weights
  int cap = 0;
  size_t barrier_bytes = 0;
};
static Ws g_ws;

static void ensure_ws() {
  int cap = b12x_capacity();
  if (g_ws.cap >= cap && g_ws.packed_a)
    return;
  int max_rows = cap * B12X_TOPK;
  // FREEZE AT P3: exact per-buffer sizes come from b12x_dyn_0.geom.txt.
  cudaMalloc(&g_ws.packed_a, (size_t)max_rows * (B12X_H / 2));
  cudaMalloc(&g_ws.sfa, (size_t)max_rows * (B12X_H / 16));
  cudaMalloc(&g_ws.packed_a_storage, (size_t)max_rows * (B12X_H / 2));
  cudaMalloc(&g_ws.scale_storage, (size_t)max_rows * (B12X_H / 16));
  cudaMalloc(&g_ws.task_arrays, (size_t)(B12X_E * 64) * 6 * sizeof(int));
  cudaMalloc(&g_ws.tile_write_count, (size_t)B12X_E * sizeof(int));
  g_ws.barrier_bytes = (size_t)4096 * sizeof(int);
  cudaMalloc(&g_ws.barriers, g_ws.barrier_bytes);
  cudaMalloc(&g_ws.row_counts, (size_t)B12X_E * 3 * sizeof(int));
  cudaMalloc(&g_ws.token_map, (size_t)max_rows * 2 * sizeof(int));
  // Zero ONLY the barrier region once at alloc — the kernel's Phase-0 self-clears
  // everything else (incl. the scatter output), so no per-call zeroing.
  cudaMemset(g_ws.barriers, 0, g_ws.barrier_bytes);
  g_ws.cap = cap;
}

extern "C" void atlas_b12x_load() {
  if (!g_loaded) {
    b12x_dyn_0_Kernel_Module_Load(&g_module); // exact name frozen at P3
    g_loaded = 1;
  }
  ensure_ws();
}

// Query for the Rust surface: the token capacity the workspace is sized for.
extern "C" int atlas_b12x_max_tokens() { return b12x_capacity(); }

// THE prefill entry the Rust FFI calls. Signature matches ops/b12x_flashinfer.rs
// PrefillFn (13 real args + num_tokens + stream). All device pointers are Atlas's.
extern "C" int atlas_b12x_moe_prefill(
    void *x_bf16, void *topk_ids_i32, void *topk_w_f32, void *out_bf16,
    void *w13_fp4, void *w13_sf, void *w2_fp4, void *w2_sf,
    void *w1_alpha, void *w2_alpha, void *fc2_gs,
    int num_tokens, void *stream) {
  cudaStream_t st = (cudaStream_t)stream;
  ensure_ws();
  if (num_tokens > g_ws.cap)
    return 2; // over capacity -> Rust falls back to grouped
  const int max_rows = num_tokens * B12X_TOPK;
  // FREEZE AT P3: build the 16 memref descriptors + 19 pointer fakes from the generated
  // b12x_dyn_0.h, wiring the cached workspace above + these Atlas pointers, then call the
  // exported wrapper with the 5 runtime Int32s (num_tokens, max_rows, rows_padded,
  // max_tasks, max_phys_tiles) and the CUstream. Placeholder returns 1 until frozen.
  (void)x_bf16; (void)topk_ids_i32; (void)topk_w_f32; (void)out_bf16;
  (void)w13_fp4; (void)w13_sf; (void)w2_fp4; (void)w2_sf;
  (void)w1_alpha; (void)w2_alpha; (void)fc2_gs; (void)max_rows; (void)st;
  return 1; // NOT YET FROZEN — parent wires the wrapper call at P3, then returns its ret.
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
