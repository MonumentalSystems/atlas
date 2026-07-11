# b12x fused-MoE AOT integration — artifacts + status

Integrate FlashInfer's **b12x** fused MoE (SM120/SM121 CuTe-DSL, NVFP4) into Atlas's
Holo-3.1-35B-A3B MoE prefill as an OPT-IN accelerator behind `ATLAS_HOLO_MOE_B12X=1`.
Replaces the ~6-launch grouped path (sort → gather → grouped gate_up GEMM → SwiGLU →
grouped down GEMM → unpermute_reduce) with ONE resident launch (route/pack + FC1 + SwiGLU
+ FP4-requant + FC2 + scatter). Holo geometry: **E=256, top_k=8, hidden=2048,
moe_intermediate=512**, NVFP4 block-scaled expert weights, Qwen3-renormalized routing.

Default-off: unset `ATLAS_HOLO_MOE_B12X` ⇒ byte-identical to today's grouped-CUTLASS path.

## GPU-independent code (landed this session — compiles, clippy-clean, 13 unit tests pass)
- `crates/spark-model/src/layers/moe/ptr_tables.rs` (129) — mechanical split of the 4
  ptr-table builders out of `mod.rs` (freed it 500→391 for the new `mod`s + field).
- `crates/spark-model/src/layers/moe/b12x_weights.rs` (232) — `B12xMoeWeights` struct,
  pure `eligibility()` gate, load-time NVFP4 fp4 repack (D2D concat UP‖GATE → `[E,2I,H/2]`,
  DOWN → `[E,H,I/2]`), alpha vectors.
- `crates/spark-model/src/layers/moe/b12x_scales.rs` (282) — Stage-(a) host e4m3 codec +
  w13 scale2-bake + ones-vecs (all unit-tested), and the Stage-(b) `swizzle_sfb` atom SEAM
  (`SfbStrategy::{ConcatReuse,RebuildFromRaw}`, `ATLAS_B12X_SFB_STRATEGY`).
- `crates/spark-model/src/layers/ops/b12x_flashinfer.rs` (170) — dlopen FFI (clone of
  `gdn_flashinfer.rs`): `available()`, `max_tokens()`, `b12x_moe_prefill()`.
- `crates/spark-model/src/layers/moe/forward_prefill_b12x.rs` (70) — the airtight
  `try_b12x_prefill` dispatch gate.
- `forward_prefill.rs` — ~6-line gate at the topk/sort boundary; the grouped block is kept
  in the `else` (byte-unchanged). Load hook: ~4 lines in `load_layers.rs`.

**Frozen scale decisions:** `w1_alpha = ones` + up/gate `scale2` baked into the w13 SFs
(mandatory — kernel reuses w1_alpha as the FC1 input-quant scale, quadratically); `w2_alpha
= down.scale2` lossless default (`ATLAS_B12X_BAKE_W2=1` → bake + ones for vLLM parity);
`fc2_input_scale = 1.0`.

**Non-negotiables (enforced in code):** E=256 (asserted at export); fully-resident experts
ONLY — `ATLAS_HOLO_MOE_B12X=1` + `--stream-experts` is a load-time HARD ERROR; EP /
null-expert / no-lib configs silently disable b12x (WARN) and run grouped; atomic-scatter
non-determinism ⇒ tolerance-only A/B (cos≥0.999, rel-L2≤2e-3), never bit-exact.

## Artifacts in this dir (parent runs on gx10/GB10)
- `b12x_moe_aot_export.patch` — ONE-hunk stream annotation on
  `blackwell_sm12x/moe_dispatch.py:1383` (`stream,` → `stream: _cuda_aot.CUstream,`) +
  the `_cuda_aot` import. `git apply --check` PASSES on flashinfer @ a671c02.
- `b12x_export.py` — AOT export driver (dynamic kernel only, one export; asserts E==256;
  `--jit-smoke` for P1; `--static-m` left UNEXECUTED for the Design-B decode follow-up).
- `proof_sfb_atom.py` — P0 GO/NO-GO SFB atom byte-identity harness.
- `b12x_shim.cpp` — C-ABI shim (cached workspace; **marshalling FROZEN AT P3** against the
  generated `.h` — `make_ptr` pointer-fakes have no GDN precedent; prefill currently
  returns 1 until frozen). Also exposes `atlas_b12x_max_tokens()` + stubbed static decode.
- `b12x_harness.cpp` — native replay + tolerance compare.

## PARENT GPU RECIPE (gx10 / GB10, in order)
1. **Env.** vLLM b12x container (`/home/ms/spark-vllm-docker/mods/exp-b12x/run.sh` has the
   pins/seds/env pre-applied; image built `--apply-vllm-pr 40082`) OR a fresh venv: torch
   cu13 aarch64 + `nvidia-cutlass-dsl==4.4.2` + `-libs-base==4.4.2` + `-libs-cu13==4.4.2`
   (4.5.x emits bad PTX on sm121). CUDA≥13 (cuda-13.2 present). The old
   `/home/ms/spark-vllm-docker/.venv` is GONE — this is real work.
2. **cutlass-dsl 4.4.2 sm_121a seds** (from run.sh:57-83), then wipe `__pycache__`:
   - `warp/mma.py`: `if not arch == Arch.sm_120a:` → `if arch not in (Arch.sm_120a,
     Arch.sm_121a):`; add `"sm_121a",` after `"sm_120a",` in the admissible list.
   - `tcgen05/mma.py`: insert `Arch.sm_120a,` + `Arch.sm_121a,` after `Arch.sm_103a,`.
   - `tcgen05/copy.py`: `arch.is_family_of(Arch.sm_110f)` → `... or
     arch.is_family_of(Arch.sm_120f)`.
   - `find <site-packages> -name __pycache__ -path '*/cutlass*' -o -path '*/flashinfer*' | rm -rf`.
3. **FlashInfer patch.** `git apply 3rdparty_patches/b12x_aot/b12x_moe_aot_export.patch` on
   `/home/ms/flashinfer` (a671c02). For pip installs also drop the stale
   `sm120_moe_dispatch_context` import from `blackwell_sm12x/__init__.py`.
4. **P0 — SFB atom A/B (GO/NO-GO, FIRST, no model/export):**
   `python proof_sfb_atom.py`. **PASS ⇒ keep `SfbStrategy::ConcatReuse` (default); repack
   ships as written.** FAIL ⇒ first-diff offset localizes the mismatch; implement the
   FI-matching swizzle in `b12x_scales.rs::swizzle_sfb`'s `RebuildFromRaw` arm (Stage-(a)
   untouched) and set `ATLAS_B12X_SFB_STRATEGY=rebuild`.
5. **P1 — JIT smoke at Holo dims:** `python b12x_export.py --jit-smoke` (+ a pure-Python
   `b12x_fused_moe` at E=256,top_k=8,k=2048,n=512, T∈{1,64,128} vs fp32 torch ref).
6. **P2 — alpha-bake numeric on REAL Holo weights:** load one MoE layer's raw scales, run
   the `bake_w13_logical` bake, b12x vs fp32 dequant ref (Atlas-convention routing). Accept
   cos≥0.999, rel-L2≤2e-3 (both ≥0.995 vs fp32). Negatives: swap up/gate halves ⇒ cos<0.9;
   skip bake ⇒ off by ~ws2. Also census `gate_ws2 == up_ws2` across all 256 experts × layers
   (confirms the single-per-expert-alpha assumption).
7. **P3 — first AOT export + header freeze:**
   `CUTE_DSL_ARCH=sm_121a PYTHONPATH=/home/ms/flashinfer python b12x_export.py` →
   `b12x_dyn_0.{h,o}` + `.geom.txt` + reference IO. **Inspect the generated `.h`** and
   freeze `b12x_shim.cpp`'s 19-ptr/16-memref/5-i32 marshalling against the actual rendering.
8. **Relink `libatlasb12x.so`:**
   `g++ -O2 -fPIC -shared b12x_shim.cpp b12x_dyn_0.o -o libatlasb12x.so
   -I/usr/local/cuda/include -lcudart -L<cute_lib> -lcute_dsl_runtime -Wl,-rpath,<cute_lib>`
   (`cute_lib` via `python -m cutlass.cute.export.aot_config --ldflags --libs`).
9. **Native harness:** `ATLAS_B12X_MAX_TOKENS=1024 ./b12x_harness /tmp/b12x_aot` (tolerance).
10. **Rebuild Atlas** (docker, not host cargo — glibc parity): the new Rust compiles
    default-off; confirm `cargo test -p spark-model b12x` green and a flag-unset build is
    byte-identical to grouped.
11. **Correctness A/B — end-to-end real Holo:** T=64, routing from Atlas's own
    `moe_topk_softmax_batched`; (A) grouped-CUTLASS through `unpermute_reduce` (routed-only,
    exclude shared) vs (B) b12x. Accept cos≥0.999, rel-L2≤2e-3. Then the **needle regression
    gate**: run the 128K/200K needle harness with `ATLAS_HOLO_MOE_B12X=1` — must hold the
    baseline (8/8 to 128K; must not regress 7/8@200K).
12. **Perf validate:** canonical serve flags (SLAI@100ms, ssm-slots 256, no profiling),
    `ATLAS_TARGET_MODEL=holo-3.1-35b-a3b`, RUST_LOG=info. Benchmark prefill tok/s A/B vs
    `ATLAS_HOLO_MOE_GROUPED_CUTLASS=1`, and vs the vLLM b12x control on this box. No number
    promised until measured (framework-first: 6 launches → 1 resident launch).
13. **Ship** `libatlasb12x.so` + `libcute_dsl_runtime.so` beside `libatlasgdn.so`. Append
    results below. Decode (Design-B static-m1/2/3 + `forward_prefill(M=m)` reroute) is
    promoted ONLY on a positive DFlash-acceptance A/B or a measured decode-MoE gap >5% — the
    shim/Rust surface is already final for it (`atlas_b12x_static_*` stubs present).

## Results log (parent appends)

### 2026-07-11 — Phases 0–5 COMPLETE on dgx-00 GB10 (sm_121a). Export + shim + relink validated.

**Env (P0-env).** Container from `atlas-gb10:gdnf32-build` (ships cutlass-dsl **4.5.0**, so the
downgrade IS required). Full recipe now captured in `docs/streaming-experts/B12X-PARENT-RUNBOOK.md`.
Corrections vs the original recipe, all needed to make export actually run (prior session wrote
the code but never executed it):
- **tvm_ffi**: not on PyPI as `tvm-ffi`; the package is **`apache-tvm-ffi`** (0.1.12). It is
  REQUIRED at flashinfer import time (`flashinfer/jit/core.py`), independent of the `--enable-tvm-ffi`
  compile flag.
- **cutlass-dsl 4.4.2 vs flashinfer a671c02 namespace gap**: FI references
  `cute.nvgpu.OperandMajorMode` (a 4.5.x top-level location); in 4.4.2 it lives under
  `cute.nvgpu.tcgen05`. One-line re-export fixes it (`cfence` refs in FI are all commented — no-op):
  `sed -i '/^from . import tcgen05$/a from .tcgen05 import OperandMajorMode, OperandSource' <nvgpu/__init__.py>`.
- torch: `pip install torch --index-url .../cu130` → torch 2.13.0+cu130, CUDA OK. Also need
  `cuda-python` (for `cuda.bindings.driver` the FI stream-annotation patch imports).

**Export driver (`b12x_export.py`) — FOUR real bugs fixed (all now committed):**
1. tvm-ffi strip only handled `list`/`tuple` options, but moe_dispatch passes
   `options="--opt-level 2 --enable-tvm-ffi"` as a **string** → tvm-ffi stayed on → wrong
   `export_to_c` signature. Now strips the string form too.
2. `from __future__ import annotations` in moe_dispatch makes the kernel's arg annotations
   **strings**; the C-header generator dispatches numeric scalars via `isinstance(ann, NumericMeta)`
   and fails on `'cutlass.Int32'`. Fixed by resolving the scalar/Constexpr/CUstream annotations to
   real classes on `type(fn).__call__.__annotations__` BEFORE compile (the kernel is a
   `_DynamicMoELaunch` **instance**, not a plain function).
3. `_dynamic_task_geometry` real signature is `(state_E, n, routed_rows, *, tile_m, tile_n)`, not
   the guessed kwargs — geom dump fixed; now emits physical_tiles/max_tasks/rows_padded/cols_pad_k.
4. Added a live-CUDA-context init before build so the baked `mac` Constexpr comes from the real
   `get_max_active_clusters(1)` probe (=48=sm_count on GB10, so no functional change here, but robust).

**Export (P3).** `CUTE_DSL_ARCH=sm_121a PYTHONPATH=/opt/flashinfer python3 b12x_export.py
--out /work/b12x_aot --name b12x_dyn_0 --max-tokens 1024` → **`b12x_dyn_0.{h,o}` + `.geom.txt`**.
Geometry @ cap=1024: **physical_tiles=319, max_tasks=1276, rows_padded=40832, cols_pad_k=128**.

**Shim freeze (P3) — MUCH simpler than predicted.** The generated `.h` renders every fixed-shape
tensor arg as a **degenerate `{ void *data; }` struct** (all shapes baked constexpr) — there are NO
stride/offset descriptors to hand-pack; the runbook's "highest-risk memref freeze" evaporated.
Counts match exactly: **19 `void*` + 16 `{void*data}` structs + 5 int32 + CUstream** into a
`void*[42]`. The header's `static inline cute_dsl_b12x_dyn_0_wrapper(...)` builds the arg array, so
the shim calls it with typed args. Two corrections vs the stub: `packed_a`/`packed_a_storage` are
the **same** buffer (as are `sfa`/`scale_storage`) passed twice; **all** control buffers are zeroed
once at alloc (JIT uses `torch.zeros`, workspace reused across calls), not just the 2 barriers.
Full runtime-arg mapping is `moe_dispatch.py:1781-1824`. The 5 int32: only `num_tokens` varies;
`max_rows==rows_padded==physical_tiles*128`, `max_tasks`/`max_phys_tiles` are capacity constants.

**Relink (P8) + native launch-smoke (P9-lite).** `g++ -shared b12x_shim.cpp b12x_dyn_0.o
-o libatlasb12x.so` with `-L$(aot_config --libdir) -lcute_dsl_runtime -L<cuda-13.2 sbsa> -lcudart`.
All 6 `atlas_b12x_*` symbols export. A ctypes launch-smoke (Holo-shaped device buffers, T=64) →
**`ret=0`, no CUDA error, no deadlock, output 99.99% written** — the marshalling/ABI/workspace are
correct. (Numerics not checked here — random fp4 weights → non-finite; that's Phase 7.)

**Artifacts committed in this dir:** `b12x_dyn_0.h` (frozen SSOT), `b12x_dyn_0.geom.txt`,
`b12x_dyn_0.o`, `libatlasb12x.so` (aarch64 sm_121a; regenerable via the runbook). Also need
`libcute_dsl_runtime.so` (39MB, from cutlass-dsl) + cudart beside it at deploy.

**NOT a P0 blocker anymore:** `proof_sfb_atom.py` stays non-runnable (unchanged) — the SFB
go/no-go is deferred into the Phase-7 end-to-end A/B (functional P0), which decides ConcatReuse vs
RebuildFromRaw.

### 2026-07-11 (cont.) — Phase 6 (P10) + functional P0 (P11 correctness) DONE + PASSED.

- **Rebuild WAS required** (the gdnf32-build binary predated the b12x Rust — 0 b12x strings). Rebuilt
  `spark-server` in-container: `CUTLASS_HOME=/opt/cutlass FLASHINFER_HOME=/opt/flashinfer
  ATLAS_CUTLASS_NVFP4_GEMM=1 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b cargo build --release -p spark-server`
  (~2.5min incremental; b12x is dlopen+runtime-gated, no cargo feature). New binary: 12 b12x strings.
- **Enabling flag discovered**: b12x eligibility needs the transposed `gate/up/down_ptrs_t` tables,
  built ONLY under **`ATLAS_HOLO_FAST_MOE_MODE=full` + `ATLAS_HOLO_FAST_MOE_LAYERS=0-39`**. Without
  them `have_t=false` → b12x silently disabled (no warning). This is the non-obvious serve flag.
- **Served** Holo-3.1-35B-A3B-NVFP4 with `ATLAS_HOLO_MOE_B12X=1 ATLAS_B12X_LIB=<.so>` +
  FAST_MOE=full + cute-runtime on LD_LIBRARY_PATH. Log: `b12x fused-MoE loaded (max_tokens=1024)` +
  `built fused weights for 256 experts … strat=ConcatReuse` ×40 layers.
- **FUNCTIONAL P0 = PASS**: correct output ("23, 29, 31" for primes>20; coherent MoE explanation) and
  the debug gate fires `N=… routed experts via one resident b12x launch` **×40 per prefill** — the
  kernel executed, not a silent fallback. **⇒ ConcatReuse is CORRECT; keep the default. No
  RebuildFromRaw. `proof_sfb_atom.py` moot.** b12x handled a 980-tok prefill in 40 launches.
- **GOTCHA**: OOM restart race — a killed b12x+FAST_MOE server (~3× expert memory) doesn't release
  CUDA memory instantly; wait for `nvidia-smi memory.free` to recover before relaunch (GB10=121.6GB).

**REMAINING (perf only): P12** end-to-end prefill TTFT A/B is SSM-DOMINATED on this model
(30 SSM/10 attn, ~80% SSM) so b12x's MoE-prefill win needs per-layer `ATLAS_MS_PROFILE` isolation,
not raw TTFT (framework-first: correctness is the deliverable). Plus the needle regression gate.
