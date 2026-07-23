# b12x fused-MoE AOT integration (Laguna-S-2.1) — artifacts + status

Port of the FlashInfer **b12x** fused MoE (SM120/SM121 CuTe-DSL, NVFP4) onto Atlas's
**Laguna-S-2.1** MoE prefill as an OPT-IN accelerator behind `ATLAS_LAGUNA_MOE_B12X=1`.
Replaces the ~6-launch grouped path (sort → gather → grouped gate_up GEMM → SwiGLU →
grouped down GEMM → unpermute_reduce) with ONE resident launch (route/pack + FC1 + SwiGLU
+ FP4-requant + FC2 + scatter).

Source: PR #23 (`feat(moe): opt-in FlashInfer b12x fused-MoE for Holo prefill`), which was
built + validated end-to-end on **Holo-3.1-35B-A3B** (E=256, H=2048, I=512, top_k=8). This
branch retargets the GPU-independent Rust + the AOT staging to Laguna geometry.

## THE GEOMETRY DELTA (Holo → Laguna)

| dim | Holo (PR #23) | **Laguna-S-2.1** |
|---|---|---|
| num_experts (E) | 256 | 256 (same) |
| hidden_size (H) | 2048 | **3072** |
| moe_intermediate_size (I) | 512 | **1024** |
| num_experts_per_tok (top_k) | 8 | **10** |
| shared_expert_intermediate_size | 512 | 1024 |

Laguna config verified at
`/tank/hf/hub/models--poolside--Laguna-S-2.1-NVFP4/.../config.json` (model_type `laguna`).

Derived b12x buffer geometry (per-expert), asserted in `b12x_weights_tests.rs`:
- `w13_fp4 [E,2I,H/2]` stride = 2·1024·1536 = **3 MiB/expert** (768 MiB total)
- `w2_fp4  [E,H,I/2]`  stride = 3072·512   = **1.5 MiB/expert** (384 MiB total)
- `w13_sf` swizzled SFB = sfb_len(2I=2048, H=3072) = **384 KiB/expert** (96 MiB)
- `w2_sf`  swizzled SFB = sfb_len(H=3072, I=1024)  = **192 KiB/expert** (48 MiB)

Default-off: unset `ATLAS_LAGUNA_MOE_B12X` ⇒ byte-identical to today's grouped-CUTLASS path.

## GPU-independent code (LANDED this branch — compiles clean, clippy-clean, 13 unit tests pass)
- `crates/spark-model/src/layers/moe/b12x_weights.rs` — `B12xMoeWeights` struct, pure
  `eligibility()` gate, load-time NVFP4 fp4 repack (D2D concat UP‖GATE → `[E,2I,H/2]`, DOWN
  → `[E,H,I/2]`, dims from `ModelConfig`), alpha vectors. (+ `b12x_weights_tests.rs`.)
- `crates/spark-model/src/layers/moe/b12x_scales.rs` — Stage-(a) host e4m3 codec + w13
  scale2-bake + ones-vecs (unit-tested), and the Stage-(b) `swizzle_sfb` atom SEAM
  (`SfbStrategy::{ConcatReuse,RebuildFromRaw}`, `ATLAS_B12X_SFB_STRATEGY`). (+ tests.)
- `crates/spark-model/src/layers/ops/b12x_flashinfer.rs` — dlopen FFI (clone of
  `gdn_flashinfer.rs`): `available()`, `max_tokens()`, `b12x_moe_prefill()`. (+ tests.)
- `crates/spark-model/src/layers/moe/forward_prefill_b12x.rs` — the airtight
  `try_b12x_prefill` dispatch gate.
- `forward_prefill.rs` — gate at the topk/sort boundary; the grouped block is kept in the
  `else` (byte-unchanged). `mod.rs` (`b12x` field + module decls), `init.rs` (`b12x: None`),
  `ops.rs` (module decl).
- **Load hook:** `crates/spark-model/src/weight_loader/laguna/load_layers.rs::load_moe_ffn`,
  right after `build_cutlass_grouped_sfb` — `if ATLAS_LAGUNA_MOE_B12X=1 { build_b12x_weights }`.
  (PR #23 hooked `qwen35/load_layers.rs`; Laguna has its OWN loader.)

### Deltas vs PR #23 (why the diff isn't a straight cherry-pick)
- **Env flag renamed** `ATLAS_HOLO_MOE_B12X` → **`ATLAS_LAGUNA_MOE_B12X`** (the one gating
  flag; every code + doc site uses it). Shim C symbols stay `atlas_b12x_*`; the
  `ATLAS_B12X_LIB` / `ATLAS_B12X_SFB_STRATEGY` / `ATLAS_B12X_BAKE_W2` knobs keep their names.
- **`ptr_tables.rs` NOT ported** — the Laguna tree already split the four ptr-table builders
  into `moe/ptr_table_build.rs`, so PR #23's `mod.rs`→`ptr_tables.rs` split is a no-op here.
- **No streaming-experts machinery on the Laguna tree** — there is no `MoeLayer.streamer`
  field. `eligibility()` keeps its `has_streamer`/`ErrStreamer` arm (unit-tested truth table,
  forward-compat), but `build_b12x_weights` passes `has_streamer = false` and
  `try_b12x_prefill` drops the belt-and-braces streamer re-check. EP re-check kept.
- **`pack_weight_sfb` gained a `src_n_major: bool` arg** on the Laguna tree. `swizzle_sfb`
  passes `false` (the bake keeps the transposed `[K/16,N]` orientation).
- **`_t` tables gate:** b12x needs `gate/up/down_ptrs_t`. On Laguna those are built by
  `transpose_for_prefill_unified`, i.e. under **`ATLAS_UNIFIED_MOE_LAYOUT=1`** (Holo's
  equivalent was `FAST_MOE_MODE=full`). Without them `have_t=false` → b12x self-disables
  (WARN) and the grouped path runs.

**Frozen scale decisions:** `w1_alpha = ones` + up/gate `scale2` baked into the w13 SFs
(mandatory — kernel reuses w1_alpha as the FC1 input-quant scale, quadratically); `w2_alpha
= down.scale2` lossless default (`ATLAS_B12X_BAKE_W2=1` → bake + ones for vLLM parity);
`fc2_input_scale = 1.0`.

**Non-negotiables (enforced in code):** E=256 (asserted at export); fully-resident experts
ONLY — EP / null-expert / no-lib / no-`_t`-tables configs silently disable b12x (WARN) and
run grouped; atomic-scatter non-determinism ⇒ tolerance-only A/B (cos≥0.999, rel-L2≤2e-3),
never bit-exact.

## Artifacts in this dir
- `b12x_export.py` — AOT export driver, **retargeted to E=256/H=3072/I=1024/top_k=10**
  (asserts the Laguna geometry). `--jit-smoke` for P1; `--static-m` left UNEXECUTED (decode
  follow-up).
- `b12x_shim.cpp` — C-ABI shim; **`B12X_H=3072 B12X_I=1024 B12X_TOPK=10`** updated.
  Marshalling layout (19-ptr/16-memref/5-i32) is geometry-independent (degenerate
  `{void*data}` structs) — unchanged from the Holo freeze.
- `b12x_harness.cpp` — native replay/tolerance compare, `H=3072` updated.
- `proof_sfb_atom.py` — P0 GO/NO-GO SFB atom byte-identity harness, Laguna dims
  (`check(2048,3072)`, `check(3072,1024)`).
- `b12x_moe_aot_export.patch` — ONE-hunk flashinfer stream annotation (geometry-agnostic;
  unchanged).
- `b12x_dyn_0.geom.txt` — Laguna input line + DSL-derived quantities marked `REGEN` (the
  parent's `b12x_export.py` run overwrites this with real `physical_tiles`/`max_tasks`/
  `rows_padded`).
- `b12x_dyn_0.h` — ⚠ **STALE Holo-generated reference** (see the banner in the file). The
  arg SHAPE is geometry-independent so it documents the ABI; the Laguna export regenerates it.
- **`libatlasb12x.so` — NOT shipped.** The PR #23 binary is Holo-geometry (H=2048); loading
  it against Laguna H=3072 pointers is wrong/UB. The parent regenerates it (steps below).

## AOT export status (this container)
**NOT run here.** A GB10 GPU is present, but the CuTe-DSL export toolchain is **not importable
in this environment** (`import flashinfer` / `import cutlass` both fail). A patched flashinfer
venv exists at `/home/ms/.claude/jobs/2be09bdc/tmp/vllm-env/` and the flashinfer source at
`/home/ms/flashinfer` — those are the leads for the parent, but this port deliberately does
NOT block on the export (default-off; the Rust half is the must-have and is DONE).

## PARENT GPU RECIPE (gx10 / GB10, in order)
Same as the Holo runbook (`docs/streaming-experts/B12X-PARENT-RUNBOOK.md`) with the Laguna
geometry now baked into `b12x_export.py` / `b12x_shim.cpp` / `proof_sfb_atom.py`:
1. **Env.** vLLM b12x container OR a venv: torch cu13 aarch64 + `nvidia-cutlass-dsl==4.4.2`
   (+ `-libs-base`/`-libs-cu13` ==4.4.2; 4.5.x emits bad PTX on sm121) + `apache-tvm-ffi` +
   `cuda-python`. CUDA≥13.
2. **cutlass-dsl 4.4.2 sm_121a seds** (warp/mma.py, tcgen05/mma.py, tcgen05/copy.py) + wipe
   `__pycache__`; **4.4.2↔flashinfer namespace shim** (re-export `OperandMajorMode` from
   `nvgpu/__init__.py`). See the Holo runbook §Env for exact seds.
3. **FlashInfer patch.** `git apply b12x_moe_aot_export.patch` on `/home/ms/flashinfer`.
4. **P0 — SFB atom A/B (GO/NO-GO):** `python proof_sfb_atom.py`. PASS ⇒ keep
   `SfbStrategy::ConcatReuse` (default). FAIL ⇒ implement the FI-matching swizzle in
   `b12x_scales.rs::swizzle_sfb`'s `RebuildFromRaw` arm, set `ATLAS_B12X_SFB_STRATEGY=rebuild`.
   (On Holo the functional P0 was folded into the end-to-end A/B and ConcatReuse PASSED; the
   Laguna swizzle uses the SAME `pack_weight_sfb` atom, so ConcatReuse is the expected result.)
5. **P1 — JIT smoke:** `python b12x_export.py --jit-smoke` (E=256, top_k=10, k=3072, n=1024).
6. **P2 — alpha-bake numeric on REAL Laguna weights** (cos≥0.999, rel-L2≤2e-3); census
   `gate_ws2 == up_ws2` across all 256 experts × MoE layers.
7. **P3 — AOT export + header freeze:**
   `CUTE_DSL_ARCH=sm_121a PYTHONPATH=/home/ms/flashinfer python b12x_export.py --out
   /tmp/b12x_aot --name b12x_dyn_0 --max-tokens 1024` → regenerates `b12x_dyn_0.{h,o}` +
   `.geom.txt` at Laguna dims. Re-confirm the shim's 19/16/5 arg mapping against the new `.h`
   (shape should be identical to Holo; only baked dims differ).
8. **Relink `libatlasb12x.so`** (g++ -shared b12x_shim.cpp b12x_dyn_0.o … -lcute_dsl_runtime
   -lcudart). Copy it + `libcute_dsl_runtime.so` beside `libatlasgdn.so`.
9. **Native harness:** `ATLAS_B12X_MAX_TOKENS=1024 ./b12x_harness /tmp/b12x_aot`.
10. **Rebuild Atlas** (`ATLAS_TARGET_MODEL='*'` or `laguna-s-2.1`), confirm
    `cargo test -p spark-model b12x` green + flag-unset byte-identical to grouped.
11. **Correctness A/B — end-to-end real Laguna:** serve with
    `ATLAS_LAGUNA_MOE_B12X=1 ATLAS_UNIFIED_MOE_LAYOUT=1 ATLAS_B12X_LIB=<.so>` (+ cute runtime
    on `LD_LIBRARY_PATH`). Expect the load log `built fused weights for 256 experts …
    strat=ConcatReuse` per MoE layer + the debug gate `N=… routed experts via one resident
    b12x launch` firing (⇒ not a silent fallback). Routed-only cos≥0.999 vs grouped; then the
    **needle regression gate** (must not regress the Laguna baseline).
12. **Perf validate:** prefill tok/s A/B vs the grouped CUTLASS path
    (`ATLAS_HOLO_MOE_GROUPED_CUTLASS=1`) and vs the vLLM b12x control. Framework-first: 6
    launches → 1 resident launch; no number promised until measured.

## Results log (parent appends)
_(empty — export + validation are parent-GPU steps; not run in the port container.)_
