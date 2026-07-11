<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Handoff 2026-07-11 — b12x MoE + GDN long-context

Session context cleared here. This is the state + the exact next steps. All PRs are **draft on
`MonumentalSystems/atlas` (origin)** — NEVER push to `avarok`; subagents never push.

## PR stack (all draft)

Tiered-cache base stack (unchanged this session, pre-existing):
`#9` ← `#10` ← `#11` ← `#12` ← `#13` ← `#15` ← `#16` ← `#17` ← `#18`(lora) ← `#19`(kv-paging UAF + decode-orphan doc)

Shipped/updated this session:
- **#20** `docs/host-build-nccl` → **main**. Standalone: `libnccl-dev` in the docker builder + single-GPU host-build docs (`--no-default-features --features cuda`) + the `set -o pipefail` stale-binary trap. Mergeable now.
- **#21** `docs/longctx-fp8-diagnosis` → `#19`. The long-context needle diagnosis (see GDN section).
- **#22** `feat/flashinfer-gdn-f32` → `#19`. GDN F32 FFI as **optional infra** (default-off, proven working) + the finding that F32 does NOT fix the long-ctx miss + a WARN that `ATLAS_GDN_FLASHINFER=1` is unsafe for long context.
- **#23** `feat/b12x-moe` → `#22`. The b12x fused-MoE **GPU-independent half** (this is the active work).

## GDN long-context — RESOLVED for correctness (no more GPU work needed)

- Root cause: the **FlashInfer GDN kernel (`libatlasgdn.so`) has a long-context defect** — cold 200K needle 7/8 vs the in-tree **FLA path's 8/8** on identical weights+fp8 KV. `ATLAS_GDN_FLASHINFER=0` (FLA, the default) is the fix, ~7% slower prefill.
- **F32 is ruled out**: re-exported the FI kernel with F32 output (variant A) AND F32 WY-inverse (variant B) — both still 7/8. The defect is not output/inverse precision. Full ledger in `LONGCTX-NEEDLE-BENCHMARK.md`.
- Kept the F32 FFI as optional infra (#22). Full diagnosis in memory `[[atlas-longctx-needle]]`.

## b12x MoE — ACTIVE. Next = the parent GPU validation

**What b12x is:** FlashInfer's SM120/121 CuTe-DSL fused MoE (route/pack+FC1+SwiGLU+FP4-requant+FC2+scatter in one launch), replacing Atlas's ~6-launch grouped-CUTLASS Holo prefill. **Resident-experts ONLY** (`num_local_experts==num_experts`) — orthogonal to over-core/streaming; the gate refuses streamed/null experts. It's a real vLLM path (PR #40082).

**#23 delivered (GPU-independent, verified, default-off behind `ATLAS_HOLO_MOE_B12X=1`):** the fp4 weight repack (`b12x_weights.rs`, UP-first `[E,2I,H/2]` + `[E,H,I/2]`, no re-quant), the scale bake with the SFB-atom risk isolated to a swappable seam (`b12x_scales.rs`, `SfbStrategy::{ConcatReuse,RebuildFromRaw}`), the dlopen FFI (`ops/b12x_flashinfer.rs`), the dispatch gate (`forward_prefill_b12x.rs`), and `3rdparty_patches/b12x_aot/`. Tests 216→229. **Holo geometry: E=256, H=2048, I=512, top_k=8** (NOT 512 experts).

**The remaining parent GPU steps (recipe in `3rdparty_patches/b12x_aot/STATUS.md`), in order:**

1. **P0 — go/no-go.** The `proof_sfb_atom.py` byte-comparison is NOT runnable as written (see its BLOCKER banner: needs a ctypes wrapper around Atlas's compiled `atlas_cutlass_pack_weight_sfb`, and the FI side must use `fp4_quantize(is_sf_swizzled_layout=True)` before `convert_sf_to_mma_layout`). **Prefer the FUNCTIONAL P0**: feed one expert's Atlas-packed weights+SF through b12x vs Atlas's grouped GEMM, compare output cos (tolerance, not bit-exact — b12x uses atomic scatter). This needs b12x actually running (step 2 env). If cos passes → keep `ConcatReuse`; if not → switch the seam to `RebuildFromRaw`.
2. **Env.** Start a container from **`atlas-gb10:gdnf32-build`** (10.7GB, the export sandbox: has `/opt/flashinfer` @ a671c02 + cutlass-dsl). **Downgrade cutlass-dsl 4.5.x → 4.4.2** (4.5 emits bad PTX on sm121) and apply the sm_121a patches to cutlass-dsl `warp/mma.py` + `tcgen05/{mma,copy}.py` — exact pins/seds are in **`/home/ms/spark-vllm-docker/mods/exp-b12x/run.sh`** (the working vLLM b12x reference on this box). Install torch (`pip install torch --index-url https://download.pytorch.org/whl/cu130` worked) + flashinfer deps (`tvm-ffi packaging ninja einops pynvml nvidia-ml-py requests tqdm tabulate`). Apply `b12x_aot/b12x_moe_aot_export.patch` (git apply --check passes) — it's the 1-hunk `stream` annotation on `moe_dispatch.py:1383`.
3. **Export.** `b12x_export.py` captures the **dynamic** prefill kernel (`_get_dynamic_kernel` → `cute.compile` at `moe_dispatch.py:1675`; num_tokens is a runtime Int32 so it exports ONCE; `share_input_across_experts=False`). Strip `--enable-tvm-ffi` from the compile options (keep `--opt-level 2`) via the gdn_export monkeypatch trick. `export_to_c` → `gdn_holo`-style .o/.h.
4. **Freeze the shim.** `b12x_shim.cpp` is stubbed (FREEZE-AT-P3) — freeze the arg marshalling against the generated `.h`: **19 pointers + 16 fixed-shape memref descriptors + 5 runtime i32 (num_tokens, max_rows, rows_padded, max_tasks, max_phys_tiles) + CUstream**, order at `moe_dispatch.py:1781-1822`. Workspace geometry in `allocate_sm120_dynamic_workspace` (`moe_dispatch.py:1236-1316`); only `barrier_count`/`barrier_epoch` need one-time zeroing (kernel Phase-0 self-clears the rest). The 4 f32 `[E]` scale args: `input_global_scale = w1_alpha` (reused!), `alpha = w1_alpha`, `down_alpha = w2_alpha`, `global_scale = fc2_input_scale` expanded to `[E]`.
5. **Relink** `libatlasb12x.so` (g++ -shared, bundle the .o + cute runtime + cudart; see the GDN relink cmd used this session). Native harness bit-check (`b12x_harness.cpp`).
6. **Rebuild Atlas** (docker builder or reuse `atlas-gb10:gdnf32-build`'s spark binary — it already compiled #23's code). Serve with `ATLAS_HOLO_MOE_B12X=1 ATLAS_GDN_LIB`-style override for the b12x .so + the prod fp8 env set. Validate: MoE correctness A/B vs grouped CUTLASS, MoE tok/s win, and the vLLM b12x control.

**Full export-surface details** (40+ args, workspace geometry, static-m=1 decode) are in the workflow result: `.../subagents/workflows/wf_9c0f9f02-e6b/journal.jsonl` (the `u:export-surface` agent) — the single best reference for the shim freeze.

## Environment facts

- **Hardware**: dgx-00 GB10 (serve/export/GPU), gx10-9959 RDMA peer (clean, <20GB test cap). GB10 = sm_121a.
- **Builder image** `atlas-gb10:gdnf32-build` (10.7GB) built from #23's tree with sccache + cache mounts — reuse it. The `gdn-export`/`gdn-serve` containers were removed; recreate from the image.
- **Host flashinfer** `/home/ms/flashinfer` @ `a671c02` (git repo, revertable). All export patches this session were applied **container-side only** — host is untouched.
- **Prod serve recipe** (Holo 35B, all the `ATLAS_HOLO_*` + CUTLASS envs): was in `$CLAUDE_JOB_DIR/tmp/needle-longctx.sh`; the canonical flag set + envs are in `LONGCTX-NEEDLE-BENCHMARK.md` and memory `[[feedback-atlas-serve-flags]]`.
- **MUST build/validate in docker** (host `cargo build` drops CUTLASS + native-fp8 = wrong numerics). Host build also fails on `-lnccl` unless `--no-default-features --features cuda` (#20).

## Open follow-ups (memory)

- Flag-naming cleanup: `ATLAS_HOLO_*` → qwen3.5/3.6 family names (`[[atlas-flag-naming-cleanup]]`). Much-later.
- Peer-side GC for orphaned decode namespaces (bounded, documented in #19; `[[atlas-longctx-needle]]` region).
- The FlashInfer GDN kernel's actual defect (not output/inverse precision) — a layer-by-layer FLA-vs-FI state-trajectory diff at depth would pin it, if someone wants FlashInfer speed + correctness. Low priority (FLA works).
