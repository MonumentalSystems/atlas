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
- (pending P0…P13)
