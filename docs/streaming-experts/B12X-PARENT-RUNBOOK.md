# b12x Parent-GPU Runbook

## State

**Done (landed, no GPU needed):** GPU-independent Rust + AOT scaffolding merged in #23 — `b12x_scales.rs`, `b12x_weights.rs`, `b12x_flashinfer.rs`, `forward_prefill_b12x.rs`, the export driver `b12x_export.py`, the C-ABI `b12x_shim.cpp` (STUBBED), `b12x_harness.cpp`, `proof_sfb_atom.py` (NOT runnable), STATUS.md/HANDOFF. Image `atlas-gb10:gdnf32-build` (10.7GB) is built and already ships `/opt/flashinfer @ a671c02` + cutlass-dsl; it also contains a spark binary compiled from #23's code.

**Remaining (this runbook, all on the parent GB10 / sm_121a / dgx-00):** env re-provision inside a fresh container → P0 SFB go/no-go → AOT export of the dynamic kernel → freeze the shim marshalling against the generated `.h` → relink `libatlasb12x.so` → native harness replay → rebuild Atlas → correctness A/B + needle gate → perf validate.

**Frozen invariants (do not violate):** geometry is `E=256, H=2048, I=512, top_k=8` (NOT 512 experts; asserted at export). Tolerance everywhere is `cos ≥ 0.999 AND rel-L2 ≤ 2e-3` (atomic-add scatter ⇒ never bit-exact). `ATLAS_HOLO_MOE_B12X=1` + `--stream-experts` is a load-time HARD ERROR; EP `world_size>1` / null-expert / missing lib / missing `_t` tables silently disable b12x → grouped fallback.

---

## PHASE 0 — Start container + env setup

The old `gdn-export`/`gdn-serve` containers were removed; recreate from the image. All prior export patches were container-side only, so host `/home/ms/flashinfer @ a671c02` is clean and revertable.

```bash
docker run --rm -it --gpus all \
  -v /home/ms/atlas/.claude/worktrees/streaming-experts-mvp:/atlas \
  -v /home/ms/flashinfer:/home/ms/flashinfer \
  atlas-gb10:gdnf32-build bash
```

**0a. Resolve the real site-packages** (the image uses `/opt/flashinfer` + venv, NOT the run.sh default `/usr/local/lib/python3.12/dist-packages`). The sed `find`s are path-agnostic, so re-point `SITE_PACKAGES` first:

```bash
export SITE_PACKAGES=$(python -c 'import cutlass,os;print(os.path.dirname(os.path.dirname(cutlass.__file__)))')
```

**0b. Downgrade cutlass-dsl 4.5.x → 4.4.2** (4.5.x emits bad PTX on sm121 — `_mma` rejected by ptxas). All three must match:

```bash
uv pip install \
  nvidia-cutlass-dsl==4.4.2 \
  nvidia-cutlass-dsl-libs-base==4.4.2 \
  nvidia-cutlass-dsl-libs-cu13==4.4.2 -q
```

**0c. sm_121a seds on cutlass-dsl** (re-apply every install — pip wipes vendored cutlass):

```bash
# warp/mma.py — runtime arch guard + admissible list
for f in $(find "$SITE_PACKAGES" -name mma.py -path '*/warp/*'); do
  sed -i "s/if not arch == Arch.sm_120a:/if arch not in (Arch.sm_120a, Arch.sm_121a):/" "$f"
  sed -i 's/^\(\s*\)"sm_120a",$/\1"sm_120a",\n\1"sm_121a",/' "$f"
done
# tcgen05/mma.py — insert both archs after sm_103a
for f in $(find "$SITE_PACKAGES" -name mma.py -path '*/tcgen05/*'); do
  sed -i "/Arch.sm_103a,/a\\        Arch.sm_120a,\n        Arch.sm_121a," "$f"
done
# tcgen05/copy.py — allow sm_120f family
for f in $(find "$SITE_PACKAGES" -name copy.py -path '*/tcgen05/*'); do
  sed -i "s/arch.is_family_of(Arch.sm_110f)/arch.is_family_of(Arch.sm_110f) or arch.is_family_of(Arch.sm_120f)/" "$f"
done
```

**0d. Wipe `__pycache__`** so patched code takes effect:

```bash
find "$SITE_PACKAGES" -name __pycache__ -path '*/cutlass*'    -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -name __pycache__ -path '*/flashinfer*' -exec rm -rf {} + 2>/dev/null || true
```

**0e. torch cu13 aarch64 + FI runtime deps** (skip if already present in image):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install tvm-ffi packaging ninja einops pynvml nvidia-ml-py requests tqdm tabulate
```

**0f. FlashInfer enablement patch** (1 hunk: adds `import cuda.bindings.driver as _cuda_aot` + annotates the dynamic-launch `stream` param → `stream: _cuda_aot.CUstream`; unannotated stream = same failure class as GDN):

```bash
git -C /home/ms/flashinfer apply /atlas/3rdparty_patches/b12x_aot/b12x_moe_aot_export.patch
```

**0g. Drop the stale import** from the pip/vendored copy (main's `__init__` references a removed `sm120_moe_dispatch_context`; symbol unused):

```bash
SM12X_INIT="$SITE_PACKAGES/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/__init__.py"
sed -i '/sm120_moe_dispatch_context/d' "$SM12X_INIT"
find "$SITE_PACKAGES/flashinfer" -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

**PASS check:** `git -C /home/ms/flashinfer apply --check .../b12x_moe_aot_export.patch` returns clean (already verified @ a671c02). CUDA ≥ 13 present (cuda-13.2). Do NOT apply run.sh lines 18-25/111-114 (vLLM-serve-only sm121-cap sed) — not needed for AOT export.

---

## PHASE 1 — P0 SFB go/no-go

**Goal:** decide `SfbStrategy`. Resolver at `b12x_scales.rs:38-43`: env `ATLAS_B12X_SFB_STRATEGY` unset/`concat` → `ConcatReuse` (default); `=rebuild` → `RebuildFromRaw`.

**DECISION POINT — which P0 to run:**

- **DO NOT run `proof_sfb_atom.py` as written.** It is NOT runnable (BLOCKER banner `:43-56`). Two defects: (1) `atlas_swizzle` imports `atlas_pack_sfb` which does not exist — would require first building a ctypes/cdylib re-exporting Atlas's compiled `atlas_cutlass_pack_weight_sfb` from `crates/spark-runtime/src/cutlass/pack.rs`; (2) `fi_swizzle` feeds raw ramp bytes to `convert_sf_to_mma_layout`, which compares the wrong thing (must first `fp4_quantize(w, is_sf_swizzled_layout=True)` then pass the swizzled 2D SF). Only pursue this if the functional P0 is ambiguous and you need atom-level layout confirmation.

- **RUN the FUNCTIONAL P0 instead** (HANDOFF step 1, preferred): feed ONE expert's Atlas-packed weights+SF through b12x vs Atlas's grouped GEMM and compare output cosine. Sidesteps atom-layout derivation entirely. Depends on b12x actually running — i.e. Phase 0 complete. This is naturally satisfied by Phase 7's MoE A/B; running it early (single expert) is the isolated go/no-go.

**Frozen scale decisions (already baked in #23 Rust, confirm not overridden):**
- `w1_alpha = ones`; up/gate `weight_scale_2` is BAKED into the w13 SFs (MANDATORY — kernel reuses `w1_alpha` as FC1 input-quant scale, applied quadratically, so a per-projection scale2 cannot be represented). Baked in `bake_w13_logical`.
- `w2_alpha = down.weight_scale_2` (lossless, unbaked). `ATLAS_B12X_BAKE_W2=1` bakes it and sets `w2_alpha=ones` — vLLM-parity/bisection ONLY, not prod.
- `fc2_input_scale = 1.0` always (`fc2_gs = ones`) — kernel does dynamic per-block FC2-input quant.

**P2 census check (do before trusting P0):** confirm `gate_ws2 == up_ws2` across all 256 experts × layers (`weights.rs`) — validates the single-per-expert-alpha assumption. Sanity negatives: swap up/gate halves ⇒ cos<0.9; skip bake ⇒ off by ~ws2.

**DECISION — P0 outcome:**
- **cos ≥ 0.999 (PASS)** ⇒ keep `ConcatReuse`. The fp4 repack + scale bake ship AS WRITTEN — zero Rust change. Proceed to Phase 2.
- **cos FAIL** ⇒ set `ATLAS_B12X_SFB_STRATEGY=rebuild` and implement the FI-matching swizzle in `swizzle_sfb`'s `RebuildFromRaw` arm (`b12x_scales.rs:186-190`, currently `anyhow::bail!`). Stage-(a) host bake/assembly is untouched either way. Re-run P0 until PASS before Phase 2.

---

## PHASE 2 — AOT export (P3)

**JIT smoke first** (P1, compile-only, returns early — validates seds+patch+arch before committing to export):

```bash
cd /atlas/3rdparty_patches/b12x_aot
CUTE_DSL_ARCH=sm_121a PYTHONPATH=/home/ms/flashinfer python b12x_export.py --jit-smoke
```

**Full export** (captures the ONE dynamic kernel; runtime `num_tokens/max_rows/rows_padded/max_tasks/max_phys_tiles` are `Int32` placeholders `1,1,1,1,1` — one export serves any prefill token count AND any workspace capacity; compile cache key contains no `m`):

```bash
CUTE_DSL_ARCH=sm_121a PYTHONPATH=/home/ms/flashinfer python b12x_export.py \
  --out /tmp/b12x_aot --name b12x_dyn_0 --max-tokens 1024
```

- `--max-tokens 1024` is a geometry-dump capacity hint only; kernel keeps num_tokens runtime.
- `--static-m ""` is a no-op (deferred Design-B decode) — leave unexecuted.
- The driver monkeypatches `cute.compile` to strip only `--enable-tvm-ffi` (keeps `--opt-level 2`); tvm-ffi changes arg marshalling and pulls a tvm runtime dep the plain C shim can't honour.
- Baked constexprs: `E=256, k=2048, n=512, top_k=8, w1_rows=2n=1024`, tiler `(128,128)`, `mac=min(get_max_active_clusters(1),sm_count)` (GB10 probe). `share_input_across_experts=False` is mandatory (Holo `w1_alpha` is per-expert `[256]`).

**Expected artifacts in `/tmp/b12x_aot`:**
- `b12x_dyn_0.h` + `b12x_dyn_0.o` (from `cf.export_to_c`)
- `b12x_dyn_0.geom.txt` — line 1 `E=256 H=2048 I=512 top_k=8 max_tokens=1024`; line 2 `repr(_dynamic_task_geometry(...))` — the SSOT for `physical_tiles`, `max_tasks`, `rows_padded`, `cols_pad_k` that the shim must not hardcode.

**PASS/FAIL:** `.h` and `.o` exist and non-empty. **If `geom.txt` line 2 prints `[warn] geometry dump skipped`** the driver's `max_tokens=` kwarg did not match FI's `_dynamic_task_geometry` signature (`moe_dispatch.py:1210`) — the dump silently skipped. You then have NO geometry SSOT for the shim; fix the kwarg (or read the formulas directly: `physical_tiles = ceil(R/128) + min(E,R) - 1`, `rows_padded = physical_tiles*128`, `slice_groups = ceil(n/128)`, `max_tasks = physical_tiles*slice_groups`, `cols_pad_k = align_up(k/16, 4)`) before Phase 3.

---

## PHASE 3 — Freeze shim (P3)

Edit `/atlas/3rdparty_patches/b12x_aot/b12x_shim.cpp` against the generated `b12x_dyn_0.h`. The authoritative ordered call is FI's `runtime_args` tuple at `moe_dispatch.py:1781-1822`; descriptor kinds are fixed by the compile-time fakes at `1569-1673`. Marshalling shape: **19 raw pointers + 16 fixed-shape memref descriptors + 5 runtime i32 + CUstream** (40 args + stream).

**Ordered wrapper prototype** (P=pointer, pass `data_ptr()`; M=memref descriptor):

| # | slot | kind | shim source |
|---|------|------|------|
|1|a_input|P|`x_bf16`|
|2|topk_ids|P|`topk_ids_i32`|
|3|topk_weights|P|`topk_w_f32`|
|4|packed_a|P|ws.packed_a_view|
|5|sfa|P|ws.packed_input_scale|
|6|packed_a_storage|P|ws.packed_a_flat|
|7|scale_storage|P|ws.scale_flat|
|8|barrier_count|M `[1]`i32|ws (ZERO once)|
|9|barrier_epoch|M `[1]`i32|ws (ZERO once)|
|10|pair_head|M `[1]`i32|ws|
|11|producers_done_count|M `[1]`i32|ws|
|12|all_work_published|M `[1]`i32|ws|
|13|task_head|M `[1]`i32|ws|
|14|task_tail|M `[1]`i32|ws|
|15-21|task_ready/expert/m_tile/slice_begin/slice_count/valid_rows `[max_tasks]`, tile_write_count `[physical_tiles]`|P|ws|
|22|b_w13|M `(w1_rows=1024,k=2048,E=256)` stride_order`(1,0,2)`|`w13_fp4`|
|23|sfb_w13|P|`w13_sf`|
|24|b_down|M `(k=2048,n=512,E=256)` stride_order`(1,0,2)`|`w2_fp4`|
|25|sfb_down|P|`w2_sf`|
|26|row_counts|M `(E,)`i32|ws|
|27|expert_write_rows|M `(E,)`i32|ws|
|28|expert_tile_base|M `(E+1,)`i32|ws|
|29|input_gs|M `(E,)`f32 align16|`w1_alpha`|
|30|alpha|M `(E,)`f32|`w1_alpha`|
|31|down_alpha|M `(E,)`f32|`w2_alpha`|
|32|global_scale|M `(E,)`f32|`fc2_gs` (fc2_input_scale expanded to `[E]`)|
|33|scatter|P|`out_bf16`|
|34|token_map|P|ws `[rows_padded]`i32|
|35|token_weights|P|ws `[rows_padded]`f32|
|36|num_tokens|i32|arg|
|37|max_rows|i32|`workspace.max_rows` (= `rows_padded`)|
|38|rows_padded|i32|`physical_tiles*tile_m`|
|39|max_tasks|i32|task capacity|
|40|max_phys_tiles|i32|physical_tiles capacity|
|+|CUstream|—|`stream`|

Count check: 19 P (1-7,15-21,23,25,33-35), 16 M (8-14,22,24,26-32), 5 i32 (36-40), +CUstream.

**Memref caution:** slots 22/24 are the ONLY non-compact descriptors — `stride_order=(1,0,2)` (k-major, E outermost) = Atlas `[E,2n,k/2]` contiguous `.permute(1,2,0)`. All other M are compact/row-major. SF-storage pointers must be the physical 2D swizzled storage `convert_sf_from_mma_layout(...).contiguous()` — Atlas repack must emit that layout directly. The exact descriptor byte layout (allocated_ptr, aligned_ptr, offset, sizes[], strides[]) has NO GDN precedent for the `make_ptr` pointer-fakes and MUST be frozen against the actual `.h` rendering, not guessed.

**Workspace the shim allocates** (from `geom.txt`, `allocate_sm120_dynamic_workspace`): `packed_input [1,rows_padded,k/2]`u8; `packed_input_scale [rows_padded,cols_pad_k]`u8; `row_counts[E]`i32, `token_map[rows_padded]`i32, `token_weights[rows_padded]`f32; `expert_write_rows[E]`i32, `expert_tile_base[E+1]`i32; 7× `[1]`i32 singletons; 6× `[max_tasks]`i32 task arrays; `tile_write_count[physical_tiles]`i32.

**Zeroing:** ONLY `barrier_count` + `barrier_epoch` need one-time zeroing at alloc; kernel Phase-0 self-clears everything else incl. `scatter_output`. **NO per-call memset.**

**Stubs that MUST change** (current file): L16 `#include "b12x_dyn_0.h"` (now exists); L34 `gdn_like_module_t g_module` → generated module type (name only known from `.h`); L77 load symbol → exact `b12x_dyn_0_*_Kernel_Module_Load` name from `.h`; `ensure_ws()` L57-72 GUESSED sizes → discrete named buffers above (fix `max_rows=cap*TOPK` → `rows_padded`); `atlas_b12x_moe_prefill` L96-105 → build 16 memrefs + 19 pointers, wire ws+Atlas ptrs, call the wrapper with the 5 i32 + stream, **return its actual ret** (kill `return 1`). Keep the `num_tokens > cap → return 2` capacity guard. Leave static-decode surface (L110-116) stubbed.

The three exported symbols the Rust `dlsym`s (`b12x_flashinfer.rs:79-81`): `atlas_b12x_load`, `atlas_b12x_moe_prefill`, `atlas_b12x_max_tokens` (must return `>0`).

---

## PHASE 4 — Relink `libatlasb12x.so` (P8)

Discover the cute runtime lib dir, then link the frozen shim against the AOT object:

```bash
cd /tmp/b12x_aot
CUTE_LIB=$(python -m cutlass.cute.export.aot_config --ldflags --libs)   # inspect output; extract the -L dir
g++ -O2 -fPIC -shared /atlas/3rdparty_patches/b12x_aot/b12x_shim.cpp b12x_dyn_0.o \
  -o libatlasb12x.so \
  -I/usr/local/cuda/include -lcudart -L<cute_lib> -lcute_dsl_runtime -Wl,-rpath,<cute_lib>
```

**PASS:** `nm -D libatlasb12x.so` shows `atlas_b12x_load`, `atlas_b12x_moe_prefill`, `atlas_b12x_max_tokens`. Missing any ⇒ Atlas WARNs and silently falls back to grouped. Ship `libatlasb12x.so` + `libcute_dsl_runtime.so` beside `libatlasgdn.so`; path override is `ATLAS_B12X_LIB` (defaults to `libatlasb12x.so` on dlopen search path).

---

## PHASE 5 — Native harness (P9)

```bash
cd /tmp/b12x_aot
ATLAS_B12X_MAX_TOKENS=1024 CUTE_DSL_ARCH=sm_121a \
LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:<cute_lib>:/usr/local/cuda/lib64 \
  /atlas/3rdparty_patches/b12x_aot/b12x_harness /tmp/b12x_aot
```

Requires reference-IO `.bin` files in the dir (raw little-endian): `ref_x_bf16.bin, ref_topk_ids_i32.bin, ref_topk_w_f32.bin, ref_w13_fp4.bin, ref_w13_sf.bin, ref_w2_fp4.bin, ref_w2_sf.bin, ref_w1_alpha.bin, ref_w2_alpha.bin, ref_fc2_gs.bin, ref_out_f32.bin`. `T = x.size()/(2*2048)`.

**PASS gate:** `cos ≥ 0.999 AND rel_l2 ≤ 2e-3`. A non-zero prefill ret prints "not yet frozen / capacity" and exits — so a still-stubbed shim (`return 1`) fails here BY DESIGN; a real relinked shim must return 0 and clear tolerance.

**NOTE / blocker:** `b12x_export.py` as written emits only `.h/.o/.geom.txt` — it contains NO reference-IO writer. STATUS P3/P7 claim export produces "reference IO," but the digests show that writer is absent from the driver. See OPEN QUESTIONS — you must locate/produce the `ref_*.bin` generator (proof/harness path) or this phase cannot run.

---

## PHASE 6 — Rebuild Atlas (P10)

**DECISION POINT — build path:** Do NOT use host cargo — it drops CUTLASS + native-fp8 ⇒ wrong numerics (and fails on `-lnccl` unless `--no-default-features --features cuda`).
- **Reuse** the spark binary already in `atlas-gb10:gdnf32-build` (it compiled #23's code) if no Rust changed since #23 (i.e. P0 PASSED → ConcatReuse, no `RebuildFromRaw` edit).
- **Rebuild via docker builder** if you implemented `RebuildFromRaw` or otherwise touched Rust.

**Unit gate (must be green — flag-unset build must be byte-identical to grouped):**

```bash
cargo test -p spark-model b12x
```

---

## PHASE 7 — Correctness A/B + needle gate (P11)

**MoE A/B, T=64:** routing from Atlas's own `moe_topk_softmax_batched`. (A) grouped-CUTLASS through `unpermute_reduce`, **routed-only, exclude shared expert** vs (B) b12x with `ATLAS_HOLO_MOE_B12X=1`.
**Accept:** `cos ≥ 0.999 AND rel-L2 ≤ 2e-3` — tolerance only, never bit-exact.

**Bisection knob if A/B fails:** `ATLAS_B12X_BAKE_W2=1` (bakes w2 alpha, sets `w2_alpha=ones`) to isolate FC2-scale issues vs vLLM parity. Revert for prod.

**Needle regression gate** (`ATLAS_HOLO_MOE_B12X=1`):

```bash
/home/ms/.claude/jobs/42b99a42/tmp/needle-longctx.sh
```

**Must hold baseline: 8/8 to 128K; must NOT regress the existing 7/8 @ 200K.** Any drop below baseline ⇒ FAIL, do not ship.

---

## PHASE 8 — Perf validate (P12)

Canonical serve flags (SLAI @ 100ms, ssm-slots 256, **no profiling**):

```bash
ATLAS_TARGET_MODEL=holo-3.1-35b-a3b RUST_LOG=info \
  <spark-serve-binary> \
  --scheduling-policy slai --tbt-deadline-ms 100 --ssm-cache-slots 256 \
  --max-seq-len 262144 --target-kv-tokens 250000 --enable-prefix-caching \
  --max-prefill-tokens 16384
```

**A/B:** prefill tok/s with `ATLAS_HOLO_MOE_B12X=1` vs grouped control `ATLAS_HOLO_MOE_GROUPED_CUTLASS=1`, and vs vLLM b12x on the same box. No number is promised until measured. Append P0–P13 results to the "Results log" in STATUS.md. Decode (Design-B static-m) stays deferred — promote only on positive DFlash-acceptance A/B or measured decode-MoE gap > 5%.

---

## OPEN QUESTIONS / RISKS

1. **Reference-IO generator is missing (blocks Phase 5).** `b12x_export.py` emits only `.h/.o/.geom.txt`; the `ref_*.bin` files the harness requires have no writer in the driver despite STATUS P3/P7 claiming otherwise. Where these come from (a section of `proof_sfb_atom.py`? a separate dump added to the export driver? hand-rolled from Atlas grouped output?) is unresolved. Operator must find or write this before the native harness can run — the harness is otherwise un-feedable.

2. **`geom.txt` may silently skip.** The geometry dump is wrapped in try/except; the driver calls `_dynamic_task_geometry` with kwarg `max_tokens=` and if FI's signature differs it prints `[warn] geometry dump skipped` and the shim loses its size SSOT. Only knowable after running Phase 2. Fallback formulas are documented but must be verified against the actual FI function.

3. **Memref descriptor byte layout is unknowable until the `.h` exists.** The 16 M-slots — especially the two non-compact `stride_order=(1,0,2)` descriptors (22/24) and the `make_ptr` pointer-fakes — have no GDN precedent. Phase 3 cannot be fully specified from the digests; the exact struct field order/offsets must be read off the generated header. Highest-risk hand-freeze in the whole runbook.

4. **E=512 contradiction (resolved but watch for it).** Journal export-surface agent (idx 5) and its target table say `E=512` — WRONG (stale brief). Driver hard-asserts `E=256`. Any tooling or doc still carrying 512 is bad; do not bake 512 anywhere.

5. **site-packages path inside the image is unconfirmed.** run.sh assumes `/usr/local/lib/python3.12/dist-packages`; the image uses `/opt/flashinfer` + a venv. Phase 0a resolves it dynamically, but if `cutlass.__file__` points somewhere the seds don't reach, the sm_121a patches silently no-op and export fails with a ptxas `_mma` rejection — diagnose by re-checking that `warp/mma.py` actually contains `sm_121a` after the sed.

6. **`proof_sfb_atom.py` remains non-runnable.** If the functional P0 is inconclusive and atom-level layout proof is needed, someone must first build the `atlas_pack_sfb` ctypes wrapper (re-export `atlas_cutlass_pack_weight_sfb`) AND fix the `fi_swizzle` to quantize a real tile before `convert_sf_to_mma_layout`. Neither is done; treat as a separate mini-task, not a step.

7. **`<cute_lib>` discovery output format is unverified.** `python -m cutlass.cute.export.aot_config --ldflags --libs` is assumed to yield the `-L`/lib dir for the g++ relink and the harness `LD_LIBRARY_PATH`. Confirm its actual output shape before wiring it into Phases 4/5.

8. **Dynamic-serves-decode assumption.** The dynamic export is asserted numerically valid at any `num_tokens` (incl. m=1), so it can serve decode initially; static-m (Design-B) is a perf follow-up. This is a design claim, not yet measured — the decode-MoE gap that would justify promoting static-m is unquantified until Phase 8.
