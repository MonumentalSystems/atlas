# Holo 3.1 Handoff

Date: 2026-06-20
Workspace: `/home/ms/atlas`
Branch: `feature/holo-port-pr177` (merged from PRs in flight, tree not clean)
Hardware: DGX GB10 (unified memory)
Test memory ceiling: 65 GB

## Objective

Keep working on Holo 3.1 throughput on Atlas with priority on raw C=4 decode and prefill, before investing in dFlash work. DFlash is known not to address prefill and should be treated as separate follow-up.

## Current runtime state (now)

A serving process is currently running:

- `127.0.0.1:8890` (listen)
- Spark process example: `pid=2333783` from `ss` output
- Last active launch was the lm-head dtype sweep run using `/tmp/holo-lmhead-nvfp4.log` and `scripts/holo_serve.sh`

## Baseline config being used

`scripts/holo_serve.sh` defaults now include:

- `ATLAS_HOLO_GPU_UTIL=0.70`
- `ATLAS_HOLO_FAST_MOE_MODE=full`
- `ATLAS_HOLO_FAST_MOE_LAYERS=0-39`
- `ATLAS_HYBRID_MOE_LAYOUT=1`
- `ATLAS_UNIFIED_MOE_LAYOUT=0`
- `ATLAS_HOLO_MAX_SEQ_LEN=32768`
- `ATLAS_HOLO_MAX_SEQS=8`
- `ATLAS_HOLO_MAX_BATCH=8`
- `ATLAS_HOLO_MAX_PREFILL=16384`
- `ATLAS_GDN_FUSED_NORM=1`
- `ATLAS_HOLO_LM_HEAD_DTYPE` is configurable and defaults to `bf16`

Fast A/B benchmark preset for 8K experiments:

```bash
ATLAS_HOLO_MAX_SEQ_LEN=8192 \
ATLAS_HOLO_MAX_SEQS=4 \
ATLAS_HOLO_MAX_BATCH=4 \
ATLAS_HOLO_MAX_PREFILL=2048 \
bash scripts/holo_serve.sh /tmp/whatever.log
```

Benchmark command:

```bash
python3 scripts/bench_holo_atlas.py 127.0.0.1:8890
```

Health/proc check:

```bash
ss -ltnp | rg ':8890'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | sort -n
```

## What is working / measured

- Fused SSM RMS path is in place and active by default via `ATLAS_GDN_FUSED_NORM=1`.
- `gated_delta_rule_decode_f32` now correctly uses `const float*` query/key/value, matching true FP32 recurrent state handling.
- SSM detail profiling has been added (`ATLAS_SSM_DETAIL_PROFILE=1`) and is stable behind graph-capture guard.
- Grouped-MoE graph capture guard fix is present; grouped decode remains slower, so keep it off in launch defaults.

Latest meaningful speed points captured this pass:

- 8K, C=4, max_prefill=2048 (fused norm):
  - c1: 69.6 tok/s
  - c2: 53.8 tok/s
  - c4: 81.8 tok/s
  - prefill pp2048: c1 2788, c2 2821, c4 2782
  - memory: ~61.0 GB
- 32K, C=8, max_prefill=16384 (full-cap run prior to this latest launch):
  - c1: 69.4 tok/s
  - c2: 55.9 tok/s
  - c4: 81.2 tok/s
  - prefill pp2048 c4: 2967
  - memory: ~62.6 GB
- SSM transposed-FP8 decode experiment was tested and reverted:
  - no meaningful speed gain at C=4 and no clear head savings
  - c4 stayed ~81.9 vs ~81.8 baseline
  - removed from tree to avoid maintenance risk

## Changes to carry forward

Files to revisit quickly:

- `scripts/holo_serve.sh`
- `kernels/gb10/common/gated_delta_rule.cu`
- `kernels/gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule.cu`
- `crates/spark-model/src/layers/ops/ssm_gdn_a.rs`
- `crates/spark-model/src/layers/qwen3_ssm/{init.rs,mod.rs,ssm_forward.rs,trait_decode_multi_seq.rs}`
- `crates/spark-model/src/layers/moe/forward_prefill_routed.rs`

## Losing experiments (kept intentionally opt-in and measured)

- `ATLAS_SSM_BATCHED_RECURRENT=1`
  - c4 decode around 59.0 tok/s (slower)
  - keep off
- `ATLAS_MOE_BATCHED_DECODE=1`
  - c4 decode around 61.9 tok/s
  - keep off
- `ATLAS_MOE_GROUPED_DECODE=1`
  - avoids crash with graph capture after fix, but slower (c4 ~77.2)

`ATLAS_GDN_FUSED_NORM=1` and launcher default `ATLAS_HOLO_LM_HEAD_DTYPE=bf16` remain the current safe baseline.

## Session 2026-06-20 (cont.) — measured A/Bs on the 8K/C4 nvfp4 config

Live config benched: `ATLAS_HOLO_LM_HEAD_DTYPE=nvfp4`, `ATLAS_HOLO_FP8_SSM_DECODE=1`,
`ATLAS_DECODE_GRAPHS_MULTISEQ=1`, bf16 KV. Baseline (reproduced twice):
- decode c1 ~69-70, c2 ~53-56, c4 ~81.5 tok/s
- prefill pp2048 c1/c2/c4 ~2780 tok/s (flat across C)
- vLLM targets: decode 75/118/145, prefill 4540/6090/6700

Two clean A/Bs this pass (both settle open questions — do NOT re-run):
- **Skip FP8 for C>1 SSM proj → REFUTED.** `ATLAS_HOLO_FP8_SSM_DECODE=0`: decode c4
  81.5→65.2 AND prefill 2785→690 (both worse). FP8 `w8a16_gemm_pipelined` is correct;
  keep it on. (Closes the roadmap "skip FP8 for C>1" open item.)
- **fp8 KV vs bf16 KV → NO DIFFERENCE at benchmark scale.** `--kv-cache-dtype fp8`:
  c4 81.6 / prefill ~2740 ≈ bf16 baseline. KV bandwidth is NOT the short-context
  bottleneck, so bf16 KV (kept for the dflash track) is free here. Matches the user's
  own read: KV dtype only helps at long context + high concurrency.

### Two pathologies / levers identified (raw decode/prefill priority, pre-dflash)
1. **C-scaling is non-monotonic: c2 (53-56) < c1 (69-70), recovers at c4 (81).** c1 uses
   the optimized single-seq decode path; c2/c4 use the batched path, which is less
   efficient per-step. Root cause partly pinpointed:
   - `multi_seq/qkv.rs` has batched qkv kernels for **n=2 and n=3 ONLY** (NVFP4).
     **n=4 (our target C=4) falls through to the sequential per-token loop**
     (qkv.rs:51-87) → each q/k/v weight read 4× instead of once. This IS the 12.4%
     `dense_gemv_bf16` slice in the nsys profile. **Next concrete code task: add an
     `ms_qkv_batch4` path (+ `w4a16_gemv_*_batch4` kernels), or replace batch2/3/4 with
     one M=n batched GEMM.**
2. **Prefill ~2920 across C vs vLLM 6700.** See dedicated prefill section below.

### PREFILL deep-dive (2026-06-20 cont.) — the user picked this track

Reframing: vLLM prefill is c1 4540 → c4 6700, only **1.47× from concurrency**. Our
single-request prefill is **2916 vs 4540 = 1.56× gap at c1, with NO batching involved.**
So single-request kernel efficiency is the bigger lever; cross-request batching is the
smaller multiplier on top.

Measured prefill A/Bs:
- **max-prefill-tokens 2048→16384 = +5% (2785→~2920) AND ttft_max c4 3.1s→1.0s.** The
  bench prompt is 2820 tok, so 2048 chunked it (2048+772, small 2nd GEMM). KEEPER: the
  launcher production default is already 16384 — the handoff "fast preset" 2048 was
  silently hurting prefill. Use 16384 (or ≥ prompt len) for prefill work.
- **`ATLAS_Q12_BATCHED_FIRST_CHUNK=1` = NO EFFECT** on this bench. Cross-request batched
  prefill IS wired (`prefill_batch_chunk_kernel_batched`, needs ≥2 lockstep streams in
  the `prefilling` vec), but with a 16K budget each 2820-tok prompt completes in ONE
  scheduler iteration → request 1 finishes before request 2 is admitted → they never
  coexist → nothing to batch. **Cross-request prefill batching needs scheduler
  co-dispatch (co-admit concurrent pending prefills before running the step), not a
  flag.** Scheduler: `crates/spark-server/src/scheduler/phase_continue_prefills.rs` +
  `phase_start_prefills.rs`; batched model path exists at
  `crates/spark-model/src/model/trait_impl/prefill_b/batch_kernel.rs`.

Single-request prefill nsys profile (c1, M~3000, eager; harness `/tmp/nsys_prefill.sh`):
- **SSM / gated-delta-rule ~39%** (BIGGEST): `gdn_chunk_delta_h_ksplit` 15%,
  `gdn_recompute_wu` 7.5%, `gdn_chunk_fwd_o` 6.1%, `causal_conv1d_update_prefill` 4.8%,
  `w8a16_gemm_t_pipelined` (SSM proj) 4.5%. ← prime lever; likely where vLLM's optimized
  Mamba/GDN prefill kernels win. Kernels in `kernels/gb10/**/gated_delta_rule.cu` +
  `qwen3_ssm/trait_prefill_gdn.rs`.
- **MoE ~28%**: `moe_w4a16_fused_gate_up_t_k64` 15.2%, `grouped_gemm_ptrtable down` 9.6%,
  unpermute/silu ~4%.
- **Attention ~20%**: `fp8_gemm_t_blockscaled` (qkv/o) 15.7%, flash `inferspark_prefill_64`
  3.1%, rope 1.1%. NOTE high variance in `fp8_gemm_t_blockscaled` (avg 4.5ms, min 1.0/
  max 7.0) and `w8a16_gemm_t_pipelined` (med 0.3/max 5.3ms) — some shapes run badly.

Prefill next steps, priority order:
1. Optimize the GDN chunked-scan prefill kernels (~39%) — biggest single lever.
2. Chase the `fp8_gemm_t_blockscaled` / `w8a16_gemm_t_pipelined` variance (bad shapes).
3. Scheduler co-dispatch for cross-request prefill batching (the ~1.5× multiplier).

### ⚠️ BUILD RECIPE (cost hours of invalid results — READ THIS)
The bin is `spark`, the PACKAGE is `spark-server`. `cargo build -p spark` SILENTLY
FAILS ("package `spark` did not match"). Kernels + nccl are env-gated at BUILD time.
Correct build (produces a runnable holo binary):
```
LD_LIBRARY_PATH=/home/ms/nccl/build/lib LIBRARY_PATH=/home/ms/nccl/build/lib \
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
cargo build --release --bin spark
```
Without LIBRARY_PATH → link error `cannot find -lnccl`. Without ATLAS_TARGET_MODEL →
compiles wrong kernels → runtime `No compiled kernel target matches 'holo3_1_moe'`.
Launcher must set `LD_LIBRARY_PATH=/home/ms/nccl/build/lib` (added to /tmp/holo_ab.sh).
ALWAYS verify `ls -la target/release/spark` mtime advanced + the binary runs after a build.

### IMPLEMENTATION PASS (2026-06-20 cont.) — workflow-planned, results
NOTE: my first builds used the wrong `-p spark` and silently no-op'd, so earlier
"Step 1 +1.4% bit-identical" and all co-dispatch tests were against the STALE binary.
Below are the CORRECTED results from properly-compiled binaries.

Ran a 6-agent analysis workflow → sequenced plan (full plan in the run transcript
`tasks/wkc1etqzj.output`). Headline thesis was "~20% of prefill is host-side stalls
(per-projection cuMemAlloc+synchronize+free) + GDN 32-CTA occupancy". Executed and
**measured** the top items — key lesson: profiler GPU-time% ≠ wall-clock when GPU-bound.

- **Step 1 SHIPPED (working tree): remove per-projection alloc/sync/free in the FP8
  block-scaled GEMM path.** Added persistent `fp8_act`/`fp8_act_scale` arena buffers
  (buffers.rs + sizes.rs) reused at all 4 sites: paged_oproj.rs, paged_qkv.rs (Q/K/V),
  trait_prefill_proj.rs (ssm qkvz), trait_prefill_helper.rs (ssm out_proj). Removed the
  per-call `ctx.gpu.alloc×2` + `synchronize` + `free×2`. **Output BIT-IDENTICAL** (golden
  diff on 2 prompts, temp 0). Result: c1 prefill 2916→2956 (~+1.4%), decode unchanged.
  Real but SMALL — the host stalls were mostly hidden behind GPU work in serving (the
  profiler's eager-mode sync gaps don't bind wall-clock). KEEP IT (free, bit-identical).
- **NVFP4_GATE_UP_M128=1 → NO-OP** (prefill 2899/2916/2937 ≈ baseline). gate_up is not
  occupancy-bound; keep off.
- **CONCLUSION: single-request prefill is GPU-COMPUTE-bound.** The cheap/safe levers are
  exhausted. The c1 gap (2956 vs vLLM 4540) requires real GPU-WORK reduction:
  - GDN V-tile chunk_delta_h (32→64-128 CTAs; 16/48 SMs idle today) — CUDA, ~5%, the
    biggest single kernel; numerically sensitive (cos=1.0 bar).
  - GDN recompute_wu blocked triangular inverse (O(C²) serial → tensor-core), conv1d
    seq-tiling (32→fill SMs), S_out round-trip fusion. All deep CUDA, medium-large.
  - NOTE: `ATLAS_MOE_PREFILL_EXACT_TILES=1` is ALREADY in the launcher, so the MoE
    empty-tile lever (plan step 8) is largely already mitigated — verify before investing.
- **Step 1 (FP8 alloc/sync/free removal) — CORRECTED: REAL +~2.7% prefill, bit-identical,
  KEEP.** Properly built, prefill ~2920→~3000-3017 across c1/c2/c4, decode unchanged,
  output bit-identical to golden. Code is in the working tree (buffers.rs/sizes.rs persistent
  `fp8_act`/`fp8_act_scale` + 4 call sites).

- **Co-dispatch (cross-request prefill batching) — IMPLEMENTED + MEASURED + ABANDONED.**
  Gated `ATLAS_PREFILL_CODISPATCH=1` (default OFF). It ENGAGES (deferred chunk-0, batched
  step fires, queue depth up to 4) but is a **36% LOSS**: pp2048 c2/c4 2925→1884 tok/s,
  TTFT 1.0s→3.0s, AND output is **not bit-identical** (coherent but diverges — co-dispatched
  chunk-0 uses PAGED batched attention vs single-stream FLASH/cache-skip). ROOT CAUSE: our
  single-request prefill is GPU-COMPUTE-bound (2820-tok prompt already saturates the GPU),
  so stacking N requests just multiplies compute-bound work — vLLM's "1.47× from
  concurrency" does NOT transfer (their win is amortizing weight reads when requests are
  small/memory-bound; ours aren't). **DO NOT ENABLE.** Code left dormant behind the flag
  (phase_start_prefills.rs want_codispatch, prefill_a_step.rs defer param, mod_helpers.rs
  micro-batch window, run_batched_prefill.rs shared geometry, batch_kernel.rs first-chunk
  alias). Consider reverting if it bothers; it's a documented losing experiment.

- **NET: the ONLY remaining prefill lever is reducing GPU kernel COMPUTE** (host stalls,
  occupancy toggles, and cross-request batching are all measured dead/loss).

### GDN chunk_delta_h is NEAR-OPTIMAL — "V-tile 32→64 CTAs" is a DEAD END (2026-06-20 doc+source review)
The kernel SOURCE (`kernels/gb10/common/gated_delta_rule_fla.cu`) documents that the
"biggest untried win" the roadmap/workflow proposed is already tried and lost:
- **V-tiling the spine REGRESSED** (fla.cu:256-263): 2 CTAs/head redundantly reload W +
  re-run the serial loop (it's bandwidth-bound, so 2× W loads = worse).
- **Tensor-core spine (chunk_delta_h_tc) REGRESSED** (staging latency unhidden) and is
  architecturally blocked (64KB f32 state must persist across chunks; no room under 99KB
  smem for TC operands; GB10 has no wgmma/TMA, ldmatrix broken).
- **K-split (chunk_delta_h_ksplit<SPLIT>) IS the deployed optimum**: SPLIT=2 → 8 warps/CTA,
  chunk_delta_h 34→26ms, cos=1.0. **SPLIT=4 gave NO further gain** — 8 warps already
  saturates latency hiding; the spine is NOT occupancy-bound past that.
So chunk_delta_h is at its achievable optimum for this algorithm on GB10. Further gain
needs a ground-up smem-state-tiling TC rewrite (huge, uncertain) OR cutting W/U/K reload
BYTES. GB10 = 48 SMs, 99KB smem/SM, 273 GB/s LPDDR (no HBM, ~178-191 GB/s practical).
**Methodology: roofline by BYTES-MOVED ÷ practical-BW, not FLOPs.** Preserve cos=1.0 +
`--fmad=false` (KERNEL.toml) + the FLA exact-replay guard.

Remaining real candidates (ranked, to be decided w/ deep research): (a) `fp8_gemm_t_blockscaled`
attention qkv/o 15.7% — BIGGEST kernel, HIGH variance (1.0-7.0ms) = a fixable bad-shape
path [NOTE: o/q_fp8w_t are None for holo → the pipelined-w8a16 alt isn't wired, would need
weight transpose]; (b) `recompute_w_u` 7.5% (already TC-Gram + 48-SM occupied; O(C²) fwd-sub
serial part); (c) MoE down-proj grouped_gemm 9.6% (untried); (d) native-FP8-SSM prefill
crash fix (layer 36, would cut SSM proj + ~3.5GB); (e) reduce chunk_delta_h reload bytes.

### DEEP RESEARCH CONCLUSION (2026-06-20) — GDN spine is HARDWARE-LIMITED on GB10
98-agent web research on SOTA gated-delta/Mamba2 chunked-scan kernels (fla, vLLM, FlashQLA,
TFLA, Gated DeltaNet). Verdict: **the techniques that beat us all require Hopper hardware we
don't have.**
- flash-linear-attention DOES tile V across CTAs (BV=32/64, assert NK=1 → K not tiled), but
  via TENSOR CORES — W loaded once, the W·S contraction is a matmul that amortizes across the
  V-tile. Atlas's V-tile was SCALAR (redundant per-column W reload) → regressed; Atlas's TC
  variant is blocked by GB10's 99KB smem + register pressure + **no wgmma / no TMA / ldmatrix
  broken** (sm_121 lacks the Hopper warpgroup ops fla/FlashQLA rely on).
- **vLLM PR #25393** attempted the exact chunk-axis parallelization of the serial spine and it
  was REVERTED — a loop-carried race ("naive chunk-axis grid parallelism races"). Even vLLM
  could not parallelize this serial state-passing safely.
- **FlashQLA** (2-3× over fla) is Hopper(sm_90)-only (wgmma/TMA). TFLA (5-way chunk 128-256)
  is mLSTM, not gated-delta.
- State fp32→bf16 on store, W/U via UT(WY) forward-substitution — which is exactly what Atlas
  already does.
CONCLUSION: Atlas's K-split chunk_delta_h is the correct optimum for GB10's constraints; there
is NO missed technique. A TC smem-state-tiling rewrite is the only theoretical path and it's
blocked by hardware (no wgmma/TMA, 99KB smem, register pressure) — not worth pursuing on GB10.

### BATCHED-FLASH PREFILL (2026-06-21) — TRIED, FAILED (wrong + 7× slower), REVERTED
Cross-request prefill batching loses because the batched forward is forced onto slow PAGED
attention. A 4-agent workflow mapped a clean "reuse the flash kernel's existing blockIdx.z
batch dim, dispatch-only wiring" plan (inferspark_prefill.cu has `batch=blockIdx.z` + per-batch
offsets; ops::prefill_attention_64 already sets grid.z=batch). Implemented it (cache_skip.rs
batch-aware via batched_meta + prefill_inner.rs gate flip). RESULT: **both incorrect AND ~7×
slower.** 2 identical concurrent streams agreed with each other (o1==o2, no cross-contamination)
but DIVERGED from single-seq at token 0; batch≥4 produced garbage (one stream); prefill c2/c4
collapsed to **487 tok/s** (vs 3510 baseline / 2060 paged-codispatch), ttft 11.6s. CONCLUSION:
the kernel's blockIdx.z batch dim is NOT compatible with the co-dispatch stacking for large
prefill chunks (likely a stride/layout mismatch the static analysis missed; it was evidently
designed for a different/small-seq regime). REVERTED the gate (prefill_inner.rs) so batched
chunk-0 stays on paged; cache_skip.rs keeps the (now-dead) batched_meta param — single-seq path
is behaviorally identical. REAL fix for prefill concurrency = a genuinely NEW batched-flash
kernel (varlen/ragged, like FlashInfer's) — not the existing kernel's batch dim. Bigger project.

### ★ BREAKTHROUGH (2026-06-21): cuBLASLt GEMM path — the gap is GEMM efficiency, RECOVERABLE
The "exhausted" conclusion below was WRONG. Measured cuBLAS on the SAME GB10, same shapes:
Atlas GEMMs run at ~30% of the cuBLAS bf16 ceiling and ~15-20% of fp8.
| shape | Atlas | cuBLAS bf16 | cuBLAS fp8 |
| SSM qkvz 3537×12288×2048 | 32 TF | 85 | 152 |
| attn Q 3537×8192×2048 | 26 | 84 | 167 |
| MoE gate_up 28296×1024×2048 | 27 | 85 | 154 |
| MoE down 28296×2048×512 | 21 | 59 | 82 |
GEMMs are ~46% of prefill; the GDN scan (~33%) is the hardware-limited floor.

**SHIPPED + MEASURED:** routed the SSM qkvz projection (biggest single GEMM, 14.7%) through
cuBLASLt BF16 behind `ATLAS_CUBLAS_GEMM=1`. Prefill **~3000 → ~3330 (+11%)**, decode
unchanged, output coherent (W16A16 > W8A8 precision). Dequant fp8→bf16 once, cached.
Implementation:
- `crates/spark-runtime/src/cublaslt.rs` — minimal cuBLASLt FFI (`bf16_gemm_act_weight_t`),
  handle+64MB workspace lazily inited. NOTE: `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES=1`
  (0 is SEARCH_MODE — using 0 gives status-7 INVALID_VALUE).
- `crates/spark-runtime/build.rs` — links `cublasLt`. Launcher needs
  `LD_LIBRARY_PATH=...:/usr/local/cuda/lib64` (added to /tmp/holo_ab.sh).
- `crates/spark-model/.../qwen3_ssm/trait_prefill_proj.rs` — `ATLAS_CUBLAS_GEMM` branch +
  `qkvz_dequant_bf16_cached` (launches `dequant_fp8_blockscaled_bf16`, caches by weight ptr).

**ROLLOUT (2026-06-21): all dense projection GEMMs now on cuBLASLt bf16 → +17% cumulative.**
Generalized to `ops::cublas_bf16_proj` + `ops::cublas_gemm_enabled()` (ops.rs) and wired:
SSM qkvz (trait_prefill_proj.rs), attn Q/K/V (cache_skip_qkv.rs), attn O (paged_oproj.rs).
Prefill **~3000 → ~3520 (+17%, flat c1-c4)** = 77% of vLLM (was 66%). Coherent, no OOM
(~2GB of cached bf16 dequant weights). NEXT: native fp8 (below) + MoE grouped GEMM.

**NATIVE FP8 — the next ~1.8× on the converted GEMMs (cuBLAS fp8 152 vs bf16 85 TF).**
CUDA-13 cuBLASLt HAS the exact scale mode: `CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F=5`
(per-128×128 fp32 block) matches Atlas's weight `row_scale` layout EXACTLY → feed
`qkvz_fp8w.weight` + `row_scale` directly (A_SCALE_MODE=5, A_SCALE_POINTER=row_scale),
zero dequant + zero extra memory. BLOCKER: the activation. `per_token_group_quant_fp8`
makes a per-[token,128K] scale `[M,K/128]` which no single cuBLAS mode matches; need EITHER
a new per-[128,128]-block activation-quant kernel (→ both operands BLK128x128, exact) OR
test per-tensor activation scale (B_SCALE_MODE=SCALAR_32F, simplest, quality-risk). Scale
mode attrs: A/B_SCALE_MODE=31/32, A/B_SCALE_POINTER=17/18. Header at
.../nvidia/cu13/include/cublasLt.h. This frees the ~2GB bf16 memory AND ~1.8× the GEMMs.

FP8 RESOLVED (2026-06-21) — BLOCK-SCALED fp8 is a HARDWARE DEAD-END on GB10.
The fp8 code is CORRECT (research-driven fix: weight BLK128x128 + activation VEC128 via the
existing per_token_group_quant_fp8; status went 7→15, i.e. valid API but no algo). BUT:
- cuBLASLt 13.1 AND 13.5.1.27 (downloaded from NVIDIA redist, LD_LIBRARY_PATH'd in) BOTH
  return status 15 NOT_SUPPORTED for the block-scaled fp8 matmul.
- torch._scaled_mm CONFIRMS: on GB10 (sm_120, cap 12.1) per-tensor fp8 WORKS (152 TF) but
  block-scaled fp8 FAILS "Invalid scaling configuration". So 128-block-scaled fp8 MMA is a
  datacenter-Blackwell (sm_100/B200) feature; consumer GB10/sm_121 LACKS it. Not a CUDA
  version issue — a silicon one. Do NOT chase block-scaled fp8 on GB10.
ROW-WISE FP8 ATTEMPTED (2026-06-21) — cuBLAS doesn't support it on GB10 EITHER.
Built the full row-wise path: new kernel `kernels/gb10/common/quant_rowwise_fp8.cu` (per-row
max→E4M3, used for both weight re-quant + per-token act quant), `cublaslt.rs
fp8_gemm_act_weight_t_rowwise` (OUTER_VEC_32F mode), `ops::cublas_fp8_rowwise_proj` +
`requant_weight_rowwise_fp8_cached` (block-fp8→bf16→row-wise-fp8, cached), wired qkvz behind
`ATLAS_CUBLAS_FP8=1`. RESULT: **status 15 NOT_SUPPORTED on cuBLAS 13.1 AND 13.5** for OUTER_VEC
fp8. So GB10's cuBLAS does fp8 ONLY with SCALAR (per-tensor) scaling — NOT block, NOT row-wise.
torch's "RowWise" works because it uses its own CUTLASS kernels, not cuBLAS. CONCLUSION: good-
quality fp8 (row-wise/block) on GB10 requires CUTLASS, not cuBLAS — a heavier integration.
cuBLAS per-tensor fp8 works (152 TF) but per-tensor on a 35B model is quality-risky + needs a
per-forward per-tensor activation reduction (awkward). The row-wise code is built + dormant
behind ATLAS_CUBLAS_FP8=1 (errors on GB10's cuBLAS; works on B200 / a cuBLAS that adds the mode).
NET: fp8 GEMM speedup on GB10 = a CUTLASS project, not a cuBLAS one. bf16 +17% stays the win.

(prior note) torch "RowWise" works on GB10 via cutlass: per-output-row weight scale
+ per-token activation scale. Needs re-quantizing the weight to per-ROW fp8 (current weight is
block-quantized) + the cuBLAS row-wise scale mode + per-token act quant. Better quality than
per-tensor. This is the fp8 follow-up IF pursued; it trades some quality for ~1.8× on the
converted GEMMs. The bf16 +17% (lossless-ish W16A16) is the shipped, working default.
fp8 code retained behind `ATLAS_CUBLAS_FP8=1` (errors on GB10 — leave OFF) for reference /
future B200 use.

FP8 ATTEMPTED (2026-06-21, behind `ATLAS_CUBLAS_FP8=1`, code in cublaslt.rs
`fp8_gemm_act_weight_t_blkscaled` + ops::cublas_fp8_proj): weight BLK128x128 + activation
SCALAR(1.0 cast via bf16_to_fp8) → `AlgoGetHeuristic status 7` (INVALID_VALUE). Diagnosis:
cuBLASLt block-scaled fp8 almost certainly requires BOTH operands block-scaled (mixed
block+scalar rejected). FIX OPTIONS: (a) add a per-128×128-block activation-quant kernel so
both operands use BLK128x128 (exact, best quality); (b) re-quantize the WEIGHT to per-tensor
fp8 at load + per-tensor activation (both SCALAR — torch._scaled_mm proved per-tensor works
at 152 TF on this GB10, but quality risk + needs weight re-quant since current weight is
block-quantized). Path (a) is the clean one. The fp8 flag is OFF by default; bf16 +17% is
the shipped, working default.

**THE PATH TO vLLM PARITY (validated model):** each GEMM converted yields its proportional
share. Convert all GEMMs (~46% of prefill at ~28 TF → cuBLAS ~85 bf16 = 3×) → that 46%
becomes ~15% → ~1.45× prefill (~4350 tok/s ≈ vLLM 4540). With fp8 cuBLAS (~150 TF) it would
EXCEED vLLM. Remaining to convert (ranked): MoE gate_up 15.2% + down 9.6% (GROUPED GEMM —
harder, needs per-expert/batched cuBLASLt), SSM out_proj + attn Q/K/V/O (~25%, easy — same
dense pattern as qkvz). Then move bf16→fp8 cuBLASLt (block-scaled, another ~1.8×) for the
overshoot. GDN scan stays the ~33% floor.

### (SUPERSEDED) FINAL SYNTHESIS — prefill optimization is EXHAUSTED on accessible levers
Every accessible lever tested/researched this session is dead/loss EXCEPT Step 1 (+2.7%).
The remaining 1.56× single-request gap vs vLLM is structural: the W8A8 blockscaled attention
GEMM runs ~9-20% of GB10 fp8 peak (a real inefficiency) and the GDN spine is hardware-limited.
Closing it needs ground-up tensor-core kernel rewrites (attention GEMM; GDN state-tiling) that
are large, numerically-delicate, and partly blocked by GB10 lacking wgmma/TMA — uncertain
multi-day payoff. The mature, honest state: KEEP Step 1; treat further prefill work as a
dedicated kernel-engineering project, not an incremental tuning pass.
`ATLAS_FP8_SINGLE_SCALE=1` tested = no speedup + quality divergence (dead). Co-dispatch code
left dormant behind `ATLAS_PREFILL_CODISPATCH=1` (default off, measured loss).

## Older next-best options (superseded items struck where measured)

1. ~~A/B lm_head dtype~~ — DONE: nvfp4 is flat vs bf16 at C=4 (81.8=81.8). Lever exhausted.
2. Promote winning options to supported paths (with quality sanity checks).
3. Structural kernel fusion (vertical fusion across SSM blocks / MoE gate→expert chains)
   — long-term work. The `ms_qkv_batch4` task above is the smallest concrete first step.
4. Keep dFlash a separate track; does not change the prefill path.

### SESSION 2026-06-21 (cont.) — DECODE cliff RESOLVED + committed to CUTLASS program

**GOAL (user, north star):** vLLM-parity — decode C=4≥145, prefill C=4≥6700; stretch C=16.
Memory <60GB, preferably <50GB. (Current banked: prefill ~3520 c1-c4 = 77% of vLLM;
decode c1 70 / c2 54 / c4 82; mem ~61GB.)

**The c2<c1 decode inversion is NOT a spec cliff — spec is OFF for Holo.** `serve.rs:464`:
`use_speculative = (args.speculative || args.dflash) && has_proposer()`. The launcher
(/tmp/holo_ab.sh) passes NEITHER `--speculative` NOR `--dflash` → `use_speculative=false`
unconditionally. All spec paths (n-gram/self-spec/MTP, gated to `active.len()==1` in
scheduler/mod.rs:328-413) are dead for this config. So c1=70 is PLAIN single-seq decode,
and c2<c1 is pure **batched-decode inefficiency** (the single-seq path is well-optimized;
the n≥2 batched path re-reads weights per-seq). The spec-cliff fix is MOOT — do not pursue.

**Every in-house "easy" decode/concurrency lever is now banked or a measured loss:**
cuBLAS bf16 (+17%, banked) / fp8 cuBLAS (GB10 per-tensor only, dead) / batched-flash (7×
slower, dead) / co-dispatch prefill (36% loss) / `ATLAS_SSM_BATCHED_RECURRENT` (c4 59) /
`ATLAS_MOE_GROUPED_DECODE` (c4 77) / spec cliff (moot). Same root cause each time: Atlas's
hand-written batched kernels don't amortize weight reads; the obvious batching regresses.
vLLM's 145/6700 come from FlashInfer + CUTLASS grouped-GEMM + fused-MoE kernels.

**DECISION (user): commit to the CUTLASS integration program.** The only path that closes
the C=4 gap. Plan: (M0 de-risk) one CUTLASS fp8/bf16 dense GEMM, prove the toolchain end-
to-end + beat hand kernel; (M1 big lever) CUTLASS grouped-GEMM → fused MoE; (M2) cuDNN
varlen flash → batched prefill; (M3) memory (fp8 weights + native-fp8 SSM).

#### Scouting map for next session (two Explore agents, this session) — START HERE

**(A) Build/FFI surface — how to add a CUTLASS .cu (gating risk, now mapped):**
- Kernels compiled by `crates/atlas-kernels/build.rs` (+ `build_target.rs`, `build_codegen.rs`)
  via direct nvcc: `nvcc --ptx -arch=sm_121f -O3 <extra> file.cu -o file.ptx`. Output is
  **PTX text, device-only**. arch `sm_121f` from `kernels/gb10/HARDWARE.toml:4`.
- TWO GOTCHAS for CUTLASS: the NVIDIA compile passes **no `-std`** (CUTLASS needs `-std=c++17`)
  and **no `-I`** (CUTLASS headers need `-I<cutlass>/include` + `/tools/util/include`); also
  it hardcodes `--ptx` (host-driven `cutlass::gemm::device::Gemm` can't be a single --ptx TU).
- ⇒ **Use the host-callable library path (B), NOT the PTX registry path.** Mirror the cuBLASLt
  integration exactly: compile `cutlass_*.cu` to an OBJECT via a NEW nvcc step in
  `crates/spark-runtime/build.rs` (analog of the cublasLt block at build.rs:39-52), declare
  `extern "C"` host wrapper in a new `crates/spark-runtime/src/cutlass.rs` (modeled on
  `cublaslt.rs:379`, takes u64 device ptrs + stream:u64), call from `ops.rs` (analog of
  ops.rs:181). Link with `cargo:rustc-link-lib=static=...` + cudart/cudadevrt.
- Files to touch: new `crates/spark-runtime/src/cutlass.rs`; edit `crates/spark-runtime/build.rs`
  (+lib.rs reg); new `kernels/.../cutlass_*.cu`; new fn in `crates/spark-model/src/layers/ops.rs`.
- (PTX-path registry, for reference: module name=file stem, `gpu.kernel(module, "extern_C_symbol")`,
  loaded via `AtlasRegistry::init` registry.rs:140; launch via `KernelLaunch` kernel_args.rs:51.)

**(B) CUTLASS target shapes (Holo 3.1, from MODEL.toml — authoritative; repo has no config.json):**
- Dims: hidden h=2048, moe_intermediate=512, num_experts=256, top_k=8, shared_expert_inter=512,
  head_dim=256, n_q_heads=16, n_kv_heads=2, attn q gated (q_proj N=2×q_dim=8192), 40 layers
  (10 attn + 30 GDN), GDN key_dim=2048 value_dim=4096.
- **Dense GEMM targets** (M=tokens, N=out, K=in), all via `cublas_fp8_rowwise_proj` today:
  SSM qkvz N=12288 K=2048 | SSM out_proj N=2048 K=4096 | attn q N=8192 K=2048 |
  attn k/v N=512 K=2048 | attn o N=2048 K=4096.
- **MoE (the big C>1 lever) — CRITICAL GOTCHA for grouped GEMM:** decode routes C×8 tokens
  over 256 experts → avg tokens/expert = C·8/256 = **0.125 (C=4), 0.5 (C=16) — far below 1**.
  Per-expert M≈1. THIS is why `ATLAS_MOE_GROUPED_DECODE` (sort/permute + ptr-table grouped
  GEMM) LOST (c4 60→33 under graphs): grouped-GEMM is built for M≫1; the sort/permute fixed
  overhead per layer ×30 dominates at M≈1. Per-token GEMV wins at decode. ⇒ **A CUTLASS
  grouped GEMM must beat M≈1 with minimal per-group overhead (or use a fused-MoE-GEMV design,
  not classic grouped GEMM).** Per-expert shapes: gate/up N=512 K=2048, down N=2048 K=512.
- Decode MoE dispatch: `qwen3_ssm/trait_decode_multi_seq.rs:146` — n=2/3 fused (`forward_k2/k3`),
  **n≥4 falls to per-token loop (:204-214)** calling `MoeLayer::forward` (moe/forward.rs:28).
- Plug points: prefill+grouped-decode share `MoeLayer::run_routed_grouped_gemm`
  (moe/forward_prefill_routed.rs:21); current grouped kernel wrappers in
  `ops/moe_grouped_a.rs:114+`. Sort/permute scaffolding (reusable group descriptors:
  expert_offsets + sorted_token_ids) in `moe/forward_prefill.rs:246,289,308`.

**ENV CHECK (2026-06-21, done):** CUTLASS is NOT installed on this box (no headers, no py pkg).
Toolchain OK: nvcc CUDA 13.0 V13.0.88 supports `sm_121f`; GPU GB10 cap 12.1 confirmed. CUTLASS is
header-only → just needs `-I<cutlass>/include -I<cutlass>/tools/util/include`.

**NEXT ACTION:** (1) `git clone https://github.com/NVIDIA/cutlass` (need ≥3.8 for consumer-Blackwell
sm_120/sm_121; GB10 has no TMA/wgmma → the **Ampere-class mma.sync (SM80) collectives** are what
will run, NOT the sm_100 tcgen05/TMA kernels — copy a `examples/` SM80 bf16 GEMM as the starting
template); confirm it compiles for `sm_121f`. (2) build M0 — a CUTLASS bf16 (or per-tensor-fp8)
dense GEMM host wrapper for ONE projection (SSM qkvz N=12288 K=2048), validate cos≈1 + perf vs
cuBLAS/hand. (3) only then M1 grouped MoE. NOTE the M≈1 MoE-decode reality above — grouped GEMM
may need a fused-GEMV design, and the win there is decode-concurrency (the real C=4 lever), not
prefill. (Re-run the deeper scout on best CUTLASS sm_121 example kernels when starting M0.)

## Sync note

There is an untracked `docs/porting/holo-3.1-handoff.md` with historical notes; use it only for background. This `HANDOFF.md` is the active handoff for the current continuation.

### SESSION 2026-06-22 — CUTLASS M0 + token-major MoE decode experiment

**CUTLASS dense BF16 M0 built and smoke-tested.** Added optional host-callable CUTLASS
integration behind `CUTLASS_HOME` build cfg + `ATLAS_CUTLASS_GEMM=1`. The wrapper compiles
as a runtime object, not through the PTX registry. FFI smoke tests execute real CUDA GEMMs:
small tile-compatible shape and real Holo SSM QKVZ shape `M=3537 N=12288 K=2048`.

Dense primitive sweep on DGX Spark/GB10 showed generic CUTLASS BF16 is **not** a blanket
replacement for cuBLASLt:
- `ssm_qkvz`: CUTLASS modestly wins, about 9-12% faster than cuBLASLt.
- `attn_q/k/v`: roughly parity / tiny wins.
- `ssm_out`, `attn_o`, dense MoE proxy shapes: cuBLASLt wins clearly.

Conclusion: keep CUTLASS M0 as useful infrastructure and maybe route `ssm_qkvz`, but this
does not solve decode concurrency. Generic dense CUTLASS is not the MoE answer on GB10.

**cuBLASLt multi-algo route-batch sweep done.** Small-M routed MoE proxies show library GEMM
only becomes plausible near `M=64-128`; Holo C=4 has only `C*top_k=32` total routes spread
across 256 experts, so classic grouped/dense GEMM remains the wrong shape for decode.

**Token-major MoE decode path implemented + measured + LOST.** Added opt-in
`ATLAS_MOE_TOKEN_MAJOR_DECODE=1`, reusing generic `moe_prefill` token-major kernels for N>=4
instead of grouped-GEMM sorting. Tested on `gx10-9959` with 8K/C4 clamp:

Baseline:
- decode c1 70.7
- decode c2 57.9
- decode c4 84.7
- prefill c4 3135
- spark memory about 55.2 GB

`ATLAS_MOE_TOKEN_MAJOR_DECODE=1`:
- decode c1 70.7
- decode c2 55.5
- decode c4 75.5
- prefill c4 3116
- spark memory about 55.3 GB

Result: token-major decode regressed C4 by about 10.9%; keep it OFF. Do not rediscover this
as a likely win. The generic token-major prefill kernels are still too prefill-shaped for
decode concurrency.

**Current concurrency conclusion:** the next real path is a purpose-built fused decode MoE
kernel, not grouped GEMM, not generic token-major prefill reuse, and not a dense library GEMM
swap. It must reduce the decode-stage chain around routed MoE directly: gate/topk + routed
expert gate/up + activation/down + weighted combine, with minimal per-expert sorting/setup.

## 2026-06-22 - C=4 atomic-add fused MoE decode prototype

Implemented an opt-in prototype behind `ATLAS_MOE_ATOMIC_C4_DECODE=1`:
- `kernels/gb10/common/moe_decode_atomic_c4.cu`
- `crates/spark-model/src/layers/ops/moe_atomic_c4.rs`
- `crates/spark-model/src/layers/moe/forward_atomic_c4.rs`
- SSM multi-seq dispatch routes `n == 4` through it before token-major decode.

Shape/scope:
- Holo NVFP4 only, shared expert required, no transposed/unified layout, no pre-expert norm.
- Reuses batched routing and token-major gate/up.
- Routed down computes weighted contributions into `[4,H]` FP32 scratch with `atomicAdd(float*)`.
- Finalize casts routed FP32 accumulation to BF16. For EP, BF16 all-reduce still happens on routed-only output, then existing `moe_batched_blend` adds shared expert once.
- FP32 scratch is carved from `BufferArena::scratch()` after top-k indices/weights, so no persistent memory increase.

Build:
```bash
LD_LIBRARY_PATH=/home/ms/nccl/build/lib LIBRARY_PATH=/home/ms/nccl/build/lib ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 cargo build --release --bin spark
```
Result: build passed; atlas-kernels compiled 139 kernels including the new module.

Remote test on `gx10-9959` with clamp:
```bash
ATLAS_HOLO_GPU_UTIL=0.80 ATLAS_HOLO_MAX_SEQ_LEN=8192 ATLAS_HOLO_MAX_SEQS=4 ATLAS_HOLO_MAX_BATCH=4 ATLAS_HOLO_MAX_PREFILL=8192 ATLAS_MOE_ATOMIC_C4_DECODE=1
```
Benchmark:
- tg128 c1: 71.7 tok/s
- tg128 c2: 55.3 tok/s
- tg128 c4: 72.7 tok/s
- pp2048 c1: 3109 tok/s
- pp2048 c2: 3126 tok/s
- pp2048 c4: 3141 tok/s

Conclusion: C=4 atomic-add prototype is a regression versus the recorded baseline (`c4 84.7 tok/s`) and even below token-major (`c4 75.5 tok/s`). Do not enable. Likely reasons: same token-major gate/up launch shape plus many small down CTAs, atomic contention/serialization, and no reduction in the dominant SSM batched projection cost.

Next useful direction: stop optimizing classic route-major decode MoE at C=4. For concurrency gains, test CUDA graphs over the current per-token MoE loop or fuse the SSM per-seq recurrent microkernels; MoE-only changes have now shown negative results for grouped, token-major, and atomic C4 paths.

## 2026-06-22 - Multi-seq CUDA graphs test

Atlas already had an opt-in multi-seq decode graph path in `crates/spark-model/src/model/trait_impl/decode_a2.rs`, gated by `ATLAS_DECODE_GRAPHS_MULTISEQ=1`. It captures/replays the full padded batch decode graph keyed by `padded_n` (`2`, `4`, `8`) after pre-step metadata uploads. Tested this directly on `gx10-9959` with the same clamp used for Holo R&D:

```bash
ATLAS_HOLO_GPU_UTIL=0.80 ATLAS_HOLO_MAX_SEQ_LEN=8192 ATLAS_HOLO_MAX_SEQS=4 ATLAS_HOLO_MAX_BATCH=4 ATLAS_HOLO_MAX_PREFILL=8192 ATLAS_DECODE_GRAPHS_MULTISEQ=1
```

Logs confirmed graph capture engaged:
- `Captured CUDA graph for batch size 2`
- `Captured CUDA graph for batch size 4`

Benchmark:
- tg128 c1: 70.8 tok/s
- tg128 c2: 57.7 tok/s
- tg128 c4: 83.7 tok/s
- pp2048 c1: 3136 tok/s
- pp2048 c2: 3135 tok/s
- pp2048 c4: 3138 tok/s

Conclusion: multi-seq CUDA graphs alone are effectively flat versus recorded baseline (`c4 84.7 tok/s`). Launch overhead is not the main C4 bottleneck for this path.

Also tested the combination predicted by comments to possibly help grouped MoE:

```bash
ATLAS_HOLO_GPU_UTIL=0.80 ATLAS_HOLO_MAX_SEQ_LEN=8192 ATLAS_HOLO_MAX_SEQS=4 ATLAS_HOLO_MAX_BATCH=4 ATLAS_HOLO_MAX_PREFILL=8192 ATLAS_DECODE_GRAPHS_MULTISEQ=1 ATLAS_MOE_GROUPED_DECODE=1
```

Benchmark:
- tg128 c1: 71.5 tok/s
- tg128 c2: 58.1 tok/s
- tg128 c4: 80.4 tok/s
- pp2048 c1: 3124 tok/s
- pp2048 c2: 3118 tok/s
- pp2048 c4: 3131 tok/s

Conclusion: graphs + grouped MoE is worse than graphs alone and worse than baseline. Do not enable.

Current negative list for C4 decode R&D:
- grouped MoE without graphs: previously measured bad
- token-major MoE: c4 75.5 tok/s
- atomic C4 MoE: c4 72.7 tok/s
- multi-seq CUDA graphs alone: c4 83.7 tok/s
- multi-seq CUDA graphs + grouped MoE: c4 80.4 tok/s

Next useful direction: SSM-side work, not MoE/launch overhead. The C4 wall appears dominated by batched SSM projection/recurrent work and memory bandwidth, not kernel launch latency.

## 2026-06-22 - SSM-side R&D: fused strided GDN+norm primitive

Docs/test review before touching code:
- `ATLAS_SSM_BATCHED_RECURRENT=1` was already measured as a loss (`c4 ~59 tok/s`), so do not re-run that unchanged.
- Relevant docs: `docs/porting/holo-3.1-optimization-roadmap.md`, `docs/adr/0011-ep-batched-decode-optimization.md`, and existing GDN examples (`gdn_verify_fused_microtest`, `gdn_batched_repro`).
- MTP verify has K=2/3/4 fused WY kernels, but those process multiple tokens in one sequence. C=4 serving is four independent SSM states, so those kernels cannot be directly reused for concurrency.

Implemented a bounded SSM primitive for the existing opt-in batched recurrent branch:
- New CUDA kernel: `gated_delta_rule_decode_f32_strided_norm` in `kernels/gb10/common/gated_delta_rule.cu`.
- New Rust wrapper: `gdn_decode_f32_strided_norm` in `crates/spark-model/src/layers/ops/ssm_gdn_a.rs`.
- New `Qwen3SsmLayer` optional handle: `gdn_f32_strided_norm_k`.
- `try_decode_multi_seq_ssm_batched` now uses the fused strided GDN+gated-RMS path when both `ATLAS_SSM_BATCHED_RECURRENT=1` and `ATLAS_GDN_FUSED_NORM=1` are set and the kernel exists; otherwise it falls back to the old strided-GDN + per-token norm sequence.

Purpose:
- Removes the FP32 intermediate GDN output global write/read and N separate `gated_rms_norm` launches from the already-existing batched recurrent experiment.
- Baseline behavior is unchanged because `ATLAS_SSM_BATCHED_RECURRENT` remains opt-in/off.

Build:
```bash
LD_LIBRARY_PATH=/home/ms/nccl/build/lib LIBRARY_PATH=/home/ms/nccl/build/lib ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 cargo build --release --bin spark
```
Result: build passed; atlas-kernels compiled 139 kernels including the new symbol.

Next validation, if spending `gx10-9959` time:
1. Add or adapt a synthetic microtest from `gdn_verify_fused_microtest` to compare `gated_delta_rule_decode_f32_strided_norm` against `gated_delta_rule_decode_f32_strided` + `gated_rms_norm` for batch=4.
2. Only after cosine passes, A/B server with:
   `ATLAS_SSM_BATCHED_RECURRENT=1 ATLAS_GDN_FUSED_NORM=1` under the usual 8K/C4 clamp.
3. Expectation should be conservative: previous batched recurrent was far behind baseline, so this may remain a loss; it just isolates whether the extra norm/write/read was a material part of that loss.

## 2026-06-22 - SSM fused strided-norm validation result

Added `crates/spark-model/examples/gdn_strided_norm_microtest.rs` and registered it under `gpu-examples`.

Microtest compares:
- old path: `gated_delta_rule_decode_f32_strided` -> `gated_rms_norm_f32_input`
- new path: `gated_delta_rule_decode_f32_strided_norm`

Important target-specific fix: Holo uses `kernels/gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule.cu` as a model-specific override, so the new symbol had to be added there as well as common. Initial symbol lookup failed until this was fixed.

Microtest command:
```bash
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
  cargo run -p spark-model --release --features cuda,gpu-examples --example gdn_strided_norm_microtest
```

Microtest result, local and on `gx10-9959`:
- output cos: `1.000000000`
- output max_abs: `0.00048828`
- h_state cos: `1.000000000`
- h_state max_abs: `0.00000006`

Server A/B on `gx10-9959` under usual clamp:
```bash
ATLAS_HOLO_GPU_UTIL=0.80 ATLAS_HOLO_MAX_SEQ_LEN=8192 ATLAS_HOLO_MAX_SEQS=4 \
ATLAS_HOLO_MAX_BATCH=4 ATLAS_HOLO_MAX_PREFILL=8192 \
ATLAS_SSM_BATCHED_RECURRENT=1 ATLAS_GDN_FUSED_NORM=1
```

Benchmark result:
- tg128 c1: 71.4 tok/s
- tg128 c2: 59.4 tok/s, but only 184/256 requested output tokens generated
- tg128 c4: 61.7 tok/s, but only 386/512 requested output tokens generated
- pp2048 c1: 3077 tok/s
- pp2048 c2: 3123 tok/s
- pp2048 c4: 3118 tok/s

Conclusion: fused strided-norm primitive is locally correct, but the overall `ATLAS_SSM_BATCHED_RECURRENT=1` branch remains a major server regression and appears behaviorally unsafe at C2/C4 (early EOS / shorter outputs). Keep `ATLAS_SSM_BATCHED_RECURRENT` off. This confirms the old batched recurrent loss was not just the extra norm launches or FP32 GDN output traffic.

Next SSM-side direction should not be this branch. If continuing SSM decode work, focus on per-seq primitive efficiency in the baseline path: profile `ATLAS_SSM_DETAIL_PROFILE=1` on `gx10-9959` and target whichever of BA gates / conv / GDN-fused-norm / out_proj dominates under current safe baseline.

## 2026-06-22 - CUTLASS native NVFP4 check on `gx10-9959`

Question answered: prior CUTLASS M0 only tested BF16/dequant GEMM. Native CUTLASS NVFP4 had
not been tested. Tested it now using NVIDIA CUTLASS 4.6 examples copied to `gx10-9959`.

Build:
```bash
cd /home/ms/cutlass
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B /tmp/cutlass-nvfp4-build \
  -DCUTLASS_NVCC_ARCHS=121a \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_TOOLS=ON \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cutlass-nvfp4-build --target 79a_blackwell_geforce_nvfp4_bf16_gemm -j8
cmake --build /tmp/cutlass-nvfp4-build --target 79d_blackwell_geforce_nvfp4_grouped_gemm -j8
```

Important constraint:
- `79a_blackwell_geforce_nvfp4_bf16_gemm` is **NVFP4 x NVFP4 -> BF16**, not W4A16.
- Atlas Holo currently has BF16 activations + NVFP4 weights, so using this class of kernel
  requires quantizing activations to NVFP4 and producing CUTLASS scale-factor layouts.
- A lower-bound standalone BF16->FP4 pack/scale microbench was measured separately; it is not
  exact CUTLASS layout code, but gives the right launch+memory-order cost.

Native NVFP4 dense GEMM exact-ish Holo prefill shapes (`79a`, iterations=30):
- `ssm_qkvz_prefill` M=3537 N=12288 K=2048: `0.703125 ms`, `253188 GFLOP/s`
- `attn_q_prefill` M=3537 N=8192 K=2048: `0.477175 ms`, `248718 GFLOP/s`
- `attn_kv_prefill` M=3537 N=512 K=2048: `0.0414528 ms`, `178942 GFLOP/s`
- `out_prefill` M=3537 N=2048 K=4096: `0.204025 ms`, `290852 GFLOP/s`
- dense MoE proxy gate/up M=28296 N=512 K=2048: `0.287398 ms`, `206476 GFLOP/s`
- dense MoE proxy down M=28296 N=2048 K=512: `0.615019 ms`, `96486.5 GFLOP/s`

Native NVFP4 dense GEMM decode/small-M sweep (`79a`, iterations=100):
- qkvz M=4 N=12288 K=2048: `0.0675421 ms`, `2980.76 GFLOP/s`
- qkvz M=8 N=12288 K=2048: `0.0651309 ms`, `6182.22 GFLOP/s`
- qkvz M=16 N=12288 K=2048: `0.0617571 ms`, `13039.9 GFLOP/s`
- qkvz M=128 N=12288 K=2048: `0.0228096 ms`, `282445 GFLOP/s`
- qkvz M=512 N=12288 K=2048: `0.11546 ms`, `223193 GFLOP/s`
- out M=4 N=2048 K=4096: `0.0271072 ms`, `2475.68 GFLOP/s`
- out M=16 N=2048 K=4096: `0.0247376 ms`, `10851.3 GFLOP/s`
- out M=128 N=2048 K=4096: `0.0205907 ms`, `104294 GFLOP/s`
- out M=512 N=2048 K=4096: `0.0371984 ms`, `230922 GFLOP/s`
- out M=2048 N=2048 K=4096: `0.105494 ms`, `325702 GFLOP/s`

Approximate activation BF16->FP4 pack/scale lower bound:
- M=4 K=2048: `0.004098 ms`
- M=8 K=2048: `0.004104 ms`
- M=16 K=2048: `0.004105 ms`
- M=128 K=2048: `0.005249 ms`
- M=512 K=2048: `0.010246 ms`
- M=2048 K=2048: `0.029629 ms`
- M=4 K=4096: `0.004108 ms`
- M=16 K=4096: `0.004109 ms`
- M=512 K=4096: `0.016420 ms`

Grouped NVFP4 MoE-like small-M sweep (`79d`, iterations=100, `--no_verif`):
- gate/up M=1 N=512 K=2048 groups=16: `0.0348819 ms`, `0.962 TFLOP/s`
- gate/up M=1 N=512 K=2048 groups=32: `0.0471581 ms`, `1.423 TFLOP/s`
- gate/up M=1 N=512 K=2048 groups=64: `0.198529 ms`, `0.676 TFLOP/s`
- gate/up M=4 N=512 K=2048 groups=16: `0.0308653 ms`, `4.349 TFLOP/s`
- gate/up M=4 N=512 K=2048 groups=32: `0.0430803 ms`, `6.231 TFLOP/s`
- down M=1 N=2048 K=512 groups=16: `0.0459126 ms`, `0.731 TFLOP/s`
- down M=1 N=2048 K=512 groups=32: `0.0718867 ms`, `0.934 TFLOP/s`
- down M=1 N=2048 K=512 groups=64: `0.181971 ms`, `0.738 TFLOP/s`
- down M=4 N=2048 K=512 groups=16: `0.0454851 ms`, `2.951 TFLOP/s`
- down M=4 N=2048 K=512 groups=32: `0.0722326 ms`, `3.716 TFLOP/s`
- gate/up M=16 N=512 K=2048 groups=32: `0.0449203 ms`, `23.903 TFLOP/s`
- down M=16 N=2048 K=512 groups=32: `0.0726979 ms`, `14.770 TFLOP/s`

CUTLASS MoE examples note:
- `examples/92_blackwell_moe_gemm/*` are CMake-gated on `CUTLASS_NVCC_ARCHS MATCHES 100a`,
  so the FP4 MoE examples are B200/SM100 targets, not DGX Spark/SM121 targets.
- The usable GB10 native-FP4 examples are the `79_blackwell_geforce_gemm` kernels.

Conclusion:
- Native CUTLASS NVFP4 absolutely shows the missing large-M prefill GEMM ceiling on GB10:
  ~180-290 TFLOP/s on Holo projection shapes, materially above cuBLAS BF16 and far above
  Atlas's hand kernels.
- It does **not** directly fix C=4 decode. At M=4/8/16, fixed overhead dominates, and adding
  activation quantization makes qkvz decode roughly `0.066-0.072 ms` just for that projection
  class. That is not a route to 145 tok/s C4.
- Classic grouped NVFP4 also does **not** solve decode MoE at Holo routing density. M=1/4 per
  expert remains very low throughput; only M=16 starts to become plausible, and C=4 avg
  per-expert occupancy is far below that.

Next useful CUTLASS direction:
1. For **prefill**, build a real Atlas host wrapper for `NVFP4 x NVFP4 -> BF16` dense GEMM
   plus activation NVFP4 quant/scale layout. Start with SSM qkvz and attention q/o.
2. For **decode C=4**, do not spend time on classic grouped GEMM. If CUTLASS helps, it needs
   a custom fused-GEMV/fused-MoE decode design or a persistent small-M design, not `79d` as-is.

## 2026-06-22 - Native CUTLASS NVFP4 dense prefill integration: WORKING + +22% prefill

Implemented opt-in native CUTLASS NVFP4 dense projection path behind:
```bash
ATLAS_CUTLASS_NVFP4_GEMM=1
CUTLASS_HOME=/home/ms/cutlass
```

Files:
- `crates/spark-runtime/cuda/cutlass_nvfp4_gemm.cu`
- `crates/spark-runtime/src/cutlass.rs`
- `crates/spark-runtime/build.rs`
- `crates/spark-model/src/layers/ops.rs`
- `crates/spark-model/src/layers/qwen3_ssm/trait_prefill_proj.rs`
- `crates/spark-model/src/layers/qwen3_attention/prefill/paged_qkv.rs`
- `crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_qkv.rs`
- `crates/spark-model/src/layers/qwen3_attention/prefill/paged_oproj.rs`

What it does:
- Host-callable CUTLASS SM120/SM121 wrapper for `NVFP4 x NVFP4 -> BF16`.
- Per-call activation pack: BF16 `[M,K]` -> CUTLASS NVFP4 packed activation + CUTLASS scale layout.
- Native NVFP4 checkpoint path: consumes Atlas transposed NVFP4 `[K/2,N]` + `[K/16,N]`, repacks scales
  into CUTLASS layout in the wrapper.
- Holo-real FP8 projection path: dequant FP8 block-scaled weights to BF16 using existing cache, then pack
  once into Atlas-transposed NVFP4 `[K/2,N]` + `[K/16,N]`, cache that packed weight, and feed the same
  CUTLASS native NVFP4 wrapper with `weight_scale_2=1.0`.
- Wired targets:
  - SSM qkvz
  - attention Q
  - attention O

Build flags needed:
```bash
CUTLASS_HOME=/home/ms/cutlass CUDA_HOME=/usr/local/cuda \
LD_LIBRARY_PATH=/home/ms/nccl/build/lib LIBRARY_PATH=/home/ms/nccl/build/lib \
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
cargo build --release --bin spark
```

Important build detail:
- CUTLASS native FP4 templates require `--expt-relaxed-constexpr`; added to the runtime CUTLASS nvcc step.

Remote test on `gx10-9959`:
```bash
ATLAS_HOLO_GPU_UTIL=0.80 ATLAS_HOLO_MAX_SEQ_LEN=8192 ATLAS_HOLO_MAX_SEQS=4 \
ATLAS_HOLO_MAX_BATCH=4 ATLAS_HOLO_MAX_PREFILL=8192 ATLAS_CUTLASS_NVFP4_GEMM=1
```

First pass:
- tg128 c1: `70.1 tok/s`
- tg128 c2: `54.9 tok/s`
- tg128 c4: `84.6 tok/s`
- pp2048 c1: `3794 tok/s`
- pp2048 c2: `3805 tok/s`
- pp2048 c4: `3810 tok/s`

Second pass:
- tg128 c1: `70.4 tok/s`
- tg128 c2: `58.9 tok/s`
- tg128 c4: `84.1 tok/s`
- pp2048 c1: `3774 tok/s`
- pp2048 c2: `3815 tok/s`
- pp2048 c4: `3808 tok/s`

Comparison:
- Prior safe baseline in this clamp was roughly pp2048 c4 `3100-3135 tok/s`.
- Native CUTLASS NVFP4 qkvz+Q+O gives pp2048 c4 `~3808-3810 tok/s`, about **+21-23%**.
- Decode remains flat (`c4 ~84 tok/s`), as expected; this is a large-M prefill path.
- Spark memory reported by `nvidia-smi` after bench: about `59808 MB`, within the user's ~65GB realistic free
  target but close enough to track.
- No runtime CUTLASS/CUDA errors found in `/tmp/holo-cutlass-nvfp4-fp8pack.log`.

Remote cleanup note:
- `gx10-9959` had a stale old `crates/spark-server/src/reasoning_parser.rs` conflicting with the current
  `reasoning_parser/mod.rs` module tree. Local repo only has the module tree. Removed the stale remote file
  so the remote build could compile.

Next useful follow-up:
1. Add instrumentation/log-once for `ATLAS_CUTLASS_NVFP4_GEMM` routing so future benches can prove which
   projections are active without inferring from throughput.
2. Extend native NVFP4 dense prefill to attention K/V if the small N=512 shape does not regress; benchmark
   carefully because K/V were not the big win in the standalone sweep.
3. Consider SSM out_proj next only after checking its call-site currently has FP8/NVFP4 source weights;
   standalone native FP4 out_proj shape was fast, but end-to-end benefit depends on whether activation pack
   plus extra quant noise pays off.

## 2026-06-22 - Native CUTLASS NVFP4 expansion + memory pass

Follow-up workflow with subagents covered dense target expansion, memory, decode, and correctness risks.

Code changes after the initial qkvz/Q/O native-FP4 win:
- Added log-once route instrumentation: `CUTLASS_NVFP4_ROUTE ... M/N/K`.
- Expanded attention prefill Q-only native-FP4 branch to Q/K/V in both paged and cache-skip paths.
- Added SSM out-proj native-FP4 override behind separate explicit flag:
  `ATLAS_CUTLASS_NVFP4_SSM_OUT=1`.
- Fixed activation pack scale consistency: activation nibbles now quantize against the decoded stored UE4M3 scale.
- Fixed native NVFP4 `weight_scale_2` semantics: no longer bakes FP32 global scale into UE4M3 per-block scale; applies it as CUTLASS epilogue alpha.
- Reduced memory for FP8-origin weights: FP8 -> BF16 temporary for NVFP4 packing is now uncached and freed after the persistent NVFP4 pack is built.
- Skipped unused attention FP8 prefill transposes when `ATLAS_CUTLASS_NVFP4_GEMM=1`.

Remote results on `gx10-9959`, clamp:
```bash
ATLAS_HOLO_GPU_UTIL=0.80 ATLAS_HOLO_MAX_SEQ_LEN=8192 ATLAS_HOLO_MAX_SEQS=4 \
ATLAS_HOLO_MAX_BATCH=4 ATLAS_HOLO_MAX_PREFILL=8192 \
ATLAS_CUTLASS_NVFP4_GEMM=1 ATLAS_CUTLASS_NVFP4_SSM_OUT=1
```

After Q/K/V/O + qkvz + SSM out + memory fixes + transpose skip:
- Pass 1:
  - tg128 c1/c2/c4: `72.1 / 57.1 / 81.2 tok/s`
  - pp2048 c1/c2/c4: `3876 / 3918 / 3922 tok/s`
  - Spark memory during capture: `57391 MB`
- Pass 2:
  - tg128 c1/c2/c4: `71.9 / 56.9 / 80.8 tok/s`
  - pp2048 c1/c2/c4: `3897 / 3905 / 3910 tok/s`
  - Spark memory after warm pass: `58028 MB`

Route logs confirmed active projections at M=2820:
- `ssm_qkvz_fp8pack M=2820 N=12288 K=2048`
- `ssm_out_fp8pack M=2820 N=2048 K=4096`
- `q_proj M=2820 N=8192 K=2048`
- `k_proj M=2820 N=512 K=2048`
- `v_proj M=2820 N=512 K=2048`
- `attn_o M=2820 N=2048 K=4096`

Comparison:
- Safe baseline before native-FP4 dense work: pp2048 c4 roughly `3100-3135 tok/s`.
- Initial qkvz/Q/O native FP4: pp2048 c4 `~3808-3810 tok/s`, memory about `59808 MB`.
- Expanded + memory pass: pp2048 c4 `~3910-3922 tok/s`, memory `~57.4-58.0 GB`.
- Net: about `+25%` prefill vs safe baseline and `~1.8-2.4 GB` lower than the first native-FP4 implementation.
- Decode remains flat/slightly noisy around `81-84 tok/s`; this work is still a large-M prefill path, not a C4 decode fix.

Correctness/quality caveat:
- This path is W4A4 for FP8-origin projection weights and can diverge from the previous W8A8/BF16-ish path. Throughput is proven; quality is not yet proven.
- Before making it default, run deterministic temp-0 prompt/logit comparisons and projection op-dump cos/max_abs for qkvz, q_proj, o_proj, and SSM out.

Next decode workflow from subagent:
1. Profile current safe baseline C4 decode with `ATLAS_SSM_MS_PROFILE=1` / `ATLAS_SSM_DETAIL_PROFILE=1` or a short nsys slice.
2. If dense projections remain material, build real batch4 decode projection kernels before touching MoE again.
3. Only start purpose-built MoE `forward_k4` if profile shows MoE >25% of C4 wall.

## 2026-06-22 - Native CUTLASS NVFP4 quality isolation

Reason for this pass: full native-FP4 dense replacement was fast but generated garbage, so the path is not deployable without isolating the bad projections.

Implemented granular route flags in addition to the old all-on flag:

- `ATLAS_CUTLASS_NVFP4_GEMM=1` still enables all native-FP4 dense routes for perf experiments.
- `ATLAS_CUTLASS_NVFP4_QKVZ=1` enables only SSM qkvz.
- `ATLAS_CUTLASS_NVFP4_ATTN_Q=1` enables only attention Q.
- `ATLAS_CUTLASS_NVFP4_ATTN_KV=1` enables attention K/V.
- `ATLAS_CUTLASS_NVFP4_ATTN_O=1` enables attention O.
- `ATLAS_CUTLASS_NVFP4_SSM_OUT=1` remains explicit-only for SSM out; it is no longer implicitly enabled by the all-on helper.

Files touched for the isolation switches:

- `crates/spark-model/src/layers/ops.rs`
- `crates/spark-model/src/layers/qwen3_ssm/trait_prefill_proj.rs`
- `crates/spark-model/src/layers/qwen3_ssm/trait_prefill_helper.rs`
- `crates/spark-model/src/layers/qwen3_attention/prefill/paged_qkv.rs`
- `crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_qkv.rs`
- `crates/spark-model/src/layers/qwen3_attention/prefill/paged_oproj.rs`

Remote build on `gx10-9959` passed:

```bash
CUTLASS_HOME=/home/ms/cutlass CUDA_HOME=/usr/local/cuda \
LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64 \
LIBRARY_PATH=/home/ms/nccl/build/lib \
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
cargo build --release --bin spark
```

Important stable-stack note: the server logs `SSM decode dtype: f32 (full precision)`. This matches the vLLM-side expectation that the SSM path is numerically sensitive. Do not replace SSM qkvz/out with W4A4 unless an op-level numerical comparator proves it safe.

Quality probe results on `gx10-9959` with `/tmp/holo_quality_probe.py`:

- Baseline/no native-FP4 dense: coherent.
- `ATLAS_CUTLASS_NVFP4_GEMM=1`: corrupt.
- `ATLAS_CUTLASS_NVFP4_GEMM=1` without SSM out: corrupt.
- `ATLAS_CUTLASS_NVFP4_QKVZ=1`: corrupt. Routes only `ssm_qkvz_fp8pack`; outputs repeat digits/Chinese fragments.
- `ATLAS_CUTLASS_NVFP4_ATTN_Q=1`: corrupt. Routes only `q_proj`; outputs repeat quotes/digits/dots.
- `ATLAS_CUTLASS_NVFP4_ATTN_KV=1`: coherent on the short probe. Routes `k_proj` and `v_proj`.
- `ATLAS_CUTLASS_NVFP4_ATTN_O=1`: coherent on the short probe. Routes `attn_o`.
- `ATLAS_CUTLASS_NVFP4_ATTN_KV=1 ATLAS_CUTLASS_NVFP4_ATTN_O=1`: corrupt despite each route passing alone, so stacked native-FP4 replacements are not safe.

O-only benchmark, standard 8K/C4 clamp, Spark memory about `57433 MB`:

```text
tg128:  c1 69.0 tok/s, c2 54.7 tok/s, c4 83.7 tok/s
pp2048: c1 3177 tok/s, c2 3198 tok/s, c4 3186 tok/s
```

Conclusion: native CUTLASS NVFP4 has a real perf path in isolation, but it is not currently a usable end-to-end speed lever. The broad route gives ~3.9k pp2048 c4 but corrupts output. O-only is quality-clean but effectively baseline-class for prefill. KV-only is quality-clean but too small to expect material speedup. The next useful step is not more server benchmarking; it is an op-level numeric comparator for the native CUTLASS wrapper and packed layout.

Next concrete task:

1. Build a projection-level comparator on `gx10-9959` for the CUTLASS NVFP4 wrapper: compare native `nvfp4_gemm_bf16_act_weight_t` output against a decoded BF16 reference using the same packed A/B/scales for shapes `q_proj M~2820 N=8192 K=2048`, `k/v M~2820 N=512 K=2048`, `attn_o M~2820 N=2048 K=4096`, and `ssm_qkvz M~2820 N=12288 K=2048`.
2. If comparator fails on qkvz/Q but passes KV/O, fix layout/scale handling before any further perf work.
3. If comparator passes but generation still corrupts, treat W4A4 activation quantization as too lossy for qkvz/Q and keep the safe BF16/cuBLAS path there.

## 2026-06-22 - NVFP4 op-level comparator: BUG, not W4A4 loss (DECISIVE)

Built the comparator as an ignored unit test (no remote needed; ran on local `dgx-00` GB10):
`crates/spark-runtime/src/cutlass.rs::tests::cutlass_nvfp4_projection_numeric_comparator`.

For each shape it compares three GEMMs over identical inputs at M=128:
- `out_cutlass`: the native CUTLASS NVFP4 kernel (`nvfp4_gemm_bf16_act_weight_t`).
- `out_ref`: a host W4A4 dequant reference. Weight side reads the **exact** `packed_t`
  nibbles + `scale_t` (e4m3) bytes the kernel consumes (bit-faithful to its weight
  operand); activation side replicates the wrapper's per-16-group `max/6 -> e2m1` quant.
- `out_true`: full unquantized BF16 GEMM.

Run:
```bash
CUTLASS_HOME=/home/ms/cutlass CUDA_HOME=/usr/local/cuda \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/ms/nccl/build/lib LIBRARY_PATH=/home/ms/nccl/build/lib \
cargo test -p spark-runtime --release --features cuda \
  cutlass_nvfp4_projection_numeric_comparator -- --ignored --nocapture
```

Results (all four shapes):
```
ssm_qkvz N=12288 K=2048  cos(cutlass,ref)=0.0026  cos(cutlass,true)=0.0024  cos(ref,true)=0.9901
attn_q   N=8192  K=2048  cos(cutlass,ref)=-0.0004 cos(cutlass,true)=-0.0007 cos(ref,true)=0.9901
attn_kv  N=512   K=2048  cos(cutlass,ref)=0.0087  cos(cutlass,true)=0.0079  cos(ref,true)=0.9902
attn_o   N=2048  K=4096  cos(cutlass,ref)=0.0008  cos(cutlass,true)=0.0005  cos(ref,true)=0.9901
```

CONCLUSIONS (overturns the prior server-probe read):
1. **It is a layout/scale BUG, not W4A4 loss.** `cos(cutlass,ref)≈0` AND `cos(cutlass,true)≈0`
   on **every** shape — the kernel output is uncorrelated with what its own packed operands
   imply. Magnitudes are sane (max_abs ~9-14 vs ref_rms ~1.5), so it's scrambling, not NaN/overflow.
2. **W4A4 itself is fine.** `cos(ref,true)=0.990` for all shapes (incl. K=4096) — the ideal
   W4A4 GEMM is a faithful ~0.99 approximation. So if the wrapper bug is fixed, native NVFP4
   is a quality-acceptable path, and the +21-25% prefill becomes a REAL win (currently it
   computes garbage).
3. **The prior "qkvz/Q corrupt, KV/O coherent" isolation was a false signal.** KV-only and
   O-only are *also* numerically broken (cos≈0); they only looked coherent on a short probe
   because those projections are small/robust enough that scrambled values still produced
   vaguely-coherent short text. The bug is NOT N-dependent.
4. **The reference + weight pack are validated.** `cos(ref,true)=0.99` proves my packed-weight
   readback decode and the runtime's `pack_bf16_weight_to_nvfp4_t` are both correct, AND that
   CUTLASS reads `weight_packed_t` directly with `stride_b`. So the fault is localized to
   `atlas_cutlass_nvfp4_gemm_bf16_act_weight_t` itself: the SFA/SFB scale-factor write layout
   (`layout_sfa(row, base, 0)` / `layout_sfb(col, group*16, 0)`), the in-wrapper activation
   pack into the A workspace, or the ColumnMajor `stride_b` interpretation of Atlas's `[K/2,N]`
   packed weight. The CUTLASS SM120 blockscaled SF layout is a swizzled atom layout, not a
   plain `[M,K/16]` — passing scalar `(coord, k_element, 0)` indices is the prime suspect.

NEXT (bug localization, ranked):
- a) Verify the SF layout: write a tiny K=16 (single group), scales=1.0, identity-ish case so
  the SF layout collapses; if that GEMM matches, the bug is purely in the SFA/SFB swizzle write.
- b) Validate `stride_b`/ColumnMajor vs Atlas `[K/2,N]` packed layout independently (a
  weight-only identity-activation probe).
- c) Cross-check against a known-good CUTLASS example (`79a_blackwell_geforce_nvfp4_bf16_gemm`)
  SF-builder to see how scales must be laid out, and mirror it.
Until fixed, native NVFP4 is OFF by default and must stay off — it was producing garbage even
where the server output looked plausible. The cuBLASLt bf16 +17% remains the shipped prefill win.

## 2026-06-22 - NVFP4 bug ROOT-CAUSED + FIXED (weight B transposed) — op-level validated

Localized the bug with a host-only CUTLASS layout probe (`/tmp/nvfp4_layout_probe.cu`,
compiled vs CUTLASS, prints element offsets for A/B/SFA/SFB and compares to the wrapper's
manual indexing). Result:
- **A data OK** (RowMajor, offset m*K+k matches). **SFA/SFB OK** (written through CUTLASS's
  own `layout_sfa/sfb`, self-consistent).
- **Weight B data was TRANSPOSED.** CUTLASS ColumnMajor B(N,K) wants element (n,k) at offset
  `n*K + k` => byte `n*(K/2)+k/2`, i.e. **`[N, K/2]` K-contiguous**. Our pack emitted Atlas's
  **`[K/2, N]`** (N-contiguous). E.g. `B(n=1,k=0)`: CUTLASS wants byte-elem 128, wrapper put it
  at 2; `B(n=0,k=16)`: wants 16, wrapper put it at 4096. Fully scrambled -> the cos≈0.

FIX (two feeding paths, both went through the same broken assumption):
- **Path 2 / FP8-origin (Holo `ssm_qkvz`, `ssm_out`):** changed the pack kernel
  `atlas_cutlass_pack_bf16_weight_nvfp4_t` (cuda/cutlass_nvfp4_gemm.cu) to emit `[N,K/2]`
  (`packed_t[col*(k/2) + base/2 + i/2]`). The FP8 pack buffer is consumed ONLY by the CUTLASS
  wrapper (verified), so this is safe.
- **Path 1 / native nvfp4 checkpoint (attn q/k/v/o; SSM `*_nvfp4` if present):** the checkpoint
  weight is `[K/2,N]` and shared with the hand kernels, so it can't be relayed. Added a cached
  **byte transpose** `[K/2,N]->[N,K/2]` (`atlas_cutlass_transpose_nvfp4_packed_kton` kernel +
  `cutlass::transpose_nvfp4_packed_kton` + `ops::cutlass_nvfp4_weight_transposed_cached`, keyed
  by weight ptr like the FP8 cache). `cutlass_nvfp4_proj` now takes `gpu` and feeds the
  transposed buffer; 5 call sites updated to pass `ctx.gpu`. Scales (`[K/16,N]`) are unchanged —
  already correct (in-wrapper SFB repack reads them identically for both paths).

VALIDATION (op-level, local `dgx-00`, `cargo test -p spark-runtime --release --features cuda
cutlass::tests::cutlass_nvfp4 -- --ignored --nocapture`):
- Comparator after fix (all four shapes): **cos(cutlass,ref) 0.00 -> 0.996**, and
  **cos(cutlass,true)=0.990 == cos(ref,true)=0.990** — CUTLASS now computes correct W4A4 at the
  ideal-quantization ceiling (residual 0.996 is just SF rounding in the host ref, not error).
- `cutlass_nvfp4_transpose_is_bit_exact`: device transpose reproduces the golden `[N,K/2]` pack
  byte-for-byte.
- Full Holo `spark` binary builds clean with CUTLASS flags (ops.rs signature + 5 call sites).

NET: the native-NVFP4 prefill path is now **numerically correct at W4A4 quality (cos 0.99 vs
true bf16)**, so the previously-measured **+21-25% prefill becomes a REAL win** instead of
garbage. Comparator + transpose tests are committed as ignored unit tests for regression.

NEXT (server validation — needs a free GPU box, e.g. `gx10-9959`; do NOT run on `dgx-00`, it
has a vLLM instance using ~42GB):
1. Quality probe (`/tmp/holo_quality_probe.py`) with `ATLAS_CUTLASS_NVFP4_GEMM=1
   ATLAS_CUTLASS_NVFP4_SSM_OUT=1` (now expected COHERENT for all routes, incl. the stacked
   case that was corrupt before).
2. pp2048 bench under the 8K/C4 clamp; expect ~3.9k c4 (+25%) and decode flat.
3. If coherent + faster, consider promoting native NVFP4 from opt-in to a default prefill path
   (with a temp-0 golden-logit diff vs the bf16/cuBLAS baseline first, since it is W4A4).

## 2026-06-23 - NVFP4 fix SERVER-VALIDATED on gx10-9959: COHERENT + +24% prefill

Synced the 8 changed files dgx-00 -> gx10-9959 (rsync -R), rebuilt the Holo `spark` binary
there (clean), launched with the 8K/C4 clamp + `ATLAS_CUTLASS_NVFP4_GEMM=1
ATLAS_CUTLASS_NVFP4_SSM_OUT=1` (CUTLASS_HOME + LD_LIBRARY_PATH=/usr/local/cuda/lib64 inherited
through the launcher's `env`). Route confirmed active ("Skipping attention FP8 prefill
transposes because ATLAS_CUTLASS_NVFP4_GEMM=1").

**Quality probe (`/tmp/holo_quality_probe.py`) — COHERENT on all 4 prompts** (was CORRUPT with
this exact config before the fix): reasoning sentence clean, math `17*23+91 = 482` (correct),
compact JSON correct, 3-bullet summary correct. The W4A4 native path now produces correct text.

**Bench (`scripts/bench_holo_atlas.py`, 8K/C4 clamp):**
```
pp2048 c1/c2/c4 = 3855 / 3885 / 3882 tok/s   (safe baseline ~3100-3135 => +24%)
tg128  c1/c2/c4 = 71.9 / 58.6 / 85.7 tok/s   (decode flat, as expected — prefill-path win)
ttft_max 0.7s across C.
```

NET: the native-NVFP4 dense prefill bug is fully resolved. The +24% prefill (≈85% of vLLM's
4540 c1) is now a REAL, coherent win, not garbage. Server torn down afterward (shared box).
Banked prefill state: ~3.88k c4 coherent (was ~3.13k cuBLAS-bf16 default). 

Remaining before making it the DEFAULT (currently still opt-in behind the flags):
- temp-0 golden-logit diff vs the bf16/cuBLAS baseline over a few prompts to quantify the W4A4
  drift end-to-end (op-level was cos 0.99/projection; want whole-model logit agreement).
- longer-context coherence pass (probe was short prompts).
- commit the working-tree changes (still uncommitted on `feature/holo-port-pr177`).
Decode C=4 (85 vs vLLM 145) is untouched by this — still the open north-star lever (SSM-side).

## 2026-06-23 - NVFP4 temp-0 golden comparison vs vLLM + cuBLAS: quality CONFIRMED safe

3-way temp-0, thinking-disabled, greedy comparison over 8 varied prompts with token logprobs:
- **vLLM `holo3.1` (Hcompany/Holo-3.1-35B-A3B-NVFP4)** running on dgx-00 `127.0.0.1:8008` = gold ref.
- **Atlas cuBLAS bf16** (`ATLAS_CUBLAS_GEMM=1`, W16A16) = high-precision Atlas ref.
- **Atlas native NVFP4** (`ATLAS_CUTLASS_NVFP4_GEMM=1 ATLAS_CUTLASS_NVFP4_SSM_OUT=1`, W4A4) = candidate.
Probes: `/tmp/golden_logit_probe.py` (param base_url+model), `/tmp/golden_compare.py` (3-way),
`/tmp/atlas_logit_diff.py` (Atlas-vs-Atlas per-token logit drift).

Content agreement with vLLM gold: **cuBLAS 4/8, NVFP4 3/8** — comparable. NVFP4's non-matches
are all benign: `"Canberra."` vs `"Canberra"` (trailing period), `"sunny"` vs `"Sunny"`, a
different-but-correct closed-form for the code task, phrasing in the reason sentence (where
NVFP4 actually matched vLLM and cuBLAS didn't). instr/longish/math2 identical across all three.

Atlas cuBLAS-vs-NVFP4 per-token logit drift (same stack/tokenizer — the clean W4A4-vs-W16A16
signal): **mean |Δlogprob| = 0.039 nats** on agreed tokens; several prompts **bit-identical**
greedy output (json 21/21, longish 40/40, instr 5/5, math2 2/2). Where they diverge it's a
single low-margin greedy flip (e.g. "increasing"->"allowing") that then cascades — both
branches coherent and correct. (Cross-stack token-string LCP reads 0% — a tokenizer-encoding
artifact between vLLM and Atlas, not a real signal; content + same-stack logit delta are the
valid metrics.)

NOTE on the `math` prompt (vLLM 482 / cuBLAS 487 / NVFP4 400, all greedy): thinking was OFF
here, and BOTH Atlas paths get it wrong & differ -> direct no-CoT arithmetic is unreliable,
NOT a precision signal. With thinking ON (production default) NVFP4 returned the correct 482
(earlier quality probe).

VERDICT: **W4A4 native NVFP4 is quality-safe.** Logit drift vs the high-precision cuBLAS path is
~0.04 nats (negligible, comparable to the cuBLAS<->vLLM gap); all outputs coherent and correct;
divergences are benign greedy branch flips. Combined with the +24% prefill, native NVFP4 is a
good candidate to promote from opt-in to DEFAULT prefill. Residual de-risk before flipping the
default (optional, lower priority): a long-context + thinking-ON reasoning eval, and a final
commit of the working tree.

## 2026-06-23 - NVFP4 made DEFAULT + workload-matched benchmark (commit 75f974d)

NVFP4 wired as the launcher default: `scripts/holo_serve.sh` now sets
`ATLAS_CUTLASS_NVFP4_GEMM=1 ATLAS_CUTLASS_NVFP4_SSM_OUT=1` + `LD_LIBRARY_PATH`
(overridable; binary must be built with CUTLASS_HOME). Whole session committed as 75f974d
(96 files, NVFP4 fix + default + accumulated R&D checkpoint).

Workload-matched bench (`/tmp/workload_bench.py`) mirroring real traffic: ~10.3K-token prompt /
short out (thinking ON) + ~6.6K-token JSON fact-extraction. gx10-9959, GPU_UTIL=0.85,
seq 16384, shared box (other tenant ~26GB). Median, c1:

| mode (10.3K prompt, thinking on) | prefill tok/s | TTFT | extract prefill | decode c1 |
| hand W8A8 (PRIOR prod default)   | 3018 | 3.42s | 3069 | 50.1 |
| cuBLAS bf16 (`ATLAS_CUBLAS_GEMM`)| 3450 | 2.99s | 3506 | 50.5 |
| **NVFP4 (NEW default)**          | **3600** | **2.87s** | **3673** | 49.0 |
| vLLM holo3.1 (ref, LIVE traffic) | ~2400* | ~4.3s* | 2051* | **254.7** |

\*vLLM under live traffic + partial TTFT capture on the long shape — indicative only.

KEY NUANCE: the headline "+24%" was vs the **hand-W8A8** path (the actual prior launcher default —
cuBLAS was never enabled in `holo_serve.sh`). So **NVFP4 vs prior production = +19% prefill /
-16% TTFT at 10.3K** (and +24% at the shorter pp2048). **NVFP4 vs cuBLAS bf16 = only ~4-5%** (the
GEMM share shrinks at long context as the hardware-limited GDN scan + flash grow). NVFP4's extra
edge over cuBLAS is memory (4-bit weights vs cuBLAS's ~2GB bf16 dequant) + it's the native path.

Quality at the real workload (NVFP4, thinking on, 10.3K ctx): **perfect** — retrieved all 3
planted FACT lines verbatim; extraction `{"total_nodes":140,"down_count":47,"regions":["ap","eu",
"us"],"max_mem_gb":63}` all fields correct (vLLM gave down_count 70, further from truth ~46).

DECODE is the standing gap: all Atlas modes ~50 tok/s c1 here vs **vLLM 254** — the known
decode-concurrency lever (SSM-side), untouched by this prefill work.

MEMORY WATCH: NVFP4-default needed GPU_UTIL 0.85 (server ~62GB; OOM'd at 0.75 with the 26GB
co-tenant). NVFP4 keeps redundant weight copies (native path: checkpoint `[K/2,N]` + transposed
`[N,K/2]`; fp8 path: fp8 original + packed nvfp4). Freeing the unused original once the CUTLASS
copy is built would reclaim a few GB toward the <65GB target — a clean next optimization.

## 2026-06-23 - pushed (aedd4a1), vision verified, memory-trim investigated, NVFP4 deployed

- **Pushed** to origin: `0b32505..75f974d` (NVFP4 default) then `..aedd4a1` (predequant guard).
- **Memory trim — investigated, mostly a no-op for Holo.** Gated attention
  `predequant_for_prefill` under NVFP4 (commit aedd4a1) mirroring the transpose skip, BUT it
  doesn't help Holo: Holo attn weights are FP8 (not NVFP4), so predequant was already a no-op
  and `q_fp8` was never allocated. The SSM `qkvz_fp8`/`out_proj_fp8` CANNOT be freed — decode
  reads them (`trait_decode_batched.rs:130,384`). The ~1GB of nvfp4 packs is intrinsic to the
  prefill speedup. NET: no easy Holo memory win; ~62GB is mostly base weights + KV, within the
  <65GB target on a dedicated box. (The transpose-skip DOES save Holo memory: fp8 transposed
  copies, 10 attn layers.)
- **Vision VERIFIED under NVFP4** (real /tank photos, attn q/k/v/o flow through NVFP4): wizards
  scene described accurately (4 robed figures, hexagonal lantern, floating hex portrait frames,
  golden fantasy); asset sheet OCR'd the title "WEST AFRICAN VILLAGE MARKET & GRANARY PROPS PACK"
  verbatim + identified modular 3D game assets. Both coherent + correct.
- **DECODE CONCURRENCY is now the active track** (item 1): Atlas ~50 tok/s c1 (long ctx) / ~85 c4
  vs vLLM 145-255. This is the dominant latency lever for the real workload (11.7K in / 137 out,
  thinking on). Launching an analysis workflow (grounded in the documented dead-ends: grouped-MoE,
  token-major, atomic-c4, CUDA-graphs, SSM-batched-recurrent all measured losses) to map the C=4
  decode chain per-stage and synthesize a ranked, novel-lever plan.

## 2026-06-23 - NVFP4 attention: WIRED + CORRECT but a LOSS (keep FP8 attn)

Question: "can we use nvfp4 attention?" Answer: it's fully wired (prefill q_nvfp4_t path +
decode w4a16_gemv at attention_forward_oproj.rs:76), gated by `ATLAS_HOLO_NATIVE_FP8_ATTN`
(launcher default =1 -> FP8 attn; ModelOpt ships Holo attention in FP8). Tested
`ATLAS_HOLO_NATIVE_FP8_ATTN=0` (NVFP4 attn) on gx10-9959, 8K/C4:
- Quality COHERENT (math 482, JSON, vision) -> native NVFP4 attn is numerically correct AND
  this validates the path-1 native-NVFP4 transpose (`cutlass_nvfp4_proj`) in production.
- BUT a clear LOSS: decode c1/c2/c4 66.0/51.9/74.8 (vs FP8-attn 70.7/57.9/84.7 = **-6/-10/-12%**);
  prefill c4 **1541 vs ~3880 = -60%** (systemic — NATIVE_FP8_ATTN=0 evidently disrupts more than
  attention via the modelopt mixed-precision routing). Memory ~neutral (~62GB).
- ROOT CAUSE (matches the decode workflow): the dense projections are ISSUE/OCCUPANCY-bound,
  not bandwidth-bound, so 4-bit weights (fewer bytes) don't help; w4a16_gemv is just less
  optimized than the FP8 w8a16 pipelined. Same lesson as ATLAS_HOLO_FP8_SSM_DECODE (fp8 81 > nvfp4 65).
VERDICT: keep `ATLAS_HOLO_NATIVE_FP8_ATTN=1` (FP8 attention). NVFP4 attn works but is slower.

## 2026-06-23 - DECODE CONCURRENCY workflow result (wuzqmr2sb) — the real plan

11-agent grounded analysis. **KEY INSIGHT: decode is NOT bandwidth-walled.** n=4 step ~47ms vs a
~11-13ms weight-bound floor (185 GB/s) => we're 3-4x above floor, dominated by **issue/occupancy
inflation in the dense GEMMs + redundant reads**, not DRAM. Specifically: the SSM QKVZ+out_proj
at n=4 already batch to ONE w8a16_gemm_pipelined launch (weights read once) BUT that kernel
**pads M=4 -> 128-row MMA tile (32x compute over-provision)** and is occupancy/issue-bound by its
own header.

TOP PROTOTYPE (highest gain/effort, NOT a dead-end): **M=4 weight-streaming W8A16 block-scaled
GEMV** (`kernels/gb10/common/w8a16_gemv_batch4.cu`, NEW) replacing the padded MMA for SSM
QKVZ+out_proj at n<=4. Clones the already-shipped & winning `w8a16_gemv.cu` (M=1) +
`dense_gemv_fp8w_batch2.cu` N-accumulator pattern: 4 FP32 accumulators, weight byte dequant'd
ONCE and reused across all 4 rows, one DRAM pass, no 128-row pad, no tensor cores. Files:
new .cu; `layers/ops/fp8_gemv_batch.rs` (w8a16_gemv_batch4 op — batch2 already declared, no
kernel); `qwen3_ssm/trait_decode_multi_seq.rs` dispatch (n<=4 -> batch GEMV, keep pipelined for
n>4); `qwen3_ssm/init.rs` kernel handle. Est **5-12% c4**, medium effort, low numeric risk
(identical reduction order; cos>=0.9999 microtest + 8K/C4 A/B). Reusable for attention QKV/O.

STACKING LEVERS (ranked, all non-dead-end): 2) same M=4 GEMV for attn QKV/O (5-12%); 3) batch
the SHARED-expert GEMV to M=4 (read 4x today in the per-token MoE loop, ~279MB/step redundant,
6-9%); 4) delete the redundant 3rd full FP32 H-state read in the GDN norm-clamp
(gated_delta_rule.cu:179-219, accumulate sum-of-squares from registers, ~252MB/step, small/safe,
4-8%); 5) async argmax D2H so host runs ahead (3-6%); 6) batched argmax+embed (2-4% cleanup).
Full result: tasks/wuzqmr2sb.output. NEXT: implement the top prototype (w8a16_gemv_batch4).

## 2026-06-23 - w8a16_gemv_batch4 SHIPPED: +23% c4 / +31% c2 decode (the prototype WON big)

Implemented the workflow's top prototype and it's the biggest decode win yet.
- NEW kernel `kernels/gb10/common/w8a16_gemv_batch4.cu`: M<=4 weight-streaming block-scaled FP8
  GEMV. Clones w8a16_gemv (M=1) block-scale path + the multi-accumulator pattern; weight byte
  dequant'd (LUT*block_scale) ONCE, MAC'd into M FP32 accumulators. Replaces w8a16_gemm_pipelined
  (which pads M=4 -> 128-row MMA tile, 32× compute over-provision, issue-bound) for n<=4.
- Op `ops::w8a16_gemv_batch4` (fp8_gemv_batch.rs); kernel handle + dispatch in
  qwen3_ssm/{init.rs,mod.rs,trait_decode_multi_seq.rs} for BOTH qkvz (:379) and out_proj (:738),
  gated `ATLAS_SSM_GEMV_BATCH4` (default ON; =0 falls back to pipelined). Same `fp8.weight` +
  `fp8.row_scale` (FP32 [N/128,K/128] block-scale — verified identical to what pipelined reads).
- MICROTEST `examples/w8a16_gemv_batch4_microtest.rs`: batch4(M=4) vs 4× w8a16_gemv(M=1) =
  **cos 1.000000000, max_abs 0.0 (bit-identical)** per row.
- SERVER A/B on gx10-9959 (8K/C4, same build, quality coherent math=482):

  | decode | batch4 OFF | batch4 ON | delta |
  | c1 | 71.8 | 70.9 | flat |
  | c2 | 58.4 | 76.3 | **+31%** |
  | c4 | 86.2 | 106.1 | **+23%** |

  Prefill flat (~3.88k). The c2<c1 inversion is GONE (now monotonic 71->76->106). c4 went from
  59% -> 73% of vLLM's 145. Far beats the workflow's 5-12% static estimate (the M-pad was a
  bigger fraction of the step than modeled). SHIPPED as default. Committed.
- STACKING NEXT (same workflow, all still open): lever 2 (same GEMV for attn QKV/O, 10 layers),
  lever 3 (batch shared-expert GEMV M=4), lever 4 (delete redundant 3rd GDN H-read), lever 5/6
  (async argmax, batched argmax+embed). Each independent and additive.

## 2026-06-23 - Levers 2/3/4 IMPLEMENTED + A/B'd: ALL MARGINAL (do not re-attempt)

Ran a 3-worktree-agent workflow (wck4npdx6) to implement levers 2/3/4 in parallel (each gated +
microtested). Integrated each onto f64505b (3-way apply; worktrees were based on an older commit
so patches needed --3way + manual conflict/visibility fixes) and ran clean same-build server A/Bs
on gx10-9959 (8K/C4). RESULT: all three are within-noise washes on top of lever 1. Reverted all;
tree stays at f64505b (lever 1 only). **The static cost-model estimates were 5-10x optimistic for
the smaller stages — only the SSM projection (lever 1) had a real large reclaimable share.**

- **Lever 2 (attn Q/K/V/O M=4 GEMV, ATLAS_ATTN_GEMV_BATCH4):** quality coherent, bit-identical
  (reuses lever-1 kernel). A/B: c4 111.9 ON vs 110.7 OFF (+1%), c2 83.9 vs 83.4 — NOISE. The
  attention is only 10 layers with smaller weights, and the batched path needs an extra d2d
  scatter into per-seq layout that offsets the weight-read saving. REVERTED.
- **Lever 3 (shared-expert M=4, ATLAS_MOE_SHARED_BATCH4):** new kernel
  moe_shared_expert_batch4_fp8.cu; microtest PASS (gate/up/down cos=1.0, down bit-exact); server
  quality COHERENT (routed-only suppression correct, no double-count). A/B: c4 112.7 ON vs 112.1
  OFF, c2 84.5 vs 85.4 — NOISE. The shared expert is a small weight fraction; reading it 1x vs 4x
  saves ~nothing vs the routed experts + SSM. REVERTED (correct but neutral; not worth the MoE
  complexity on the decode path that has many dead-ends).
- **Lever 4 (GDN 3rd-read deletion, ATLAS_GDN_FUSED_NORM_READ):** the agent edited the COMMON
  gated_delta_rule.cu's `decode`/`decode_f32`, but Holo's hot decode kernel is
  `gated_delta_rule_decode_f32_norm` in the OVERRIDE (qwen3.6-35b-a3b/nvfp4/...). So it's a NO-OP
  for Holo without porting to the override's f32_norm — a sub-1% bit-equivalent gain not worth it.
  NOT APPLIED.

NET decode state: **lever 1 (w8a16_gemv_batch4, +23% c4) is the banked decode win** (c4 ~106-112,
c2 ~76-85, c1 ~71; = ~73-78% of vLLM c4). Levers 2/3/4 are documented-marginal — do not redo.
Remaining workflow levers 5/6 (async argmax D2H, batched argmax+embed) are launch/latency micro-
opts (est 2-6%) and untested; lower priority. The next REAL decode lever would need to attack the
routed-MoE per-token weight streaming or the GDN recurrent occupancy directly — both are in the
documented hard/dead-end territory. Consider decode largely tapped on accessible levers; the
bigger remaining gaps vs vLLM (decode 112 vs 145; memory 62 vs 40GB w/ larger ctx) are structural
(FlashInfer/CUTLASS-grade fused kernels; paged-KV memory efficiency).

## 2026-06-23 - Concurrency scaling + w8a16_gemv_batch16 (high-C decode) + FlashInfer scout

**Scaling (shipped lever-1 binary, 8K):** decode c1 72 / c4 110 / c8 120 / c16 145 (sub-linear).
**Prefill FLAT ~3880 at ALL concurrency** (c1=c16) — ZERO cross-request prefill benefit vs vLLM
scaling 4540->6700->7180 by packing requests into its 16K-token batch. Flat prefill is the biggest
high-concurrency gap, hits the real 11.7K-prompt concurrent workload hardest.

**SHIPPED: w8a16_gemv_batch16 — lever 1 extended to M<=16.** batch4 was gated n<=4 so it turned OFF
at C=8/16 (fell back to M-padded pipelined). Refactored w8a16_gemv_batch4.cu into a
`template<int MAX_M>` device helper + two extern-C globals: `w8a16_gemv_batch4` (M<=4) +
`w8a16_gemv_batch16` (M<=16). Dispatch picks batch4 for n<=4, batch16 for n=5..16, pipelined n>16;
same ATLAS_SSM_GEMV_BATCH4 gate (default on). Microtest extended: batch4@M=4 + batch16@M=8,16 all
cos>=0.99999. Same-build A/B (MAX_SEQS=16, quality coherent math=482):
  c4: 86.6->107.5 (+24%, batch4); c8: 120.3->136.3 (**+13%, batch16**); c16: 142.8->150.2 (**+5%**).
c16 gains less (M=16 register pressure acc[16]+wf[16]; MoE/GDN per-token ×16 dominate). New highs:
c8 136, c16 150. SHIPPED default.

**FlashInfer scout — VERDICT: GO (proven feasible on GB10).** Background agent EMPIRICALLY proved
FlashInfer's `BatchPrefillWithRaggedKVCacheDispatched` (flashinfer/attention/prefill.cuh) is a
FlashAttention-2 SM80-class kernel (mma.sync/ldmatrix/cp.async), NOT FA3/Hopper. COMPILES to
sm_121f SASS (520 HMMA + 160 LDSM + 96 LDGSTS, ZERO wgmma/TMA/tcgen05); isolated LDSM+HMMA probe
RAN on live GB10 (launch_rc=0, finite — "ldmatrix broken" doesn't apply to the m8n8.x4 form used).
head_dim=256 + GQA supported on SM80 path. Headers TORCH-FREE (torch coupling only in csrc/*.cu,
bypassed like CUTLASS's Python). Corroborated by flashinfer #3170 (DGX Spark audit: consumer
Blackwell = FA2 via SM89; only prebuilt trtllm-gen cubins missing, we compile from source).
- INTEGRATION (mirror CUTLASS): new crates/spark-runtime/cuda/flashinfer_ragged_prefill.cu w/ two
  extern-C wrappers `atlas_fi_prefill_plan(...)` + `atlas_fi_ragged_prefill_hd256(q,k,v,q_indptr,
  kv_indptr,o,...,stream)`; build_flashinfer_object in build.rs gated on FLASHINFER_HOME; new
  crates/spark-runtime/src/flashinfer.rs (mirror cutlass.rs); call from qwen3_attention/prefill.
  BUILD WRINKLE: needs FlashInfer's pinned CCCL via -isystem ($FLASHINFER_HOME/3rdparty/cccl/
  {libcudacxx/include,cub,thrust}) BEFORE CUDA-13 toolkit CCCL (CUDA 13 lacks cuda::fast_mod_div;
  vendor the submodule).
- EFFORT ~1 week: wrapper+build.rs+FFI ~1-2d; real work = PrefillPlan plumbing ~3-5d (two-call
  plan/run, workspace sizing, plan-blob scheduler ptrs -> BatchPrefillRaggedParams, build
  q_indptr/kv_indptr from Atlas batched layout, BF16 validate vs paged). Caveat: consumer Blackwell
  halves FP32-accum MMA tput, but the prefill win is BATCHING (kill per-request paged-attn overhead
  that flatlines us at 3880), not peak MMA. THE lever for cross-request prefill scaling. Full detail
  + extern-C sigs: tasks/adfb2c4274c6365fc.output.

## 2026-06-23 - FlashInfer ragged prefill: STAGE 1 BUILT + VALIDATED (FFI works, cos 0.999998)

Built the host-callable FlashInfer ragged/varlen prefill integration end-to-end and proved it
correct in-tree. Inert without FLASHINFER_HOME (atlas_flashinfer cfg off -> bail path), so default
builds are unchanged.
- VENDORED: /home/ms/flashinfer (permanent). GOTCHA: its 3rdparty/cccl submodule was empty/too-old
  (lacks cuda::fast_mod_div); copied a working CCCL (HEAD f150d51, has fast_modulo_division.h) to
  /home/ms/flashinfer/3rdparty/cccl. build.rs needs that CCCL via -isystem BEFORE CUDA-13's CCCL.
- crates/spark-runtime/cuda/flashinfer_ragged_prefill.cu (agent-written): two extern-C fns —
  atlas_fi_ragged_prefill_bf16_hd256 (PrefillPlan -> BatchPrefillRaggedParams<bf16,bf16,bf16,int32>
  -> BatchPrefillWithRaggedKVCacheDispatched<CTA_TILE_Q,256,256,kNone,false,MaskMode,
  DefaultAttention<false,false,false,false>>). MaskMode is a COMPILE-TIME template (dispatches
  causal/none); CTA_TILE_Q via DISPATCH_CTA_TILE_Q over plan_info. Compiles to sm_121f (FA2/SM80
  HMMA+LDSM+LDGSTS).
- build.rs: build_flashinfer_object gated on FLASHINFER_HOME (mirrors build_cutlass_object) ->
  libatlas_flashinfer.a, cfg atlas_flashinfer.
- crates/spark-runtime/src/flashinfer.rs: FFI + safe `ragged_prefill_bf16_hd256(...)`. WORKSPACES:
  the FlashInfer float workspace is a BUDGET PrefillPlan splits within (NOT a computed size — the
  wrapper's size-query returned 552GB nonsense). Use FIXED budgets: float 256MB, int 64MB, pinned
  64MB (cuMemAlloc + cudaHostAlloc, lazy OnceLock). lib.rs registers the module.
- VALIDATION (ignored test flashinfer::tests::flashinfer_ragged_prefill_matches_cpu_reference,
  FLASHINFER_HOME build): 2 ragged requests (6+10 tok), GQA 4qo/2kv, hd=256, causal vs a CPU
  reference -> **worst_cos=0.999998**. The FFI runs + is numerically correct.

KEY DATA-CONTRACT NOTES for STAGE 2 (from the wrapper agent — must hold when feeding real data):
- indptr DUALITY: pass qo/kv indptr BOTH host (PrefillPlan derefs on CPU) AND device (kernel reads).
- STRIDES assume FULLY CONTIGUOUS [rows, heads, head_dim] BF16, no head/row padding. q_stride_n=
  n_qo*256, q_stride_h=256; k/v_stride_n=n_kv*256.
- sm_scale is RAW (1/sqrt(256)=0.0625); kernel applies log2e internally — do NOT pre-multiply.
- total_kv_rows is unused by the wrapper (KV extent from kv_indptr); total_qo_rows -> PrefillPlan
  total_num_rows.

STAGE 2 (next): wire ragged_prefill into the batched prefill attention path
(qwen3_attention/prefill/paged_attn_batched.rs — the slow paged path the dormant co-dispatch fell
back to), build qo/kv_indptr from the co-dispatched requests' seq lens (prefill_b/ batched geometry +
scheduler phase_continue_prefills/run_batched_prefill.rs), produce contiguous [rows,heads,256] Q/K/V
(RoPE applied before), validate vs the single-request path, then re-enable co-dispatch + bench
cross-request prefill scaling (target: prefill that scales with concurrency instead of flat 3880).
Production build will add FLASHINFER_HOME=/home/ms/flashinfer to the recipe once Stage 2 lands.

NOTE (user, 2026-06-23): vLLM serves Holo in ~40GB CUDA mem with a much LARGER context window vs
Atlas ~62GB — a real memory-efficiency gap (paged-KV / weight layout). Memory/context efficiency
is a separate open track from the decode-speed work above.
