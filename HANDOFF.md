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

## Sync note

There is an untracked `docs/porting/holo-3.1-handoff.md` with historical notes; use it only for background. This `HANDOFF.md` is the active handoff for the current continuation.
