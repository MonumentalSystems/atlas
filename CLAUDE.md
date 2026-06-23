# Atlas — Claude working notes

Atlas is a from-scratch inference engine. Active work: **Holo 3.1**
(`Hcompany/Holo-3.1-35B-A3B-NVFP4`) on **DGX GB10** (sm_121, consumer Blackwell).
Hybrid model: 30 GDN/SSM (Mamba-like) + 10 full-attention + 256-expert MoE;
hidden=2048, head_dim=256, 16 q-heads / 2 kv-heads.

## ⚠️ BUILD — use this EXACT command (read before any rebuild)

The build runs on the **remote GB10 host `gx10-9959`** (this session's shell is on
`dgx-00`; the two are **separate filesystems** — edits must be `rsync`'d over).
Package is `spark-server`, bin is `spark`. **Default features build the WRONG
thing** — you must pass the target env, `--no-default-features --features cuda`,
**and** point at the CUTLASS / FlashInfer / NCCL trees. Omitting any of these
produces a binary that *loads* but fails at runtime.

```bash
ssh gx10-9959 'cd ~/atlas && source ~/.cargo/env
  export PATH=/usr/local/cuda/bin:$PATH          # nvcc
  export CUTLASS_HOME=$HOME/cutlass              # else: "CUTLASS support was not built"
  export FLASHINFER_HOME=$HOME/flashinfer        # else: varlen FlashInfer prefill absent
  export RUSTFLAGS="-L/home/ms/nccl/build/lib -L/usr/local/cuda/lib64"  # else: -lnccl not found
  export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4
  cargo build --release -p spark-server --bin spark --no-default-features --features cuda'
```

**Why each flag matters — symptoms if you forget it (all observed 2026-06-23):**
- No `ATLAS_TARGET_MODEL=holo-3.1-35b-a3b` → kernels compile for the default
  `qwen3-next-80b-a3b` target → runtime panic: *"No compiled kernel target matches
  model_type 'holo3_1_moe' / hidden_size=2048"*. Build log must say
  `compiled N kernels for target 0 (gb10, holo-3.1-35b-a3b, nvfp4)`.
- No `CUTLASS_HOME` → runtime ERROR *"CUTLASS support was not built; set CUTLASS_HOME
  when building"* on the first prefill layer (NVFP4 GEMM + co-dispatch route there).
- No `FLASHINFER_HOME` → `cfg(atlas_flashinfer)` off → varlen ragged-prefill path
  silently unavailable.
- No `RUSTFLAGS -L…/nccl…` → link fails: *"cannot find -lnccl"*.
- These env vars are **not** in the login profile, so a non-interactive `ssh` build
  won't pick them up — always export them inline.

Lesson (2026-06-23): a bare `cargo build --release -p spark-server` clobbered a
working binary with a qwen3-next-target / no-CUTLASS build. Always use the block above.

## Serve + benchmark

- Launch: `bash scripts/holo_serve.sh /tmp/holo.log` (uses the prebuilt
  `target/release/spark`; does **not** build). Binds `127.0.0.1:8890`, model name
  `holo3.1-atlas-poc`. Tunables via env: `ATLAS_HOLO_GPU_UTIL`, `ATLAS_HOLO_MAX_SEQS`,
  `ATLAS_HOLO_MAX_PREFILL`, etc.
- Process gotcha: `pgrep -f "release/spark serve --model"` (the `--model` keeps it
  from matching your own polling shells). One server = 2 PIDs (parent+child).
- Bench scripts (run on remote, live in `/tmp`): `single_bench.py <url> <tag>`
  (one 1403-tok req → prefill + decode tok/s), `varlen_bench.py <url> <tag>`
  (4 concurrent *different-length* reqs → aggregate prefill tok/s). Both use
  `max_tokens=160` (realistic gen; 8 was too short).
- **Do not re-run vLLM** (user directive). Baselines: `/home/ms/spark-vllm-docker/results.csv`.

## REAL-workload perf (image + fact-extraction — the bench that matters)

The synthetic pp2048 text bench HID the real bottleneck. On realistic traffic
(image input + 4.5K→1K fact-extraction, /tmp/real_bench.py):
- **The vision ViT dominated image requests** (~5s/image ViT-encode vs ~0.1s for the
  290 LLM tokens). Root: `vision_gemm_bias` was a naive scalar GEMM. FIXED →
  tensor-core (`dense_gemm_bf16_pipelined` + `vision_add_bias`), helper
  `VisionEncoder::vit_gemm_bias` (all 27 blocks + merger + patch_embed). Verified
  correct (Saturn → "a planet with rings, resembling Saturn").
- **Image prefill c1: 5.0s → 0.69s (7.2×). Real mixed c4: ttft 9.96→3.52s, prefill
  808→2284 tok/s (2.8×).** This was FAR more impactful than any LLM-prefill batching
  lever — for image-heavy traffic the ViT is #1. Always bench real shapes.
- Next real-workload levers: vision_attention_rope (the ViT attention, still custom);
  decode of long (1K) fact-extraction outputs; vLLM real-shape baseline (only had
  synthetic pp2048).

## Memory (fit alongside co-tenants; the GB10 is SHARED)

The GB10 (121.6GB unified) is shared — ComfyUI (sage-attn/fp8 image-gen) + o_voxel/
FlexGEMM (the 3D-viewport pipeline) + sparkview use ~36GB baseline. vLLM runs Holo at
`--gpu-memory-utilization 0.34` (~41GB) with 200K context + 16 seqs. Atlas was using
~56GB — the excess blocked large-context/concurrency testing.

- **ROOT CAUSE: MoE "Full prefill copies" (~30GB).** `transpose_for_prefill` +
  `predequant_for_prefill` (helpers_a.rs) keep resident transposed/predequant expert
  copies for fast prefill (`moe_w4a16_fused_gate_up_t_k64`). Gated by
  `FAST_MOE_MODE=full` + `FAST_MOE_LAYERS=0-39`. (Attention already skips its prefill
  transposes via the CUTLASS-NVFP4 work; MoE didn't.)
- **FIX (works): `ATLAS_HOLO_FAST_MOE_MODE=off`** (any non-{gate_up,full,unified}
  value → `holo_fast_moe_mode()` None → `skip_moe_prefill_copies=true`). Saves ~30GB;
  prefill falls to the non-transposed `moe_w4a16_grouped_gemm_ptrtable` path
  (`forward_prefill_routed.rs:168`). VERIFIED correct (image still right). Atlas
  footprint ~42GB. NOTE the serve script's `${VAR:-full}` turns EMPTY into full —
  pass the literal `off`, not "".
- **TRADEOFF: prefill ~1.5× slower** (c1 1704 vs 2662 tok/s) — non-transposed MoE GEMM.
  Recovering this is the open follow-up. ATTEMPTS so far (both negative):
  - Coalescing the non-t kernel's B-load: NO change → it's NOT load-bound.
  - A contained FP8-MMA swap (`moe_w4a16_grouped_gemm_ptrtable_fp8`, m16n8k32 e4m3
    keeping the naive synchronous structure): SLOWER (1227 vs 1704) AND greedy output
    diverged slightly — reverted. LESSON: the transposed kernel's speed is its
    **cp.async pipelining + vectorized dequant + fusion**, NOT just the FP8 MMA. A
    naive FP8 kernel loses to its own load/sync overhead.
  - REAL FIX (substantial): port the FULL transposed `fused_gate_up_t_k64` structure
    (double-buffered cp.async, vectorized dequant, fused gate+up, FP8 MMA) to the
    `[N,K/2]` layout. The hard part is the load: transposed cp.async assumes
    contiguous-N `[K/2,N]`; non-transposed is contiguous-K `[N,K/2]`, so the
    cp.async + smem layout must be restructured (not just the address). Validate
    token-for-token + logits vs the copies-ON path before shipping. Alternative:
    accept the 1.5× (memory matters more) — on-the-fly transpose is WORSE (triples
    weight traffic on a bandwidth-bound prefill).
- **`ATLAS_KV_OVERCOMMIT=1`** → 200K max-seq-len + 16 seqs admits on-demand (276K-token
  pool ≈ vLLM's 300K) instead of the worst-case refusal.
- **2nd issue — budget formula counts co-tenant memory.** factory/build.rs checks
  `total_consumed + reserve ≤ util×total_GPU`; with 36GB co-tenants it needs util≥0.65
  even though Atlas uses ~42GB. vLLM's util is self-relative. To hit vLLM's 0.34, the
  KV-budget check should size against Atlas's own footprint / free memory, not total.
  Also the inference reserve is 10GB at max-batch=16 (5.6GB at 8) — a tunable chunk.
- Working low-mem config: `ATLAS_HOLO_FAST_MOE_MODE=off ATLAS_KV_OVERCOMMIT=1
  ATLAS_HOLO_GPU_UTIL=0.7 ATLAS_HOLO_MAX_SEQ_LEN=200000 ATLAS_HOLO_MAX_SEQS=16`.

## North star

vLLM-parity. **Real apples-to-apples baseline** (vLLM `pp2048` from
`/home/ms/spark-vllm-docker/results.csv`, measured Jun 3; DO NOT re-run vLLM):

| conc | vLLM pp / tg | Atlas pp (OFF) | Atlas tg (decode) |
|------|-------------|----------------|-------------------|
| c1   | 4540 / 75   | ~3700 (**80%**)| **75 (100%)**     |
| c2   | 6090 / 118  | ~3800 (62%)    | 84 (71%)          |
| c4   | — / —       | ~3760          | 117               |
| c5   | 6830 / 151  | (c4 ~3760, 55%)| (c8 = 153)        |
| c10  | 7180 / 196  | —              | —                 |

- **Single-stream: decode is AT PARITY (100%), prefill ~80%.** Blended c1 ≈ 90%.
- Concurrency: per-req decode Atlas (75→42→30→19 at c1/c2/c4/c8) vs vLLM
  (75→59→30→20 at c1/c2/c5/c10) CONVERGE by c4–c5 (~30/req). So decode concurrency is
  fine at high conc (the c16 goal); the gap is only at c2.
- **The real concurrency gap is PREFILL** (flat c1≈c4≈3700, ~55% of vLLM at c5).
  vLLM batches prefill efficiently; Atlas OFF serializes and varlen ON is overhead-
  bound (net loss). Closing it = the batched-prefill overhead rework (eliminate the
  per-token Z-copy ~350k/forward, per-request conv1d/GDN loops, single serial stream,
  alloc churn), and even then it may only MATCH serial unless cross-request kernel
  overlap is added. Highest-leverage but largest concurrency item.
- TOOLING — SOLVED: nsys truncation (only last ~2ms survived) was the default
  flush-on-exit overflowing on long traces. FIX: add **`--cuda-flush-interval=100`**
  to the nsys cmd (see /tmp/holo_serve_nsys_flush.sh) → full capture, confirmed on a
  batched c4 forward. For per-kernel metrics (achieved FLOP/s, tensor-core util, stall
  reasons): `ncu` IS installed (`/usr/local/cuda/bin/ncu`); GPU counters are admin-
  locked (`RmProfilingAdminOnly:1`) but we have passwordless sudo → `sudo ncu --set
  full -k <kernel> --target-processes all <cmd>`.
- BATCHED-PREFILL PROFILE (varlen c4, flush-interval): GPU = GDN `wy64` 45% +
  MoE 23% + dequant/pack 13%; API = 237k cuStreamSynchronize + 151k cuMemAlloc +
  151k HtoD + 85k DtoH (load + per-request churn). Batched path loses on BOTH fronts.
  Batching lever: (1) GDN per-request `wy64` loop → batched FLA (grid
  [num_chunks,nv,batch] fills GPU as batch grows — the M2 work), (2) collapse the
  per-request sync/alloc/copy churn (one stream/one sync per layer, not per-request).

- **Single-stream prefill is already ~80% of vLLM** — 90% is ~1.13×, NOT 2.3×.
  (Earlier "43%/2.3×" was a baseline ERROR: compared a 1403-tok single-stream run
  to vLLM's c5 *aggregate*. Prefill tok/s rises with prompt length; measure at the
  SAME length.)
- **The real gap is concurrency scaling**: Atlas prefill is FLAT (c1≈c2≈c4≈3700-3800)
  while vLLM scales 1.5× (4540→6830). vLLM overlaps/batches prefill; Atlas OFF
  serializes. varlen ON currently makes it WORSE (1985-2033 at c2/c4) because the
  batched path still uses the slow occupancy-limited `wy64` GDN (single-stream uses
  FLA). Keep varlen OFF; the win is (a) single-stream 80→90% via modest kernel
  tuning, and (b) make concurrent prefill actually scale (batched GDN→FLA + overlap).
- Measured (Jun 23): varlen ON at c4 with EXACTLY-uniform prompts is still 2042
  tok/s vs OFF 3756 — the batched forward is ~1.8× SLOWER than running the requests
  serially. So it's NOT the `uniform`-check or wy64-vs-FLA alone: the co-dispatch
  ARCHITECTURE is overhead-bound (per-token Z-copy D2D loops — ~350k calls/forward at
  c4; per-request conv1d/GDN; single serial stream; alloc churn). Making batched
  prefill beat serial is a real rework, not a config flip. OFF itself is flat (no
  overlap — 4 requests run back-to-back). Tractable near-term: single-stream 80→90%
  (the GEMMs); concurrency is higher-ceiling but needs the overhead rework first.

## The concrete single-stream lever: MoE FP8 → FP4 (precision, NOT tiling)

- MoE is the biggest single-stream section (~28%). CORRECTION: it does NOT run BF16 —
  the `moe_w4a16` grouped GEMM stages activations as bf16 in smem but converts them to
  `e4m3` and runs `mma.sync.m16n8k32.f32.e4m3.e4m3` = **FP8×FP8 tensor cores** (4-bit
  weight dequant→fp8, act bf16→fp8). So MoE is effectively W4A8-FP8-compute. The real
  lever is **FP8 → FP4** (~2× FP4-vs-FP8 peak) — a PRECISION change, needs a new FP4
  grouped expert kernel + accuracy validation (FP4 MoE acts may drift; vLLM ships NVFP4
  so plausibly fine). No cutlass FP4 *grouped* GEMM exists yet (only dense).
- TESTED + REJECTED (Jun 23): the `m128` fused-gate-up tile (M_TILE 64→128, same FP8
  math) — ported into Holo, registered, ran correct (identical greedy outputs) but
  NO speedup (3529-3567 vs 3700-3798 baseline; uneven token-per-expert distribution +
  2× smem lowers occupancy). Reverted. Together with the M1 large-M-hoist null result,
  this confirms **tiling/batching tweaks don't move these throughput-limited GEMMs** —
  the gains require precision (FP8→FP4) or fundamentally better kernels.

## Prefill bottleneck map (single-stream FLA path, the goal path)

Measured with `ATLAS_PROFILE=1` (per-section sync+timestamp; nsys can't cleanly
capture the OFF/single-stream forward on this box — it keeps only a ~2 ms tail).
Run a single prefill, then aggregate `SSM prefill [<section>] N=…: …µs` over the
30 SSM layers. Per-SSM-layer split:

- **moe_ffn ≈ 28%**, **qkvz_gemm ≈ 27%** (CUTLASS-NVFP4 proj), **gdn_prefill ≈ 27%**
  (FLA scan), out_proj ≈ 11%, norms/gates ≈ 2.5%.
- **No single 2.3× lever** — it's a 3-way tie (MoE / QKVZ-proj / GDN). Closing the
  vLLM gap needs all three. (The earlier "GDN = 53%" was the *batched* path's slow
  `wy64` kernel, NOT the FLA single-stream path.)
- Sub-levers: GDN-FLA spine `chunk_delta_h` is 52% of GDN and occupancy-limited
  (`grid=[nv=32,batch]`, serial over chunks); QKVZ NVFP4 GEMM ≈16 TFLOP/s (low,
  headroom); MoE `_m128` fused gate+up variant is unregistered (faster large-M tiling).
- Batched/varlen GDN still uses `wy64` (occupancy-limited) while single-stream uses
  FLA — same stale-kernel pattern as the (now fixed) dense_gemm bug; bringing the
  batched path onto FLA would help varlen but not the OFF goal path.

## The crux: GEMM throughput is ~16 TFLOP/s and we ALREADY use true FP4

- M1 large-M hoist (commit a5bc5c9) gave NO clear c4 speedup → the GEMMs are
  **throughput-limited, not M-limited**. The QKVZ GEMM at M=1403 is ~4.25 ms =
  **~16 TFLOP/s**, compute-bound (memory ~0.3 ms).
- We are NOT leaving the FP4 cores idle: the QKVZ/SSM/attention CUTLASS path is
  `atlas_cutlass_nvfp4_gemm_bf16_act_weight_t` → `cutlass.rs:198`
  `out = quant_nvfp4(act) @ weight_t`, kernel `ElementA=ElementB=nv_float4_t<e2m1>`,
  `ArchTag=Sm120`. The `pack_bf16_act_nvfp4` kernel quantizes the activation to FP4.
  So it IS true FP4×FP4 on the Blackwell FP4 tensor cores. (MoE uses `moe_w4a16` =
  4-bit weight × bf16 act — that one is hand-rolled, NOT the cutlass FP4 path, and
  may have more headroom.)
- So 16 TFLOP/s is the achieved CUTLASS Sm120 FP4 GEMM rate on GB10 (sm_121).
- ROOT CAUSE (as far as static analysis can go): the cutlass GEMM is
  `ArchTag=Sm120`, `OpClassBlockScaledTensorOp`, `ThreadBlockShape<128,128,128>`,
  `ClusterShape<1,1,1>`, `KernelScheduleAuto` (`cutlass_nvfp4_gemm.cu:39-73`),
  compiled `-arch=sm_121f`. At M=1403/N=12288 there are ~1056 tiles over 48 SMs
  (~22 waves) → NOT occupancy/tile-limited. It's compute-bound at ~7% of GB10 FP4
  peak → the limit is per-tile MMA/blockscale-dequant efficiency (or sm_121's real
  FP4 MMA rate), which tile/schedule tuning won't move. This is the prefill
  single-stream wall — closing it needs either a hand-tuned sm_121 FP4 GEMM (what
  vLLM likely ships) or confirming GB10's FP4 ceiling is genuinely this low.
- BLOCKED: confirming MMA-vs-overhead split needs GEMM-internal profiling (nsys
  ncu), but nsys can't capture these workloads on this box (see tooling note).
  This is the gating blocker — without a working profiler, GEMM optimization is
  blind. Unblock profiling FIRST, then attack the cutlass FP4 GEMM.

## Varlen continuous-batching prefill (in progress)

Goal: co-dispatch varied-length prefills into ONE forward so out_proj/MoE read
weights once for the whole batch. Flags: `ATLAS_PREFILL_VARLEN=1
ATLAS_FLASHINFER_PREFILL=1 ATLAS_PREFILL_CODISPATCH=1`. **Off by default** until it
beats the per-request scheduler.

State (2026-06-23):
- Correct (no cross-request contamination). Attention uses FlashInfer ragged
  (`cu_seqlens`); GDN Phase 1/Phase 2 run per-request, Phase 3 hoisted to one call.
- **3× regression ROOT-CAUSED and FIXED via nsys.** The profile showed scalar
  `dense_gemm_bf16` at ~60–76% of batched-prefill GPU time (19–31 ms/call). With
  `#[track_caller]` it pinned to `trait_prefill_phase1.rs:158` (QKVZ, n=12288) and
  `trait_prefill_phase3.rs:55` (out_proj). Cause: the **batched** phase1/phase3
  inlined a GEMM dispatch that LACKED the CUTLASS-NVFP4 branches the **single-stream**
  path (`prefill_qkvz_proj` / `prefill_out_proj_dispatch`) already had — so
  co-dispatched requests fell back to the scalar BF16 GEMM while single-stream ran
  NVFP4. Fix: route phase1/phase3 through those shared helpers. Prefill TTFT on the
  4-req set dropped **~4.9s → ~1.6–2.0s**; varlen ON prefill **737 → ~1500–1750 tok/s**.
- **Strategic finding**: after the fix, varlen ON (~1500–1750) ≈ per-request OFF
  (~1530–2572) — neither clearly wins, both ~2000 tok/s at c4. **Batching is NOT the
  lever to vLLM parity.** vLLM prefill c4 ≈ 6700 = ~2.3× our single-stream 2957 — the
  gap is **per-token kernel efficiency** (GDN scan / MoE / attention prefill on GB10),
  not orchestration. Pursue core-kernel speed next, not more varlen plumbing.
- nsys recipe: `bash /tmp/holo_serve_nsys.sh` (runs the binary directly under nsys,
  varlen env baked in) → warm + one bench → SIGTERM the spark pid → `/tmp/vl.nsys-rep`
  → `nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum`.
- Bench metric: report **prefill via max_ttft**, NOT wall (wall now includes decode
  when `max_tokens` is realistic). `/tmp/varlen_bench.py` prints both.
- Diagnostic-sync gating (still in): `prefill_phase1_inner` entry sync + GDN log gated
  behind `k>4096` / `debug!` (was per-request churn).

## Debugging principle that paid off here

When a batched path regresses, **isolate batch=1 vs batch=N before optimizing**.
batch=1 being fine while batch=N collapses pointed straight at per-request-loop
overhead (the sync), not GEMM-M efficiency — which would have been the wrong fix.
