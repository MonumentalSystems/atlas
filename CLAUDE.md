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

## North star

vLLM-parity: decode c4 ≥ 145 tok/s, prefill c4 ≥ 6700 tok/s.
Current single-stream prefill ≈ 2900 tok/s (~43% of the c4 target) → need ~2.3×.

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
- So 16 TFLOP/s is the achieved CUTLASS Sm120 FP4 GEMM rate on GB10 (sm_121). The
  gap to vLLM is therefore one (or more) of: (a) cutlass Sm120 schedule not tuned
  for sm_121 (consumer Blackwell — no tcgen05/TMA fast path → cutlass falls back);
  (b) vLLM ships a better hand-tuned sm_121 FP4 kernel; (c) the 6700 baseline isn't
  apples-to-apples with our 4-mixed-prompt test. Cause is UNCONFIRMED.
- Honest status: the remaining 2.3× is deep FP4-kernel-tuning territory on consumer
  Blackwell (or a baseline-validation question), not a quick fix. Decisive next
  experiment: micro-bench the cutlass FP4 GEMM at the exact shapes vs a known-good
  reference, and/or confirm the vLLM 6700 conditions.

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
