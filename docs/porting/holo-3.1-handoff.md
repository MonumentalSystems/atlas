# Holo 3.1 on Atlas — Handoff

**Date:** 2026-06-20
**Branch:** `feature/holo-port-pr177` (HEAD `3e853f1`, tree clean)
**Target:** `Hcompany/Holo-3.1-35B-A3B-NVFP4` — Qwen3.5-MoE VLM (hybrid GDN/SSM +
full-attention every 4th, 40 layers, 256 experts/8 active, NVFP4 W4A16), GB10/DGX Spark.
**Measuring stick:** the production vLLM Holo service (see config below).

---

## Goal

Make Holo 3.1 serving on Atlas "blazingly fast, accurate, leaner on memory" — at
parity with the vLLM service. Concretely:

| metric | start | now (Atlas) | vLLM target |
|---|---|---|---|
| C=1 decode | 45 | **61** (81%) | ~75 |
| C=4 decode | 48 | **68** (47%) | ~145 |
| C=8 decode | — | **90** (52%) | ~175 |
| pre-KV memory | 53 GB | **~54 GB** ✅ | (leaner is better) |
| prefill | — | ~960 tok/s (flat) | ~4500 |
| DFlash accept (γ=4-5) | — | **6.5%** (wrong drafter) | **2.5–3×** |

---

## How to run

**Production server (non-spec, stable, fast):**
```bash
bash scripts/holo_serve.sh /tmp/holo-atlas.log
# self-cleans any prior server; setsid-detached; ready in ~75s on :8890
curl -s http://127.0.0.1:8890/health     # {"status":"ready",...}
```
Config baked in: `ATLAS_DECODE_GRAPHS_MULTISEQ=1 ATLAS_HOLO_FP8_SSM_DECODE=1`,
32K ctx / C8 / `--max-prefill-tokens 8192` / `--gpu-memory-utilization 0.55`.

**DFlash acceptance test (the correct drafter — see findings):**
```bash
bash scripts/holo_serve_dflash.sh /tmp/holo-atlas-dflash.log
# Qwen3.5 drafter, γ=4. A/B knobs: ATLAS_DFLASH_NO_MSCALE, ATLAS_DFLASH_DEBUG_CTX_OFF
```

**Build (incremental, ~18s; ~29s on kernel change):**
```bash
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server --no-default-features --features cuda
```

**Bench:** `scripts/bench_holo_atlas.py`. vLLM baseline numbers are in
`/home/ms/spark-vllm-docker/results.csv` — **do not re-run vLLM** (user directive).

### Process-management gotcha (cost me a lot of time — read this)
`pgrep -f "target/release/spark serve"` **matches your own polling shells** (their
argv contains that string) and the launcher's `setsid -f` detaches, so a stale
server survives and the new launch orphans (the "2 servers at once" symptom).
- Match the real binary with **`pgrep -f "release/spark serve --model"`** (your
  shells don't contain `--model`).
- One logical server shows as **2 PIDs** (parent + child) — that's normal, not a dup.
- Find the real port owner: `ss -ltnp | grep 8890`.
- Don't fire the launcher repeatedly while one is loading — each self-clean kills
  the in-flight load and you get racing half-loaded processes.

---

## What landed this session (all committed)

1. **Multi-seq CUDA graphs** (`ATLAS_DECODE_GRAPHS_MULTISEQ=1`, `decode_a2.rs`):
   the n≥2 disable was over-conservative — scheduler keeps active seqs in
   contiguous SSM slots `[0..n)`, so replay is safe. **c8 48→78, c4 48→60.**
2. **FP8 SSM decode overlay** (`ATLAS_HOLO_FP8_SSM_DECODE=1`): block-scaled FP8
   QKVZ/out_proj for decode, BF16 kept for prefill (avoids layer-36 prefill crash).
   **Big C=1 win.** +~750 MB.
3. **Pipelined w8a16 GEMM** in batched SSM decode (nsys-driven — `w8a16_gemm` was
   44.6% of the C=4 step): **c4 60→68, c8 78→90.** This was *the* profiling win.
4. **~10 GB reclaimed**: free dead post-concat BF16 SSM buffers; FP8 overlay.
5. **PR #177 synced** (block-scaled FP8 accuracy).
6. **DFlash lm_head crash fixed**: drafter was doing BF16 `dense_gemm` on Holo's
   NVFP4 lm_head (254 MB read as 1 GB → garbage + ~4× OOB → CUDA-700). Now
   dispatches `w4a16_gemm`. (A *separate* CUDA-700 still remains — see below.)
7. **Launcher hardening + prefill/util tuning** (`0c7ea8d`).
8. **DFlash launcher** with correct drafter (`3e853f1`).

---

## Key findings (so they aren't re-discovered)

### Decode is dependency-chain-DEPTH bound at C>1 (NOT bandwidth/launch/occupancy)
Proven by **7 no-op/loss experiments** (all correct, all zero or negative speedup):
LM-head→GEMM, o_proj→GEMM, smem-activation-cache, K-loop unroll, grouped-MoE,
strided-batched-GDN, **transposed-n128 lm_head** (this session: c8 90→86, reverted).
Measured ~15× off aggregate bandwidth (GPU ~93% idle). CUDA graphs already pipeline
independent kernels, so horizontal batching can't shorten the chain.
**The ONLY remaining decode lever = VERTICAL KERNEL FUSION** (fuse conv+gdn+gated_norm;
fuse MoE gate→topk→experts→down→combine). This is how vLLM is ~2×. It's a real
multi-day CUDA effort, not a swap. Do NOT re-attempt horizontal/per-kernel decode opts.
Post-w8a16 hot kernels (eager nsys): w8a16_pipelined 31%, lm_head w4a16_gemm 16%
(~32 GB/s, huge-N/M=4, no ready faster kernel), attention dense_gemv 12.4%, MoE 20%.

### Prefill is per-token kernel-efficiency bound (NOT chunk-size bound)
Throughput is **flat ~940–965 tok/s for any `--max-prefill-tokens`** (4096→940,
8192→964, 16384→964). Matching vLLM's 16K batch does NOT close the ~4.7× gap; 16K
costs +5.5 GB pre-KV for ~2.5%. Settled on 8192/util 0.55. The gap is the prefill
MoE grouped-GEMM + attention prefill kernels — same deep-kernel story as decode.

### Profiling method (the data-driven loop that found the wins)
- `ncu` can't attach (must launch). `nsys` **hides kernels inside CUDA graphs** →
  must capture **EAGER** (no `ATLAS_DECODE_GRAPHS_MULTISEQ`).
- Harness: `/tmp/nsys_eager.sh` (delay 100 / duration 8, fire C=4 load,
  `nsys stats --report cuda_gpu_kern_sum`). Per-phase eager timing: `ATLAS_MS_PROFILE=1`.

---

## DFlash — the highest-ceiling lever (2.5–3×), and exactly where it's stuck

**Architecture is faithful.** Verified line-by-line against the reference drafter
`dflash.py` (`/tank/hf/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash/snapshots/*/dflash.py`):
fc(10240→2048)+hidden_norm once globally; ctx used as K/V re-derived each layer from
the fixed fc projection (not evolving, not through input_layernorm); capture =
post-layer BF16 residual at `[1,10,19,28,37]` with the correct HF `−1` offset
(`[0,9,18,27,36]`); noise = `[last_token, mask×(γ-1)]`; positions, mask_token=248070,
drafter-specific yarn (θ=1e7, factor=64) — all match. **The 6.5% is a data/numerics
problem, not an architecture or multi-week-rewrite problem.**

**Runtime A/B (acceptance via `/metrics atlas_spec_decode_verify_total{k="2"}`):**
- ctx OFF → **0%** (0/444); ctx ON → **6.5%** (27/417). ⇒ ctx path WORKS, not dead.
- yarn mscale "fix" (fold (0.1·ln64+1)²≈2.0 into inv_sqrt_d) → **2.7%, HURTS.** Reverted.
  Don't re-add (reference's HF Qwen3RotaryEmbedding nominally applies it, but
  empirically wrong here — likely baked into exported weights).

**★ ROOT-CAUSE LEAD (high confidence): WRONG DRAFTER.** All A/B above used the
**Qwen3.6** drafter. Holo 3.1 is `Qwen3_5MoeForConditionalGeneration`; the recipe
(`/tank/holo-3.1-bf16-dflash.recipe.yaml`) and the user's vLLM both specify
**`z-lab/Qwen3.5-35B-A3B-DFlash`, n=4**. The 3.5 and 3.6 drafters have identical
config (target_layer_ids, mask, θ) but **different trained weights** — the 3.6 one
reads Holo's hiddens out-of-distribution → exactly the weak 6.5%. `holo_serve_dflash.sh`
now points at 3.5 (`...Qwen3.5-35B-A3B-DFlash/snapshots/a6ab3a277f856d91c43f28711611e7929073d56d`).

**BLOCKER: a 2nd CUDA-700 crash.** With the 3.5 drafter the server crashes after
~1 request (log ends abruptly, no Rust error = async device fault). Prevents a clean
acceptance measurement. Earlier "crash fixed" only covered the lm_head NVFP4 path.

### Next steps for DFlash (in order)
1. **Fix the recurring CUDA-700.** Use `compute-sanitizer --tool memcheck` on a
   short run (NOT rapid relaunch — that fights the debugger). Suspects: SSM-state
   corruption from the no-rollback / K=γ verify path that doesn't populate
   per-position SSM intermediates (only K=2/3/4 WY-chunkwise kernels do); or a
   scratch/KV overrun as `ctx_len` grows across requests. Files:
   `crates/spark-model/src/layers/dflash_head/{propose,forward_block,forward_block_layer}.rs`,
   `crates/spark-model/src/model/trait_impl/{verify_b,decode_a}.rs`,
   `crates/spark-server/src/scheduler/{verify_k2_step,verify_dflash_step}.rs`.
2. **Re-measure acceptance with the 3.5 drafter** (cap stays 1 → routes through the
   correct K=2 verify; read `k="2"` accept/reject from `/metrics`). If it jumps to
   ~50-70%, the drafter mismatch was the bug.
3. If still low, run the built-in **element-wise bisection** (infra already exists,
   unused): launch with `ATLAS_DFLASH_DEBUG_FORCE_PATTERN=1 ATLAS_DFLASH_DEBUG_DUMP_FULL=1`
   → writes `/tmp/atlas_target_hidden.bin`, `/tmp/atlas_attn_out.bin`,
   `/tmp/atlas_o_proj_out.bin` + logs drafts. Write a ~40-line PyTorch harness that
   imports `dflash.py`, loads `model.safetensors`, feeds the SAME force-pattern input,
   and compares layer-0 q/k/v/attn_out element-wise — first divergence localizes it.
4. Only then tackle γ-parallel: `ATLAS_DFLASH_DRAFT_CAP` is 1 because the K=γ eager
   verify corrupts SSM state. The γ-parallel speedup needs a K=γ GDN verify kernel
   that populates per-position intermediates (kernel work).

### Whether quantization degrades the captures (open, lower priority)
Atlas runs the target with FP8 SSM decode + NVFP4 MoE; the drafter was trained on
BF16 target hiddens. Couldn't test cleanly (server kept crashing). vLLM also runs
NVFP4 and gets 2.5-3×, so probably secondary to the drafter mismatch — but worth an
A/B (drop `ATLAS_HOLO_FP8_SSM_DECODE` so captures are BF16) once the crash is fixed.

---

## Production vLLM config (for parity reference)

```
--attention-backend flashinfer --gpu-memory-utilization 0.34
--max-model-len 131072 --max-num-seqs 32 --max-num-batched-tokens 16384
--mamba_ssm_cache_dtype float32           # Atlas already uses FP32 recurrent state ✓
--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32]}'
--enable-prefix-caching --trust-remote-code --enable-auto-tool-choice
--tool-call-parser qwen3_coder --reasoning-parser qwen3
--limit-mm-per-prompt '{"image":3,"video":0}' --tensor-parallel-size 1
--speculative-config '{"method":"dflash",
  "model":".../Qwen3.5-35B-A3B-DFlash","num_speculative_tokens":4,
  "attention_backend":"flash_attn"}'
```
They run util **0.34** at **131K** ctx / **32** seqs because DFlash keeps its own
full-context KV (extra cost) — a different memory regime than Atlas's current 0.55/32K/8.

---

## Pointers
- Memory notes (persist across sessions):
  `~/.claude/projects/-home-ms-atlas/memory/holo-{perf-goal,decode-batching,dflash-state}.md`
- Roadmap + ranked fusion targets: `docs/porting/holo-3.1-optimization-roadmap.md`
- Drafter checkpoints: `/tank/hf/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash/...` (correct),
  `...Qwen3.6...` (wrong — different target).
- Target weights: `/tank/holo-bf16kv-test`. Recipe: `/tank/holo-3.1-bf16-dflash.recipe.yaml`.

## One-line bottom line
Quick wins are spent (graphs + FP8-SSM + pipelined-w8a16 landed: C=8 +88%). Both raw
gaps (decode 52%, prefill 21% of vLLM) now provably need **vertical kernel fusion** —
a scoped multi-day CUDA project. The biggest single lever is **DFlash** (2.5-3×): now
de-risked to *use the Qwen3.5 drafter + fix one CUDA-700 crash*, not an open mystery.
