# Atlas → vLLM parity POC (concurrent decode + prefill) — Holo-3.1 / GB10

Baseline (Ornith, C=1..8, 256-tok): Atlas C8 108 t/s / TTFT ~990ms ; vLLM C8 193 t/s / TTFT ~190ms.
Atlas WINS single-stream (C1 90 vs 45). Gaps: (A) concurrent-decode agg ~1.8x, (B) prefill/TTFT ~5x.

## Corrected findings (design workflow wf_be5f4897 + adversarial review)
- **A1 batched CUDA graphs — NOT a lever.** Already built (decode_a2.rs, padded_n {2,4,8}), default-on
  (ATLAS_DECODE_GRAPHS_MULTISEQ=1), measured NEUTRAL. Decode is dependency-chain-DEPTH bound (GPU ~93% idle),
  not launch-bound; graphs can't shorten a serial conv→gdn→norm→MoE chain ×40 layers.
- **POC-1 batched GDN scan (ATLAS_SSM_BATCHED_RECURRENT) — PROVEN NO-OP.** One of 6 zero-speedup horizontal-
  batching experiments. Demote to a 5-min confirmatory A/B only.
- **POC-2 defer chunk-0 → fused mixed step (ATLAS_HOLO_ALWAYS_MIXED) — BIGGEST lever.** Inline-blocking chunk-0
  freezes all decode for a full prompt forward → TTFT linear in C. Machinery exists, default OFF. Prior A/B:
  decode-freeze 4406→1310ms (3.4x). Pure env flip to validate.
- **POC-4 vertical fusion — only validated decode lever** (ATLAS_GDN_FUSED_NORM +18%@C4). Incremental, not 1.8x.
- **POC-0 ATLAS_GAP_TIMING instrumentation** — gap_timing.rs (OnceLock+AtomicU64), splits C8 wall into
  queue_wait / prefill_compute / decode-only tok/s / mean_batch / stall_ratio. Build first.

## Order: POC-0 (measure) → POC-2 (TTFT, flag then default-on) → POC-3 (batched mixed for multi-admit) → POC-4 (vertical fusion)
Target landing: C8 agg ~165-185 t/s (vLLM 193), TTFT ~200-300ms flat (vLLM ~190), keeping C1 win.
Full design: workflow wf_be5f4897-1a0.

## MEASURED (POC-0 ATLAS_GAP_TIMING + ATLAS_MS_PROFILE, Holo-3.1, GB10) — 2026-06-28
Baseline Holo C=1..8 (burst): C1 88/135ms, C2 82/258ms, C4 93/540ms, C8 104/1066ms (TTFT≈C×135, mean≈p50≈max).
ATLAS_HOLO_ALWAYS_MIXED=1 (POC-2 step1): NEUTRAL (C8 TTFT 1074 vs 1066). Confirmed dead for burst arrival.

GAP decomposition (cumulative across sweep):
- stall_ratio = 0.000 throughout  → decode NEVER blocked by prefill. POC-2/prefill-stall is NOT the lever.
- decode_only_tok_s FLAT ~90→98 while mean_batch 1.0→2.9  → batched decode gives ~zero throughput amortization.
- prefill_compute per C=8 co-dispatch tick ≈ 827ms for 8 short prompts  → TTFT is prefill-LATENCY bound, co-dispatch
  doesn't truly parallelize prefill compute (single short-prompt prefill ~100-200ms).

Per-phase decode (C=4, eager MS_PROFILE, total≈38.3ms/step, per-tok 9.57ms):
- SSM/GDN  24.8ms  (65%, 30 layers)   <- dominant; per-seq serial chain across 30 layers
- head      7.5ms  (20%, vocab GEMM, ~fixed per step)
- attn      5.9ms  (15%, 10 layers)

## CONCLUSION (data-driven, corrects original A1 hypothesis AND synthesis POC-2)
Two REAL levers, both kernel/structural — no free flag closes them:
 (A) DECODE amortization: SSM is 65% of decode and runs a per-seq serial 30-layer chain; batching doesn't amortize
     (chain-depth bound). Lever = occupancy-preserving VERTICAL FUSION of the SSM layer (ba_gates→conv→gdn→norm),
     extending the proven ATLAS_GDN_FUSED_NORM (+18%@c4). NOT horizontal batching (proven no-op).
 (B) TTFT: prefill latency-bound + co-dispatch serializes internally. Lever = truly batched prefill forward
     (kernel-level, not per-stream loop) + GDN-prefill throughput. NOT ALWAYS_MIXED (stall_ratio=0).
Dead ends proven: A1 batched CUDA graphs (shipped, neutral), POC-1 batched-recurrent GDN (no-op), POC-2 ALWAYS_MIXED (neutral here).

## PREFILL LEVER PROVEN (2026-06-28) — kernel-batched prefill gated off under prefix cache
The truly-batched packed prefill forward EXISTS and WORKS: `prefill_batch_chunk_kernel_batched`
(crates/spark-model/src/model/trait_impl/prefill_b/batch.rs), gated by `kernel_batched_eligible`.
BUG/GATE: `kernel_batched_eligible` returns false whenever `self.prefix_cache.is_active()` ("Fix #4" —
defensive guard against partial-mutation when co-dispatched streams have MIXED cache-hit depths).
Production serves `--enable-prefix-caching true` → gate always fires → per-stream serialization
(verified: 4 concurrent prompts → 4 separate 27-tok forwards @ ~72ms each).

Disabling prefix cache engages it (q12 trace: "kernel-batched dispatch attempt n=4 → succeeded"):
  TTFT  C4 547→294ms,  C8 1066→524ms  (~2x), decode unchanged. Bench uses unique prompts (0 cache hits)
  so prefix caching buys nothing here anyway.

FIX (next, scoped — not new kernel work): the gate is too broad. The mixed-depth bug only occurs with
ACTUAL cache hits. Pre-compute all co-dispatched streams' cache-hit depth (read-only radix lookup) BEFORE
any state mutation; allow the kernel-batched path when all streams are cold / uniform depth, else fall
through cleanly (no partial mutation). Unlocks the 2x TTFT win with prefix caching ON.
Residual gap to vLLM (524 vs ~190ms) = follow-on (packed forward still partly per-layer-overhead bound + admission).

## PREFILL FIX VALIDATED (2026-06-28) — guard relaxation is correct + 2x TTFT
The Fix #4 mixed-cache-depth bug appears already fixed elsewhere. Relaxed the guard behind
ATLAS_Q12_BATCHED_WITH_CACHE=1 (batch_kernel.rs kernel_batched_eligible).
Correctness: warmed cache S (Roman-aqueduct prefix), then co-dispatched n=6 with MIXED hit depths
(full-hit S, partial-hit S+suffix, cold unique) — q12: "kernel-batched dispatch attempt n=6 → succeeded".
All 6 outputs coherent, on-topic, semantically == per-stream reference; NO cross-stream bleed / gibberish
(only ULP-level wording diffs from batched-vs-GEMV accumulation).
Perf (prefix caching ON + flag): TTFT C2 258→178, C4 547→289 (1.9x), C8 1066→495ms (2.15x); agg C8 103→108.
Decode unchanged (per-stream still ~13.8 — that's the SSM-fusion lever, separate).

REMAINING before default-on: my test exercised the SUCCESS path (no mid-batch bail). The original bug was
partial-mutation on BAIL under cache. Validate the bail path + longctx_needle + soak with the flag before
removing the guard / defaulting it on. Then it's a ~free 2x TTFT win with caching on.

## CONTEXT-OVERFLOW GRACEFUL DENIAL (2026-06-28)
Soak surfaced CUDA-700 (cuMemsetD8Async status 700) — a server-killing fault, present in BOTH baseline
and WITH_CACHE=1 (control-confirmed: NOT the relaxation). Causes are over-context vectors:
  1. max_tokens >> max_seq_len (soak longgen: max_tokens=128000 vs max_seq_len=32768) → decode past KV → 700.
     FIXED: chat/mod.rs clamps max_tokens to (max_seq_len - prompt_len), logs warn, finishes length.
     Verified: 128000→32733, HTTP 200, no 700.
  2. REMAINING (pre-existing, separate hardening): soak still 700s on first batch — heterogeneous
     vision(<|image_pad|>)+text+tools co-dispatch and/or the 190K big_ctx path. CORRECTION: auto-compaction
     is NOT the cause — it is OPT-IN via `--auto-compact [THRESHOLD]` (Option<f32>, OFF when omitted; our
     serve omits it) AND only fires for messages.len()>4, while big_ctx is a single message. So the 0
     "prompt too long" rejects is unexplained (big_ctx likely never produced >=max_seq_len tokens at the
     guard, or failed pre-guard). The 64k/190k soak targets a larger-context + vision-image server; not a
     valid test on the 32K text config. Fixing all over-context vectors (vision token budget, decode-time
     hard KV stop, reject-not-truncate oversized single prompts) is a separate robustness task (deferred).

RELAXATION STATUS: WITH_CACHE=1 validated correct WITHIN context (needle 3/3, mixed-cache correctness,
120s sustained stress 0×700 + 0 cross-stream-bleed, corrupt-rate == baseline). 2x TTFT win stands for
in-context serving. Soak 700 is orthogonal.

## DECODE LEVER (c) — design + A/B findings (2026-06-28, workflow wf_04635cc4)
Profile (nsys C=4 GPU-time share, prod kernels): SSM out_proj GEMM (w8a16_pipelined) ~31%, lm_head w4a16_gemm ~16%
(~32 GB/s — ~8x below GB10 peak, "no fast swap"), attn dense_gemv ~12%, MoE ~20%, conv1d ~0%.
Per-step C=4: SSM 24.8ms(65%), head 7.5ms(20%), attn 5.9ms(15%). C=8: ssm 73%.

A/B RESULTS (live, gx10:8890, vs FUSED_NORM baseline 88/82/93/104 @ C1/2/4/8):
- ATLAS_SSM_BATCHED_RECURRENT=1 → 89.5/81.9/93.0/104.6 = NO-OP (confirmed; horizontal batching is chain-depth-bound, GPU ~93% idle).
- FUSED_NORM (already on) = the one proven fusion (+18%@C4). FUSED_CONV = neutral (per-k-head grid halves CTAs 32→16).
- conv1d fusion is pointless (conv ~0% of decode).

DESIGN candidates (occupancy-preserving fusion, NOT horizontal batching):
- C1 BA-into-QKVZ concat GEMV (hoist per-seq ba_gates into the batched qkvz pass): low risk, +3-8% (review down-rated from +8-15%).
- C2 FUSED_CONV v2 (per-V-head grid remap, restore 32 CTAs): +10-18% possible, medium-high risk (conv_state race / recompute).
- Head/vocab GEMM (lm_head w4a16_gemv @ ~32 GB/s, 20% of step): FATTEST lower-risk single-kernel target — ~8x bandwidth
  headroom; amortizes across batch. Review's pick over the 30-layer fusion.

HONEST CEILING: decode is fundamentally chain-depth bound (30 serial layers, GPU 93% idle). Realistic cumulative decode
gain ~+12-20% (head-GEMV + a fusion), NOT the ~1.8x needed for full concurrent-decode parity — that needs whole-layer
vertical fusion (architectural). Atlas ALREADY WINS single-stream (C1 90 vs vLLM 45); the gap is purely concurrent decode.
NEXT (best gain/risk): optimize the lm_head w4a16_gemv bandwidth (20%/8x headroom), then C1.

## DECODE AMORTIZATION — FINAL VERDICT (2026-06-28, FLA bench + code audit)
FLA/FlashInfer gated_delta_rule_decode (the kernel vLLM uses), GB10, 1 GDN layer:
  B=1 27us, B=4 87us (3.24x), B=8 155us (5.74x) — does NOT amortize. At B=8: 32MB state traffic/155us
  = ~206 GB/s (~75% of GB10 peak) → STATE-BANDWIDTH bound (state = 2MB/seq, scales with batch).
  => Swapping in NVIDIA's GDN decode kernel will NOT close the concurrent gap. vLLM hits the same physics.

Code audit of Atlas decode SSM (trait_decode_multi_seq.rs):
- out_proj: ALREADY batched (line 827 "ONE GEMM, weights ×1", w8a16_gemm_pipelined / w4a16_gemv_batchm). amortizes.
- qkvz: ALREADY batched. ba_gates: per-seq GEMV but weight tiny [64,2048] → batching = launch saving only (~C1).
- MoE decode: sparse; n tokens hit up to n*top_k distinct experts → expert-weight reads scale with batch (no amortize).

CONCLUSION: concurrent decode of this GDN+sparse-MoE hybrid is PHYSICALLY un-amortizable — both dominant costs
(GDN state I/O + MoE distinct-expert reads) scale with batch. vLLM's ~2x per-stream edge at C8 is per-kernel
EFFICIENCY / framework overhead, NOT a structural amortization Atlas lacks. Fusion (C1/C2) and GDN-kernel swap
are dead ends for the concurrent gap. Atlas ALREADY wins single-stream (C1 90 vs 45).
=> The only real decode wins are per-kernel bandwidth efficiency. Fattest: lm_head w4a16 GEMV @ ~32 GB/s (8x
headroom, 20% of step). Realistic decode ceiling ~+15-20%, not concurrent-parity. lm_head GEMV is THE target.

## lm_head GEMV — NEGATIVE (2026-06-28): dispatch swap doesn't help, FP4-read-bandwidth bound
Routed the decode lm_head (decode_a2.rs + lm_head_batched) through w4a16_gemv_batch16 (the small-M
weight-once GEMV out_proj uses) instead of w4a16_gemm. MS_PROFILE head us IDENTICAL: GEMV 7566us vs
GEMM 7549us (both ~34 GB/s reading the 254MB NVFP4 head weight once). Needle 3/3 (numerics fine).
=> The head bottleneck is the FP4 weight-READ access pattern (8x below GB10 peak) shared by BOTH kernels,
NOT the GEMM M-tiling. A dispatch swap can't fix it; needs a NEW bandwidth-optimized NVFP4 GEMV kernel
(vectorized 128-bit loads, coalesced nibble+scale). Reverted the no-op swap.
Note: head is FIXED ~7.5ms/step (weight read once) → it DOMINATES low-concurrency decode (~most of the
C=1 token) but AMORTIZES at high C (7.5ms/8tok). So a fast FP4 head kernel would extend Atlas's existing
C=1 win (single-user latency), not close the concurrent gap. Real kernel R&D, modest/low-conc payoff.

## (b) OVER-CONTEXT HARDENING + SOAK-CRASH ROOT CAUSE (2026-06-28)
Over-context graceful denial: ALREADY WORKING for text. Isolated probes 40K/50K/64K-token prompts all
return clean HTTP 400 "Prompt too long: N exceeds max_seq_len 32768", server stays ALIVE (chat/mod.rs:191
guard + the max_tokens clamp). 8 concurrent within-limit 14K prefills (112K total KV demand) → all HTTP200,
alive (KV pool pages/queues fine). So the user's ask (deny over-context instead of exploding) is satisfied.

SOAK CUDA-700 ROOT CAUSE = a VISION BUG, not over-context:
- A single image request CUDA-700s the server (HTTP 500) when the image is NON-SQUARE (soak uses 640x400):
  "cuLaunchKernel CUDA_ERROR_ILLEGAL_ADDRESS (700) grid=[293,1,1] block=[256,1,1]" in the vision prefill,
  then cascades (free_sequence zero_slot 700). Reproducible on a FRESH server, first request.
- SQUARE images work: 448x448 → HTTP200, correct answer, 0x700.
=> The vision encoder mishandles non-square aspect ratios (patch-grid / merge OOB). This is what kills the
soak (it rotates in a 640x400 screenshot). Separate from over-context + from the parity POC. Needs a
vision-encoder fix (find the grid=293 kernel's non-square bounds bug). Square-image vision is fine.

## VISION FIX VALIDATED via soak (2026-06-28)
After the non-square over-max-pixels fix: SOAK faults 12378 → 12, server STAYS ALIVE (served ~969 reqs/60s,
15 image encodes, 33 completions) vs total context-death before. The catastrophic crash is RESOLVED.
- Reproducible non-square 640×400 → CUDA 700: FIXED (now encodes 240 patches, answers correctly).
- Residual: 12× status 716 (MISALIGNED) on vision requests under CONCURRENT load (~1.2%), now ISOLATED
  (server survives, no cascade). Could NOT reproduce in isolation (vision_probe + 6 varied/odd sizes
  sequentially = 0 faults); only surfaces under the 6-client concurrent soak. Separate, intermittent
  vision-encoder buffer-alignment bug (grid=278 kernel) — deeper, concurrency-dependent, non-catastrophic.
