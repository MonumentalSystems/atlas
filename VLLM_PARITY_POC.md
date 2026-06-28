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
