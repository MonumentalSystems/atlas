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
