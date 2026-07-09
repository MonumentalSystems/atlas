# Holo-3.1-35B-A3B-NVFP4 long-context needle benchmark (no offloading)

**Date:** 2026-07-09 · **Hardware:** dgx-00 GB10 · **Model:** `Hcompany/Holo-3.1-35B-A3B-NVFP4` (natural max ctx 262144)
**Config:** NO `--high-speed-swap` (full performance, all KV resident) · `--max-seq-len 262144` · `--target-kv-tokens 250000` ·
`--ssm-cache-slots 256` · `--enable-prefix-caching` (radix) · `--scheduling-policy slai --tbt-deadline-ms 100` ·
`--max-prefill-tokens 16384` · no profiling.

## Method

8 secrets (`VAULT-ALPHA=ZEBRA-7719` … `VAULT-HALCYON=CINNABAR-2596`) planted every ~25K tokens into growing
deterministic filler. At 32K / 64K / 128K / 200K the model is asked to list **every** secret seen so far as
`LABEL=CODE`. Scored on **missed** needles *and* **hallucinated** codes (a code listed before it was planted),
so it cannot pass by guessing. Prefix caching is on and each round's filler is a strict token prefix of the next,
so only the **new** tokens are prefilled per round; `prefill tok/s` = new tokens ÷ TTFT. `decode tok/s` and TTFT
are taken from the server's own `Done: N tokens (…) X tok/s, TTFT=Yms` log line.

## Results

| ctx | prompt_tok | new_tok | TTFT (fp8) | prefill t/s (fp8) | decode t/s (fp8) | TTFT (bf16) | prefill t/s (bf16) | decode t/s (bf16) | needles | result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 32K | 32,419 | 32,406 | 9.7s | 3,357 | 28.2 | 6.9s | **4,678** | 28.3 | 2/2 | PASS |
| 64K | 65,939 | 33,520 | 42.2s | 795 | 17.1 | 27.0s | **1,244** | 17.2 | 3/3 | PASS |
| 128K | 133,354 | 67,415 | 112.8s | 598 | 9.5 | 59.9s | **1,125** | 9.6 | 6/6 | PASS |
| 200K | 209,241 | 75,887 | 205.9s | 369 | 6.4 | 101.2s | **750** | 6.4 | 7/8 | **FAIL** |

fp8 arm = `--kv-cache-dtype fp8` (KV 2.4 GB). bf16 arm = `+ --kv-high-precision-layers max` → "10/10 attention
layers at bf16" (KV 4.8 GB). Identical needles, identical context, only KV precision differs.

## Findings

1. **Recall is perfect through 128K; 200K drops one needle (7/8).** The miss is `VAULT-CIRRUS` (planted ~51K deep,
   ≈25% into the context) — recalled correctly at 64K and 128K, lost only once the context reaches 200K. Classic
   **lost-in-the-middle** attention dilution. Nothing was hallucinated in either arm.
2. **It is NOT a KV-quantization artifact.** The bf16-KV arm missed *the same* needle with *the same* symptom, so
   fp8 KV is exonerated. The server's startup warning ("NVFP4 models may hallucinate or lose coherence at long
   context") does not reproduce as a quality effect here.
3. **The 200K failure is a coherence signature, not a pure retrieval miss.** In both arms the model also silently
   abandoned the requested `LABEL=CODE` format and emitted bare codes. Losing instruction-following *together with*
   one needle suggests degradation of the whole long-context state, not just one lookup.
4. **fp8 KV costs prefill throughput and buys nothing here.** bf16 KV is **1.4–2.0× faster on prefill at every depth**
   (750 vs 369 tok/s at 200K) while decode is identical (6.4 vs 6.4 t/s). fp8's only win is memory (2.4 vs 4.8 GB),
   which is irrelevant at this scale (39 GB free). **Recommendation: prefer `--kv-high-precision-layers max` for
   long-context serving on this model.** The prefill-side fp8 attention path is the suspect; worth profiling.
5. **Prefill throughput collapses with depth** (quadratic attention over the growing prefix): the 64K round prefills
   the *same* ~33K new tokens as the 32K round but takes 4.4× longer (fp8). Decode likewise degrades 28 → 6.4 t/s as
   every step attends the full resident KV.

## Cross-check against the HSS decode tax

No-HSS decode here is **28.2 t/s at 32K ctx**. The earlier HSS run at `--high-speed-swap-cache-blocks-per-seq 1024`
measured **7.1 t/s at only ~11K ctx** — i.e. HSS-engaged decode at 1/3 the context was ~4× slower. Consistent with
the eager write-through: the offload loop re-offloads the active/boundary block on **every decode step** even when
nothing ever evicts. Independent corroboration of the write-through tax documented in
`PREFILL-WINDOW-SWEEP-FINDINGS.md`, and further motivation for the pressure-gated write-back offload.

## Reproduce

`/home/ms/.claude/jobs/42b99a42/tmp/needle-longctx.sh` (fp8) and `needle-hp.sh` (bf16, adds
`--kv-high-precision-layers max`).
