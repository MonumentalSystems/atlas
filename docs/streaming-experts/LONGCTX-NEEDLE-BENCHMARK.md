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

## Cross-engine control: the same model on vLLM

Run through the engine-agnostic `bench/needle_longctx.py` against vLLM (`holo-aeon`, vLLM 0.24.0, flashinfer
attention, **the same NVFP4 weights**, `--kv-cache-dtype fp8`, `--enable-prefix-caching`, `--max-model-len 262144`,
`--max-num-batched-tokens 16384`, KV pool 602,931 tokens, thinking disabled to match Atlas's default):

| ctx | prefill Atlas-fp8 | prefill Atlas-bf16 | **prefill vLLM** | decode Atlas | **decode vLLM** | needles Atlas | **needles vLLM** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 3,357 | 4,678 | **5,016** | 28.2 | **50.0** | 2/2 | **2/2** |
| 64K | 795 | 1,244 | **3,178** | 17.1 | **46.7** | 3/3 | **3/3** |
| 128K | 598 | 1,125 | **2,171** | 9.5 | **33.3** | 6/6 | **6/6** |
| 200K | 369 | 750 | **1,436** | 6.4 | **28.6** | **7/8 FAIL** | **8/8 PASS** |

## Findings (revised — the cross-engine control overturned the first reading)

1. **vLLM recalls 8/8 at 200K, including the exact needle Atlas misses.** Same weights, same fp8 KV, same context,
   same prompt. So the model is fully capable of this retrieval: **Atlas has a long-context correctness regression.**
   ⚠ An earlier revision of this doc concluded "lost-in-the-middle, a model limitation." That was wrong. The bf16 arm
   ruled out *KV quantization*, but not the rest of Atlas's long-context path — the cross-engine control did.
2. **It is NOT a KV-quantization artifact** (this half still holds). Atlas's bf16-KV arm missed *the same* needle
   (`VAULT-CIRRUS`, planted ~51K deep) with *the same* symptom, so fp8 KV is exonerated as the cause of the miss.
3. **The 200K failure is a coherence signature, not a pure retrieval miss.** In both Atlas arms the model also
   silently abandoned the requested `LABEL=CODE` format and emitted bare codes. vLLM held the format at 200K.
   Losing instruction-following *together with* a needle points at degradation of Atlas's long-context state.
4. **Atlas degrades far more steeply with depth than vLLM — the real tell.** Normalizing each engine to its own 32K:

   | 32K → 200K | Atlas-fp8 | Atlas-bf16 | vLLM |
   |---|---:|---:|---:|
   | prefill slowdown | 9.1× | 6.2× | **3.5×** |
   | decode slowdown | 4.4× | — | **1.7×** |

   At 32K Atlas-bf16 is within **1.1×** of vLLM; by 200K it is **1.9×** behind. vLLM runs DFlash speculative decoding
   (k=4) so its absolute decode numbers are not comparable — **but speculation multiplies throughput roughly
   independently of context depth, so it cannot explain a slope difference.** Atlas's long-context attention path
   (prefill *and* decode) is the suspect, and it is the same suspect as the correctness miss.
5. **fp8 KV costs Atlas prefill throughput and buys nothing.** bf16 KV is **1.4–2.0× faster on prefill at every depth**
   (750 vs 369 tok/s at 200K) while decode is identical (6.4 vs 6.4 t/s). fp8's only win is memory (2.4 vs 4.8 GB),
   irrelevant at 39 GB free. **Prefer `--kv-high-precision-layers max` for long-context serving on this model.**
6. **Prefill throughput collapses with depth on both engines** (quadratic attention over the growing prefix), but the
   Atlas collapse is ~2.6× steeper.

## Open follow-up

The Atlas-vs-vLLM perf rows come from two harnesses (Atlas's own `Done:` log line vs the portable two-point timing).
Recall is harness-independent and solid; for a fully symmetric perf comparison, re-run Atlas through
`bench/needle_longctx.py --salt <s> --disable-thinking` on the same box. Prime suspects for the regression:
long-context attention kernel / RoPE at depth, and the chunked-prefill boundary handling.

## Cross-check against the HSS decode tax

No-HSS decode here is **28.2 t/s at 32K ctx**. The earlier HSS run at `--high-speed-swap-cache-blocks-per-seq 1024`
measured **7.1 t/s at only ~11K ctx** — i.e. HSS-engaged decode at 1/3 the context was ~4× slower. Consistent with
the eager write-through: the offload loop re-offloads the active/boundary block on **every decode step** even when
nothing ever evicts. Independent corroboration of the write-through tax documented in
`PREFILL-WINDOW-SWEEP-FINDINGS.md`, and further motivation for the pressure-gated write-back offload.

## Reproduce

`/home/ms/.claude/jobs/42b99a42/tmp/needle-longctx.sh` (fp8) and `needle-hp.sh` (bf16, adds
`--kv-high-precision-layers max`).
