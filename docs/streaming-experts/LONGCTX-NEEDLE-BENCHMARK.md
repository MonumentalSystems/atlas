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

## Diagnosis (2026-07-10)

### UPDATE — the root cause is the FlashInfer GDN kernel, not "cumulative precision"

The precision sweep below (kept for the record) was bisecting the wrong axis: every run used the **FlashInfer
GDN kernel** (`libatlasgdn.so`, prod default `ATLAS_GDN_FLASHINFER=1`). Flipping **only** that flag settles it:

| GDN kernel (cold 200K, prod fp8 everywhere else — native-fp8 attn+ssm, fp8 KV) | needles |
|---|---|
| FlashInfer `.so` (`ATLAS_GDN_FLASHINFER=1`, prod) | **7/8 FAIL** (VAULT-CIRRUS) |
| in-tree FLA chunked (`ATLAS_GDN_FLASHINFER=0`) | **8/8 PASS** |

Only `ATLAS_GDN_FLASHINFER` changed. So the **FlashInfer GDN kernel loses long-context coherence** in a way the
in-tree FLA chunked kernel (`gated_delta_rule_fla`: `recompute_wu → chunk_delta_h_ksplit → chunk_fwd_o`, fp32
`h_state`) does not. The "cumulative precision" result is real but was **compensation**: with the lossy
FlashInfer kernel you need bf16 projections + bf16 KV to add enough margin to pass; with the accurate FLA kernel
you pass at full fp8. Everything reconciles (FlashInfer+fp8 fail; FlashInfer+bf16-proj+bf16-KV pass;
FLA+fp8 pass). The FlashInfer `.so` hard-codes bf16 GDN output with no Rust-visible dtype knob — very likely the
same BF16-truncation source tracked for the in-tree kernels in Avarok #248 / PR #290, but baked into the cubin.

**Recommendation (revised):** serve long-context with **`ATLAS_GDN_FLASHINFER=0`** — 8/8 at 200K while keeping
full fp8 speed on projections + KV, at ~7% slower prefill (FLA vs FlashInfer). Durable perf-preserving fix:
rebuild `libatlasgdn.so` with an F32 (or better-accumulated) GDN output — the F32-output pattern in PR #290 /
#248 is the reference. The ~2.6× prefill/decode slope (F4) remains a separate, benign kernel-efficiency effect.

### Original precision sweep (superseded as the root cause; retained for the record)

Root-caused by a cold-vs-warm bisect + a one-knob-at-a-time precision sweep, all on the **same CUTLASS prod
image** (`atlas-gb10:spillpool`, full native-fp8 + CUTLASS env set — a host `cargo build` is the WRONG numeric
config and cannot reproduce this).

**Step 1 — kill the warm hypothesis.** The model is a **hybrid: 30 of 40 layers are GDN linear-attention**
(recurrent state *outside* the KV cache); only 10 are full-attention. The benchmark runs warm (at 200K only
~76K of 209K is freshly prefilled), so it was natural to suspect SSM-snapshot / prefix-replay drift. But a
**cold single-pass 200K** (fresh salt, whole context prefilled in one contiguous pass, no snapshot restore, no
KV reuse — `--request-timeout 0` required or the long cold prefill dies mid-decode) **misses the SAME needle,
7/8.** ⇒ the miss is **depth-intrinsic**, and the snapshot / prefix-cache machinery is **exonerated**.

**Step 2 — precision, not structure.** Cold 200K at max precision (bf16 attn+ssm projections + bf16 KV)
recalls **8/8**. A structural bug (RoPE range, chunk-boundary math, index truncation) would not be fixed by
raising precision — those are exonerated.

**Step 3 — no single fp8 path is the culprit; the error is cumulative.** One knob at a time, cold 200K:

| config | needles | missed |
|---|---|---|
| prod: fp8 proj + fp8 KV | 7/8 | VAULT-CIRRUS (~51K, mid) |
| `NATIVE_FP8_SSM=0` only (bf16 SSM proj) | 7/8 | VAULT-CIRRUS |
| `NATIVE_FP8_ATTN=0` only (bf16 attn proj) | 7/8 | VAULT-CIRRUS |
| bf16 KV only (`--kv-high-precision-layers max`) | 7/8 | VAULT-CIRRUS |
| bf16 **both** proj, fp8 KV | 7/8 | **VAULT-HALCYON (~200K, deepest)** ← miss shifted |
| bf16 proj **+** bf16 KV | **8/8 PASS** | — |

The missed needle **shifting** from mid-context (fixed by bf16 projections) to the deepest position (still sunk
by residual fp8-KV error, which accumulates most at 200K) is the signature of cumulative, threshold-crossing
precision loss across **three independent fp8 paths** — attention projections, SSM projections, and KV cache.

**Why vLLM passes (F1):** it does not stack these downgrades — notably it keeps the SSM/GDN recurrent path at
higher precision (Atlas's `h_state` carry is fp32 at `ssm_pool.rs:230`, but its SSM *projections* and FLA WY
chunk intermediates are bf16/native-fp8).

**Recommendation.** For long-context *correctness* on this model, serve with `ATLAS_HOLO_NATIVE_FP8_ATTN=0
ATLAS_HOLO_NATIVE_FP8_SSM=0 --kv-high-precision-layers max` (8/8 at 200K), at a prefill-throughput cost
(native-fp8 was the big speed lever). The durable fix is better fp8 *scaling* / fp32-accumulate in these paths
(especially pushing GDN/SSM toward fp32) so speed and correctness coexist. **The ~2.6× prefill/decode slope
(F4) is a separate, benign kernel-efficiency effect (small tiles + serial per-CTA KV loop + fp8 sync-dequant);
the prefill online-softmax is provably correct — do not conflate the slope with the miss.**
