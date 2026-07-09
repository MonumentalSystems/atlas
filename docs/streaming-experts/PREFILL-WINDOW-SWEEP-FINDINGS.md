# HSS prefill: the window sweep — chunking dominates, write-IO is invisible, and the over-core knee

**Date:** 2026-07-09 · **Branch:** `feat/streaming-experts-mvp` · **Model:** Holo-3.1-35B-A3B-NVFP4 (dgx-00 GB10)
**Measured with:** `ATLAS_PREFILL_PROFILE=1` (phase-level, per-chunk), one ~10.9K-token **cold** prompt, `max_tokens=1`, fp8 KV, SLAI@100ms.

## TL;DR

1. **The ~124-token chunk clamp — not write-IO — is the entire HSS prefill tax.** At the same 10,881 tokens and the same bytes written, going from 1 chunk to 89 chunks inflates prefill **7.1×** (1833 → 13064 ms). `forward` is **99–100% of prefill in every config** — the cost is compute (SM under-fill + growing-prefix attention recompute per chunk), not I/O.
2. **Write-run coalescing (`ATLAS_HSS_COALESCE_WRITE_RUNS`) is invisible at prefill: −0.1%.** With 89× the writes of baseline, the coalesced arm matched the per-block arm to noise (13049 vs 13064 ms). On local NVMe the writes hide entirely behind compute. Ships **default-OFF, no prefill perf claim** — its value is decode (Tier-1/2 gave ~4×) and the latency-bound NFS/peer tier.
3. **There are two different optima.** For raw prefill latency (HBM abundant), bigger is always slightly better — `ms/tok` creeps toward the 0.168 baseline out to the full 8–16K-token batch. For **over-core** (HBM is the scarce resource HSS exists to reclaim), the **economic knee is window 32–64 (512–1024 resident tokens/seq)**, capturing **72–84%** of the achievable speedup; past it each extra HBM token buys almost nothing.

## The window sweep

`--high-speed-swap-cache-blocks-per-seq W` clamps `max_prefill_tokens = W·16 − max_batch` (= the chunk size). Baseline `A` runs no HSS ⇒ no clamp ⇒ the whole 10.9K prompt as **one** ≤16384 chunk (≈ the "16K optimal batch").

| window W | HBM tok/seq | chunks | prefill ms | ms/tok | vs base | recovery* |
|---------:|------------:|-------:|-----------:|-------:|--------:|----------:|
| — (no HSS) | — (16K)   |   1    |   1833     | 0.168  |  1.0×   |  100%     |
|    8     |    128      |  89    |  13064     | 1.201  |  7.1×   |    0%     |
|   16     |    256      |  45    |   8186     | 0.752  |  4.5×   |   43%     |
|   32     |    512      |  23    |   4937     | 0.454  |  2.7×   |   72%     |
|   64     |   1024      |  12    |   3686     | 0.339  |  2.0×   |   84%     |
|  128     |   2048      |   7    |   3105     | 0.285  |  1.7×   |   89%     |
|  256     |   4096      |   4    |   2758     | 0.254  |  1.5×   |   92%     |
|  512     |   8192      |   3    |   2576     | 0.237  |  1.4×   |   93%     |

\* recovery = fraction of the max achievable prefill reduction (W08 → baseline) captured.

**Write-coalesce arm:** `D08` = W08 + `ATLAS_HSS_COALESCE_WRITE_RUNS=1` → 13049 ms (−0.1% vs W08), `forward` 13016 vs 13031 ms. Noise.

## The over-core knee (µs of prefill saved per extra resident HBM token)

```
 win   8→16 : +128  HBM tok → save 4878 ms = 38109 µs/HBM-tok
 win  16→32 : +256  HBM tok → save 3250 ms = 12694 µs/HBM-tok
 win  32→64 : +512  HBM tok → save 1251 ms =  2443 µs/HBM-tok   ← 5× drop
 win  64→128: +1024 HBM tok → save  581 ms =   568 µs/HBM-tok   ← 4× drop
 win 128→256: +2048 HBM tok → save  346 ms =   169 µs/HBM-tok
 win 256→512: +4096 HBM tok → save  182 ms =    44 µs/HBM-tok
```

The marginal value falls off a cliff after **window 32–64**. That is the over-core sweet spot: **512–1024 resident tokens/seq recovers 72–84% of the prefill speedup at ⅛–1/16 the HBM of an 8K-batch window.** Sizing bigger trades over-core headroom (the whole point of HSS) for the last ~10% of prefill latency.

## Why W512 (8K batch) still isn't baseline

Even an 8188-token chunk (W512, ~2 chunks for a 10.9K prompt) sits at 1.4× baseline. Residual cost = per-chunk fixed overhead (lookup+restore, snapshot save, boundary syncs) × chunk count + the growing-prefix attention each later chunk recomputes over earlier KV. Baseline `A` pays none of that — one pass, one attention sweep. The gap shrinks with fewer chunks but never fully closes until 1 chunk.

## Practical guidance

- **Prefill latency is a pass-count (window) dial, full stop.** The KV write-side I/O optimizations (Tier-1 block coalescing, Tier-2 read run-merge, write-run coalescing) do **nothing** for prefill on local NVMe — they are decode / slow-tier levers. Don't reach for them to speed up TTFT.
- **For over-core deployments, default the window to ~32–64** (512–1024 resident tok/seq). Raise it only if prefill TTFT is the binding constraint and HBM is spare.
- **Raw-throughput / HBM-abundant:** the GB10 saturates at an 8–16K-token prefill batch (per operator measurement); run the window at `max_seq_len/block_size` (no HBM shrink) to match baseline.

## Addendum: attributing the full-window "write-through tax" (W1024)

At window=1024 (max_prefill 16380 ⇒ the ~11K prompt is **1 chunk**, no chunking, everything resident) HSS still costs **~500ms over the no-HSS baseline** (2327 vs 1833 ms). W1024 is the *best case* for write-run coalescing — one chunk means all 681 blocks/layer are strictly-consecutive disk-ids (681 pwrites → ~11 runs) — so it's the sharpest test of whether the write-side I/O opts move it. They don't:

| W1024 arm | prefill | Δ vs BASE | vs baseline 1833ms |
|---|--:|--:|--:|
| BASE (default write path) | 2327ms | — | +494ms |
| + `ATLAS_HSS_COALESCE_WRITE_RUNS=1` | 2304ms | −22ms (~1%) | +471ms |
| + `ATLAS_HSS_PINNED_OFFLOAD=1` | 2241ms | −86ms (~4%) | +408ms |
| + both | 2232ms | −95ms (~4%) | +399ms |

- **Write-run coalescing = −22ms even here** (near the ~10ms run-to-run noise). Collapsing 681→11 pwrites/layer recovers almost nothing ⇒ prefill writes hide behind compute. Consistent with the −0.1% at window-8.
- **Pinned D2H = −86ms**, which *refutes* the "unpinned 106 MiB @ 0.17 GB/s ⇒ 600ms" hypothesis: the offload D2H largely overlaps compute; only a small tail is exposed.
- **~400ms of the residual is neither writes nor the copy.** It stays inside `forward` (2207ms with both flags vs baseline's 1809ms) — the offload's host-side gather/assembly (`stage_block_into`), per-block predictor projection, and pool bookkeeping, all run to write a disk copy that at this window is **never read back**. The write-side I/O opts address <20% of the tax; the bulk is the offload host-path, not I/O.

**Decode at W1024 (non-spilling, all KV resident):** 7.1 tok/s, TTFT 1712ms over the ~11K context (server-logged, cross-checked by a two-point non-streaming timing). Slow because decode attends the full ~11K resident KV each step on a 3B-active MoE — not an HSS effect (no streaming fires at this window).

**KV data volume:** Holo-3.1-35B-A3B is hybrid — only **10 of 40 layers are full-attention** (GQA 2 KV heads, head_dim 256); the 30 SSM layers hold fixed recurrent state, no KV. fp8 KV = **10 KiB/token** ⇒ ~11K tokens = **106 MiB** (bf16 = 212 MiB). (The `/hss` files are pre-allocated capacity, not live data.)

## Reproduce

`/home/ms/.claude/jobs/42b99a42/tmp/prefill-chunking-ab.sh` (A + W08/W16/W32/W64 + D08) and `prefill-knee-ext.sh` (W128/W256/W512). Serve Holo-35B with `ATLAS_PREFILL_PROFILE=1`, one cold ~11K prompt, grep `PREFILL_PROFILE`. See also the decode-side companion: `DECODE-SPILL-HOST-STAGING-FINDINGS.md`.
