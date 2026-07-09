# Decode-ring spill: host-staging is the cost, not the transport (GB10)

**2026-07-09.** Empirical findings from multi-threading the SSM decode-ring spill
worker and A/B-testing NVMe vs RDMA cold tiers on Holo-3.1-35B (GB10, sm_121).
Reads alongside the GPUDirect-replacement research (no true GPUDirect on GB10 →
host-staged bulk transfer is the correct architecture). **Bottom line: the
decode-ring spill is a concrete instance of the host-staged-vs-GPUDirect
tradeoff — the transport (NVMe vs RDMA) is irrelevant until the GPU→host staging
is fixed, because the spill bounces through a plain host buffer instead of the
zero-copy UMA landing the KV tier already uses.**

## What we built

`DecodeSpiller` (rolling decode-ring tier, `ATLAS_SSM_DECODE_RING_ROLL`) was a
single background worker draining aged hot SSM lanes HBM→cold-tier. At high
concurrency, boundary saves outrun it → `SaveDecision::Backpressure` → the decode
thread busy-spins (`sleep 50µs`) waiting for a freed lane. We replaced it with a
**seq-sharded pool of N workers** (`ATLAS_SSM_DECODE_SPILL_WORKERS`, tier-aware
default nvme=4 / host-RAM,peer=1; `spill_worker_index(seq,n)=splitmix(seq)%n`, so
all incarnations of a `cold_key=f(seq,logical)` stay FIFO on one worker →
byte-reproduces the single-consumer order; each worker its own CUDA stream;
`N=1`+margin=1 byte-identical to before). Added `ATLAS_DECODE_RING_STATS` →
`bp_spins_per_spill` as the causal metric. Scope: **only** the SSM decode-rollback
ring — not the Marconi prefill-snapshot tier (`ATLAS_SSM_TIER`), the KV/RDMA-KV
peer, experts, LoRA, or weights.

## Measurements (Holo-3.1-35B, C=8, ISL 1024 / OSL 512, NVMe unless noted)

| Cold tier | Workers | Tput (t/s) | `bp_spins_per_spill` |
|---|---|---:|---:|
| NVMe | 1 | 47.7 | ~3282 |
| NVMe | 8 | 47.0 | **~1474** |
| RDMA peer (gx10:9920) | 1 | 46.0 | ~3037–3398 |
| RDMA peer | 4 | 44.6 | ~3060–3086 |

Also: `workers=1` alone swings **42.9–47.7 t/s** run-to-run (~±10%) — that
variance swamps any pool effect on throughput at C=8.

### Findings

1. **No throughput win at testable concurrency.** At C=8 the pool leaves Tput
   inside run-to-run noise. The regime where relieved backpressure would convert
   to throughput is C≥64, which **crashes** on this model (`CUDA_ERROR_ILLEGAL_ADDRESS`
   in decode — reproduces with the rolling tier OFF, so it's a pre-existing
   large-batch bug, *not* the spill pool). So the throughput benefit is real in
   principle but **unproven on currently-testable configs**.

2. **The pool works as a mechanism** — on NVMe it halves `bp_spins_per_spill`
   (~3282→~1474) because `pwrite` parallelizes across disjoint file offsets.

3. **RDMA is not a lever.** NVMe and RDMA show the *same* `bp_spins_per_spill`
   (~3000–3400 @ workers=1) and tied Tput. Identical cost across a slow-disk and
   a fast-fabric transport ⇒ **the transport is not the bottleneck.**

4. **The pool does nothing for the peer tier** (workers 1→4: ~3037→~3060, flat).
   The RDMA paging peer serializes every `put` on one connection (2 TCP RTTs +
   RDMA write under one mutex); N seq-sharded workers all block on that single
   arena — `N workers ≈ 1`. Parallelizing it needs one arena *connection per
   worker* (heavier; multiplies peer-side load).

## Root cause: the spill bounces, the KV tier zero-copies

The spill worker gathers into a **plain, non-pinned host `Vec`**, then hands it to
whichever tier (`ssm_snapshot.rs`):
```rust
let mut blob = vec![0u8; blob_bytes];                    // plain heap Vec — not pinned, not UMA
for i in 0..num_layers { copy_d2h_on_stream(..., &mut blob[...]) }   // explicit per-layer GPU→host gather
store.put(req.cold_key, &blob)                           // NVMe pwrite OR RDMA write to peer
```
The code already names the gap (`ssm_snapshot.rs:440`):
> *"cuMemAllocManaged is **not the pinned-UMA the KV zero-copy path uses**."*

The KV tier (`rdma_kv_backend.rs`, `ATLAS_KV_ZERO_COPY`) does the GB10-correct
thing: *"RDMA READ lands directly into a UMA (GPU-addressable) dst"* via `reg_mr`
— no bounce. The SSM decode-ring spill was never wired for that, so every spill
pays an explicit gather into non-pinned host memory *before* the transport ever
runs. That is why NVMe==RDMA: the transport is downstream of the cost.

## Mapping to the GPUDirect-replacement research

| Measured here | Research pattern |
|---|---|
| NVMe ≈ RDMA (~3000 bp_spins both) | "Stop designing around the transport" — GB10 is host-staged; optimize the staging, not the wire. |
| Pool: NVMe 2×, peer 0× | Pattern 9 — serial/fine-grained remote access is a poor fit; the peer's single-connection `put` can't parallelize. |
| Plain non-pinned `Vec` gather | Violates Pattern 1/2 (pinned host rings, double-buffering); no compression (Pattern 4). |
| KV already zero-copies, SSM bounces | The doc's ideal is implemented for KV (`ATLAS_KV_ZERO_COPY` → UMA landing), absent for the SSM spill. |

## Recommended direction

The high-leverage move is the **SSM analog of `ATLAS_KV_ZERO_COPY`**: land the
spill in **pinned-UMA scratch** the GPU writes once and the tier (NVMe O_DIRECT /
NIC) reads directly — eliminating the explicit gather + the non-pinned copy. Then
layer the research doc's patterns: pinned host rings + double-buffering, and
optional FP8/NVFP4 block-compression of the 63 MB blob before the wire. The
spill-worker pool and the cold-tier choice are **second-order**; the staging is
first-order.

**Caveat before any rewrite:** ~3000 spins × 50 µs ≈ **150 ms/spill** is far more
than a 63 MB copy on GB10 unified LPDDR should cost (raw bandwidth ≈ low
single-digit ms). So a large share of the floor is likely **per-layer launch/sync
overhead** — 30 layers × 2 `copy_d2h_on_stream` launches + `stream_wait_event` +
the worker/decode stream contention — not the copy itself. **Profile the gather
first:** the win may be "one *fused* gather into pinned-UMA" (collapse 60 launches
to 1) as much as "avoid the bounce." Instrumentation is in place
(`ATLAS_DECODE_RING_STATS`); a gather-vs-put phase split is the next measurement.

## Status

Landed on `feat/streaming-experts-mvp` (reviewed, unit-tested, default-safe):
seq-sharded pool + `ATLAS_DECODE_RING_STATS`. Keep as the correct mechanism;
the throughput win awaits (a) the C≥64 crash fix and (b) the pinned-UMA fused
gather. Related: `DECODE-RING-ROLLING-TIER.md`.
