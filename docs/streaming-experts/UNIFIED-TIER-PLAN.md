# Unified Tiered Cache — roadmap for the next ultracode push (2026-07-07)

**Architecture: one spill tier for BOTH KV blocks and SSM snapshots.** Route both
through the *already-shipped* byte-agnostic `StorageBackend` cascade
(`CascadeBackend` local-pinned-RAM → `RdmaKvBackend` peer → `IoUringBackend` NVMe)
and the same `SlotCache` LRU policy. Compute paths stay consumer-specific (KV = tiled
attention; SSM = D2D/H2D copy) — what's shared is the byte-movement cascade + one
promotion/eviction mechanism. Staged **by value**, converge to one namespace only if
it earns its keep.

## Why this ordering (the operator's correction is the tiebreaker)

In the *realistic* regime — a big resident cache (~32K tokens/seq), context rolls to
the tier only for the deep tail, 8 concurrent agents — the dominant loss is **NOT** KV
read volume. It's the **SSM snapshot pool thrashing**: 16 HBM slots across 8 sessions →
**~0 Marconi hits** (measured live: 16 saved / 0 hits) → a **30-SSM-layer recompute over
the prefix on every warm turn**. So the north-star test shows its biggest, most-certain
win the moment SSM gets a spill tier — that lands first, before the deep KV throughput
levers. The pathological `cap=8` microbench (where 3 prior attempts flatlined) is *not*
the north star.

## The three failed attempts → encoded as INVARIANTS, not phases

| attempt (wip branch) | result | invariant it forces |
|---|---|---|
| `kv35-batched-streaming` | 2.7× regression | **Never drop the resident cache** (batched reads always keep per-seq lookup/eviction) |
| `kv-tile-pipeline` | no-op + 6/8 C=8 race | **Per-seq state isolation** is a prerequisite (Phase 2) |
| `kv34-sync-deferral` | no-op at C=1 | **Don't redo sync-deferral** — overflow decode is multi-tile I/O-bound, not sync-bound |

## North star

`scripts/dev/agentic_test.py` at **8 concurrent** multi-turn tool-calling agents, in the
**realistic ~32K-resident regime** (not cap=8). Dual-gated:
- **(A) win**: Marconi snapshot hit-rate ~0 → majority; warm-turn TTFT drops
  (recompute-cost → tier-restore-cost); overflow-tail decode tok/s up on 35B.
- **(B) correctness**: C=8 overflow recall **parity** with C=1; flag-off bit-identical to main.

## Phased roadmap

| # | phase | goal | risk |
|---|---|---|---|
| **0** | Instrument + enlarge Marconi pool | Measure the SSM thrash + realistic baseline; prove pool-size (not a bug) is the loss. Sweep `ssm_cache_slots` (16→128), add hit-rate / warm-TTFT / read-bytes / CPU telemetry. **Cheapest lever first.** | very low (config+telemetry; watch HBM OOM) |
| **1** | **SSM snapshots onto the cascade — spill-not-drop** | *Headline win.* Second cascade instance (`GroupLayout num_layers=1`, synthetic elem so `group_stride == num_ssm_layers×(h+conv)`) reusing every hardened byte-mover — zero change to trait/GroupKey/ScratchPool/KV-attn. `SnapshotEntry` → `Location{Hbm(slot)｜Tier(key)}`; evict **spills** not drops; prefix-lookup faults back in. Gated `$ATLAS_SSM_TIER` (default off = byte-identical). | low-med — **cross-stream ordering** vs `wait_snapshot_saves_dispatch` (a fault-in reading a half-written snapshot = silent state corruption); pin slots during fault-in |
| **2** | **Per-seq orchestrator isolation** | Move transient orchestrator state (copy_stream, S-planes, events, `war_armed`) off the shared thread-local `HighSpeedSwap` into a per-seq context. Ships no perf; **unblocks 3 & 5**. Re-state the `unsafe Sync` soundness. | med — under-scoping reintroduces the exact 6/8 C=8 race that killed the tile-pipeline |
| **3** | Cross-layer prefetch (deep lever A) | Hide tier reads behind the **SSM+MoE/FFN compute between two attention layers** (not the thin within-layer slice the pipeline tried) — for both KV and SSM fault-ins. The only overlap with enough compute. | med — a pin bug silently corrupts attention (wrong-layer blocks); must be measured at the agentic regime not cap=8 |
| **4** | Reduce read volume — **native-quant tier storage** (deep lever B) | Stop the BF16 inflation: `decode/high_speed_swap.rs:232-289` dequants FP8/NVFP4/4-bit → BF16 *before* `write_from_host`, so quantized history is stored + re-read at **2–4×**. Store native-quant bytes; dequant on read. **Biggest concrete byte cut.** | med — wide per-format correctness surface (FP8/NVFP4/Turbo3/4/8); needs a per-format numeric-diff harness |
| **5** | Parallel per-seq CPU orchestration (deep lever C) | De-serialize the `multi_seq/attn.rs:203` per-seq loop **without dropping the cache** — kills the observed **2-cores-pegged / spiky-throughput** symptom under C=8. | med-high — highest concurrency risk; breaks `unsafe Sync` if widened silently (UB, not just a race); thread-pool footgun on the shared box |
| **6** | *(optional capstone)* one namespaced `BlobKey` space | Permanently kill KV/SSM re-divergence: one address space, one policy, per-namespace budgets. Only if cross-arbitration / shared budget proves worthwhile. | med — addressing/off-by-one cross-write hazard (KV over SSM bytes); pure hygiene, zero new user value |

## Key open questions
- **Snapshot capacity sizing**: how many warm snapshots/session must survive to convert
  a deep agentic chain? (max_context / checkpoint_interval × sessions) vs HBM/local-RAM budget.
- **Zero-copy SSM restore**: `register_landing_region` assumes one contiguous UMA pool;
  SSM lands into per-layer `SsmStatePool` device ptrs (60 destinations) — bounce first, or
  extend landing registration.
- **Fault-in-vs-recompute crossover**: for shallow prefixes a 60-destination tier read may
  be slower than recomputing a few SSM layers → `Location` may need a cost-aware policy.
- **Ownership/threading**: HSS is `thread_local` on the scheduler thread; SSM pools are
  `Arc`-shared on `TransformerModel` — Phase 1 must reconcile these.
- Whether Phase 6 is ever worth its wide-diff risk given the two-instance version already
  delivers all measured value.

## Task mapping
- **#36** = Phase 0+1 (SSM spill tier — the headline). **#35/#34** are demoted/absorbed:
  the throughput levers live in Phases 3/5, gated on the Phase-2 per-seq isolation, and
  #34's sync-deferral is explicitly *not* redone.

## Live baseline (2026-07-07, 8-concurrent agentic, 35B, 8K-resident)
Serving correct (0 errors / 0 exhausted); tier ~idle (contexts < cap); **SSM 16 saved /
0 hits** (the thrash this plan targets); ~4 tok/s/session. Confirms: at realistic scale
the win is SSM tiering + decode throughput, not cap=8 overflow-streaming micro-opts.
