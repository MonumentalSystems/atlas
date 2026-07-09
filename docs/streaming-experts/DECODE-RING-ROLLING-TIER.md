# Rolling hot-HBM / cold-spill SSM decode-rollback ring

Status: DESIGN (synthesized from three competing angles — reuse-blobstore,
purpose-built, uma-hybrid). Owner: streaming-experts. Date: 2026-07-08.

## IMPLEMENTATION STATUS (2026-07-08) — WIRED LIVE + GPU-VALIDATED

Landed on `feat/streaming-experts-mvp`, gated OFF by default
(`ATLAS_SSM_DECODE_RING_ROLL`), byte-identical when the flag is unset. Compiles;
**unit tests pass** (13 `decode_ring_manager` incl. `reset_seq`, 13 `ssm_decode_ring`,
18 `rollback`, 17 `ssm_tier`).

**GPU-validated on Holo-3.1-35B (C=8, cold=local NVMe, target-kv-tokens 100000,
fp8+calibration, full CUTLASS/grouped-MoE fast set):**
- Rolling tier engaged: `2 hot + 1 margin lanes/seq, 8 fault-scratch`; cold tier
  `LOCAL NVMe … 65 slots (non-dropping ≥ min_slots 64)` — the adversarial-fix arena
  sizing (`ring_slots × max_batch` = 8C) is live.
- **Correctness: both waves 8/8 ok** — wave 2 recycled the seq-slots, so the
  `reset_decode_ring_seq` path (cross-sequence reuse) produces coherent restores.
- **HBM: 4080 → 2040 MB** at C=8 (decode region halved; L_phys=3 vs 8 logical) →
  projects ~32 → ~12.75 GB at C=64.
- **Decode cost: ~9–10%** vs the HBM baseline (38.97 → 35.3 tok/s w/ CUTLASS),
  decode-bound single-thread spill overhead — real, not free. This is the high-C
  HBM-vs-throughput trade; the cost is worth it only where the 32 GB HBM ring is the
  binding constraint.

### The live wiring (what drives the manager)
- `save_decode`/`restore_decode` branch to `save/restore_decode_managed` when
  `ATLAS_SSM_DECODE_RING_ROLL` is set (existing scheduler `save/restore_decode_ssm_snapshot`
  calls route through unchanged).
- `reset_decode_ring_seq` (trait → `SsmSnapshotPool::reset_decode_seq` →
  `DecodeRingManager::reset_seq`) fires from `rollback.rs::snapshot_boundary_if_ssm` when
  the scheduler ring `is_empty()` (fresh sequence recycled the seq-slot, or full truncate)
  — prevents a new sequence inheriting the prior one's stale lanes/LRU/residency.
- `drop_decode_ring_slot` fires for each slot returned by `SsmDecodeRing::truncate_after`
  (partial rollback) — frees discarded lanes so `plan_save` can't backpressure with an
  empty drain (deadlock).

### Multi-worker spill pool (landed, gated conservative)
The single-thread spill worker is now a **sequence-sharded pool of N workers**
(`ssm_snapshot.rs`). Each worker owns an independent `mpsc` channel **and its own
CUDA stream**; `decode_enqueue_spill` routes a job to
`spill_worker_index(seq, N)` (a pure `splitmix(seq) % N`, in `decode_ring_manager.rs`).
Because `cold_key` is a pure function of `(seq, logical)`, all spills of a
sequence — hence every incarnation sharing a `cold_key` — stay FIFO on ONE worker,
byte-reproducing the single-consumer `put → complete_spill → remove` order the
epoch guard was designed around. **No `DecodeRingManager` state changed**;
coherence comes from routing, not new locking. `N == 1` is byte-identical to the
pre-pool build.

Knobs (`atlas-kernels`, runtime env, pure-fn + unit-tested):
- `ATLAS_SSM_DECODE_SPILL_WORKERS=<1..8>` — explicit pool size. Unset ⇒ tier-aware
  default via `ATLAS_SSM_DECODE_TIER`: **`nvme`=4** (disjoint-offset `pwrite`
  genuinely parallelizes), **host-RAM (unset)=1**, **`peer`=1**.
- `ATLAS_SSM_DECODE_SPILL_MARGIN=<n>` — async-drain lanes/seq, default
  `DECODE_SPILL_MARGIN(1)`, floored 1, ceilinged so `hot+margin ≤ ring depth`.
  Consumed INSIDE `decode_hbm_lanes_per_seq` (the SSOT) so the pool alloc and the
  preflight HBM reservation widen in lockstep; costs `margin × max_batch × layers ×
  (h+conv)` HBM per raised lane. Worker-count and margin are the coupled levers that
  shrink the backpressure/sync-fallback window at high C.

**Scope / honesty:** the pool recovers throughput on **local NVMe** (and trivially
host-RAM); the **RDMA `peer` tier stays serial** — a shared `RdmaSnapshotArena`
holds one `Mutex` across two TCP RTTs + the RDMA write, so extra workers do not
scale it. Per-worker peer connections are **out of scope** (a `put` on connection A
is absent from connection B's slot map, breaking restore). No measured perf win is
claimed here: only ~9–10% at C=8 was ever GPU-measured; the ~10–13% high-C figure
is a projection, and the pool's recovery is a **docker A/B follow-up** (§9). The
flat `ATLAS_SSM_DECODE_RING_UMA` knob (no physical cap on GB10) remains the separate
high-C escape valve.

<details><summary>original scaffold status (superseded)</summary>

Landed gated OFF, compiled + 29 unit tests:

- **Built (pool side):** `decode_ring_manager.rs` (residency state machine
  Absent/Resident/Spilling/Cold + epoch guard + backpressure), `ssm_snapshot.rs`
  (`DecodeRingManager` wiring, async spill worker, `save/restore_decode_managed`,
  `spill/fault_in_decode_slot`), `ssm_tier.rs` (`FileSnapshotArena` for local
  NVMe, `RdmaSnapshotStore`→`ArenaSnapshotStore`, `build_decode_tier_store` with
  nvme/peer/host-RAM backends), `impl_a1.rs` (store build + spiller spawn),
  `ssm_decode_ring.rs` (record/truncate_after return evicted entries).
- **Adversarial-verify fixes applied:** NVMe cold arena sized to the true
  worst-case key cardinality `ring_slots × max_batch` (was `(ring_slots −
  hot_lanes) × max_batch` → would drop a LIVE rollback target at C=64/nvme); a
  failed `store.put` no longer commits `Cold` (leaves the slot `Spilling`, lane
  pinned, bytes safe) instead of erasing the only copy.

- **NOT DONE — required before this is a live feature:**
  1. **Scheduler dispatch is not wired.** `rollback.rs` / `verify_a.rs` /
     `trait_impl/mod.rs` still call the OLD flat `save_decode`/`restore_decode`;
     the rolling manager is currently dead code (hence `#![allow(dead_code)]`).
     Route the dispatch to `*_managed`, thread `token_position`/logical slot,
     and drive fault-in-before-restore on `rollback_to_boundary`.
  2. **`drop_decode_slot` has no caller** — wire it into
     `SsmDecodeRing::truncate_after` + sequence teardown so evicted logical
     slots' cold keys are actually reclaimed (otherwise the cold store holds the
     full `ring_slots × max_batch` residency forever — bounded, but not the
     `(ring_slots − hot_lanes)` the HBM-cap doc assumes).
  3. **GPU validation** — prove bit-exact re-steer through a cold fault-in, and
     A/B the high-C perf (the single-thread spill worker + margin=1 can hit
     steady-state backpressure at C=64, per the PERF lens — measure before
     trusting the 12 GB@C=64 cap in practice).
  4. Supersede/remove the flat `ATLAS_SSM_DECODE_RING_UMA` knob once the rolling
     tier is live (UMA gives no physical cap on GB10 — see §on the uma-hybrid).

</details>

## 0. Problem & thesis

The Phase-C re-steer watchdog keeps a per-sequence ring of SSM boundary
snapshots so a degeneration rollback can restore **bit-exact** recurrent state.
It is pure HBM:

```
DECODE_ROLLBACK_RING_SLOTS(8) × max_batch_size × num_ssm_layers × (h+conv)
   ≈ 4 GB at C=8,  ≈ 32 GB at C=64
```

The ring is **written every sentence boundary** (hot D2D) and **read only on a
watchdog rollback** (≤ `ROLLBACK_RESTEER_CAP=2` per seq, never on streaming).
The existing flat escape valve — `ATLAS_SSM_DECODE_RING_UMA=1`, whole ring in
`cuMemAllocManaged` — was measured at **−5.8% decode** (37.31→35.14 tok/s,
Holo-3.1-35B, C=8), because the per-boundary D2D into managed memory is slower
than HBM→HBM. It also buys **no physical cap on GB10** (managed pages live in
the same LPDDR; they free bytes only by swapping to disk).

**Thesis.** Keep the ring's logical depth at 8 and keep `SsmDecodeRing` as the
recency authority, but shrink the *physical* HBM to `L_phys = K_hot + 1` lanes
per active seq. The most-recent `K_hot` boundaries stay HBM-resident (every
boundary save is an HBM→HBM D2D — constraint 1). Aged boundaries spill
**asynchronously** on a side stream + worker (constraint 2) into a **second,
non-dropping `SnapshotBlobStore`** — operator-selectable local NVMe or the
gx10:9920 paging peer. A rollback that targets a cold boundary faults it back
(`fault_in`-style, ~26–57 ms, capped, never streaming) **before** the restore
read (constraints 3, 4). At C=64, K_hot=2 the ring is capped at **~12 GB HBM**
(vs 32 GB), independent of logical depth and generation length.

We **reject the managed/UMA middle tier on GB10** (uma-hybrid's own honest
verdict): on physically-unified LPDDR a managed WARM slot that stays resident
occupies the identical bytes it would in HBM — it caps only `cuMemAlloc`
accounting, not physical footprint, and leaning on it re-introduces the −6% via
demand-paging on the write path. The only real relief is HBM-hot → NVMe/peer
cold. `ATLAS_SSM_DECODE_RING_UMA` is superseded and should be removed.

## 1. The partition: INTRA-SEQ-DEPTH (proven the only cap at all-active C)

Across-seq partitioning (reclaiming idle seq-slots' rings) yields **nothing**
when all `C = max_batch_size` sequences are active — that is the whole point of
the task. The only partition that caps HBM at all-active C is *within* each
seq's 8 logical boundaries:

- **HOT** = the `K_hot` most-recent boundaries per active seq. Boundary saves
  (`save_decode`, `ssm_snapshot.rs:237`) land here as HBM→HBM D2D.
- **COLD** = the deeper `8 − K_hot` boundaries, spilled async to the tier.

Justified by the measured access asymmetry (write:read ≈ hundreds:0–2):

- **Write** is round-robin into the newest lane, at every boundary, on the
  decode critical path → must be HBM.
- **Read** fires only on a watchdog rollback. `find_last_boundary_with_snapshot`
  (`rollback.rs:137`) scans backward and returns the **shallowest eligible**
  boundary ≥ `min_keep` back — normally within the hot window. Only a deep
  re-steer (large watchdog `min_keep`, multi-boundary attractor) reaches a cold
  boundary → the rare cold fault is affordable. **All 8 live entries remain
  legitimate targets**, so every one must be resident *or* faultable.

## 2. Physical/logical model & the HBM cap

Logical depth stays 8 (`SsmDecodeRing.capacity` unchanged). Physical HBM region
is re-sized at `ssm_snapshot.rs:147`:

```
decode_region = decode_max_seqs * L_phys        // was decode_max_seqs * 8
L_phys        = K_hot + SPILL_MARGIN            // recommend 2 + 1 = 3
```

still via `gpu.alloc` (HBM — **not** `alloc_managed`; the UMA branch
`:169-185` is deleted). Plus a small **shared** fault-scratch pool `F` lanes
(recommend 4–8 lanes global, ≥ `ROLLBACK_RESTEER_CAP` concurrent rollbacks).

**Why `SPILL_MARGIN ≥ 1` is fundamental.** True async spill means the demoted
lane's D2H gather is still draining when the next boundary save needs a lane.
If the save reused the draining lane it would either corrupt the cold blob
mid-gather or have to stall on the drain (violating constraint 1). So one spare
drain lane per seq is required: the save always writes a lane that is neither
hot-resident nor draining. `SPILL_MARGIN=1` gives zero critical-path stalls.

**Cap at C=64, all active** (blob = `num_ssm_layers×(h+conv)` ≈ 64 MB =
`spill_blob_bytes()`, `ssm_snapshot.rs:457`):

| config | decode HBM @ C=64 | cold residue → tier | per-boundary write |
|---|---|---|---|
| today (8 HBM) | 64×8×64MB = **32 GB** | 0 | HBM D2D (fast) |
| UMA knob | 0 HBM (managed) | 0 | managed D2D, **−5.8%** |
| **this, K_hot=2, margin=1** | 64×3×64MB = **12 GB** + F·64MB | 64×5×64MB ≈ 20 GB | HBM D2D (fast) |
| this, K_hot=1, margin=1 | 64×2×64MB = **8 GB** + F·64MB | 64×6×64MB ≈ 24 GB | HBM D2D (fast) |

The cap is `C × L_phys × 64 MB + F·64 MB`, **independent of logical depth 8 and
of generation length / rollback depth**, because deep boundaries never occupy
HBM. That is a real cap at all-active C. Recommended default **K_hot=2 ⇒ 12 GB**
(2.7× cut, one hot rollback margin); K_hot=1 ⇒ 8 GB but faults on nearly every
deep re-steer. Host bounce memory (`inflight × 64 MB`) also competes on LPDDR —
a few GB worst case at C=64.

`preflight.rs:118-125` MUST mirror this: change the decode term from
`DECODE_ROLLBACK_RING_SLOTS × max_batch_size` to
`L_phys × max_batch_size + F` (SSOT — the `:109-117` comment already warns of
exactly this decoupling hazard).

## 3. Residency state machine (the correctness core)

Residency lives in `SsmSnapshotPool` (device truth), indexed by
`(ssm_slot, logical_slot)` via `decode_flat_index`. Per logical boundary:

```
enum PhysResidency {
    Absent,
    Resident { lane: u32 },                         // HBM lane, canonical
    Spilling { lane: u32, key: u64, epoch: u64 },   // lane bits STILL VALID; put in flight
    Cold     { key: u64 },                           // present in a NON-DROPPING store
}
```

Invariant: a live ring entry is always **Resident**, **Spilling** (lane still
holds valid bits), or **Cold** (key present in a store that never drops). The
only off-HBM transition is `Spilling → Cold`, and it flips **only after the
async `store.put(key,blob)` returns `Ok`** (worker signals completion). Until
then the entry stays `Spilling`, its lane reserved.

Key mint: decode boundaries have no Marconi prefix hash, so
`key = splitmix64(seq_uid ^ (token_position << 3) ^ DECODE_DOMAIN)`. A
`DECODE_DOMAIN` salt + a **separate store instance** (§5) keep decode keys off
Marconi's namespace.

**Epoch guard (cancel-during-spill).** Each `(ssm_slot, logical_slot)` carries a
generation counter bumped on eviction/truncate. The spill worker commits
`Spilling → Cold` only if the epoch is unchanged; if the entry was truncated or
round-robin-evicted meanwhile, the worker instead `store.remove(key)`s and frees
the lane. This closes the race where a `store.put` completes *after* the ring
already dropped the entry (otherwise `truncate_after`'s `remove(key)` would race
a not-yet-present key and leak the arena slot).

## 4. Write path — async spill + ordering (the hot path)

Call chain unchanged: `decode_logits_step.rs:564` →
`rollback::snapshot_boundary_if_ssm` (`rollback.rs:311`) → `ring.record()` →
`model.save_decode_ssm_snapshot(seq, logical_slot)` → dispatch
(`verify_a.rs:379`) → **new** `DecodeRingManager::save_decode_managed`.

On the **decode (default) stream** only:
1. Pop a free phys lane `L` from the per-seq free list (non-empty by
   `L_phys = K_hot+1`).
2. `save_decode` D2D live pool → `lane_h/conv[L]` — identical to
   `ssm_snapshot.rs:237-261`, addressing `L` instead of `flat`. **HBM→HBM,
   constraint 1.**
3. `record_event` a per-lane `last_write_event` (`gpu.rs:283`).
4. `Resident{lane:L}`; push `logical_slot` on the per-seq MRU.
5. If MRU length `> K_hot`: pop the LRU resident logical, flip it
   `Resident{lane} → Spilling{lane,key,epoch}`, and **enqueue**
   `(ssm_slot, logical, lane, key, epoch, last_write_event)` to the spiller.
   Return immediately.

The decode thread does **only** D2D + event + enqueue — no D2H, no `store.put`,
no `synchronize` on the boundary path. **Constraint 2 honored.**

**Spill worker** (dedicated thread, `gpu.bind_to_thread()`, own
`decode_spill_stream = gpu.create_stream()`, pinned host bounce per in-flight
spill):
1. `stream_wait_event(decode_spill_stream, last_write_event)` (`gpu.rs:288`) —
   GPU-side wait, no CPU block; guarantees step-2 D2D landed before the gather
   reads the lane. (This replaces `spill_slot`'s leading `synchronize`, which
   would have blocked the decode thread.)
2. Per-layer `copy_d2h_on_stream` gather into the pinned bounce;
   `synchronize(decode_spill_stream)` **on the worker thread**.
3. `store.put(key, &blob)`.
4. Under the manager lock: if epoch current and entry still `Spilling` →
   `Cold{key}`, return lane to the per-seq free list. Else (`epoch` bumped) →
   `store.remove(key)`, return lane.

**Backpressure.** Bounded queue; if it fills (slow NVMe/peer at high C), the
demotion falls back to a **synchronous** spill (rare critical-path stall) rather
than unbounded host-RAM growth. Counter exposed. This never happens on the
common path; it is the safety valve for the C=64 spill firehose (§8).

## 5. Read path — fault-before-rollback (the rare path)

Call chain unchanged: `decode_logits_step.rs:664` →
`rollback::rollback_to_boundary` (`rollback.rs:203`). After
`ring.slot_for_position(keep_len)` yields logical slot `L`, dispatch to **new**
`restore_decode_managed` **before** buffer truncation (preserving the existing
decline-clean ordering at `rollback.rs:265-283`):

```
match residency[ssm_slot][L] {
  Resident{lane} | Spilling{lane,..} =>
      restore_decode(lane) D2D → live pool          // as today, bit-exact
  Cold{key} => {
      let fl = fault_scratch.pop();                 // shared scratch lane
      fault_in_decode_slot(key, fl):                // store.get → copy_h2d_async
          → synchronize(stream)                     // COMMIT before read
      restore_decode(fl) D2D → live pool            // reads faulted lane
      fault_scratch.push(fl);
  }
}
```

**The `Spilling` case reads the pinned lane directly** — the spill is a *read*
of the lane, its bits are still valid, so no fault and no wait. The lane stays
pinned (not reused) until the spill completes or truncate cancels it. This is
the key ordering trick that makes async spill invisible to the rare read.

The `Cold` path mirrors `fault_in_slot` verbatim, whose trailing
`synchronize(stream)` (`ssm_snapshot.rs:554`) guarantees the H2D scatter
committed **before** the D2D restore reads the lane (constraint 4). Latency
~26–57 ms (peer) / lower (local NVMe), inside `ROLLBACK_RESTEER_CAP=2`,
watchdog-gated, non-streaming-only (`rollback.rs:214` `StreamUnsafe`) —
constraint 3. Streaming still declines the read but keeps writing+spilling every
boundary, which is fine.

A `store.get → false` (miss) on a live target would be **SSM corruption** (not a
recompute as in Marconi). By construction it is unreachable: the store is
non-dropping and the key is present (§6). A `false` is a hard bug, asserted.

## 6. Cold backend — one seam, both destinations

Reuse the existing **`SnapshotBlobStore`** trait (`ssm_tier.rs:42`) — host-byte
oriented, `u64`-keyed; **all** device gather/scatter + stream ordering already
happen in the pool before the store is touched (`ssm_tier.rs:259-263`). No new
trait. Build a **separate instance** from Marconi's `ssm_tier_store` via a new
`build_decode_tier_store(blob_bytes, min_slots)` next to `build_tier_store`
(`ssm_tier.rs:189`), with its own env namespace so capacity budgets and keys
never collide.

- **(a) local NVMe** — `ATLAS_SSM_DECODE_TIER=nvme` + `ATLAS_SSM_DECODE_NVME_DIR=/path`.
  Add one new type `FileSnapshotArena: SnapshotTransport` (`ssm_tier.rs:273`):
  `pwrite`/`pread` at absolute offset, O_DIRECT with a pinned bounce for
  alignment (as `posix.rs:65-68`), io_uring default → posix fallback honoring
  `ATLAS_KV_BACKEND` (mirroring `high_speed_swap.rs:41-88`). Run the **existing**
  offset-addressed fixed-slot store over it unchanged (rename
  `RdmaSnapshotStore → ArenaSnapshotStore` — it already takes
  `Box<dyn SnapshotTransport>`, `ssm_tier.rs:406`). **This `FileSnapshotArena` is
  the only genuinely new store machinery.**
- **(b) RDMA peer** — `ATLAS_SSM_DECODE_TIER=peer` +
  `ATLAS_SSM_DECODE_RDMA_TIER=gx10:9920` (+ paging). Reuse
  `RdmaSnapshotArena::connect_paging` + `PagingSnapshotStore`
  (`rdma_snapshot.rs:136`, `ssm_tier.rs:333`) exactly as the Marconi tier does,
  with its own `ATLAS_SSM_DECODE_NS` fold. `paging_put` **never rejects** (peer
  LRU-spills coldest to its own O_DIRECT NVMe) — non-dropping for free. ~26.5 ms
  RDMA-read of one 66 MB blob (measured, `HANDOFF-2026-07-08-ssm-tier.md`).
- default (unset) → `MemBlobStore::new(0)` (unbounded host RAM).

**Non-dropping requirement (hard).** A rejected/dropped put = lost rollback
target = corrupt restore. The decode tier MUST be one of: `MemBlobStore::new(0)`,
`PagingSnapshotStore` (never rejects), or `ArenaSnapshotStore` (local NVMe)
**provably sized ≥ `(8 − K_hot) × max_batch_size` slots** (24 GB at C=64,K=2).
An undersized local arena is a **hard preflight error, not a warn** — rejection
becomes an impossible state, surfaced at init.

## 7. File-by-file change plan

1. `crates/atlas-kernels/src/lib.rs` — add `DECODE_HOT_LANES=2`,
   `DECODE_SPILL_MARGIN=1`, `DECODE_FAULT_SCRATCH`, `DECODE_DOMAIN` salt; keep
   `DECODE_ROLLBACK_RING_SLOTS=8` (logical) and `ROLLBACK_RESTEER_CAP=2`. Fix
   the stale `ssm_decode_ring.rs:24` "CAP+1" doc.
2. `crates/spark-model/src/model/ssm_snapshot.rs` — resize `decode_region` to
   `decode_max_seqs × L_phys` (`:147`); delete the `ring_uma` branch
   (`:169-185`); `decode_flat_index` (`:292`) indexes by `L_phys`; add
   `DecodeRingManager` (residency map, per-seq free lists, shared fault-scratch,
   MRU, epoch, spiller handle); add `save_decode_managed`,
   `restore_decode_managed`, `drop_decode_entry`, `spill_decode_slot`,
   `fault_in_decode_slot`; factor the per-layer gather/scatter body of
   `spill_slot`/`fault_in_slot` (`:472-565`) into shared
   `gather_blob(vecs,stride,flat)`/`scatter_blob(...)` used by both regions
   (constraint 6). `spill_blob_bytes` (`:457`) reused unchanged.
3. `crates/spark-model/src/model/ssm_tier.rs` — add `FileSnapshotArena:
   SnapshotTransport`; rename `RdmaSnapshotStore → ArenaSnapshotStore` (keep
   alias); add `build_decode_tier_store` (nvme/peer/mem branches, all
   non-dropping; hard-assert local arena `slots ≥ min_slots`).
4. `crates/spark-model/src/model/impl_a1.rs` (`:159-205`) — read `K_hot` from
   `ATLAS_SSM_DECODE_HOT_SLOTS` (default 2 → tiering on; 8 → off = today's
   behavior); pass `L_phys=K_hot+margin`, `F` to `SsmSnapshotPool::new`; build
   the second store `ssm_decode_tier_store` via
   `build_decode_tier_store(spill_blob_bytes(), (8−K_hot)*max_batch_size)`;
   create `decode_spill_stream`; spawn the spill worker.
5. `crates/spark-model/src/model/trait_impl/verify_a.rs` (`:379-419`) +
   `trait_impl/mod.rs` (`:247-252`) — route save/restore dispatch to the managed
   methods; thread `token_position`/`logical_slot`; add
   `decode_hot_lanes()`; keep `decode_rollback_ring_slots()`=8.
   `traits/model.rs:346` — default no-op/0 for pure-attention.
6. `crates/spark-server/src/scheduler/ssm_decode_ring.rs` — `record` (`:98`) and
   `truncate_after` (`:149`) return the evicted/dropped entries so the caller can
   drive `drop_decode_entry` (free lane / cancel spill via epoch bump /
   `store.remove`). Capacity, round-robin, selection logic unchanged.
7. `crates/spark-server/src/scheduler/rollback.rs` — `snapshot_boundary_if_ssm`
   (`:311`) passes `token_position` + dispatches the spill emitted by `record`;
   `rollback_to_boundary` (`:203`) dispatches drops from `truncate_after`
   (`:282`); restore stays **before** truncate; seq teardown `store.remove`s all
   cold keys (leak guard).
8. `crates/spark-server/src/scheduler/decode_logits_step.rs` (`:564,:664`) —
   call structure unchanged.
9. `crates/spark-server/src/main_modules/serve_phases/preflight.rs`
   (`:118-125`) — reserve `L_phys × max_batch_size + F` (was
   `DECODE_ROLLBACK_RING_SLOTS × max_batch_size`). SSOT — move in lockstep.
10. `SsmDecodeRing::new(...)` call sites (`lifecycle.rs:319`,
    `prefill_a_step.rs:420,509`, `prefill_b_step.rs:255,340`,
    `phase_promote_prefills.rs:70,188`) — thread `seq_uid`/`hot_lanes` args
    (mechanical).
11. `crates/spark-storage/src/{high_speed_swap.rs,backend/{io_uring,posix}.rs}` —
    reused by `FileSnapshotArena`; `rdma_snapshot.rs` peer path reused verbatim.
    `crates/spark-runtime/src/gpu.rs` + `cuda_backend/gpu_impl.rs` — no new
    primitives (`create_stream`, `record_event`, `stream_wait_event`,
    `copy_d2h_on_stream`, `copy_h2d_async`, `bind_to_thread` all exist).

## 8. Correctness argument

**Bit-exact re-steer.** The cold blob is a verbatim D2H gather (no transform) of
each boundary's per-layer `[h|conv]`, stored byte-for-byte, faulted back H2D
verbatim into an HBM lane; `restore_decode` then does the *identical* D2D it does
for a hot lane. Restoring from a Resident lane, a Spilling (pinned) lane, or a
faulted-back lane all produce identical device bytes — the same round-trip
Marconi already relies on. No quantization, no lossy step anywhere.

**Resident-or-faulted-before-read.** §5 dispatch guarantees: `Cold` ⇒
`store.get` + H2D scatter + `synchronize` **precede** the D2D restore;
`Spilling` ⇒ read the still-valid pinned lane; `Resident` ⇒ D2D as today. The
trailing sync in `fault_in_decode_slot` orders the scatter before the read.

**Never lose a valid rollback target (constraint 4).** At every instant a live
entry is Resident, Spilling (lane valid), or Cold in a non-dropping store — there
is no window where it is neither. `Spilling → Cold` flips only after
`store.put` returns `Ok`. The epoch guard makes a spill that finishes after its
entry was truncated a no-op (`store.remove` + free lane), not a stale `Cold`
commit that would leak or mis-key. Any entry dropped by ring eviction or
`truncate_after` is `store.remove`d and never targeted again (its
`token_position` left the ring, so `slot_for_position` cannot return it).

**No-shared-lane.** `record`'s by-slot eviction (`ssm_decode_ring.rs:116`) plus
the manager's per-seq free list ensure one phys lane maps to at most one live
logical slot at a time.

**HBM cap at C=64.** Resident = `C × L_phys × 64 MB + F·64 MB`, independent of
ring depth (§2). Preflight reserves exactly this.

**No synchronous spill on the boundary-write path.** The decode thread's only
tier-adjacent action is a bounded-queue enqueue; the D2H gather, `store.put`, and
all syncs run on the worker/side stream. The only synchronous fallback is the
explicit backpressure valve (§4), which fires only when the queue saturates and
is instrumented — it is a safety stall, never the common path.

## 9. Measurement plan (GPU A/B)

Rig: the harness that produced 37.31/35.14 tok/s — Holo-3.1-35B-A3B
(`ATLAS_TARGET_MODEL=holo-3.1-35b-a3b`), SLAI@100ms, `--ssm-slots 256`, 32K,
INFO logging. Run at **both C=8 (write-bound proof) and C=64 (cap + firehose
proof)**; median of ≥5 runs, pinned clocks.

**Arms.** A0 = today's 8-slot HBM ring (reference, ≈37.31). A1 = flat UMA
(`ATLAS_SSM_DECODE_RING_UMA=1`, ≈35.14, negative control). A2 = this design,
`K_hot=2`, `ATLAS_SSM_DECODE_TIER=nvme`. A3 = this design, `K_hot=2`,
`ATLAS_SSM_DECODE_TIER=peer`=gx10:9920.

**(1) Write stays HBM-fast (primary).** Steady-state decode tok/s: A2/A3 within
±1% of A0 at C=8, and strictly above A1. Recovering the 5.8% gap is the headline
proof the write never left HBM. CUDA-event-timed per-boundary `save_decode` D2D
under `ATLAS_SSM_DECODE_TIMING` must equal A0's D2D, ≪ A1's managed D2D.

**(2) HBM capped.** Pool log line (`ssm_snapshot.rs:195`) + `cuMemGetInfo_v2`
before/after warm-up: A2/A3 decode region = `C × 3 × 64MB` (12 GB at C=64) vs
A0's 32 GB. At C=64, A0 OOMs / steals KV; A2/A3 fit.

**(3) Async spill non-blocking.** Per-boundary decode-step latency p50/p99 (A2/A3
vs A0): no spill-stall tail. Spiller counters (`ATLAS_DECODE_RING_STATS`):
spills/sec, mean spill latency, max queue depth, **zero free-lane waits**,
synchronous-fallback count ≈ 0 at C=8.

**(4) Spill firehose at C=64 (honest stress).** At capped ring, every new
boundary evicts one → spill rate = boundary rate ≈ 128 spills/s × 64 MB ≈ ~8
GB/s sustained D2H + store write (§ risks). Measure aggregate gather + `store.put`
throughput (`ATLAS_SSM_TIER_TIMING`). For A3-peer check against CX7 egress
(~2.5–5 GB/s) — **expect peer to bottleneck at C=64**; for A2-nvme against the
device write ceiling (multi-GB/s, should sustain). This is where local NVMe is
the high-C default and peer is capacity-sharing / lower-C.

**(5) GB10 illusory-relief guard (grafted from uma-hybrid).** At C=64 capture
`/proc/meminfo MemAvailable` + `swapon --show` for A1 vs A2/A3: A1 (managed)
must NOT free ~24 GB physical (bytes stay resident) or, if SwapUsed grows, shows
uncontrolled disk-paging; A2/A3 (explicit tier) must raise `MemAvailable` by the
cold residue with SwapUsed flat — proving only the explicit tier creates real
headroom.

**(6) Bit-exact rollback (CI + GPU).** CI unit test with `MemBlobStore(0)`:
save → force-cold spill → fault back → restore → assert device bytes ==
original. GPU microbench (non-streaming, greedy, fixed seed): a hook forces the
watchdog to re-steer to the **deepest (cold)** boundary; assert
`fault_in+restore` ~26–57 ms and resumed `h_state`/`conv_state` **byte-identical**
to an A0 run rolling back to the same boundary while HBM-resident, and identical
post-rollback token stream. Plus a unit test firing a rollback between spill
enqueue and worker completion (the `Spilling`-lane restore race).

Acceptance: (1) A2/A3 ≥ 0.99×A0 and ≥ A1+5%; (2) HBM = 3/8; (4) C=64 fits and
holds tok/s (NVMe); (6) bit-exact + latency in budget.

## 10. Risks / honest downsides

- The spill firehose is intrinsic, not incidental: capping the ring below 8
  means spill rate = boundary rate (~8 GB/s at C=64). Peer backend may not
  sustain it — local NVMe is the high-C default.
- `SPILL_MARGIN=1` costs an extra `C×64MB` (12 GB not the "ideal" 8 GB at
  K_hot=2). A shared transit pool could recover it but under synchronized
  boundaries at C=64 would need ~C drain lanes anyway and risks stalls — deferred.
- Non-dropping constrains the store: local NVMe arena must be pre-provisioned to
  worst case (24 GB) with a hard preflight error; a miss on a live target is
  corruption, not a graceful recompute (the crucial difference from Marconi).
- Real new correctness surface: residency state machine + Spilling-vs-rollback
  race + epoch cancel-on-truncate. The pinned-lane restore is the subtlest
  invariant — covered by the dedicated race test.
- Two overlapping knobs during transition; remove `ATLAS_SSM_DECODE_RING_UMA`.
- Low value at low C / small models (HBM not binding); gate to engage only when
  `K_hot < 8` requested. Framework-first: keep the mechanism regardless.
- Key-lifecycle leak risk (missed `store.remove`) → per-seq owned key set
  dropped on teardown + periodic reconcile.
- Peer contention: gx10:9920 is the shared atlas-cache-peer; decode spills need
  their own namespace + arena budget.
