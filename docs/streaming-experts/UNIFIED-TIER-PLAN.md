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

## Update 2026-07-07 — PR #278 reconciliation (the operator flagged it; it reshapes Phase 0/1)

[PR #278](https://github.com/Avarok-Cybersecurity/atlas/pull/278)
(`perf/35b-agentic-wall-sub1500`, OPEN, forked from main `6d79e14`) independently
found and fixed *exactly* the thrash this plan targets, from the eviction-victim angle.
It is **not** merged to main and **not** on this branch. Key facts to fold in:

**The measured win (external validation of Phase 0's thesis).** 35B agentic
`webserver_ok` wall Σ **2765 → 1364s** (10/10 ws_ok, 0 caps). Two levers:
- `--ssm-cache-slots 256` (not our default **16**) — a bigger resident pool.
- `ATLAS_SSM_TAIL_PROTECT=1` — the dominant lever. Root cause: `evict_lru`'s
  `escore = last_access·(1+hit_count)` has **no depth term**, so the just-aged deep
  tail (hit_count=0) is evicted before the hot token-8192 anchor (self-reinforced
  hits) → warm restore falls back to 8192 → **~4,400 SSM tokens recomputed/turn
  (~7.6s TTFT)**. Fix: exempt the *live session's deepest* snapshot from eviction.
  Restores-from-8192 50%→~9%, mean recompute 4438→262 tok.
- (also in #278, orthogonal to us: partial top-k sampler + opencode harness pipe-hang fix.)

**WIN recorded — session-aware eviction (ours) and tail-protect (#278) are COMPLEMENTARY, not competing.**
This branch already ships *session-aware* eviction (`snapshot.rs:134`, default ON:
evict from the stalest **conversation** first, protecting the live session's whole
chain vs *dormant* sessions). But it does **not** fix the single-session deep-tail case:
within one live/dominant session, session-freshness doesn't discriminate, so the victim
falls through to lowest-escore = the just-aged deep tail — the very bug #278 fixes.
- session-aware (ours) → protects live session vs **other/dormant** conversations.
- tail-protect (#278) → protects the deep tail **within** the live conversation.
- The north star (8 concurrent × deep) hits **both** failure modes → we want both.

**ANTI-PATTERN recorded — "one eviction heuristic fixes the deep-tail thrash" is false.**
Two independent efforts (this branch's session-aware; #278's tail-protect) each fixed
*half* the problem and each closed #33/#30 believing it was whole. Neither alone covers
concurrent-**and**-deep. The structural Phase 1 (spill-not-drop) **subsumes both**: if an
evicted snapshot spills to the cascade instead of dropping, a fault-in is always available,
so victim *choice* stops gating TTFT correctness — the heuristics only optimize *which*
snapshots stay in fast HBM. That is the real reason Phase 1 is the headline, not a nicer knob.

**Phase 0 re-scoped accordingly:** (a) port tail-protect onto this branch (adapting to
our 4-arg `lookup` w/ `adapter_id`) so it composes with session-aware eviction; (b) adopt
#278's mechanism telemetry (restore-anchor histogram, recompute-tok/turn) plus a
**residual-drop** counter; (c) sweep slots (16→256) and measure the *residual* recompute
that survives 256-slots + both heuristics — that residual is precisely what Phase 1 converts
from recompute → fault-in.

## Phased roadmap

| # | phase | goal | risk |
|---|---|---|---|
| **0** ✅ | Instrument + enlarge Marconi pool | Measure the SSM thrash + realistic baseline; prove pool-size (not a bug) is the loss. Sweep `ssm_cache_slots` (16→128), add hit-rate / warm-TTFT / read-bytes / CPU telemetry. **Cheapest lever first.** | very low (config+telemetry; watch HBM OOM) |
| **1** 🔨 | **SSM snapshots onto the cascade — spill-not-drop** (1a byte-mover+tier ✅ · 1b-core index state-machine ✅ · 1b-integration serving-wiring NEXT, needs live 35B) | *Headline win.* Second cascade instance (`GroupLayout num_layers=1`, synthetic elem so `group_stride == num_ssm_layers×(h+conv)`) reusing every hardened byte-mover — zero change to trait/GroupKey/ScratchPool/KV-attn. `SnapshotEntry` → `Location{Hbm(slot)｜Tier(key)}`; evict **spills** not drops; prefix-lookup faults back in. Gated `$ATLAS_SSM_TIER` (default off = byte-identical). | low-med — **cross-stream ordering** vs `wait_snapshot_saves_dispatch` (a fault-in reading a half-written snapshot = silent state corruption); pin slots during fault-in |
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

## Progress log

### Phase 0 — LANDED (2026-07-07)
Code in `crates/spark-runtime/src/radix_tree/snapshot.rs` (+ `serve_args.rs` guidance):
- **Tail-protect ported from #278** onto this branch's 4-arg `lookup` (`adapter_id`),
  composed with the existing session-aware eviction. Refactored victim selection into a
  pure `session_aware_victim(tail_protect: bool)` so it's unit-testable without mutating
  process env (edition-2024 `set_var` is `unsafe`; `#![deny(warnings)]` is on).
  Gated `ATLAS_SSM_TAIL_PROTECT` (default off = byte-identical), matching #278.
- **Telemetry** (`ATLAS_SSM_SNAP_STATS`): aggregate hit-rate, mean restored-anchor depth,
  **mean recompute-tok-on-hit** (the #278 metric — the residual Phase 1 removes),
  recompute-on-miss, saves, evictions(=drops-today). Env-gated summary every 64 lookups;
  zero hot-path perturbation.
- **Slot guidance**: `--ssm-cache-slots` help now points deep-agentic runs at 256 (+#278
  recipe), default kept at 16 (raising it shifts the VRAM budget for all users).
- **Tests**: 6 unit tests green — deep-tail evicted w/o protect (reproduces #278 root
  cause), survives w/ protect, dormant-session tail still evictable, single-entry pool
  still evictable (no deadlock), lookup latches live session, telemetry hit/recompute math.
- **WIN**: session-aware + tail-protect proven complementary (see reconciliation above).
- **ANTI-PATTERN**: `war_armed` named in the original plan (Phase 1 risk row) **does not
  exist** in the tree — the real construct is the RDMA WAR barrier at
  `rdma_kv_backend.rs:341` and the snapshot-save ordering via
  `wait_snapshot_saves_dispatch` (`async_chkpt.rs:166`). Phase 1 must reconcile against
  *those*, not a phantom flag.
- **OPEN (needs GPU)**: the live 256-slot + tail-protect re-measure of *residual* recompute
  (the number Phase 1 targets). Deferred — dgx-00 is a shared prod box; will run a scoped
  35B agentic pass when headroom is clear, or fold into the Phase-1 validation.

### Phase 1a — LANDED (2026-07-07): spill/fault byte-mover + host-RAM tier
The mechanism that turns *drop* into *spill* — tested end-to-end at the pool layer.
- **`crates/spark-model/src/model/ssm_tier.rs`** (new): `SnapshotBlobStore` trait +
  `MemBlobStore` (bounded host-RAM tier, FIFO evict, byte-budget + telemetry). On GB10 UMA
  this is a *real* T1 tier, not a stand-in: spilling frees a scarce pinned snapshot-pool
  slot while bytes live in abundant LPDDR. Gate helper `ssm_tier_enabled()` (`ATLAS_SSM_TIER`,
  default off = byte-identical drop).
- **`ssm_snapshot.rs`**: `spill_slot` (gather scattered per-layer `(h,conv)` D2H → one blob →
  `store.put`) and `fault_in_slot` (`store.get` → scatter H2D into a slot). Both close their
  half of the **cross-stream ordering hazard**: spill `synchronize(stream)`s to drain the
  in-flight `save` before reading (no half-written spill); fault-in `synchronize`s after the
  H2D so the caller's `restore` can't read an un-committed slot.
- **Tests (9 green)**: headline **spill→fault-into-a-different-slot is bit-for-bit identical**
  (on `MockGpuBackend`, no GPU needed); absent-key = clean miss; wrong-size = refused;
  cap-FIFO evict; over-cap blob refused; overwrite reclaims bytes.
- **DESIGN DECISION (learning)**: **host-mediated**, not zero-copy device-landing. Snapshot
  state is scattered across `2×num_ssm_layers` device allocs, but `StorageBackend::read` lands
  ONE contiguous blob at ONE ptr — mismatched. And `MockGpuBackend::copy_d2d` is a no-op, so a
  D2D-scatter path couldn't be byte-tested at all. Host-mediation (D2H-gather / H2D-scatter) is
  correct, matches the plan's "bounce first / 60 destinations" open question, *and* is fully
  CPU-unit-testable. Zero-copy landing stays a later perf optimization, not a correctness need.
- **ANTI-PATTERN avoided**: did NOT half-wire spill-on-evict without fault-in-on-lookup — a
  spill nothing ever reads back is pure wasted I/O. Spill and fault-in must land together in the
  index (Phase 1b), so 1a ships the *mechanism* proven end-to-end and defers the atomic wiring.

### Phase 1b-core — LANDED (2026-07-07): index `Location{Hbm｜Tier}` state machine
`crates/spark-runtime/src/radix_tree/snapshot.rs` — new **gated, unit-tested** methods on
`SsmSnapshotIndex`; flag-off is byte-identical (no entry is ever tiered when `ATLAS_SSM_TIER`
is unset, so every default path is unchanged and the existing 45 radix tests still pass):
- `SnapshotEntry.tiered: bool`; `SnapLoc{Hbm(slot)｜Tier(key)}` + `SnapMatch`.
- `evict_to_tier()` → spill victim (same session-aware/tail-protect policy, HBM-resident only),
  marks it spilled, returns `(freed_slot, key)`; **entry kept** (findable), not removed. `None`
  when nothing resident remains (caller must not spin).
- `lookup_tiered()` → deepest anchor + where it lives; feeds Phase-0 telemetry + `tier_hits`.
- `promote(key, new_slot)` → re-home a faulted-in entry to HBM.
- `session_aware_victim` gains `skip_tiered` + returns `Option` (a spilled entry holds no slot →
  never a drop victim; freeing its stale id would double-free). Old `lookup`/`evict_lru` skip
  tiered defensively.
- **11 snapshot tests green**: spill-not-remove, spilled-entry lookup semantics (invisible to
  `lookup`, `Tier` via `lookup_tiered`), promote→Hbm, None-when-all-spilled, reinsert-un-spills.
- **ANTI-PATTERN avoided**: a spilled entry MUST NOT be a drop/`evict_lru` victim — its
  `snapshot_id` is stale, so freeing it would return an already-free slot (double-free /
  slot-aliasing). `skip_tiered` enforces this; a test asserts `evict_lru` frees only the resident.

### Phase 1b-integration — NEXT (serving-path wiring; needs live 35B validation)
The tested core (1a mechanism + 1b state machine) is complete; this is the plumbing that makes
it fire in serving, all gated `ATLAS_SSM_TIER` (default off = byte-identical, so it can merge
and be validated opt-in — the same shipping model #278 used for tail-protect):
- Own an `Arc<dyn SnapshotBlobStore>` on `TransformerModel`; expose spill/fault through the
  `PrefixCache` trait (or a thin sibling) so `reclaim_from_cache` calls `evict_to_tier` →
  `pool.spill_slot(freed, key, store, gpu, stream)` instead of `free`, and the prefill
  prefix-lookup site calls `lookup_tiered`; on a `Tier` hit → claim a slot →
  `pool.fault_in_slot(slot, key, store, gpu, stream)` → `promote(key, slot)` → restore as usual.
- Thread store+gpu+stream into `reclaim_from_cache` and the 4 exhaustion branches
  (`save_checkpoint`/`decode_checkpoint`/`finalize_last`/`prefill_d`) + the prefill lookup.
- **Cost-aware guard** (plan open q): for shallow prefixes a 60-destination fault may cost more
  than recomputing a few SSM layers → depth threshold before faulting.
- Order against `wait_snapshot_saves_dispatch` (`async_chkpt.rs:166`) — the real construct, not
  the plan's phantom `war_armed`. (1a's `spill_slot`/`fault_in_slot` already `synchronize` their
  own half; the integration must additionally not spill a slot whose save event is unrecorded.)
- **Validate**: gated 35B agentic pass — the `ssm-snap-stats` line should show `tier_hits` rising
  and `mean_recompute_on_hit` dropping toward the tail-protect+256 residual. **Held for GPU
  headroom** on the shared dgx-00 (Holo/AEON/hub co-resident); wiring blind without this live
  read would repeat the plan's own anti-pattern (3 prior untested-at-scale attempts failed).
