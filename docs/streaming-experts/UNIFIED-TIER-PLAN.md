# Unified Tiered Cache — roadmap for the next ultracode push (2026-07-07)

> **Status @ 2026-07-07 (this push):** **Phase 1 (spill-not-drop) is COMPLETE + LIVE-VALIDATED
> on GB10 CUDA.** Phase 0 ✅ · 1a ✅ · 1b-core ✅ · 1b-integration ✅ · **live smoke ✅**. All
> gated `ATLAS_SSM_TIER`/`ATLAS_SSM_TAIL_PROTECT` (default off = byte-identical), full
> single-GPU build links, `#![deny(warnings)]`-clean, ~25 unit tests + a real-CUDA serving run.
>
> **Live smoke result** (Qwen3.6-35B-A3B-NVFP4 on dgx-00 GB10, `--ssm-cache-slots 4`,
> 6 interleaved multi-turn sessions, 72 requests): **72/72 OK, 0 errors, coherent output
> throughout** (fault-in restores are correct — no state corruption). The full serving path —
> `evict_to_tier` → `spill_slot` (D2H) → tier → `lookup_tiered` → `fault_in_slot` (H2D) →
> `promote` → restore — executes correctly under concurrent graph replay on real hardware, with
> **`evictions(drops)=0`** (every eviction became a spill) and **`mean_recompute_on_hit=15 tok`**
> (vs the #278 ~4400-tok recompute baseline).
> - initial fault-in (immediate-free-slot only): `tier_hits=43`, **`tier_fault_ins=13`**,
>   hit_rate 0.67 — 30 warm hits lost to a busy 4-slot pool → recompute.
> - **+ full-pool fault-in** (`acquire_or_spill_slot` spills a victim to make room):
>   `tier_hits=58`, **`tier_fault_ins=57`**, **hit_rate 0.91**, `tier_spills=274`,
>   `resident=4` (now correctly = slots). Nearly every warm hit now tier-restores instead of
>   recomputing.
>
> **Phases 2 & 3 ✅ LANDED + validated on GB10.** Phase 3 (cross-layer KV prefetch, 3a+3b-i+3b-ii)
> is complete and **live-validated correct** (prefetch on vs off = byte-identical output on a real
> KV-overflow decode); it shows **no perf gain on fast local NVMe** (read not the bottleneck) and is a
> ready lever for the slow-tier (NFS/RDMA) / deep-overflow regime. Phases 4–6 remain (Phase 4a =
> native-quant KV tier — deferred: a lot of kernel/layout work for a spot not currently bottlenecked).

### Phase 3 — foundation LANDED (2026-07-07), integration OPEN
Cross-layer prefetch: hide the tier read for attention layer L+1 behind the SSM+MoE compute
between L and L+1. Built + GB10-tested the correctness-critical foundation:
- **3a** (`fe5274d`): **persistent refcounted slot-pin** in `ScratchPool`/`EvictionPolicy`,
  honored by `assign`'s victim scan. This is the guard against the plan's headline hazard —
  without it a prefetch's `assign` could evict a slot layer L is *actively reading* → silent
  wrong-layer attention corruption. Unused by existing paths ⇒ byte-identical. 2 GPU tests.
- **3b-i** (`11c7683`): `prefetch_layer_on_stream` (reserve + load + PIN a layer's blocks, no
  attend) + `attend_layer` unpins **per-tile** after `step_tile` (frees slots for later tiles +
  next-layer prefetch; avoids a full-pinned-tile deadlock; stream-ordered so evict+overwrite is
  safe). GB10-tested: prefetch pins block 0 → attend consumes it → byte-correct output → block
  unpinned.
- **3b-ii — LANDED + live-validated (2026-07-07).** Chose the **side-stream, no-thread** design over
  the I/O-thread/`Arc<Mutex>` rework: the io_uring read already `stream_sync`s, so running prefetch on
  a side CUDA stream (`cuda_min::create_stream`) makes its H2D overlap main-stream compute while the CPU
  block overlaps already-enqueued SSM/MoE kernels — no shared-HSS, no touching the `unsafe Sync`
  assumption. `decode_a2.rs` triggers `hss.prefetch_layer(next_attn_idx, disk_block_ids)` for each
  overflowed seq when the next layer is full-attention. Gated `ATLAS_KV_PREFETCH`.
  - **Live A/B (Qwen3.6-35B-A3B, GB10, `--high-speed-swap-cache-blocks-per-seq 64`, 2250-tok prompt →
    KV overflow):** prefetch on vs off → **byte-identical output**, **identical 22 tok/s**. So it's
    **correct** (the pin + prefetch don't corrupt) but delivers **no measurable win here** — local NVMe
    (~GB/s) + ~1200-tok overflow means the tier read isn't the bottleneck. The overlap only pays when the
    read dominates: a **slow tier (NFS ~2 GB/s / RDMA) or deep overflow**. It's a correct, ready lever
    that will show its value in the over-core NFS/RDMA regime, not on fast local NVMe.
  - **LEARNING**: use the **release** build for 35B serve iteration — debug loads in ~7 min (unoptimized
    host-side BF16→FP32 weight promotion), release in **~40 s** and decodes faster; the fast weight
    loader is auto-on either way.
- **(historical) 3b-ii scoping note — architecturally significant.** The scoping surfaced a real constraint: HSS is a
  **thread-local `RefCell` singleton** and `backend.read` is **CPU-blocking**. A prefetch issued
  from the scheduler thread therefore overlaps only kernels *already enqueued* (L's FFN), NOT the
  SSM layers the same single CPU thread enqueues *after* the prefetch call — so triggering
  prefetch inline gives little real overlap. **Meaningful overlap needs a dedicated I/O thread +
  making HSS shareable across threads** (Arc/Mutex or per-seq contexts — revisiting the
  `RdmaKvBackend` `unsafe impl Sync` single-owner assumption), plus a side-stream + event
  (`stream_wait_event`) so L+1's attend waits on the prefetch's H2D. This is a distinct
  architectural change requiring live agentic-regime measurement (the existing
  `long_context_bench` measures `attend` in isolation and won't show the hidden-behind-compute
  win — a decode-tail bench must be built). Held for a dedicated push; the foundation above makes
  it safe to build on. Phases 4–6 remain.

### Phase 2 — LANDED (2026-07-07): per-seq orchestrator scratch
A contained **scratch/accumulator split**, not a wholesale HSS-per-seq (the scoping corrected the
plan's model: HSS has no `copy_stream`/CUDA-events/`war_armed`/bounce-ring — those were phantoms).
- **2a** (`6b4b6ac`): split `TiledAttention` into shared kernel handles + an external
  `TiledAttnPlanes` (m/l/o) passed as a param to begin_step/step_tile/finalize.
- **2b** (`47160d2`): the 6 transient HSS fields (planes + q_proj/block_scores/block_table/counts/
  score_host_buf) → a per-seq `SeqScratch`, held as `Vec<SeqScratch>` indexed by `seq_slot` (the
  batch position), lazily grown (no `max_batch_size` plumbing). `attend_layer` takes `seq_slot`;
  `multi_seq/attn.rs` passes the batch index `i` (the real per-seq site). Shared
  pool/backend/predictor/eviction/disk_state stay single-owner (seq-agnostic / global by design —
  honoring the `RdmaKvBackend` `unsafe impl Sync` single-owner assumption).
- **Ships NO perf** on its own (serial path still barriers per seq) — it's purely the enabler that
  lets Phase 3/5 overlap sequences without the 6/8 C=8 softmax-clobber race.
- **Byte-identical, GB10-validated**: `tiled_attention_parity` 2/2, `streaming_attention_e2e` 3/3,
  `high_speed_swap_e2e` 1/1 **+ a new slot-0-vs-slot-1 bit-identical equivalence assertion** (proves
  per-seq scratch is equivalence-preserving and lazy-grow works). Full single-GPU serve links.

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
| **1** ✅ | **SSM snapshots onto the cascade — spill-not-drop** — DONE + LIVE-VALIDATED on GB10 (154 spills / 0 drops / 13 fault-ins over 72 requests, 0 errors, recompute-on-hit 4400→15 tok). 1a byte-mover+tier · 1b-core state-machine · 1b-integration serving-wiring. | *Headline win.* Second cascade instance (`GroupLayout num_layers=1`, synthetic elem so `group_stride == num_ssm_layers×(h+conv)`) reusing every hardened byte-mover — zero change to trait/GroupKey/ScratchPool/KV-attn. `SnapshotEntry` → `Location{Hbm(slot)｜Tier(key)}`; evict **spills** not drops; prefix-lookup faults back in. Gated `$ATLAS_SSM_TIER` (default off = byte-identical). | low-med — **cross-stream ordering** vs `wait_snapshot_saves_dispatch` (a fault-in reading a half-written snapshot = silent state corruption); pin slots during fault-in |
| **2** ✅ | **Per-seq orchestrator isolation** | Move transient orchestrator state (copy_stream, S-planes, events, `war_armed`) off the shared thread-local `HighSpeedSwap` into a per-seq context. Ships no perf; **unblocks 3 & 5**. Re-state the `unsafe Sync` soundness. | med — under-scoping reintroduces the exact 6/8 C=8 race that killed the tile-pipeline |
| **3** ✅ | Cross-layer prefetch (deep lever A) — **LANDED + live-validated correct** (3a pin + 3b-i prefetch + 3b-ii side-stream + decode-loop trigger, gated `ATLAS_KV_PREFETCH`). **No perf gain on fast local NVMe** (read not the bottleneck there); the overlap pays off for slow tiers (NFS/RDMA) / deep overflow. | Hide tier reads behind the **SSM+MoE/FFN compute between two attention layers**. | med — a pin bug silently corrupts attention (wrong-layer blocks); must be measured at the agentic regime not cap=8 |
| **4** | Reduce read volume — **native-quant tier storage** (deep lever B) | Stop the BF16 inflation: `decode/high_speed_swap.rs:232-289` dequants FP8/NVFP4/4-bit → BF16 *before* `write_from_host`, so quantized history is stored + re-read at **2–4×**. Store native-quant bytes; dequant on read. **Biggest concrete byte cut.** | med — wide per-format correctness surface (FP8/NVFP4/Turbo3/4/8); needs a per-format numeric-diff harness |
| **5** 📋 | Parallel per-seq CPU orchestration (deep lever C) — **ultracode-designed 2026-07-07: BATCHING, not threads** (see plan below) | De-serialize the `multi_seq/attn.rs:203` per-seq loop **without dropping the cache** — kills the observed **2-cores-pegged / spiky-throughput** symptom under C=8. | med-high — highest concurrency risk; breaks `unsafe Sync` if widened silently (UB, not just a race); thread-pool footgun on the shared box |
| **6** | *(optional capstone)* one namespaced `BlobKey` space | Permanently kill KV/SSM re-divergence: one address space, one policy, per-namespace budgets. Only if cross-arbitration / shared budget proves worthwhile. | med — addressing/off-by-one cross-write hazard (KV over SSM bytes); pure hygiene, zero new user value |

## Phase 5 — vetted plan (ultracode workflow, 2026-07-07): BATCH, don't thread

A 14-agent understand→design→verify→synthesize workflow. **All four threaded/sharded designs came
back "risky"/unsound** under adversarial review; the unanimous low-risk winner is **single-owner
BATCHING, gated on measurement.** Threads chase overlapping N NVMe reads — I/O that this session
*proved* is not the bottleneck on local NVMe — and every threaded shape trips a real UB/liveness hazard.

**Grounding finding:** `kernels/gb10/.../paged_decode_attn_tiled.cu` is ALREADY batch-native
(seq=blockIdx.x, per-seq Q `[seq*nq+qh]`, `tile_blocks[seq*tile_cap+b]`, `counts[seq]`, per-seq m/l/o,
`grid=(num_seqs,nq,1)`), and `step_tile`/`begin_step`/`finalize` already take `num_seqs`. Batching the C
overflowed seqs into ONE `num_seqs=C` pass is a **pure host-orchestration rewrite** — no threads, no
locking, `unsafe impl Sync` (rdma_kv_backend.rs:209) untouched. Collapses C mid-attend `stream_sync`s
(impl_more.rs:238) → 1, and C under-occupied `grid=(1,nq,1)` launches → one `grid=(C,nq,1)`.

**MEASUREMENT DONE (2026-07-07) — CONFIRMS the bottleneck, green-lights batching.** C=8 concurrent
overflow decode (Qwen3.6-35B-A3B, `--high-speed-swap-cache-blocks-per-seq 8` = 128-tok window,
`--max-batch-size 8`, `ATLAS_MS_PROFILE=1`): **attention dominates decode (~70%)** — `attn≈133ms(10L)`
vs `ssm≈40ms(30L)` at n=8. Attention is the **one part not batching**: it scales ~linearly with N
(n=2→8: attn 35→133ms = 3.8× for 4× batch, ~16.6ms/seq) while SSM/MoE batch well (per-tok DROPS
29.8→23ms as N grows). A core **pegs to 100% in bursts** (the "spiky" symptom) from the ~80 mid-attend
`stream_sync`s/step (10 attn layers × 8 seqs, impl_more.rs:238). So the per-seq-serial attend is the real
cost; batching it (collapse ~80 syncs→~10 + one wide `num_seqs=8` launch) should bring attn in line with
how SSM/MoE already batch. **Green light for the plan below.** (The exact gain needs the implementation
to measure; batching helps the CPU-orchestration/sync/launch fraction of the 133ms, not the GPU-kernel or
disk-read fraction.)

**Sequence: MEASURE ✅ → BATCH → (stop).**
1. **Measure** (mandatory): serve `--high-speed-swap --high-speed-swap-cache-blocks-per-seq 8
   --max-batch-size 8`, 8 concurrent long (>128-tok/window) prompts, `ATLAS_SSM_MS_PROFILE=1`.
   **Triad, all three must agree = CPU-serial confirmed:** (a) `attn_us ∝ N` (decode_a2.rs profile line)
   while GPU paged-decode is ~flat in N; (b) `mpstat -P ALL` shows ~2 cores pegged, rest idle; (c) tok/s
   plateaus over C=1,2,4,8. If GPU/bandwidth-bound instead → Phase 5 can't help, **stop**.
2. **Batch** (only if confirmed). Incremental, each byte-identical + GPU-testable:
   - **Inc 1** — plumb `max_seqs:1`→C (high_speed_swap.rs:241): size batched `TiledAttnPlanes` +
     `block_table_dev[C*tile_cap]` + `counts_dev[C]`; decouple `num_slots` from `resident_blocks`
     (hss.rs:173), grow to `C*per_seq_budget`. `debug_assert!(num_seqs<=max_seqs)` **at the plane-alloc
     site**. C=1 path byte-identical.
   - **Inc 2** — score-only batching (lowest blast radius, the actual sync win): per-seq
     `project_q`/`score_blocks` into `SeqScratch[s]` with NO interleaved sync, then ONE `stream_sync`.
     Land independently; should recover most of the symptom.
   - **Inc 3** — union read + wide launch: union all C seqs' missing blocks into one `backend.read`;
     `[C×tile_cap]` block_table + `[C]` counts; `step_tile(num_seqs=C)`. **Pin every slot assigned across
     all C for the tile, unpin after `step_tile`**; enforce `C*per_seq_budget<=num_slots`.
   - **Inc 4** (separate track) — the serial offload stripe-repack (impl_more.rs:117-133, nested
     bs×nkv×hd per-element `to_le_bytes`, runs for EVERY HSS seq every step, attn.rs:207) is a prime
     suspect for a pegged core batching doesn't touch — vectorize or move to GPU.
   - **Inc 5** (only if rank dominates) — `eviction.rank()` full-sorts `num_slots` per missing block; the
     grown pool × C× missing blocks can make it the new peg (~64× host work at C=8). Incremental/heap.
3. Keep `attend_layer_on_stream_with_q_pos` for prefill (per-seq `last_block_valid_slots` can't share one
   scalar). **C=1 MUST keep `budget==num_slots`** so the single-seq golden path stays bitwise-identical.

**Top 3 hazards** (all in batched attend): (1) **OOB device write** — planes sized `max_seqs=1` but
`grid.x=C` writes `m/l/o[seq*nq+…]` past the buffer = silent HBM corruption → rebuild planes for C +
the plane-alloc assert; (2) **cross-seq intra-tile WAR** — one union read fills all C seqs' slots, s2's
`assign` mustn't evict s1's just-placed unconsumed slot → pin-on-assign-across-C, unpin after step_tile;
(3) **ragged-tail stale counts** — an exhausted seq must present `counts[s]=0` every later tile → rebuild
`counts_host = vec![0;C]` fresh each tile, never carry.

**Do NOT:** `Arc<Mutex<ScratchPool>>` (unsound cross-stream torn-KV via enqueue-gated `unpin_key` +
non-live single-seq-sized pool → `assign` Err → decode abort); per-worker io_uring backends (`setup_sqpoll`
io_uring.rs:32 → busy-spinning SQPOLL kernel threads on the shared GB10 co-running Holo/hyades — *worsens*
the peg); a blanket `read_on_stream` dropping the terminal sync (UB for RDMA — reap frees bounce with no
CudaEvent, zero-copy NIC write isn't CUDA-stream-ordered — and posix single shared bounce; io_uring-only if
ever); parallelize before the triad confirms CPU-bound; change the C=1 tile budget.

Tests: extend the ignored-GPU `tiled_attention` tests — C seqs batched, each row within tolerance of the
per-seq result (float non-associativity ⇒ NOT bitwise for C>1); `counts=0` tail = no-op; `C*budget<=num_slots`.

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

### Phase 1b-integration — LANDED code-complete (2026-07-07): serving path wired
All gated `ATLAS_SSM_TIER` (default off = byte-identical; verified: flag-off never tiers an
entry, `reclaim` takes the drop branch, the fault-in block is skipped, and `RadixTree::lookup`'s
tier-aware path degenerates to the resident-only lookup).
- `TransformerModel` owns `Option<Arc<dyn SnapshotBlobStore>>` (`types.rs`), built in `impl_a1`
  as an **unbounded** host-RAM `MemBlobStore` when `ATLAS_SSM_TIER` set AND the model has SSM
  layers. Unbounded ⇒ `put` never rejects (bounded-tier drop-on-reject is a follow-up).
- `PrefixCache` trait gained `evict_snapshot_to_tier()` + `promote_snapshot()` (default None/false;
  `RadixTree` delegates to the index). `PrefixMatch` gained additive `ssm_snapshot_tier_key` +
  `ssm_snapshot_tier_tokens`. `RadixTree::lookup` routes through `lookup_tiered` (one call, no
  telemetry double-count).
- `reclaim_from_cache(prefix_cache, kv, tier, gpu)`: with a tier, spill the victim
  (`evict_snapshot_to_tier` → `spill_slot` → `free`) instead of dropping. **All 6 exhaustion
  call sites** (`save_checkpoint`/`decode_checkpoint`×2/`finalize_last`/`prefill_d`×2) pass the
  tier, so every path spills via this one choke point.
- `prefill_b` prefix-lookup: on a spilled deepest anchor, `try_pop_free_slot` → `fault_in_slot`
  → `promote_snapshot` → restore uniformly. Ordering: Marconi saves run on the **default
  stream**, so `spill_slot`'s `synchronize` drains them before the D2H (no half-written spill);
  `fault_in_slot` `synchronize`s after the H2D before restore reads the slot.
- **Follow-ups**: full-pool fault-in ✅ DONE + live-validated (`acquire_or_spill_slot` — see
  above, 13→57 fault-ins). Remaining (documented, not blocking): bounded-tier drop-on-reject,
  cost-aware fault-vs-recompute depth guard, and wiring `prefill_a`/`prefill_c` (they ignore the
  tier key → recompute, which is correct just unoptimized).
- **CPU-validated**: `test_spill_tier_lookup_transitions` drives resident→spilled→resident
  through the real `RadixTree`; pool byte-mover + slot-recycling on `MockGpuBackend`.

### Live-validation recipe (✅ EXECUTED 2026-07-07 — PASSED)
**Learning — the workload must be INTERLEAVED multi-session, not single-turn.** First attempts
(short single-turn prompts; identical re-sends) produced **0 spills / 0 fault-ins**: short
prompts save no intermediate checkpoints, and a snapshot is only *re-needed* after it's been
evicted by *other* sessions' activity. Only **6 interleaved sessions** (> the 4-slot pool), each
with a distinct >1024-token root (stable `compute_session_hash` = first ≤1024 tokens) + multi-turn
history, reproduced the #278 pattern: session A's snapshot spills while B..F run, then A resumes
and faults it back in. That run gave 154 spills / 43 tier hits / 13 fault-ins, 0 drops. Repro
script: an interleaved-session driver like `scripts/dev/agentic_test.py` (or the ad-hoc
`ssm_interleaved.py` used here). Also: `--gpu-memory-utilization` must clear weights+buffers
(~46 GB for 35B-A3B) *before* any KV budget — 0.35 OOMs, use ≥0.55.

Build kernels for the SSM target, then serve with a **tiny** pool to force spill+fault-in:
```
# rebuild so kernels match the served model (default target is qwen3-next-80b-a3b)
ATLAS_TARGET_MODEL=qwen3.6-35b-a3b cargo build --release -p spark-server \
  --no-default-features --features cuda
# serve (spare port; scoped; tear down by port, never blanket-pkill)
CKPT=~/.cache/huggingface/hub/models--AEON-7--Qwen3.6-35B-A3B-heretic-NVFP4/snapshots/*/
ATLAS_SSM_TIER=1 ATLAS_SSM_TAIL_PROTECT=1 ATLAS_SSM_SNAP_STATS=1 RUST_LOG=info \
  ./target/release/spark serve --model-from-path $CKPT --model-name a3b --port 18999 \
  --lm-head-dtype bf16 --gpu-memory-utilization 0.60 --max-seq-len 8192 --max-batch-size 2 \
  --enable-prefix-caching --ssm-cache-slots 4 --ssm-checkpoint-interval 16
# NB: util must clear weights+buffers+reserve BEFORE any KV budget — 35B-A3B needs
# ~46 GB committed, so 0.35 (=42.6 GB) OOMs "No memory left for KV cache"; use >=0.55.
# workload: a few multi-turn repeated-prefix chats across 2-3 sessions (scripts/dev/agentic_test.py
# or curl) so the 4-slot pool overflows → spill, and warm turns re-hit → fault-in.
```
**Pass gate**: log shows `SSM spill tier ENABLED`; `ssm-snap-stats` shows `tier_spills>0` AND
`tier_hits>0` with `tier_fault_ins>0`; warm-turn outputs are coherent (a corrupted fault-in
restore would garble them); no crash/error. **Correctness A/B**: same prompts at
`ATLAS_SSM_TIER=0` vs `=1` should agree (greedy/temp-0 for an exact compare). Then, for the
north-star perf read, run the #278 35B agentic wall and confirm `mean_recompute_on_hit` drops
below the tail-protect+256 residual as `tier_hits` rises.

### Phase 5 Inc 1+2 — LANDED code-complete + GPU-parity-validated (2026-07-07, ultracode fable)
A 10-agent understand→implement→verify fable workflow (5 parallel readers de-staled the plan
against current code, 1 implementer, 4 adversarial verifiers). Inc 1 (plumb `max_seqs=C`) +
Inc 2 (score-only sync-collapse) landed in `spark-storage`; the batched `num_seqs=C` kernel path
is now GPU-verified even though host orchestration (Inc 3) doesn't drive it yet.

**Plan-staleness the readers caught (the plan was written from a prior reading):**
- `block_table_dev`/`counts_dev`/`planes` are **per-seq inside `SeqScratch`** (a `Vec`), NOT single
  buffers. The plan's "grow `block_table_dev` to `[C*tile_cap]`" only makes sense for a *shared*
  batch buffer (the kernel indexes one contiguous `tile_blocks[seq*tile_cap+b]`/`counts[seq]`).
  Realized as a new `BatchScratch {C-sized planes, [C×tile_cap] table, [C] counts}`, allocated only
  when `C>1` (dead-code-gated until Inc 3), leaving per-seq scratch single-sized.
- No `num_seqs` exists at the plane-alloc site (`TiledAttnPlanes::new`); the real unguarded OOB
  launch sites are `step_tile_on_stream`/`finalize_on_stream`. Realized as: record plane capacity
  (`n_q_slots`) at alloc + guards at all three launch fns.
- `debug_assert!(num_seqs<=max_seqs)` already existed as a hard `bail!` in `begin_step`; but it
  checks `dims.max_seqs`, not the *actual* plane capacity — a `num_seqs=C` launch bound to a
  1-seq per-seq plane clears it and writes OOB. **Promoted to a release-active `bail!` on
  `num_seqs*num_q_heads > planes.n_q_slots`** (the true H1 guard; the moment Inc 3 lands a wide
  launch, a mis-pairing errors instead of silently corrupting HBM).

**The adversarial pass earned its keep — one CONFIRMED `major` (cross-stream WAR, C-fold widened):**
collapsing the C per-seq mid-attend `stream_sync`s removed the *only* barrier that incidentally
drained each prior seq's `step_tile` before the next began. Without it, all C seqs' tiles are
enqueued-but-unexecuted (slots unpinned at `step_tile` *enqueue*, `impl_more.rs`) when
`prefetch_layer` — on a **separate stream with no CudaEvent ordering** — `assign`s + overwrites the
oldest-touched slots. Reachable with `ATLAS_KV_PREFETCH=1` + C≥2 = silent, timing-dependent KV
corruption. **Verified against the code by hand** (`unpin_key` at enqueue `impl_more.rs:453`;
prefetch on `prefetch_stream` `impl_more.rs:523`, `assign` at `:493`, no event; the existing
safety comment's "stream ordering makes it safe" only covers *same-stream* ops).
- **Fix (landed):** `HighSpeedSwap` reads `ATLAS_KV_PREFETCH` once at construction
  (`kv_prefetch_enabled`); `attend_layer_batch_on_stream` falls back to the **serial per-seq
  attend** (each with its own `score→sync→tile`, restoring the pre-change 1-seq in-flight window)
  whenever prefetch is live. Sync-collapse and prefetch are now *mutually exclusive*, not silently
  racy. The default (prefetch off) still gets the full Inc-2 collapse.
- **Deferred follow-up (Inc 3 prerequisite):** proper coexistence = record a CudaEvent on the main
  stream at the end of the batched attend, `cuStreamWaitEvent` it on `prefetch_stream` before
  prefetch `assign`/reads. Then the two can run together. Also flagged: `attend_tile_phase` calls
  `backend.read` even on an empty `reqs` (io_uring/RDMA pay a terminal sync per tile) — the
  `if !reqs.is_empty()` guard must land *with* the event fix (that empty-read sync is currently the
  accidental io_uring WAR narrowing), and `score_host_buf` is pageable (D2H may sync-block) — pin it
  if the C=8 measurement shows the collapse didn't materialize.

**Minor hardening also landed:** duplicate-`seq_slot` in a batch is now a hard `bail!` (aliased
per-seq scratch = silent wrong results; host O(C²), C≤~8, negligible).

**GPU-validated on dgx-00 GB10 (all ignored parity tests pass):**
- `tiled_attention_parity`: `batched_seqs_match_per_seq` (C=4 `num_seqs=4` pass matches each seq
  solo, per-row tol<1e-2), **`counts_zero_tail_is_noop`** (H3 exact-no-op, bitwise) — 4/4 ok.
- `high_speed_swap_e2e`: **`batched_attend_matches_single_seq_bitwise`** (two ragged seqs through
  ONE `attend_layer_batch_on_stream` == two independent `attend_layer`, bit-for-bit per row — the
  Inc-2 sync-collapse is output-identical), `pool_sized_for_c_times_per_seq_budget`
  (`C*budget<=num_slots` by construction) — 3/3 ok.
- `cargo build -p spark-storage --features cuda` + `cargo check --workspace` +
  `cargo build --release -p spark-server --no-default-features --features cuda` all green;
  71 storage unit tests pass.

**Still open (operator live run, box permitting):** the C=8 sync-count / latency measurement —
serve `ATLAS_HSS_MAX_SEQS=8` + 8 concurrent overflow prompts, `ATLAS_SSM_MS_PROFILE=1`, prefetch
OFF, and confirm the mid-attend `stream_sync`s collapse (~80→~10/step) and `attn_us` scales
sub-linearly in N. This quantifies the win; correctness is already GPU-verified above.

### Phase 5 Inc 1+2 — C=8 LIVE MEASUREMENT (2026-07-07): Inc 2 is NOT the win (redirects to Inc 3)
Served Qwen3.6-35B-A3B on dgx-00 GB10 (scoped port 18997, torn down by PID), `ATLAS_HSS_MAX_SEQS=8`
(`batched attend sized for max_seqs=8` confirmed in log), `ATLAS_MS_PROFILE=1`, **prefetch OFF** (so
the Inc-2 sync-collapse path is live), `--high-speed-swap-cache-blocks-per-seq 8` (128-tok window),
`--max-batch-size 8`, 8 concurrent ~500-tok overflow prompts. Steady state n=8 (197 profiled steps):

| metric | this run (Inc 1+2) | pre-change baseline (prior session) |
|---|---|---|
| attn (10 layers) | **139.7 ms** | ~133 ms |
| attn / seq | **17.4 ms** | ~16.6 ms |
| attn % of decode | 79% | ~70% |
| per-tok | 23.3 ms | — |
| tok/s (8-way) | 4.8 | ~4 |

**attn/seq is FLAT (17.4 vs 16.6 ms = within noise). The sync-collapse delivered no measurable decode
speedup.** This is the important result, not a disappointment: it falsifies the prior session's inference
that the `attn ∝ N` cost was CPU-sync-bound. The ~80 mid-attend `stream_sync`s → ~10 collapse has a
theoretical ceiling of ~1% of the 140 ms attn wall (70 fewer syncs × ~20 µs); flat is exactly expected.
The pegged core was CPU-busy **overlapped** with GPU+disk, not serially gating decode.

**Where the `∝N` cost actually lives → Inc 3.** At C=8 each seq fires an **under-occupied
`grid=(1,nq,1)` launch and waits on its own NVMe read, back-to-back on one stream** — 8 sequential
under-occupied GPU launches + 8 sequential disk waits. That is what scales with N, and Inc 2 does not
touch it. **Inc 3 (one wide `grid=(C,nq,1)` launch = 8× occupancy + a union disk read = one wait not 8)
is the real lever.** Inc 2 remains the correct, GPU-validated structural prerequisite that makes the wide
launch expressible; it is not a standalone throughput win.

**Caveats (stated, not spun):** (1) baseline is cross-session, not a same-config HEAD A/B — the 16.6 vs
17.4 gap is within run-to-run noise either way; a same-box HEAD-vs-branch A/B would make "no win" airtight
but the ~1% ceiling already predicts flat. (2) `ATLAS_MS_PROFILE=1` perturbs the regime (forces eager /
per-phase syncs; log also showed FP8-calibration re-enabling graphs mid-run), and wave-A n=1 emitted no
profile lines (ran during the graph/calibration transition) so no clean per-N scaling curve from this run.
(3) Per [[feedback-test-models]], **future Phase 5 validation (esp. Inc 3 correctness + any quality A/B)
should serve Holo 3.1 35B**, not qwen3.6-35b-a3b — the sync-collapse being timed here is model-agnostic
host orchestration so qwen was fine for it, but quality judgments need the trusted-baseline model.

**Recommendation:** land Inc 3 (union read + `step_tile(num_seqs=C)` wide launch, per the §Phase 5 plan
+ the H1 release-active plane-capacity bail that already guards it) as the next increment — that is where
the measured `∝N` attention cost is actually attackable. Re-measure on Holo 3.1 35B.

### Phase 5 Inc 3 — LANDED + GPU-validated (2026-07-07); C=8 NVMe measurement = read-bound, NO win (RDMA next)
Wide `grid=(C,nq,1)` launch + union tier read (359591f). Kernel untouched: BatchScratch gains
q_gather/o_gather `[C×nq×hd]` (the kernel reads `Q[(seq×nq+qh)×hd]` seq=0..C-1 but overflowed seqs
sit at sparse batch positions → d2d-gather Q in, wide launch, finalize C rows, scatter O back; new
`copy_d_to_d_async`). `attend_tile_phase_batched`: lockstep tiles, one union `backend.read` + one
`step_tile(num_seqs=C)` per tile. Gated on `batch.is_some() && seqs.len()<=max_seqs` (else Inc-2 serial).
Hazards held: H1 (begin_step release-active `n_q_slots` bail is the live guard now), H2 (pin ALL C per tile,
unpin after), H3 (counts fresh `[0;C]`, exhausted seq → `counts[s]=0` no-op).

**GPU-validated bitwise (dgx-00 GB10):** NEW `batched_wide_launch_matches_serial_multitile` — 3 ragged
multi-tile seqs (32/20/8 blocks, tile_cap 8), pool = EXACTLY C×tile_cap=24 (every slot pinned per tile →
H2 load-bearing) — wide launch == serial per-seq **bit-for-bit**. All parity suites pass; 71 unit tests;
release spark-server green.

**C=8 LIVE MEASUREMENT (same config/harness as the Inc-2 run, qwen3.6-35b-a3b, prefetch OFF,
ATLAS_HSS_MAX_SEQS=8 sized): NO throughput win — read-bound.**
- n=8, 197 steps: attn **median 133.6 ms / mean 153.1 ms** (vs Inc-2 ~140 ms) — **flat**. **tok/s 4.8,
  unchanged.** Only 8/197 steps (4%) were <30 ms, and they are the FIRST 8 steps (early decode, little
  overflow), NOT a wide-launch win; 89% of steps >120 ms; max step 2.65 s (disk stall).
- **Mechanism: the C=8 overflow-decode attention is NVMe-read-bound** — each of the 8 seqs re-streams its
  growing >128-tok KV history from NVMe every step. attn climbs with overflow volume (fast early → ~133 ms
  once windows are full). Neither Inc 2 (sync-collapse) nor Inc 3 (wide launch + union read) reduces the
  *bytes* read, so both are flat. The wide-launch GPU-occupancy win and union-read syscall win are real
  mechanisms but immaterial while the GPU idles waiting on NVMe.

**CORRECTION:** an initial read of the log *tail* suggested a 7.5× attn drop — that was wrong (those were a
later batch's early low-overflow steps). The median/tok-s show no win. Reported straight per
[[feedback-no-fudged-data]].

**This is the expected result under [[streaming-experts-framework-first]] — KEEP the mechanism; the win is
on the faster tier / at larger scale, not here.** Concretely, next:
1. **RDMA tier measurement** (`ATLAS_KV_PEER=gx10`, ~11–24 GB/s vs NVMe ~2 GB/s): the read wall drops
   ~6–12×, which should EXPOSE the wide-launch/union-read framework win. Requires an `atlas_rdma_verbs`
   build + healthy CX7 to gx10. **This is the real test of the over-core thesis; the framework (Inc 1+2+3)
   is now all in place + bitwise-validated to run it.**
2. **Prefetch overlap** (Phase 3) to hide the read behind SSM/MoE compute — but the batched path currently
   serial-falls-back under `ATLAS_KV_PREFETCH` (the WAR fix); the CudaEvent coexistence follow-up is the
   prerequisite to combine batched + prefetch.
3. Larger model / deeper over-core, where attention compute (which the wide launch does speed up on
   resident steps) is a bigger fraction than the per-step KV re-read.

### Phase 5 — reasonable-buffers + RDMA-tier measurements (2026-07-07)
Two follow-ups to the pathological 128-tok-window NVMe run (which was read-bound, 4.8 tok/s). ALL numbers
below are **profiling-on** (`ATLAS_MS_PROFILE=1` forces eager + per-phase syncs) so absolute tok/s is a
FLOOR, not production — but A/B deltas (symmetric tax) are valid.

**(1) Reasonable buffers — perf is healthy.** Realistic config (`--max-seq-len 32768 --ssm-cache-slots 256
--enable-prefix-caching`, ~2K-tok prompts, 8 concurrent, working set fits → streaming dormant): **native
15.0 tok/s/req** (profiling-on) vs the pathological 128-tok-window's 4.8 → **~3× faster with sane buffers.**
So the low numbers were the deliberately-pathological tiny window, not a real regression; with buffers
sized to hold the context, decode is healthy. (HSS-resident-32K config OOM-free after fixing an orphaned
72 GB server that had leaked from a killed harness — hygiene note: `kill -9` serve, it wedges on SIGTERM.)

**(2) RDMA tier vs NVMe (same 128-tok window, gx10 RAM over CX7, bounce mode):**

| tier | attn (n=8) | attn/seq | per-tok | tok/s |
|---|---|---|---|---|
| NVMe (local io_uring) | 133.6 ms | 16.7 ms | 23.3 ms | 4.8 |
| RDMA (gx10 blade) | 119.0 ms | 14.9 ms | 21.3 ms | 5.3 |

**RDMA is faster but only ~10%** — far short of the raw bandwidth ratio (CX7 ~11–24 GB/s vs NVMe ~2 GB/s),
with a fat tail (attn max **3034 ms** = a network/bounce stall). Why it's not 6–12×:
- **Bounce mode, not zero-copy.** `ATLAS_KV_ZERO_COPY` needs UMA KV scratch that isn't wired yet (RDMA-
  KV-TIER.md §6) → every read is D2H→RDMA→H2D, capping effective BW far below the link.
- **Latency-bound, not bandwidth-bound.** Each step reads only the window's few new blocks/seq — small
  transfers where round-trip + bounce overhead dominate, so 10× BW barely moves the needle.
- Profiling/eager serialization on top.

**Takeaways:** the over-core thesis is directionally confirmed (RDMA IS faster) but the current bounce-mode
path doesn't expose the bandwidth advantage. Concrete levers, in order: **(a) wire zero-copy RDMA (UMA KV
scratch)** — the pending RDMA-KV-TIER §6 item, likely the biggest single win; (b) prefetch-overlap to hide
the read entirely (needs the CudaEvent coexistence fix so batched + prefetch combine); (c) larger reads /
bigger models where the per-step re-read is a smaller fraction. Peer now runs as a durable systemd service
on gx10 ([[atlas-kv-peer-service]]). Production tok/s (graphs-on, no profiling) still TODO — all numbers
here are profiling floors.

### Phase 5 — HONEST PRODUCTION tok/s (2026-07-07, graphs-on, SLAI@100ms, verified via runtime log)
Correct flag set THIS time (verified in the STARTUP LOG, not just --help): `--scheduling-policy slai
--tbt-deadline-ms 100 --ssm-cache-slots 256 --enable-prefix-caching --max-seq-len 32768 --max-batch-size 8
--lm-head-dtype bf16 --gpu-memory-utilization 0.70`, graphs ON (no ATLAS_MS_PROFILE). Log confirmed
`Scheduling policy: SLAI (TBT deadline=100ms)` + `Marconi 256 slots` + `Prefix caching: ENABLED`.
~2K-tok prompts, 8-way. (Prior tok/s were profiling floors AND FIFO — both wrong; discard them.)

| config | 1-concurrent (peak) | 8-concurrent (per-req / aggregate) |
|---|---|---|
| **native** | **64.8 tok/s** | 15.0 / **120.2 tok/s** |
| **HSS-resident** (window 2048=32K, NO overflow) | **19.3 tok/s** | 12.8 / 102.6 tok/s |

**RETRACTION: HSS is NOT "free when resident."** Earlier this session I claimed the batched machinery is
dormant/free when the working set fits — the measurement refutes it: HSS-enabled is **3.4× slower
single-stream** (19.3 vs 64.8) and ~15% slower at 8-way, even sized to hold everything resident (no
overflow). There is a real standing per-step cost in the `--high-speed-swap` path that native doesn't pay.
Candidate causes (VERIFY before asserting): (a) the per-step offload stripe-repack (impl_more.rs:117-133,
runs per HSS seq per step regardless of overflow — the Inc 4 target); (b) host-side FP8 KV dequant (startup
log: "10 attn layers FP8 … host dequant for FP8/NVFP4"). Single-stream hit hardest ⇒ fixed per-step CPU
overhead that doesn't amortize at low concurrency.

**Implications:** native production is healthy (~65 single-stream, ~120 aggregate @8-way, SLAI@100ms).
Enable HSS ONLY for genuine over-core (context > HBM), not as an always-on layer — for in-HBM contexts
native is much faster. Reframes Phase 5: the batched attention (Inc 1+2+3) optimizes the STREAMING path,
which inherently costs more than native; the value is enabling over-core at all + minimizing that cost.
**NEXT: pin down the HSS standing-cost source** (offload stripe-repack vs host FP8 dequant) — likely the
highest-leverage remaining lever, ahead of the RDMA zero-copy work.

### ISL/OSL baseline (container CUDA 13.2 + CUTLASS, native resident, 2026-07-07)
Reasonable resident buffers (256 SSM snapshot slots, KV in HBM, gpu-mem 0.70 → 83.8 GB free after weights,
NOTHING spilling), SLAI@100ms, graphs-on, batch=1, cold. Via bench/bench_isl_osl.py:

| ISL/OSL | TTFT | prefill tok/s | decode TPOT | decode tok/s |
|---|---|---|---|---|
| 1024/128 | 2.5s | 411 | 6.7ms | 148 |
| 4096/128 | 3.7s | 1107 | 7.6ms | 132 |
| 8192/128 | 4.2s | 1932 | 9.3ms | 107 |
| 16384/128 | 8.6s | 1914 | 11.7ms | 85 |
| 32768/128 | 17.4s | 1883 | 17.0ms | 59 |
| 32768/1024 | 26.7s | — | 19.8ms | 51 |

Prefill throughput plateaus ~1900 tok/s at 8K+ (chunked prefill, no cliff to 32K); decode degrades
148→51 tok/s as KV attention grows with context (TPOT 6.7→19.8ms). This is the pure-compute resident
baseline on the CUTLASS+13.2 binary (decode +14% at concurrency vs the CUDA-13.0/no-cutlass host floor).

### SSM-tier map — CORRECTION + the SSM→RDMA reframing (2026-07-07)
Earlier I said "SSM only spills to host-RAM" — WRONG. SSM state has TWO landed disk/tier paths, plus the
KV tier, and one gap:

| what | trigger | destination | status |
|---|---|---|---|
| Sequence (KV+SSM) swap-out | seq preempted to free HBM | **NVMe** (swap file, `sequence.rs` Ph3, `--swap-space-gb`) | landed ✓ |
| SSM **snapshot** spill (Marconi) | prefix-cache pool full | **host-RAM** (`MemBlobStore`, `ATLAS_SSM_TIER`) | landed ✓ |
| KV **overflow** | over-core resident window | NVMe *or* **RDMA peer** (HSS) | landed ✓ |
| **SSM snapshot → RDMA** | pool full + local tier full (deep agentic) | remote peer | **NOT wired** |

**Why SSM→RDMA is the high-value next increment (the operator's reframing):** SSM snapshots are the
*expensive* resource, in two senses neither KV path shares. (1) **Recompute-on-loss is compute-bound**:
lose a KV block → re-*read* it (I/O, cheap); lose an SSM snapshot → **recompute the entire SSM prefix**
(sequential recurrent scan over every prior token — #278: ~4400 recompute-tok/hit at 16 slots). (2) **HBM
footprint is huge**: this run's log = `Marconi 256 slots (16320 MB)` + `decode-rollback (4080 MB)` ≈
**20 GB HBM** for SSM state, often > the KV cache. So spilling snapshots to a *remote* tier both frees
~16 GB HBM for resident KV/context AND scales warm-snapshot capacity past local LPDDR/NVMe — converting
expensive recomputes into cheap remote restores for deep concurrent agentic.

**The engineering catch (open question, now the increment):** KV is contiguous blocks (trivial zero-copy
RDMA); SSM snapshot state is **scattered across ~60 per-layer device pointers** (`h_state`/`conv_state` ×
30 SSM layers), so `register_landing_region`'s one-contiguous-UMA-pool assumption breaks. Options: (a)
bounce-gather to one contiguous blob then reuse `RdmaKvBackend.write_from_host`/read (host-mediated, matches
the current `MemBlobStore` D2H-gather/H2D-scatter — simplest, correctness-first); (b) extend landing
registration to 60 per-layer MRs for zero-copy scatter (fastest, hardest). NEXT: an ultracode design pass
to vet the approach (a new `RdmaSnapshotStore : SnapshotBlobStore` over the existing verbs) — see below.

### Phase 4b — SSM-snapshot RDMA spill tier (ULTRACODE-DESIGNED 2026-07-07)
> Design via a 5-phase ultracode workflow (5 readers CONFIRMED the key insight 5/5; synthesis produced this plan). The workflow's Design+Verify agent phases crashed (StructuredOutput schema cap), so the adversarial verification was done BY HAND against the code — CONFIRMED: (1) spill_slot builds ONE contiguous blob + leading synchronize (ssm_snapshot.rs:434-448); (2) fault_in_slot trailing synchronize before restore (:488); (3) reclaim_from_cache handles put->Ok(false) → free slot → miss-into-recompute (:565-581). The 3 load-bearing correctness claims hold.

> **Thesis.** The Marconi snapshot pool (~20 GB HBM: 256 slots × 63.75 MB + decode-rollback) today spills evicted snapshots only to local host RAM (`MemBlobStore`, gated `ATLAS_SSM_TIER`). Give it a **second** `SnapshotBlobStore` implementation that ships the *already-contiguous* spill blob to the gx10 RAM blade over the existing CX7 RDMA transport. This scales warm-snapshot capacity past local LPDDR and frees ~16–20 GB HBM, converting the *expensive* loss (an SSM-prefix **recompute**, ~4,400 recompute-tok/hit at 16 slots, #278) into a cheap ~5–7 ms remote restore. Default-off ⇒ byte-identical to today.

### Verified architectural ground (why this is the small increment, not the scary one)

Confirmed against code before design — these are the load-bearing facts:

- **The integration surface is exactly `put`/`get`.** `SnapshotBlobStore` is a **host-bytes** trait — `put(key:u64, bytes:&[u8])->Result<bool>`, `get(key:u64, out:&mut[u8])->Result<bool>`, `remove/len/bytes_resident` (`ssm_tier.rs:42-61`). An `RdmaSnapshotStore` never touches a device stream.
- **The blob is one contiguous host `Vec`.** `spill_slot` (`ssm_snapshot.rs:435-448`) allocates `vec![0u8; spill_blob_bytes()]`, D2H-gathers the 60 scattered per-layer chunks into it in layout `[h₀ conv₀ h₁ conv₁ … h_{L-1} conv_{L-1}]`, then calls `store.put(key,&blob)`. `fault_in_slot` (`459-490`) does `store.get(key,&mut blob)` then H2D-scatters. **The "60 scattered device MRs" problem is already solved at the trait boundary** — no `register_landing_region`-to-60-MRs work is needed. Zero-copy is a later perf phase, not this increment.
- **Both cross-stream ordering hazards are already discharged.** `spill_slot` does a leading `synchronize(stream)` (`ssm_snapshot.rs:434`) to drain the in-flight async save before the D2H read; `fault_in_slot` does a trailing `synchronize(stream)` (`488`) before the caller's `restore` reads the slot. A host-bytes store inherits both for free.
- **The peer-full path is already correct.** `reclaim_from_cache` (`ssm_snapshot.rs:565-581`) handles `put→Ok(false)` by logging and freeing the slot; the index entry stays marked-tiered but holds no bytes, so a later `lookup_tiered`→fault-in cleanly **misses into recompute**. A *bounded* remote tier therefore ships correct with **zero new rollback logic**.
- **The KV blade server is layout-agnostic.** `kv_peer` wire protocol (`kv_peer.rs:16-23`) is: client sends `[u64 total_bytes]` → peer allocates + registers ONE RW MR of that size → serves one-sided WRITE/READ at `base + offset`. **The peer knows nothing about `GroupLayout`** — all group addressing lives client-side in `RdmaKvBackend`. So a **second `kv_peer` instance** serves a snapshot arena with **zero peer-side code changes**; we only need a different client-side addressing scheme.
- **KV `GroupKey` addressing is the wrong layout and would corrupt live KV.** `RdmaKvBackend::write_from_host` (`rdma_kv_backend.rs:485-509`) asserts `src.len() == group_bytes` and addresses `base + group_id*group_stride`. A 63.75 MB snapshot blob is neither group-sized nor group-addressable — reusing `GroupKey` is verified-broken. Snapshots need a **`u64 → arena offset` allocator**, not the KV group layout.

### Mechanism

**New type `RdmaSnapshotStore` (crate `spark-storage`) implementing `SnapshotBlobStore`.** Three parts:

1. **Transport (reuse KV machinery, replace addressing).** Reuse `RdmaKvBackend`'s connection bring-up verbatim — the TCP handshake (`[u64 total_bytes][u8 n_rails]` → `KvServerParams` → `VerbsClientParams` → QP connect), dual-rail striping (`ATLAS_EXPERT_RDMA_DEV`/`GID` rail 0, `ATLAS_KV_RAIL2_DEV`/`GID` rail 1, `ATLAS_KV_DUAL_RAIL`), and pipeline depth. Extract the rail-set bring-up (`connect`'s rail-creation + handshake loop, `rdma_kv_backend.rs:214-322`) into a shared `RdmaRailSet::connect(addr, arena_bytes, bounce_bytes) -> Vec<Rail>` used by both backends, so the snapshot store gets dual-rail throughput without duplicating verbs setup. **Do not** reuse `GroupKey`/`group_stride`.

2. **Peer arena + `u64 → offset` allocator.** The snapshot store connects to its **own** `kv_peer` instance (or a distinct port) with `total_bytes = arena_slots × blob_bytes`. Because every snapshot blob is the same fixed size (`spill_blob_bytes()`), the allocator is a trivial **fixed-slot arena**: a free-list of `arena_slots` offsets plus `HashMap<u64 key → slot_offset>`. `put(key,blob)`: if `key` already mapped, overwrite in place at its offset; else pop a free slot — **if none, return `Ok(false)`** (the graceful-drop contract; the entry misses into recompute). RDMA-WRITE the blob to `remote_base + slot_offset`, record the mapping, return `Ok(true)`. `get(key,out)`: look up the offset; absent ⇒ `Ok(false)`; `out.len() != blob_bytes` ⇒ `Ok(false)` (defensive, never scatter a wrong-sized blob); else RDMA-READ `remote_base + slot_offset` into `out`, return `Ok(true)`. `remove(key)`: return its slot to the free-list. `len`/`bytes_resident` from the map. **Grafted from design C:** carve the arena so per-snapshot offsets are STABLE and per-layer-addressable (offset + `i*(h_bytes+conv_bytes)`), so the future zero-copy path can register the same ~60 device pointers as MRs writing into the identical peer offsets — making zero-copy a drop-in write-path swap, not a re-architecture.

3. **Persistent bounce MR (graft from C).** A 63.75 MB blob per op makes the KV-style depth-16 bounce ring (~1 GB pinned/rail) wasteful. Keep **one persistent registered bounce buffer of `blob_bytes` per rail** (or chunk the blob into pipeline-depth segments striped across rails — a perf tuning knob), reused across every spill/fault. This avoids per-spill `reg_mr` cost **without** giving up the byte-testable host boundary: `put` copies the blob into the persistent bounce and RDMA-WRITEs; `get` RDMA-READs into the bounce then `copy_from_slice` into `out`.

**Plug-in point (`impl_a1.rs:~191`).** Today `ssm_tier_store: Option<Arc<dyn SnapshotBlobStore>>` is `Some(MemBlobStore::new(0))` when `ATLAS_SSM_TIER` + SSM layers. Extend the selector: when `ATLAS_SSM_RDMA_TIER=host:port` is *also* set, construct `Arc::new(RdmaSnapshotStore::connect(peer, spill_blob_bytes, arena_slots)?)` instead; on connect failure, **log and fall back** to `MemBlobStore` (never fail model init over an optional tier). `spill_blob_bytes()` and `arena_slots` (from `ATLAS_SSM_RDMA_ARENA_SLOTS`, default e.g. 512) size the arena. Everything downstream (`spill_slot`/`fault_in_slot`/`reclaim_from_cache`) is unchanged — they only see the trait.

### Ordered, byte-identical increments

- **4b-0 — Extract shared transport.** Factor the rail bring-up/handshake out of `RdmaKvBackend::connect` into `RdmaRailSet`. Pure refactor; KV path byte-identical (existing `#[ignore]` live KV round-trip test still passes). No new behavior.
- **4b-1 — `RdmaSnapshotStore` + offset allocator, mock-validated.** Implement the trait against the shared transport with a **loopback / mock rail** (or an in-process `kv_peer` over TCP+loopback verbs where available). Unit test: `put`/`get`/`remove`/`len`/`bytes_resident` **parity with `MemBlobStore`** including the over-cap/full-arena `Ok(false)` and wrong-size-`get` `Ok(false)` cases. Not wired into the model yet ⇒ byte-identical.
- **4b-2 — Gate wiring (default-off).** Add the `ATLAS_SSM_RDMA_TIER` selector at `impl_a1.rs:191` with connect-failure fallback to `MemBlobStore`. Unset ⇒ `None`/`MemBlobStore` exactly as today ⇒ byte-identical. Add a unit test that the selector returns `None` when the flag is absent and `MemBlobStore` when only `ATLAS_SSM_TIER` is set.
- **4b-3 — Live gx10 peer validation.** Stand up a second `kv_peer` instance sized `arena_slots × blob_bytes`. `#[ignore]`-gated live test (mirrors `rdma_kv_round_trip`): reuse the existing pool-level `spill_then_fault_in_preserves_bytes` / `tier_survives_slot_recycling` invariants (`ssm_snapshot.rs:637-742`) but with `RdmaSnapshotStore` as the store — assert the spill→RDMA→fault round-trip is **bit-identical**, and that a full arena returns `Ok(false)` → graceful miss. Then a live agentic smoke (35B, `ATLAS_TARGET_MODEL=holo-3.1-35b-a3b`) confirming HBM freed and warm-turn fault-ins land.

### Top hazards, with guards

1. **Cross-stream ordering (half-written spill / premature restore).** *Guard:* inherited for free — `spill_slot`'s leading `synchronize` (`ssm_snapshot.rs:434`) drains the async save before the store ever sees bytes; `fault_in_slot`'s trailing `synchronize` (`488`) commits the H2D scatter before `restore`. The store operates purely on host bytes, so it cannot violate device ordering. Additionally, `put` must **fully drain its RDMA-WRITE completion before returning** (synchronous store contract), and `get` must drain its READ before copying to `out`.
2. **Byte-identity (a corrupted restore garbles output).** *Guard:* the spill→RDMA→fault round-trip is unit-tested against `MemBlobStore` parity (4b-1) and bit-compared live (4b-3), reusing the pool's existing bit-for-bit invariants. The persistent-bounce copy is a `copy_nonoverlapping`/`copy_from_slice` of exactly `blob_bytes`.
3. **Peer OOM (256+ snapshots × 64 MB × many sessions).** *Guard:* the **fixed-slot arena caps residency**; `put` returns `Ok(false)` when the free-list is empty. `reclaim_from_cache` (`569-580`) already handles the reject → free slot → miss-into-recompute. No per-connection budget needed beyond the arena size. (Bounded-tier index drop-on-reject is a *perf* follow-up, not correctness — the miss path is already safe.)
4. **Addressing (must not be `GroupKey`).** *Guard:* the `u64 → slot-offset` allocator is a distinct scheme; the store never constructs a `GroupKey` and connects to a **separate arena/instance**, so it cannot collide with live KV groups.
5. **Latency vs recompute (remote restore must stay a win).** *Guard:* correctness is flag-independent; as a **follow-up telemetry gate**, add a cost-aware fault-in-vs-recompute decision (CX7 ~64 MB restore ≈5–7 ms vs ~4,400 recompute-tok/hit) so a slow peer degrades to recompute rather than losing throughput. Not a blocker for default-off correctness.

### DO NOT

- **Do not** register the 60 per-layer device pointers as MRs (zero-copy scatter) in this increment — the host-mediated contiguous blob already exists and is bit-testable; zero-copy is a later perf swap (design C), enabled by the stable per-layer arena offsets we lay down now.
- **Do not** reuse `RdmaKvBackend`'s `GroupKey`/`group_stride` addressing or its `write_from_host` (`src.len()==group_bytes` assert) — verified wrong layout, would corrupt live KV.
- **Do not** make the tier fail hard on peer full, connect failure, or slow restore — always degrade (`Ok(false)` / fall back to `MemBlobStore` / recompute).
- **Do not** change any default: no `ATLAS_SSM_RDMA_TIER` ⇒ existing `MemBlobStore`/drop path ⇒ byte-identical.

### Flag / gating (default-off = byte-identical)

- `ATLAS_SSM_TIER` (existing) — engages the spill tier at all. Unset ⇒ evictions drop exactly as before.
- `ATLAS_SSM_RDMA_TIER=host:port` (new) — when set *with* `ATLAS_SSM_TIER`, selects `RdmaSnapshotStore` instead of `MemBlobStore`; connect failure logs and falls back to `MemBlobStore`.
- `ATLAS_SSM_RDMA_ARENA_SLOTS` (new, default e.g. 512) — sizes the peer arena (`slots × spill_blob_bytes`).
- Reuses KV transport env: `ATLAS_KV_DUAL_RAIL`, `ATLAS_EXPERT_RDMA_DEV`/`GID`, `ATLAS_KV_RAIL2_DEV`/`GID`, `ATLAS_KV_PIPELINE_DEPTH`.

### Test strategy

- **Unit (host, no GPU/NIC):** `RdmaSnapshotStore` over a loopback/mock rail — full `SnapshotBlobStore` parity with `MemBlobStore` (round-trip, absent-key miss, wrong-size-get refused, full-arena `Ok(false)`, `remove`, `bytes_resident`). Selector test at the `impl_a1` gate (absent flag ⇒ `None`; `ATLAS_SSM_TIER` only ⇒ `MemBlobStore`).
- **Pool-level (mock GPU):** reuse `spill_then_fault_in_preserves_bytes` and `tier_survives_slot_recycling` (`ssm_snapshot.rs`) parameterized over the store — proves spill-not-drop end-to-end with the RDMA store's semantics.
- **Live (gx10 peer, `#[ignore]`):** second `kv_peer` instance; bit-identical spill→RDMA→fault round-trip + full-arena graceful-miss; then a 35B agentic smoke confirming HBM freed and warm-turn fault-ins landing.

### Follow-ups (out of scope for the correct default-off increment)

- Cost-aware fault-in-vs-recompute telemetry gate (hazard 5).
- Bounded-tier drop-on-reject index cleanup (perf; correctness already safe via miss-on-fault).
- Zero-copy per-layer device-MR write path into the pre-stabilized arena offsets (design C drop-in).

### Phase 4b — LANDED + LIVE-VALIDATED on gx10 (2026-07-07)
All three increments shipped (`d43c065` Inc 1, `8cdf8e8` Inc 2+3) + live-validated over the real CX7 link.
- **Inc 1** — RdmaSnapshotStore core (spark-model/ssm_tier.rs): fixed-slot arena allocator (free-list +
  key→slot map; overwrite-in-place / pop-free-slot / Ok(false)-when-full; transport op outside the map lock
  + rolls the allocator back on write-fail so a bad write never leaves a garbage slot mapped) +
  SnapshotTransport seam + MockSnapshotTransport. 7 parity tests vs MemBlobStore.
- **Inc 2** — RdmaSnapshotArena (spark-storage/rdma_snapshot.rs): synchronous, offset-addressed RDMA over
  the same Verbs + atlas-kv-peer blade protocol, keyed by byte offset (NOT KV GroupKey — verified-wrong).
  Dual-rail; each op drained to completion. Available unconditionally via a connect-errors stub when
  cuda+atlas_rdma_verbs isn't built (so spark-model needs no cfg and degrades to host-RAM).
- **Inc 3** — ssm_tier::build_tier_store + impl_a1 selector: `ATLAS_SSM_RDMA_TIER=host:port` +
  `ATLAS_SSM_RDMA_ARENA_SLOTS` (default 512) → RDMA arena; connect-fail/no-verbs → MemBlobStore fallback.
  Unset ⇒ byte-identical. 14/14 ssm_tier tests green; workspace (stub) + release spark-server (verbs) link.

**LIVE (gx10 peer :9917, 2nd atlas-kv-peer instance, dual-rail; dgx-00 35B-A3B, --ssm-cache-slots 4 to force
spill, ATLAS_SSM_RDMA_ARENA_SLOTS=64 = 3.98 GiB arena, blob 66,846,720 B):** log confirms `RdmaSnapshotArena
connected … 4.0 GiB arena, 2 rail(s)` + peer `client connected: 2 rail(s), 4.0 GiB RW blade`; **8 spills + 6
tier fault-ins** ("restored spilled snapshot at token 1328 into slot N") through the RDMA store; **18/18
coherent, domain-matched** responses under an interleaved 6-session workload — a corrupted remote restore
would garble temp-0 output, so the spill→RDMA→fault round-trip is byte-correct. (One-sided RDMA = the peer
CPU is uninvolved per-op and releases the arena on client disconnect, so peer RSS after teardown is ~0 —
expected, not a defect.) Peer runs as a durable systemd `--user` service (atlas-kv-peer-snap) on gx10.

**Open follow-ups (perf, not correctness):** cost-aware fault-vs-recompute gate (CX7 ~64 MB restore ≈5–7 ms
vs recompute); bounded-tier drop-on-reject index cleanup; zero-copy per-layer device-MR scatter into the
pre-stabilized arena offsets. A same-config RDMA-vs-host-RAM byte-identical A/B would further harden the
correctness claim beyond the coherent-restore evidence.
