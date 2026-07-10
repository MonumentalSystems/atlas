# Tiered-cache consolidation + PR#9 split plan (2026-07-09)

> **Status:** design/plan only. No code changed by this document. Written after a read-only
> audit of `feat/streaming-experts-mvp` @ `0c522204`. Every claim below carries a `file:line`
> anchor against that commit; the counts were produced by grep, not from the PR description.
>
> **Thesis.** The peer daemon already contains the correct, generic tiered-cache core
> (`SlotArena` + `SwapStore` + `SnapshotResidency`). The inference process never adopted it, so
> the same mechanism has been re-implemented per tier family — five RDMA clients, four NVMe
> backends, six residency policies, five keyspaces. The consolidation is to **lift the peer's
> core into a shared crate** and make "peer over RDMA" and "local NVMe" two `SwapStore` impls of
> that core. Consolidate **code**, not **processes**.
>
> This also splits PR#9 (254 files / +36.7k / 127 commits) into nine independently reviewable
> chunks.

---

## 1. Duplication census (measured, not estimated)

| axis | distinct impls | where |
|---|---|---|
| RDMA client to a peer | **5** | `RdmaKvBackend` (`rdma_kv_backend.rs:226`), `RdmaTier` (`expert_tier_rdma.rs:56`), `RdmaWeightLoader` (`weight_tier_rdma.rs:36`), `RdmaLoraLoader` (`weight_lora_rdma.rs:113`), `RdmaSnapshotArena` (`rdma_snapshot.rs:104`) |
| NVMe / disk tier | **4** client + 1 peer | `IoUringBackend` (`backend/io_uring.rs:18`), `PosixBackend` (`backend/posix.rs:16`), `UmaArenaTier` (`expert_tier.rs:178`, O_DIRECT), `FileSnapshotArena` (`ssm_tier.rs:556`), + peer `DirectSwapFile` (`snapshot_swap.rs:454`) |
| residency / eviction policy | **6** client + 1 peer | `EvictionPolicy` (`eviction.rs:19`), `SlotCache` (`cascade_policy.rs:26`), `ScratchPool` free-list (`scratch_pool.rs:83`), `MemBlobStore` FIFO (`ssm_tier.rs:78`), `RdmaSnapshotStore` free-list (`ssm_tier.rs:406`), expert-streamer slab occupancy (`moe/streamer.rs:55`), + peer `SnapshotResidency` (`snapshot_swap.rs:84`) |
| key / namespace scheme | **5** | `GroupKey` (`group.rs`), `ExpertKey` (`expert_tier.rs:48`), snapshot `u64` prefix-hash (`ssm_tier.rs`), weight tensor-name, LoRA tensor-name |
| peer **daemons** | **3** | `atlas-cache-peer` (RW), `atlas-expert-peer` (RO), `atlas-weight-peer` (RO) — all in `crates/atlas-expert-pack/Cargo.toml` |

All five RDMA clients share exactly one primitive: `rdma_verbs::Verbs` (`rdma_verbs.rs:75`).
Everything above it — rail structs, bounce rings, dual-rail env handling, TCP handshake,
QP `INIT→RTR→RTS`, completion reaping — is hand-rolled five times.

### 1a. Two claims in the PR description that the code does not support

Recording these so the next reader doesn't plan against them:

1. **"`atlas-cache-peer` — one unified generic cache blade that serves every RDMA tier
   (experts / KV / SSM-snapshots / LoRA), each in its own arena."** There are **three** daemons.
   `atlas-cache-peer` is the only read-write one; experts, model weights and LoRA are served by
   the two **read-only** peers (`reg_mr(..., remote_read)`, `expert_peer.rs:394`,
   `weight_peer.rs:448`). The unification is a doc claim, not a code fact.

2. **The multi-tier peer seam is scaffolding, never driven.** `PagingKind` (`SSM=0`, `KV=1`,
   `snapshot_swap.rs:603`) and the v2 kind-byte handshake (`PAGING_MAGIC_V2`,
   `snapshot_swap.rs:597`) exist **server-side only**. `grep -rn 'PAGING_MAGIC_V2\|PagingKind'
   crates/` finds no client sender — every deployed client uses v1, which carries no kind byte
   and is therefore always `SSM`. No client has ever driven KV through the paging peer.

---

## 2. The reframe: the peer is not a process, it is a `SwapStore`

The buffers cannot literally move into the peer process. It is CUDA-free by construction
(`atlas-expert-pack` depends on `spark-storage` with `default-features = false`), and the hot
tier is GPU memory that kernels index directly — `tile_blocks[seq*tile_cap + b]`, the SSM pool's
~60 per-layer device pointers, and `ExpertArena` asserting host VA == device VA
(`expert_arena.rs:52`). Another process's anonymous mmap is not in the client's VA space;
recovering it means `cuMemHostRegister` over `shm_open`, or CUDA IPC handles. That is real work
for no throughput win.

What is worth taking from the peer is not its process boundary. It is the trio in
`snapshot_swap.rs`:

```rust
pub trait SlotArena: Send { ... }                              // snapshot_swap.rs:34  — where hot bytes live
pub trait SwapStore: Send { ... }                              // snapshot_swap.rs:48  — where cold bytes go
pub struct SnapshotResidency<A: SlotArena, S: SwapStore> {...} // snapshot_swap.rs:84  — the policy
```

Byte-agnostic. Opaque `u64` key. Two-level LRU (RAM over disk). Read-pins so a slot cannot be
evicted mid-RDMA-READ (`pin_read`/`unpin_read`, `snapshot_swap.rs:305-343`). **Never rejects a
`put`** — it spills instead (`alloc` → `evict_coldest_to_disk` → `make_disk_room`,
`snapshot_swap.rs:185 / :357 / :394`).

That is already a correct, generic tiered cache. The peer built the right abstraction and the
inference process never adopted it.

**The recursion that makes it general:** `S` may itself be a `SnapshotResidency`. *A tier is a
residency whose backing store is another tier.* Then:

- `RdmaSwapStore` — "cold storage is someone else's RAM" — is **one** `SwapStore` impl.
- `DirectSwapFile` — local NVMe — is **one** `SwapStore` impl, and it is the best of the five
  disk implementations on the branch (O_DIRECT, fixed stride, page-aligned bounce).
- `NullSwapStore` — drop — is the flag-off default, preserving byte-identity.
- The peer daemon is just `Residency<MmapSlotArena, DirectSwapFile>` behind a thin RPC shim.

`CascadeBackend` is today hard-wired to exactly two levels — one `PinnedStore` T1 fronting a
single `backing: Box<dyn StorageBackend>` (`cascade_backend.rs:73-76`), with the
"local-pinned → peer → NVMe" cascade achieved only by nesting the choice by hand at
`high_speed_swap.rs:575`. Under the recursion, cascade depth becomes compositional and that
hard-wiring dissolves.

---

## 3. The distinction that explains why the families diverged

This is the piece `UNIFIED-TIER-PLAN.md` Phase 6 ("one namespaced `BlobKey` space") misses,
and it is why that phase keeps getting deferred as scary hygiene with "zero new user value."

There are genuinely **two roles**. They should be **layered, not merged**:

- **`AddressedStore`** — a pure deterministic `address → bytes` projection. No allocator, no
  metadata, no lookup. **KV overflow is this**: `GroupLayout` gives a total bijection
  `(layer, block, kv_head, kind) → offset` (`group.rs:86`), never stored, never allocated. So
  are expert records at fixed stride, and weight tensors by name. *The absence of metadata is a
  feature* — it is why `IoUringBackend` can run SQPOLL with registered buffers and coalesce
  block runs.

- **`BlobStore`** — allocate / lookup / evict. Has residency and metadata. **Snapshot spill is
  this**: keys are content hashes with no natural address. So are Cascade T1, `MemBlobStore`,
  and the peer's paging arena.

`ssm_tier.rs:270` and `rdma_snapshot.rs:6-9` both explicitly **refuse** to reuse
`StorageBackend`, warning that the `GroupKey`/`group_stride` addressing "would corrupt live KV."
Both refusals are correct: they are a `BlobStore` being offered an `AddressedStore` trait. There
is a second, independent blocker — `StorageBackend::read` (`backend/mod.rs:213`) lands **one
contiguous blob at one pointer**, which cannot express the SSM snapshot's 60-destination
scatter.

**The unification is that `BlobStore` is implemented *over* an `AddressedStore`** — an allocator
on top of a flat address space. That is precisely what `RdmaSnapshotStore` (free-list over
`SnapshotTransport`, `ssm_tier.rs:406`) and the peer (`SnapshotResidency` over `MmapSlotArena`)
each independently hand-rolled. Write it once. Then two things fall out cheaply:

- **`BlobKey { ns: u32, kind: u8, id: u64 }`**, where `id` is *derived* per consumer (`GroupKey`
  → dense `group_id`; `ExpertKey` → `layer*E + e`; snapshot → prefix hash; tensor name → hash).
  The peer already only ever sees a `u64` and already folds the namespace client-side via
  `wire()` (`ssm_tier.rs:355`). **Phase 6 is much cheaper than the plan fears** — it is a
  consequence of §5 steps 2–4, not a standalone wide-diff rewrite.
- **`StorageBackend::read` grows a landing descriptor**
  `Landing { Contiguous(ptr, len) | Scatter(&[(ptr, len)]) }` plus an opaque ordering token
  instead of assuming a CUDA stream. The SSM refusal evaporates and
  `register_landing_region`'s one-contiguous-UMA-pool assumption (`backend/mod.rs:339`) stops
  being load-bearing.

---

## 4. This is not cosmetics: there is a live policy bug in it

The **same logical tier** — SSM snapshot spill — gets a different eviction policy depending on
which store happens to back it:

| backing store | policy | site |
|---|---|---|
| host RAM (`MemBlobStore`) | **FIFO** (`order.pop_front()`) | `ssm_tier.rs:78-95` |
| RDMA arena (`RdmaSnapshotStore`) | **drop-on-full**, no recency at all (`free.pop()` → `Ok(false)`) | `ssm_tier.rs:406` |
| paging peer (`SnapshotResidency`) | **two-level LRU, never rejects** | `snapshot_swap.rs:84` |

FIFO and drop-on-full are precisely wrong for the deep-tail eviction pathology that `#278`,
`ATLAS_SSM_TAIL_PROTECT`, and the session-aware victim selection all exist to fix. Two
independent engineering efforts fixed victim *selection* in the HBM pool
(`radix_tree/snapshot.rs`), and then the victim spills into a tier that picks its own victim by
**insertion order**. Unifying on the peer's `Residency` does not tidy this; it fixes it.

Related smell, same root cause: `build_tier_store` (`ssm_tier.rs:189`) and
`build_decode_tier_store` (`ssm_tier.rs:623`) are two separate selectors reading five
environment variables — `ATLAS_SSM_TIER`, `ATLAS_SSM_RDMA_TIER` (`:191`), `ATLAS_SSM_SWAP`
(`:204`), `ATLAS_SSM_DECODE_TIER` (`:628`), `ATLAS_SSM_DECODE_RDMA_TIER` (`:661`) — to build two
parallel store stacks for the *same data type*.

---

## 5. Target architecture

```
atlas-tier                     (new crate: no CUDA, no verbs, pure, fully unit-testable)
  BlobKey { ns, kind, id }
  trait SlotArena              -- Mmap | Uma | Hbm
  trait SwapStore              -- DirectSwapFile | RdmaSwapStore | Residency<..> | Null
  struct Residency<A, S>       -- one LRU, read-pins, never-reject   [lifted from snapshot_swap.rs]
  trait AddressedStore         -- deterministic addr -> bytes, no metadata
  trait BlobStore              -- alloc/lookup/evict; blanket impl over <AddressedStore + Residency>

atlas-rdma                     (new crate)
  RailSet::connect(addr, arena_bytes, bounce_bytes) -> Vec<Rail>   [the 5-way dedup]
  RdmaSwapStore : SwapStore

atlas-cache-peer               (ONE daemon, N arenas, kind from the v2 handshake byte)
  = Residency<MmapSlotArena, DirectSwapFile> per PagingKind
  + read-only arenas for experts / weights / lora (reg_mr flags differ; nothing else does)

spark-storage / spark-model    (consumers: keep the compute paths, keep GroupLayout)
```

**Consolidate code, not processes.** The one *process* consolidation worth doing is the three
daemons → one binary with `--serve kv,ssm,experts,lora`. Once `atlas-tier` exists they differ
only in `reg_mr` flags and whether a manifest is served. That makes true the thing §1a.1
currently only claims.

> **Deployment correction (measured on gx10, 2026-07-09, `systemctl --user list-units --all`).**
> An earlier revision of this section said "three-plus instances already run (`:9916` unified,
> `:9917` snapshot arena, `:9920` WS-A paging)." Not so. **Two** units run, *both* the same
> `atlas-cache-peer` binary — `atlas-cache-peer.service` (`:9916`) and
> `atlas-cache-peer-paging.service` (`:9920`). Nothing listens on `:9917`, and there are **no
> `atlas-expert-peer` / `atlas-weight-peer` units at all.** Since experts speak the expert-peer
> RO-manifest protocol (`expert_tier_rdma.rs:32` imports `expert_peer::{encode_request,
> read_manifest}`) and LoRA speaks the weight-peer one (`weight_lora_rdma.rs:163`), **the expert
> and LoRA RDMA tiers currently have no peer to talk to** — they are default-off. Only KV and
> SSM-snapshots (the `[u64 total_bytes]` → RW-arena protocol, `cache_peer.rs:265` `reg_mr_rw`)
> have a live server. This *strengthens* §1a.1: the "one unified blade" is not merely a doc
> claim over three binaries, it is a doc claim over three **incompatible wire protocols**.

---

## 6. PR#9 split

254 files / +36,682 / −1,167 across 127 commits is past the point where review means anything.
Ordered by what unblocks what. Every chunk keeps the branch's existing discipline: default-off,
byte-identical when the gate is unset.

| # | chunk | depends on | note |
|---|---|---|---|
| 1 | **Build/infra** — CUDA 13.2 Dockerfile, GDN AOT link fixes, ISL/OSL baseline | — | zero runtime risk; lands first, unblocks CI |
| 2 | **`atlas-rdma`: extract `RailSet`** | 1 | pure refactor of the 5-way duplicated bring-up. No behavior change, byte-identical, and it **shrinks every subsequent PR**. Scoped in `UNIFIED-TIER-PLAN.md` as "4b-0"; only partially done |
| 3 | **`atlas-tier`: lift the core out of `snapshot_swap.rs`** | — | moves code, adds tests, changes nothing at runtime |
| 4 | **`atlas-cache-peer` on 2+3** — WS-A NVMe paging, shared arena, per-model NS, disk cap, and *actually wiring* `PagingKind` v2 | 2, 3 | **highest-value reviewable chunk, and it has zero dependency on the inference process** — separate crate, does not link CUDA. Currently tangled with everything for no reason |
| 5 | **LoRA (§C) — its own PR, entirely** | 2 | see below |
| 6 | **KV overflow tier** — `StorageBackend` family, `CascadeBackend`, HSS, batched attend Inc 1/2/3, `#33` concurrency fix | 2 | |
| 7 | **Expert streaming / over-core MoE** — `ExpertTier`, `ExpertArena`, expert-pack, WS1/WS4/WS5 | 2 | independent of 6 |
| 8 | **SSM snapshot tier** — `ssm_pool`/`ssm_snapshot`/`ssm_tier`, radix `Location{Hbm｜Tier}`, Phases 1–4b | 3, 4 | independent of 7 |
| 9 | **Weight staging tier** (`RdmaWeightLoader`) | 2 | small |

**LoRA does not belong under the offload umbrella.** It is a weights feature that happens to
fetch over the same wire; it shares the `weight_peer` transport and nothing else. It is roughly
a third of the PR by narrative (M0/M1, per-request routing, the fused `bgmv` kernel, reviewer
roadmap `#22`–`#32`). Cutting it out alone probably halves the review surface. The reason PR#9
feels like it covers *tons of stuff* is that it is two unrelated projects sharing a branch.

The natural reading of "Offload/Overprovisioning as its own part, with KV / SSM / LoRA beneath
it" maps to chunks 6/7/8 sitting on 2+3+4 — **minus LoRA**.

---

## 7. Do not

- **Do not move the local tiers into a local peer process.** VA/registration dance, CUDA IPC, no
  throughput win. The win is code reuse, and a shared crate delivers all of it.
- **Do not collapse `GroupKey` into the blob keyspace.** The deterministic bijection is exactly
  why `IoUringBackend` needs no allocator and can coalesce block runs. Layer `BlobStore` over
  `AddressedStore`; do not merge them. `ssm_tier.rs:270`'s warning that reusing `GroupKey` would
  corrupt live KV is correct and must stay true.
- **Do not attempt Phase 6 as a big-bang namespace rewrite.** It is cheap *as a consequence of*
  chunks 2–4 (the peer already takes an opaque `u64`), and expensive as its own wide-diff
  hygiene pass — which is precisely why `UNIFIED-TIER-PLAN.md` keeps deferring it.
- **Do not unify the daemons before `atlas-tier` exists.** Merging three binaries that each
  hand-roll their own residency buys a bigger binary and nothing else.

---

## 8. Relationship to `UNIFIED-TIER-PLAN.md`

That document plans the *client-side* tier from the consumer down, and its Phase 6 gestures at
the same endpoint ("one namespaced `BlobKey` space… permanently kill KV/SSM re-divergence").
This document argues the same endpoint is reached far more cheaply from the *peer side up*,
because the peer already implements the generic core; and that the two-role split
(`AddressedStore` under `BlobStore`) is what makes the merge safe rather than a cross-write
hazard. Phases 0–5 of that plan are orthogonal and unaffected — they optimize the compute and
overlap paths, which stay consumer-specific under every proposal here.

## 9. Verification trail

Claims in this document were checked against the branch rather than the PR description:

- `crates/atlas-expert-pack/Cargo.toml` declares four `[[bin]]` targets, **three of which are
  peer daemons** — `atlas-expert-peer`, `atlas-cache-peer`, `atlas-weight-peer` (the fourth,
  `atlas-expert-pack`, is the offline expert-store builder, not a daemon);
- `grep -rn 'PAGING_MAGIC_V2\|PagingKind' crates/ --include=*.rs` → definitions, tests and two
  comments only; **no client sender**;
- the three divergent eviction policies at `ssm_tier.rs:78` (FIFO `order.pop_front()`),
  `ssm_tier.rs:406` (`free.pop()` → `Ok(false)`), `snapshot_swap.rs:185` (`alloc` →
  `evict_coldest_to_disk`, never rejects);
- all trait/struct anchors in §1 and §2 confirmed by `grep -n` at the cited lines.

Added 2026-07-09 (independent re-verification):

- the three peers speak **three different wire protocols**, not one: `cache_peer.rs:265`
  `reg_mr_rw` + `[u64 total_bytes]` arena handshake (KV, SSM) vs `expert_peer.rs:394`
  `reg_mr(..,true)` REMOTE_READ + JSON manifest (experts) vs `weight_peer.rs:448` REMOTE_READ +
  weight manifest (weights, **LoRA**). Clients confirm the split: `expert_tier_rdma.rs:32`
  imports from `expert_peer`, `weight_lora_rdma.rs:163` from `weight_peer`;
- gx10 runs only `atlas-cache-peer` (×2: `:9916`, `:9920`); no expert/weight peer units exist and
  `:9917` is not listening ⇒ the expert and LoRA RDMA tiers have no live server (see §5 note).
