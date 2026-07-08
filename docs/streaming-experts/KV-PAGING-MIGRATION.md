# Retiring the legacy one-sided KV path — migration plan

**Status:** blocked-work scoped (2026-07-08). The legacy one-sided path is **live**
— it is the sole transport for KV overflow (`RdmaKvBackend`). Retirement is a real
client rewrite, not a dead-code sweep. This doc is the actionable checklist so the
migration can be done in one focused session without regressing the validated
KV-tier perf (#9/#10/#11, zero-copy, dual-rail, pipeline depth).

## Why two paths exist (the divergence is deliberate)

`cache_peer` speaks one wire protocol with two handshake branches, forked on the
first u64:

- **Legacy one-sided** (bare `total_bytes` first-u64): per-connection anon arena,
  registered RW, rkey published; **peer CPU idles** while the client drives all I/O
  with one-sided RDMA READ/WRITE at `base+offset`. Client addressing is KV's
  `GroupKey`/`group_stride` space. Server data plane = `cache_peer.rs:314-326`
  (idle-until-hangup). This is what `RdmaKvBackend` uses today.
- **Paging** (`PAGING_MAGIC`/`_V2` first-u64): process-global **shared** arena, a
  live TCP control channel of tiny `[op][key]` msgs (`OP_BYE/ALLOC/COMMIT/GET/REMOVE`,
  `snapshot_swap.rs:683-687`), server-managed slot residency + **NVMe swap** for
  infinite-depth spill. Used only by the SSM snapshot tier
  (`ssm_tier.rs:220`, `RdmaSnapshotArena::connect_paging`).

Per `UNIFIED-TIER-PLAN.md:664,696`: KV's group-stride layout is the *wrong* shape
for the paging u64→offset slot allocator, and reusing `write_from_host` verbatim is
"verified-broken". The paths diverged on purpose; KV was left on legacy.

## What migration buys (why bother)

1. **Peer-side disk spill for KV overflow.** Today the peer KV arena can't grow past
   its registered size; with the new `--max-blade-gb` cap it *rejects* rather than
   OOMs, but it still can't spill. Paging gives NVMe-backed overflow on the peer.
2. **Shared arena across clients** (vs per-connection arena) — better multi-client
   memory sharing; converges onto the one server data plane.
3. **Delete the legacy branch** — one transport to maintain.

## What migration risks (why it's not a bg-job half-start)

- Re-plumbs the **validated** KV perf features — dual-rail, zero-copy UMA landing,
  pipeline depth, the #11 async prefetch-completion — onto the paging protocol,
  which "manages rails/slots differently". High regression risk against numbers we
  just measured clean (bounce 10.20 / zero-copy 9.99 tok/s).
- Requires reconciling **incompatible addressing** (KV group-stride ↔ paging
  slots), not just swapping a handshake.

## Retirement checklist (file:line)

### Client — `crates/spark-storage/src/rdma_kv_backend.rs` (the real work)
- [ ] `connect` `:246-344` — replace the bare-`total_bytes` handshake (`:272-274`)
      with a paging handshake (magic + arena/blob + `OP_*` control channel), or
      delegate to the SSM-tier's `RdmaSnapshotArena`/`RdmaRailSet`.
- [ ] `write_from_host` / `read` (`StorageBackend` impl, `:563-593`) — rework to
      `OP_ALLOC` + `OP_COMMIT` (write) and `OP_GET` (read) against server slots.
- [ ] One-sided helpers `post_read`/`post_write` (`:186-222`) and their call sites
      (`:403,:520,:589`) — become server-slot-addressed paging ops.
- [ ] Reconcile the KV group-stride addressing with the paging slot allocator
      (the load-bearing design problem — see UNIFIED-TIER-PLAN §above).
- [ ] Header doc `:3-29`.

### Server — `crates/spark-storage/src/cache_peer.rs`
- [ ] Handshake fork `:203-220` — drop the `None` (legacy) arm; require `PAGING_MAGIC`.
- [ ] Legacy per-connection arena `:234-256` (`local` binding, `try_reserve`+`Mmap::anon`)
      — delete once nothing takes the `shared.is_none()` branch.
- [ ] Legacy idle data-plane loop + teardown `:314-326` — the core deletion.
- [ ] File header `:3-22`.
- [ ] Leave the paging registry `:239-499` intact.

### Wire-in / flags
- [ ] `high_speed_swap.rs:253-278` — repoint the `$ATLAS_KV_PEER` selection to the
      migrated backend.
- [ ] Keep `$ATLAS_KV_PEER`; revisit `$ATLAS_KV_ZERO_COPY` / `ATLAS_KV_DUAL_RAIL` /
      `ATLAS_KV_RAIL2_*` / `ATLAS_KV_PIPELINE_DEPTH` (`rdma_kv_backend.rs:256-267,352`)
      — paging manages rails/slots differently.
- [ ] Deployed peer must run a `--swap-dir` paging build (already true for :9916/:9920).

### Tests / examples
- [ ] `rdma_kv_backend.rs:616-702` live `#[ignore]` tests (`rdma_kv_round_trip`,
      `rdma_kv_bandwidth`) — rewrite against the paging handshake or retire.
- [ ] `snapshot_paging_smoke.rs` already exists as the replacement pattern.

### Docs to update on completion
- `HANDOFF-2026-07-08-ssm-tier.md:133-134`, `RDMA-KV-TIER.md:78-101`,
  `UNIFIED-TIER-PLAN.md:663-696`, and this file.

## Not a blocker anymore

The `4 TiB` "sanity only" concern is closed: `--max-blade-gb` + `blade_cap::CommitLedger`
now rejects an over-budget client at the handshake before any RAM is pinned
(`cache_peer.rs:246,469`). So the peer is safe for unattended use **on the legacy
path today** — migration is about capability (disk spill / shared arena) and code
convergence, not safety.
