// SPDX-License-Identifier: AGPL-3.0-only
//
// WS-A Inc 1: peer-side paging core for the SSM-snapshot spill tier.
//
// Turns the atlas-cache-peer's fixed RDMA arena into a bounded page-cache over
// an UNBOUNDED lower tier (NVMe swap file) — "infinite depth like the LoRA
// setup" (operator, 2026-07-07). The peer owns the residency map (so all fleet
// clients SHARE one warm cache instead of each owning a colliding client-side
// allocator), and the stable per-rail arena MR is NEVER re-registered — bytes
// swap under the fixed rkey, driven by a TCP control channel (Inc 2).
//
// Tiered-cache consolidation step 2: the GENERIC half of this module — the
// `SlotArena`/`SwapStore` seams, the `SnapshotResidency` page table (now
// `atlas_tier::Residency`) and the O_DIRECT `DirectSwapFile` — was lifted
// VERBATIM into the CUDA-/verbs-free `atlas-tier` crate and is re-exported
// below, so `cache_peer.rs` / `rdma_snapshot.rs` / `cache_peer_main.rs` keep
// their `crate::snapshot_swap::*` paths unchanged. What REMAINS here is the
// peer-specific half: the TCP control protocol (byte-frozen — the deployed
// gx10 peers speak v1), the paging loops, the client codec, and the
// `MmapSlotArena` over the peer's RDMA-registered mmap MR — split into the
// `wire` and `mmap_arena` sub-modules.

#![allow(dead_code)]

/// The generic paging core, lifted to `atlas-tier` (CUDA- and verbs-free).
/// `Residency` keeps its historical `SnapshotResidency` name at this path.
pub use atlas_tier::{
    DirectSwapFile, MemSwapStore, Residency as SnapshotResidency, SlotArena, SwapStats, SwapStore,
    VecSlotArena,
};

mod mmap_arena;
mod wire;

pub use mmap_arena::MmapSlotArena;
pub use wire::*;
