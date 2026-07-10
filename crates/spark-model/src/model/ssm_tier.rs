// SPDX-License-Identifier: AGPL-3.0-only

//! SSM snapshot spill tier — Phase 1 of UNIFIED-TIER-PLAN.
//!
//! Today an evicted Marconi snapshot is **dropped**: [`SsmSnapshotPool::free`]
//! returns the HBM slot to the free list and the recurrent state is discarded,
//! so the next warm turn that needed it recomputes the whole SSM prefix
//! (measured ~4,400 tok / ~7.6s TTFT on 35B — see the plan doc). This module is
//! the **spill-not-drop** substrate: an evicted snapshot's bytes are moved to a
//! cheaper tier and faulted back in on a later hit, converting *recompute* into
//! *tier-restore*.
//!
//! ## Why host-mediated (bytes → one blob → store)
//!
//! A snapshot's state is **scattered** across `2 × num_ssm_layers` device
//! allocations (`h_snapshots[i]`, `conv_snapshots[i]`, each strided by slot),
//! whereas the shipped [`spark_storage::StorageBackend::read`] lands *one*
//! contiguous blob at *one* device pointer. So the tier gathers a slot's
//! per-layer chunks D2H into a single host blob on spill, and scatters the blob
//! H2D back into a (possibly different) slot on fault-in. On GB10's unified
//! LPDDR this host blob store is itself a valid T1 tier: it frees a pinned
//! snapshot-pool slot (the scarce, fixed-size resource) for another session
//! while the bytes live in abundant UMA. A zero-copy device-landing path
//! (`register_landing_region` over the 60 per-layer destinations) is a later
//! optimization — the plan's open question — not needed for correctness.
//!
//! The byte-movement mechanism lives on [`SsmSnapshotPool`] (it needs the pool's
//! private device pointers); this file defines the **store** the bytes land in.

mod arena_store;
mod fingerprint;
mod selectors;
mod store;
mod transport;
mod unified;

pub(crate) use arena_store::{ArenaSnapshotStore, PagingSnapshotStore, RdmaSnapshotStore};
pub(crate) use fingerprint::ModelFingerprint;
pub(crate) use selectors::{build_decode_tier_store, build_tier_store, ssm_tier_enabled};
pub(crate) use store::{BlobStoreStats, MemBlobStore, SnapshotBlobStore};
pub(crate) use transport::{FileSnapshotArena, MockSnapshotTransport, SnapshotTransport};
pub(crate) use unified::{UnifiedSnapshotStore, ssm_tier_unified};
