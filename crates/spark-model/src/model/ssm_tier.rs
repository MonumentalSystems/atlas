// SPDX-License-Identifier: AGPL-3.0-only

//! The SSM snapshot spill tier: the model-safety contract (the model-agnostic
//! durable-key [`ModelFingerprint`] and the [`ensure_ssm_tier_capability`]
//! gate) plus the byte-store substrate an evicted snapshot spills into.
//!
//! * [`SnapshotBlobStore`] — the seam: a keyed fixed-size blob store. Backends
//!   are hardware-free and in-process here ([`MemBlobStore`] host-RAM;
//!   [`ArenaSnapshotStore`]/[`PagingSnapshotStore`] over a [`SnapshotTransport`]/
//!   [`PagingTransport`], proven on [`MockSnapshotTransport`]/[`FileSnapshotArena`]).
//!   The RDMA transport binding and the env-driven store selection land in
//!   follow-up PRs.

mod arena_store;
mod capability;
mod fingerprint;
mod store;
mod transport;

pub(crate) use arena_store::{ArenaSnapshotStore, PagingSnapshotStore, RdmaSnapshotStore};
pub(crate) use capability::ensure_ssm_tier_capability;
pub(crate) use fingerprint::ModelFingerprint;
pub(crate) use store::{BlobStoreStats, MemBlobStore, SnapshotBlobStore};
pub(crate) use transport::{
    FileSnapshotArena, MockSnapshotTransport, PagingTransport, SnapshotTransport,
};
