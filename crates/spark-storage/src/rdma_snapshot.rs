// SPDX-License-Identifier: AGPL-3.0-only

//! Offset-addressed RDMA arena for the SSM-snapshot spill tier — a synchronous
//! transport that moves a snapshot blob to/from a remote RW blade, addressed by
//! a flat byte offset (snapshots are keyed by an opaque id → arena slot) rather
//! than the KV `GroupKey` layout.
//!
//! This is the STUB: `connect`/`connect_paging` always error, so a tier selector
//! that requests the RDMA arena degrades to the host-RAM tier. The real verbs +
//! CUDA-pinned data path (gated behind `feature = "cuda"` + `atlas_rdma_verbs`)
//! lands with the SSM-snapshot spill wiring in a follow-up PR; dependents can
//! reference the type unconditionally in the meantime.

use anyhow::{Result, bail};

/// Placeholder RDMA snapshot arena. `connect` always errors so dependents
/// degrade to the host-RAM tier; the data-plane methods are unreachable (a stub
/// arena is never successfully constructed).
pub struct RdmaSnapshotArena;

impl RdmaSnapshotArena {
    pub fn connect(_addr: &str, _arena_bytes: u64, _blob_bytes: usize) -> Result<Self> {
        bail!("RDMA snapshot tier not built (needs feature `cuda` + atlas_rdma_verbs)")
    }
    pub fn connect_paging(_addr: &str, _arena_bytes: u64, _blob_bytes: usize) -> Result<Self> {
        bail!("RDMA snapshot tier not built (needs feature `cuda` + atlas_rdma_verbs)")
    }
    pub fn write(&self, _offset: u64, _bytes: &[u8]) -> Result<()> {
        unreachable!("stub RdmaSnapshotArena is never constructed")
    }
    pub fn read(&self, _offset: u64, _out: &mut [u8]) -> Result<()> {
        unreachable!("stub RdmaSnapshotArena is never constructed")
    }
    pub fn paging_put(&self, _key: u64, _bytes: &[u8]) -> Result<()> {
        unreachable!("stub RdmaSnapshotArena is never constructed")
    }
    pub fn paging_get(&self, _key: u64, _out: &mut [u8]) -> Result<bool> {
        unreachable!("stub RdmaSnapshotArena is never constructed")
    }
    pub fn paging_remove(&self, _key: u64) -> Result<()> {
        unreachable!("stub RdmaSnapshotArena is never constructed")
    }
}
