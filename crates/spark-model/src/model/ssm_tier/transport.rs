// SPDX-License-Identifier: AGPL-3.0-only

//! Byte-mover plumbing: the [`SnapshotTransport`] seam and its in-process,
//! RDMA, and local-NVMe implementations.

use anyhow::Result;
use parking_lot::Mutex;

// ─────────────────────────────────────────────────────────────────────────
// Phase 4b — RDMA snapshot spill tier (`RdmaSnapshotStore`)
//
// A second `SnapshotBlobStore` that ships the (already-contiguous) spill blob to
// a remote RAM blade over RDMA instead of local host RAM. Scales warm-snapshot
// capacity past local LPDDR and frees ~16-20 GB HBM; converts an SSM-prefix
// *recompute* into a ~5-7 ms remote restore. Default-off ⇒ byte-identical.
//
// The blob gather/scatter and ALL device ordering (leading/trailing
// `synchronize`) already happen in `SsmSnapshotPool::{spill_slot,fault_in_slot}`
// before/after the store is called, so a transport only ever moves HOST bytes —
// the "60 scattered device pointers" problem is solved at the trait boundary.
// ─────────────────────────────────────────────────────────────────────────

/// Transport seam for the RDMA snapshot tier: a flat remote byte arena addressed
/// by absolute offset. The RDMA implementation (Phase 4b Inc 2, behind
/// `atlas_rdma_verbs`) ships each contiguous spill blob to a peer RAM blade over
/// CX7; `MockSnapshotTransport` is an in-process arena for unit tests. Snapshots
/// must NOT reuse the KV `RdmaKvBackend` `GroupKey`/`group_stride` addressing
/// (wrong layout — would corrupt live KV); this arena is offset-addressed only.
#[allow(dead_code)] // real (RDMA) transport + gate wiring land in Inc 2/3
pub(crate) trait SnapshotTransport: Send + Sync {
    /// Write `bytes` to the arena at absolute `offset`. The caller
    /// (`RdmaSnapshotStore`) guarantees `offset + bytes.len()` is within
    /// capacity and drains the op's completion before returning.
    fn write_blob(&self, offset: u64, bytes: &[u8]) -> Result<()>;
    /// Read `out.len()` bytes from the arena at absolute `offset` into `out`.
    fn read_blob(&self, offset: u64, out: &mut [u8]) -> Result<()>;
}

/// In-process arena transport — the unit-test / no-NIC backing for
/// `RdmaSnapshotStore`. Byte-for-byte faithful to the RDMA transport contract (a
/// flat offset-addressed arena) so store-level tests exercise the real store.
#[allow(dead_code)] // used by tests now; by the RDMA transport swap in Inc 2
pub(crate) struct MockSnapshotTransport {
    arena: Mutex<Vec<u8>>,
}

#[allow(dead_code)]
impl MockSnapshotTransport {
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        Self {
            arena: Mutex::new(vec![0u8; capacity_bytes]),
        }
    }
}

impl SnapshotTransport for MockSnapshotTransport {
    fn write_blob(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        let mut a = self.arena.lock();
        let off = offset as usize;
        a[off..off + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    fn read_blob(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        let a = self.arena.lock();
        let off = offset as usize;
        out.copy_from_slice(&a[off..off + out.len()]);
        Ok(())
    }
}

// Phase 4b Inc 2: the real transport is spark-storage's offset-addressed
// `RdmaSnapshotArena` (CX7 verbs + kv-peer blade; a `connect`-errors stub when
// verbs aren't built). We own `SnapshotTransport` here, so implementing it for
// the foreign type is allowed (no orphan rule).
impl SnapshotTransport for spark_storage::RdmaSnapshotArena {
    fn write_blob(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.write(offset, bytes)
    }
    fn read_blob(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        self.read(offset, out)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Local-NVMe transport for the decode rolling tier (`FileSnapshotArena`)
//
// The decode cold tier needs a HOST-LOCAL NVMe destination as an alternative to
// the RDMA paging peer. `spark-storage`'s `StorageBackend` lands bytes directly
// at a *device* pointer and is KV-`Layout`-coupled — the wrong contract here,
// where the pool has already gathered host bytes and wants a flat u64→bytes
// arena. So we plug a `pwrite`/`pread`-at-offset file into the SAME fixed-slot
// `ArenaSnapshotStore` the RDMA path uses. O_DIRECT is deferred (a pinned bounce
// like `posix.rs` is a later optimization); a plain buffered file is correct.
// ─────────────────────────────────────────────────────────────────────────

/// A flat offset-addressed NVMe arena backing the decode cold tier. One
/// pre-sized file; slot `i`'s blob lives at `i * blob_bytes`. `pwrite`/`pread`
/// via `FileExt::{write_at,read_at}` are offset-absolute (no shared cursor), so
/// the store's `Mutex`-guarded allocator is the only serialization needed and
/// the (blocking) I/O runs on the caller's thread — for the decode tier that is
/// always the async spill worker, never the decode critical path.
#[allow(dead_code)]
pub(crate) struct FileSnapshotArena {
    file: std::fs::File,
    capacity: u64,
}

#[allow(dead_code)]
impl FileSnapshotArena {
    /// Create/truncate a backing file of exactly `capacity` bytes under `dir`.
    /// The file name embeds the pid so two servers on one box never share a
    /// backing store (decode blobs are ephemeral, never recovered across runs).
    pub(crate) fn create(dir: &str, capacity: u64) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = std::path::Path::new(dir)
            .join(format!("atlas-decode-ring.{}.arena", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        file.set_len(capacity)?;
        Ok(Self { file, capacity })
    }
}

impl SnapshotTransport for FileSnapshotArena {
    fn write_blob(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        if offset + bytes.len() as u64 > self.capacity {
            anyhow::bail!(
                "FileSnapshotArena write {offset}+{} exceeds capacity {}",
                bytes.len(),
                self.capacity
            );
        }
        self.file.write_all_at(bytes, offset)?;
        Ok(())
    }
    fn read_blob(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        if offset + out.len() as u64 > self.capacity {
            anyhow::bail!(
                "FileSnapshotArena read {offset}+{} exceeds capacity {}",
                out.len(),
                self.capacity
            );
        }
        self.file.read_exact_at(out, offset)?;
        Ok(())
    }
}
