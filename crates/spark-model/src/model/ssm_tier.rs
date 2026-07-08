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

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use parking_lot::Mutex;

/// A keyed byte-blob store backing the SSM spill tier. One blob == one
/// snapshot's full `(h,conv)×layers` state, keyed by its prefix hash (the same
/// stable identity the [`super::super::traits`] snapshot index keys on).
///
/// Implementations must be cheap to share (`Send + Sync`) — the tier is
/// consulted from the scheduler thread on eviction and on prefix lookup.
pub(crate) trait SnapshotBlobStore: Send + Sync {
    /// Store `bytes` under `key`, replacing any prior value. Returns `false`
    /// when the tier is full and refused the write — the caller then falls back
    /// to a plain drop (correct degradation, never a hard error).
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool>;

    /// Copy the blob for `key` into `out` (which must be sized to the blob).
    /// Returns `false` if `key` is absent (caller recomputes) or the length
    /// mismatches (defensive: never scatter a wrong-sized blob into a slot).
    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool>;

    /// Drop the blob for `key` if present (e.g. when its prefix is invalidated).
    fn remove(&self, key: u64);

    /// Resident blob count.
    fn len(&self) -> usize;

    /// Total resident bytes — for budget enforcement and telemetry.
    fn bytes_resident(&self) -> usize;
}

/// Aggregate spill-tier telemetry (mirrors the Phase-0 snapshot stats).
#[derive(Default)]
pub(crate) struct BlobStoreStats {
    pub puts: AtomicUsize,
    pub put_rejects: AtomicUsize,
    pub get_hits: AtomicUsize,
    pub get_misses: AtomicUsize,
    pub evictions: AtomicUsize,
}

/// In-memory host-RAM spill tier. On GB10 (unified LPDDR) this is a real T1
/// tier, not a test stand-in: spilling here frees a scarce pinned snapshot-pool
/// slot while the bytes remain in abundant UMA. Bounded by `cap_bytes` with
/// FIFO eviction so a runaway session can't exhaust host memory; `cap_bytes ==
/// 0` means unbounded.
pub(crate) struct MemBlobStore {
    inner: Mutex<MemInner>,
    bytes: AtomicUsize,
    cap_bytes: usize,
    pub stats: BlobStoreStats,
}

struct MemInner {
    map: HashMap<u64, Vec<u8>>,
    /// Insertion order for FIFO eviction when `cap_bytes` is exceeded. A key is
    /// pushed on first insert; re-`put` of an existing key overwrites in place
    /// without reordering (keeps eviction simple and allocation-free).
    order: std::collections::VecDeque<u64>,
}

impl MemBlobStore {
    /// `cap_bytes == 0` → unbounded.
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(MemInner {
                map: HashMap::new(),
                order: std::collections::VecDeque::new(),
            }),
            bytes: AtomicUsize::new(0),
            cap_bytes,
            stats: BlobStoreStats::default(),
        }
    }
}

impl SnapshotBlobStore for MemBlobStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        // A single blob larger than the whole cap can never fit — refuse rather
        // than evict everything and still fail.
        if self.cap_bytes != 0 && bytes.len() > self.cap_bytes {
            self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let mut g = self.inner.lock();
        // Overwrite in place: reclaim the old blob's bytes first.
        if let Some(old) = g.map.get(&key) {
            self.bytes.fetch_sub(old.len(), Ordering::Relaxed);
        } else {
            g.order.push_back(key);
        }
        // Evict oldest until the new blob fits under the cap.
        if self.cap_bytes != 0 {
            while self.bytes.load(Ordering::Relaxed) + bytes.len() > self.cap_bytes {
                let Some(victim) = g.order.pop_front() else { break };
                if victim == key {
                    // Don't evict the key we're inserting; requeue and stop.
                    g.order.push_front(victim);
                    break;
                }
                if let Some(v) = g.map.remove(&victim) {
                    self.bytes.fetch_sub(v.len(), Ordering::Relaxed);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        self.bytes.fetch_add(bytes.len(), Ordering::Relaxed);
        g.map.insert(key, bytes.to_vec());
        self.stats.puts.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        let g = self.inner.lock();
        match g.map.get(&key) {
            Some(v) if v.len() == out.len() => {
                out.copy_from_slice(v);
                self.stats.get_hits.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            _ => {
                self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }
        }
    }

    fn remove(&self, key: u64) {
        let mut g = self.inner.lock();
        if let Some(v) = g.map.remove(&key) {
            self.bytes.fetch_sub(v.len(), Ordering::Relaxed);
            g.order.retain(|&k| k != key);
        }
    }

    fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    fn bytes_resident(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }
}

/// Whether the SSM spill tier is engaged (`ATLAS_SSM_TIER`). Default off ⇒
/// eviction drops exactly as before ⇒ byte-identical to a pre-tier build.
pub(crate) fn ssm_tier_enabled() -> bool {
    std::env::var_os("ATLAS_SSM_TIER").is_some()
}

/// Build the SSM spill-tier store (called only when `ssm_tier_enabled()`).
/// `ATLAS_SSM_RDMA_TIER=host:port` selects the RDMA arena
/// ([`RdmaSnapshotStore`] over a peer blade, `ATLAS_SSM_RDMA_ARENA_SLOTS` slots,
/// default 512); otherwise the host-RAM [`MemBlobStore`]. A connect failure (or
/// a build without RDMA verbs) LOGS and falls back to host-RAM — the tier is
/// optional, never a hard model-init error. With `ATLAS_SSM_RDMA_TIER` unset the
/// result is exactly `MemBlobStore::new(0)` as before ⇒ byte-identical.
pub(crate) fn build_tier_store(blob_bytes: usize) -> std::sync::Arc<dyn SnapshotBlobStore> {
    use std::sync::Arc;
    if let Some(peer) = std::env::var("ATLAS_SSM_RDMA_TIER")
        .ok()
        .filter(|s| !s.is_empty())
    {
        let slots: usize = std::env::var("ATLAS_SSM_RDMA_ARENA_SLOTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let arena_bytes = slots as u64 * blob_bytes as u64;
        // WS-A: ATLAS_SSM_SWAP=1 selects PAGING mode — the peer (started with
        // --swap-dir) owns residency and backs the RAM arena with an NVMe swap
        // file, giving infinite depth (never drops) shared across clients. Falls
        // through to the bounded RDMA store / host-RAM on any connect failure.
        if std::env::var("ATLAS_SSM_SWAP").ok().as_deref() == Some("1") {
            match spark_storage::RdmaSnapshotArena::connect_paging(&peer, arena_bytes, blob_bytes) {
                Ok(arena) => {
                    tracing::info!(
                        "SSM spill tier = RDMA PAGING peer {peer} ({slots}-slot RAM cache × \
                         {blob_bytes} B + NVMe swap = infinite depth, peer-owned residency)"
                    );
                    return Arc::new(PagingSnapshotStore::new(arena, blob_bytes));
                }
                Err(e) => tracing::warn!(
                    "SSM RDMA paging connect to {peer} failed ({e:#}); trying bounded RDMA"
                ),
            }
        }
        // `connect` errors (and we fall back) both on a real connect failure and
        // in a build without the RDMA verbs (the stub arena always errors).
        match spark_storage::RdmaSnapshotArena::connect(&peer, arena_bytes, blob_bytes) {
            Ok(arena) => {
                tracing::info!(
                    "SSM spill tier = RDMA peer {peer} ({slots} slots × {blob_bytes} B = \
                     {:.2} GiB arena)",
                    arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                return Arc::new(RdmaSnapshotStore::new(Box::new(arena), blob_bytes, slots));
            }
            Err(e) => tracing::warn!(
                "SSM RDMA tier connect to {peer} failed ({e:#}); falling back to host-RAM"
            ),
        }
    }
    Arc::new(MemBlobStore::new(0))
}

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

/// WS-A paging-mode store: the PEER owns residency + an NVMe swap file, so this
/// store just forwards PUT/GET/REMOVE over the arena's control channel. Unlike
/// [`RdmaSnapshotStore`] (client-side fixed-slot allocator, `Ok(false)` when the
/// arena fills), a paging PUT **never rejects** — the peer spills the coldest
/// slot to disk. This is the "infinite depth" tier + shared across clients (one
/// peer-owned map) that the bounded RDMA/host-RAM stores can't give.
pub(crate) struct PagingSnapshotStore {
    arena: spark_storage::RdmaSnapshotArena,
    blob_bytes: usize,
}

impl PagingSnapshotStore {
    pub(crate) fn new(arena: spark_storage::RdmaSnapshotArena, blob_bytes: usize) -> Self {
        Self { arena, blob_bytes }
    }
}

impl SnapshotBlobStore for PagingSnapshotStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        if bytes.len() != self.blob_bytes {
            anyhow::bail!("paging put: {} != blob_bytes {}", bytes.len(), self.blob_bytes);
        }
        self.arena.paging_put(key, bytes)?;
        Ok(true) // never full — the peer spills to NVMe
    }
    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        if out.len() != self.blob_bytes {
            anyhow::bail!("paging get: {} != blob_bytes {}", out.len(), self.blob_bytes);
        }
        self.arena.paging_get(key, out)
    }
    fn remove(&self, key: u64) {
        if let Err(e) = self.arena.paging_remove(key) {
            tracing::debug!("paging remove {key:#x} failed: {e:#}");
        }
    }
    // Residency lives on the peer (RAM cache + NVMe); the client doesn't track it.
    fn len(&self) -> usize {
        0
    }
    fn bytes_resident(&self) -> usize {
        0
    }
}

/// RDMA snapshot spill tier: a `SnapshotBlobStore` over a remote byte arena.
/// Because every snapshot blob is the SAME fixed size (`spill_blob_bytes()`),
/// the allocator is a trivial **fixed-slot arena**: slot `i` lives at offset
/// `i * blob_bytes`; a free-list of slot indices + a `key → slot` map track
/// residency. A full arena makes `put` return `Ok(false)` — the graceful-drop
/// contract `reclaim_from_cache` already handles (free the pool slot; the entry
/// misses into recompute). All map/free-list mutation is `Mutex`-guarded; the
/// (blocking) transport op runs outside the lock and rolls the allocator back on
/// failure so a half-written slot is never left mapped (never read as garbage).
#[allow(dead_code)] // constructed by the Inc-3 gate wiring (impl_a1 selector)
pub(crate) struct RdmaSnapshotStore {
    transport: Box<dyn SnapshotTransport>,
    blob_bytes: usize,
    inner: Mutex<RdmaInner>,
    pub stats: BlobStoreStats,
}

struct RdmaInner {
    /// key → slot index (byte offset = slot × blob_bytes).
    map: HashMap<u64, usize>,
    /// Free slot indices (LIFO reuse).
    free: Vec<usize>,
}

#[allow(dead_code)]
impl RdmaSnapshotStore {
    /// Build a store over `transport` with `arena_slots` fixed slots of
    /// `blob_bytes` each. The transport's arena must cover
    /// `arena_slots × blob_bytes` bytes.
    pub(crate) fn new(
        transport: Box<dyn SnapshotTransport>,
        blob_bytes: usize,
        arena_slots: usize,
    ) -> Self {
        let free: Vec<usize> = (0..arena_slots).rev().collect();
        Self {
            transport,
            blob_bytes,
            inner: Mutex::new(RdmaInner {
                map: HashMap::new(),
                free,
            }),
            stats: BlobStoreStats::default(),
        }
    }

    #[inline]
    fn offset_of(&self, slot: usize) -> u64 {
        (slot * self.blob_bytes) as u64
    }
}

impl SnapshotBlobStore for RdmaSnapshotStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        // Fixed-slot arena: only the snapshot blob size fits. A size mismatch is
        // a caller bug — refuse gracefully rather than corrupt a slot.
        if bytes.len() != self.blob_bytes {
            self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        // Pick the slot under the lock, but DON'T commit a new mapping until the
        // write succeeds (so a failed write never leaves a garbage slot mapped).
        let (slot, was_new) = {
            let mut g = self.inner.lock();
            match g.map.get(&key) {
                Some(&slot) => (slot, false), // overwrite in place
                None => {
                    let Some(slot) = g.free.pop() else {
                        // Arena full → graceful drop (entry misses → recompute).
                        self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
                        return Ok(false);
                    };
                    (slot, true)
                }
            }
        };
        match self.transport.write_blob(self.offset_of(slot), bytes) {
            Ok(()) => {
                if was_new {
                    self.inner.lock().map.insert(key, slot);
                }
                self.stats.puts.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(e) => {
                // Roll back: a new slot returns to the free-list (nothing
                // mapped); an overwrite drops the mapping AND frees the slot —
                // its bytes may be half-overwritten, so a later `get` must miss
                // (recompute), never read a corrupted slot.
                let mut g = self.inner.lock();
                if was_new {
                    g.free.push(slot);
                } else if let Some(s) = g.map.remove(&key) {
                    g.free.push(s);
                }
                Err(e)
            }
        }
    }

    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        // Defensive: never scatter a wrong-sized blob into a slot.
        if out.len() != self.blob_bytes {
            self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let slot = match self.inner.lock().map.get(&key) {
            Some(&slot) => slot,
            None => {
                self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
                return Ok(false);
            }
        };
        self.transport.read_blob(self.offset_of(slot), out)?;
        self.stats.get_hits.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    fn remove(&self, key: u64) {
        let mut g = self.inner.lock();
        if let Some(slot) = g.map.remove(&key) {
            g.free.push(slot);
        }
    }

    fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    fn bytes_resident(&self) -> usize {
        self.inner.lock().map.len() * self.blob_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_round_trip() {
        let s = MemBlobStore::new(0);
        assert!(s.put(42, &[1, 2, 3, 4]).unwrap());
        let mut out = [0u8; 4];
        assert!(s.get(42, &mut out).unwrap());
        assert_eq!(out, [1, 2, 3, 4]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.bytes_resident(), 4);
    }

    #[test]
    fn get_absent_is_miss_not_error() {
        let s = MemBlobStore::new(0);
        let mut out = [0u8; 4];
        assert!(!s.get(7, &mut out).unwrap());
        assert_eq!(s.stats.get_misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn wrong_size_get_refused() {
        let s = MemBlobStore::new(0);
        s.put(1, &[9; 8]).unwrap();
        let mut out = [0u8; 4]; // mismatched
        assert!(!s.get(1, &mut out).unwrap(), "never scatter a wrong-sized blob");
    }

    #[test]
    fn overwrite_reclaims_bytes() {
        let s = MemBlobStore::new(0);
        s.put(1, &[0; 10]).unwrap();
        s.put(1, &[0; 3]).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.bytes_resident(), 3, "old blob bytes reclaimed on overwrite");
    }

    #[test]
    fn cap_evicts_fifo() {
        let s = MemBlobStore::new(10);
        s.put(1, &[0; 4]).unwrap(); // 4
        s.put(2, &[0; 4]).unwrap(); // 8
        s.put(3, &[0; 4]).unwrap(); // would be 12 > 10 → evict key 1 (oldest)
        assert!(s.bytes_resident() <= 10);
        let mut o = [0u8; 4];
        assert!(!s.get(1, &mut o).unwrap(), "oldest evicted");
        assert!(s.get(3, &mut o).unwrap(), "newest resident");
        assert!(s.stats.evictions.load(Ordering::Relaxed) >= 1);
    }

    #[test]
    fn blob_larger_than_cap_refused() {
        let s = MemBlobStore::new(4);
        assert!(!s.put(1, &[0; 8]).unwrap(), "over-cap blob refused, not partial");
        assert_eq!(s.len(), 0);
        assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 1);
    }

    // ── Phase 4b: RdmaSnapshotStore (over MockSnapshotTransport) ──────────
    // Fixed blob size (BLOB) + finite arena (SLOTS) so full-arena / slot-reuse
    // paths are exercised. These mirror the MemBlobStore contract above.
    const BLOB: usize = 4;
    fn rdma_store(slots: usize) -> RdmaSnapshotStore {
        let t = Box::new(MockSnapshotTransport::new(slots * BLOB));
        RdmaSnapshotStore::new(t, BLOB, slots)
    }

    #[test]
    fn rdma_put_get_round_trip_bit_identical() {
        let s = rdma_store(4);
        assert!(s.put(42, &[1, 2, 3, 4]).unwrap());
        let mut out = [0u8; BLOB];
        assert!(s.get(42, &mut out).unwrap());
        assert_eq!(out, [1, 2, 3, 4], "spill->arena->fault is bit-identical");
        assert_eq!(s.len(), 1);
        assert_eq!(s.bytes_resident(), BLOB);
    }

    #[test]
    fn rdma_get_absent_is_miss_not_error() {
        let s = rdma_store(4);
        let mut out = [0u8; BLOB];
        assert!(!s.get(7, &mut out).unwrap());
        assert_eq!(s.stats.get_misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn rdma_wrong_size_get_refused_out_untouched() {
        let s = rdma_store(4);
        s.put(1, &[9; BLOB]).unwrap();
        let mut out = [0u8; BLOB + 4]; // mismatched
        assert!(!s.get(1, &mut out).unwrap(), "never scatter a wrong-sized blob");
        assert_eq!(out, [0u8; BLOB + 4], "out left untouched on refusal");
    }

    #[test]
    fn rdma_wrong_size_put_refused() {
        let s = rdma_store(4);
        assert!(!s.put(1, &[0; BLOB + 1]).unwrap(), "off-size blob refused, not corrupt");
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn rdma_full_arena_put_returns_false_not_err() {
        let s = rdma_store(2);
        assert!(s.put(1, &[1; BLOB]).unwrap());
        assert!(s.put(2, &[2; BLOB]).unwrap());
        // Third key, no free slot → graceful Ok(false), NOT Err, NOT overwrite.
        assert!(!s.put(3, &[3; BLOB]).unwrap());
        assert_eq!(s.len(), 2);
        assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 1);
        // The two resident keys are intact.
        let mut o = [0u8; BLOB];
        assert!(s.get(1, &mut o).unwrap() && o == [1; BLOB]);
        assert!(s.get(2, &mut o).unwrap() && o == [2; BLOB]);
    }

    #[test]
    fn rdma_remove_frees_slot_for_reuse() {
        let s = rdma_store(1); // single slot
        assert!(s.put(1, &[1; BLOB]).unwrap());
        assert!(!s.put(2, &[2; BLOB]).unwrap(), "arena full");
        s.remove(1);
        assert!(s.put(2, &[2; BLOB]).unwrap(), "freed slot reused");
        let mut o = [0u8; BLOB];
        assert!(s.get(2, &mut o).unwrap() && o == [2; BLOB]);
        assert!(!s.get(1, &mut o).unwrap(), "removed key gone");
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn rdma_overwrite_in_place_no_slot_leak() {
        let s = rdma_store(1); // single slot forces in-place overwrite
        assert!(s.put(1, &[1; BLOB]).unwrap());
        assert!(s.put(1, &[2; BLOB]).unwrap(), "overwrite reuses the same slot");
        assert_eq!(s.len(), 1);
        assert_eq!(s.bytes_resident(), BLOB);
        let mut o = [0u8; BLOB];
        assert!(s.get(1, &mut o).unwrap());
        assert_eq!(o, [2; BLOB], "reads the overwritten value");
    }

    #[test]
    fn build_tier_store_defaults_to_host_ram_unbounded() {
        // With ATLAS_SSM_RDMA_TIER absent (the byte-identical default), the
        // selector yields the unbounded host-RAM store. Guarded on the var being
        // unset so a concurrent env-setting test can't flake this.
        if std::env::var_os("ATLAS_SSM_RDMA_TIER").is_none() {
            let s = build_tier_store(4);
            assert!(s.put(1, &[1, 2, 3, 4]).unwrap());
            let mut o = [0u8; 4];
            assert!(s.get(1, &mut o).unwrap());
            assert_eq!(o, [1, 2, 3, 4]);
            for k in 0..1000u64 {
                assert!(s.put(k, &[0; 4]).unwrap(), "unbounded: nothing dropped");
            }
            assert_eq!(s.len(), 1000);
        }
    }
}
