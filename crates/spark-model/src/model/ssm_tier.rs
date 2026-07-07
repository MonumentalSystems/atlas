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
}
