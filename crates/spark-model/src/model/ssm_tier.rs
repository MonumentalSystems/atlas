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

/// Opt-in truthy parse for `ATLAS_SSM_TIER_UNIFIED` (style-matching
/// `ATLAS_HSS_COALESCE_WRITE_RUNS` in spark-storage/high_speed_swap.rs).
fn unified_flag_truthy(v: Option<&str>) -> bool {
    matches!(v.map(str::trim), Some("1") | Some("true") | Some("on") | Some("yes"))
}

/// TIERED-CACHE-CONSOLIDATION §4 fix, step 3: whether the client-side SSM
/// spill stores route through the ONE lifted policy core
/// ([`atlas_tier::Residency`] — two-level LRU, never rejects) instead of the
/// per-store policies (MemBlobStore FIFO, RdmaSnapshotStore drop-on-full).
/// DEFAULT OFF ⇒ the selectors construct exactly today's stores, byte- and
/// behavior-identical.
///
/// ⚠ **BEFORE FLIPPING THIS DEFAULT ON**, three flag-ON-only defects found by the
/// step-3 adversarial review must be fixed. None affect the default path; all three
/// are latent the moment the flag is engaged in production:
///
/// 1. **Lock held across transport I/O.** The RDMA arm's `put` path holds the
///    `UnifiedSnapshotStore` mutex across a victim evict (remote READ of a ~64 MB
///    blob, ~5–7 ms) *plus* the new blob's remote WRITE. Today's `RdmaSnapshotStore`
///    does not. Split the residency map ops from the byte moves — the core already
///    exposes the two-phase `alloc`/`commit` API needed to run transport I/O outside
///    the lock.
/// 2. **Silent downgrade to unbounded host RAM.** If `UnifiedSnapshotStore::new`
///    fails *after* a successful peer connect, the RDMA arm falls back to
///    `MemBlobStore::new(0)` (unbounded host RAM) with only a warn, abandoning the
///    connected arena. It should fall through to the legacy `RdmaSnapshotStore`
///    instead — the arena is already connected.
/// 3. **Swap files leak.** Flag-ON swap files (`atlas-ssm-{tag}.{pid}.swap`,
///    `atlas-decode-ring.{pid}.swap`) are per-PID and never unlinked, and the disk
///    tier grows unbounded by design. Unlink same-tag stale files on create, or open
///    with `O_TMPFILE`.
///
/// Coverage gap to close alongside: the flag-ON **RDMA** and **decode-NVMe** selector
/// arms are only component-tested, never exercised through `build_tier_store` /
/// `build_decode_tier_store` with the env set (the host-RAM arm is).
pub(crate) fn ssm_tier_unified() -> bool {
    unified_flag_truthy(std::env::var("ATLAS_SSM_TIER_UNIFIED").ok().as_deref())
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
            // Namespace = ATLAS_SSM_SWAP_NS (explicit u64) or a hash of
            // ATLAS_TARGET_MODEL, so different models sharing one peer can't
            // collide; 0 = single-model fleet (passthrough).
            let namespace = std::env::var("ATLAS_SSM_SWAP_NS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or_else(|| match std::env::var("ATLAS_TARGET_MODEL") {
                    Ok(m) if !m.is_empty() && m != "*" => {
                        use std::hash::{Hash, Hasher};
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        m.hash(&mut h);
                        h.finish()
                    }
                    _ => 0,
                });
            match spark_storage::RdmaSnapshotArena::connect_paging(&peer, arena_bytes, blob_bytes) {
                Ok(arena) => {
                    tracing::info!(
                        "SSM spill tier = RDMA PAGING peer {peer} ({slots}-slot shared RAM cache × \
                         {blob_bytes} B + NVMe swap = infinite depth; ns={namespace:#x})"
                    );
                    return Arc::new(PagingSnapshotStore::new(arena, blob_bytes, namespace));
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
                if ssm_tier_unified() {
                    // §4 fix (the LIVE bug arm): replace the drop-on-full
                    // fixed-slot allocator with the peer's LRU/never-reject
                    // Residency, client-side, over the same remote arena — an
                    // arena-full PUT now LRU-spills the coldest blob to the
                    // local swap tier instead of silently discarding the spill.
                    let hot = Box::new(TransportSlotArena {
                        transport: Box::new(arena),
                        slot_bytes: blob_bytes,
                        num_slots: slots,
                    });
                    let swap = build_unified_swap(blob_bytes, "marconi-rdma");
                    match UnifiedSnapshotStore::new(hot, swap, blob_bytes) {
                        Ok(s) => {
                            tracing::info!(
                                "SSM spill tier = UNIFIED residency over RDMA peer {peer} \
                                 ({slots} hot slots × {blob_bytes} B, LRU spill, never rejects)"
                            );
                            return Arc::new(s);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "SSM unified residency init failed ({e:#}); \
                                 falling back to host-RAM"
                            );
                            return Arc::new(MemBlobStore::new(0));
                        }
                    }
                }
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
    if ssm_tier_unified() {
        // §4 fix (host-RAM arm): one policy core instead of the FIFO
        // MemBlobStore — a bounded LRU hot arena that spills (never rejects)
        // into the swap tier. NOTE: unlike today's lazily-growing unbounded
        // store, the hot arena is allocated up front (slots × blob_bytes).
        let hot_slots = unified_hot_slots();
        let hot = Box::new(atlas_tier::VecSlotArena::new(blob_bytes, hot_slots));
        let swap = build_unified_swap(blob_bytes, "marconi-host");
        match UnifiedSnapshotStore::new(hot, swap, blob_bytes) {
            Ok(s) => {
                tracing::info!(
                    "SSM spill tier = UNIFIED residency in host RAM ({hot_slots} hot slots × \
                     {blob_bytes} B, LRU spill, never rejects)"
                );
                return Arc::new(s);
            }
            Err(e) => tracing::warn!(
                "SSM unified residency init failed ({e:#}); falling back to host-RAM store"
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
    /// Per-model namespace folded into every key so a SHARED peer never serves
    /// one model's SSM state to another (the prefix_hash is model-independent —
    /// same tokens collide across models, but their recurrent state differs).
    /// 0 = no namespacing (single-model fleet / passthrough).
    namespace: u64,
}

impl PagingSnapshotStore {
    pub(crate) fn new(
        arena: spark_storage::RdmaSnapshotArena,
        blob_bytes: usize,
        namespace: u64,
    ) -> Self {
        Self { arena, blob_bytes, namespace }
    }

    /// Fold the namespace into a key (splitmix64 on `key ^ ns`). Deterministic
    /// (same model+key → same wire key → cache hit) with negligible cross-model
    /// collision, same 64-bit contract as `prefix_hash` itself.
    fn wire(&self, key: u64) -> u64 {
        if self.namespace == 0 {
            return key;
        }
        let mut h = key ^ self.namespace.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 30;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^ (h >> 31)
    }
}

impl SnapshotBlobStore for PagingSnapshotStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        if bytes.len() != self.blob_bytes {
            anyhow::bail!("paging put: {} != blob_bytes {}", bytes.len(), self.blob_bytes);
        }
        self.arena.paging_put(self.wire(key), bytes)?;
        Ok(true) // never full — the peer spills to NVMe
    }
    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        if out.len() != self.blob_bytes {
            anyhow::bail!("paging get: {} != blob_bytes {}", out.len(), self.blob_bytes);
        }
        self.arena.paging_get(self.wire(key), out)
    }
    fn remove(&self, key: u64) {
        if let Err(e) = self.arena.paging_remove(self.wire(key)) {
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

/// The fixed-slot offset-addressed arena store ([`RdmaSnapshotStore`]) is
/// transport-agnostic — it runs equally over the RDMA arena or a local file
/// (see [`FileSnapshotArena`]). `ArenaSnapshotStore` is the transport-neutral
/// name the decode rolling tier selects; `RdmaSnapshotStore` remains as an alias
/// for the existing Marconi call sites.
pub(crate) type ArenaSnapshotStore = RdmaSnapshotStore;

// ─────────────────────────────────────────────────────────────────────────
// §4 unification (TIERED-CACHE-CONSOLIDATION step 3) — ATLAS_SSM_TIER_UNIFIED
//
// The SAME logical tier historically got a DIFFERENT eviction policy per
// backing store: MemBlobStore evicts FIFO by insertion order (latent — the
// production cap is always 0), RdmaSnapshotStore drops-on-full with no recency
// at all (live), while the peer's paging Residency does two-level LRU and
// never rejects. FIFO/drop-on-full defeat the HBM pool's session-aware victim
// selection: the carefully chosen victim spills into a tier that re-picks its
// own victim by insertion order — or silently discards it.
//
// Flag ON routes the client-side spill stores through the ONE policy core
// lifted from the peer (`atlas_tier::Residency`: LRU over a hot arena, spill
// to a swap tier, NEVER reject, uncapped disk ⇒ nothing ever dropped). Flag
// OFF (default) constructs exactly today's stores — byte/behavior-identical.
// The gather/scatter of the ~60 per-layer device regions stays ABOVE this
// boundary in SsmSnapshotPool::{spill_slot,fault_in_slot}; the store only ever
// moves ONE contiguous host blob, so no scatter-capable SwapStore is needed
// and the StorageBackend refusals above remain true.
// ─────────────────────────────────────────────────────────────────────────

/// Adapts a [`SnapshotTransport`] (flat offset-addressed remote/file arena) to
/// the [`atlas_tier::SlotArena`] hot-tier seam: slot `i` lives at offset
/// `i × slot_bytes` — the same fixed-slot geometry [`RdmaSnapshotStore`] uses,
/// so the peer arena layout is unchanged under the flag.
struct TransportSlotArena {
    transport: Box<dyn SnapshotTransport>,
    slot_bytes: usize,
    num_slots: usize,
}

impl atlas_tier::SlotArena for TransportSlotArena {
    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    fn num_slots(&self) -> usize {
        self.num_slots
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        if slot >= self.num_slots || out.len() != self.slot_bytes {
            anyhow::bail!("TransportSlotArena::read_slot({slot}) out of range / size mismatch");
        }
        self.transport.read_blob((slot * self.slot_bytes) as u64, out)
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        if slot >= self.num_slots || bytes.len() != self.slot_bytes {
            anyhow::bail!("TransportSlotArena::write_slot({slot}) out of range / size mismatch");
        }
        self.transport.write_blob((slot * self.slot_bytes) as u64, bytes)
    }
}

/// The flag-ON [`SnapshotBlobStore`]: a `Mutex`-shared [`atlas_tier::Residency`]
/// (the peer's exact paging core, in-process). PUT never returns `Ok(false)`
/// for a right-sized blob — a full hot arena LRU-spills its coldest resident
/// into the swap tier, and the uncapped disk (`max_disk_slots = 0`) means
/// nothing is ever dropped, which also satisfies the decode tier's HARD
/// non-dropping requirement BY CONSTRUCTION rather than by sizing. The Mutex
/// is held across the byte move — the same tradeoff the peer's
/// `run_paging_loop_shared` documents (map op + one blob memcpy per call).
pub(crate) struct UnifiedSnapshotStore {
    inner: Mutex<atlas_tier::Residency<Box<dyn atlas_tier::SlotArena>, Box<dyn atlas_tier::SwapStore>>>,
    blob_bytes: usize,
    pub stats: BlobStoreStats,
}

impl UnifiedSnapshotStore {
    fn new(
        arena: Box<dyn atlas_tier::SlotArena>,
        swap: Box<dyn atlas_tier::SwapStore>,
        blob_bytes: usize,
    ) -> Result<Self> {
        // Uncapped disk tier: keys are NEVER dropped (a capped disk would let
        // make_disk_room silently discard live decode rollback targets).
        let residency = atlas_tier::Residency::new(arena, swap)?;
        Ok(Self { inner: Mutex::new(residency), blob_bytes, stats: BlobStoreStats::default() })
    }
}

impl SnapshotBlobStore for UnifiedSnapshotStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        // Fixed-size tier: an off-size blob is a caller bug — refuse gracefully
        // (same contract as RdmaSnapshotStore), never corrupt a slot.
        if bytes.len() != self.blob_bytes {
            self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        self.inner.lock().put_blob(key, bytes)?;
        self.stats.puts.fetch_add(1, Ordering::Relaxed);
        Ok(true) // never full — the residency spills, it doesn't reject
    }

    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        // Defensive: never scatter a wrong-sized blob into a slot.
        if out.len() != self.blob_bytes {
            self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let hit = self.inner.lock().get_blob(key, out)?;
        if hit {
            self.stats.get_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
        }
        Ok(hit)
    }

    fn remove(&self, key: u64) {
        self.inner.lock().remove(key);
    }

    fn len(&self) -> usize {
        self.inner.lock().total_keys()
    }

    fn bytes_resident(&self) -> usize {
        // Hot (RAM-arena) bytes; swapped records live in the swap tier.
        self.inner.lock().resident_count() * self.blob_bytes
    }
}

/// The unified stores' swap tier. `ATLAS_SSM_TIER_SWAP_DIR` selects the lifted
/// O_DIRECT NVMe swap file (needs a 4 KiB-multiple blob — the O_DIRECT
/// stride); otherwise (or on any setup failure) host-RAM records — still
/// LRU-ordered and never-reject, just RAM-resident like today's stores.
fn build_unified_swap(blob_bytes: usize, tag: &str) -> Box<dyn atlas_tier::SwapStore> {
    if let Some(dir) = std::env::var("ATLAS_SSM_TIER_SWAP_DIR").ok().filter(|s| !s.is_empty()) {
        if blob_bytes > 0 && blob_bytes.is_multiple_of(4096) {
            let make = || -> Result<atlas_tier::DirectSwapFile> {
                std::fs::create_dir_all(&dir)?;
                let path = std::path::Path::new(&dir)
                    .join(format!("atlas-ssm-{tag}.{}.swap", std::process::id()));
                atlas_tier::DirectSwapFile::create(&path, blob_bytes)
            };
            match make() {
                Ok(f) => {
                    tracing::info!("unified SSM tier ({tag}): O_DIRECT swap file in {dir}");
                    return Box::new(f);
                }
                Err(e) => tracing::info!(
                    "unified SSM tier ({tag}): swap dir {dir} unusable ({e:#}); \
                     using host-RAM swap"
                ),
            }
        } else {
            tracing::info!(
                "unified SSM tier ({tag}): blob_bytes {blob_bytes} is not a 4 KiB multiple \
                 (O_DIRECT stride); using host-RAM swap"
            );
        }
    }
    Box::new(atlas_tier::MemSwapStore::new(blob_bytes))
}

/// Hot-arena slot count for the unified stores (`ATLAS_SSM_TIER_SLOTS`,
/// default 64). The hot arena is allocated up front at `slots × blob_bytes`.
fn unified_hot_slots() -> usize {
    std::env::var("ATLAS_SSM_TIER_SLOTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1)
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

/// Build the **decode rolling-tier** cold store (a SEPARATE instance from the
/// Marconi `build_tier_store`, its own `ATLAS_SSM_DECODE_*` env namespace so keys
/// and budgets never collide). Non-dropping is a HARD requirement: a dropped
/// decode blob is a lost rollback target = corrupt restore (unlike Marconi's
/// miss→recompute). `min_slots` = `(ring_slots − hot_lanes) × max_batch_size` is
/// the worst-case cold residency; the local NVMe arena is sized ≥ that and its
/// undersizing is a preflight ERROR, never a warn.
///
/// Selection (`ATLAS_SSM_DECODE_TIER`):
///   - `nvme` + `ATLAS_SSM_DECODE_NVME_DIR=<dir>` → [`FileSnapshotArena`] behind
///     [`ArenaSnapshotStore`], provably sized ≥ `min_slots`.
///   - `peer`  + `ATLAS_SSM_DECODE_RDMA_TIER=host:port` → the never-dropping
///     [`PagingSnapshotStore`] (peer LRU-spills to its own NVMe), own
///     `ATLAS_SSM_DECODE_NS` namespace fold.
///   - unset / anything else → unbounded host-RAM [`MemBlobStore::new(0)`].
pub(crate) fn build_decode_tier_store(
    blob_bytes: usize,
    min_slots: usize,
) -> Result<std::sync::Arc<dyn SnapshotBlobStore>> {
    use std::sync::Arc;
    match std::env::var("ATLAS_SSM_DECODE_TIER").ok().as_deref() {
        Some("nvme") => {
            let dir = std::env::var("ATLAS_SSM_DECODE_NVME_DIR")
                .ok()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "ATLAS_SSM_DECODE_TIER=nvme requires ATLAS_SSM_DECODE_NVME_DIR=<dir>"
                    )
                })?;
            if ssm_tier_unified() && blob_bytes > 0 && blob_bytes.is_multiple_of(4096) {
                // §4 unification: a RAM hot cache over the lifted O_DIRECT swap
                // file. The uncapped disk tier (max_disk_slots = 0) makes
                // NON-DROPPING hold BY CONSTRUCTION instead of by arena sizing
                // — a decode rollback target can never be refused or dropped.
                std::fs::create_dir_all(&dir)?;
                let path = std::path::Path::new(&dir)
                    .join(format!("atlas-decode-ring.{}.swap", std::process::id()));
                let swap = atlas_tier::DirectSwapFile::create(&path, blob_bytes)?;
                let hot_slots = unified_hot_slots().min(min_slots + 1);
                let hot = Box::new(atlas_tier::VecSlotArena::new(blob_bytes, hot_slots));
                let store = UnifiedSnapshotStore::new(hot, Box::new(swap), blob_bytes)?;
                tracing::info!(
                    "SSM decode cold tier = UNIFIED residency ({hot_slots} hot RAM slots + \
                     O_DIRECT swap in {dir}; non-dropping by construction ≥ min_slots \
                     {min_slots})"
                );
                return Ok(Arc::new(store));
            }
            if ssm_tier_unified() {
                tracing::info!(
                    "SSM decode cold tier: ATLAS_SSM_TIER_UNIFIED set but blob_bytes \
                     {blob_bytes} is not a 4 KiB multiple (O_DIRECT stride); keeping the \
                     sized arena store"
                );
            }
            // Provision to the worst-case cold residency + headroom slot so the
            // non-dropping invariant holds by construction (an undersized arena
            // would return Ok(false) on a live target = corruption).
            let slots = min_slots + 1;
            let capacity = slots as u64 * blob_bytes as u64;
            let arena = FileSnapshotArena::create(&dir, capacity)?;
            tracing::info!(
                "SSM decode cold tier = LOCAL NVMe {dir} ({slots} slots × {blob_bytes} B = \
                 {:.2} GiB, non-dropping ≥ min_slots {min_slots})",
                capacity as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            Ok(Arc::new(ArenaSnapshotStore::new(
                Box::new(arena),
                blob_bytes,
                slots,
            )))
        }
        Some("peer") => {
            let peer = std::env::var("ATLAS_SSM_DECODE_RDMA_TIER")
                .ok()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "ATLAS_SSM_DECODE_TIER=peer requires ATLAS_SSM_DECODE_RDMA_TIER=host:port"
                    )
                })?;
            // Own namespace so decode spills never contend with the primary
            // Marconi tier keys on the shared atlas-cache-peer.
            let namespace = std::env::var("ATLAS_SSM_DECODE_NS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(atlas_kernels::DECODE_DOMAIN);
            // Arena RAM cache size is a hint; the peer pages to its own NVMe so
            // the store never drops regardless of this slot count.
            let slots = (min_slots + 1).max(512);
            let arena_bytes = slots as u64 * blob_bytes as u64;
            let arena =
                spark_storage::RdmaSnapshotArena::connect_paging(&peer, arena_bytes, blob_bytes)?;
            tracing::info!(
                "SSM decode cold tier = RDMA PAGING peer {peer} (non-dropping, ns={namespace:#x})"
            );
            Ok(Arc::new(PagingSnapshotStore::new(arena, blob_bytes, namespace)))
        }
        _ => {
            tracing::info!("SSM decode cold tier = host-RAM (unbounded, non-dropping)");
            Ok(Arc::new(MemBlobStore::new(0)))
        }
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

    // ── Decode rolling tier: FileSnapshotArena (local NVMe) + selector ────
    #[test]
    fn file_arena_round_trip_bit_identical() {
        let dir = std::env::temp_dir().join(format!("atlas-decode-test-{}", std::process::id()));
        let dir = dir.to_str().unwrap();
        let store = ArenaSnapshotStore::new(
            Box::new(FileSnapshotArena::create(dir, 4 * BLOB as u64).unwrap()),
            BLOB,
            4,
        );
        assert!(store.put(0xDEAD, &[9, 8, 7, 6]).unwrap());
        let mut out = [0u8; BLOB];
        assert!(store.get(0xDEAD, &mut out).unwrap());
        assert_eq!(out, [9, 8, 7, 6], "file spill->arena->fault is bit-identical");
        // Slot recycle: distinct key reuses a fresh slot, both recover.
        assert!(store.put(0xBEEF, &[1, 2, 3, 4]).unwrap());
        let mut o2 = [0u8; BLOB];
        assert!(store.get(0xBEEF, &mut o2).unwrap() && o2 == [1, 2, 3, 4]);
        assert!(store.get(0xDEAD, &mut out).unwrap() && out == [9, 8, 7, 6]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn file_arena_write_past_capacity_errs_not_corrupts() {
        let dir = std::env::temp_dir().join(format!("atlas-decode-cap-{}", std::process::id()));
        let dir = dir.to_str().unwrap();
        let arena = FileSnapshotArena::create(dir, BLOB as u64).unwrap();
        assert!(arena.write_blob(0, &[1; BLOB]).is_ok());
        assert!(arena.write_blob(1, &[1; BLOB]).is_err(), "over-capacity write refused");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn decode_tier_defaults_to_host_ram_non_dropping() {
        // With ATLAS_SSM_DECODE_TIER unset the decode store is unbounded host-RAM
        // and never drops (the correctness floor). Guard on the var being unset.
        if std::env::var_os("ATLAS_SSM_DECODE_TIER").is_none() {
            let s = build_decode_tier_store(4, /*min_slots*/ 8).unwrap();
            for k in 0..2000u64 {
                assert!(s.put(k, &[0; 4]).unwrap(), "non-dropping: nothing refused");
            }
            assert_eq!(s.len(), 2000);
        }
    }

    // ── §4 unification (ATLAS_SSM_TIER_UNIFIED) ────────────────────────────

    #[test]
    fn unified_flag_truthy_parse_matches_hss_style() {
        for on in ["1", "true", "on", "yes", " 1 ", "yes "] {
            assert!(unified_flag_truthy(Some(on)), "{on:?} must engage the flag");
        }
        for off in ["", "0", "false", "off", "no", "TRUE", "2"] {
            assert!(!unified_flag_truthy(Some(off)), "{off:?} must stay off");
        }
        assert!(!unified_flag_truthy(None), "unset = default OFF");
    }

    fn unified_store(slots: usize) -> UnifiedSnapshotStore {
        UnifiedSnapshotStore::new(
            Box::new(atlas_tier::VecSlotArena::new(BLOB, slots)),
            Box::new(atlas_tier::MemSwapStore::new(BLOB)),
            BLOB,
        )
        .unwrap()
    }

    /// THE §4 fix: where the bounded stores FIFO-evict or drop-on-full, the
    /// unified store never rejects — overflow LRU-spills to the swap tier and
    /// every key faults back byte-identical.
    #[test]
    fn unified_store_never_rejects_and_faults_back() {
        let s = unified_store(2);
        for k in 0..32u64 {
            assert!(s.put(k, &[k as u8; BLOB]).unwrap(), "put {k} must never be refused");
        }
        assert_eq!(s.len(), 32, "all keys tracked — nothing dropped");
        let mut o = [0u8; BLOB];
        for k in 0..32u64 {
            assert!(s.get(k, &mut o).unwrap(), "key {k} present");
            assert_eq!(o, [k as u8; BLOB], "key {k} byte-identical");
        }
        assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 0);
    }

    /// LRU (not FIFO): touching the oldest-inserted key protects it — the
    /// spill victim is the least-recently-USED key. (A capped MemBlobStore
    /// would evict key 1 here; RdmaSnapshotStore would refuse key 3 outright.)
    #[test]
    fn unified_store_victim_is_lru_not_fifo_and_not_a_reject() {
        let s = unified_store(2);
        assert!(s.put(1, &[1; BLOB]).unwrap());
        assert!(s.put(2, &[2; BLOB]).unwrap());
        let mut o = [0u8; BLOB];
        assert!(s.get(1, &mut o).unwrap()); // touch 1 → 2 is now coldest
        assert!(s.put(3, &[3; BLOB]).unwrap(), "no drop-on-full");
        assert_eq!(s.bytes_resident(), 2 * BLOB, "two hot slots resident");
        // The hot-again key SURVIVED IN THE HOT TIER: getting key 1 is a
        // resident hit (no disk fault), where FIFO would have evicted it as
        // oldest-inserted; key 2 was the LRU spill victim and faults back.
        let faults0 = s.inner.lock().stats().faults_from_disk;
        assert!(s.get(1, &mut o).unwrap(), "hot-again key survives");
        assert_eq!(o, [1u8; BLOB]);
        assert_eq!(
            s.inner.lock().stats().faults_from_disk,
            faults0,
            "key 1 was still RESIDENT — the LRU victim was key 2, not the FIFO-oldest"
        );
        assert!(s.get(2, &mut o).unwrap(), "spilled key faults back, never dropped");
        assert_eq!(o, [2u8; BLOB]);
        assert_eq!(
            s.inner.lock().stats().faults_from_disk,
            faults0 + 1,
            "key 2 came back via a disk fault"
        );
        assert!(s.get(3, &mut o).unwrap());
        assert_eq!(o, [3u8; BLOB]);
    }

    /// Read-pins are honored through the unified store: a read-pinned key can
    /// never be chosen as the LRU spill victim while pinned (the peer's
    /// mid-RDMA-READ guarantee survives the in-process adoption), and returns
    /// to normal LRU rotation after the last unpin.
    #[test]
    fn unified_store_honors_read_pins() {
        let s = unified_store(2);
        assert!(s.put(1, &[1; BLOB]).unwrap());
        assert!(s.put(2, &[2; BLOB]).unwrap());
        s.inner.lock().pin_read(1);
        // Churn well past arena capacity: every spill victim must be a key
        // OTHER than the pinned one.
        for k in 10..20u64 {
            assert!(s.put(k, &[k as u8; BLOB]).unwrap(), "puts never rejected while pinned");
        }
        let mut o = [0u8; BLOB];
        {
            let mut r = s.inner.lock();
            assert_eq!(r.read_pin_count(1), 1);
            let faults0 = r.stats().faults_from_disk;
            assert!(r.get_blob(1, &mut o).unwrap(), "pinned key present");
            assert_eq!(
                r.stats().faults_from_disk,
                faults0,
                "pinned key stayed RESIDENT through the churn — never spilled"
            );
            r.unpin_read(1);
            assert_eq!(r.read_pin_count(1), 0);
        }
        assert_eq!(o, [1u8; BLOB], "pinned key bytes intact");
        // After the last unpin the key is evictable again: more churn spills
        // it, and it faults back byte-identical (never dropped).
        for k in 20..30u64 {
            assert!(s.put(k, &[k as u8; BLOB]).unwrap());
        }
        let faults1 = s.inner.lock().stats().faults_from_disk;
        assert!(s.get(1, &mut o).unwrap(), "unpinned key spilled but never dropped");
        assert_eq!(o, [1u8; BLOB]);
        assert_eq!(
            s.inner.lock().stats().faults_from_disk,
            faults1 + 1,
            "unpinned key was evicted normally and faulted back from swap"
        );
    }

    #[test]
    fn unified_store_wrong_size_refused_gracefully() {
        let s = unified_store(2);
        assert!(!s.put(1, &[0; BLOB + 1]).unwrap(), "off-size put refused, not corrupt");
        assert!(s.put(1, &[7; BLOB]).unwrap());
        let mut big = [0u8; BLOB + 4];
        assert!(!s.get(1, &mut big).unwrap(), "never scatter a wrong-sized blob");
        assert_eq!(big, [0u8; BLOB + 4], "out untouched on refusal");
    }

    #[test]
    fn unified_store_remove_is_clean_miss() {
        let s = unified_store(2);
        assert!(s.put(1, &[1; BLOB]).unwrap());
        s.remove(1);
        let mut o = [0u8; BLOB];
        assert!(!s.get(1, &mut o).unwrap());
        assert_eq!(s.len(), 0);
    }

    /// Unified over the SAME transport geometry the bounded RDMA store uses:
    /// where `RdmaSnapshotStore` returns Ok(false) at slot 5, the unified wrap
    /// keeps accepting (LRU spill to the swap tier) — the live §4 bug arm.
    #[test]
    fn unified_over_transport_never_drops_where_bounded_store_did() {
        const SLOTS: usize = 4;
        let hot = Box::new(TransportSlotArena {
            transport: Box::new(MockSnapshotTransport::new(SLOTS * BLOB)),
            slot_bytes: BLOB,
            num_slots: SLOTS,
        });
        let s = UnifiedSnapshotStore::new(
            hot,
            Box::new(atlas_tier::MemSwapStore::new(BLOB)),
            BLOB,
        )
        .unwrap();
        let mut o = [0u8; BLOB];
        for k in 0..16u64 {
            assert!(s.put(k, &[k as u8; BLOB]).unwrap(), "arena-full put {k} accepted");
        }
        for k in 0..16u64 {
            assert!(s.get(k, &mut o).unwrap(), "key {k} recoverable");
            assert_eq!(o, [k as u8; BLOB]);
        }
    }

    /// DEFAULT-OFF byte/behavior identity: with the flag unset the selectors
    /// construct exactly today's stores with today's policies — the bounded
    /// arena still drop-on-fulls here, the FIFO MemBlobStore still evicts
    /// oldest-inserted (`cap_evicts_fifo` above), and `build_tier_store` still
    /// yields the unbounded host-RAM store
    /// (`build_tier_store_defaults_to_host_ram_unbounded` below).
    #[test]
    fn unified_flag_default_off_preserves_todays_policies() {
        // Pure-logic half: holds regardless of the ambient environment.
        assert!(!unified_flag_truthy(None), "absent env ⇒ flag OFF");
        // Env-dependent half. This is the §4 default-OFF regression guard, so it
        // must NEVER pass vacuously: fail loudly rather than skip when the flag is
        // exported into the test environment.
        assert!(
            std::env::var_os("ATLAS_SSM_TIER_UNIFIED").is_none(),
            "ATLAS_SSM_TIER_UNIFIED is set in the test environment — unset it. This test is \
             the default-OFF regression guard for the TIERED-CACHE-CONSOLIDATION §4 fix; \
             skipping it would green-light a change to the default path. To exercise the \
             flag-ON arms, run the flag-ON tests selectively instead of exporting the var \
             across the whole suite."
        );
        assert!(!ssm_tier_unified(), "flag must default OFF");
        let s = rdma_store(1);
        assert!(s.put(1, &[1; BLOB]).unwrap());
        assert!(!s.put(2, &[2; BLOB]).unwrap(), "flag OFF: drop-on-full unchanged");
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
