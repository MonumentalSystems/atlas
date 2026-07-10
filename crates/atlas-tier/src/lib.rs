// SPDX-License-Identifier: AGPL-3.0-only

//! The generic tiered-cache core — lifted VERBATIM from
//! `spark-storage/src/snapshot_swap.rs` (tiered-cache consolidation step 2,
//! docs/streaming-experts/TIERED-CACHE-CONSOLIDATION.md).
//!
//! One mechanism, three seams:
//!   * [`SlotArena`]  — the bounded HOT tier: `num_slots` fixed-size byte slots
//!     (the peer's mmap'd RDMA MR, a host-RAM `Vec`, …). No CUDA/HBM impl may
//!     live in this crate — that belongs to consumer crates.
//!   * [`SwapStore`]  — the unbounded COLD tier: a fixed-stride record store
//!     ([`DirectSwapFile`] on NVMe, [`MemSwapStore`] in RAM, …).
//!   * [`Residency`]  — the page table over both (was `SnapshotResidency`):
//!     opaque `u64` key → byte-agnostic fixed-size blob, two-level LRU (RAM
//!     `lru` above `disk_lru`), read-pins so a slot is never reused mid-read,
//!     and NEVER-reject puts (a full arena spills the coldest resident to
//!     disk; a capped disk drops its coldest key → clean later miss).
//!
//! This crate is CPU/disk-only: deps are `anyhow` + `libc` (unix). It is fully
//! unit-testable without RDMA or a GPU, and it is the reason the peer daemons
//! (`atlas-expert-pack`, `spark-storage` with `default-features = false`)
//! build CUDA-free. The peer wire protocol (PAGING_MAGIC, paging loops, client
//! codec) deliberately did NOT move — it stays in `spark-storage::snapshot_swap`,
//! which re-exports this core so existing consumers compile unchanged.

use std::collections::{HashMap, VecDeque};

use anyhow::{Result, bail};

/// The hot tier: a RAM arena as a set of `num_slots` fixed-size slots. The
/// cache peer implements this over its `mmap`'d MR (page-aligned →
/// O_DIRECT-safe); in-process consumers use [`VecSlotArena`].
pub trait SlotArena: Send {
    fn slot_bytes(&self) -> usize;
    fn num_slots(&self) -> usize;
    /// Copy arena slot → `out` (for spilling a victim to disk). `out.len()`
    /// MUST equal `slot_bytes()`.
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()>;
    /// Copy `bytes` → arena slot (for faulting a record back in). `bytes.len()`
    /// MUST equal `slot_bytes()`.
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()>;
}

/// The cold tier: an unbounded fixed-stride record store addressed by a
/// monotonic `disk_slot` index. The peer implements this over an O_DIRECT NVMe
/// file ([`DirectSwapFile`]); [`MemSwapStore`] is the host-RAM variant.
pub trait SwapStore: Send {
    fn record_bytes(&self) -> usize;
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()>;
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()>;
    /// Optional: reclaim disk space for a freed slot (default no-op; a hole in
    /// a preallocated file is fine — the free-list reuses the index).
    fn discard_record(&mut self, _disk_slot: usize) {}
}

// Boxed trait objects compose (lets a consumer pick arena/swap impls at
// runtime: `Residency<Box<dyn SlotArena>, Box<dyn SwapStore>>`).
impl<T: SlotArena + ?Sized> SlotArena for Box<T> {
    fn slot_bytes(&self) -> usize {
        (**self).slot_bytes()
    }
    fn num_slots(&self) -> usize {
        (**self).num_slots()
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        (**self).read_slot(slot, out)
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        (**self).write_slot(slot, bytes)
    }
}

impl<T: SwapStore + ?Sized> SwapStore for Box<T> {
    fn record_bytes(&self) -> usize {
        (**self).record_bytes()
    }
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        (**self).write_record(disk_slot, bytes)
    }
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        (**self).read_record(disk_slot, out)
    }
    fn discard_record(&mut self, disk_slot: usize) {
        (**self).discard_record(disk_slot)
    }
}

/// Where a key's blob currently lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loc {
    /// Arena slot handed out for an in-flight PUT; pinned (not evictable) until
    /// `commit`. Holds the caller's about-to-be-written bytes.
    Reserved(usize),
    /// Live in an arena slot (RDMA-readable now).
    Resident(usize),
    /// Spilled to a disk record; a GET faults it back into a slot.
    OnDisk(usize),
}

#[derive(Default, Debug, Clone)]
pub struct SwapStats {
    pub puts: u64,
    pub gets: u64,
    pub get_miss: u64,
    pub spills_to_disk: u64,
    pub faults_from_disk: u64,
    pub resident_hits: u64,
    /// Cold on-disk snapshots dropped because the disk cap was hit (a later GET
    /// for one cleanly misses → recompute).
    pub disk_evictions: u64,
}

/// The page table: `key → Loc` over a bounded [`SlotArena`] (hot) backed by an
/// unbounded [`SwapStore`] (cold), with LRU eviction of resident slots to disk.
/// (Historically `SnapshotResidency` — `spark-storage::snapshot_swap` re-exports
/// it under that name for the peer.)
pub struct Residency<A: SlotArena, S: SwapStore> {
    arena: A,
    swap: S,
    blob_bytes: usize,
    map: HashMap<u64, Loc>,
    /// Free arena slot indices (LIFO reuse).
    free_slots: Vec<usize>,
    /// Resident keys, front = coldest (LRU eviction victim). Reserved keys are
    /// NOT in here (pinned).
    lru: VecDeque<u64>,
    /// On-disk keys, front = coldest — the disk-cap eviction victim. Every
    /// `OnDisk` entry is exactly once in here (a bounded two-level LRU:
    /// RAM `lru` above disk `disk_lru`).
    disk_lru: VecDeque<u64>,
    /// Max simultaneous on-disk records (the disk cap / blob_bytes). 0 =
    /// unbounded. When full, the coldest on-disk snapshot is dropped to make
    /// room — a later GET for it misses and the model recomputes (correct
    /// degradation, keeps the swap file bounded).
    max_disk_slots: usize,
    /// Free disk record indices (reused before growing the high-water mark).
    free_disk: Vec<usize>,
    next_disk: usize,
    /// Reusable scratch for a single blob move (spill/fault), sized once.
    scratch: Vec<u8>,
    /// Read-pins: `key → active reader count`. A GET hands the client an arena
    /// offset it then one-sided-RDMA-READs; the peer drops the residency lock
    /// before that read, so a concurrent ALLOC on another connection could pick
    /// the slot as an eviction victim and reuse it mid-read (torn restore). A
    /// pinned key is held OUT of `lru` (like a `Reserved` slot) so
    /// `evict_coldest_to_disk` can never choose it. Ref-counted for concurrent
    /// readers of the same key. Invariant: `key ∈ lru ⟺ Resident AND unpinned`.
    read_pins: HashMap<u64, u32>,
    stats: SwapStats,
}

impl<A: SlotArena, S: SwapStore> Residency<A, S> {
    /// Unbounded disk tier (no cap). Prefer [`Residency::new_capped`] in
    /// production.
    pub fn new(arena: A, swap: S) -> Result<Self> {
        Self::new_capped(arena, swap, 0)
    }

    /// `max_disk_slots` bounds the on-disk record count (0 = unbounded). When
    /// full, spilling evicts the coldest on-disk snapshot (dropped → later GET
    /// misses → recompute), keeping the swap file at ≤ `max_disk_slots` records.
    pub fn new_capped(arena: A, swap: S, max_disk_slots: usize) -> Result<Self> {
        let blob_bytes = arena.slot_bytes();
        if blob_bytes == 0 {
            bail!("Residency: slot_bytes must be > 0");
        }
        if swap.record_bytes() != blob_bytes {
            bail!(
                "Residency: arena slot ({}) and swap record ({}) sizes differ",
                blob_bytes,
                swap.record_bytes()
            );
        }
        let n = arena.num_slots();
        if n == 0 {
            bail!("Residency: arena must have >= 1 slot");
        }
        Ok(Self {
            arena,
            swap,
            blob_bytes,
            map: HashMap::new(),
            free_slots: (0..n).rev().collect(),
            lru: VecDeque::new(),
            disk_lru: VecDeque::new(),
            max_disk_slots,
            free_disk: Vec::new(),
            next_disk: 0,
            scratch: vec![0u8; blob_bytes],
            read_pins: HashMap::new(),
            stats: SwapStats::default(),
        })
    }

    pub fn blob_bytes(&self) -> usize {
        self.blob_bytes
    }
    pub fn stats(&self) -> &SwapStats {
        &self.stats
    }
    pub fn resident_count(&self) -> usize {
        self.lru.len()
    }
    pub fn total_keys(&self) -> usize {
        self.map.len()
    }

    /// Direct arena access for the data plane. The peer writes the slots its
    /// clients one-sided-RDMA into/out of; in-process consumers should prefer
    /// [`Residency::put_blob`] / [`Residency::get_blob`].
    pub fn arena(&self) -> &A {
        &self.arena
    }
    pub fn arena_mut(&mut self) -> &mut A {
        &mut self.arena
    }

    /// Byte offset of an arena slot (what the client RDMA-reads/writes).
    pub fn slot_offset(&self, slot: usize) -> u64 {
        (slot as u64) * (self.blob_bytes as u64)
    }

    // ─────────────────────────── control-plane ops ───────────────────────────

    /// PUT step 1 — reserve an arena slot for `key`. Evicts the coldest resident
    /// slot to disk if the arena is full (never rejects). The caller then
    /// RDMA-WRITEs the blob into `slot_offset(slot)` and calls `commit(key)`.
    /// Re-PUT of a live key reuses its current slot (idempotent overwrite).
    pub fn alloc(&mut self, key: u64) -> Result<usize> {
        self.stats.puts += 1;
        // Overwrite-in-place: a key already resident/reserved keeps its slot.
        match self.map.get(&key).copied() {
            Some(Loc::Resident(slot)) => {
                self.lru_remove(key); // pin during the rewrite
                self.map.insert(key, Loc::Reserved(slot));
                return Ok(slot);
            }
            Some(Loc::Reserved(slot)) => return Ok(slot),
            Some(Loc::OnDisk(disk_slot)) => {
                // Rewriting a spilled key: reclaim its disk record, give a slot.
                self.disk_lru_remove(key);
                self.free_disk.push(disk_slot);
                self.swap.discard_record(disk_slot);
            }
            None => {}
        }
        let slot = self.acquire_slot()?;
        self.map.insert(key, Loc::Reserved(slot));
        Ok(slot)
    }

    /// PUT step 2 — the client's RDMA-WRITE into the reserved slot has landed;
    /// mark `key` resident (and hottest in the LRU).
    pub fn commit(&mut self, key: u64) -> Result<()> {
        match self.map.get(&key).copied() {
            Some(Loc::Reserved(slot)) => {
                self.map.insert(key, Loc::Resident(slot));
                // Maintain the `in-lru ⟺ Resident AND unpinned` invariant: if a
                // reader pinned this key while the re-PUT was in flight, leave it
                // out of the LRU — `unpin_read` re-adds it when the last reader
                // releases.
                if !self.read_pins.contains_key(&key) {
                    self.lru.push_back(key); // hottest
                }
                Ok(())
            }
            Some(Loc::Resident(_)) => Ok(()), // already committed — idempotent
            _ => bail!("commit({key:#x}): no reserved slot (alloc not called / evicted)"),
        }
    }

    /// GET — ensure `key` is resident and return its arena slot (offset via
    /// `slot_offset`). Faults from disk into a slot if it was spilled (evicting
    /// a victim to make room). `Ok(None)` = unknown key (caller recomputes).
    pub fn locate(&mut self, key: u64) -> Result<Option<usize>> {
        self.stats.gets += 1;
        match self.map.get(&key).copied() {
            Some(Loc::Resident(slot)) => {
                self.stats.resident_hits += 1;
                // A concurrently-pinned key is held out of `lru`; touching it
                // would re-insert it (breaking the invariant + making it an
                // eviction victim while still being read). The caller pins right
                // after this returns, so unpinned hits get refreshed here and
                // pinned ones stay out.
                if !self.read_pins.contains_key(&key) {
                    self.lru_touch(key);
                }
                Ok(Some(slot))
            }
            Some(Loc::Reserved(slot)) => {
                // A GET racing an uncommitted PUT: the bytes are (being) written
                // by the same caller; hand back the slot.
                Ok(Some(slot))
            }
            Some(Loc::OnDisk(disk_slot)) => {
                // Pin against `acquire_slot`'s spill+make_disk_room evicting THIS
                // key (it is still OnDisk until the fault below completes).
                self.disk_lru_remove(key);
                let slot = match self.acquire_slot() {
                    Ok(s) => s,
                    Err(e) => {
                        self.disk_lru.push_front(key); // un-pin (still on disk)
                        return Err(e);
                    }
                };
                // scratch is exclusive to one move at a time (control loop is
                // single-threaded per connection); read disk → arena slot.
                let mut buf = std::mem::take(&mut self.scratch);
                let r = self.swap.read_record(disk_slot, &mut buf);
                if r.is_ok() {
                    r.and_then(|_| self.arena.write_slot(slot, &buf))?;
                } else {
                    self.scratch = buf;
                    self.free_slots.push(slot);
                    self.disk_lru.push_front(key); // still on disk; re-pin (cold)
                    return r.map(|_| None);
                }
                self.scratch = buf;
                self.free_disk.push(disk_slot);
                self.swap.discard_record(disk_slot);
                self.map.insert(key, Loc::Resident(slot));
                self.lru.push_back(key);
                self.stats.faults_from_disk += 1;
                Ok(Some(slot))
            }
            None => {
                self.stats.get_miss += 1;
                Ok(None)
            }
        }
    }

    /// Drop `key` entirely, reclaiming its arena slot or disk record.
    pub fn remove(&mut self, key: u64) {
        match self.map.remove(&key) {
            Some(Loc::Resident(slot)) | Some(Loc::Reserved(slot)) => {
                self.lru_remove(key);
                self.free_slots.push(slot);
            }
            Some(Loc::OnDisk(disk_slot)) => {
                self.disk_lru_remove(key);
                self.free_disk.push(disk_slot);
                self.swap.discard_record(disk_slot);
            }
            None => {}
        }
    }

    /// Read-pin `key` so its resident slot cannot be chosen as an eviction
    /// victim while a client's one-sided RDMA READ of it is in flight (the
    /// GET→RDMA-read race: the peer replies with the offset and drops the lock
    /// before the client reads, so a concurrent ALLOC could otherwise
    /// spill+reuse the slot). Ref-counted for concurrent readers; the first pin
    /// removes the key from `lru` (like a `Reserved` slot). No-op unless the key
    /// is currently `Resident`.
    pub fn pin_read(&mut self, key: u64) {
        if !matches!(self.map.get(&key), Some(Loc::Resident(_))) {
            return;
        }
        let n = self.read_pins.get(&key).copied().unwrap_or(0);
        if n == 0 {
            self.lru_remove(key); // exclude from eviction victims while read
        }
        self.read_pins.insert(key, n + 1);
    }

    /// Release one read-pin taken by [`Residency::pin_read`]. When the last
    /// reader releases, the key rejoins `lru` as hottest (it was just read).
    /// No-op if the key holds no pin. Robust to the key having been removed
    /// while pinned: only re-adds to `lru` if still `Resident` and not already
    /// present.
    pub fn unpin_read(&mut self, key: u64) {
        let Some(n) = self.read_pins.get_mut(&key) else {
            return;
        };
        *n -= 1;
        if *n == 0 {
            self.read_pins.remove(&key);
            if matches!(self.map.get(&key), Some(Loc::Resident(_))) && !self.lru.contains(&key) {
                self.lru.push_back(key); // hottest — just accessed
            }
        }
    }

    /// Active read-pin count (test/introspection).
    pub fn read_pin_count(&self, key: u64) -> u32 {
        self.read_pins.get(&key).copied().unwrap_or(0)
    }

    // ───────────────────── in-process one-shot helpers ─────────────────────

    /// One-shot in-process PUT: reserve a slot, copy `bytes` into it, commit.
    /// Same NEVER-reject chain as the two-phase peer path (alloc → spill the
    /// coldest resident → drop the coldest on-disk key when capped). For
    /// consumers whose data plane is a memcpy rather than a client RDMA-WRITE.
    pub fn put_blob(&mut self, key: u64, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.blob_bytes {
            bail!("put_blob({key:#x}): {} bytes, expected {}", bytes.len(), self.blob_bytes);
        }
        let slot = self.alloc(key)?;
        if let Err(e) = self.arena.write_slot(slot, bytes) {
            // Roll back the reservation so the slot is not stranded Reserved
            // (a later GET must miss cleanly, never read a torn slot).
            self.remove(key);
            return Err(e);
        }
        self.commit(key)
    }

    /// One-shot in-process GET: fault `key` in (if spilled) and copy its blob
    /// into `out`. `Ok(false)` = unknown key (caller recomputes).
    pub fn get_blob(&mut self, key: u64, out: &mut [u8]) -> Result<bool> {
        if out.len() != self.blob_bytes {
            bail!("get_blob({key:#x}): {} bytes, expected {}", out.len(), self.blob_bytes);
        }
        match self.locate(key)? {
            Some(slot) => {
                self.arena.read_slot(slot, out)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    // ─────────────────────────── internals ───────────────────────────

    /// A free arena slot, spilling the coldest resident slot to disk if none.
    fn acquire_slot(&mut self) -> Result<usize> {
        if let Some(s) = self.free_slots.pop() {
            return Ok(s);
        }
        self.evict_coldest_to_disk()
    }

    /// Spill the LRU-coldest RESIDENT key to a disk record and return its freed
    /// arena slot. Reserved (pinned) keys are never victims.
    fn evict_coldest_to_disk(&mut self) -> Result<usize> {
        let Some(victim) = self.lru.pop_front() else {
            bail!(
                "Residency: arena exhausted — all {} slots reserved (uncommitted \
                 PUTs) or read-pinned (in-flight RDMA READs)",
                self.arena.num_slots()
            );
        };
        let slot = match self.map.get(&victim).copied() {
            Some(Loc::Resident(slot)) => slot,
            other => bail!("LRU/map desync: victim {victim:#x} is {other:?}, expected Resident"),
        };
        // Bound the disk tier: drop the coldest on-disk snapshot(s) if at cap
        // BEFORE claiming a disk slot for this spill.
        self.make_disk_room();
        let disk_slot = self.alloc_disk_slot();
        let mut buf = std::mem::take(&mut self.scratch);
        let res = self
            .arena
            .read_slot(slot, &mut buf)
            .and_then(|_| self.swap.write_record(disk_slot, &buf));
        self.scratch = buf;
        if let Err(e) = res {
            // Roll back: victim stays resident, disk slot returns to the pool.
            self.free_disk.push(disk_slot);
            self.lru.push_front(victim);
            return Err(e);
        }
        self.map.insert(victim, Loc::OnDisk(disk_slot));
        self.disk_lru.push_back(victim); // warmest on-disk entry
        self.stats.spills_to_disk += 1;
        Ok(slot)
    }

    /// Evict the coldest on-disk snapshot(s) until there is room for one more
    /// under `max_disk_slots` (no-op when unbounded). A dropped snapshot's key
    /// leaves the map entirely → a later GET misses → the model recomputes.
    fn make_disk_room(&mut self) {
        if self.max_disk_slots == 0 {
            return;
        }
        while self.disk_lru.len() >= self.max_disk_slots {
            let Some(cold) = self.disk_lru.pop_front() else {
                break;
            };
            if let Some(Loc::OnDisk(ds)) = self.map.remove(&cold) {
                self.free_disk.push(ds);
                self.swap.discard_record(ds);
                self.stats.disk_evictions += 1;
            }
        }
    }

    fn alloc_disk_slot(&mut self) -> usize {
        if let Some(d) = self.free_disk.pop() {
            d
        } else {
            let d = self.next_disk;
            self.next_disk += 1;
            d
        }
    }

    fn disk_lru_remove(&mut self, key: u64) {
        if let Some(pos) = self.disk_lru.iter().position(|&k| k == key) {
            self.disk_lru.remove(pos);
        }
    }

    fn lru_touch(&mut self, key: u64) {
        self.lru_remove(key);
        self.lru.push_back(key);
    }

    fn lru_remove(&mut self, key: u64) {
        if let Some(pos) = self.lru.iter().position(|&k| k == key) {
            self.lru.remove(pos);
        }
    }
}

// ───────────────────────── host-RAM reference impls ─────────────────────────

/// Host-RAM [`SlotArena`] over one flat `Vec<u8>` (promoted from the original
/// test fake). The hot tier for in-process consumers — e.g. the unified SSM
/// spill store's RAM cache. Allocates `slot_bytes * num_slots` up front.
pub struct VecSlotArena {
    buf: Vec<u8>,
    slot_bytes: usize,
    n: usize,
}

impl VecSlotArena {
    pub fn new(slot_bytes: usize, num_slots: usize) -> Self {
        Self { buf: vec![0u8; slot_bytes * num_slots], slot_bytes, n: num_slots }
    }
}

impl SlotArena for VecSlotArena {
    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    fn num_slots(&self) -> usize {
        self.n
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        if slot >= self.n || out.len() != self.slot_bytes {
            bail!("VecSlotArena::read_slot({slot}) out of range / size mismatch");
        }
        let o = slot * self.slot_bytes;
        out.copy_from_slice(&self.buf[o..o + self.slot_bytes]);
        Ok(())
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        if slot >= self.n || bytes.len() != self.slot_bytes {
            bail!("VecSlotArena::write_slot({slot}) out of range / size mismatch");
        }
        let o = slot * self.slot_bytes;
        self.buf[o..o + self.slot_bytes].copy_from_slice(bytes);
        Ok(())
    }
}

/// Host-RAM [`SwapStore`] over a `HashMap` (promoted from the original test
/// fake). Records live in ordinary heap memory — the "swap" tier when no NVMe
/// directory is configured (unbounded, still LRU-ordered by the residency).
pub struct MemSwapStore {
    recs: HashMap<usize, Vec<u8>>,
    record_bytes: usize,
}

impl MemSwapStore {
    pub fn new(record_bytes: usize) -> Self {
        Self { recs: HashMap::new(), record_bytes }
    }
}

impl SwapStore for MemSwapStore {
    fn record_bytes(&self) -> usize {
        self.record_bytes
    }
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.record_bytes {
            bail!("MemSwapStore::write_record: {} bytes, expected {}", bytes.len(), self.record_bytes);
        }
        self.recs.insert(disk_slot, bytes.to_vec());
        Ok(())
    }
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        match self.recs.get(&disk_slot) {
            Some(v) => {
                out.copy_from_slice(v);
                Ok(())
            }
            None => bail!("MemSwapStore: no record {disk_slot}"),
        }
    }
    fn discard_record(&mut self, disk_slot: usize) {
        self.recs.remove(&disk_slot);
    }
}

// ───────────────────────────── real disk store ─────────────────────────────

use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// O_DIRECT fixed-stride swap file on NVMe (the peer's cold tier). `record_bytes`
/// MUST be a 4 KiB multiple (O_DIRECT) — the SSM snapshot blob (66,846,720 B =
/// 16,320 × 4 KiB) already is. Records are addressed by `disk_slot` at
/// `disk_slot * record_bytes`; the file grows sparsely as slots are allocated.
///
/// Buffers passed to read/write must be page-aligned for O_DIRECT. The peer's
/// callers pass the mmap'd arena scratch (page-aligned); the residency scratch
/// is a plain Vec — see `read/write_record` which stage through an aligned
/// bounce only when the caller's buffer isn't aligned.
pub struct DirectSwapFile {
    fd: OwnedFd,
    record_bytes: usize,
    /// Page-aligned bounce for callers whose buffer isn't O_DIRECT-aligned.
    bounce: AlignedBuf,
}

impl DirectSwapFile {
    pub fn create(path: &Path, record_bytes: usize) -> Result<Self> {
        if record_bytes == 0 || !record_bytes.is_multiple_of(4096) {
            bail!("DirectSwapFile: record_bytes ({record_bytes}) must be a non-zero 4 KiB multiple");
        }
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(|e| anyhow::anyhow!("open O_DIRECT {}: {e}", path.display()))?;
        Ok(Self {
            fd: OwnedFd::from(f),
            record_bytes,
            bounce: AlignedBuf::new(record_bytes),
        })
    }

    fn offset(&self, disk_slot: usize) -> libc::off_t {
        (disk_slot as u64 * self.record_bytes as u64) as libc::off_t
    }
}

impl SwapStore for DirectSwapFile {
    fn record_bytes(&self) -> usize {
        self.record_bytes
    }

    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.record_bytes {
            bail!("write_record: {} bytes, expected {}", bytes.len(), self.record_bytes);
        }
        let off = self.offset(disk_slot);
        let src = if is_aligned(bytes.as_ptr()) {
            bytes.as_ptr()
        } else {
            self.bounce.as_mut_slice().copy_from_slice(bytes);
            self.bounce.ptr()
        };
        let n = unsafe {
            libc::pwrite(self.fd.as_raw_fd(), src as *const libc::c_void, self.record_bytes, off)
        };
        if n != self.record_bytes as isize {
            bail!("pwrite record {disk_slot} returned {n}, errno {}", errno());
        }
        Ok(())
    }

    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        if out.len() != self.record_bytes {
            bail!("read_record: {} bytes, expected {}", out.len(), self.record_bytes);
        }
        let off = self.offset(disk_slot);
        if is_aligned(out.as_ptr()) {
            let n = unsafe {
                libc::pread(self.fd.as_raw_fd(), out.as_mut_ptr() as *mut libc::c_void, self.record_bytes, off)
            };
            if n != self.record_bytes as isize {
                bail!("pread record {disk_slot} returned {n}, errno {}", errno());
            }
        } else {
            // Stage through the aligned bounce, then copy out. `&self` — the
            // bounce is interior; take a raw ptr (single-threaded peer loop).
            let bp = self.bounce.ptr();
            let n = unsafe {
                libc::pread(self.fd.as_raw_fd(), bp as *mut libc::c_void, self.record_bytes, off)
            };
            if n != self.record_bytes as isize {
                bail!("pread(bounce) record {disk_slot} returned {n}, errno {}", errno());
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bp, out.as_mut_ptr(), self.record_bytes);
            }
        }
        Ok(())
    }
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

fn is_aligned(p: *const u8) -> bool {
    (p as usize) & 0xfff == 0
}

/// A page-aligned heap buffer (posix_memalign) for O_DIRECT staging.
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for AlignedBuf {}
impl AlignedBuf {
    fn new(len: usize) -> Self {
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut p, 4096, len) };
        assert!(rc == 0 && !p.is_null(), "posix_memalign({len}) failed rc={rc}");
        Self { ptr: p as *mut u8, len }
    }
    fn ptr(&self) -> *mut u8 {
        self.ptr
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { libc::free(self.ptr as *mut libc::c_void) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: usize = 8; // tiny blob for tests

    fn blob(tag: u8) -> Vec<u8> {
        vec![tag; B]
    }

    /// Client-side helper: alloc → write bytes into the arena slot → commit.
    fn put(r: &mut Residency<VecSlotArena, MemSwapStore>, key: u64, tag: u8) {
        let slot = r.alloc(key).unwrap();
        r.arena_mut().write_slot(slot, &blob(tag)).unwrap();
        r.commit(key).unwrap();
    }
    fn get(r: &mut Residency<VecSlotArena, MemSwapStore>, key: u64) -> Option<Vec<u8>> {
        r.locate(key).unwrap().map(|slot| {
            let mut out = vec![0u8; B];
            r.arena().read_slot(slot, &mut out).unwrap();
            out
        })
    }

    fn residency(slots: usize) -> Residency<VecSlotArena, MemSwapStore> {
        Residency::new(VecSlotArena::new(B, slots), MemSwapStore::new(B)).unwrap()
    }

    fn residency_capped(slots: usize, max_disk: usize) -> Residency<VecSlotArena, MemSwapStore> {
        Residency::new_capped(VecSlotArena::new(B, slots), MemSwapStore::new(B), max_disk).unwrap()
    }

    /// Disk cap: the swap tier is BOUNDED — beyond RAM slots + `max_disk`, the
    /// COLDEST on-disk snapshot is dropped (a later GET misses → recompute), so
    /// the swap file never grows past the cap. This is the operator's 50 GB
    /// sanity limit at the paging layer.
    #[test]
    fn disk_cap_bounds_swap_and_drops_coldest() {
        let mut r = residency_capped(2, 3); // 2 RAM + 3 disk = 5 total capacity
        for k in 0..10u64 {
            put(&mut r, k, k as u8);
        }
        assert!(r.stats().disk_evictions >= 5, "coldest disk snaps must be dropped at cap");
        assert!(r.total_keys() <= 2 + 3, "total tracked keys bounded by RAM + disk cap");
        // Coldest keys were dropped → clean miss (checked first: a miss doesn't
        // perturb residency).
        assert_eq!(get(&mut r, 0), None, "oldest key evicted from the capped disk");
        assert_eq!(get(&mut r, 1), None);
        // The hottest keys survive (resident) and are byte-identical.
        assert_eq!(get(&mut r, 9).as_deref(), Some(&blob(9)[..]));
        assert_eq!(get(&mut r, 8).as_deref(), Some(&blob(8)[..]));
    }

    /// THE headline invariant: put far more keys than the arena holds; the
    /// coldest spill to the disk tier and fault back BYTE-IDENTICAL. "Infinite
    /// depth, never dropped" proven at the paging layer.
    #[test]
    fn infinite_depth_spill_and_fault_byte_identical() {
        let mut r = residency(4); // 4 slots
        for k in 0..64u64 {
            put(&mut r, k, k as u8);
        }
        assert!(r.stats().spills_to_disk >= 60, "most keys must have spilled to disk");
        assert_eq!(r.resident_count(), 4, "only 4 slots resident at once");
        assert_eq!(r.total_keys(), 64, "all 64 keys tracked — nothing dropped");
        // Every key faults back to its exact bytes.
        for k in 0..64u64 {
            assert_eq!(get(&mut r, k).as_deref(), Some(&blob(k as u8)[..]), "key {k}");
        }
        assert!(r.stats().faults_from_disk > 0);
    }

    /// THE eviction-pin guarantee (WS-A GET→RDMA-read race): a read-pinned key
    /// is never chosen as an eviction victim, even when it is the LRU-coldest —
    /// a concurrent ALLOC spills the next-coldest UNPINNED key instead, so the
    /// client's in-flight one-sided RDMA READ is never torn by slot reuse.
    #[test]
    fn read_pin_survives_concurrent_eviction() {
        let mut r = residency(2);
        put(&mut r, 0, 0); // resident: [0] (coldest)
        put(&mut r, 1, 1); // resident: [0,1]
        // Client A GETs key 0 (the coldest) and begins its RDMA read → pin it.
        assert!(r.locate(0).unwrap().is_some());
        r.pin_read(0);
        assert_eq!(r.read_pin_count(0), 1);
        let faults_before = r.stats().faults_from_disk;

        // Client B ALLOCs a new key → arena full → must evict. Key 0 is coldest
        // but pinned, so key 1 is spilled instead.
        put(&mut r, 2, 2);
        assert_eq!(r.stats().spills_to_disk, 1, "exactly one eviction");
        assert_eq!(get(&mut r, 1), Some(blob(1)), "the UNPINNED key 1 was the victim");
        // Key 0 is still resident (byte-intact) and never touched disk.
        assert_eq!(get(&mut r, 0), Some(blob(0)), "pinned key 0 survived intact");
        assert_eq!(
            r.stats().faults_from_disk,
            faults_before + 1,
            "only key 1 faulted back; key 0 never spilled"
        );
    }

    /// Ref-counted: concurrent readers each add a pin; the key stays protected
    /// until the LAST reader releases, then rejoins the LRU (no double-insert).
    #[test]
    fn refcounted_read_pins_release_to_evictable() {
        let mut r = residency(2);
        put(&mut r, 0, 0);
        put(&mut r, 1, 1);
        r.pin_read(0);
        r.pin_read(0); // second concurrent reader of key 0
        assert_eq!(r.read_pin_count(0), 2);
        r.unpin_read(0);
        assert_eq!(r.read_pin_count(0), 1, "still one reader → still pinned");
        // Force an eviction: key 0 is still protected → key 1 spills.
        put(&mut r, 2, 2);
        assert_eq!(get(&mut r, 1), Some(blob(1)), "key 1 evicted while key 0 still pinned");
        // Last reader releases → key 0 rejoins the LRU exactly once and is now
        // an eligible victim again.
        r.unpin_read(0);
        assert_eq!(r.read_pin_count(0), 0);
        assert_eq!(r.resident_count(), 2, "keys 0 and 2 resident; no LRU double-insert");
        put(&mut r, 3, 3); // evicts the now-unpinned coldest (key 0)
        put(&mut r, 4, 4);
        assert_eq!(get(&mut r, 0), Some(blob(0)), "unpinned key 0 spilled+faulted byte-identical");
    }

    #[test]
    fn resident_hit_does_not_touch_disk() {
        let mut r = residency(4);
        for k in 0..3u64 {
            put(&mut r, k, k as u8);
        }
        let spills_before = r.stats().spills_to_disk;
        assert_eq!(get(&mut r, 1), Some(blob(1)));
        assert_eq!(r.stats().spills_to_disk, spills_before, "resident hit spills nothing");
        assert!(r.stats().resident_hits >= 1);
    }

    #[test]
    fn lru_evicts_coldest_first() {
        let mut r = residency(2);
        put(&mut r, 10, 10); // resident: [10]
        put(&mut r, 11, 11); // resident: [10,11]
        get(&mut r, 10); // touch 10 → [11,10]; 11 now coldest
        put(&mut r, 12, 12); // arena full → evict coldest (11) to disk
        // 11 must be on disk, 10 & 12 resident.
        assert_eq!(get(&mut r, 11), Some(blob(11))); // faults back correctly
        assert!(r.stats().faults_from_disk >= 1);
    }

    #[test]
    fn overwrite_in_place_reuses_slot_no_leak() {
        let mut r = residency(2);
        put(&mut r, 5, 100);
        put(&mut r, 5, 200); // rewrite same key
        assert_eq!(get(&mut r, 5), Some(blob(200)));
        assert_eq!(r.total_keys(), 1, "no phantom duplicate");
        assert_eq!(r.resident_count(), 1);
    }

    #[test]
    fn overwrite_spilled_key_reclaims_disk() {
        let mut r = residency(1); // force spilling
        put(&mut r, 1, 1);
        put(&mut r, 2, 2); // spills key 1 to disk
        put(&mut r, 1, 99); // rewrite the SPILLED key 1
        assert_eq!(get(&mut r, 1), Some(blob(99)));
        assert_eq!(get(&mut r, 2), Some(blob(2)));
    }

    #[test]
    fn remove_frees_resources() {
        let mut r = residency(2);
        put(&mut r, 1, 1);
        put(&mut r, 2, 2);
        put(&mut r, 3, 3); // 1 spills to disk
        r.remove(1); // remove a spilled key
        r.remove(2); // remove a resident key
        assert_eq!(get(&mut r, 1), None, "removed key is a clean miss");
        assert_eq!(get(&mut r, 2), None);
        assert_eq!(get(&mut r, 3), Some(blob(3)));
        assert_eq!(r.total_keys(), 1);
    }

    #[test]
    fn unknown_key_is_clean_miss() {
        let mut r = residency(2);
        assert_eq!(r.locate(0xdead).unwrap(), None);
        assert_eq!(r.stats().get_miss, 1);
    }

    #[test]
    fn reserved_slot_pinned_during_put() {
        // A slot handed out by alloc must not be evictable before commit.
        let mut r = residency(1);
        let slot = r.alloc(1).unwrap();
        // Second alloc with the only slot reserved-and-uncommitted must error,
        // not silently evict the in-flight PUT.
        let err = r.alloc(2);
        assert!(err.is_err(), "must not evict an uncommitted reserved slot");
        // Finish the first PUT and it all works.
        r.arena_mut().write_slot(slot, &blob(1)).unwrap();
        r.commit(1).unwrap();
        assert_eq!(get(&mut r, 1), Some(blob(1)));
    }

    #[test]
    fn size_mismatch_rejected() {
        let bad = Residency::new(VecSlotArena::new(8, 2), MemSwapStore::new(16));
        assert!(bad.is_err(), "arena/swap size mismatch must be rejected at construction");
    }

    // ───────────── new coverage: one-shot helpers + boxed composition ─────────────

    /// `put_blob`/`get_blob` never reject and round-trip bytes exactly like the
    /// two-phase alloc/commit path (the in-process consumer contract).
    #[test]
    fn put_get_blob_helpers_never_reject_and_roundtrip() {
        let mut r = residency(2);
        for k in 0..32u64 {
            r.put_blob(k, &blob(k as u8)).unwrap();
        }
        assert_eq!(r.total_keys(), 32, "never-reject: every key tracked");
        assert_eq!(r.resident_count(), 2);
        let mut out = vec![0u8; B];
        for k in 0..32u64 {
            assert!(r.get_blob(k, &mut out).unwrap(), "key {k} present");
            assert_eq!(out, blob(k as u8), "key {k} byte-identical");
        }
        assert!(!r.get_blob(999, &mut out).unwrap(), "unknown key is a clean miss");
        // Size mismatches are hard errors, never silent corruption.
        assert!(r.put_blob(1, &vec![0u8; B + 1]).is_err());
        let mut short = vec![0u8; B - 1];
        assert!(r.get_blob(1, &mut short).is_err());
    }

    /// `Residency<Box<dyn SlotArena>, Box<dyn SwapStore>>` composes (runtime
    /// arena/swap selection — what the unified SSM store uses).
    #[test]
    fn boxed_trait_objects_compose() {
        let arena: Box<dyn SlotArena> = Box::new(VecSlotArena::new(B, 2));
        let swap: Box<dyn SwapStore> = Box::new(MemSwapStore::new(B));
        let mut r = Residency::new(arena, swap).unwrap();
        for k in 0..16u64 {
            r.put_blob(k, &blob(k as u8)).unwrap();
        }
        let mut out = vec![0u8; B];
        for k in 0..16u64 {
            assert!(r.get_blob(k, &mut out).unwrap());
            assert_eq!(out, blob(k as u8));
        }
    }

    // ───────────── new coverage: DirectSwapFile (never tested before) ─────────────

    /// A real-filesystem dir for O_DIRECT (tmpfs/overlay EINVALs on O_DIRECT —
    /// tolerated as a skip so containerized CI doesn't break).
    fn o_direct_file(record_bytes: usize, tag: &str) -> Option<(DirectSwapFile, std::path::PathBuf)> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/atlas-tier-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("dsf-{tag}-{}.swap", std::process::id()));
        match DirectSwapFile::create(&path, record_bytes) {
            Ok(f) => Some((f, path)),
            Err(e) => {
                eprintln!("skipping O_DIRECT test (filesystem refused O_DIRECT): {e:#}");
                None
            }
        }
    }

    #[test]
    fn direct_swap_file_rejects_bad_record_bytes() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/atlas-tier-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dsf-bad.swap");
        assert!(DirectSwapFile::create(&path, 0).is_err(), "zero record_bytes rejected");
        assert!(DirectSwapFile::create(&path, 1000).is_err(), "non-4KiB multiple rejected");
    }

    /// O_DIRECT write/read round-trips through the page-aligned bounce (a plain
    /// `Vec` caller buffer is usually unaligned, exercising both bounce paths).
    #[test]
    fn direct_swap_file_roundtrips_records() {
        let rb = 4096usize;
        let Some((mut f, path)) = o_direct_file(rb, "rt") else { return };
        assert_eq!(f.record_bytes(), rb);
        let mut pat = vec![0u8; rb];
        for (i, b) in pat.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        f.write_record(3, &pat).unwrap(); // sparse: slot 3 before slot 0
        f.write_record(0, &vec![0xEE; rb]).unwrap();
        let mut out = vec![0u8; rb];
        f.read_record(3, &mut out).unwrap();
        assert_eq!(out, pat, "record 3 byte-identical");
        f.read_record(0, &mut out).unwrap();
        assert_eq!(out, vec![0xEE; rb], "record 0 byte-identical");
        // Size validation is a hard error, not a short IO.
        assert!(f.write_record(1, &pat[..100]).is_err());
        let mut short = vec![0u8; 100];
        assert!(f.read_record(0, &mut short).is_err());
        let _ = std::fs::remove_file(path);
    }

    /// End-to-end: the residency spills to a REAL O_DIRECT file and faults back
    /// byte-identical (the exact peer configuration, minus RDMA).
    #[test]
    fn residency_over_o_direct_swap_byte_identical() {
        let rb = 4096usize;
        let Some((f, path)) = o_direct_file(rb, "resid") else { return };
        let mut r = Residency::new(VecSlotArena::new(rb, 2), f).unwrap();
        for k in 0..8u64 {
            r.put_blob(k, &vec![k as u8; rb]).unwrap();
        }
        assert_eq!(r.total_keys(), 8);
        assert!(r.stats().spills_to_disk >= 6, "cold keys spilled to the O_DIRECT file");
        let mut out = vec![0u8; rb];
        for k in 0..8u64 {
            assert!(r.get_blob(k, &mut out).unwrap(), "key {k}");
            assert_eq!(out, vec![k as u8; rb], "key {k} byte-identical through O_DIRECT");
        }
        let _ = std::fs::remove_file(path);
    }
}
