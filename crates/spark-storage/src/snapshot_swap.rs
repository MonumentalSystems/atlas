// SPDX-License-Identifier: AGPL-3.0-only
//
// WS-A Inc 1: peer-side paging core for the SSM-snapshot spill tier.
//
// Turns the atlas-cache-peer's fixed RDMA arena into a bounded page-cache over
// an UNBOUNDED lower tier (NVMe swap file) — "infinite depth like the LoRA
// setup" (operator, 2026-07-07). The peer owns the residency map (so all fleet
// clients SHARE one warm cache instead of each owning a colliding client-side
// allocator), and the stable per-rail arena MR is NEVER re-registered — bytes
// swap under the fixed rkey, driven by a TCP control channel (Inc 2). This
// module is the pure paging logic + disk record store; it is CPU/disk-only and
// fully unit-testable without RDMA or a GPU.
//
// Design constraints honored:
//   * No MR churn: the arena is a set of fixed slots at a stable VA; we swap
//     BYTES between an arena slot and a disk record, never remap.
//   * Never reject: a `put` that overflows the RAM arena spills the coldest
//     resident slot to disk instead of returning "full" (kills the bounded-tier
//     drop-on-reject hazard the client side flagged as a follow-up).
//   * Byte-fidelity: spill→fault round-trips the exact blob (unit-tested).
//
// Two seams keep the paging logic testable with in-memory fakes; the real impls
// (peer mmap `SlotArena`, O_DIRECT `DirectSwapFile`) plug in at Inc 2 wiring.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

use anyhow::{Context, Result, bail};

/// The hot tier: the RDMA-registered RAM arena as a set of `num_slots`
/// fixed-size slots. The peer implements this over its `mmap`'d MR (page-aligned
/// → O_DIRECT-safe); tests implement it over a `Vec<u8>`.
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
/// file (`DirectSwapFile`); tests implement it over a `HashMap`.
pub trait SwapStore: Send {
    fn record_bytes(&self) -> usize;
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()>;
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()>;
    /// Optional: reclaim disk space for a freed slot (default no-op; a hole in
    /// a preallocated file is fine — the free-list reuses the index).
    fn discard_record(&mut self, _disk_slot: usize) {}
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
}

/// The page table: `key → Loc` over a bounded `SlotArena` (hot) backed by an
/// unbounded `SwapStore` (cold), with LRU eviction of resident slots to disk.
pub struct SnapshotResidency<A: SlotArena, S: SwapStore> {
    arena: A,
    swap: S,
    blob_bytes: usize,
    map: HashMap<u64, Loc>,
    /// Free arena slot indices (LIFO reuse).
    free_slots: Vec<usize>,
    /// Resident keys, front = coldest (LRU eviction victim). Reserved keys are
    /// NOT in here (pinned).
    lru: VecDeque<u64>,
    /// Free disk record indices (reused before growing the high-water mark).
    free_disk: Vec<usize>,
    next_disk: usize,
    /// Reusable scratch for a single blob move (spill/fault), sized once.
    scratch: Vec<u8>,
    stats: SwapStats,
}

impl<A: SlotArena, S: SwapStore> SnapshotResidency<A, S> {
    pub fn new(arena: A, swap: S) -> Result<Self> {
        let blob_bytes = arena.slot_bytes();
        if blob_bytes == 0 {
            bail!("SnapshotResidency: slot_bytes must be > 0");
        }
        if swap.record_bytes() != blob_bytes {
            bail!(
                "SnapshotResidency: arena slot ({}) and swap record ({}) sizes differ",
                blob_bytes,
                swap.record_bytes()
            );
        }
        let n = arena.num_slots();
        if n == 0 {
            bail!("SnapshotResidency: arena must have >= 1 slot");
        }
        Ok(Self {
            arena,
            swap,
            blob_bytes,
            map: HashMap::new(),
            free_slots: (0..n).rev().collect(),
            lru: VecDeque::new(),
            free_disk: Vec::new(),
            next_disk: 0,
            scratch: vec![0u8; blob_bytes],
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
                self.lru.push_back(key); // hottest
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
                self.lru_touch(key);
                Ok(Some(slot))
            }
            Some(Loc::Reserved(slot)) => {
                // A GET racing an uncommitted PUT: the bytes are (being) written
                // by the same caller; hand back the slot.
                Ok(Some(slot))
            }
            Some(Loc::OnDisk(disk_slot)) => {
                let slot = self.acquire_slot()?;
                // scratch is exclusive to one move at a time (control loop is
                // single-threaded per connection); read disk → arena slot.
                let mut buf = std::mem::take(&mut self.scratch);
                let r = self.swap.read_record(disk_slot, &mut buf);
                if r.is_ok() {
                    r.and_then(|_| self.arena.write_slot(slot, &buf))?;
                } else {
                    self.scratch = buf;
                    self.free_slots.push(slot);
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
                self.free_disk.push(disk_slot);
                self.swap.discard_record(disk_slot);
            }
            None => {}
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
                "SnapshotResidency: arena exhausted — all {} slots reserved (uncommitted PUTs)",
                self.arena.num_slots()
            );
        };
        let slot = match self.map.get(&victim).copied() {
            Some(Loc::Resident(slot)) => slot,
            other => bail!("LRU/map desync: victim {victim:#x} is {other:?}, expected Resident"),
        };
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
        self.stats.spills_to_disk += 1;
        Ok(slot)
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

// ───────────────────────────── control protocol ─────────────────────────────
//
// Inc 2: the TCP control channel between a paging client and the peer. It rides
// the SAME stream the peer used for the RDMA handshake (which today just idles).
// Backward-compatible: a paging client sends `PAGING_MAGIC` as its first u64
// where a legacy KV client sends `total_bytes` (validated <= 1<<42); the magic
// is far above that range, so legacy clients are never mis-parsed.
//
// After the shared rail handshake, the loop is: client sends [op][key], peer
// replies [status] (+ [offset] for ALLOC/GET-hit). Data still moves one-sided
// over RDMA into/out of `slot_offset(slot)`; only tiny control messages cross
// TCP.

use std::io::{Read, Write};

/// First-u64 sentinel selecting the paging protocol (> 1<<42, so a legacy
/// `total_bytes` can never collide). "PAGE" + version.
pub const PAGING_MAGIC: u64 = 0x5041_4745_0000_0001;

pub const OP_BYE: u8 = 0;
pub const OP_ALLOC: u8 = 1;
pub const OP_COMMIT: u8 = 2;
pub const OP_GET: u8 = 3;
pub const OP_REMOVE: u8 = 4;

pub const ST_OK: u8 = 0;
pub const ST_MISS: u8 = 1;
pub const ST_ERR: u8 = 2;

/// Result of one control request, ready to serialize.
#[derive(Debug, PartialEq, Eq)]
pub enum PagingReply {
    /// ST_OK with a following u64 arena offset (ALLOC and GET-hit).
    Located(u64),
    /// ST_OK with no payload (COMMIT, REMOVE).
    Ok,
    /// ST_MISS — unknown key (GET).
    Miss,
    /// ST_ERR — operation failed (e.g. arena exhausted by reservations).
    Err,
    /// Client asked to close.
    Bye,
}

/// Execute one control op against the residency and return the reply. Pure over
/// the (already unit-tested) `SnapshotResidency`, so the protocol is testable
/// without a socket or RDMA.
pub fn dispatch<A: SlotArena, S: SwapStore>(
    res: &mut SnapshotResidency<A, S>,
    op: u8,
    key: u64,
) -> PagingReply {
    match op {
        OP_BYE => PagingReply::Bye,
        OP_ALLOC => match res.alloc(key) {
            Ok(slot) => PagingReply::Located(res.slot_offset(slot)),
            Err(e) => {
                tracing::warn!("paging ALLOC {key:#x} failed: {e:#}");
                PagingReply::Err
            }
        },
        OP_COMMIT => match res.commit(key) {
            Ok(()) => PagingReply::Ok,
            Err(e) => {
                tracing::warn!("paging COMMIT {key:#x} failed: {e:#}");
                PagingReply::Err
            }
        },
        OP_GET => match res.locate(key) {
            Ok(Some(slot)) => PagingReply::Located(res.slot_offset(slot)),
            Ok(None) => PagingReply::Miss,
            Err(e) => {
                tracing::warn!("paging GET {key:#x} failed: {e:#}");
                PagingReply::Err
            }
        },
        OP_REMOVE => {
            res.remove(key);
            PagingReply::Ok
        }
        other => {
            tracing::warn!("paging: unknown op {other}");
            PagingReply::Err
        }
    }
}

fn write_reply<W: Write>(w: &mut W, reply: &PagingReply) -> Result<()> {
    match reply {
        PagingReply::Located(off) => {
            w.write_all(&[ST_OK])?;
            w.write_all(&off.to_le_bytes())?;
        }
        PagingReply::Ok => w.write_all(&[ST_OK])?,
        PagingReply::Miss => w.write_all(&[ST_MISS])?,
        PagingReply::Err => w.write_all(&[ST_ERR])?,
        PagingReply::Bye => {}
    }
    w.flush()?;
    Ok(())
}

/// The peer-side control loop: read `[op][u64 key]` requests, dispatch against
/// `res`, write replies, until BYE or hangup. Generic over the stream so it runs
/// against a real `TcpStream` in the peer and a fake duplex in tests.
pub fn run_paging_loop<T: Read + Write, A: SlotArena, S: SwapStore>(
    stream: &mut T,
    res: &mut SnapshotResidency<A, S>,
) -> Result<()> {
    loop {
        let mut op = [0u8; 1];
        if stream.read_exact(&mut op).is_err() {
            break; // client hung up
        }
        if op[0] == OP_BYE {
            break;
        }
        let mut kb = [0u8; 8];
        stream.read_exact(&mut kb).context("read paging key")?;
        let key = u64::from_le_bytes(kb);
        let reply = dispatch(res, op[0], key);
        write_reply(stream, &reply)?;
    }
    Ok(())
}

// ─────────────────────── client-side protocol helpers ───────────────────────
//
// The CLIENT half of the control channel, sharing the wire format above so peer
// and client can never drift. Each sends `[op][u64 key]` and reads the reply.
// The RDMA data-plane WRITE/READ (client-side) happens between `client_alloc`
// and `client_commit` (PUT) or after `client_get` (GET) — see RdmaSnapshotArena.

fn send_req<T: Write>(s: &mut T, op: u8, key: u64) -> Result<()> {
    let mut buf = [0u8; 9];
    buf[0] = op;
    buf[1..].copy_from_slice(&key.to_le_bytes());
    s.write_all(&buf)?;
    s.flush()?;
    Ok(())
}

fn read_status<T: Read>(s: &mut T) -> Result<u8> {
    let mut st = [0u8; 1];
    s.read_exact(&mut st).context("read paging status")?;
    Ok(st[0])
}

fn read_offset<T: Read>(s: &mut T) -> Result<u64> {
    let mut b = [0u8; 8];
    s.read_exact(&mut b).context("read paging offset")?;
    Ok(u64::from_le_bytes(b))
}

/// PUT step 1: reserve a slot for `key`; returns the arena offset to RDMA-WRITE.
pub fn client_alloc<T: Read + Write>(s: &mut T, key: u64) -> Result<u64> {
    send_req(s, OP_ALLOC, key)?;
    match read_status(s)? {
        ST_OK => read_offset(s),
        st => bail!("paging ALLOC {key:#x} refused (status {st})"),
    }
}

/// PUT step 2: the RDMA-WRITE has drained; mark `key` resident.
pub fn client_commit<T: Read + Write>(s: &mut T, key: u64) -> Result<()> {
    send_req(s, OP_COMMIT, key)?;
    match read_status(s)? {
        ST_OK => Ok(()),
        st => bail!("paging COMMIT {key:#x} failed (status {st})"),
    }
}

/// GET: `Some(offset)` to RDMA-READ, or `None` if the peer has no such key.
pub fn client_get<T: Read + Write>(s: &mut T, key: u64) -> Result<Option<u64>> {
    send_req(s, OP_GET, key)?;
    match read_status(s)? {
        ST_OK => Ok(Some(read_offset(s)?)),
        ST_MISS => Ok(None),
        st => bail!("paging GET {key:#x} error (status {st})"),
    }
}

/// Drop `key` from the peer cache.
pub fn client_remove<T: Read + Write>(s: &mut T, key: u64) -> Result<()> {
    send_req(s, OP_REMOVE, key)?;
    match read_status(s)? {
        ST_OK => Ok(()),
        st => bail!("paging REMOVE {key:#x} failed (status {st})"),
    }
}

/// Politely tell the peer to close the paging loop.
pub fn client_bye<T: Write>(s: &mut T) -> Result<()> {
    send_req(s, OP_BYE, 0)
}

// ─────────────────────────── real peer-mmap arena ───────────────────────────

/// `SlotArena` over the peer's RDMA-registered `mmap` region (a raw base ptr).
/// The peer memcpys between an arena slot and the disk swap on spill/fault; the
/// client one-sided-RDMAs into/out of the same slots. The base VA is stable and
/// registered ONCE per rail — this NEVER re-registers (no MR churn).
///
/// SAFETY: `base` must point at a live mapping of at least `num_slots *
/// slot_bytes` bytes, page-aligned (mmap guarantees this), outliving the arena.
pub struct MmapSlotArena {
    base: *mut u8,
    slot_bytes: usize,
    num_slots: usize,
}
unsafe impl Send for MmapSlotArena {}

impl MmapSlotArena {
    /// # Safety
    /// `base` must be a valid, writable mapping of `>= num_slots*slot_bytes`
    /// bytes that outlives this arena.
    pub unsafe fn new(base: *mut u8, slot_bytes: usize, num_slots: usize) -> Self {
        Self { base, slot_bytes, num_slots }
    }
    fn slot_ptr(&self, slot: usize) -> *mut u8 {
        // slot < num_slots enforced by callers (residency free-list).
        unsafe { self.base.add(slot * self.slot_bytes) }
    }
}

impl SlotArena for MmapSlotArena {
    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    fn num_slots(&self) -> usize {
        self.num_slots
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        if slot >= self.num_slots || out.len() != self.slot_bytes {
            bail!("read_slot({slot}) out of range / size mismatch");
        }
        unsafe { std::ptr::copy_nonoverlapping(self.slot_ptr(slot), out.as_mut_ptr(), self.slot_bytes) };
        Ok(())
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        if slot >= self.num_slots || bytes.len() != self.slot_bytes {
            bail!("write_slot({slot}) out of range / size mismatch");
        }
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.slot_ptr(slot), self.slot_bytes) };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory arena over a flat Vec — the peer's mmap MR stand-in.
    struct VecArena {
        buf: Vec<u8>,
        slot_bytes: usize,
        n: usize,
    }
    impl VecArena {
        fn new(slot_bytes: usize, n: usize) -> Self {
            Self { buf: vec![0u8; slot_bytes * n], slot_bytes, n }
        }
    }
    impl SlotArena for VecArena {
        fn slot_bytes(&self) -> usize {
            self.slot_bytes
        }
        fn num_slots(&self) -> usize {
            self.n
        }
        fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
            let o = slot * self.slot_bytes;
            out.copy_from_slice(&self.buf[o..o + self.slot_bytes]);
            Ok(())
        }
        fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
            let o = slot * self.slot_bytes;
            self.buf[o..o + self.slot_bytes].copy_from_slice(bytes);
            Ok(())
        }
    }

    /// In-memory swap over a HashMap — the NVMe file stand-in.
    #[derive(Default)]
    struct MemSwap {
        recs: HashMap<usize, Vec<u8>>,
        record_bytes: usize,
    }
    impl MemSwap {
        fn new(record_bytes: usize) -> Self {
            Self { recs: HashMap::new(), record_bytes }
        }
    }
    impl SwapStore for MemSwap {
        fn record_bytes(&self) -> usize {
            self.record_bytes
        }
        fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
            self.recs.insert(disk_slot, bytes.to_vec());
            Ok(())
        }
        fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
            match self.recs.get(&disk_slot) {
                Some(v) => {
                    out.copy_from_slice(v);
                    Ok(())
                }
                None => bail!("MemSwap: no record {disk_slot}"),
            }
        }
        fn discard_record(&mut self, disk_slot: usize) {
            self.recs.remove(&disk_slot);
        }
    }

    const B: usize = 8; // tiny blob for tests

    fn blob(tag: u8) -> Vec<u8> {
        vec![tag; B]
    }

    /// Client-side helper: alloc → write bytes into the arena slot → commit.
    fn put(r: &mut SnapshotResidency<VecArena, MemSwap>, key: u64, tag: u8) {
        let slot = r.alloc(key).unwrap();
        r.arena.write_slot(slot, &blob(tag)).unwrap();
        r.commit(key).unwrap();
    }
    fn get(r: &mut SnapshotResidency<VecArena, MemSwap>, key: u64) -> Option<Vec<u8>> {
        r.locate(key).unwrap().map(|slot| {
            let mut out = vec![0u8; B];
            r.arena.read_slot(slot, &mut out).unwrap();
            out
        })
    }

    fn residency(slots: usize) -> SnapshotResidency<VecArena, MemSwap> {
        SnapshotResidency::new(VecArena::new(B, slots), MemSwap::new(B)).unwrap()
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
        r.arena.write_slot(slot, &blob(1)).unwrap();
        r.commit(1).unwrap();
        assert_eq!(get(&mut r, 1), Some(blob(1)));
    }

    #[test]
    fn size_mismatch_rejected() {
        let bad = SnapshotResidency::new(VecArena::new(8, 2), MemSwap::new(16));
        assert!(bad.is_err(), "arena/swap size mismatch must be rejected at construction");
    }

    // ─────────────────────────── protocol tests ───────────────────────────

    #[test]
    fn magic_above_legacy_range() {
        // A legacy client's total_bytes is validated <= 1<<42; the paging magic
        // must be strictly above so it can never be mistaken for a size.
        assert!(PAGING_MAGIC > (1u64 << 42));
    }

    /// One PUT (alloc→arena write→commit) then GET, driven through `dispatch`,
    /// with the caller's RDMA-write emulated by writing the returned slot.
    #[test]
    fn dispatch_put_then_get_roundtrips() {
        let mut r = residency(4);
        // ALLOC key 7
        let PagingReply::Located(off) = dispatch(&mut r, OP_ALLOC, 7) else {
            panic!("alloc reply")
        };
        let slot = (off as usize) / B;
        // client RDMA-WRITE emulation
        r.arena.write_slot(slot, &blob(0xAB)).unwrap();
        assert_eq!(dispatch(&mut r, OP_COMMIT, 7), PagingReply::Ok);
        // GET key 7
        let PagingReply::Located(goff) = dispatch(&mut r, OP_GET, 7) else {
            panic!("get reply")
        };
        let mut out = vec![0u8; B];
        r.arena.read_slot((goff as usize) / B, &mut out).unwrap();
        assert_eq!(out, blob(0xAB));
        // unknown key → miss
        assert_eq!(dispatch(&mut r, OP_GET, 999), PagingReply::Miss);
    }

    /// Fake bidirectional stream: scripted input, captured output.
    struct Duplex {
        inp: std::io::Cursor<Vec<u8>>,
        out: Vec<u8>,
    }
    impl Read for Duplex {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            self.inp.read(b)
        }
    }
    impl Write for Duplex {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.out.extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn req(op: u8, key: u64) -> Vec<u8> {
        let mut v = vec![op];
        v.extend_from_slice(&key.to_le_bytes());
        v
    }

    /// Drive the full `run_paging_loop` over a scripted request stream and
    /// assert the reply bytes — the protocol end-to-end (sans RDMA data plane).
    #[test]
    fn run_paging_loop_scripts_ok() {
        let mut r = residency(4);
        // ALLOC 1 → we need its offset before we can COMMIT meaningfully, but
        // run_paging_loop reads all input at once; emulate the client by first
        // allocating via dispatch to learn the slot, writing bytes, THEN scripting
        // commit+get through the loop. (Mirrors real client ordering.)
        let PagingReply::Located(off) = dispatch(&mut r, OP_ALLOC, 1) else {
            panic!()
        };
        r.arena.write_slot((off as usize) / B, &blob(0x5A)).unwrap();

        let mut script = Vec::new();
        script.extend(req(OP_COMMIT, 1));
        script.extend(req(OP_GET, 1));
        script.extend(req(OP_GET, 42)); // miss
        script.extend(req(OP_REMOVE, 1));
        script.extend(req(OP_BYE, 0));
        let mut dx = Duplex { inp: std::io::Cursor::new(script), out: Vec::new() };
        run_paging_loop(&mut dx, &mut r).unwrap();

        // Expected replies: COMMIT→[OK]; GET1→[OK][off]; GET42→[MISS]; REMOVE→[OK].
        let mut exp = Vec::new();
        exp.push(ST_OK); // commit
        exp.push(ST_OK);
        exp.extend_from_slice(&off.to_le_bytes()); // get 1 (same slot/offset)
        exp.push(ST_MISS); // get 42
        exp.push(ST_OK); // remove
        assert_eq!(dx.out, exp);
    }

    /// Client codec: request bytes on the wire + reply decode.
    #[test]
    fn client_codec_alloc_get_miss() {
        // ALLOC → [ST_OK][offset 0x40]
        let mut reply = vec![ST_OK];
        reply.extend_from_slice(&0x40u64.to_le_bytes());
        let mut dx = Duplex { inp: std::io::Cursor::new(reply), out: Vec::new() };
        let off = client_alloc(&mut dx, 0xAB).unwrap();
        assert_eq!(off, 0x40);
        assert_eq!(dx.out, req(OP_ALLOC, 0xAB), "request bytes on the wire");

        // GET miss → [ST_MISS]
        let mut dx = Duplex { inp: std::io::Cursor::new(vec![ST_MISS]), out: Vec::new() };
        assert_eq!(client_get(&mut dx, 7).unwrap(), None);

        // COMMIT ok → [ST_OK]
        let mut dx = Duplex { inp: std::io::Cursor::new(vec![ST_OK]), out: Vec::new() };
        client_commit(&mut dx, 9).unwrap();
        assert_eq!(dx.out, req(OP_COMMIT, 9));
    }

    /// End-to-end loopback: the client codec's request bytes feed the peer
    /// `dispatch`; the peer's reply bytes feed the client codec — the two halves
    /// agree on the wire and a PUT→GET round-trips a blob byte-identical (the
    /// RDMA data plane emulated via direct arena writes at the returned offset).
    #[test]
    fn client_peer_loopback_roundtrip() {
        let mut r = residency(4);
        // Run one client request through the peer and return the client-decoded
        // reply channel (a cursor over the peer's reply bytes).
        fn peer_roundtrip(
            r: &mut SnapshotResidency<VecArena, MemSwap>,
            req_bytes: &[u8],
        ) -> std::io::Cursor<Vec<u8>> {
            let op = req_bytes[0];
            let key = u64::from_le_bytes(req_bytes[1..9].try_into().unwrap());
            let mut reply = Vec::new();
            write_reply(&mut reply, &dispatch(r, op, key)).unwrap();
            std::io::Cursor::new(reply)
        }

        // PUT key 3: ALLOC → emulate RDMA write → COMMIT.
        let mut wire = Vec::new();
        send_req(&mut wire, OP_ALLOC, 3).unwrap();
        let mut rep = peer_roundtrip(&mut r, &wire);
        assert_eq!(read_status(&mut rep).unwrap(), ST_OK);
        let off = read_offset(&mut rep).unwrap();
        r.arena.write_slot((off as usize) / B, &blob(0x77)).unwrap();

        wire.clear();
        send_req(&mut wire, OP_COMMIT, 3).unwrap();
        assert_eq!(read_status(&mut peer_roundtrip(&mut r, &wire)).unwrap(), ST_OK);

        // GET key 3 → read back byte-identical.
        wire.clear();
        send_req(&mut wire, OP_GET, 3).unwrap();
        let mut rep = peer_roundtrip(&mut r, &wire);
        assert_eq!(read_status(&mut rep).unwrap(), ST_OK);
        let goff = read_offset(&mut rep).unwrap();
        let mut out = vec![0u8; B];
        r.arena.read_slot((goff as usize) / B, &mut out).unwrap();
        assert_eq!(out, blob(0x77), "PUT→GET round-trips byte-identical over the protocol");
    }

    /// `MmapSlotArena` over a real page-aligned heap buffer round-trips bytes.
    #[test]
    fn mmap_slot_arena_roundtrips() {
        let slot_bytes = 4096usize;
        let n = 3usize;
        let buf = AlignedBuf::new(slot_bytes * n);
        let mut arena = unsafe { MmapSlotArena::new(buf.ptr, slot_bytes, n) };
        let pat = vec![0x3C_u8; slot_bytes];
        arena.write_slot(1, &pat).unwrap();
        let mut out = vec![0u8; slot_bytes];
        arena.read_slot(1, &mut out).unwrap();
        assert_eq!(out, pat);
        assert!(arena.write_slot(3, &pat).is_err(), "slot out of range rejected");
        drop(arena);
        drop(buf); // buf outlives arena (asserted by drop order)
    }
}
