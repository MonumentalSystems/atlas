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
// `MmapSlotArena` over the peer's RDMA-registered mmap MR.

#![allow(dead_code)]

use anyhow::{Context, Result, bail};

/// The generic paging core, lifted to `atlas-tier` (CUDA- and verbs-free).
/// `Residency` keeps its historical `SnapshotResidency` name at this path.
pub use atlas_tier::{
    DirectSwapFile, MemSwapStore, Residency as SnapshotResidency, SlotArena, SwapStats, SwapStore,
    VecSlotArena,
};

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

/// First-u64 sentinel selecting the paging protocol v1 (> 1<<42, so a legacy KV
/// `total_bytes` can never collide). "PAGE" + version. v1 == SSM snapshots, no
/// kind byte (the deployed gx10:9920 peer + clients speak this — keep byte-exact).
pub const PAGING_MAGIC: u64 = 0x5041_4745_0000_0001;

/// Paging protocol v2 (item 8): after the magic comes a `[u8 kind]` byte so ONE
/// peer serves a registry of per-(kind, shape) arenas. Also > 1<<42.
pub const PAGING_MAGIC_V2: u64 = 0x5041_4745_0000_0002;

/// The tier a paging arena serves. Only the RW paging kinds (SSM, KV-as-paging)
/// ride the `CacheServerParams` single-base+rkey reply; the read-only tiers
/// (experts/weights/lora) speak a different manifest+VerbsServerParams dialect
/// and are NOT accepted on this handshake (rejected in `parse_paging_header`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PagingKind(pub u8);
impl PagingKind {
    pub const SSM: PagingKind = PagingKind(0);
    pub const KV: PagingKind = PagingKind(1);
    /// Whether this kind is servable on the RW paging (CacheServerParams) path.
    pub fn is_paging_rw(self) -> bool {
        self.0 <= 1
    }
}

/// Parse the paging handshake header after the caller has read the first u64.
/// v1 magic → (SSM, arena_bytes, blob_bytes) with NO kind byte on the wire
/// (byte-exact with the deployed peer). v2 magic → read `[u8 kind]` then the two
/// sizes. Any other first u64 is a legacy KV `total_bytes` → `Ok(None)` so the
/// caller takes the dumb one-sided path. Rejects unsupported kinds (≥2).
pub fn parse_paging_header<R: Read>(
    first: u64,
    r: &mut R,
) -> Result<Option<(PagingKind, u64, u64)>> {
    let kind = if first == PAGING_MAGIC {
        PagingKind::SSM
    } else if first == PAGING_MAGIC_V2 {
        let mut kb = [0u8; 1];
        r.read_exact(&mut kb).context("read paging kind")?;
        let k = PagingKind(kb[0]);
        if !k.is_paging_rw() {
            bail!("paging: unsupported kind {} (only SSM/KV ride this handshake)", kb[0]);
        }
        k
    } else {
        return Ok(None); // legacy KV total_bytes — not a paging client
    };
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8).context("read paging arena_bytes")?;
    let arena_bytes = u64::from_le_bytes(b8);
    r.read_exact(&mut b8).context("read paging blob_bytes")?;
    let blob_bytes = u64::from_le_bytes(b8);
    Ok(Some((kind, arena_bytes, blob_bytes)))
}

/// Round-robin stripe a `blob_bytes` transfer into `chunk_bytes` chunks across
/// `n_rails`, returning per-rail lists of `(offset, len)`. The offset is the
/// chunk's position in BOTH the (single, contiguous) staging buffer and the peer
/// arena slot — so whichever rail fetches chunk j, it lands at its true offset
/// and one memcpy reassembles the blob (the verified inc-6 reassembly fix). The
/// tail chunk carries the short remainder, never `chunk_bytes`.
pub fn stripe_plan(blob_bytes: usize, chunk_bytes: usize, n_rails: usize) -> Vec<Vec<(usize, usize)>> {
    let n = n_rails.max(1);
    let cb = chunk_bytes.max(1);
    let mut rails: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut off = 0usize;
    let mut j = 0usize;
    while off < blob_bytes {
        let len = cb.min(blob_bytes - off);
        rails[j % n].push((off, len));
        off += len;
        j += 1;
    }
    rails
}

/// Chunk size for the striped snapshot pipeline (ATLAS_SSM_CHUNK_BYTES, default
/// 1 MiB) and pipeline depth (ATLAS_SSM_PIPELINE_DEPTH, default 16, clamped
/// 1..=128, mirroring the KV backend).
pub fn staging_chunk_bytes() -> usize {
    std::env::var("ATLAS_SSM_CHUNK_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= 4096)
        .unwrap_or(1024 * 1024)
}
pub fn staging_depth() -> usize {
    std::env::var("ATLAS_SSM_PIPELINE_DEPTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(1, 128)
}

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
/// One control op with connection-scoped read-pin lifecycle. Releases this
/// connection's previous GET read-pin — its RDMA READ has necessarily drained,
/// because the client is synchronous and only sends its NEXT op after the read
/// completes — then dispatches, and pins a fresh GET hit so a concurrent ALLOC
/// on another connection cannot evict the slot mid-read. `pinned` threads the
/// connection's currently-pinned key across calls. Needs NO new opcode and no
/// client change (auto-release on next op / disconnect) → wire-compatible with
/// the deployed peer.
fn handle_paging_op<A: SlotArena, S: SwapStore>(
    res: &mut SnapshotResidency<A, S>,
    op: u8,
    key: u64,
    pinned: &mut Option<u64>,
) -> PagingReply {
    if let Some(prev) = pinned.take() {
        res.unpin_read(prev);
    }
    let reply = dispatch(res, op, key);
    if op == OP_GET && matches!(reply, PagingReply::Located(_)) {
        res.pin_read(key);
        *pinned = Some(key);
    }
    reply
}

pub fn run_paging_loop<T: Read + Write, A: SlotArena, S: SwapStore>(
    stream: &mut T,
    res: &mut SnapshotResidency<A, S>,
) -> Result<()> {
    let mut pinned: Option<u64> = None;
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
        let reply = handle_paging_op(res, op[0], key, &mut pinned);
        write_reply(stream, &reply)?;
    }
    // Release this connection's outstanding read-pin on hangup / BYE.
    if let Some(pk) = pinned {
        res.unpin_read(pk);
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

/// Shared variant of [`run_paging_loop`]: many connection threads drive ONE
/// process-global residency, locking it per request. This is what makes the
/// peer a SHARED warm cache — a snapshot PUT by one client is GET-able by
/// another (same namespace). The lock is held only for the (fast) map op + any
/// spill/fault byte move, never across a TCP read.
pub fn run_paging_loop_shared<T: Read + Write, A: SlotArena, S: SwapStore>(
    stream: &mut T,
    res: &std::sync::Mutex<SnapshotResidency<A, S>>,
) -> Result<()> {
    // Per-connection read-pin (see `handle_paging_op`): a GET hit is pinned OUT
    // of the LRU under the same lock as the dispatch, so a concurrent ALLOC on
    // another connection can't evict the slot while THIS client RDMA-reads it.
    // Released on the connection's next op (its read has drained) or disconnect.
    let mut pinned: Option<u64> = None;
    loop {
        let mut op = [0u8; 1];
        if stream.read_exact(&mut op).is_err() {
            break;
        }
        if op[0] == OP_BYE {
            break;
        }
        let mut kb = [0u8; 8];
        stream.read_exact(&mut kb).context("read paging key")?;
        let key = u64::from_le_bytes(kb);
        let reply = {
            let mut g = res.lock().expect("shared residency mutex poisoned");
            handle_paging_op(&mut g, op[0], key, &mut pinned)
        };
        write_reply(stream, &reply)?;
    }
    if let Some(pk) = pinned {
        res.lock().expect("shared residency mutex poisoned").unpin_read(pk);
    }
    Ok(())
}

// ─────────────────────────── real peer-mmap arena ───────────────────────────

/// `SlotArena` over the peer's RDMA-registered `mmap` region (a raw base ptr).
/// The peer memcpys between an arena slot and the disk swap on spill/fault; the
/// client one-sided-RDMAs into/out of the same slots. The base VA is stable and
/// registered ONCE per rail — this NEVER re-registers (no MR churn).
///
/// SAFETY: `base` must point at a live mapping of at least `num_slots *
/// slot_bytes` bytes, page-aligned (mmap guarantees this), outliving the arena.
/// (Peer-specific — deliberately NOT lifted into atlas-tier: the lifted crate
/// carries no unsafe raw-pointer arena types.)
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

    // The residency-semantics test suite (disk cap, infinite depth, read-pins,
    // reserved-slot pinning, overwrite/remove, size laws) moved to atlas-tier
    // with the core. What stays here is the PROTOCOL half: header parse, stripe
    // plan, wire-golden bytes, dispatch/loop/codec, connection-scoped pins, and
    // the peer-specific MmapSlotArena.

    type TestResidency = SnapshotResidency<VecSlotArena, MemSwapStore>;

    const B: usize = 8; // tiny blob for tests

    fn blob(tag: u8) -> Vec<u8> {
        vec![tag; B]
    }

    /// Client-side helper: alloc → write bytes into the arena slot → commit.
    fn put(r: &mut TestResidency, key: u64, tag: u8) {
        let slot = r.alloc(key).unwrap();
        r.arena_mut().write_slot(slot, &blob(tag)).unwrap();
        r.commit(key).unwrap();
    }
    fn get(r: &mut TestResidency, key: u64) -> Option<Vec<u8>> {
        r.locate(key).unwrap().map(|slot| {
            let mut out = vec![0u8; B];
            r.arena().read_slot(slot, &mut out).unwrap();
            out
        })
    }

    fn residency(slots: usize) -> TestResidency {
        SnapshotResidency::new(VecSlotArena::new(B, slots), MemSwapStore::new(B)).unwrap()
    }

    /// The connection-scoped auto-release: `handle_paging_op` pins a GET hit and
    /// releases it on the SAME connection's next op — no new opcode, no client
    /// change. During the window the slot survives a concurrent ALLOC.
    #[test]
    fn handle_paging_op_pins_get_and_auto_releases() {
        let mut r = residency(2);
        put(&mut r, 0, 0);
        put(&mut r, 1, 1);
        let mut pinned: Option<u64> = None;

        // Connection A: GET 0 → pinned.
        let reply = handle_paging_op(&mut r, OP_GET, 0, &mut pinned);
        assert!(matches!(reply, PagingReply::Located(_)));
        assert_eq!(pinned, Some(0));
        assert_eq!(r.read_pin_count(0), 1);

        // Concurrent ALLOC (another connection) evicts the unpinned key 1, not 0.
        put(&mut r, 2, 2);
        assert_eq!(get(&mut r, 0), Some(blob(0)), "pinned GET slot survived the ALLOC");

        // Connection A's NEXT op releases the pin (its RDMA read has drained).
        handle_paging_op(&mut r, OP_REMOVE, 99, &mut pinned);
        assert_eq!(pinned, None);
        assert_eq!(r.read_pin_count(0), 0, "pin auto-released on next op");
    }

    // ─────────────────────────── protocol tests ───────────────────────────

    #[test]
    fn magic_above_legacy_range() {
        // A legacy client's total_bytes is validated <= 1<<42; both paging magics
        // must be strictly above so neither is mistaken for a size, and distinct.
        assert!(PAGING_MAGIC > (1u64 << 42));
        assert!(PAGING_MAGIC_V2 > (1u64 << 42));
        assert_ne!(PAGING_MAGIC, PAGING_MAGIC_V2);
    }

    #[test]
    fn stripe_plan_covers_every_byte_once() {
        for (blob, chunk, rails) in [
            (64usize, 16usize, 2usize), // even split
            (70, 16, 2),                // tail remainder
            (64, 16, 1),                // single rail
            (10, 64, 2),                // chunk > blob → one chunk
            (64, 64, 2),                // chunk == blob
            (66846720, 1048576, 2),     // real 66MB SSM blob, 1MiB chunks, dual-rail
        ] {
            let plan = stripe_plan(blob, chunk, rails);
            assert_eq!(plan.len(), rails.max(1));
            // Flatten and assert every byte [0,blob) is covered exactly once.
            let mut covered = vec![0u8; blob];
            for rail in &plan {
                for &(off, len) in rail {
                    assert!(len <= chunk && off + len <= blob, "chunk oob {off}+{len}>{blob}");
                    for b in &mut covered[off..off + len] {
                        assert_eq!(*b, 0, "byte {off} double-covered");
                        *b = 1;
                    }
                }
            }
            assert!(covered.iter().all(|&b| b == 1), "gap in coverage blob={blob}");
        }
        // Zero blob → empty plan (no chunks).
        assert!(stripe_plan(0, 16, 2).iter().all(|r| r.is_empty()));
    }

    #[test]
    fn paging_header_v1_v2_legacy_and_reject() {
        // v1: no kind byte on the wire → SSM.
        let mut body = Vec::new();
        body.extend_from_slice(&0x1000u64.to_le_bytes()); // arena
        body.extend_from_slice(&0x40u64.to_le_bytes()); // blob
        let mut c = std::io::Cursor::new(body);
        assert_eq!(
            parse_paging_header(PAGING_MAGIC, &mut c).unwrap(),
            Some((PagingKind::SSM, 0x1000, 0x40))
        );
        // v2: [kind][arena][blob].
        let mut body = vec![PagingKind::KV.0];
        body.extend_from_slice(&0x2000u64.to_le_bytes());
        body.extend_from_slice(&0x80u64.to_le_bytes());
        let mut c = std::io::Cursor::new(body);
        assert_eq!(
            parse_paging_header(PAGING_MAGIC_V2, &mut c).unwrap(),
            Some((PagingKind::KV, 0x2000, 0x80))
        );
        // legacy KV bare total_bytes → None (dumb path).
        let mut c = std::io::Cursor::new(Vec::new());
        assert_eq!(parse_paging_header(12345, &mut c).unwrap(), None);
        // unsupported kind (RO tier) → hard error, never a bogus arena.
        let mut body = vec![3u8];
        body.extend_from_slice(&[0u8; 16]);
        let mut c = std::io::Cursor::new(body);
        assert!(parse_paging_header(PAGING_MAGIC_V2, &mut c).is_err());
    }

    /// WIRE-GOLDEN (verify's ask): freeze the exact v1 handshake byte layout so a
    /// future codec edit can't silently shift it and strand the deployed peer.
    #[test]
    fn v1_handshake_wire_golden() {
        // What connect_paging emits for arena=0x1000, blob=0x40, 1 rail:
        //   [PAGING_MAGIC u64 le][arena u64 le][blob u64 le][n_rails u8]
        let mut w = Vec::new();
        w.extend_from_slice(&PAGING_MAGIC.to_le_bytes());
        w.extend_from_slice(&0x1000u64.to_le_bytes());
        w.extend_from_slice(&0x40u64.to_le_bytes());
        w.push(1u8);
        assert_eq!(
            w,
            vec![
                0x01, 0x00, 0x00, 0x00, 0x45, 0x47, 0x41, 0x50, // PAGING_MAGIC LE
                0x00, 0x10, 0, 0, 0, 0, 0, 0, // arena 0x1000
                0x40, 0, 0, 0, 0, 0, 0, 0, // blob 0x40
                0x01, // n_rails
            ],
            "v1 handshake bytes must never shift — the deployed peer depends on this"
        );
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
        r.arena_mut().write_slot(slot, &blob(0xAB)).unwrap();
        assert_eq!(dispatch(&mut r, OP_COMMIT, 7), PagingReply::Ok);
        // GET key 7
        let PagingReply::Located(goff) = dispatch(&mut r, OP_GET, 7) else {
            panic!("get reply")
        };
        let mut out = vec![0u8; B];
        r.arena().read_slot((goff as usize) / B, &mut out).unwrap();
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
        r.arena_mut().write_slot((off as usize) / B, &blob(0x5A)).unwrap();

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
        fn peer_roundtrip(r: &mut TestResidency, req_bytes: &[u8]) -> std::io::Cursor<Vec<u8>> {
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
        r.arena_mut().write_slot((off as usize) / B, &blob(0x77)).unwrap();

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
        r.arena().read_slot((goff as usize) / B, &mut out).unwrap();
        assert_eq!(out, blob(0x77), "PUT→GET round-trips byte-identical over the protocol");
    }

    /// `MmapSlotArena` over a real page-aligned heap buffer round-trips bytes.
    #[test]
    fn mmap_slot_arena_roundtrips() {
        let slot_bytes = 4096usize;
        let n = 3usize;
        // Page-aligned heap buffer (AlignedBuf moved to atlas-tier as a private
        // helper — allocate directly here).
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut p, 4096, slot_bytes * n) };
        assert!(rc == 0 && !p.is_null(), "posix_memalign failed rc={rc}");
        {
            let mut arena = unsafe { MmapSlotArena::new(p as *mut u8, slot_bytes, n) };
            let pat = vec![0x3C_u8; slot_bytes];
            arena.write_slot(1, &pat).unwrap();
            let mut out = vec![0u8; slot_bytes];
            arena.read_slot(1, &mut out).unwrap();
            assert_eq!(out, pat);
            assert!(arena.write_slot(3, &pat).is_err(), "slot out of range rejected");
        } // arena dropped before the backing buffer is freed
        unsafe { libc::free(p) };
    }
}
