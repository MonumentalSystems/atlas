// SPDX-License-Identifier: AGPL-3.0-only
//
// RdmaKvBackend — the KV cache overflow tier over one-sided RDMA.
//
// A drop-in `StorageBackend` (same trait the io_uring / posix NVMe backends
// implement), except the store is a peer's RAM blade (`cache_peer`) reached over
// RoCE instead of a local file:
//   * `write_from_host` (offload a cold group) -> `IBV_WR_RDMA_WRITE` the group
//     into the peer at `base + group_id * group_stride`.
//   * `read` (restore groups)                  -> `IBV_WR_RDMA_READ` them back
//     into pinned bounces, then `copy_h2d` to the HBM destinations.
//
// PIPELINED + DUAL-RAIL. Each RAIL is a QP on one CX7 adapter with its own ring
// of `depth` registered bounce buffers (env `ATLAS_KV_PIPELINE_DEPTH`, default
// 16). With `ATLAS_KV_DUAL_RAIL=1` the client opens 2 rails (env
// `ATLAS_EXPERT_RDMA_DEV`/`GID` = rail 0, `ATLAS_KV_RAIL2_DEV`/`GID` = rail 1)
// and stripes ops round-robin across both adapters — the two GB10 CX7 ports are
// independent PCIe paths (~1.75x aggregate). The peer registers its arena once
// per rail (shared physical pages, refcounted pinning → not N× RAM).
//
// The pipeline keeps up to `depth` RDMA ops in flight per rail so per-op latency
// overlaps across a batch and RDMA READs overlap `copy_h2d`. `read` posts the
// batch across all rails and reaps completions (interleaved so both rails run in
// parallel), one `stream_sync` at the end. `write_from_host` posts async and
// reaps lazily; writes are drained before any read (a restore always sees prior
// offloads) and on drop for durability.
//
// This is the "faster than the SSD" tier: peer RAM over CX7 vs the ~2 GB/s USB
// SSD. Peer CPU idle (one-sided); each group belongs to one client, no coherence.

use std::collections::HashMap;
use std::ffi::c_void;
use std::io::Write;
use std::net::TcpStream;

use anyhow::{Context, Result, bail};

use crate::backend::{ReadRequest, StorageBackend};
use crate::cuda_min::{CudaEvent, PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout};
use crate::rdma_verbs::Verbs;

/// One registered pinned bounce in a rail's pipeline ring.
struct Bounce {
    buf: PinnedBuffer,
    lkey: u32,
    /// #11-refinement: async prefetch reuse guard. `read_async` records a CUDA
    /// event on the copy stream AFTER `copy_h_to_d_async` drains this bounce;
    /// the next reuse of this bounce `sync`s it first (via `wait_bounce_free`),
    /// so the host cannot re-fill the bounce before the device copy has read it —
    /// the reuse hazard the deleted terminal host stream_sync used to prevent.
    /// `None` on the synchronous path (that path keeps its terminal stream_sync),
    /// so the guard is a pure no-op there → byte-identical for prefetch-OFF.
    copy_done: Option<CudaEvent>,
}

/// An in-flight RDMA op, keyed by its `wr_id`, so a completion can be dispatched.
enum InFlight {
    /// A restore: after the READ lands, copy the bounce to this HBM dest.
    Read { bounce: usize, dst: u64 },
    /// An offload: the WRITE from this bounce; just free it on completion.
    Write { bounce: usize },
}

/// One QP on one CX7 adapter, with its own bounce ring + completion tracking.
struct Rail {
    verbs: Verbs,
    remote_rkey: u32,
    bounces: Vec<Bounce>,
    free: std::collections::VecDeque<usize>,
    inflight: HashMap<u64, InFlight>,
    next_wr: u64,
    /// Zero-copy restore: lkeys of destination MRs registered on demand (a UMA
    /// dst is GPU-addressable, so RDMA lands there directly — no bounce, no
    /// copy_h2d). Cached by dst address; KV scratch slots are reused.
    dst_lkeys: HashMap<u64, u32>,
    /// Pre-registered whole landing region `(base, len, lkey)`: one MR covering
    /// the entire (UMA) scratch pool. Any dst inside it reuses this lkey, so we
    /// never register per-slot sub-regions (which fail on GB10).
    region: Option<(u64, u64, u32)>,
    /// In-flight direct (zero-copy) reads on this rail — no bounce to free.
    direct_inflight: usize,
}

impl Rail {
    #[inline]
    fn fresh_wr(&mut self) -> u64 {
        let w = self.next_wr;
        self.next_wr = self.next_wr.wrapping_add(1);
        w
    }

    /// Register (once, cached) a `bytes`-sized destination MR at `addr` for a
    /// zero-copy RDMA READ landing. On GB10 UMA the dst is GPU-addressable pinned
    /// host memory, so `ibv_reg_mr` on its VA succeeds and the GPU reads the
    /// landed bytes at the same address — no `copy_h2d`.
    /// Register `[base, base+len)` as ONE landing MR on this rail (the whole UMA
    /// scratch pool). Called once, before any restore.
    fn register_region(&mut self, base: u64, len: usize) -> Result<()> {
        // SAFETY: base/len describe the pool's live UMA (pinned) allocation,
        // which outlives every rail (deregistered on drop before the pool frees).
        let keys = unsafe { self.verbs.reg_mr(base as *mut c_void, len, false) }
            .context("register UMA landing region")?;
        self.region = Some((base, len as u64, keys.lkey));
        Ok(())
    }

    fn reg_dst(&mut self, addr: u64, bytes: usize) -> Result<u32> {
        // Whole-region fast path: any dst inside the pre-registered pool reuses
        // its single lkey (no per-slot registration — that fails on GB10).
        if let Some((base, len, lkey)) = self.region
            && addr >= base
            && addr + bytes as u64 <= base + len
        {
            return Ok(lkey);
        }
        if let Some(&lk) = self.dst_lkeys.get(&addr) {
            return Ok(lk);
        }
        // SAFETY: caller guarantees zero-copy mode => addr is a live UMA buffer
        // of at least `bytes` (else reg_mr fails, surfacing a clear error).
        let keys = unsafe { self.verbs.reg_mr(addr as *mut c_void, bytes, false) }
            .context("zero-copy restore needs a UMA (GPU-addressable) dst; reg_mr failed")?;
        self.dst_lkeys.insert(addr, keys.lkey);
        Ok(keys.lkey)
    }

    /// Reap exactly one completion on this rail, freeing its bounce. For a READ,
    /// first `copy_h2d` the landed bytes to its HBM dest on `stream`. When
    /// `track` is set (async prefetch), record a CUDA event on `stream` after
    /// that copy so a later reuse of this bounce can `sync` on the copy draining
    /// the bounce (replacing the deleted terminal host stream_sync). `track` is
    /// only ever true on `read_async`'s READ reaps; the sync path and all write
    /// drains pass false, leaving `copy_done = None` → byte-identical.
    fn reap_one(&mut self, group_bytes: usize, stream: u64, track: bool) -> Result<()> {
        let wr = self.verbs.poll()?;
        let op = self
            .inflight
            .remove(&wr)
            .with_context(|| format!("kv: completion for unknown wr_id {wr:#x}"))?;
        let bounce = match op {
            InFlight::Read { bounce, dst } => {
                copy_h_to_d_async(
                    dst,
                    self.bounces[bounce].buf.ptr as *const _,
                    group_bytes,
                    stream,
                )?;
                if track {
                    let ev = CudaEvent::new()?;
                    ev.record(stream)?;
                    self.bounces[bounce].copy_done = Some(ev);
                }
                bounce
            }
            InFlight::Write { bounce } => bounce,
        };
        self.free.push_back(bounce);
        Ok(())
    }

    /// #11-refinement: before reusing bounce `b`, wait for any still-in-flight
    /// async copy_h2d that is draining it (recorded by a prior `read_async`
    /// reap). No-op on the sync path (`copy_done` is always `None`), so it never
    /// perturbs prefetch-OFF. Off the decode run-ahead loop it only ever fires
    /// under genuine copy-engine backpressure (correct async pushback).
    fn wait_bounce_free(&mut self, b: usize) -> Result<()> {
        if let Some(ev) = self.bounces[b].copy_done.take() {
            ev.sync()?;
        }
        Ok(())
    }

    fn drain(&mut self, group_bytes: usize, stream: u64) -> Result<()> {
        while !self.inflight.is_empty() {
            // Drained ops here are writes (before a read) or teardown reaps — no
            // new mirror-RAW consumer needs the copy event, so track=false.
            self.reap_one(group_bytes, stream, false)?;
        }
        Ok(())
    }

    /// # Safety: bounce/len/remote must describe a live MR and the peer arena.
    unsafe fn post_read(
        &mut self,
        bounce: usize,
        raddr: u64,
        bytes: usize,
        dst: u64,
    ) -> Result<()> {
        let wr = self.fresh_wr();
        unsafe {
            self.verbs.post_read(
                self.bounces[bounce].buf.ptr,
                self.bounces[bounce].lkey,
                raddr,
                self.remote_rkey,
                bytes as u32,
                wr,
            )?;
        }
        self.inflight.insert(wr, InFlight::Read { bounce, dst });
        Ok(())
    }

    /// # Safety: as `post_read`; `src` bytes already copied into the bounce.
    unsafe fn post_write(&mut self, bounce: usize, raddr: u64, bytes: usize) -> Result<()> {
        let wr = self.fresh_wr();
        unsafe {
            self.verbs.post_write(
                self.bounces[bounce].buf.ptr,
                self.bounces[bounce].lkey,
                raddr,
                self.remote_rkey,
                bytes as u32,
                wr,
            )?;
        }
        self.inflight.insert(wr, InFlight::Write { bounce });
        Ok(())
    }
}

pub struct RdmaKvBackend {
    rails: Vec<Rail>,
    layout: GroupLayout,
    remote_base: u64,
    rr: usize, // round-robin rail cursor for writes
    /// Zero-copy restore (ATLAS_KV_ZERO_COPY=1): RDMA READ lands directly into
    /// the (UMA) destination, skipping the bounce + copy_h2d that otherwise caps
    /// restore at the copy-engine bandwidth.
    zero_copy: bool,
    _stream: TcpStream,
}

// See the single-rail rationale: both trait methods take `&mut self` and no
// `&self` method touches a QP, so `Sync` is sound (the swap orchestrator owns it
// single-threaded regardless).
unsafe impl Sync for RdmaKvBackend {}

impl RdmaKvBackend {
    /// Connect to a KV blade at `addr`, size + register the peer arena, bring up
    /// N rails (RC QPs across the CX7 adapters) via
    /// [`atlas_rdma::railset::RailSet`], and allocate each rail's ring.
    pub fn connect(addr: &str, layout: GroupLayout) -> Result<Self> {
        use atlas_rdma::env::{first_set, first_set_u32};
        use atlas_rdma::railset::{RailSet, RailSpec};

        let group_bytes = layout.group_bytes() as usize;
        let num_groups = (layout.num_layers as u64)
            * 2
            * (layout.num_blocks as u64)
            * (layout.num_kv_heads as u64);
        let total_bytes = num_groups * layout.group_stride;

        // Rail devices: rail 0 from the expert env (shared CX7 link), rail 1 from
        // the KV rail-2 env. Dual-rail only when ATLAS_KV_DUAL_RAIL=1. Fresh
        // random 24-bit PSN per rail.
        let spec =
            |dev: String, gid: u32| RailSpec::new(dev, gid, rand::random::<u32>() & 0xff_ffff);
        let rail0 = spec(
            first_set(&["ATLAS_EXPERT_RDMA_DEV"], "roceP2p1s0f1"),
            first_set_u32(&["ATLAS_EXPERT_RDMA_GID"], 3),
        );
        let dual = std::env::var("ATLAS_KV_DUAL_RAIL").ok().as_deref() == Some("1");
        let specs: Vec<RailSpec> = if dual {
            let rail1 = spec(
                first_set(&["ATLAS_KV_RAIL2_DEV"], "rocep1s0f1"),
                first_set_u32(&["ATLAS_KV_RAIL2_GID"], 3),
            );
            vec![rail0, rail1]
        } else {
            vec![rail0]
        };
        let n_rails = specs.len();
        let depth: usize = first_set_u32(&["ATLAS_KV_PIPELINE_DEPTH"], 16).clamp(1, 128) as usize;

        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect kv peer {addr}"))?;
        stream.set_nodelay(true).ok();
        stream
            .write_all(&total_bytes.to_le_bytes())
            .context("send kv total_bytes")?;

        // [u8 n_rails] + one QP per rail, then each rail's bounce ring
        // (LOCAL_WRITE-only landing MRs — `remote_read == false`, invariant).
        let mut rs = RailSet::begin(&mut stream, &specs)?;
        let mut rings: Vec<Vec<Bounce>> = Vec::with_capacity(n_rails);
        for rail in &mut rs.rails {
            let mut bounces = Vec::with_capacity(depth);
            for _ in 0..depth {
                let buf = PinnedBuffer::new(group_bytes).context("alloc pinned kv bounce")?;
                // SAFETY: buf lives as long as the rail (and thus the MR).
                let keys = unsafe { rail.verbs.reg_mr(buf.ptr, group_bytes, false)? };
                bounces.push(Bounce {
                    buf,
                    lkey: keys.lkey,
                    copy_done: None,
                });
            }
            rings.push(bounces);
        }

        // Peer's per-rail QP + rkey (shared base), client params, connect, ack.
        let server = rs.finish_rw(&mut stream, "kv peer")?;
        // Shared arena base: every rail publishes the same one (keep the LAST,
        // the pre-RailSet loop-overwrite behavior).
        let base = server.last().map(|sp| sp.base_addr).unwrap_or(0);
        let rails: Vec<Rail> = rs
            .into_verbs()
            .into_iter()
            .zip(rings)
            .zip(&server)
            .map(|((verbs, bounces), sp)| Rail {
                verbs,
                remote_rkey: sp.rkey,
                free: (0..depth).collect(),
                bounces,
                inflight: HashMap::new(),
                next_wr: 0,
                dst_lkeys: HashMap::new(),
                region: None,
                direct_inflight: 0,
            })
            .collect();
        tracing::info!(
            "RdmaKvBackend connected to {addr}: {:.1} GiB blade, {n_rails} rail(s), \
             group_stride {}, pipeline depth {depth}",
            total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            layout.group_stride,
        );
        Ok(Self {
            rails,
            layout,
            remote_base: base,
            rr: 0,
            zero_copy: std::env::var("ATLAS_KV_ZERO_COPY").ok().as_deref() == Some("1"),
            _stream: stream,
        })
    }

    #[inline]
    fn remote_addr(&self, key: GroupKey) -> u64 {
        self.remote_base + self.layout.group_id(key).0 * self.layout.group_stride
    }

    /// Zero-copy restore: RDMA READ each group DIRECTLY into its (UMA) HBM dest —
    /// the dst is registered as the landing MR, so there is no bounce and no
    /// `copy_h2d`. On completion the bytes are already GPU-visible at `dst`
    /// (same host==dev VA), so no `stream_sync` either. Removes the copy-engine
    /// bottleneck that pinned single-rail restore at ~9.7 GB/s, letting it
    /// dual-rail. Requires UMA destinations (else `reg_dst` errors clearly).
    fn read_zero_copy(
        &mut self,
        requests: &[ReadRequest],
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        // WAR barrier: the NIC is about to DMA into UMA slots that the PREVIOUS
        // tile's attention kernel may still be reading on `stream`. Unlike the
        // bounce path (whose copy_h2d is stream-ordered after attention + ends in
        // stream_sync), the NIC write is off-stream, so we must drain in-flight
        // consumers of these slots first — else zero-copy restore under eviction
        // pressure silently corrupts KV. This restores the bounce path's implicit
        // barrier. (RAW is already safe: the poll below means the bytes have
        // landed before the next kernel that reads them is queued.)
        stream_sync(stream)?;
        let n = self.rails.len();
        let depth = self.rails[0].bounces.len(); // in-flight cap per rail
        let mut pend: Vec<std::collections::VecDeque<usize>> = vec![Default::default(); n];
        for (j, _) in requests.iter().enumerate() {
            pend[j % n].push_back(j);
        }
        loop {
            let mut active = false;
            for (ri, rail) in self.rails.iter_mut().enumerate() {
                while rail.direct_inflight < depth {
                    let Some(j) = pend[ri].pop_front() else { break };
                    let dst = requests[j].dst_dev_ptr;
                    let lkey = rail.reg_dst(dst, bytes)?;
                    let raddr = self.remote_base
                        + self.layout.group_id(requests[j].group).0 * self.layout.group_stride;
                    let wr = rail.fresh_wr();
                    // SAFETY: dst is a live UMA MR (lkey) of `bytes`; raddr/rkey
                    // address the peer blade. The NIC DMAs straight into the
                    // GPU-addressable dst.
                    unsafe {
                        rail.verbs.post_read(
                            dst as *mut c_void,
                            lkey,
                            raddr,
                            rail.remote_rkey,
                            bytes as u32,
                            wr,
                        )?;
                    }
                    rail.direct_inflight += 1;
                }
                if rail.direct_inflight > 0 {
                    rail.verbs.poll()?; // completion => bytes GPU-visible in dst
                    rail.direct_inflight -= 1;
                    active = true;
                }
            }
            if !active && pend.iter().all(|q| q.is_empty()) {
                break;
            }
        }
        Ok(())
    }
}

impl RdmaKvBackend {
    /// Body shared by `read` (sync) and `read_async` (#11-refinement). Bounce
    /// path only differs by `is_async`:
    ///   * sync  (`false`): LIFO bounce reuse (`pop_back` = original `pop`),
    ///     `track=false` (no per-bounce copy event), terminal host `stream_sync`.
    ///     Byte-identical to the pre-refinement `read`; `wait_bounce_free` is a
    ///     pure no-op (all `copy_done == None`).
    ///   * async (`true`): FIFO bounce reuse (`pop_front` — reuse the OLDEST
    ///     freed bounce so its `copy_done` has had `depth`-many copy-times to
    ///     drain, making `wait_bounce_free` a steady-state no-op; LIFO would
    ///     stall on every reuse and defeat run-ahead), `track=true` (record a
    ///     copy event per reaped READ), and NO terminal `stream_sync` — mirror-
    ///     RAW is closed by the caller's `kv_prefetch_done`.
    /// The zero-copy branch is shared verbatim (see `read_zero_copy` / §G: it
    /// keeps its own leading WAR host sync, the one honestly-unremovable block).
    fn read_common(
        &mut self,
        requests: &[ReadRequest],
        stream: u64,
        is_async: bool,
    ) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        // Ensure any pending offloads land first, so a restore sees them.
        for rail in &mut self.rails {
            rail.drain(bytes, stream)?;
        }
        if self.zero_copy {
            if requests.is_empty() {
                return Ok(());
            }
            // Register EVERY dst on EVERY rail up front — before any RDMA READ is
            // posted — so a reg_mr failure (dst not UMA-registerable on some rail)
            // degrades to the bounce path CLEANLY, with no half-posted batch. A
            // per-slot / per-rail probe is required: rail 1's device/PD can reject
            // a host region rail 0 accepted, and later scratch slots differ from
            // the first. reg_dst caches, so read_zero_copy reuses these lkeys.
            let mut all_ok = true;
            'reg: for req in requests {
                for rail in &mut self.rails {
                    if let Err(e) = rail.reg_dst(req.dst_dev_ptr, bytes) {
                        tracing::warn!(
                            "kv restore dst not UMA-registerable ({e:#}); \
                             permanently using bounce restore"
                        );
                        all_ok = false;
                        break 'reg;
                    }
                }
            }
            if all_ok {
                return self.read_zero_copy(requests, bytes, stream);
            }
            // Non-UMA dst — fall through to the bounce path for this and all
            // future reads.
            self.zero_copy = false;
        }
        let n = self.rails.len();
        // Per-rail queues of pending request indices, striped round-robin.
        let mut pend: Vec<std::collections::VecDeque<usize>> = vec![Default::default(); n];
        for (j, _) in requests.iter().enumerate() {
            pend[j % n].push_back(j);
        }
        // Drive all rails in parallel: each outer pass fills every rail's free
        // bounces with new READs, then reaps one from each rail that has work.
        loop {
            let mut active = false;
            for (ri, rail) in self.rails.iter_mut().enumerate() {
                while !rail.free.is_empty() {
                    let Some(j) = pend[ri].pop_front() else { break };
                    // FIFO on async (oldest freed bounce, copy_done drained),
                    // LIFO on sync (== original `pop()`, byte-identical order).
                    let b = if is_async {
                        rail.free.pop_front()
                    } else {
                        rail.free.pop_back()
                    }
                    .unwrap();
                    // Reuse gate: wait any async copy_h2d still draining this
                    // bounce before the NIC refills it. No-op unless a prior
                    // read_async left a copy_done on `b` (always None on the pure
                    // sync/prefetch-OFF path → byte-identical).
                    rail.wait_bounce_free(b)?;
                    let raddr = self.remote_base
                        + self.layout.group_id(requests[j].group).0 * self.layout.group_stride;
                    // SAFETY: bounce b is a live MR; raddr/rkey are the blade.
                    unsafe { rail.post_read(b, raddr, bytes, requests[j].dst_dev_ptr)? };
                }
                if !rail.inflight.is_empty() {
                    rail.reap_one(bytes, stream, is_async)?;
                    active = true;
                }
            }
            if !active && pend.iter().all(|q| q.is_empty()) {
                break;
            }
        }
        if !is_async {
            // Sync path: finalise the stream (PosixBackend semantics). The async
            // path deliberately omits this — that deleted host block IS the win.
            stream_sync(stream)?;
        }
        Ok(())
    }
}

impl StorageBackend for RdmaKvBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read_common(requests, stream, false)
    }

    fn read_async(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read_common(requests, stream, true)
    }

    fn register_landing_region(&mut self, base: u64, len: usize) -> Result<()> {
        // Register the whole (UMA) scratch pool as one MR per rail so zero-copy
        // restore reuses that lkey for every slot — no per-slot registration.
        for rail in &mut self.rails {
            rail.register_region(base, len)?;
        }
        tracing::info!(
            "RdmaKvBackend: registered UMA landing region {:.1} MiB on {} rail(s) — zero-copy restore live",
            len as f64 / (1024.0 * 1024.0),
            self.rails.len(),
        );
        Ok(())
    }

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        if src.len() != bytes {
            bail!(
                "write_from_host: src len {} != group bytes {bytes}",
                src.len()
            );
        }
        let raddr = self.remote_addr(key);
        let n = self.rails.len();
        let ri = self.rr % n;
        self.rr = self.rr.wrapping_add(1);
        let rail = &mut self.rails[ri];
        // Acquire a free bounce on this rail, reaping a completion if full.
        if rail.free.is_empty() {
            rail.reap_one(bytes, 0, false)?; // only writes are in flight here (no copy)
        }
        let b = rail.free.pop_back().expect("free bounce after reap");
        // #11-refinement: an offload staging-write must not race a still-draining
        // async prefetch copy_h2d out of this bounce. No-op unless a prior
        // read_async left a copy_done on `b` (always None on the pure sync path);
        // when it fires it is on the offload/eviction path, off the decode loop.
        rail.wait_bounce_free(b)?;
        // SAFETY: bounce b holds `bytes`; copy the group in, then RDMA-WRITE it.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), rail.bounces[b].buf.ptr as *mut u8, bytes);
            rail.post_write(b, raddr, bytes)?;
        }
        Ok(()) // async — reaped lazily / drained before the next read
    }

    fn group_layout(&self) -> GroupLayout {
        // Inherits the DEFAULT block read/write (per-head fan-out over the
        // one-sided verbs path) — correct, un-coalesced. Native multi-group RDMA
        // coalescing is a noted follow-up.
        self.layout
    }
}

impl Drop for RdmaKvBackend {
    fn drop(&mut self) {
        let bytes = self.layout.group_bytes() as usize;
        for rail in &mut self.rails {
            let _ = rail.drain(bytes, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_min::CudaCtx;
    use crate::group::KvKind;

    // Read a UMA (pinned, host==dev VA) dst's bytes host-side — valid for both
    // the bounce path (copy_h2d lands there) and zero-copy (RDMA lands there).
    unsafe fn uma_bytes(buf: &PinnedBuffer, n: usize) -> &[u8] {
        unsafe { std::slice::from_raw_parts(buf.ptr as *const u8, n) }
    }

    #[test]
    #[ignore = "requires GPU + live cache-peer at $ATLAS_KV_PEER"]
    fn rdma_kv_round_trip() {
        let _ctx = CudaCtx::new(0).expect("cuda init");
        let peer = std::env::var("ATLAS_KV_PEER").expect("set ATLAS_KV_PEER=host:port");
        let layout = GroupLayout::new(2, 4, 2, 16, 128, 2, 4096);
        let bytes = layout.group_bytes() as usize;
        let mut be = RdmaKvBackend::connect(&peer, layout).expect("connect kv peer");
        let keys = [
            GroupKey::new(0, 0, 0, KvKind::K),
            GroupKey::new(0, 3, 1, KvKind::V),
            GroupKey::new(1, 2, 0, KvKind::V),
            GroupKey::new(1, 0, 1, KvKind::K),
        ];
        let pat =
            |i: usize| -> Vec<u8> { (0..bytes).map(|b| ((b + i * 37) & 0xFF) as u8).collect() };
        for (i, k) in keys.iter().enumerate() {
            be.write_from_host(*k, &pat(i)).expect("write_from_host");
        }
        // UMA dsts so the same test validates both the bounce and zero-copy paths.
        let devs: Vec<_> = keys
            .iter()
            .map(|_| PinnedBuffer::new(bytes).unwrap())
            .collect();
        let reqs: Vec<_> = keys
            .iter()
            .zip(&devs)
            .map(|(k, d)| ReadRequest {
                group: *k,
                dst_dev_ptr: d.device_ptr().unwrap(),
            })
            .collect();
        be.read(&reqs, _ctx.stream).expect("read");
        for (i, d) in devs.iter().enumerate() {
            let back = unsafe { uma_bytes(d, bytes) };
            assert_eq!(
                back,
                &pat(i)[..],
                "group {:?} corrupted through the RDMA blade",
                keys[i]
            );
        }
    }

    #[test]
    #[ignore = "requires GPU + live cache-peer at $ATLAS_KV_PEER"]
    fn rdma_kv_bandwidth() {
        let ctx = CudaCtx::new(0).expect("cuda init");
        let peer = std::env::var("ATLAS_KV_PEER").expect("set ATLAS_KV_PEER=host:port");
        let layout = GroupLayout::new(16, 64, 8, 64, 128, 2, 4096);
        let gbytes = layout.group_bytes() as usize;
        let mut be = RdmaKvBackend::connect(&peer, layout).expect("connect kv peer");
        let ngroups = (layout.num_layers as u64)
            * 2
            * (layout.num_blocks as u64)
            * (layout.num_kv_heads as u64);
        let total = ngroups * gbytes as u64;
        let keys: Vec<GroupKey> = (0..layout.num_layers)
            .flat_map(|l| {
                (0..layout.num_blocks).flat_map(move |b| {
                    (0..layout.num_kv_heads).flat_map(move |h| {
                        [
                            GroupKey::new(l, b, h, KvKind::K),
                            GroupKey::new(l, b, h, KvKind::V),
                        ]
                    })
                })
            })
            .collect();
        let src = vec![0xABu8; gbytes];
        // UMA dst so zero-copy (ATLAS_KV_ZERO_COPY=1) can RDMA straight in.
        let dst = PinnedBuffer::new(gbytes).unwrap();
        let dptr = dst.device_ptr().unwrap();

        let t0 = std::time::Instant::now();
        for k in &keys {
            be.write_from_host(*k, &src).expect("write");
        }
        for rail in &mut be.rails {
            rail.drain(gbytes, 0).expect("drain");
        }
        let wdt = t0.elapsed().as_secs_f64();

        let reqs: Vec<_> = keys
            .iter()
            .map(|k| ReadRequest {
                group: *k,
                dst_dev_ptr: dptr,
            })
            .collect();
        let t1 = std::time::Instant::now();
        be.read(&reqs, ctx.stream).expect("read");
        let rdt = t1.elapsed().as_secs_f64();

        let gbps = |dt: f64| (total as f64) / dt / 1e9;
        println!(
            "\nRDMA KV tier ({} rail(s), {}, pipelined): {} groups × {} B = {:.0} MiB\n  \
             OFFLOAD (RDMA WRITE): {:.3}s => {:.2} GB/s\n  \
             RESTORE (RDMA READ): {:.3}s => {:.2} GB/s",
            be.rails.len(),
            if be.zero_copy {
                "zero-copy"
            } else {
                "bounce+h2d"
            },
            ngroups,
            gbytes,
            total as f64 / 1048576.0,
            wdt,
            gbps(wdt),
            rdt,
            gbps(rdt),
        );
    }
}
