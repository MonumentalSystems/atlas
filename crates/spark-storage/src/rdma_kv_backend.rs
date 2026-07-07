// SPDX-License-Identifier: AGPL-3.0-only
//
// RdmaKvBackend — the KV cache overflow tier over one-sided RDMA.
//
// A drop-in `StorageBackend` (same trait the io_uring / posix NVMe backends
// implement), except the store is a peer's RAM blade (`kv_peer`) reached over
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
use std::io::{Read, Write};
use std::net::TcpStream;

use anyhow::{Context, Result, bail};

use crate::backend::{ReadRequest, StorageBackend};
use crate::cuda_min::{PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::expert_peer::{STATUS_OK, VerbsClientParams};
use crate::group::{GroupKey, GroupLayout};
use crate::kv_peer::KvServerParams;
use crate::rdma_verbs::Verbs;

/// One registered pinned bounce in a rail's pipeline ring.
struct Bounce {
    buf: PinnedBuffer,
    lkey: u32,
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
    free: Vec<usize>,
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
    /// first `copy_h2d` the landed bytes to its HBM dest on `stream`.
    fn reap_one(&mut self, group_bytes: usize, stream: u64) -> Result<()> {
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
                bounce
            }
            InFlight::Write { bounce } => bounce,
        };
        self.free.push(bounce);
        Ok(())
    }

    fn drain(&mut self, group_bytes: usize, stream: u64) -> Result<()> {
        while !self.inflight.is_empty() {
            self.reap_one(group_bytes, stream)?;
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
    /// N rails (RC QPs across the CX7 adapters), and allocate each rail's ring.
    pub fn connect(addr: &str, layout: GroupLayout) -> Result<Self> {
        let group_bytes = layout.group_bytes() as usize;
        let num_groups = (layout.num_layers as u64)
            * 2
            * (layout.num_blocks as u64)
            * (layout.num_kv_heads as u64);
        let total_bytes = num_groups * layout.group_stride;

        // Rail devices: rail 0 from the expert env (shared CX7 link), rail 1 from
        // the KV rail-2 env. Dual-rail only when ATLAS_KV_DUAL_RAIL=1.
        let dev0 = std::env::var("ATLAS_EXPERT_RDMA_DEV").unwrap_or_else(|_| "roceP2p1s0f1".into());
        let gid0 = env_u32("ATLAS_EXPERT_RDMA_GID", 3);
        let dev1 = std::env::var("ATLAS_KV_RAIL2_DEV").unwrap_or_else(|_| "rocep1s0f1".into());
        let gid1 = env_u32("ATLAS_KV_RAIL2_GID", 3);
        let dual = std::env::var("ATLAS_KV_DUAL_RAIL").ok().as_deref() == Some("1");
        let rail_devs: Vec<(String, u32)> = if dual {
            vec![(dev0, gid0), (dev1, gid1)]
        } else {
            vec![(dev0, gid0)]
        };
        let n_rails = rail_devs.len();
        let depth: usize = env_u32("ATLAS_KV_PIPELINE_DEPTH", 16).clamp(1, 128) as usize;

        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect kv peer {addr}"))?;
        stream.set_nodelay(true).ok();
        stream
            .write_all(&total_bytes.to_le_bytes())
            .context("send kv total_bytes")?;
        stream.write_all(&[n_rails as u8]).context("send n_rails")?;

        // Create each rail's QP + bounce ring.
        let mut rails: Vec<Rail> = Vec::with_capacity(n_rails);
        for (dev, gid) in &rail_devs {
            let psn = rand::random::<u32>() & 0xff_ffff;
            let mut verbs = Verbs::create(dev, *gid, psn)?;
            let mut bounces = Vec::with_capacity(depth);
            for _ in 0..depth {
                let buf = PinnedBuffer::new(group_bytes).context("alloc pinned kv bounce")?;
                // SAFETY: buf lives as long as the rail (and thus the MR).
                let keys = unsafe { verbs.reg_mr(buf.ptr, group_bytes, false)? };
                bounces.push(Bounce {
                    buf,
                    lkey: keys.lkey,
                });
            }
            rails.push(Rail {
                verbs,
                remote_rkey: 0, // filled from the handshake below
                free: (0..depth).collect(),
                bounces,
                inflight: HashMap::new(),
                next_wr: 0,
                dst_lkeys: HashMap::new(),
                region: None,
                direct_inflight: 0,
            });
        }

        // Read the peer's per-rail QP + rkey (shared base).
        let mut b1 = [0u8; 1];
        stream.read_exact(&mut b1).context("read peer n_rails")?;
        if b1[0] as usize != n_rails {
            bail!("peer granted {} rails, wanted {n_rails}", b1[0]);
        }
        let mut base = 0u64;
        let mut server: Vec<KvServerParams> = Vec::with_capacity(n_rails);
        for _ in 0..n_rails {
            let sp = KvServerParams::read_from(&mut stream).context("read kv server params")?;
            base = sp.base_addr;
            server.push(sp);
        }
        // Reply with each rail's client QP, then connect.
        stream
            .write_all(&[n_rails as u8])
            .context("send client n_rails")?;
        for rail in &rails {
            VerbsClientParams {
                qpn: rail.verbs.qpn(),
                psn: rail.verbs.psn(),
                gid: rail.verbs.gid(),
            }
            .write_to(&mut stream)
            .context("send kv client params")?;
        }
        for (rail, sp) in rails.iter_mut().zip(&server) {
            rail.verbs.connect(sp.qpn, sp.psn, &sp.gid)?;
            rail.remote_rkey = sp.rkey;
        }
        let mut ack = [0u8; 1];
        stream.read_exact(&mut ack).context("read kv ready ack")?;
        if ack[0] != STATUS_OK {
            bail!("kv peer refused connection (ack {})", ack[0]);
        }
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
        // landed before the next kernel that reads them is queued.) This leading
        // WAR sync is intentionally RETAINED under the relaxed StorageBackend::read
        // contract: the off-stream NIC DMA is not stream-ordered against the prior
        // kernel, so it needs an explicit CPU-visible fence that same-stream
        // ordering cannot provide.
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

fn env_u32(k: &str, default: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

impl StorageBackend for RdmaKvBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
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
                    let b = rail.free.pop().unwrap();
                    let raddr = self.remote_base
                        + self.layout.group_id(requests[j].group).0 * self.layout.group_stride;
                    // SAFETY: bounce b is a live MR; raddr/rkey are the blade.
                    unsafe { rail.post_read(b, raddr, bytes, requests[j].dst_dev_ptr)? };
                }
                if !rail.inflight.is_empty() {
                    rail.reap_one(bytes, stream)?;
                    active = true;
                }
            }
            if !active && pend.iter().all(|q| q.is_empty()) {
                break;
            }
        }
        stream_sync(stream)?;
        Ok(())
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
            rail.reap_one(bytes, 0)?; // only writes are in flight here (no copy)
        }
        let b = rail.free.pop().expect("free bounce after reap");
        // SAFETY: bounce b holds `bytes`; copy the group in, then RDMA-WRITE it.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), rail.bounces[b].buf.ptr as *mut u8, bytes);
            rail.post_write(b, raddr, bytes)?;
        }
        Ok(()) // async — reaped lazily / drained before the next read
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
    #[ignore = "requires GPU + live kv-peer at $ATLAS_KV_PEER"]
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
    #[ignore = "requires GPU + live kv-peer at $ATLAS_KV_PEER"]
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
