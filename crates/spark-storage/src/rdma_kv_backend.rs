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
}

impl Rail {
    #[inline]
    fn fresh_wr(&mut self) -> u64 {
        let w = self.next_wr;
        self.next_wr = self.next_wr.wrapping_add(1);
        w
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
                copy_h_to_d_async(dst, self.bounces[bounce].buf.ptr as *const _, group_bytes, stream)?;
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
    unsafe fn post_read(&mut self, bounce: usize, raddr: u64, bytes: usize, dst: u64) -> Result<()> {
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
        stream
            .write_all(&[n_rails as u8])
            .context("send n_rails")?;

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
                bounces.push(Bounce { buf, lkey: keys.lkey });
            }
            rails.push(Rail {
                verbs,
                remote_rkey: 0, // filled from the handshake below
                free: (0..depth).collect(),
                bounces,
                inflight: HashMap::new(),
                next_wr: 0,
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
        stream.write_all(&[n_rails as u8]).context("send client n_rails")?;
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
            _stream: stream,
        })
    }

    #[inline]
    fn remote_addr(&self, key: GroupKey) -> u64 {
        self.remote_base + self.layout.group_id(key).0 * self.layout.group_stride
    }
}

fn env_u32(k: &str, default: u32) -> u32 {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

impl StorageBackend for RdmaKvBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        // Ensure any pending offloads land first, so a restore sees them.
        for rail in &mut self.rails {
            rail.drain(bytes, stream)?;
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
                    let raddr =
                        self.remote_base + self.layout.group_id(requests[j].group).0 * self.layout.group_stride;
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

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        if src.len() != bytes {
            bail!("write_from_host: src len {} != group bytes {bytes}", src.len());
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
    use crate::cuda_min::{CudaCtx, DeviceBuffer, copy_d_to_h_async};
    use crate::group::KvKind;

    #[test]
    #[ignore = "requires GPU + live kv-peer at $ATLAS_KV_PEER"]
    fn rdma_kv_round_trip() {
        let ctx = CudaCtx::new(0).expect("cuda init");
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
        let pat = |i: usize| -> Vec<u8> { (0..bytes).map(|b| ((b + i * 37) & 0xFF) as u8).collect() };
        for (i, k) in keys.iter().enumerate() {
            be.write_from_host(*k, &pat(i)).expect("write_from_host");
        }
        let devs: Vec<_> = keys.iter().map(|_| DeviceBuffer::new(bytes).unwrap()).collect();
        let reqs: Vec<_> = keys
            .iter()
            .zip(&devs)
            .map(|(k, d)| ReadRequest { group: *k, dst_dev_ptr: d.ptr })
            .collect();
        be.read(&reqs, ctx.stream).expect("read");
        for (i, d) in devs.iter().enumerate() {
            let mut back = vec![0u8; bytes];
            copy_d_to_h_async(back.as_mut_ptr() as *mut _, d.ptr, bytes, ctx.stream).unwrap();
            stream_sync(ctx.stream).unwrap();
            assert_eq!(back, pat(i), "group {:?} corrupted through the RDMA blade", keys[i]);
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
                        [GroupKey::new(l, b, h, KvKind::K), GroupKey::new(l, b, h, KvKind::V)]
                    })
                })
            })
            .collect();
        let src = vec![0xABu8; gbytes];
        let dev = DeviceBuffer::new(gbytes).unwrap();

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
            .map(|k| ReadRequest { group: *k, dst_dev_ptr: dev.ptr })
            .collect();
        let t1 = std::time::Instant::now();
        be.read(&reqs, ctx.stream).expect("read");
        let rdt = t1.elapsed().as_secs_f64();

        let gbps = |dt: f64| (total as f64) / dt / 1e9;
        println!(
            "\nRDMA KV tier ({} rail(s), pipelined): {} groups × {} B = {:.0} MiB\n  \
             OFFLOAD (RDMA WRITE): {:.3}s => {:.2} GB/s\n  \
             RESTORE (RDMA READ + h2d): {:.3}s => {:.2} GB/s",
            be.rails.len(),
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
