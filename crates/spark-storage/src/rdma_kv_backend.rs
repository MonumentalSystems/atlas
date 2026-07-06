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
// PIPELINED: a ring of `depth` registered bounce buffers (default 16, env
// `ATLAS_KV_PIPELINE_DEPTH`) keeps up to `depth` RDMA ops in flight so per-op
// latency overlaps across a batch, and RDMA READs overlap the `copy_h2d` — the
// serial single-bounce version was latency-bound at KV group sizes. `read` posts
// the whole batch through the ring and does one `stream_sync`; `write_from_host`
// posts async and reaps completions lazily (writes are drained before any read,
// so a restore always sees prior offloads, and on drop for durability).
//
// This is the "faster than the SSD" tier: peer RAM at (up to) ~12 GB/s over CX7
// vs the ~2 GB/s USB SSD. Peer CPU is idle (one-sided); each group belongs to
// one client, so there is no coherence protocol — the client owns the blade.
//
// Device/GID from `$ATLAS_EXPERT_RDMA_DEV` / `$ATLAS_EXPERT_RDMA_GID` (the same
// cabled CX7 link the expert tier uses), peer at `$ATLAS_KV_PEER=host:port`.

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

/// One registered pinned bounce in the pipeline ring.
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

pub struct RdmaKvBackend {
    verbs: Verbs,
    layout: GroupLayout,
    /// Registered bounce ring (pipeline depth).
    bounces: Vec<Bounce>,
    /// Indices of bounces not currently holding an in-flight op.
    free: Vec<usize>,
    /// wr_id -> the op occupying a bounce.
    inflight: HashMap<u64, InFlight>,
    next_wr: u64,
    remote_base: u64,
    remote_rkey: u32,
    _stream: TcpStream, // hold the control channel open; drop => peer tears down
}

// `StorageBackend: Send + Sync`. `Verbs` (raw `*mut RsConn`) is Send but not
// Sync. Both trait methods take `&mut self`, and the backend exposes no `&self`
// method that touches the QP, so a shared `&RdmaKvBackend` can do nothing — the
// QP is only ever driven under exclusive access. Sync is therefore sound; the
// swap orchestrator owns the backend on a single thread regardless.
unsafe impl Sync for RdmaKvBackend {}

impl RdmaKvBackend {
    /// Connect to a KV blade at `addr`, size + register the peer arena for this
    /// `layout`, allocate the bounce ring, and bring up the RC QP.
    pub fn connect(addr: &str, layout: GroupLayout) -> Result<Self> {
        let group_bytes = layout.group_bytes() as usize;
        // Flat group-id space: (max group_id + 1) * stride.
        let num_groups = (layout.num_layers as u64)
            * 2
            * (layout.num_blocks as u64)
            * (layout.num_kv_heads as u64);
        let total_bytes = num_groups * layout.group_stride;

        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect kv peer {addr}"))?;
        stream.set_nodelay(true).ok();
        stream
            .write_all(&total_bytes.to_le_bytes())
            .context("send kv total_bytes")?;

        let dev = std::env::var("ATLAS_EXPERT_RDMA_DEV").unwrap_or_else(|_| "roceP2p1s0f1".into());
        let gid_idx = std::env::var("ATLAS_EXPERT_RDMA_GID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3u32);
        let depth: usize = std::env::var("ATLAS_KV_PIPELINE_DEPTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16)
            .clamp(1, 128);
        let psn = rand::random::<u32>() & 0xff_ffff;
        let mut verbs = Verbs::create(&dev, gid_idx, psn)?;

        // The bounce ring: each is both a RDMA-READ landing buffer and a
        // RDMA-WRITE source; LOCAL_WRITE suffices for both.
        let mut bounces = Vec::with_capacity(depth);
        for _ in 0..depth {
            let buf = PinnedBuffer::new(group_bytes).context("alloc pinned kv bounce")?;
            // SAFETY: buf lives as long as self (and thus the MR).
            let keys = unsafe { verbs.reg_mr(buf.ptr, group_bytes, false)? };
            bounces.push(Bounce {
                buf,
                lkey: keys.lkey,
            });
        }
        let free = (0..depth).collect();

        let sp = KvServerParams::read_from(&mut stream).context("read kv server params")?;
        VerbsClientParams {
            qpn: verbs.qpn(),
            psn: verbs.psn(),
            gid: verbs.gid(),
        }
        .write_to(&mut stream)
        .context("send kv client params")?;
        verbs.connect(sp.qpn, sp.psn, &sp.gid)?;
        let mut ack = [0u8; 1];
        stream.read_exact(&mut ack).context("read kv ready ack")?;
        if ack[0] != STATUS_OK {
            bail!("kv peer refused connection (ack {})", ack[0]);
        }
        tracing::info!(
            "RdmaKvBackend connected to {addr}: {:.1} GiB blade, group_stride {}, pipeline depth {depth}",
            total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            layout.group_stride,
        );
        Ok(Self {
            verbs,
            layout,
            bounces,
            free,
            inflight: HashMap::new(),
            next_wr: 0,
            remote_base: sp.base_addr,
            remote_rkey: sp.rkey,
            _stream: stream,
        })
    }

    #[inline]
    fn remote_addr(&self, key: GroupKey) -> u64 {
        self.remote_base + self.layout.group_id(key).0 * self.layout.group_stride
    }

    #[inline]
    fn fresh_wr(&mut self) -> u64 {
        let w = self.next_wr;
        self.next_wr = self.next_wr.wrapping_add(1);
        w
    }

    /// Reap exactly one completion, freeing its bounce. For a READ, first
    /// `copy_h2d` the landed bytes to its HBM dest on `stream`. Returns the freed
    /// bounce index.
    fn reap_one(&mut self, stream: u64) -> Result<usize> {
        let wr = self.verbs.poll()?;
        let op = self
            .inflight
            .remove(&wr)
            .with_context(|| format!("kv: completion for unknown wr_id {wr:#x}"))?;
        let bounce = match op {
            InFlight::Read { bounce, dst } => {
                let bytes = self.layout.group_bytes() as usize;
                copy_h_to_d_async(dst, self.bounces[bounce].buf.ptr as *const _, bytes, stream)?;
                bounce
            }
            InFlight::Write { bounce } => bounce,
        };
        self.free.push(bounce);
        Ok(bounce)
    }

    /// Drain all in-flight ops (used before a read so it sees prior writes, and
    /// on drop for durability). Reads land + copy on `stream`.
    fn drain(&mut self, stream: u64) -> Result<()> {
        while !self.inflight.is_empty() {
            self.reap_one(stream)?;
        }
        Ok(())
    }
}

impl StorageBackend for RdmaKvBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        // Ensure any pending offloads land first, so a restore sees them.
        self.drain(stream)?;
        let bytes = self.layout.group_bytes() as usize;
        let mut it = requests.iter();
        let mut done = false;
        loop {
            // Fill free bounces with new READs (keep the pipeline full).
            while let Some(&b) = self.free.last() {
                let Some(req) = it.next() else {
                    done = true;
                    break;
                };
                self.free.pop();
                let raddr = self.remote_addr(req.group);
                let wr = self.fresh_wr();
                // SAFETY: bounce b is a `bytes`-sized MR; raddr/rkey are the blade.
                unsafe {
                    self.verbs.post_read(
                        self.bounces[b].buf.ptr,
                        self.bounces[b].lkey,
                        raddr,
                        self.remote_rkey,
                        bytes as u32,
                        wr,
                    )?;
                }
                self.inflight.insert(
                    wr,
                    InFlight::Read {
                        bounce: b,
                        dst: req.dst_dev_ptr,
                    },
                );
            }
            if self.inflight.is_empty() && done {
                break;
            }
            // Reap one (READ -> copy_h2d), freeing a bounce to refill above.
            self.reap_one(stream)?;
        }
        // One sync for all the copies issued on `stream`.
        stream_sync(stream)?;
        Ok(())
    }

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        if src.len() != bytes {
            bail!("write_from_host: src len {} != group bytes {bytes}", src.len());
        }
        // Acquire a free bounce, reaping a completion if the ring is full.
        if self.free.is_empty() {
            // No reads should be outstanding here (reads fully drain), so this
            // reaps a write; NULL stream is fine (writes issue no copy_h2d).
            self.reap_one(0)?;
        }
        let b = self.free.pop().expect("free bounce after reap");
        // SAFETY: bounce b holds `bytes`; copy the group in, then RDMA-WRITE it.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.bounces[b].buf.ptr as *mut u8, bytes);
        }
        let raddr = self.remote_addr(key);
        let wr = self.fresh_wr();
        // SAFETY: bounce b is a live `bytes`-sized MR; raddr/rkey are the blade.
        unsafe {
            self.verbs.post_write(
                self.bounces[b].buf.ptr,
                self.bounces[b].lkey,
                raddr,
                self.remote_rkey,
                bytes as u32,
                wr,
            )?;
        }
        self.inflight.insert(wr, InFlight::Write { bounce: b });
        Ok(()) // async — reaped lazily / drained before the next read
    }
}

impl Drop for RdmaKvBackend {
    fn drop(&mut self) {
        // Drain in-flight writes so the blade holds all offloaded bytes before
        // the connection closes (best-effort; ignore poll errors on teardown).
        let _ = self.drain(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_min::{CudaCtx, DeviceBuffer, copy_d_to_h_async};
    use crate::group::KvKind;

    // Bit-identical KV round-trip over RDMA through the pipeline: offload a set
    // of distinct groups, restore each, confirm bytes survive WRITE -> peer RAM
    // -> READ -> HBM unchanged. Needs a GPU and a live atlas-kv-peer.
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
        // Restore all in ONE batched read (exercises the pipeline).
        let devs: Vec<_> = keys.iter().map(|_| DeviceBuffer::new(bytes).unwrap()).collect();
        let reqs: Vec<_> = keys
            .iter()
            .zip(&devs)
            .map(|(k, d)| ReadRequest {
                group: *k,
                dst_dev_ptr: d.ptr,
            })
            .collect();
        be.read(&reqs, ctx.stream).expect("read");
        for (i, d) in devs.iter().enumerate() {
            let mut back = vec![0u8; bytes];
            copy_d_to_h_async(back.as_mut_ptr() as *mut _, d.ptr, bytes, ctx.stream).unwrap();
            stream_sync(ctx.stream).unwrap();
            assert_eq!(back, pat(i), "group {:?} corrupted through the RDMA blade", keys[i]);
        }
    }

    // Throughput of the pipelined KV overflow tier: offload then restore a large
    // group set, reporting GB/s each way. Needs a GPU and a live atlas-kv-peer.
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
        let dev = DeviceBuffer::new(gbytes).unwrap();

        let t0 = std::time::Instant::now();
        for k in &keys {
            be.write_from_host(*k, &src).expect("write");
        }
        // Drain the async write pipeline so the timing includes durability.
        be.drain(0).expect("drain writes");
        let wdt = t0.elapsed().as_secs_f64();

        // Restore all groups in one batched, pipelined read (reuse one dst — we
        // measure transport, not distinct destinations).
        let reqs: Vec<_> = keys
            .iter()
            .map(|k| ReadRequest {
                group: *k,
                dst_dev_ptr: dev.ptr,
            })
            .collect();
        let t1 = std::time::Instant::now();
        be.read(&reqs, ctx.stream).expect("read");
        let rdt = t1.elapsed().as_secs_f64();

        let gbps = |dt: f64| (total as f64) / dt / 1e9;
        println!(
            "\nRDMA KV tier (pipelined): {} groups × {} B = {:.0} MiB\n  \
             OFFLOAD (RDMA WRITE): {:.3}s => {:.2} GB/s ({:.1} us/group)\n  \
             RESTORE (RDMA READ + h2d): {:.3}s => {:.2} GB/s ({:.1} us/group)",
            ngroups,
            gbytes,
            total as f64 / 1048576.0,
            wdt,
            gbps(wdt),
            wdt / ngroups as f64 * 1e6,
            rdt,
            gbps(rdt),
            rdt / ngroups as f64 * 1e6,
        );
    }
}
