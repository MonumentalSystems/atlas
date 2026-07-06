// SPDX-License-Identifier: AGPL-3.0-only
//
// RdmaKvBackend — the KV cache overflow tier over one-sided RDMA.
//
// A drop-in `StorageBackend` (same trait the io_uring / posix NVMe backends
// implement), except the store is a peer's RAM blade (`kv_peer`) reached over
// RoCE instead of a local file:
//   * `write_from_host` (offload a cold group) -> `IBV_WR_RDMA_WRITE` the group
//     into the peer at `base + group_id * group_stride`.
//   * `read` (restore a group)                 -> `IBV_WR_RDMA_READ` it back
//     into a pinned bounce, then `copy_h2d` to the HBM destination.
//
// This is the "faster than the SSD" tier: peer RAM at ~12 GB/s over CX7 vs the
// ~2 GB/s USB SSD. Structurally identical to `PosixBackend` (single pinned
// bounce, serialize per request) — only the transport differs, so the existing
// scratch-pool / predictor / eviction machinery above it is unchanged. The peer
// CPU is idle (one-sided); each group belongs to one client, so there is no
// coherence protocol — the client is the sole owner of the blade's contents.
//
// Device/GID from `$ATLAS_EXPERT_RDMA_DEV` / `$ATLAS_EXPERT_RDMA_GID` (the same
// cabled CX7 link the expert tier uses), peer at `$ATLAS_KV_PEER=host:port`.

use std::io::{Read, Write};
use std::net::TcpStream;

use anyhow::{Context, Result, bail};

use crate::backend::{ReadRequest, StorageBackend};
use crate::cuda_min::{PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::expert_peer::{STATUS_OK, VerbsClientParams};
use crate::group::{GroupKey, GroupLayout};
use crate::kv_peer::KvServerParams;
use crate::rdma_verbs::Verbs;

pub struct RdmaKvBackend {
    verbs: Verbs,
    layout: GroupLayout,
    /// Single pinned bounce (group_stride bytes), registered LOCAL_WRITE.
    bounce: PinnedBuffer,
    bounce_lkey: u32,
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
    /// `layout`, and bring up the RC QP. The peer allocates exactly the flat
    /// group-id address space this layout spans.
    pub fn connect(addr: &str, layout: GroupLayout) -> Result<Self> {
        let group_bytes = layout.group_bytes() as usize;
        // Flat group-id space: (max group_id + 1) * stride == num_layers layers'
        // worth of (K+V × blocks × kv_heads) groups.
        let num_groups = (layout.num_layers as u64)
            * 2
            * (layout.num_blocks as u64)
            * (layout.num_kv_heads as u64);
        let total_bytes = num_groups * layout.group_stride;

        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect kv peer {addr}"))?;
        stream.set_nodelay(true).ok();
        // 1. Tell the peer how much RAM to register.
        stream
            .write_all(&total_bytes.to_le_bytes())
            .context("send kv total_bytes")?;

        let dev = std::env::var("ATLAS_EXPERT_RDMA_DEV").unwrap_or_else(|_| "roceP2p1s0f1".into());
        let gid_idx = std::env::var("ATLAS_EXPERT_RDMA_GID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3u32);
        let psn = rand::random::<u32>() & 0xff_ffff;
        let mut verbs = Verbs::create(&dev, gid_idx, psn)?;

        // The bounce is both the RDMA-READ landing buffer and the RDMA-WRITE
        // source; LOCAL_WRITE suffices for both.
        let bounce = PinnedBuffer::new(group_bytes).context("alloc pinned kv bounce")?;
        // SAFETY: bounce lives as long as self (and thus the MR).
        let bkeys = unsafe { verbs.reg_mr(bounce.ptr, group_bytes, false)? };

        // 2-4. Exchange QP params, connect, ack.
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
            "RdmaKvBackend connected to {addr}: {:.1} GiB blade, group_stride {}",
            total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            layout.group_stride,
        );
        Ok(Self {
            verbs,
            layout,
            bounce,
            bounce_lkey: bkeys.lkey,
            remote_base: sp.base_addr,
            remote_rkey: sp.rkey,
            _stream: stream,
        })
    }

    #[inline]
    fn remote_addr(&self, key: GroupKey) -> u64 {
        self.remote_base + self.layout.group_id(key).0 * self.layout.group_stride
    }
}

impl StorageBackend for RdmaKvBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        let bounce_ptr = self.bounce.ptr;
        for req in requests {
            let raddr = self.remote_addr(req.group);
            let wr = self.layout.group_id(req.group).0;
            // SAFETY: bounce is a `bytes`-sized MR (bounce_lkey); raddr/rkey
            // address the peer's RW blade at this group's flat offset.
            unsafe {
                self.verbs.post_read(
                    bounce_ptr,
                    self.bounce_lkey,
                    raddr,
                    self.remote_rkey,
                    bytes as u32,
                    wr,
                )?;
            }
            let got = self.verbs.poll()?;
            if got != wr {
                bail!("kv read completion wr_id {got:#x} != expected {wr:#x}");
            }
            // Land into HBM; sync before the next request reuses the bounce
            // (single-bounce serialization, exactly like PosixBackend).
            copy_h_to_d_async(req.dst_dev_ptr, bounce_ptr as *const _, bytes, stream)?;
            stream_sync(stream)?;
        }
        Ok(())
    }

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        if src.len() != bytes {
            bail!("write_from_host: src len {} != group bytes {bytes}", src.len());
        }
        // SAFETY: bounce holds `bytes`; copy the group in, then RDMA-WRITE it.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.bounce.ptr as *mut u8, bytes);
        }
        let raddr = self.remote_addr(key);
        let wr = self.layout.group_id(key).0;
        // SAFETY: bounce is a live `bytes`-sized MR; raddr/rkey are the blade.
        unsafe {
            self.verbs.post_write(
                self.bounce.ptr,
                self.bounce_lkey,
                raddr,
                self.remote_rkey,
                bytes as u32,
                wr,
            )?;
        }
        let got = self.verbs.poll()?;
        if got != wr {
            bail!("kv write completion wr_id {got:#x} != expected {wr:#x}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_min::{CudaCtx, DeviceBuffer, copy_d_to_h_async};
    use crate::group::KvKind;

    // Bit-identical KV round-trip over RDMA: offload a set of distinct groups to
    // the peer blade, restore each into a fresh device buffer, and confirm the
    // bytes survive WRITE -> peer RAM -> READ -> HBM unchanged. Proves the
    // overflow tier is a lossless StorageBackend. Needs a GPU and a live
    // atlas-kv-peer at $ATLAS_KV_PEER.
    #[test]
    #[ignore = "requires GPU + live kv-peer at $ATLAS_KV_PEER"]
    fn rdma_kv_round_trip() {
        let ctx = CudaCtx::new(0).expect("cuda init");
        let peer = std::env::var("ATLAS_KV_PEER").expect("set ATLAS_KV_PEER=host:port");
        // Small flat space: 2 layers × 4 blocks × 2 kv_heads, K+V.
        let layout = GroupLayout::new(2, 4, 2, 16, 128, 2, 4096);
        let bytes = layout.group_bytes() as usize;
        let mut be = RdmaKvBackend::connect(&peer, layout).expect("connect kv peer");

        let keys = [
            GroupKey::new(0, 0, 0, KvKind::K),
            GroupKey::new(0, 3, 1, KvKind::V),
            GroupKey::new(1, 2, 0, KvKind::V),
            GroupKey::new(1, 0, 1, KvKind::K),
        ];
        let pat = |i: usize| -> Vec<u8> {
            (0..bytes).map(|b| ((b + i * 37) & 0xFF) as u8).collect()
        };

        // Offload each group (RDMA WRITE to the blade).
        for (i, k) in keys.iter().enumerate() {
            be.write_from_host(*k, &pat(i)).expect("write_from_host");
        }
        // Restore each into a fresh device buffer (RDMA READ + copy_h2d) and cmp.
        for (i, k) in keys.iter().enumerate() {
            let dev = DeviceBuffer::new(bytes).unwrap();
            be.read(
                &[ReadRequest {
                    group: *k,
                    dst_dev_ptr: dev.ptr,
                }],
                ctx.stream,
            )
            .expect("read");
            let mut back = vec![0u8; bytes];
            copy_d_to_h_async(back.as_mut_ptr() as *mut _, dev.ptr, bytes, ctx.stream).unwrap();
            stream_sync(ctx.stream).unwrap();
            assert_eq!(back, pat(i), "group {k:?} corrupted through the RDMA blade");
        }
    }
}
