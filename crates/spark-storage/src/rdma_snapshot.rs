// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 4b — offset-addressed RDMA arena for the SSM-snapshot spill tier.
//!
//! A minimal, **synchronous** transport over the same CX7 verbs + `atlas-cache-peer`
//! RW-blade protocol the KV overflow tier uses, but addressed by a flat byte
//! **offset** (snapshots are keyed by an opaque id → arena slot) rather than the
//! KV `GroupKey`/`group_stride` layout — reusing that layout would corrupt live
//! KV (its `write_from_host` asserts `src.len()==group_bytes`).
//!
//! The `atlas-cache-peer` server is layout-agnostic (client sends `total_bytes`, the
//! peer registers ONE RW MR and serves `base+offset`), so a **second peer
//! instance** on its own port serves the snapshot arena with zero peer-side
//! change. Each op is drained to completion before returning (one blob ~64 MB,
//! ~5–7 ms — the spill/fault path is latency-, not throughput-critical), so the
//! caller's `SnapshotBlobStore::{put,get}` contract (durable on return) holds.
//!
//! Gathering the scattered per-layer SSM state into the contiguous blob and all
//! device-stream ordering already happen in `SsmSnapshotPool::{spill_slot,
//! fault_in_slot}`; this transport only moves host bytes.

// The real transport needs the CUDA pinned bounce + the verbs FFI; when either
// is absent, a stub whose `connect` always errors lets dependents reference the
// type unconditionally (the tier selector then falls back to host-RAM).
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub use imp::RdmaSnapshotArena;
#[cfg(not(all(feature = "cuda", atlas_rdma_verbs)))]
pub use stub::RdmaSnapshotArena;

#[cfg(not(all(feature = "cuda", atlas_rdma_verbs)))]
mod stub {
    use anyhow::{Result, bail};
    /// Placeholder when RDMA verbs / CUDA aren't built. `connect` always errors,
    /// so [`crate`] dependents degrade to the host-RAM tier; `write`/`read` are
    /// unreachable (a stub arena is never successfully constructed).
    pub struct RdmaSnapshotArena;
    impl RdmaSnapshotArena {
        pub fn connect(_addr: &str, _arena_bytes: u64, _blob_bytes: usize) -> Result<Self> {
            bail!("RDMA snapshot tier not built (needs feature `cuda` + atlas_rdma_verbs)")
        }
        pub fn connect_paging(_addr: &str, _arena_bytes: u64, _blob_bytes: usize) -> Result<Self> {
            bail!("RDMA snapshot tier not built (needs feature `cuda` + atlas_rdma_verbs)")
        }
        pub fn write(&self, _offset: u64, _bytes: &[u8]) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn read(&self, _offset: u64, _out: &mut [u8]) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn paging_put(&self, _key: u64, _bytes: &[u8]) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn paging_get(&self, _key: u64, _out: &mut [u8]) -> Result<bool> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn paging_remove(&self, _key: u64) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
    }
}

#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
mod imp {
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Mutex;

use anyhow::{Result, bail};

use crate::cuda_min::PinnedBuffer;
use crate::expert_peer::{STATUS_OK, VerbsClientParams};
use crate::cache_peer::CacheServerParams;
use crate::rdma_verbs::Verbs;

/// One rail: its QP + a single persistent registered bounce (`blob_bytes`).
struct SnapRail {
    verbs: Verbs,
    bounce: PinnedBuffer,
    lkey: u32,
    remote_rkey: u32,
}

/// Mutable rail state, serialized under one lock (the trait exposes `&self`).
struct ArenaInner {
    rails: Vec<SnapRail>,
    rr: usize,
    next_wr: u64,
    /// In legacy (dumb) mode: kept alive for the QP's lifetime, otherwise idle.
    /// In paging mode (WS-A): the live control channel — alloc/commit/get/remove
    /// requests ride this stream, interleaved with the RDMA data plane below.
    stream: TcpStream,
}

/// Offset-addressed RDMA snapshot arena. Connect to an `atlas-cache-peer` sized for
/// `arena_slots × blob_bytes`; `write`/`read` move one `blob_bytes` blob to/from
/// `base + offset`.
pub struct RdmaSnapshotArena {
    inner: Mutex<ArenaInner>,
    remote_base: u64,
    blob_bytes: usize,
}

// SAFETY: every access to the raw verbs/bounce state goes through `inner`'s
// Mutex, so there is no unsynchronized sharing; mirrors `RdmaKvBackend`'s
// single-owner contract. `Verbs` is `Send`; `PinnedBuffer` is `Send + Sync`.
unsafe impl Send for RdmaSnapshotArena {}
unsafe impl Sync for RdmaSnapshotArena {}

fn env_u32(k: &str, default: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

impl RdmaSnapshotArena {
    /// Handshake with the snapshot peer at `addr` and register `blob_bytes`
    /// bounces. Rail devices/GIDs reuse the KV env (`ATLAS_EXPERT_RDMA_DEV`/`GID`
    /// = rail 0, `ATLAS_KV_RAIL2_DEV`/`GID` = rail 1, dual only when
    /// `ATLAS_KV_DUAL_RAIL=1`). `arena_bytes` = `arena_slots × blob_bytes`.
    pub fn connect(addr: &str, arena_bytes: u64, blob_bytes: usize) -> Result<Self> {
        Self::connect_inner(addr, arena_bytes, blob_bytes, false)
    }

    /// Paging-mode connect (WS-A): the peer arena becomes a page-cache over an
    /// NVMe swap file and OWNS residency; this client uses the control channel
    /// (`paging_put`/`paging_get`/`paging_remove`) instead of a client-side
    /// allocator. Requires the peer be started with `--swap-dir`.
    pub fn connect_paging(addr: &str, arena_bytes: u64, blob_bytes: usize) -> Result<Self> {
        Self::connect_inner(addr, arena_bytes, blob_bytes, true)
    }

    fn connect_inner(addr: &str, arena_bytes: u64, blob_bytes: usize, paging: bool) -> Result<Self> {
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

        let mut stream =
            TcpStream::connect(addr).map_err(|e| anyhow::anyhow!("connect snapshot peer {addr}: {e}"))?;
        stream.set_nodelay(true).ok();
        // Paging clients select the protocol with the magic + blob size; legacy
        // (dumb one-sided) clients send only arena_bytes. See cache_peer.rs.
        if paging {
            stream.write_all(&crate::snapshot_swap::PAGING_MAGIC.to_le_bytes())?;
            stream.write_all(&arena_bytes.to_le_bytes())?;
            stream.write_all(&(blob_bytes as u64).to_le_bytes())?;
        } else {
            stream.write_all(&arena_bytes.to_le_bytes())?;
        }
        stream.write_all(&[n_rails as u8])?;

        let mut rails: Vec<SnapRail> = Vec::with_capacity(n_rails);
        for (dev, gid) in &rail_devs {
            let psn = rand::random::<u32>() & 0xff_ffff;
            let mut verbs = Verbs::create(dev, *gid, psn)?;
            let bounce = PinnedBuffer::new(blob_bytes)?;
            // SAFETY: bounce lives as long as the rail (and thus the MR); local
            // read (remote_read=false — we WRITE from it and READ into it).
            let keys = unsafe { verbs.reg_mr(bounce.ptr, blob_bytes, false)? };
            rails.push(SnapRail {
                verbs,
                bounce,
                lkey: keys.lkey,
                remote_rkey: 0,
            });
        }

        // Peer's per-rail QP + shared arena base/rkey.
        let mut b1 = [0u8; 1];
        stream.read_exact(&mut b1)?;
        if b1[0] as usize != n_rails {
            bail!("snapshot peer granted {} rails, wanted {n_rails}", b1[0]);
        }
        let mut base = 0u64;
        let mut server: Vec<CacheServerParams> = Vec::with_capacity(n_rails);
        for _ in 0..n_rails {
            let sp = CacheServerParams::read_from(&mut stream)?;
            base = sp.base_addr;
            server.push(sp);
        }
        // Reply with each rail's client QP, then connect.
        stream.write_all(&[n_rails as u8])?;
        for rail in &rails {
            VerbsClientParams {
                qpn: rail.verbs.qpn(),
                psn: rail.verbs.psn(),
                gid: rail.verbs.gid(),
            }
            .write_to(&mut stream)?;
        }
        for (rail, sp) in rails.iter_mut().zip(&server) {
            rail.verbs.connect(sp.qpn, sp.psn, &sp.gid)?;
            rail.remote_rkey = sp.rkey;
        }
        let mut ack = [0u8; 1];
        stream.read_exact(&mut ack)?;
        if ack[0] != STATUS_OK {
            bail!("snapshot peer refused connection (ack {})", ack[0]);
        }
        tracing::info!(
            "RdmaSnapshotArena connected to {addr}: {:.1} GiB arena, {n_rails} rail(s), blob {blob_bytes} B",
            arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        Ok(Self {
            inner: Mutex::new(ArenaInner {
                rails,
                rr: 0,
                next_wr: 1, // 0 == "no completion yet" sentinel in the poll loop
                stream,
            }),
            remote_base: base,
            blob_bytes,
        })
    }

    #[inline]
    pub fn blob_bytes(&self) -> usize {
        self.blob_bytes
    }

    /// RDMA-WRITE one `blob_bytes` blob to `base + offset`, drained to completion.
    pub fn write(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.blob_bytes {
            bail!("snapshot write: {} != blob_bytes {}", bytes.len(), self.blob_bytes);
        }
        let mut g = self.inner.lock().expect("snapshot arena mutex");
        self.rdma_write_locked(&mut g, self.remote_base + offset, bytes)
    }

    /// RDMA-READ one `blob_bytes` blob from `base + offset` into `out`, drained.
    pub fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        if out.len() != self.blob_bytes {
            bail!("snapshot read: {} != blob_bytes {}", out.len(), self.blob_bytes);
        }
        let mut g = self.inner.lock().expect("snapshot arena mutex");
        self.rdma_read_locked(&mut g, self.remote_base + offset, out)
    }

    /// Pick a rail (round-robin) and a fresh wr id.
    fn rail_and_wr(g: &mut ArenaInner) -> (usize, u64) {
        let n = g.rails.len();
        let ri = g.rr % n;
        g.rr = g.rr.wrapping_add(1);
        let wr = g.next_wr;
        g.next_wr = g.next_wr.wrapping_add(1).max(1);
        (ri, wr)
    }

    fn rdma_write_locked(&self, g: &mut ArenaInner, raddr: u64, bytes: &[u8]) -> Result<()> {
        let (ri, wr) = Self::rail_and_wr(g);
        let rail = &mut g.rails[ri];
        // SAFETY: bounce is a live registered MR of blob_bytes; copy the blob in,
        // RDMA-WRITE it to the peer arena, drain the single completion.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), rail.bounce.ptr as *mut u8, self.blob_bytes);
            rail.verbs.post_write(
                rail.bounce.ptr,
                rail.lkey,
                raddr,
                rail.remote_rkey,
                self.blob_bytes as u32,
                wr,
            )?;
        }
        while rail.verbs.poll()? != wr {}
        Ok(())
    }

    fn rdma_read_locked(&self, g: &mut ArenaInner, raddr: u64, out: &mut [u8]) -> Result<()> {
        let (ri, wr) = Self::rail_and_wr(g);
        let rail = &mut g.rails[ri];
        // SAFETY: read into the live bounce MR, drain, then copy host-side to out.
        unsafe {
            rail.verbs.post_read(
                rail.bounce.ptr,
                rail.lkey,
                raddr,
                rail.remote_rkey,
                self.blob_bytes as u32,
                wr,
            )?;
        }
        while rail.verbs.poll()? != wr {}
        unsafe {
            std::ptr::copy_nonoverlapping(rail.bounce.ptr as *const u8, out.as_mut_ptr(), self.blob_bytes);
        }
        Ok(())
    }

    // ─────────────────────────── paging data path (WS-A) ───────────────────────
    // The peer owns residency; we ALLOC a slot (control), RDMA-WRITE the blob,
    // then COMMIT — all under one lock so the peer's single-threaded per-conn
    // request order is respected. GET faults from the peer's NVMe swap if needed.

    /// PUT `key`'s blob into the tier. Never "full" — the peer spills to NVMe.
    pub fn paging_put(&self, key: u64, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.blob_bytes {
            bail!("paging_put: {} != blob_bytes {}", bytes.len(), self.blob_bytes);
        }
        let mut g = self.inner.lock().expect("snapshot arena mutex");
        let off = crate::snapshot_swap::client_alloc(&mut g.stream, key)?;
        self.rdma_write_locked(&mut g, self.remote_base + off, bytes)?;
        crate::snapshot_swap::client_commit(&mut g.stream, key)
    }

    /// GET `key`'s blob into `out`. `Ok(false)` = the peer has no such key.
    pub fn paging_get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        if out.len() != self.blob_bytes {
            bail!("paging_get: {} != blob_bytes {}", out.len(), self.blob_bytes);
        }
        let mut g = self.inner.lock().expect("snapshot arena mutex");
        match crate::snapshot_swap::client_get(&mut g.stream, key)? {
            Some(off) => {
                self.rdma_read_locked(&mut g, self.remote_base + off, out)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Drop `key` from the peer cache.
    pub fn paging_remove(&self, key: u64) -> Result<()> {
        let mut g = self.inner.lock().expect("snapshot arena mutex");
        crate::snapshot_swap::client_remove(&mut g.stream, key)
    }
}
}
