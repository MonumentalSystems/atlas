// SPDX-License-Identifier: AGPL-3.0-only
//
// KV overflow blade — a dumb remote-RAM tier for the high-speed-swap KV cache.
//
// Where `expert_peer` serves READ-ONLY expert weights over one-sided RDMA READ,
// this serves a READ-WRITE slab of RAM: a streaming client OFFLOADS cold K/V
// groups into it with `IBV_WR_RDMA_WRITE` and RESTORES them with
// `IBV_WR_RDMA_READ`, both one-sided, peer CPU idle. It is the "faster than the
// SSD" overflow tier: local pinned RAM → **peer RAM (~12 GB/s over CX7)** →
// local NVMe/USB SSD (~2 GB/s). The peer owns nothing — each group belongs to
// exactly one client sequence; this process is a passive memory blade.
//
// Addressing is the flat group-id space of `GroupLayout`: a group lands at
// `base + group_id * group_stride`, so no per-group bookkeeping on the peer.
//
// Wire protocol (little-endian), connection-oriented:
//   1. client -> [u64 total_bytes]  (num_groups * group_stride it will address)
//   2. peer allocates + registers a RW MR of that size, replies with
//      CacheServerParams [u32 qpn][u32 psn][16 gid][u64 base_addr][u32 rkey]
//   3. client -> VerbsClientParams [u32 qpn][u32 psn][16 gid]
//   4. peer connects its QP, replies [u8 STATUS_OK]
//   5. client does one-sided WRITE/READ; peer idles until the client hangs up,
//      then unregisters + unmaps the blade.

use anyhow::{Context, Result, bail};

/// The peer's half of the KV handshake: its QP identity + the single RW MR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheServerParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    pub base_addr: u64,
    pub rkey: u32,
}

impl CacheServerParams {
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.qpn.to_le_bytes())?;
        w.write_all(&self.psn.to_le_bytes())?;
        w.write_all(&self.gid)?;
        w.write_all(&self.base_addr.to_le_bytes())?;
        w.write_all(&self.rkey.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let mut b4 = [0u8; 4];
        let mut b8 = [0u8; 8];
        let mut gid = [0u8; 16];
        r.read_exact(&mut b4).context("kv qpn")?;
        let qpn = u32::from_le_bytes(b4);
        r.read_exact(&mut b4).context("kv psn")?;
        let psn = u32::from_le_bytes(b4);
        r.read_exact(&mut gid).context("kv gid")?;
        r.read_exact(&mut b8).context("kv base")?;
        let base_addr = u64::from_le_bytes(b8);
        r.read_exact(&mut b4).context("kv rkey")?;
        let rkey = u32::from_le_bytes(b4);
        Ok(Self {
            qpn,
            psn,
            gid,
            base_addr,
            rkey,
        })
    }
}

#[cfg(unix)]
pub use server_impl::{RdmaConfig, serve};

#[cfg(unix)]
mod server_impl {
    use super::*;
    use std::net::{TcpListener, TcpStream, ToSocketAddrs};

    /// RDMA rail selection for the blade. One `(dev, gid_idx)` per CX7 adapter;
    /// a client requests N rails and the peer registers its arena on each so the
    /// client can stripe traffic across both adapters (~1.75x aggregate on GB10).
    #[derive(Clone, Debug)]
    pub struct RdmaConfig {
        /// `(device, gid_idx)` per rail, in link order (rail 0 = .178, 1 = .177).
        pub rails: Vec<(String, u32)>,
        /// Ceiling on total committed (registered) blade RAM across all
        /// concurrent connections, in bytes. `0` = unlimited (the default).
        pub max_blade_bytes: u64,
        /// Directory for NVMe swap files backing paging-mode connections
        /// (WS-A). `None` = paging clients are refused (RAM-only). When set, a
        /// paging connection's RDMA arena becomes a page-cache over a per-conn
        /// O_DIRECT swap file here → "infinite depth".
        pub swap_dir: Option<std::path::PathBuf>,
    }

    impl Default for RdmaConfig {
        fn default() -> Self {
            Self {
                rails: vec![("roceP2p1s0f1".into(), 3), ("rocep1s0f1".into(), 3)],
                max_blade_bytes: 0,
                swap_dir: None,
            }
        }
    }

    /// Serve a KV overflow blade on `addr` until interrupted. One thread per
    /// connection; each connection gets its own RW arena sized by the client.
    pub fn serve<A: ToSocketAddrs>(addr: A, rdma: RdmaConfig) -> Result<()> {
        let listener = TcpListener::bind(addr).context("bind cache-peer listener")?;
        let local = listener.local_addr().ok();
        // One process-global ledger, shared by every connection thread; a
        // connection reserves its arena size before it maps/registers any RAM.
        let ledger = std::sync::Arc::new(crate::blade_cap::CommitLedger::new(rdma.max_blade_bytes));
        tracing::info!(
            "cache-peer (RW RDMA overflow blade) listening on {:?} (rails {:?}, cap {})",
            local,
            rdma.rails,
            if rdma.max_blade_bytes == 0 {
                "unlimited".to_string()
            } else {
                format!(
                    "{:.1} GiB",
                    rdma.max_blade_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                )
            },
        );
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("cache-peer accept error: {e}");
                    continue;
                }
            };
            let rdma = rdma.clone();
            let ledger = ledger.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_conn(stream, &rdma, &ledger) {
                    tracing::warn!("cache-peer connection ended: {e}");
                }
            });
        }
        Ok(())
    }

    #[cfg(not(atlas_rdma_verbs))]
    fn handle_conn(
        _stream: TcpStream,
        _rdma: &RdmaConfig,
        _ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<()> {
        bail!("cache-peer needs a build with rdma-core (atlas_rdma_verbs)");
    }

    #[cfg(atlas_rdma_verbs)]
    fn handle_conn(
        mut stream: TcpStream,
        rdma: &RdmaConfig,
        ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<()> {
        use crate::expert_peer::{STATUS_OK, VerbsClientParams};
        use crate::rdma_verbs::Verbs;
        use std::io::{Read, Write};
        stream.set_nodelay(true).ok();

        // 1. Client tells us how much RAM to register and how many rails it
        //    wants. Backward-compatible paging select (WS-A): a paging client
        //    sends `PAGING_MAGIC` as its first u64 (far above the legacy
        //    `total_bytes` range, which is validated <= 1<<42), then
        //    `[u64 arena_bytes][u64 blob_bytes]`. A legacy KV client's first u64
        //    IS `total_bytes`, so it is never mis-parsed.
        let mut b8 = [0u8; 8];
        stream.read_exact(&mut b8).context("read total_bytes/magic")?;
        let first = u64::from_le_bytes(b8);
        let (total, paging_blob): (usize, Option<usize>) =
            if first == crate::snapshot_swap::PAGING_MAGIC {
                stream.read_exact(&mut b8).context("read paging arena_bytes")?;
                let arena_bytes = u64::from_le_bytes(b8) as usize;
                stream.read_exact(&mut b8).context("read paging blob_bytes")?;
                let blob = u64::from_le_bytes(b8) as usize;
                if blob == 0 || arena_bytes == 0 || !arena_bytes.is_multiple_of(blob) {
                    bail!("paging: bad arena_bytes {arena_bytes} / blob_bytes {blob}");
                }
                // Reject paging BEFORE the rail handshake when this peer has no
                // swap dir — the client's `connect_paging` then errors cleanly on
                // the missing peer reply and falls back to the bounded/host-RAM
                // tier (vs a confusing mid-session failure after STATUS_OK).
                if rdma.swap_dir.is_none() {
                    bail!("paging client but peer started without --swap-dir; refusing");
                }
                (arena_bytes, Some(blob))
            } else {
                (first as usize, None)
            };
        if total == 0 || total > (1usize << 42) {
            bail!("implausible kv blade size: {total}");
        }
        let mut b1 = [0u8; 1];
        stream.read_exact(&mut b1).context("read n_rails")?;
        let n_rails = b1[0] as usize;
        if n_rails == 0 || n_rails > rdma.rails.len() {
            bail!(
                "client asked for {n_rails} rails; peer has {}",
                rdma.rails.len()
            );
        }

        // Admission gate: charge the arena size ONCE (the N per-rail MRs pin the
        // SAME refcounted pages, so the committed footprint is `total`, not
        // total*n_rails). Reserve BEFORE any mmap/reg_mr pins RAM; the RAII guard
        // releases on every exit below (early bail, reg_mr error, normal hangup).
        let _reservation = ledger.try_reserve(total as u64).context("kv blade cap")?;

        // Anonymous, page-aligned, lazily-zeroed arena — registered ONCE per rail
        // (each device its own PD/rkey). The physical pages are shared (pinned
        // refcounted), so N rails do NOT cost N× RAM — only N MR handles + rkeys.
        let arena = Mmap::anon(total).context("mmap kv blade arena")?;
        let pid = std::process::id();
        let mut rails: Vec<Verbs> = Vec::with_capacity(n_rails);
        let mut rkeys: Vec<u32> = Vec::with_capacity(n_rails);
        for (i, (dev, gid)) in rdma.rails.iter().take(n_rails).enumerate() {
            let psn = (0x5a5a5a ^ pid ^ ((i as u32) << 20)) & 0xff_ffff;
            let mut v = Verbs::create(dev, *gid, psn)?;
            // SAFETY: the arena outlives every rail (dropped after them below).
            let keys = unsafe { v.reg_mr_rw(arena.addr as *mut _, arena.len)? };
            rkeys.push(keys.rkey);
            rails.push(v);
        }

        // 2. Publish rail count + each rail's QP + rkey (shared base).
        stream.write_all(&[n_rails as u8]).context("send n_rails")?;
        for (v, rkey) in rails.iter().zip(&rkeys) {
            CacheServerParams {
                qpn: v.qpn(),
                psn: v.psn(),
                gid: v.gid(),
                base_addr: arena.addr as u64,
                rkey: *rkey,
            }
            .write_to(&mut stream)
            .context("send kv server params")?;
        }

        // 3-4. Learn each client rail's QP, connect, ack.
        stream.read_exact(&mut b1).context("read client n_rails")?;
        if b1[0] as usize != n_rails {
            bail!("client rail count mismatch");
        }
        for v in rails.iter_mut() {
            let cp = VerbsClientParams::read_from(&mut stream).context("read kv client params")?;
            v.connect(cp.qpn, cp.psn, &cp.gid)?;
        }
        stream
            .write_all(&[STATUS_OK])
            .context("send kv ready ack")?;
        tracing::info!(
            "cache-peer client connected: {n_rails} rail(s), {:.1} GiB RW blade",
            total as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // 5. Data plane.
        if let Some(blob_bytes) = paging_blob {
            // Paging mode (WS-A): the arena is a page-cache over an O_DIRECT NVMe
            // swap file; the peer owns residency and faults from disk on GET.
            // Bytes still move one-sided over RDMA into/out of the arena slots;
            // only tiny [op][key] control messages cross this TCP stream. The MR
            // is never re-registered — swap happens under the stable rkey.
            use std::os::fd::AsRawFd;
            let num_slots = total / blob_bytes;
            let run = |stream: &mut TcpStream| -> Result<()> {
                let swap_dir = rdma
                    .swap_dir
                    .as_ref()
                    .context("paging client but peer has no --swap-dir configured")?;
                let swap_path =
                    swap_dir.join(format!("snap-{pid}-{:x}.swap", stream.as_raw_fd() as u64));
                let swap = crate::snapshot_swap::DirectSwapFile::create(&swap_path, blob_bytes)?;
                // SAFETY: `arena` outlives this loop (dropped just below).
                let slot_arena = unsafe {
                    crate::snapshot_swap::MmapSlotArena::new(
                        arena.addr as *mut u8,
                        blob_bytes,
                        num_slots,
                    )
                };
                let mut residency =
                    crate::snapshot_swap::SnapshotResidency::new(slot_arena, swap)?;
                tracing::info!(
                    "cache-peer PAGING client: {num_slots} slots × {blob_bytes} B RAM arena \
                     + NVMe swap {} (infinite depth)",
                    swap_path.display(),
                );
                let r = crate::snapshot_swap::run_paging_loop(stream, &mut residency);
                let _ = std::fs::remove_file(&swap_path); // per-conn swap is ephemeral
                r
            };
            let res = run(&mut stream);
            drop(rails);
            drop(arena);
            return res;
        }

        // Legacy one-sided KV blade: idle until the client hangs up.
        let mut sink = [0u8; 8];
        loop {
            match stream.read(&mut sink) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        // Dereg (rails) before unmap (arena): drop rails first.
        drop(rails);
        drop(arena);
        Ok(())
    }

    /// A page-aligned anonymous mapping, unmapped on drop.
    #[cfg(atlas_rdma_verbs)]
    struct Mmap {
        addr: *mut libc::c_void,
        len: usize,
    }

    #[cfg(atlas_rdma_verbs)]
    impl Mmap {
        fn anon(len: usize) -> Result<Self> {
            // SAFETY: standard anonymous private mapping of `len` bytes.
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if addr == libc::MAP_FAILED {
                bail!(
                    "mmap anon {len} failed: {}",
                    std::io::Error::last_os_error()
                );
            }
            Ok(Self { addr, len })
        }
    }

    #[cfg(atlas_rdma_verbs)]
    impl Drop for Mmap {
        fn drop(&mut self) {
            // SAFETY: addr/len from a successful mmap, unmapped once.
            unsafe { libc::munmap(self.addr, self.len) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_server_params_round_trip() {
        let sp = CacheServerParams {
            qpn: 0x4242,
            psn: 0x0012_3456 & 0xff_ffff,
            gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 178, 12],
            base_addr: 0x7f00_1234_0000,
            rkey: 0xdead_beef,
        };
        let mut buf = Vec::new();
        sp.write_to(&mut buf).unwrap();
        assert_eq!(CacheServerParams::read_from(&mut &buf[..]).unwrap(), sp);
    }
}
