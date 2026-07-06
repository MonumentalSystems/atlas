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
//      KvServerParams [u32 qpn][u32 psn][16 gid][u64 base_addr][u32 rkey]
//   3. client -> VerbsClientParams [u32 qpn][u32 psn][16 gid]
//   4. peer connects its QP, replies [u8 STATUS_OK]
//   5. client does one-sided WRITE/READ; peer idles until the client hangs up,
//      then unregisters + unmaps the blade.

use anyhow::{Context, Result, bail};

/// The peer's half of the KV handshake: its QP identity + the single RW MR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvServerParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    pub base_addr: u64,
    pub rkey: u32,
}

impl KvServerParams {
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

    /// RDMA device selection for the blade (defaults match the cabled GB10 link).
    #[derive(Clone, Debug)]
    pub struct RdmaConfig {
        pub dev: String,
        pub gid_idx: u32,
    }

    impl Default for RdmaConfig {
        fn default() -> Self {
            Self {
                dev: "roceP2p1s0f1".into(),
                gid_idx: 3,
            }
        }
    }

    /// Serve a KV overflow blade on `addr` until interrupted. One thread per
    /// connection; each connection gets its own RW arena sized by the client.
    pub fn serve<A: ToSocketAddrs>(addr: A, rdma: RdmaConfig) -> Result<()> {
        let listener = TcpListener::bind(addr).context("bind kv-peer listener")?;
        let local = listener.local_addr().ok();
        tracing::info!(
            "kv-peer (RW RDMA overflow blade) listening on {:?} (verbs dev={} gid={})",
            local,
            rdma.dev,
            rdma.gid_idx,
        );
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("kv-peer accept error: {e}");
                    continue;
                }
            };
            let rdma = rdma.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_conn(stream, &rdma) {
                    tracing::warn!("kv-peer connection ended: {e}");
                }
            });
        }
        Ok(())
    }

    #[cfg(not(atlas_rdma_verbs))]
    fn handle_conn(_stream: TcpStream, _rdma: &RdmaConfig) -> Result<()> {
        bail!("kv-peer needs a build with rdma-core (atlas_rdma_verbs)");
    }

    #[cfg(atlas_rdma_verbs)]
    fn handle_conn(mut stream: TcpStream, rdma: &RdmaConfig) -> Result<()> {
        use crate::expert_peer::{STATUS_OK, VerbsClientParams};
        use crate::rdma_verbs::Verbs;
        use std::io::{Read, Write};
        stream.set_nodelay(true).ok();

        // 1. Client tells us how much RAM it will address.
        let mut b8 = [0u8; 8];
        stream.read_exact(&mut b8).context("read total_bytes")?;
        let total = u64::from_le_bytes(b8) as usize;
        if total == 0 || total > (1usize << 42) {
            bail!("implausible kv blade size: {total}");
        }

        // Anonymous, page-aligned, lazily-zeroed arena (faulted+pinned by reg_mr).
        let arena = Mmap::anon(total).context("mmap kv blade arena")?;
        let psn: u32 = 0x5a5a5a ^ std::process::id();
        let mut verbs = Verbs::create(&rdma.dev, rdma.gid_idx, psn & 0xff_ffff)?;
        // SAFETY: the arena outlives `verbs` (dropped after it below).
        let keys = unsafe { verbs.reg_mr_rw(arena.addr as *mut _, arena.len)? };

        // 2. Publish our QP + the RW MR.
        KvServerParams {
            qpn: verbs.qpn(),
            psn: verbs.psn(),
            gid: verbs.gid(),
            base_addr: arena.addr as u64,
            rkey: keys.rkey,
        }
        .write_to(&mut stream)
        .context("send kv server params")?;

        // 3-4. Learn the client's QP, connect, ack.
        let cp = VerbsClientParams::read_from(&mut stream).context("read kv client params")?;
        verbs.connect(cp.qpn, cp.psn, &cp.gid)?;
        stream.write_all(&[STATUS_OK]).context("send kv ready ack")?;
        tracing::info!(
            "kv-peer client connected (qpn {} -> {}, {:.1} GiB RW blade)",
            verbs.qpn(),
            cp.qpn,
            total as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // 5. Idle until the client hangs up. All movement is one-sided.
        let mut sink = [0u8; 8];
        loop {
            match stream.read(&mut sink) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        // Dereg (verbs) before unmap (arena): drop verbs first.
        drop(verbs);
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
                bail!("mmap anon {len} failed: {}", std::io::Error::last_os_error());
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
        let sp = KvServerParams {
            qpn: 0x4242,
            psn: 0x0012_3456 & 0xff_ffff,
            gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 178, 12],
            base_addr: 0x7f00_1234_0000,
            rkey: 0xdead_beef,
        };
        let mut buf = Vec::new();
        sp.write_to(&mut buf).unwrap();
        assert_eq!(KvServerParams::read_from(&mut &buf[..]).unwrap(), sp);
    }
}
