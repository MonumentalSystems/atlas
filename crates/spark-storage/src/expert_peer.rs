// SPDX-License-Identifier: AGPL-3.0-only
//
// Expert weight-serving peer + wire protocol (Stage 4, Phase A: TCP).
//
// The RDMA weight-tier's first incarnation: a peer that holds the resident
// expert store and serves records to a streaming client over a socket, into the
// client's pinned arena. This proves the residency-tier abstraction (a peer as
// a fetch tier, distinct from EP sharding) with zero verbs risk, over the RoCE
// Ethernet netdev. Phase B swaps the transport for one-sided RDMA_READ into the
// SAME arena (see RESEARCH-RDMA-TIER.md) — the protocol geometry is identical.
//
// Wire protocol (little-endian), connection-oriented:
//   1. On accept the server sends the manifest: [u32 len][len bytes of JSON].
//      The client parses it to size its arena (same ExpertIndex geometry).
//   2. Request loop: client sends [u32 layer][u32 expert]; server replies
//      [u8 status][record_stride bytes] (status 0 = OK, nonzero = error, no
//      payload). A layer/expert of u32::MAX/u32::MAX is a graceful shutdown.
//
// The peer is pure I/O (no CUDA); the client half lives in the cuda-gated
// `expert_tier_rdma` module because it lands bytes in the pinned arena.

use anyhow::{Context, Result, bail};

pub const STATUS_OK: u8 = 0;
pub const STATUS_ERR: u8 = 1;
/// Sentinel request that asks the server to close the connection.
pub const SHUTDOWN_MARKER: u32 = u32::MAX;

// ── Transport selection (sent by the client right after the manifest) ──
/// Two-sided TCP record streaming (the Stage-4 Phase-A path).
pub const MODE_TCP: u8 = 0;
/// One-sided RDMA READ over verbs (WS2 Phase B): the server publishes its
/// store's MRs and the client READs records directly into its arena.
pub const MODE_VERBS: u8 = 1;

/// The server's half of the verbs handshake: its QP identity plus, per MoE
/// layer, the base virtual address + rkey of that layer file's registered MR.
/// `remote_addr(layer, expert) = layers[layer].0 + expert * record_stride`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerbsServerParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    /// `(mr_base_addr, rkey)` for each MoE layer, layer-indexed.
    pub layers: Vec<(u64, u32)>,
}

/// The client's half: just its QP identity (its arena MR is local-only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerbsClientParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
}

impl VerbsServerParams {
    /// Wire form: `[u32 qpn][u32 psn][16 gid][u32 n_layers]{[u64 base][u32 rkey]}*`.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.qpn.to_le_bytes())?;
        w.write_all(&self.psn.to_le_bytes())?;
        w.write_all(&self.gid)?;
        w.write_all(&(self.layers.len() as u32).to_le_bytes())?;
        for (base, rkey) in &self.layers {
            w.write_all(&base.to_le_bytes())?;
            w.write_all(&rkey.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let qpn = read_u32(r)?;
        let psn = read_u32(r)?;
        let mut gid = [0u8; 16];
        r.read_exact(&mut gid).context("read server gid")?;
        let n = read_u32(r)? as usize;
        if n == 0 || n > 4096 {
            bail!("implausible verbs layer count: {n}");
        }
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let mut b8 = [0u8; 8];
            r.read_exact(&mut b8).context("read mr base")?;
            let base = u64::from_le_bytes(b8);
            let rkey = read_u32(r)?;
            layers.push((base, rkey));
        }
        Ok(Self {
            qpn,
            psn,
            gid,
            layers,
        })
    }
}

impl VerbsClientParams {
    /// Wire form: `[u32 qpn][u32 psn][16 gid]`.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.qpn.to_le_bytes())?;
        w.write_all(&self.psn.to_le_bytes())?;
        w.write_all(&self.gid)?;
        Ok(())
    }

    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let qpn = read_u32(r)?;
        let psn = read_u32(r)?;
        let mut gid = [0u8; 16];
        r.read_exact(&mut gid).context("read client gid")?;
        Ok(Self { qpn, psn, gid })
    }
}

fn read_u32<R: std::io::Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).context("read u32")?;
    Ok(u32::from_le_bytes(b))
}

/// Serialize a request: `(layer, expert)`.
pub fn encode_request(layer: u32, expert: u32) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[0..4].copy_from_slice(&layer.to_le_bytes());
    b[4..8].copy_from_slice(&expert.to_le_bytes());
    b
}

/// Parse a request buffer.
pub fn decode_request(b: &[u8; 8]) -> (u32, u32) {
    let layer = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let expert = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    (layer, expert)
}

#[cfg(unix)]
pub use server_impl::{RdmaConfig, serve};

#[cfg(unix)]
mod server_impl {
    use super::*;
    use crate::expert::ExpertKey;
    use crate::expert_pack::ExpertFileReader;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream, ToSocketAddrs};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    /// RDMA device selection for the verbs (`MODE_VERBS`) transport. Ignored for
    /// TCP clients. Defaults match the cabled GB10 CX7 link (`roceP2p1s0f1`,
    /// RoCEv2 GID index 3).
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

    /// Serve records from `dir` on `addr` until interrupted. One thread per
    /// connection. Blocking; intended to run as its own process (`atlas-expert-peer`).
    pub fn serve<A: ToSocketAddrs>(dir: &Path, addr: A, rdma: RdmaConfig) -> Result<()> {
        let reader = Arc::new(ExpertFileReader::open(dir)?);
        let manifest = serde_json::to_vec(reader.index())?;
        let dir: Arc<PathBuf> = Arc::new(dir.to_path_buf());
        let rdma = Arc::new(rdma);
        let listener = TcpListener::bind(addr).context("bind expert-peer listener")?;
        let local = listener.local_addr().ok();
        tracing::info!(
            "expert-peer serving {} ({} layers, {} experts, stride {}) on {:?} \
             (verbs dev={} gid={})",
            dir.display(),
            reader.index().num_moe_layers,
            reader.index().num_experts,
            reader.index().record_stride,
            local,
            rdma.dev,
            rdma.gid_idx,
        );
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("expert-peer accept error: {e}");
                    continue;
                }
            };
            let reader = reader.clone();
            let manifest = manifest.clone();
            let dir = dir.clone();
            let rdma = rdma.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_conn(stream, &reader, &manifest, &dir, &rdma) {
                    tracing::warn!("expert-peer connection ended: {e}");
                }
            });
        }
        Ok(())
    }

    fn handle_conn(
        mut stream: TcpStream,
        reader: &ExpertFileReader,
        manifest: &[u8],
        dir: &Path,
        rdma: &RdmaConfig,
    ) -> Result<()> {
        stream.set_nodelay(true).ok();
        // 1. Send the manifest.
        stream.write_all(&(manifest.len() as u32).to_le_bytes())?;
        stream.write_all(manifest)?;

        // 2. The client picks a transport.
        let mut mode = [0u8; 1];
        stream.read_exact(&mut mode).context("read transport mode")?;
        match mode[0] {
            MODE_TCP => serve_tcp(stream, reader),
            MODE_VERBS => serve_verbs(stream, reader, dir, rdma),
            other => bail!("client requested unknown transport mode {other}"),
        }
    }

    /// Two-sided record streaming (Phase A). The request loop is unchanged.
    fn serve_tcp(mut stream: TcpStream, reader: &ExpertFileReader) -> Result<()> {
        let stride = reader.index().record_stride as usize;
        let mut req = [0u8; 8];
        loop {
            if stream.read_exact(&mut req).is_err() {
                break; // client hung up
            }
            let (layer, expert) = decode_request(&req);
            if layer == SHUTDOWN_MARKER && expert == SHUTDOWN_MARKER {
                break;
            }
            match reader.read_record_raw(ExpertKey::new(layer, expert)) {
                Ok(rec) => {
                    debug_assert_eq!(rec.len(), stride);
                    stream.write_all(&[STATUS_OK])?;
                    stream.write_all(&rec)?;
                }
                Err(e) => {
                    tracing::warn!("expert-peer read {layer}/{expert}: {e}");
                    stream.write_all(&[STATUS_ERR])?;
                }
            }
        }
        Ok(())
    }

    #[cfg(not(atlas_rdma_verbs))]
    fn serve_verbs(
        _stream: TcpStream,
        _reader: &ExpertFileReader,
        _dir: &Path,
        _rdma: &RdmaConfig,
    ) -> Result<()> {
        bail!("client requested verbs transport but this peer was built without rdma-core");
    }

    /// One-sided RDMA READ (Phase B). The server `mmap`s + registers each layer
    /// file with REMOTE_READ, publishes the MRs' `{base, rkey}` + its QP params,
    /// connects to the client's QP, then goes idle: the client pulls records
    /// directly out of these MRs. The server CPU never touches a record byte.
    #[cfg(atlas_rdma_verbs)]
    fn serve_verbs(
        mut stream: TcpStream,
        reader: &ExpertFileReader,
        dir: &Path,
        rdma: &RdmaConfig,
    ) -> Result<()> {
        use crate::rdma_verbs::Verbs;

        let index = reader.index();
        let num_layers = index.num_moe_layers;
        // A per-connection PSN so successive clients don't collide.
        let psn: u32 = 0x424242 ^ (std::process::id().rotate_left(3));
        let mut verbs = Verbs::create(&rdma.dev, rdma.gid_idx, psn & 0xff_ffff)?;

        // mmap + register every layer file (REMOTE_READ). Keep the mappings alive
        // for the whole connection — the NIC DMAs out of them.
        let mut mmaps: Vec<Mmap> = Vec::with_capacity(num_layers as usize);
        let mut layers: Vec<(u64, u32)> = Vec::with_capacity(num_layers as usize);
        for l in 0..num_layers {
            let path = dir.join(index.file_name(l));
            let m = Mmap::open_ro(&path)
                .with_context(|| format!("mmap {}", path.display()))?;
            // SAFETY: the mapping covers `m.len` bytes at `m.addr` and outlives
            // `verbs` (mmaps dropped after verbs at scope end — see note below).
            let keys = unsafe { verbs.reg_mr(m.addr as *mut _, m.len, true)? };
            layers.push((m.addr as u64, keys.rkey));
            mmaps.push(m);
        }

        // Publish our QP + MR table, learn the client's QP, connect, ack.
        let sp = VerbsServerParams {
            qpn: verbs.qpn(),
            psn: verbs.psn(),
            gid: verbs.gid(),
            layers,
        };
        sp.write_to(&mut stream).context("send verbs server params")?;
        let cp = VerbsClientParams::read_from(&mut stream).context("read verbs client params")?;
        verbs.connect(cp.qpn, cp.psn, &cp.gid)?;
        stream.write_all(&[STATUS_OK]).context("send verbs ready ack")?;
        tracing::info!(
            "expert-peer verbs client connected (qpn {} -> {}, {} layer MRs)",
            verbs.qpn(),
            cp.qpn,
            num_layers,
        );

        // Idle until the client hangs up. All record movement is one-sided RDMA
        // READ initiated by the client; the server just holds the MRs open.
        let mut sink = [0u8; 8];
        loop {
            match stream.read(&mut sink) {
                Ok(0) => break, // client closed
                Ok(_) => {}     // ignore (shutdown marker or stray bytes)
                Err(_) => break,
            }
        }
        // Drop order: `verbs` (which dereg's the MRs) must fall before `mmaps`
        // are unmapped. Rust drops locals in reverse declaration order, and
        // `verbs` was declared before `mmaps`, so `mmaps` unmaps first — reorder
        // by dropping verbs explicitly first.
        drop(verbs);
        drop(mmaps);
        Ok(())
    }

    /// A read-only `mmap` of a whole file, unmapped on drop.
    #[cfg(atlas_rdma_verbs)]
    struct Mmap {
        addr: *mut libc::c_void,
        len: usize,
    }

    #[cfg(atlas_rdma_verbs)]
    impl Mmap {
        fn open_ro(path: &Path) -> Result<Self> {
            use std::os::fd::AsRawFd;
            let f = std::fs::File::open(path)?;
            let len = f.metadata()?.len() as usize;
            if len == 0 {
                bail!("empty layer file {}", path.display());
            }
            // SAFETY: fd is a valid open RO file; MAP_SHARED read mapping of `len`
            // bytes. The kernel keeps the mapping valid after the fd closes.
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    f.as_raw_fd(),
                    0,
                )
            };
            if addr == libc::MAP_FAILED {
                bail!(
                    "mmap {} failed: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                );
            }
            Ok(Self { addr, len })
        }
    }

    #[cfg(atlas_rdma_verbs)]
    impl Drop for Mmap {
        fn drop(&mut self) {
            // SAFETY: addr/len came from a successful mmap and are unmapped once.
            unsafe { libc::munmap(self.addr, self.len) };
        }
    }
}

/// Read the length-prefixed manifest from a freshly-connected stream and parse
/// it. Shared by the client (`expert_tier_rdma`).
#[cfg(unix)]
pub fn read_manifest<R: std::io::Read>(
    stream: &mut R,
) -> Result<crate::expert_pack::ExpertIndex> {
    let mut lenb = [0u8; 4];
    stream.read_exact(&mut lenb).context("read manifest length")?;
    let len = u32::from_le_bytes(lenb) as usize;
    if len == 0 || len > 16 * 1024 * 1024 {
        bail!("implausible peer manifest length: {len}");
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).context("read manifest json")?;
    let index: crate::expert_pack::ExpertIndex =
        serde_json::from_slice(&buf).context("parse peer manifest")?;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips() {
        let b = encode_request(7, 42);
        assert_eq!(decode_request(&b), (7, 42));
        let s = encode_request(SHUTDOWN_MARKER, SHUTDOWN_MARKER);
        assert_eq!(decode_request(&s), (SHUTDOWN_MARKER, SHUTDOWN_MARKER));
    }

    #[test]
    fn verbs_server_params_round_trip() {
        let sp = VerbsServerParams {
            qpn: 0x1234,
            psn: 0x00ab_cdef & 0xff_ffff,
            gid: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 178, 12,
            ],
            layers: vec![(0x7f00_0000_0000, 1001), (0x7f00_0100_0000, 1002)],
        };
        let mut buf = Vec::new();
        sp.write_to(&mut buf).unwrap();
        let back = VerbsServerParams::read_from(&mut &buf[..]).unwrap();
        assert_eq!(sp, back);
    }

    #[test]
    fn verbs_client_params_round_trip() {
        let cp = VerbsClientParams {
            qpn: 0x9999,
            psn: 0x0055_5555,
            gid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        };
        let mut buf = Vec::new();
        cp.write_to(&mut buf).unwrap();
        let back = VerbsClientParams::read_from(&mut &buf[..]).unwrap();
        assert_eq!(cp, back);
    }

    #[test]
    fn verbs_server_params_reject_absurd_layer_count() {
        // A corrupt/hostile count must Err, not attempt a huge allocation.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes()); // qpn
        buf.extend_from_slice(&2u32.to_le_bytes()); // psn
        buf.extend_from_slice(&[0u8; 16]); // gid
        buf.extend_from_slice(&99_999u32.to_le_bytes()); // n_layers (absurd)
        assert!(VerbsServerParams::read_from(&mut &buf[..]).is_err());
    }
}
