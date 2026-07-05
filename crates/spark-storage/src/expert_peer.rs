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
pub use server_impl::serve;

#[cfg(unix)]
mod server_impl {
    use super::*;
    use crate::expert::ExpertKey;
    use crate::expert_pack::ExpertFileReader;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream, ToSocketAddrs};
    use std::path::Path;
    use std::sync::Arc;

    /// Serve records from `dir` on `addr` until interrupted. One thread per
    /// connection. Blocking; intended to run as its own process (`atlas-expert-peer`).
    pub fn serve<A: ToSocketAddrs>(dir: &Path, addr: A) -> Result<()> {
        let reader = Arc::new(ExpertFileReader::open(dir)?);
        let manifest = serde_json::to_vec(reader.index())?;
        let listener = TcpListener::bind(addr).context("bind expert-peer listener")?;
        let local = listener.local_addr().ok();
        tracing::info!(
            "expert-peer serving {} ({} layers, {} experts, stride {}) on {:?}",
            dir.display(),
            reader.index().num_moe_layers,
            reader.index().num_experts,
            reader.index().record_stride,
            local
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
            std::thread::spawn(move || {
                if let Err(e) = handle_conn(stream, &reader, &manifest) {
                    tracing::warn!("expert-peer connection ended: {e}");
                }
            });
        }
        Ok(())
    }

    fn handle_conn(mut stream: TcpStream, reader: &ExpertFileReader, manifest: &[u8]) -> Result<()> {
        stream.set_nodelay(true).ok();
        // 1. Send the manifest.
        stream.write_all(&(manifest.len() as u32).to_le_bytes())?;
        stream.write_all(manifest)?;

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
}
