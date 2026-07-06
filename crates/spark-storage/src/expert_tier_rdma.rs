// SPDX-License-Identifier: AGPL-3.0-only
//
// RdmaTier — the peer weight-fetch tier (Stage 4).
//
// Fetches expert records from an `atlas-expert-peer` straight into the pinned
// arena, then returns residency addresses pointing INTO that arena — exactly
// like `UmaArenaTier`, only the source is a peer instead of local NVMe. Two
// transports share the tier and the arena machinery:
//
//   * `Transport::Tcp`   — Phase A: two-sided record streaming. The peer `pread`s
//     each record and writes it back over the socket; simple, bit-identical, but
//     the peer CPU is busy and single-stream bandwidth is ~5 GB/s.
//   * `Transport::Verbs` — Phase B: one-sided `IBV_WR_RDMA_READ`. The client
//     pulls each record directly out of the peer's registered store MR into the
//     arena slot with zero peer-CPU involvement (~14 GB/s, measured). This is a
//     pure transport swap — the bytes still land in the same pinned LPDDR that
//     the GPU reads at the same VA, and `residency_from` + the record header's
//     identity check catch any misplacement, so it cannot change a GEMM byte.
//
// The transport is chosen by the `--expert-backend` value: `rdma` = TCP,
// `rdma-verbs` = one-sided verbs. Device/GID for verbs come from
// `$ATLAS_EXPERT_RDMA_DEV` (default `roceP2p1s0f1`) / `$ATLAS_EXPERT_RDMA_GID`
// (default 3, the RoCEv2 IPv4 GID on GB10/CX7).

use std::io::{Read, Write};
use std::net::TcpStream;

use anyhow::{Context, Result, bail};

use crate::expert::{ExpertKey, ExpertLayout, ExpertRecordSpec};
use crate::expert_arena::ExpertArena;
use crate::expert_peer::{MODE_TCP, STATUS_OK, encode_request, read_manifest};
use crate::expert_tier::{ArenaSlot, ExpertResidency, ExpertTier, TierKind, residency_from};

/// The active peer transport. `Verbs` only exists where the shim is compiled.
/// The verbs transport is N-rail: one `Rail` per CX7 adapter, and a fetch is
/// striped to `rail = expert % n_rails` (single-rail = the unchanged path).
enum Transport {
    Tcp,
    #[cfg(atlas_rdma_verbs)]
    Verbs(Vec<Rail>),
}

/// One-sided verbs state for a single rail: the QP, this rail's arena MR lkey,
/// and the per-layer remote MR `{base, rkey}` table the peer published for it.
/// The base VA is shared across rails (the peer mmaps each layer once); only the
/// rkey (and QP/NIC) differ per rail.
#[cfg(atlas_rdma_verbs)]
struct Rail {
    verbs: crate::rdma_verbs::Verbs,
    arena_lkey: u32,
    /// `(remote_base_addr, rkey)` per MoE layer, layer-indexed.
    layers: Vec<(u64, u32)>,
}

pub struct RdmaTier {
    stream: TcpStream,
    // `transport` is declared BEFORE `arena` so it drops first: the verbs rails
    // hold MRs registered over the arena's pinned pages, so their `ibv_dereg_mr`
    // must run before the arena frees those pages. (Struct fields drop in
    // declaration order.) With N rails this is load-bearing — reverting the order
    // would dereg N MRs over freed memory.
    transport: Transport,
    arena: ExpertArena,
    spec: ExpertRecordSpec,
    layout: ExpertLayout,
    healthy: bool,
}

impl RdmaTier {
    /// Connect to a peer at `addr`, receive its manifest, allocate the arena, and
    /// bring up the chosen transport. The peer's `ExpertIndex` geometry defines
    /// the record stride, so the arena matches the remote store exactly.
    pub fn connect(
        addr: &str,
        num_slabs: u32,
        slots_per_slab: u32,
        use_verbs: bool,
    ) -> Result<Self> {
        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect expert peer {addr}"))?;
        stream.set_nodelay(true).ok();
        let index = read_manifest(&mut stream)?;
        let spec = index.spec();
        let layout = index.layout();
        let arena = ExpertArena::new(num_slabs, slots_per_slab, layout.record_stride as usize)?;

        let transport = if use_verbs {
            #[cfg(atlas_rdma_verbs)]
            {
                connect_verbs(&mut stream, &arena, index.num_moe_layers)?
            }
            // Built without rdma-core (no C shim) — verbs is unavailable; the
            // TCP `rdma` backend still works. Keeps the crate compiling under
            // ATLAS_SKIP_BUILD / hosts without libibverbs.
            #[cfg(not(atlas_rdma_verbs))]
            {
                let _ = &arena;
                bail!(
                    "--expert-backend rdma-verbs needs a build with rdma-core \
                     (atlas_rdma_verbs cfg); use --expert-backend rdma (TCP) instead"
                );
            }
        } else {
            stream
                .write_all(&[MODE_TCP])
                .context("send TCP transport mode")?;
            Transport::Tcp
        };

        let label = match &transport {
            Transport::Tcp => "TCP".to_string(),
            #[cfg(atlas_rdma_verbs)]
            Transport::Verbs(rails) => {
                format!("verbs (one-sided RDMA READ, {} rail(s))", rails.len())
            }
        };
        tracing::info!(
            "RdmaTier[{label}] connected to {addr}: {} layers, {} experts, stride {}",
            index.num_moe_layers,
            index.num_experts,
            layout.record_stride
        );
        Ok(Self {
            stream,
            arena,
            spec,
            layout,
            transport,
            healthy: true,
        })
    }

    pub fn arena(&self) -> &ExpertArena {
        &self.arena
    }

    /// Two-sided TCP fetch: request the record, read `[status][stride bytes]`
    /// straight into the pinned slot.
    fn fetch_tcp(&mut self, key: ExpertKey, host: *mut u8, stride: usize) -> Result<()> {
        if let Err(e) = self
            .stream
            .write_all(&encode_request(key.layer, key.expert))
        {
            self.healthy = false;
            return Err(e).with_context(|| format!("peer request {:?}", key));
        }
        let mut status = [0u8; 1];
        if let Err(e) = self.stream.read_exact(&mut status) {
            self.healthy = false;
            return Err(e).with_context(|| format!("peer status {:?}", key));
        }
        if status[0] != STATUS_OK {
            bail!("peer returned error status {} for {:?}", status[0], key);
        }
        // Land the record bytes DIRECTLY into the pinned, GPU-addressable slot.
        // SAFETY: `host` points at a `stride`-byte slot inside the pinned arena.
        let dst = unsafe { std::slice::from_raw_parts_mut(host, stride) };
        if let Err(e) = self.stream.read_exact(dst) {
            self.healthy = false;
            return Err(e).with_context(|| format!("peer payload {:?}", key));
        }
        Ok(())
    }
}

#[cfg(atlas_rdma_verbs)]
fn env_u32(k: &str, default: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Bring up the one-sided verbs transport: create N rails, register the arena on
/// each, exchange per-rail QP params over the TCP control channel, connect
/// INIT->RTR->RTS, await the ack. Dual-rail is env-driven (ATLAS_EXPERT_DUAL_RAIL
/// =1): rail 0 = ATLAS_EXPERT_RDMA_DEV/GID (the existing single-rail defaults),
/// rail 1 = ATLAS_EXPERT_RAIL2_DEV/GID (default rocep1s0f1 / 3). Single-rail is
/// the default and is byte-for-byte the previous path.
#[cfg(atlas_rdma_verbs)]
fn connect_verbs(
    stream: &mut TcpStream,
    arena: &ExpertArena,
    num_layers: u32,
) -> Result<Transport> {
    use crate::expert_peer::{MODE_VERBS, VerbsClientParams, read_server_rails};
    use crate::rdma_verbs::Verbs;

    stream
        .write_all(&[MODE_VERBS])
        .context("send verbs transport mode")?;

    // Rail 0 from the expert env (the cabled CX7 link); rail 1 from the expert
    // rail-2 env. Dual-rail only when ATLAS_EXPERT_DUAL_RAIL=1.
    let dev0 = std::env::var("ATLAS_EXPERT_RDMA_DEV").unwrap_or_else(|_| "roceP2p1s0f1".into());
    let gid0 = env_u32("ATLAS_EXPERT_RDMA_GID", 3);
    let dev1 = std::env::var("ATLAS_EXPERT_RAIL2_DEV").unwrap_or_else(|_| "rocep1s0f1".into());
    let gid1 = env_u32("ATLAS_EXPERT_RAIL2_GID", 3);
    let dual = std::env::var("ATLAS_EXPERT_DUAL_RAIL").ok().as_deref() == Some("1");
    let rail_devs: Vec<(String, u32)> = if dual {
        vec![(dev0, gid0), (dev1, gid1)]
    } else {
        vec![(dev0, gid0)]
    };
    let n_rails = rail_devs.len();

    // Tell the peer how many rails we want to stripe across.
    stream.write_all(&[n_rails as u8]).context("send n_rails")?;

    // Create each rail's QP and register the WHOLE arena as its READ landing MR
    // (LOCAL_WRITE only). The N MRs pin the SAME arena pages (one lkey per rail).
    // SAFETY: the arena's pinned buffer lives as long as the tier (and thus every
    // MR); base_ptr()/total_bytes() describe exactly that allocation.
    let mut verbs_rails: Vec<Verbs> = Vec::with_capacity(n_rails);
    let mut arena_lkeys: Vec<u32> = Vec::with_capacity(n_rails);
    for (dev, gid) in &rail_devs {
        let psn = rand::random::<u32>() & 0xff_ffff;
        let mut v = Verbs::create(dev, *gid, psn)?;
        let keys = unsafe { v.reg_mr(arena.base_ptr(), arena.total_bytes(), false)? };
        arena_lkeys.push(keys.lkey);
        verbs_rails.push(v);
    }

    // Peer publishes N per-rail QP + per-layer MR tables; validate each rail's
    // layer count against the manifest.
    let server = read_server_rails(stream, n_rails).context("read verbs server params")?;
    for sp in &server {
        if sp.layers.len() != num_layers as usize {
            bail!(
                "verbs peer published {} layer MRs but manifest has {num_layers} MoE layers",
                sp.layers.len()
            );
        }
    }

    // Reply with each rail's client QP, then connect each rail to its peer rail.
    stream
        .write_all(&[n_rails as u8])
        .context("send client n_rails")?;
    for v in &verbs_rails {
        VerbsClientParams {
            qpn: v.qpn(),
            psn: v.psn(),
            gid: v.gid(),
        }
        .write_to(stream)
        .context("send verbs client params")?;
    }
    let mut rails: Vec<Rail> = Vec::with_capacity(n_rails);
    for ((mut v, arena_lkey), sp) in verbs_rails.into_iter().zip(arena_lkeys).zip(server) {
        v.connect(sp.qpn, sp.psn, &sp.gid)?;
        rails.push(Rail {
            verbs: v,
            arena_lkey,
            layers: sp.layers,
        });
    }

    let mut ack = [0u8; 1];
    stream
        .read_exact(&mut ack)
        .context("read verbs ready ack")?;
    if ack[0] != STATUS_OK {
        bail!("verbs peer refused connection (ack {})", ack[0]);
    }
    Ok(Transport::Verbs(rails))
}

impl ExpertTier for RdmaTier {
    fn fetch(&mut self, key: ExpertKey, slot: ArenaSlot, _stream: u64) -> Result<ExpertResidency> {
        let stride = self.layout.record_stride as usize;
        let host = self.arena.slot_host_ptr(slot.slab, slot.slot)?;
        let dev_va = self.arena.slot_dev_va(slot.slab, slot.slot)?;
        let spec = self.spec; // Copy — release the field borrow before matching.

        match &mut self.transport {
            Transport::Tcp => {
                self.fetch_tcp(key, host, stride)?;
            }
            #[cfg(atlas_rdma_verbs)]
            Transport::Verbs(rails) => {
                // Stripe the fetch onto rail = expert % n_rails. Single-rail
                // (n == 1) => always rail 0, the unchanged path.
                let ri = (key.expert as usize) % rails.len();
                let rail = &mut rails[ri];
                let (base, rkey) = *rail.layers.get(key.layer as usize).with_context(|| {
                    format!("verbs: no layer MR for layer {} ({:?})", key.layer, key)
                })?;
                let remote_addr = base + (key.expert as u64) * (stride as u64);
                let wr_id = ((key.layer as u64) << 32) | (key.expert as u64);
                // SAFETY: `host` is a `stride`-byte slot inside this rail's arena
                // MR (arena_lkey); remote_addr/rkey address the peer's layer MR on
                // the SAME rail.
                let post = unsafe {
                    rail.verbs.post_read(
                        host as *mut std::ffi::c_void,
                        rail.arena_lkey,
                        remote_addr,
                        rkey,
                        stride as u32,
                        wr_id,
                    )
                };
                if let Err(e) = post {
                    self.healthy = false;
                    return Err(e).with_context(|| format!("verbs post_read {:?}", key));
                }
                match rail.verbs.poll() {
                    Ok(got) if got == wr_id => {}
                    Ok(got) => {
                        self.healthy = false;
                        bail!("verbs completion wr_id {got:#x} != expected {wr_id:#x} ({key:?})");
                    }
                    Err(e) => {
                        self.healthy = false;
                        return Err(e).with_context(|| format!("verbs poll {:?}", key));
                    }
                }
            }
        }

        // SAFETY: the slot now holds `stride` valid bytes (landed by TCP or RDMA).
        let record = unsafe { std::slice::from_raw_parts(host, stride) };
        residency_from(&spec, record, dev_va, key)
    }

    fn kind(&self) -> TierKind {
        TierKind::Rdma
    }

    /// Link health: false after any transport error so the streamer can fall
    /// back to the local NVMe UMA tier (graceful degradation on CX7 flap).
    fn healthy(&self) -> bool {
        self.healthy
    }
}
