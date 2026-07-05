// SPDX-License-Identifier: AGPL-3.0-only
//
// RdmaTier — the peer weight-fetch tier (Stage 4, Phase A: TCP transport).
//
// Fetches expert records from an `atlas-expert-peer` over a socket straight into
// the pinned arena, then returns residency addresses pointing INTO that arena —
// exactly like `UmaArenaTier`, only the source is a peer instead of local NVMe.
// This proves the residency-tier abstraction (a peer as a fetch tier, distinct
// from EP sharding) and shares the arena machinery, so Phase B (one-sided
// RDMA_READ into the same arena — probe-confirmed GPU-readable zero-copy on
// GB10) is a transport swap with no re-architecture. See RESEARCH-RDMA-TIER.md.
//
// Transport note: over the RoCE Ethernet netdev this is TCP; the bytes still
// land in pinned LPDDR that the GPU reads at the same VA. The verbs path
// replaces the `read_exact` below with an `IBV_WR_RDMA_READ` completion.

use std::io::{Read, Write};
use std::net::TcpStream;

use anyhow::{Context, Result, bail};

use crate::expert::{ExpertKey, ExpertLayout, ExpertRecordSpec};
use crate::expert_arena::ExpertArena;
use crate::expert_peer::{STATUS_OK, encode_request, read_manifest};
use crate::expert_tier::{ArenaSlot, ExpertResidency, ExpertTier, TierKind, residency_from};

pub struct RdmaTier {
    stream: TcpStream,
    arena: ExpertArena,
    spec: ExpertRecordSpec,
    layout: ExpertLayout,
    healthy: bool,
}

impl RdmaTier {
    /// Connect to a peer at `addr`, receive its manifest, and allocate the arena
    /// ring. The peer's `ExpertIndex` geometry defines the record stride, so the
    /// arena matches the remote store exactly.
    pub fn connect(addr: &str, num_slabs: u32, slots_per_slab: u32) -> Result<Self> {
        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect expert peer {addr}"))?;
        stream.set_nodelay(true).ok();
        let index = read_manifest(&mut stream)?;
        let spec = index.spec();
        let layout = index.layout();
        let arena = ExpertArena::new(num_slabs, slots_per_slab, layout.record_stride as usize)?;
        tracing::info!(
            "RdmaTier(TCP) connected to {addr}: {} layers, {} experts, stride {}",
            index.num_moe_layers,
            index.num_experts,
            layout.record_stride
        );
        Ok(Self {
            stream,
            arena,
            spec,
            layout,
            healthy: true,
        })
    }

    pub fn arena(&self) -> &ExpertArena {
        &self.arena
    }
}

impl ExpertTier for RdmaTier {
    fn fetch(&mut self, key: ExpertKey, slot: ArenaSlot, _stream: u64) -> Result<ExpertResidency> {
        let stride = self.layout.record_stride as usize;
        let host = self.arena.slot_host_ptr(slot.slab, slot.slot)?;

        // Request the record; peer replies [status][stride bytes].
        if let Err(e) = self.stream.write_all(&encode_request(key.layer, key.expert)) {
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
        let record = unsafe { std::slice::from_raw_parts(host, stride) };
        let dev_va = self.arena.slot_dev_va(slot.slab, slot.slot)?;
        residency_from(&self.spec, record, dev_va, key)
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
