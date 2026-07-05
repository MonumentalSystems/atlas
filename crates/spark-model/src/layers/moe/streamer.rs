// SPDX-License-Identifier: AGPL-3.0-only
//
// Model-wide expert streamer shared by every MoE layer (Arc).
//
// Holds one `ExpertTier` (Posix bounce oracle or UMA zero-copy arena) plus the
// store geometry. Stage 2 uses a BLOCKING fetch (no compute overlap): each MoE
// layer, just before its routed grouped GEMM, fetches its local experts into
// the ring slab for that layer and patches the transposed pointer tables to the
// fetched addresses. The `Mutex` serializes tier access; prefill is
// single-threaded per model so there is no contention, only Send/Sync hygiene.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use atlas_core::config::ModelConfig;

use spark_storage::expert::ExpertKey;
use spark_storage::expert_tier::{ArenaSlot, ExpertResidency, ExpertTier, TierKind, open_tier};
use spark_storage::ExpertIndex;

// `index`/`slots_per_slab`/`kind` + several accessors are consumed by Stage 3
// (async prefetch bounds) / Stage 4 (RDMA health + logging); kept as the stable
// streamer API surface.
#[allow(dead_code)]
pub struct ExpertStreamerShared {
    tier: Mutex<Box<dyn ExpertTier>>,
    index: ExpertIndex,
    num_slabs: u32,
    slots_per_slab: u32,
    kind: TierKind,
}

impl ExpertStreamerShared {
    /// Open the store named by `config.expert_store_dir` with the configured
    /// backend and size the arena ring. `slots_per_slab` = the count of LOCAL
    /// experts (EP-scoped); `num_slabs` = `expert_arena_layers` (0 => one slab
    /// per MoE layer, i.e. fully resident ring, no eviction).
    pub fn open(config: &ModelConfig) -> Result<Self> {
        let dir: &Path = config
            .expert_store_dir
            .as_deref()
            .context("expert_streaming set but --stream-experts dir is None")?;
        let index = ExpertIndex::load(dir)?;
        if index.num_experts as usize != config.num_experts {
            bail!(
                "expert store has {} experts but model config has {}",
                index.num_experts,
                config.num_experts
            );
        }
        let (lo, hi) = config.local_expert_range();
        let slots_per_slab = (hi - lo) as u32;
        if slots_per_slab == 0 {
            bail!("expert streaming: local_expert_range is empty");
        }
        let num_moe_layers = index.num_moe_layers;
        let num_slabs = if config.expert_arena_layers == 0 {
            num_moe_layers
        } else {
            (config.expert_arena_layers as u32).clamp(1, num_moe_layers)
        };
        let tier = open_tier(&config.expert_backend, dir, num_slabs, slots_per_slab)?;
        let kind = tier.kind();
        tracing::info!(
            "expert streamer: backend={:?} store={} experts={} local={} slabs={} (arena_layers={}) \
             record_stride={}",
            kind,
            dir.display(),
            index.num_experts,
            slots_per_slab,
            num_slabs,
            config.expert_arena_layers,
            index.record_stride,
        );
        Ok(Self {
            tier: Mutex::new(tier),
            index,
            num_slabs,
            slots_per_slab,
            kind,
        })
    }

    #[allow(dead_code)]
    pub fn num_slabs(&self) -> u32 {
        self.num_slabs
    }
    #[allow(dead_code)]
    pub fn num_moe_layers(&self) -> u32 {
        self.index.num_moe_layers
    }
    #[allow(dead_code)]
    pub fn slots_per_slab(&self) -> u32 {
        self.slots_per_slab
    }
    #[allow(dead_code)]
    pub fn kind(&self) -> TierKind {
        self.kind
    }

    /// Fetch `expert` of dense MoE layer `dense_layer` into the ring slab for
    /// that layer at `local_slot`, returning the six sub-buffer device
    /// addresses to patch into the pointer tables.
    pub fn fetch(
        &self,
        dense_layer: u32,
        expert: u32,
        local_slot: u32,
        stream: u64,
    ) -> Result<ExpertResidency> {
        let slab = dense_layer % self.num_slabs;
        let slot = ArenaSlot::new(slab, local_slot);
        let key = ExpertKey::new(dense_layer, expert);
        let mut tier = self.tier.lock().expect("expert streamer tier mutex poisoned");
        tier.fetch(key, slot, stream)
    }
}
