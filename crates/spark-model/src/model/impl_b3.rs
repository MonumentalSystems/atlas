// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn run_mtp_propose_inner(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(Vec::new()),
        };
        // ATLAS_DFLASH_DEBUG_DUMP_FULL=1: emit the full token sequence
        // ONCE so a Python reference can run the SAME tokens through HF
        // transformers and dump matching hidden-state captures.
        static TOKENS_DUMPED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !TOKENS_DUMPED.load(std::sync::atomic::Ordering::Relaxed)
            && std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                .ok()
                .as_deref()
                == Some("1")
        {
            let tokens_json = serde_json::json!({
                "prompt_len": position - seq.tokens.len() + seq.tokens.len(),
                "position": position,
                "last_token": token,
                "all_tokens": seq.tokens.clone(),
                "generated_tokens": seq.tokens.iter().skip(seq.prompt_len).copied().collect::<Vec<u32>>(),
            });
            if let Err(e) = std::fs::write(
                "/tmp/atlas_tokens.json",
                serde_json::to_string_pretty(&tokens_json).unwrap_or_default(),
            ) {
                tracing::warn!("DFLASH DUMP_FULL: tokens write failed: {e}");
            } else {
                tracing::info!(
                    "DFLASH DUMP_FULL: wrote /tmp/atlas_tokens.json (position={}, all_tokens.len()={}, prompt_len={})",
                    position,
                    seq.tokens.len(),
                    seq.prompt_len,
                );
            }
            TOKENS_DUMPED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let stream = self.gpu.default_stream();
        let draft_embed_target = None;
        // MTP loads ALL experts on every rank (no EP filtering), so its MoE
        // output is already complete — no all_reduce needed. Passing comm: None
        // prevents MoeLayer::forward() from doubling the output via SUM.
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
        };
        let prop_state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No proposer state for sequence"))?;
        proposer.propose(
            token,
            self.mtp_hidden_save,
            position,
            num_drafts,
            prop_state.as_mut(),
            &ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            self.dflash_hidden_save,
        )
    }

    /// Borrow the GPU backend for post-construction wiring (e.g. installing
    /// a DFlash proposer that needs to allocate paged KV caches against the
    /// same GPU the target uses).
    pub fn gpu_backend(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
    }

    /// Borrow the model config for post-construction wiring (e.g. building the
    /// DeepSeek-V4 MTP proposer, which needs `hidden_size` / `kv_lora_rank` /
    /// `qk_rope_head_dim` to size its private MLA KV cache).
    pub fn config_ref(&self) -> &ModelConfig {
        &self.config
    }

    /// Install a DFlash drafter as the active proposer, replacing whatever
    /// MTP proposer (if any) `TransformerModel::new` built. The target's
    /// hidden-state capture buffer is already allocated when the config's
    /// `dflash_capture_layers` is non-empty (factory.rs populates it before
    /// construction), so this method only swaps the proposer slot.
    ///
    /// Mutually exclusive with `--speculative` MTP at the CLI level
    /// (clap `conflicts_with`); this method does not enforce that — the
    /// caller is expected to have validated the flag combination already.
    pub fn set_dflash_proposer(&mut self, proposer: std::sync::Arc<dyn DraftProposer>) {
        if self.proposer.is_some() {
            tracing::info!("DFlash: replacing existing MTP proposer with BlockDiffusionDraftHead");
        }
        self.proposer = Some(proposer);
    }

    /// Install a startup-static LoRA adapter (post-construction, mirroring
    /// [`Self::set_dflash_proposer`]). Walks the model layers by GLOBAL
    /// index — `LoraWeights.layers` is indexed the same way — and copies
    /// each adapted layer's K/V/O (+ optional gate/up/down) pairs into the
    /// `Qwen3AttentionLayer` (which routes FFN pairs into its dense FFN
    /// component). M0: layers only STORE the adapter; base output is
    /// unchanged until the M1 compute insertions read it.
    pub fn set_lora_weights(&mut self, lora: Option<crate::lora::LoraWeights>) -> Result<()> {
        if let Some(ref lw) = lora {
            // eager-on-rotate: >1 resident adapter, or the rotate/peer env,
            // ARMS rotation and forces eager decode (see `lora_rotatable`).
            self.lora_rotatable = lw.slots.len() > 1
                || crate::lora::lora_rotate_env()
                || crate::lora::lora_peer_env().is_some();
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            // Clone the active slot's pairs (small; LoraPair is Copy) so the
            // install walk can hold a shared borrow while it &mut-borrows
            // `self.layers`.
            let active = lw.active_layers().to_vec();
            let installed = self.install_lora_layers(&active, kernels)?;
            tracing::info!(
                "LoRA: {} adapter(s) resident [{}], active '{}' installed on \
                 {installed} layers (r={}, max_rank={}, max_loras={}, \
                 pool={:.1} MiB, rotatable={})",
                lw.slots.len(),
                lw.adapter_names().join(", "),
                lw.name,
                lw.adapter_config.r,
                lw.max_rank,
                lw.max_loras,
                lw.pool_bytes as f64 / (1024.0 * 1024.0),
                self.lora_rotatable,
            );
        }
        self.lora = lora;
        Ok(())
    }

    /// Install one slot's per-layer pairs onto the layer structs (the shared
    /// walk used by both initial install and runtime rotation). `layers` is
    /// GLOBAL-layer-indexed. Returns the number of layers installed.
    fn install_lora_layers(
        &mut self,
        layers: &[Option<crate::lora::LoraLayerWeights>],
        kernels: ops::lora_delta::LoraKernels,
    ) -> Result<usize> {
        let mut installed = 0usize;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let Some(layer_weights) = layers.get(idx).and_then(|o| o.as_ref()) else {
                continue;
            };
            let attn = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<crate::layers::Qwen3AttentionLayer>())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "LoRA: adapted layer {idx} is not a Qwen3AttentionLayer \
                         (loader/adapter layer-type mismatch)"
                    )
                })?;
            let attn_weights = ops::lora_delta::LoraAttnWeights {
                k: layer_weights.k_proj,
                v: layer_weights.v_proj,
                o: layer_weights.o_proj,
                kernels,
            };
            let ffn_weights = if layer_weights.gate_proj.is_some()
                || layer_weights.up_proj.is_some()
                || layer_weights.down_proj.is_some()
            {
                Some(ops::lora_delta::LoraFfnWeights {
                    gate: layer_weights.gate_proj,
                    up: layer_weights.up_proj,
                    down: layer_weights.down_proj,
                    kernels,
                })
            } else {
                None
            };
            attn.set_lora_weights(attn_weights, ffn_weights)?;
            installed += 1;
        }
        Ok(installed)
    }

    /// Runtime adapter rotation (eager-on-rotate). Selects the resident
    /// adapter named `name` as ACTIVE: re-points every layer's LoraPair (a/b
    /// DevicePtr + rank/scale) to that slot's sub-region, then clears the
    /// decode-graph caches defensively (empty under forced eager). MUST be
    /// called at a scheduler QUIESCENT point (no in-flight decode reading the
    /// old slot). Graph-safety rests on `lora_rotatable` forcing eager decode
    /// — this method never re-captures a graph.
    pub fn rotate_lora_to(&mut self, name: &str) -> Result<()> {
        let slot = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA rotation: no adapter loaded"))?;
            lw.slot_of(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "LoRA rotation: adapter '{name}' is not resident (have [{}])",
                    lw.adapter_names().join(", ")
                )
            })?
        };
        if !self.lora_rotatable {
            // A single startup adapter with no rotation env is baked into the
            // decode graph; re-pointing would be replayed stale. Refuse rather
            // than silently mis-serve.
            anyhow::bail!(
                "LoRA rotation not armed (single adapter, ATLAS_LORA_ROTATE unset); \
                 set ATLAS_LORA_ROTATE=1 (forces eager decode) to rotate at runtime"
            );
        }
        // Re-point onto the new active slot.
        let (layers, active_name, r) = {
            let lw = self.lora.as_mut().unwrap();
            lw.active = slot;
            lw.name = lw.slots[slot].name.clone();
            lw.adapter_config = lw.slots[slot].adapter_config.clone();
            (
                lw.slots[slot].layers.clone(),
                lw.name.clone(),
                lw.adapter_config.r,
            )
        };
        let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
        let installed = self.install_lora_layers(&layers, kernels)?;
        // Defensive: drop any captured decode graphs so a stale-pointer replay
        // is impossible even if `lora_rotatable` were ever mis-derived. Under
        // forced eager these are already empty.
        self.decode_graph.lock().clear();
        self.batch_decode_graphs.lock().clear();
        tracing::info!(
            "LoRA rotation → slot {slot} '{active_name}' (r={r}) re-installed on \
             {installed} layers"
        );
        Ok(())
    }

    /// RDMA-swap the adapter named `adapter_name` (staged on `$ATLAS_LORA_PEER`
    /// at `adapter_id`) INTO pool `slot`, in place, then make it that slot's
    /// resident adapter. Byte-identical to a disk pack (the loader does the same
    /// F16/F32→BF16 convert + B row-repack). MUST be called at a scheduler
    /// QUIESCENT point (no in-flight decode reading `slot`). Re-zeroes the slot
    /// sub-region first (a reused slot may hold the prior adapter's bytes), then
    /// rebuilds the slot's `LoraLayerWeights` with the NEW adapter's r/scale —
    /// re-installing if the swapped slot is currently active. Requires rotation
    /// armed (`ATLAS_LORA_ROTATE`/`$ATLAS_LORA_PEER`) so decode is eager.
    #[cfg(feature = "cuda")]
    pub fn swap_lora_slot_from_peer(
        &mut self,
        peer_addr: &str,
        adapter_id: &str,
        adapter_name: &str,
        slot: usize,
        peft: atlas_core::config::PeftAdapterConfig,
    ) -> Result<()> {
        use crate::lora::rdma_stage;

        let (pool, max_rank, max_loras) = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA RDMA swap: no adapter pool loaded"))?;
            (lw.pool, lw.max_rank, lw.max_loras)
        };
        if !self.lora_rotatable {
            anyhow::bail!(
                "LoRA RDMA swap needs rotation armed (set $ATLAS_LORA_PEER or \
                 ATLAS_LORA_ROTATE=1 so decode runs eager)"
            );
        }
        if slot >= max_loras {
            anyhow::bail!("LoRA RDMA swap: slot {slot} >= max_loras {max_loras}");
        }

        // 1) Fetch manifest + build landing targets (classify + slot offsets).
        let manifest = rdma_stage::fetch_adapter_manifest(peer_addr, adapter_id)?;
        let targets =
            rdma_stage::build_land_targets(&manifest, &self.config, pool, slot, max_rank)?;

        // 2) Re-zero the slot sub-region (in-place reload of a dirty slot),
        //    then RDMA-land the adapter's A/B into it.
        let slot_bytes = rdma_stage::slot_bytes(&self.config, max_rank);
        let slot_base = DevicePtr(pool.0 + (slot * slot_bytes) as u64);
        self.gpu.memset(slot_base, 0, slot_bytes)?;
        let loader =
            spark_storage::RdmaLoraLoader::new(peer_addr.to_string(), adapter_id.to_string());
        loader.stage_into_slot(self.gpu.as_ref(), &targets)?;

        // 3) Rebuild the slot's per-layer pairs (new r/scale), stamp the slot.
        let layers =
            rdma_stage::rebuild_slot_layers(&targets, &self.config, &peft, pool, slot, max_rank)?;
        {
            let lw = self.lora.as_mut().unwrap();
            let s = lw
                .slots
                .get_mut(slot)
                .ok_or_else(|| anyhow::anyhow!("LoRA RDMA swap: slot {slot} not resident"))?;
            s.name = adapter_name.to_string();
            s.adapter_config = peft;
            s.layers = layers;
        }

        // 4) If the swapped slot is active, re-install onto the layer structs.
        let active = self.lora.as_ref().unwrap().active;
        if active == slot {
            let installed_layers = self.lora.as_ref().unwrap().slots[slot].layers.clone();
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&installed_layers, kernels)?;
            self.lora.as_mut().unwrap().name = adapter_name.to_string();
            self.decode_graph.lock().clear();
            self.batch_decode_graphs.lock().clear();
        }
        tracing::info!(
            "LoRA RDMA swap: '{adapter_name}' landed in slot {slot} \
             ({} targets, active_slot={active})",
            targets.len()
        );
        Ok(())
    }

    /// DFlash prefill capture: copy `proc_count` tokens × hidden_size BF16
    /// from `self.buffers.hidden_states()` (filled by the just-completed
    /// prefill layer) into the per-sequence DFlash accumulator. Called
    /// inside the prefill layer loop after each layer. No-op when:
    ///   - DFlash is disabled (capture_layers empty)
    ///   - `layer_idx` is not in `dflash_capture_layers`
    ///   - The seq has no `DflashProposerState`
    ///   - Rank > 0 under EP/TP (drafter is rank-0 only)
    ///
    /// Layout: writes `hidden[t]` BF16 into
    /// `acc[(chunk_start + t) * 5 * h + slot_idx * h]` for each t.
    /// Per-layer call performs `proc_count` strided d2d_async copies —
    /// at typical prefill of 128–4096 tokens × 5 capture layers, total
    /// 640–20480 launches per prefill. Acceptable launch overhead for
    /// first land; replace with a strided-scatter kernel if profiling
    /// shows it's a bottleneck.
    pub(super) fn try_dflash_prefill_capture_layer(
        &self,
        seq: &mut crate::traits::SequenceState,
        layer_idx: usize,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let slot_idx = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let dstate = match seq.proposer_state.as_mut() {
            Some(ps) => match ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
            {
                Some(s) => s,
                None => return Ok(()),
            },
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();
        let acc_base = dstate.ctx_hidden_acc;
        let max_ctx = dstate.max_ctx_len;
        let src_base = self.buffers.hidden_states();
        for t in 0..proc_count {
            let abs_pos = chunk_start + t;
            if abs_pos >= max_ctx {
                break; // accumulator full; drop later positions
            }
            let src = src_base.offset(t * h * bf16);
            let dst_offset = abs_pos * n_capture * h * bf16 + slot_idx * h * bf16;
            self.gpu
                .copy_d2d_async(src, acc_base.offset(dst_offset), h * bf16, stream)?;
        }
        Ok(())
    }

    /// After prefill completes, advance the seq's DFlash `ctx_len` to
    /// `chunk_start + proc_count` so the drafter sees all captured prompt
    /// positions on the first propose() call.
    pub(super) fn update_dflash_ctx_len_after_prefill(
        &self,
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        proc_count: usize,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        if let Some(ps) = seq.proposer_state.as_mut()
            && let Some(dstate) = ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        {
            let new_len = (chunk_start + proc_count).min(dstate.max_ctx_len);
            dstate.ctx_len = new_len;
        }
        Ok(())
    }

    /// DFlash 5-layer hidden capture. Called inside each per-layer loop after
    /// `layer.decode(...)` returns. No-op when DFlash is disabled (the buffer
    /// is `None`) or when `layer_idx` is not in `dflash_capture_layers`.
    ///
    /// Captures only the latest-decoded-token's hidden, matching the
    /// `save_hidden_for_mtp` semantics. The `token_idx` argument selects
    /// which row of `self.buffers.hidden_states()` to read — pass 0 for the
    /// single-token decode path.
    ///
    /// Under EP/TP world > 1: only rank 0 owns the drafter (replicated, not
    /// sharded — same pattern as MTP under EP — see model.rs:7232 comment),
    /// so non-rank-0 ranks skip the capture. The captured hiddens are
    /// post-TP-allreduce so semantically correct on rank 0.
    pub(super) fn try_dflash_capture(
        &self,
        layer_idx: usize,
        token_idx: usize,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        // Rank-0 gate (mirrors save_hidden_for_mtp's effective behavior).
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        // The residual stream is always BF16, so DFlash hidden capture
        // copies BF16 bytes directly with no downcast.
        let src = self.buffers.hidden_states().offset(token_idx * h * bf16);
        let dst_slot = dst.offset(slot * h * bf16);
        self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        Ok(())
    }
}
