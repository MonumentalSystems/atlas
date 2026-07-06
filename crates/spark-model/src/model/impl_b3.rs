// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
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
            routed_lora_layers: None, // #30: MTP/draft decode never routes prefill.
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
    /// Task #24: stable adapter_id for a per-request pool-slot selector. Returns
    /// the base sentinel `0` when no LoRA pool is resident (byte-identical base),
    /// else the NAME-derived id of the resolved slot (`-1 -> active`). Resolved
    /// here at prefill time because `LoraWeights.active` can rotate between HTTP
    /// request resolution and prefill.
    pub fn adapter_id_for_slot(&self, slot: i32) -> u64 {
        match self.lora.as_ref() {
            Some(lw) => lw.adapter_id_for_slot(slot),
            None => 0,
        }
    }

    /// Task #25: acquire a per-slot ref for a sequence beginning to use its
    /// adapter (called at prefill, resolving `-1 -> active` exactly like
    /// [`Self::adapter_id_for_slot`]). Returns the RESOLVED pool index the ref
    /// was taken on — the caller stores it and releases EXACTLY that index at
    /// terminal free, so an intervening rotate changing `active` cannot make
    /// release hit a different counter. `-1` ("nothing acquired") when no LoRA
    /// pool is resident or the slot is out of range → byte-identical no-op base.
    pub fn acquire_adapter_slot(&self, slot: i32) -> i32 {
        match self.lora.as_ref() {
            Some(lw) => lw.acquire_slot(slot),
            None => -1,
        }
    }

    /// Task #25: release a ref acquired by [`Self::acquire_adapter_slot`], by the
    /// RESOLVED index it returned. `-1` and no-pool are no-ops (base path).
    pub fn release_adapter_slot(&self, resolved: i32) {
        if let Some(lw) = self.lora.as_ref() {
            lw.release_slot(resolved);
        }
    }

    pub fn set_lora_weights(&mut self, lora: Option<crate::lora::LoraWeights>) -> Result<()> {
        if let Some(ref lw) = lora {
            // eager-on-rotate: ONLY the global rotate/swap re-point path forces
            // eager decode. A multi-adapter pool no longer implies eager —
            // per-request routing (M2) is graph-safe by construction (the
            // per-seq slot buffer is per-step-uploaded to a stable address, the
            // pool tables are load-time-fixed), so decode graphs STAY captured
            // under routing. Equating slots.len()>1 with eager here would throw
            // away the entire point of batched routing.
            self.lora_rotatable =
                crate::lora::lora_rotate_env() || crate::lora::lora_peer_env().is_some();
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            // Clone the active slot's pairs (small; LoraPair is Copy) so the
            // install walk can hold a shared borrow while it &mut-borrows
            // `self.layers`. Clone the (Copy) pool table pointers + scale table
            // too so the routed batched-decode path can read them per layer.
            let active = lw.active_layers().to_vec();
            let tables = lw.tables.clone();
            let scale_table = lw.scale_table;
            let installed = self.install_lora_layers(&active, kernels, &tables, scale_table)?;
            // Task #27: `slots` is pre-sized to max_loras with empty cache
            // placeholders; report only the filled (named) adapters.
            let resident: Vec<String> = lw
                .adapter_names()
                .into_iter()
                .filter(|n| !n.is_empty())
                .collect();
            tracing::info!(
                "LoRA: {} adapter(s) resident [{}], active '{}' installed on \
                 {installed} layers (r={}, max_rank={}, max_loras={}, \
                 pool={:.1} MiB, rotatable={})",
                resident.len(),
                resident.join(", "),
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
        tables: &std::collections::BTreeMap<
            (usize, crate::lora::LoraModule),
            (spark_runtime::gpu::DevicePtr, spark_runtime::gpu::DevicePtr),
        >,
        scale_table: spark_runtime::gpu::DevicePtr,
    ) -> Result<usize> {
        use crate::lora::LoraModule;
        // Build the per-module routing table from the frozen pool tables + the
        // active-slot pair dims (k_in/n_out/max_rank identical across slots, so
        // the active pair supplies them). `None` when the module has no table
        // (base-only) — the bgmv apply site then no-ops for that module.
        let mk_route = |layer_idx: usize,
                        module: LoraModule,
                        pair: &Option<ops::lora_delta::LoraPair>|
         -> Option<ops::lora_delta::LoraRoute> {
            let p = pair.as_ref()?;
            let (a_table, b_table) = *tables.get(&(layer_idx, module))?;
            Some(ops::lora_delta::LoraRoute {
                a_table,
                b_table,
                scale_table,
                k_in: p.k_in,
                n_out: p.n_out,
                max_rank: p.max_rank,
            })
        };
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
                // #30: the global layer index (from `self.layers.enumerate()`) —
                // the key the request slot's GLOBAL-layer-indexed pairs use.
                layer_idx: idx,
                k: layer_weights.k_proj,
                v: layer_weights.v_proj,
                o: layer_weights.o_proj,
                kernels,
                k_route: mk_route(idx, LoraModule::KProj, &layer_weights.k_proj),
                v_route: mk_route(idx, LoraModule::VProj, &layer_weights.v_proj),
                o_route: mk_route(idx, LoraModule::OProj, &layer_weights.o_proj),
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

    /// #28: drain + DESTROY the decode graph caches (`decode_graph` +
    /// `batch_decode_graphs`) on a rotate/swap. `GraphHandle` has no `Drop`, so a
    /// bare `.clear()` would LEAK the CUDA graphs — and now that a swappable pool
    /// decodes GRAPHED (not forced-eager, #28) these caches actually hold graphs.
    /// The compound `(slot, active_id)` key already makes replay safe, so this is
    /// belt-and-suspenders — but it must DESTROY, not just drop. Runs at scheduler
    /// quiescence on the CUDA-bound model thread (like `free_sequence`'s destroys).
    fn destroy_lora_decode_graphs(&self) {
        for (_, g) in self.decode_graph.lock().drain() {
            if g.0 != 0
                && let Err(e) = self.gpu.destroy_graph(g)
            {
                tracing::warn!("LoRA graph clear: destroy decode_graph: {e:#}");
            }
        }
        for (_, g) in self.batch_decode_graphs.lock().drain() {
            if g.0 != 0
                && let Err(e) = self.gpu.destroy_graph(g)
            {
                tracing::warn!("LoRA graph clear: destroy batch_decode_graph: {e:#}");
            }
        }
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
        // #25 safety: rotation RE-INSTALLS the new slot's pairs onto the layer
        // structs, so any in-flight sequence still decoding on the OLD active
        // adapter (via the installed pair) would replay with the wrong delta.
        // Refuse while the current active slot has in-flight sequences — rotate
        // only at a scheduler-quiescent point (matches this method's contract).
        {
            let lw = self.lora.as_ref().unwrap();
            let cur = lw.active;
            if lw.slot_ref_count(cur) > 0 {
                anyhow::bail!(
                    "LoRA rotation refused: active slot {cur} has in-flight \
                     sequences (ref_count>0); rotate at a quiescent point"
                );
            }
        }
        // Re-point onto the new active slot.
        let (layers, active_name, r, tables, scale_table) = {
            let lw = self.lora.as_mut().unwrap();
            lw.active = slot;
            lw.name = lw.slots[slot].name.clone();
            lw.adapter_config = lw.slots[slot].adapter_config.clone();
            (
                lw.slots[slot].layers.clone(),
                lw.name.clone(),
                lw.adapter_config.r,
                lw.tables.clone(),
                lw.scale_table,
            )
        };
        let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
        let installed = self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
        // Defensive: drop any captured decode graphs so a stale-pointer replay
        // is impossible even if `lora_rotatable` were ever mis-derived. Under
        // forced eager these are already empty.
        self.destroy_lora_decode_graphs();
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
        // Task #25 busy-slot refusal: bail BEFORE the destructive memset/stage
        // below so a refused swap leaves the slot's bytes + identity untouched.
        // Replacing an adapter while sequences are mid-decode on it would corrupt
        // their KV and replay a captured graph over swapped pool bytes.
        {
            let busy = self.lora.as_ref().unwrap().slot_ref_count(slot);
            if busy > 0 {
                anyhow::bail!(
                    "LoRA RDMA swap REFUSED: slot {slot} has {busy} in-flight \
                     sequence(s) (ref_count>0); cannot replace an adapter mid-decode"
                );
            }
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
        // Task #26: refresh this slot's a/b pointer tables + scale table from the
        // freshly-staged adapter's actual coverage BEFORE `peft`/`layers` are moved
        // into the slot stamp — a promoted adapter with different module coverage
        // than the evicted one would otherwise keep a stale bgmv route entry for the
        // reused cache slot (missed / wrong-scaled delta). Same fix as the disk swap.
        self.lora.as_ref().unwrap().refresh_slot_tables(
            slot,
            &layers,
            peft.scaling(),
            self.gpu.as_ref(),
        )?;
        {
            let lw = self.lora.as_mut().unwrap();
            let s = lw
                .slots
                .get_mut(slot)
                .ok_or_else(|| anyhow::anyhow!("LoRA RDMA swap: slot {slot} not resident"))?;
            s.name = adapter_name.to_string();
            s.adapter_config = peft;
            s.layers = layers;
            // Task #25: contents changed → bump generation so this re-staged slot
            // yields a FRESH adapter_id (a later same-name request misses the
            // stale prior KV). Pure rotate does NOT reach here.
            s.generation = s.generation.wrapping_add(1);
        }

        // 4) If the swapped slot is active, re-install onto the layer structs.
        let active = self.lora.as_ref().unwrap().active;
        if active == slot {
            let installed_layers = self.lora.as_ref().unwrap().slots[slot].layers.clone();
            let tables = self.lora.as_ref().unwrap().tables.clone();
            let scale_table = self.lora.as_ref().unwrap().scale_table;
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&installed_layers, kernels, &tables, scale_table)?;
            self.lora.as_mut().unwrap().name = adapter_name.to_string();
            self.destroy_lora_decode_graphs();
        }
        tracing::info!(
            "LoRA RDMA swap: '{adapter_name}' landed in slot {slot} \
             ({} targets, active_slot={active})",
            targets.len()
        );
        Ok(())
    }

    /// Task #27 (demand-driven promotion): promote the adapter `adapter_name`
    /// (staged on `peer_addr` at `adapter_id`) from the peer into a CACHE-region
    /// pool slot and make it ACTIVE, returning `(slot, evicted_name)`. Runs on
    /// the scheduler thread at a QUIESCENT point (the only place per-slot
    /// `ref_count` is authoritative). Victim policy (pure [`select_victim_slot`]):
    /// a never-filled placeholder first, else the LRU idle (`ref_count == 0`)
    /// cache slot, else `POOL_FULL` (retryable — a busy slot is NEVER evicted).
    /// The underlying [`Self::swap_lora_slot_from_peer`] re-checks `ref_count>0`
    /// and bails as a backstop, and bumps the slot generation so #24 KV stays
    /// correct. Making the promoted slot active mirrors the rotate/load control
    /// plane so the delta actually applies under batch-1 (the per-slot bgmv route
    /// tables are still dormant — compute reads the installed active adapter).
    #[cfg(feature = "cuda")]
    pub fn promote_lora_slot_from_peer(
        &mut self,
        peer_addr: &str,
        adapter_id: &str,
        adapter_name: &str,
        peft: atlas_core::config::PeftAdapterConfig,
    ) -> Result<(usize, Option<String>)> {
        // 1) Snapshot the cache region + pick a victim (pure policy).
        let (slot, evicted) = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA promote: no adapter pool loaded"))?;
            let views = lw.cache_slot_views();
            let slot = crate::lora::select_victim_slot(&views).map_err(|e| match e {
                crate::lora::VictimError::PoolFull => anyhow::anyhow!(
                    "POOL_FULL: all {} cache slot(s) are busy (ref_count>0); retry",
                    views.len()
                ),
            })?;
            // The name being replaced (if the victim already held an adapter) so
            // the caller can drop the stale name->slot overlay entry.
            let evicted = lw
                .slots
                .get(slot)
                .map(|s| s.name.clone())
                .filter(|n| !n.is_empty());
            (slot, evicted)
        };

        // 2) RDMA-stage into the victim slot (re-checks ref_count>0, bumps gen).
        self.swap_lora_slot_from_peer(peer_addr, adapter_id, adapter_name, slot, peft)?;

        // 3) Make the promoted slot ACTIVE so its delta applies (batch-1 honest).
        //    `swap_lora_slot_from_peer` already re-installed if the victim WAS the
        //    active slot; otherwise re-point the installed pairs onto it here.
        let already_active = self.lora.as_ref().unwrap().active == slot;
        if !already_active {
            let (layers, tables, scale_table) = {
                let lw = self.lora.as_mut().unwrap();
                lw.active = slot;
                lw.name = lw.slots[slot].name.clone();
                lw.adapter_config = lw.slots[slot].adapter_config.clone();
                (
                    lw.slots[slot].layers.clone(),
                    lw.tables.clone(),
                    lw.scale_table,
                )
            };
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
            self.destroy_lora_decode_graphs();
        }
        // Stamp the freshly-promoted slot as most-recently-used so a back-to-back
        // promote of a DIFFERENT cold adapter picks an older victim, not this one,
        // before the request that triggered this promote has acquired its ref.
        self.lora.as_ref().unwrap().touch_slot(slot);
        tracing::info!(
            "LoRA promote: '{adapter_name}' hot in cache slot {slot} \
             (evicted={:?}), now active",
            evicted
        );
        Ok((slot, evicted))
    }

    /// Disk-swap the adapter at `adapter_dir` INTO pool `slot`, in place, then
    /// make it that slot's resident adapter (re-installing onto the layer structs
    /// if the slot is currently active). The local-disk analog of
    /// [`Self::swap_lora_slot_from_peer`] — same audit + pack + re-point, no RDMA.
    /// This is the pool-size-1 dynamic-load path: load a DIFFERENT adapter into
    /// the single slot at runtime (per-request weight change). MUST be called at
    /// a scheduler QUIESCENT point (no in-flight decode reading `slot`) and needs
    /// rotation armed (`ATLAS_LORA_ROTATE=1`/`$ATLAS_LORA_PEER`) so decode is
    /// eager and no captured graph replays the swapped slot's stale pointers.
    pub fn swap_lora_slot_from_disk(
        &mut self,
        adapter_dir: &std::path::Path,
        name: &str,
        slot: usize,
    ) -> Result<()> {
        if !self.lora_rotatable {
            anyhow::bail!(
                "LoRA disk swap needs rotation armed (set ATLAS_LORA_ROTATE=1 so \
                 decode runs eager); a single startup adapter with no rotation env \
                 is baked into the decode graph and a re-point would replay stale"
            );
        }
        // Task #25 busy-slot refusal: fail fast (before the disk load + pack)
        // when the target slot has in-flight sequences. `pack_store_into_slot`
        // re-checks under `&mut lw` right before the destructive memset (the
        // authoritative gate); this early check just avoids the wasted load.
        if let Some(lw) = self.lora.as_ref() {
            let busy = lw.slot_ref_count(slot);
            if busy > 0 {
                anyhow::bail!(
                    "LoRA disk swap REFUSED: slot {slot} has {busy} in-flight \
                     sequence(s) (ref_count>0); cannot replace an adapter mid-decode"
                );
            }
        }
        // Parse the adapter's own PEFT config (scaling read per adapter, never
        // defaulted) — the same hard-fail parser the startup path uses.
        let cfg_path = adapter_dir.join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let peft = atlas_core::config::parse_peft_adapter_config(&raw)
            .with_context(|| format!("parse {}", cfg_path.display()))?;
        // Load the adapter's A/B into a device WeightStore (host F16/F32→BF16),
        // then pack it into the slot (same layout as a startup pack).
        let store = spark_runtime::weights::adapter::load_adapter_safetensors(
            adapter_dir,
            self.gpu.as_ref(),
            0,
        )
        .context("load LoRA adapter weights for disk swap")?;
        let layers = {
            let lw = self
                .lora
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("LoRA disk swap: no adapter pool loaded"))?;
            crate::lora::pack_store_into_slot(
                lw,
                slot,
                name,
                &store,
                &peft,
                &self.config,
                self.gpu.as_ref(),
            )?
        };
        // If the swapped slot is the active one, re-install onto the layer structs
        // so subsequent requests apply the new adapter's delta.
        let active = self.lora.as_ref().unwrap().active;
        if active == slot {
            let tables = self.lora.as_ref().unwrap().tables.clone();
            let scale_table = self.lora.as_ref().unwrap().scale_table;
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
            self.lora.as_mut().unwrap().name = name.to_string();
            self.destroy_lora_decode_graphs();
        }
        tracing::info!(
            "LoRA disk swap: '{name}' packed into slot {slot} (r={}, active_slot={active})",
            peft.r
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
