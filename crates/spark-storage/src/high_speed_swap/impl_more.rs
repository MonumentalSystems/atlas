// SPDX-License-Identifier: AGPL-3.0-only
//
//! Additional `HighSpeedSwap` methods (offload + attention orchestration).

use anyhow::Result;
use std::ffi::c_void;

use super::{HighSpeedSwap, SeqScratch};
use crate::backend::ReadRequest;
use crate::config::HighSpeedSwapConfig;
use crate::cuda_min::{CudaCtx, copy_d_to_h_async, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, KvKind};
use crate::predictor::Predictor;
use crate::scratch_pool::{ResidentKey, ScratchPool};

impl HighSpeedSwap {
    /// Persist a freshly-written KV block to disk and update the predictor's
    /// per-block K_lr. K block layout is `[block_size, num_kv_heads, head_dim]`
    /// BF16 in both `*_dev` (used for projection) and `*_host` (used for the
    /// per-(kv_head) disk stripe).
    pub fn offload_block(
        &mut self,
        ctx: &CudaCtx,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> Result<()> {
        self.offload_block_on_stream(
            ctx.stream,
            layer,
            block,
            k_block_dev,
            k_block_host,
            v_block_host,
        )
    }

    /// Stream-only variant for production callers (spark-model decode path).
    /// `stream` must already be bound to the current thread's CUDA context.
    pub fn offload_block_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> Result<()> {
        // True when the production HBM buffer at `k_block_dev` is BF16-laid-out;
        // the predictor's project_kv_block kernel reads it as BF16. Non-BF16
        // callers must use `offload_block_no_predict_on_stream`.
        self.offload_block_inner_on_stream(
            stream,
            layer,
            block,
            k_block_dev,
            k_block_host,
            v_block_host,
            true,
        )
    }

    /// FP8/quantized callers: identical to `offload_block_on_stream` but skips
    /// the predictor's per-block K projection (since `k_block_dev` is not
    /// BF16-laid-out — running the BF16 kernel on it would OOB-read into
    /// adjacent blocks). Eviction policy degrades to LRU-only for these
    /// blocks; correctness is preserved.
    pub fn offload_block_no_predict_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> Result<()> {
        self.offload_block_inner_on_stream(
            stream,
            layer,
            block,
            0,
            k_block_host,
            v_block_host,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn offload_block_inner_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
        do_predict: bool,
    ) -> Result<()> {
        if do_predict {
            self.predictor.project_kv_block_on_stream(
                stream,
                layer as usize,
                block as usize,
                k_block_dev,
            )?;
        }
        let bs = self.model.block_size as usize;
        let nkv = self.model.num_kv_heads as usize;
        let hd = self.model.head_dim as usize;
        if k_block_host.len() != bs * nkv * hd || v_block_host.len() != bs * nkv * hd {
            anyhow::bail!(
                "offload_block: host buffers must be {} BF16 elements",
                bs * nkv * hd
            );
        }
        for kh in 0..nkv {
            let mut k_stripe = Vec::with_capacity(bs * hd * 2);
            let mut v_stripe = Vec::with_capacity(bs * hd * 2);
            for tok in 0..bs {
                let base = (tok * nkv + kh) * hd;
                for x in &k_block_host[base..base + hd] {
                    k_stripe.extend_from_slice(&x.to_le_bytes());
                }
                for x in &v_block_host[base..base + hd] {
                    v_stripe.extend_from_slice(&x.to_le_bytes());
                }
            }
            self.backend
                .write_from_host(GroupKey::new(layer, block, kh as u16, KvKind::K), &k_stripe)?;
            self.backend
                .write_from_host(GroupKey::new(layer, block, kh as u16, KvKind::V), &v_stripe)?;
        }
        // Drop the resident-cache copy (if any). The on-disk image was just
        // overwritten; without invalidation, attend_layer_on_stream would
        // keep serving the stale slot. Critical for decode where the active
        // block is re-offloaded every step with new slots filled.
        self.pool.invalidate(ResidentKey { layer, block });
        Ok(())
    }

    /// Run streaming attention for one (layer, sequence). `q_dev` is the
    /// full [num_q_heads × head_dim] BF16 query for this step;
    /// `seq_block_ids` is the sequence's full block list; `output_dev`
    /// receives the [num_q_heads × head_dim] BF16 attention output.
    pub fn attend_layer(
        &mut self,
        seq_slot: usize,
        ctx: &CudaCtx,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
    ) -> Result<()> {
        self.attend_layer_on_stream(seq_slot, ctx.stream, layer, seq_block_ids, q_dev, output_dev)
    }

    /// Stream-only variant for production callers (spark-model decode path).
    /// `stream` must already be bound to the current thread's CUDA context.
    ///
    /// Backwards-compat: defaults `last_block_valid_slots` to `block_size`,
    /// i.e. no causal masking — appropriate for decode where the active
    /// block's stale slots are zero-init from `zero_block`. For prefill,
    /// callers MUST use `attend_layer_on_stream_with_q_pos` to pass the
    /// query's absolute position, otherwise future tokens within the
    /// active block leak into past queries.
    pub fn attend_layer_on_stream(
        &mut self,
        seq_slot: usize,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
    ) -> Result<()> {
        let bs = self.model.block_size as i32;
        self.attend_layer_on_stream_with_q_pos(
            seq_slot,
            stream,
            layer,
            seq_block_ids,
            q_dev,
            output_dev,
            bs,
        )
    }

    /// Causal-masking variant: `last_block_valid_slots` controls how many
    /// slots of the LAST block in `seq_block_ids` are consumed by the
    /// attention kernel. For prefill query at absolute position `q_pos`,
    /// pass `(q_pos % block_size) + 1` to mask out future positions in
    /// the active block.
    pub fn attend_layer_on_stream_with_q_pos(
        &mut self,
        seq_slot: usize,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
        last_block_valid_slots: i32,
    ) -> Result<()> {
        // Phase 5 Inc 2 recomposition: score phase → ONE stream_sync → tile
        // phase. Identical op sequence, byte counts, and stream order as the
        // pre-split single-seq body; the split only exists so the batched
        // entry below can enqueue C score phases before paying one sync.
        self.attend_score_phase(seq_slot, stream, layer, q_dev)?;
        // The tile phase's eviction ranking reads `score_host_buf` on the
        // HOST, so the async D2H score copy must have completed.
        stream_sync(stream)?;
        self.attend_tile_phase(
            seq_slot,
            stream,
            layer,
            seq_block_ids,
            q_dev,
            output_dev,
            last_block_valid_slots,
        )
    }

    /// Phase 5 Inc 2: batched overflow-decode attend. Enqueues ALL `seqs`'
    /// score phases (project_q → score_blocks → async D2H into each seq's
    /// own `score_host_buf` — disjoint per-slot buffers, no aliasing) with
    /// NO interleaved sync, pays ONE `stream_sync`, then runs each seq's
    /// tile phase serially on `stream`. Collapses the mid-attend syncs from
    /// C per attention layer to 1 (~80 → ~10 per step at C=8).
    ///
    /// Decode-only: every seq shares `last_block_valid_slots = block_size`
    /// (no causal mask — the active block's stale slots are zero-init).
    /// Prefill keeps the per-seq `attend_layer_on_stream_with_q_pos`, whose
    /// per-seq mask scalar a shared launch cannot express.
    ///
    /// H2 (cross-seq intra-tile WAR) is not regressed: the tile phases run
    /// SEQUENTIALLY on ONE stream with today's per-tile pin/assign/read/
    /// step_tile/unpin ordering, so a later seq's evict+overwrite of an
    /// earlier seq's slot is enqueued after that slot was consumed.
    pub fn attend_layer_batch_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        seqs: &[super::AttendSeqReq<'_>],
    ) -> Result<()> {
        if seqs.is_empty() {
            return Ok(());
        }
        // Phase 5 Inc 2 safety gate: the sync-collapse below is unsafe to run
        // concurrently with the Phase-3 side-stream KV prefetch. The per-seq
        // mid-attend `stream_sync`s it removes were the only barrier draining
        // each prior seq's `step_tile` before the next began; without them ALL
        // C seqs' tiles are enqueued-but-unexecuted (and their slots unpinned)
        // when `prefetch_layer` — on a separate stream with no CudaEvent
        // ordering — `assign`s + overwrites the oldest-touched slots. That is a
        // C-fold-widened cross-stream WAR → silent, timing-dependent KV
        // corruption. When prefetch is live, serve each seq with the serial
        // per-seq attend (its own score→sync→tile), restoring the pre-change
        // 1-seq in-flight window prefetch was validated against. (A proper
        // coexistence — a main-stream CudaEvent waited on `prefetch_stream` —
        // is a tracked follow-up; see the `kv_prefetch_enabled` field doc.)
        if self.kv_prefetch_enabled {
            for s in seqs {
                self.attend_layer_on_stream(
                    s.seq_slot,
                    stream,
                    layer,
                    s.seq_block_ids,
                    s.q_dev,
                    s.output_dev,
                )?;
            }
            return Ok(());
        }
        // A batch of one IS the single-seq path — delegate so the C=1
        // byte-identical guarantee is a shared code path, not a proof.
        if let [s] = seqs {
            return self.attend_layer_on_stream(
                s.seq_slot,
                stream,
                layer,
                s.seq_block_ids,
                s.q_dev,
                s.output_dev,
            );
        }
        // Per-seq scratch is selected by `seq_slot`; a duplicate would make two
        // seqs' score phases D2H into the same host buffer and share planes —
        // silent wrong results. Hard-reject it (host-side O(C²), C≤~8, and the
        // call already pays a stream_sync, so the cost is noise).
        for (i, a) in seqs.iter().enumerate() {
            for b in &seqs[i + 1..] {
                if a.seq_slot == b.seq_slot {
                    anyhow::bail!(
                        "duplicate seq_slot {} in batched attend — per-seq scratch would alias",
                        a.seq_slot
                    );
                }
            }
        }
        for s in seqs {
            self.attend_score_phase(s.seq_slot, stream, layer, s.q_dev)?;
        }
        // ONE sync for all C seqs — the Inc-2 win. It MUST precede every
        // tile phase: eviction ranking reads each seq's host score buffer.
        stream_sync(stream)?;
        let lbvs = self.model.block_size as i32;
        for s in seqs {
            self.attend_tile_phase(
                s.seq_slot,
                stream,
                layer,
                s.seq_block_ids,
                s.q_dev,
                s.output_dev,
                lbvs,
            )?;
        }
        Ok(())
    }

    /// Phase 5 Inc 2 — score phase: project Q, score every block at `layer`,
    /// and enqueue the async D2H copy of the scores into this seq's
    /// `score_host_buf`. Does NOT sync — the caller must `stream_sync` before
    /// any host read of the scores (i.e. before the tile phase).
    fn attend_score_phase(
        &mut self,
        seq_slot: usize,
        stream: u64,
        layer: u32,
        q_dev: u64,
    ) -> Result<()> {
        // Phase 2: use this sequence's OWN transient scratch so concurrent seqs
        // don't clobber each other. Lazily grow the pool to cover `seq_slot`.
        while self.scratch.len() <= seq_slot {
            let s = SeqScratch::new(&self.attn, &self.model, &self.cfg)?;
            self.scratch.push(s);
        }
        // 1. Project Q. 2. Score every block at this layer (only seq subset
        //    is consumed; the rest is wasted compute but score_blocks is µs).
        self.predictor
            .project_q_on_stream(stream, q_dev, self.scratch[seq_slot].q_proj.ptr)?;
        let m = &self.model;
        let layer_a_g = self.predictor.a_g_dev_ptr()
            + (layer as u64)
                * (m.max_blocks_per_layer as u64)
                * (m.num_kv_heads as u64)
                * (m.block_size as u64)
                * (self.cfg.rank as u64)
                * 2;
        self.predictor.score_blocks_on_stream(
            stream,
            self.scratch[seq_slot].q_proj.ptr,
            layer_a_g,
            self.scratch[seq_slot].block_scores_dev.ptr,
            m.max_blocks_per_layer as usize,
        )?;
        // Extract the immutable ptr/len before the mutable `as_mut_ptr` borrow
        // so we don't alias `self.scratch[seq_slot]` mut+shared in one call.
        let bs_ptr = self.scratch[seq_slot].block_scores_dev.ptr;
        let hbuf_bytes = self.scratch[seq_slot].score_host_buf.len() * 4;
        copy_d_to_h_async(
            self.scratch[seq_slot].score_host_buf.as_mut_ptr() as *mut c_void,
            bs_ptr,
            hbuf_bytes,
            stream,
        )?;
        Ok(())
    }

    /// Phase 5 Inc 2 — tile phase: the begin_step → step_tile* → finalize
    /// loop over `seq_block_ids`, single-seq (`num_seqs = 1`) launches into
    /// this seq's own planes. Precondition: `attend_score_phase(seq_slot, ..)`
    /// ran and the stream has been synced since (eviction ranking reads the
    /// host score buffer).
    #[allow(clippy::too_many_arguments)]
    fn attend_tile_phase(
        &mut self,
        seq_slot: usize,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
        last_block_valid_slots: i32,
    ) -> Result<()> {
        // 3. Tile loop.
        self.attn
            .begin_step_on_stream(&self.scratch[seq_slot].planes, stream, 1)?;
        let tile_cap = self.cfg.resident_blocks as usize;
        let mut tile_idx = 0;
        while tile_idx < seq_block_ids.len() {
            let tile_end = (tile_idx + tile_cap).min(seq_block_ids.len());
            let tile = &seq_block_ids[tile_idx..tile_end];

            // Pin slots already resident for tile blocks; mark them touched.
            let mut block_table = vec![0_i32; tile_cap];
            let mut pinned: Vec<u32> = Vec::new();
            // First pass: identify which tile blocks are missing.
            let mut missing: Vec<u32> = Vec::new();
            for (i, &blk) in tile.iter().enumerate() {
                let key = ResidentKey { layer, block: blk };
                if let Some(slot) = self.pool.lookup(key) {
                    block_table[i] = slot as i32;
                    pinned.push(slot);
                    self.eviction.touch(slot);
                } else {
                    missing.push(blk);
                }
            }
            // Second pass: assign + read missing blocks.
            let mut reqs: Vec<ReadRequest> = Vec::new();
            for &blk in &missing {
                let key = ResidentKey { layer, block: blk };
                let candidates = self.eviction.rank(&pinned);
                let slot = self.pool.assign(key, &candidates)?;
                pinned.push(slot);
                self.eviction.touch(slot);
                self.eviction
                    .record_score(slot, self.scratch[seq_slot].score_host_buf[blk as usize]);
                // Find this block's index in the tile so the block_table is right.
                let idx = tile.iter().position(|&x| x == blk).unwrap();
                block_table[idx] = slot as i32;
                for kh in 0..self.model.num_kv_heads {
                    reqs.push(ReadRequest {
                        group: GroupKey::new(layer, blk, kh, KvKind::K),
                        dst_dev_ptr: self.pool.slot_k_ptr(slot, kh),
                    });
                    reqs.push(ReadRequest {
                        group: GroupKey::new(layer, blk, kh, KvKind::V),
                        dst_dev_ptr: self.pool.slot_v_ptr(slot, kh),
                    });
                }
            }
            self.backend.read(&reqs, stream)?;

            // 4. Tiled attention launch.
            let counts = [(tile.len()) as i32];
            copy_h_to_d_async(
                self.scratch[seq_slot].block_table_dev.ptr,
                block_table.as_ptr() as *const c_void,
                tile_cap * 4,
                stream,
            )?;
            copy_h_to_d_async(
                self.scratch[seq_slot].counts_dev.ptr,
                counts.as_ptr() as *const c_void,
                4,
                stream,
            )?;
            let (s_blk, s_tok, s_kvh) = self.attn.scratch_pool_strides();
            let v_off = (self.model.num_kv_heads as u64)
                * (self.model.block_size as u64)
                * (self.model.head_dim as u64)
                * 2;
            // Causal mask: only apply on the FINAL tile of the seq's block
            // list. Earlier tiles are full blocks of historical K/V.
            let lbvs = if tile_end == seq_block_ids.len() {
                last_block_valid_slots
            } else {
                self.model.block_size as i32
            };
            self.attn.step_tile_on_stream(
                &self.scratch[seq_slot].planes,
                stream,
                q_dev,
                self.pool.pool_dev_ptr(),
                self.pool.pool_dev_ptr() + v_off,
                self.scratch[seq_slot].block_table_dev.ptr,
                self.scratch[seq_slot].counts_dev.ptr,
                1,
                s_blk,
                s_tok,
                s_kvh,
                lbvs,
            )?;
            // Phase 3: this tile's blocks have now been consumed by step_tile
            // (enqueued on `stream`), so release any prefetch pin on them —
            // freeing the slots for LATER tiles' on-demand reads and for the
            // next layer's prefetch. Stream ordering makes a subsequent
            // evict+overwrite of one of these slots safe (its read is enqueued
            // after this step_tile). `unpin_key` is a saturating no-op for
            // blocks read on-demand (never prefetched), so the non-prefetch
            // path is unaffected.
            for &blk in tile {
                self.pool.unpin_key(ResidentKey { layer, block: blk });
            }
            tile_idx = tile_end;
        }
        self.attn
            .finalize_on_stream(&self.scratch[seq_slot].planes, stream, output_dev, 1)?;
        Ok(())
    }

    /// Phase 3: **prefetch** (reserve + load + PIN) every block of `layer` into
    /// the scratch pool WITHOUT attending, so a later `attend_layer` for the
    /// same `layer` finds them resident (no on-demand read) and just consumes +
    /// unpins them. Issue on a SIDE stream to overlap the NVMe/tier read with
    /// the SSM+MoE compute between two attention layers — the Phase-3 win.
    ///
    /// The pins protect the reserved slots from eviction (by another seq's
    /// attend, or offload `invalidate`) during that compute window. Best-effort:
    /// if the pool can't fit more (every candidate is pinned), it stops early —
    /// the un-prefetched tail is simply read on-demand by `attend_layer`, still
    /// correct, just unhidden. The caller MUST pair this with an `attend_layer`
    /// for the same `layer` (which releases the pins); an unmatched prefetch
    /// leaks pins.
    pub fn prefetch_layer_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
    ) -> Result<()> {
        // Score order for eviction victims; `assign` skips already-pinned slots.
        let candidates = self.eviction.rank(&[]);
        let mut reqs: Vec<ReadRequest> = Vec::new();
        for &blk in seq_block_ids {
            let key = ResidentKey { layer, block: blk };
            if let Some(slot) = self.pool.lookup(key) {
                // Already resident (prior step / earlier tile) — pin to protect,
                // no re-read.
                self.pool.pin(slot);
                self.eviction.touch(slot);
                continue;
            }
            let slot = match self.pool.assign(key, &candidates) {
                Ok(s) => s,
                // Pool exhausted by pins: stop prefetching; attend reads the rest.
                Err(_) => break,
            };
            self.pool.pin(slot);
            self.eviction.touch(slot);
            for kh in 0..self.model.num_kv_heads {
                reqs.push(ReadRequest {
                    group: GroupKey::new(layer, blk, kh, KvKind::K),
                    dst_dev_ptr: self.pool.slot_k_ptr(slot, kh),
                });
                reqs.push(ReadRequest {
                    group: GroupKey::new(layer, blk, kh, KvKind::V),
                    dst_dev_ptr: self.pool.slot_v_ptr(slot, kh),
                });
            }
        }
        if !reqs.is_empty() {
            self.backend.read(&reqs, stream)?;
        }
        Ok(())
    }

    /// Phase 3: prefetch `layer`'s blocks on the internal **side stream** so the
    /// H2D copies overlap the main stream's compute. Convenience over
    /// `prefetch_layer_on_stream`. The scheduler calls this for the NEXT
    /// attention layer's blocks while the intervening SSM+MoE layers' kernels
    /// are enqueued on the main stream, hiding the tier read behind that compute.
    pub fn prefetch_layer(&mut self, layer: u32, seq_block_ids: &[u32]) -> Result<()> {
        let s = self.prefetch_stream;
        self.prefetch_layer_on_stream(s, layer, seq_block_ids)
    }

    /// Test/diag accessors.
    pub fn pool(&self) -> &ScratchPool {
        &self.pool
    }
    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }
    pub fn config(&self) -> &HighSpeedSwapConfig {
        &self.cfg
    }
}
