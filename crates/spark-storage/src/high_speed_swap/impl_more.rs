// SPDX-License-Identifier: AGPL-3.0-only
//
//! Additional `HighSpeedSwap` methods (offload + attention orchestration).

use anyhow::Result;
use std::ffi::c_void;

use super::{HighSpeedSwap, SeqScratch};
use crate::backend::{BlockReadRequest, ReadRequest};
use crate::config::HighSpeedSwapConfig;
use crate::cuda_min::{
    CudaCtx, copy_d_to_d_async, copy_d_to_h_async, copy_h_to_d_async, stream_sync,
};
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
        if self.coalesce_blocks {
            // ATLAS_HSS_COALESCE_BLOCKS: pack all nkv K/V stripes into ONE
            // block_bytes host image (each raw stripe padded to group_stride) and
            // issue ONE contiguous block write instead of 2·nkv per-head pwrites.
            let buf = self.assemble_block_image_from_host(k_block_host, v_block_host);
            self.backend
                .write_block_from_host(GroupKey::new(layer, block, 0, KvKind::K), &buf)?;
        } else {
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
        }
        // Drop the resident-cache copy (if any). The on-disk image was just
        // overwritten; without invalidation, attend_layer_on_stream would
        // keep serving the stale slot. Critical for decode where the active
        // block is re-offloaded every step with new slots filled.
        self.pool.invalidate(ResidentKey { layer, block });
        Ok(())
    }

    /// Build the single `block_bytes` on-disk image for one block from its host
    /// K/V (`[block_size, num_kv_heads, head_dim]` BF16): pack each kv_head's K/V
    /// stripe (raw `block_size·head_dim·2` bytes) then pad to `group_stride` via
    /// `assemble_block_write_buffer`. The ONE source of the coalesced block image,
    /// shared by the immediate `offload_block_inner_on_stream` write and the
    /// deferred `stage_block_into` accumulator — so the two can never drift.
    fn assemble_block_image_from_host(
        &self,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> Vec<u8> {
        let bs = self.model.block_size as usize;
        let nkv = self.model.num_kv_heads as usize;
        let hd = self.model.head_dim as usize;
        let group_stride = self.backend.group_layout().group_stride as usize;
        let mut k_stripes: Vec<Vec<u8>> = Vec::with_capacity(nkv);
        let mut v_stripes: Vec<Vec<u8>> = Vec::with_capacity(nkv);
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
            k_stripes.push(k_stripe);
            v_stripes.push(v_stripe);
        }
        assemble_block_write_buffer(nkv, group_stride, &k_stripes, &v_stripes)
    }

    /// ATLAS_HSS_COALESCE_WRITE_RUNS stage step: run the per-block work of the
    /// coalesced offload EXCEPT the disk write — predictor projection (only when
    /// `do_predict`, i.e. the BF16 arms), then assemble the block's `block_bytes`
    /// image into `out`. Does NOT write and does NOT invalidate: those are
    /// deferred to `flush_write_run` so the run's wide pwrite lands the bytes
    /// before the resident copies are dropped (preserving the disk-overwritten ⇒
    /// resident-dropped atomicity of `offload_block_inner_on_stream`).
    ///
    /// `out` MUST be exactly `block_bytes` — the caller slices its staging buffer
    /// at `run_len · block_bytes` so block `i` lands at `staging[i·block_bytes]`,
    /// making the concatenation byte-identical to the block's on-disk span.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_block_into(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        do_predict: bool,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
        out: &mut [u8],
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
                "stage_block_into: host buffers must be {} BF16 elements",
                bs * nkv * hd
            );
        }
        let group_stride = self.backend.group_layout().group_stride as usize;
        let block_bytes = 2 * nkv * group_stride;
        if out.len() != block_bytes {
            anyhow::bail!(
                "stage_block_into: out slice {} != block_bytes {block_bytes}",
                out.len()
            );
        }
        let buf = self.assemble_block_image_from_host(k_block_host, v_block_host);
        out.copy_from_slice(&buf);
        Ok(())
    }

    /// ATLAS_HSS_COALESCE_WRITE_RUNS flush: write a run of `run_len`
    /// strictly-consecutive same-layer blocks starting at `run_start` in ONE wide
    /// pwrite (`write_blocks_run`), then drop the resident-cache copy for EVERY
    /// block in the run. Invalidate is deferred to HERE (after the bytes land) so
    /// a concurrent attend can never read a slot whose disk image is mid-write.
    /// No-op on an empty run (`run_len == 0`).
    pub fn flush_write_run(
        &mut self,
        layer: u32,
        run_start: u32,
        run_len: usize,
        staging: &[u8],
    ) -> Result<()> {
        if run_len == 0 {
            return Ok(());
        }
        let nkv = self.model.num_kv_heads as usize;
        let group_stride = self.backend.group_layout().group_stride as usize;
        let block_bytes = 2 * nkv * group_stride;
        let run_bytes = run_len * block_bytes;
        if staging.len() < run_bytes {
            anyhow::bail!(
                "flush_write_run: staging {} < run bytes {run_bytes} ({run_len} × {block_bytes})",
                staging.len()
            );
        }
        // O_DIRECT keeps offset+length 4096-aligned; block_bytes is a multiple of
        // group_stride (4096), so run_bytes is too.
        debug_assert_eq!(run_bytes % 4096, 0, "run bytes must stay O_DIRECT-aligned");
        self.backend.write_blocks_run(
            GroupKey::new(layer, run_start, 0, KvKind::K),
            run_len,
            &staging[..run_bytes],
        )?;
        for i in 0..run_len {
            self.pool.invalidate(ResidentKey {
                layer,
                block: run_start + i as u32,
            });
        }
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
        // #11: the Phase-3 side-stream KV prefetch now coexists with the
        // sync-collapse below. The cross-stream WAR (prefetch `assign`+overwrite
        // racing an enqueued-but-unexecuted `step_tile` read) is closed by the
        // `kv_war_event` fence: the decode loop records it on THIS (main) stream
        // at each prefetch boundary — after every attend enqueued so far this
        // step — and `prefetch_layer_on_stream` waits it on `prefetch_stream`
        // before the overwriting H2D. So the former serial-fallback gate is
        // gone; batched + prefetch run together.
        //
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
        // Phase 5 Inc 3: fuse the C per-seq tile phases into ONE
        // grid=(C, nq, 1) launch per tile with a union tier read, when the
        // C-sized BatchScratch exists (ATLAS_HSS_MAX_SEQS>1 sized the pool for
        // C×tile_cap) and the batch fits it. Otherwise keep the Inc-2 serial
        // tile phases (num_seqs=1 each) — correct, just unfused; this covers
        // the env-unset default (max_seqs==1, no BatchScratch) and any batch
        // wider than the configured C (pool can't hold C×tile_cap for it).
        if self.batch.is_some() && seqs.len() <= self.max_seqs {
            self.attend_tile_phase_batched(stream, layer, seqs, lbvs)?;
        } else {
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
        }
        Ok(())
    }

    /// Phase 5 Inc 3 — batched tile phase: process all C seqs in lockstep
    /// tiles through ONE `grid=(C, num_q_heads, 1)` launch per tile, with a
    /// **union** tier read across every seq's missing blocks. Replaces the C
    /// serial `num_seqs=1` launches (Inc 2) — attacking the real `attn ∝ N`
    /// cost the C=8 measurement found (8 under-occupied launches + 8 serial
    /// tier waits → one wide launch + one union read).
    ///
    /// Preconditions: `attend_score_phase` ran for every seq and the stream
    /// has been synced since (eviction ranking reads each seq's host score
    /// buffer); `self.batch` is `Some` (`max_seqs > 1`) and `seqs.len() <=
    /// max_seqs` (checked by the caller). Decode-only: every seq shares
    /// `lbvs` (= block_size); ragged tails are expressed purely via
    /// per-seq `counts` (an exhausted seq presents `counts[s]=0`, an exact
    /// kernel no-op — hazard H3).
    fn attend_tile_phase_batched(
        &mut self,
        stream: u64,
        layer: u32,
        seqs: &[super::AttendSeqReq<'_>],
        lbvs: i32,
    ) -> Result<()> {
        let c = seqs.len();
        // #11-refinement: consume the async prefetch's mirror-RAW event. The
        // producer (prefetch_layer_on_stream) recorded `kv_prefetch_done` on
        // `prefetch_stream` after the prefetched H2D; wait it DEVICE-side here so
        // this attend's slot reads see the landed bytes. The host does NOT block
        // (unlike the old terminal stream_sync inside backend.read). Gated on
        // prefetch so prefetch-OFF emits zero ops (byte-identity).
        if self.kv_prefetch_enabled {
            self.kv_prefetch_done.wait(stream)?;
        }
        let tile_cap = self.cfg.resident_blocks as usize;
        let q_row_elems = self.model.num_q_heads as usize * self.model.head_dim as usize;
        let q_row_bytes = q_row_elems * 2; // BF16
        // Device ptrs of the shared batch buffers (Copy u64 — extracted up
        // front so the &mut self.pool/eviction/backend calls below don't clash
        // with a live borrow of self.batch).
        let (bt_ptr, ct_ptr, qg_ptr, og_ptr) = {
            let b = self.batch.as_ref().expect("batch present (checked by caller)");
            (
                b.block_table_dev.ptr,
                b.counts_dev.ptr,
                b.q_gather_dev.ptr,
                b.o_gather_dev.ptr,
            )
        };

        // Gather each seq's Q into contiguous row c of q_gather (the kernel
        // reads Q[(seq×nq+qh)×hd] with seq=0..C-1, but the seqs sit at their
        // original, possibly sparse, batch positions).
        for (c_idx, s) in seqs.iter().enumerate() {
            copy_d_to_d_async(
                qg_ptr + (c_idx * q_row_bytes) as u64,
                s.q_dev,
                q_row_bytes,
                stream,
            )?;
        }

        self.attn.begin_step_on_stream(
            &self.batch.as_ref().expect("batch present").planes,
            stream,
            c,
        )?;

        let max_tiles = seqs
            .iter()
            .map(|s| s.seq_block_ids.len().div_ceil(tile_cap))
            .max()
            .unwrap_or(0);
        let (s_blk, s_tok, s_kvh) = self.attn.scratch_pool_strides();
        let v_off = (self.model.num_kv_heads as u64)
            * (self.model.block_size as u64)
            * (self.model.head_dim as u64)
            * 2;

        for t in 0..max_tiles {
            // H1: block_table sized C×tile_cap, counts sized C. H3: counts
            // rebuilt fresh [0;C] every tile — an exhausted seq presents 0 and
            // its (zero-init) block_table row is a kernel no-op.
            let mut block_table = vec![0_i32; c * tile_cap];
            let mut counts = vec![0_i32; c];
            // H2: pin EVERY slot assigned/resident across ALL C seqs for this
            // tile before the single step_tile, unpin after. A later seq's
            // assign therefore can't evict an earlier seq's just-placed,
            // not-yet-consumed slot (`rank(&pinned)` excludes them).
            let mut pinned: Vec<u32> = Vec::new();
            let mut reqs: Vec<ReadRequest> = Vec::new();
            // ATLAS_HSS_COALESCE_BLOCKS: one BlockReadRequest per missing block
            // (dst = slot base) instead of 2·nkv per-head reqs. Exactly ONE of
            // `reqs`/`breqs` is populated per call (flag chooses); both are
            // non-empty iff ≥1 block is missing, so the #11 empty-skip guard and
            // its op-identity reasoning are preserved.
            let mut breqs: Vec<BlockReadRequest> = Vec::new();

            for (c_idx, s) in seqs.iter().enumerate() {
                let blocks = s.seq_block_ids;
                let start = t * tile_cap;
                if start >= blocks.len() {
                    continue; // seq exhausted — counts[c_idx] stays 0 (H3)
                }
                let end = (start + tile_cap).min(blocks.len());
                let tile = &blocks[start..end];
                counts[c_idx] = tile.len() as i32;
                let row = c_idx * tile_cap;
                for (i, &blk) in tile.iter().enumerate() {
                    let key = ResidentKey { layer, block: blk };
                    if let Some(slot) = self.pool.lookup(key) {
                        block_table[row + i] = slot as i32;
                        pinned.push(slot);
                        self.eviction.touch(slot);
                    } else {
                        let candidates = self.eviction.rank(&pinned);
                        let slot = self.pool.assign(key, &candidates)?;
                        pinned.push(slot);
                        self.eviction.touch(slot);
                        self.eviction.record_score(
                            slot,
                            self.scratch[s.seq_slot].score_host_buf[blk as usize],
                        );
                        block_table[row + i] = slot as i32;
                        if self.coalesce_blocks {
                            breqs.push(BlockReadRequest {
                                base_key: GroupKey::new(layer, blk, 0, KvKind::K),
                                dst_dev_ptr: self.pool.slot_dev_ptr(slot),
                            });
                        } else {
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
                    }
                }
            }

            // ONE union read for the whole C-wide tile. #11: skip the no-op
            // read ONLY when prefetch is live — that is the run where
            // io_uring's unconditional trailing `stream_sync` (an accidental
            // WAR-narrowing barrier) is now replaced by the `kv_war_event`
            // fence. Prefetch-off keeps the unconditional read+sync for
            // byte-for-byte op-identity. `breqs.is_empty() ⟺ reqs.is_empty()`
            // (both non-empty iff ≥1 missing block), so this guard is unchanged.
            if !self.kv_prefetch_enabled || !reqs.is_empty() || !breqs.is_empty() {
                if self.coalesce_blocks {
                    self.backend.read_blocks(&breqs, stream)?;
                } else {
                    self.backend.read(&reqs, stream)?;
                }
            }

            copy_h_to_d_async(
                bt_ptr,
                block_table.as_ptr() as *const c_void,
                c * tile_cap * 4,
                stream,
            )?;
            copy_h_to_d_async(ct_ptr, counts.as_ptr() as *const c_void, c * 4, stream)?;

            // ONE wide launch across all C seqs.
            self.attn.step_tile_on_stream(
                &self.batch.as_ref().expect("batch present").planes,
                stream,
                qg_ptr,
                self.pool.pool_dev_ptr(),
                self.pool.pool_dev_ptr() + v_off,
                bt_ptr,
                ct_ptr,
                c,
                s_blk,
                s_tok,
                s_kvh,
                lbvs,
            )?;

            // Unpin this tile's blocks across all C seqs (Phase 3 pin release —
            // stream-ordered, so a later same-stream evict+overwrite is safe;
            // the cross-STREAM prefetch overwrite is ordered after these
            // `step_tile` reads by the `kv_war_event` fence — #11).
            for s in seqs {
                let blocks = s.seq_block_ids;
                let start = t * tile_cap;
                if start >= blocks.len() {
                    continue;
                }
                let end = (start + tile_cap).min(blocks.len());
                for &blk in &blocks[start..end] {
                    self.pool.unpin_key(ResidentKey { layer, block: blk });
                }
            }
        }

        // finalize(num_seqs=C) writes all C rows into o_gather; scatter each
        // row back to its seq's destination.
        self.attn.finalize_on_stream(
            &self.batch.as_ref().expect("batch present").planes,
            stream,
            og_ptr,
            c,
        )?;
        for (c_idx, s) in seqs.iter().enumerate() {
            copy_d_to_d_async(
                s.output_dev,
                og_ptr + (c_idx * q_row_bytes) as u64,
                q_row_bytes,
                stream,
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
        // #11-refinement: consume the async prefetch's mirror-RAW event before
        // any pool read on the main stream (see attend_tile_phase_batched).
        // Device-side wait; host does not block. `begin_step` doesn't touch the
        // pool, so gating here covers every subsequent `step_tile` slot read.
        if self.kv_prefetch_enabled {
            self.kv_prefetch_done.wait(stream)?;
        }
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
            // ATLAS_HSS_COALESCE_BLOCKS: one BlockReadRequest per missing block
            // (see the batched path). Exactly one of reqs/breqs is populated.
            let mut breqs: Vec<BlockReadRequest> = Vec::new();
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
                if self.coalesce_blocks {
                    breqs.push(BlockReadRequest {
                        base_key: GroupKey::new(layer, blk, 0, KvKind::K),
                        dst_dev_ptr: self.pool.slot_dev_ptr(slot),
                    });
                } else {
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
            }
            // #11: skip the no-op read ONLY when prefetch is live (see the
            // fused path and the `kv_prefetch_enabled` field doc). Prefetch-off
            // keeps the unconditional read+sync for byte-for-byte op-identity.
            // `breqs.is_empty() ⟺ reqs.is_empty()`, so the guard is unchanged.
            if !self.kv_prefetch_enabled || !reqs.is_empty() || !breqs.is_empty() {
                if self.coalesce_blocks {
                    self.backend.read_blocks(&breqs, stream)?;
                } else {
                    self.backend.read(&reqs, stream)?;
                }
            }

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
        // ATLAS_HSS_COALESCE_BLOCKS: one BlockReadRequest per assigned block
        // (dst = slot base). Exactly one of reqs/breqs is populated per call.
        let mut breqs: Vec<BlockReadRequest> = Vec::new();
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
            if self.coalesce_blocks {
                breqs.push(BlockReadRequest {
                    base_key: GroupKey::new(layer, blk, 0, KvKind::K),
                    dst_dev_ptr: self.pool.slot_dev_ptr(slot),
                });
            } else {
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
        }
        if !reqs.is_empty() || !breqs.is_empty() {
            // #11: order this evict-victim H2D AFTER the main stream's
            // `step_tile` reads (recorded by the decode loop via
            // `record_kv_read_event`). `stream` here is `prefetch_stream`; the
            // CPU does not block. non-empty reqs/breqs ⟺ ≥1 `assign` ⟺ ≥1 slot
            // about to be overwritten — exactly when the fence is needed. A
            // pins-only prefetch (both empty) never waits.
            self.kv_war_event.wait(stream)?; // #11 WAR — UNCHANGED
            // #11-refinement: fully-async prefetch — enqueue the tier read + H2D
            // on `prefetch_stream` WITHOUT a terminal host stream_sync (the
            // decode host thread must not block on main compute here). Mirror-RAW
            // is then closed device-side: record `kv_prefetch_done` on this
            // in-order stream AFTER the H2D, and the NEXT attend waits it
            // cross-stream on the main stream. Staging/bounce reuse is made safe
            // internally by each async backend (per-bounce copy events + FIFO).
            if self.coalesce_blocks {
                self.backend.read_blocks_async(&breqs, stream)?;
            } else {
                self.backend.read_async(&reqs, stream)?;
            }
            self.kv_prefetch_done.record(stream)?;
        }
        Ok(())
    }

    /// #11: record the WAR fence event on the MAIN compute stream. The decode
    /// loop calls this at each prefetch boundary, before issuing prefetch on
    /// the side stream. In-order stream execution ⇒ this event dominates every
    /// `step_tile` KV read enqueued so far this step (all prior attention
    /// layers), so a subsequent `prefetch_layer` (which `wait`s it on
    /// `prefetch_stream`) cannot overwrite a slot a pending `step_tile` reads.
    pub fn record_kv_read_event(&self, stream: u64) -> Result<()> {
        self.kv_war_event.record(stream)
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

/// ATLAS_HSS_COALESCE_BLOCKS write-buffer assembler (pure, host-testable — this
/// guards the highest-risk correctness item). Produces the single `block_bytes =
/// 2·nkv·group_stride` host image the coalesced offload write pwrites: K stripe
/// `kh` at `kh·group_stride`, V stripe `kh` at `(nkv+kh)·group_stride`, each raw
/// `block_size·head_dim·2`-byte stripe zero-padded up to `group_stride`. This is
/// byte-identical to the on-disk image the `2·nkv` per-head padded writes
/// produce (the per-head path also moves the full padded `group_stride` per op),
/// so a coalesced read-back lands the same bytes — a raw concatenation (no
/// per-stripe padding) would pass the length check but write a WRONG image that
/// only surfaces as garbage KV at attend time.
pub(crate) fn assemble_block_write_buffer(
    nkv: usize,
    group_stride: usize,
    k_stripes: &[Vec<u8>],
    v_stripes: &[Vec<u8>],
) -> Vec<u8> {
    debug_assert_eq!(k_stripes.len(), nkv);
    debug_assert_eq!(v_stripes.len(), nkv);
    let mut buf = vec![0u8; 2 * nkv * group_stride];
    for kh in 0..nkv {
        let k = &k_stripes[kh];
        let v = &v_stripes[kh];
        debug_assert!(k.len() <= group_stride && v.len() <= group_stride);
        let k_off = kh * group_stride;
        let v_off = (nkv + kh) * group_stride;
        buf[k_off..k_off + k.len()].copy_from_slice(k);
        buf[v_off..v_off + v.len()].copy_from_slice(v);
    }
    buf
}

#[cfg(test)]
mod coalesce_write_tests {
    use super::assemble_block_write_buffer;

    /// The assembled buffer is exactly block_bytes; each stripe is recovered at
    /// its group_stride-pitched slot; with group_stride == raw it byte-equals
    /// the K0..K(nkv-1)‖V0..V(nkv-1) concatenation; pad bytes are zero.
    #[test]
    fn assembles_padded_block_image() {
        // Two kv_heads, group_stride 8, raw stripe 6 (2 bytes tail padding).
        let nkv = 2;
        let gs = 8;
        let raw = 6;
        let k: Vec<Vec<u8>> = (0..nkv)
            .map(|h| (0..raw).map(|i| 0x10 * (h as u8 + 1) + i as u8).collect())
            .collect();
        let v: Vec<Vec<u8>> = (0..nkv)
            .map(|h| (0..raw).map(|i| 0x80 + 0x10 * (h as u8) + i as u8).collect())
            .collect();
        let buf = assemble_block_write_buffer(nkv, gs, &k, &v);
        assert_eq!(buf.len(), 2 * nkv * gs);
        for kh in 0..nkv {
            let k_off = kh * gs;
            let v_off = (nkv + kh) * gs;
            assert_eq!(&buf[k_off..k_off + raw], k[kh].as_slice());
            assert_eq!(&buf[v_off..v_off + raw], v[kh].as_slice());
            // Tail padding is zero.
            assert!(buf[k_off + raw..k_off + gs].iter().all(|&b| b == 0));
            assert!(buf[v_off + raw..v_off + gs].iter().all(|&b| b == 0));
        }
    }

    /// group_stride == raw ⇒ no padding ⇒ pure ordered concatenation
    /// K0‖K1‖…‖V0‖V1‖… (the Holo case bs*hd*2 == gs == 4096).
    #[test]
    fn unpadded_is_plain_concatenation() {
        let nkv = 3;
        let gs = 4;
        let k: Vec<Vec<u8>> = (0..nkv).map(|h| vec![h as u8; gs]).collect();
        let v: Vec<Vec<u8>> = (0..nkv).map(|h| vec![0xF0 | h as u8; gs]).collect();
        let buf = assemble_block_write_buffer(nkv, gs, &k, &v);
        let mut expected = Vec::new();
        for s in &k {
            expected.extend_from_slice(s);
        }
        for s in &v {
            expected.extend_from_slice(s);
        }
        assert_eq!(buf, expected);
    }
}
