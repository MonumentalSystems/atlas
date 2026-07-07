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
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Pre-allocated GPU memory pool for SSM state snapshots.
///
/// Each snapshot slot stores a copy of h_state + conv_state for all SSM layers
/// at a specific point in a token sequence.
///
/// The pool serves **two** independent consumers from one set of GPU
/// allocations (SSOT — one snapshot mechanism, one D2D copy primitive):
///
/// 1. **Marconi prefix caching** — the LRU-managed `[0, num_slots)` slot
///    region, allocated/freed via [`save`](Self::save) / [`free`](Self::free)
///    against the `free_slots` list. When a prefix cache hit occurs the
///    snapshot is restored to skip SSM recompute for cached tokens.
///
/// 2. **Phase-C decode-time boundary rollback** — a *separate*,
///    deterministically-addressed `[0, decode_ring_slots)` region (per
///    active sequence). No free list: ring slot `r` for SSM-pool
///    sequence slot `s` lives at flat index `s * ring_slots + r`, so a
///    sequence's snapshots never collide with another's and never
///    contend with Marconi's LRU slots. Sized for `max_batch_size`
///    sequences so the watchdog rollback always has capacity.
pub(crate) struct SsmSnapshotPool {
    pub(super) h_snapshots: Vec<DevicePtr>,
    pub(super) conv_snapshots: Vec<DevicePtr>,
    pub(super) free_slots: Mutex<Vec<usize>>,
    pub(super) num_slots: usize,
    pub(super) h_bytes: usize,
    pub(super) conv_bytes: usize,
    pub(super) num_ssm_layers: usize,
    /// Maps snapshot_slot_id → session_hash for session-scoped isolation.
    /// When restoring, skip snapshots that belong to a different session.
    pub(super) session_tags: Mutex<std::collections::HashMap<usize, u64>>,
    /// Decode-rollback region: `h_snapshots` for the Phase-C ring.
    /// Layout per layer: `[max_batch_size * decode_ring_slots * h_bytes]`.
    /// Empty when `decode_ring_slots == 0`.
    pub(super) decode_h_snapshots: Vec<DevicePtr>,
    /// Decode-rollback region: `conv_snapshots` for the Phase-C ring.
    pub(super) decode_conv_snapshots: Vec<DevicePtr>,
    /// Number of decode-rollback ring slots reserved per active sequence.
    /// 0 disables the decode-rollback region entirely.
    pub(super) decode_ring_slots: usize,
    /// Number of active-sequence slots the decode region is sized for
    /// (equals `max_batch_size`). A sequence's SSM-pool `slot_idx` must
    /// be `< decode_max_seqs` to use the decode region.
    pub(super) decode_max_seqs: usize,
    /// Last-token post-final-norm hidden state for each Marconi snapshot
    /// slot. Single buffer of `num_slots * hidden_bytes`; slot `s` lives
    /// at `offset(s * hidden_bytes)`. NULL when Marconi is disabled.
    ///
    /// Marconi's leaf snapshot stores SSM recurrent state *after* the last
    /// token (state@N). On an exact full-prompt hit the engine must
    /// produce the first generated token's logits — which normally come
    /// from re-running the last prompt token's forward. For SSM layers
    /// that re-run would apply the last token's recurrent update a second
    /// time on top of state@N (double-advance → corruption). Instead we
    /// stash the last token's post-norm hidden here at save time and feed
    /// it straight to `lm_head` on the hit, skipping any SSM re-run.
    pub(super) hidden_snapshot: DevicePtr,
    /// Byte size of one slot's last-token hidden (`hidden_size * 2`, BF16).
    pub(super) hidden_bytes: usize,
    /// Marconi slots that currently hold a valid `hidden_snapshot` entry
    /// (only leaf saves populate it; intermediate checkpoints do not).
    pub(super) slot_has_hidden: Mutex<std::collections::HashSet<usize>>,
}

impl SsmSnapshotPool {
    /// Build the snapshot pool.
    ///
    /// `num_slots` sizes the Marconi LRU region; `decode_ring_slots` ×
    /// `decode_max_seqs` sizes the Phase-C decode-rollback region. A
    /// pool with `num_slots == 0` but `decode_ring_slots > 0` is valid
    /// (decode rollback enabled, Marconi caching disabled) and vice
    /// versa — the two regions are independent.
    pub(super) fn new(
        num_slots: usize,
        h_bytes: usize,
        conv_bytes: usize,
        num_ssm_layers: usize,
        decode_ring_slots: usize,
        decode_max_seqs: usize,
        hidden_bytes: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let decode_enabled = num_ssm_layers > 0 && decode_ring_slots > 0 && decode_max_seqs > 0;
        let marconi_enabled = num_ssm_layers > 0 && num_slots > 0;

        if !marconi_enabled && !decode_enabled {
            return Ok(Self {
                h_snapshots: Vec::new(),
                conv_snapshots: Vec::new(),
                free_slots: Mutex::new(Vec::new()),
                num_slots: 0,
                h_bytes,
                conv_bytes,
                num_ssm_layers,
                session_tags: Mutex::new(std::collections::HashMap::new()),
                decode_h_snapshots: Vec::new(),
                decode_conv_snapshots: Vec::new(),
                decode_ring_slots: 0,
                decode_max_seqs: 0,
                hidden_snapshot: DevicePtr::NULL,
                hidden_bytes,
                slot_has_hidden: Mutex::new(std::collections::HashSet::new()),
            });
        }

        let mut h_snapshots = Vec::new();
        let mut conv_snapshots = Vec::new();
        let mut hidden_snapshot = DevicePtr::NULL;
        if marconi_enabled {
            for _ in 0..num_ssm_layers {
                h_snapshots.push(gpu.alloc(num_slots * h_bytes)?);
                conv_snapshots.push(gpu.alloc(num_slots * conv_bytes)?);
            }
            hidden_snapshot = gpu.alloc(num_slots * hidden_bytes)?;
        }

        let mut decode_h_snapshots = Vec::new();
        let mut decode_conv_snapshots = Vec::new();
        let decode_region = if decode_enabled {
            decode_max_seqs * decode_ring_slots
        } else {
            0
        };
        if decode_enabled {
            for _ in 0..num_ssm_layers {
                decode_h_snapshots.push(gpu.alloc(decode_region * h_bytes)?);
                decode_conv_snapshots.push(gpu.alloc(decode_region * conv_bytes)?);
            }
        }

        let free_slots: Vec<usize> = if marconi_enabled {
            (0..num_slots).rev().collect()
        } else {
            Vec::new()
        };
        let marconi_mb = num_ssm_layers * num_slots * (h_bytes + conv_bytes) / (1024 * 1024);
        let decode_mb = num_ssm_layers * decode_region * (h_bytes + conv_bytes) / (1024 * 1024);
        tracing::info!(
            "SSM snapshot pool: Marconi {num_slots} slots ({marconi_mb} MB), \
             decode-rollback {decode_ring_slots} slots × {decode_max_seqs} seqs \
             ({decode_mb} MB), {num_ssm_layers} layers",
        );

        Ok(Self {
            h_snapshots,
            conv_snapshots,
            free_slots: Mutex::new(free_slots),
            num_slots: if marconi_enabled { num_slots } else { 0 },
            h_bytes,
            conv_bytes,
            num_ssm_layers,
            session_tags: Mutex::new(std::collections::HashMap::new()),
            decode_h_snapshots,
            decode_conv_snapshots,
            decode_ring_slots: if decode_enabled { decode_ring_slots } else { 0 },
            decode_max_seqs: if decode_enabled { decode_max_seqs } else { 0 },
            hidden_snapshot,
            hidden_bytes,
            slot_has_hidden: Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Marconi prefix-cache region availability.
    pub(super) fn is_enabled(&self) -> bool {
        self.num_slots > 0
    }

    /// Phase-C decode-rollback region availability.
    pub(super) fn decode_rollback_enabled(&self) -> bool {
        self.decode_ring_slots > 0 && !self.decode_h_snapshots.is_empty()
    }

    /// Save the SSM state of pool slot `ssm_slot` into the decode-rollback
    /// ring slot `(ssm_slot, ring_slot)`. Deterministic addressing — no
    /// free list, no eviction. Errors if the decode region is disabled
    /// or the indices are out of the reserved range (fail fast — a
    /// silent skip would leave the watchdog rollback unable to undo SSM
    /// state, corrupting every subsequent decode).
    pub(super) fn save_decode(
        &self,
        ssm_slot: usize,
        ring_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let flat = self.decode_flat_index(ssm_slot, ring_slot)?;
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                main_pool.h_state(i, ssm_slot),
                self.decode_h_snapshots[i].offset(flat * self.h_bytes),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                main_pool.conv_state(i, ssm_slot),
                self.decode_conv_snapshots[i].offset(flat * self.conv_bytes),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Restore the SSM state of pool slot `ssm_slot` from the
    /// decode-rollback ring slot `(ssm_slot, ring_slot)`.
    pub(super) fn restore_decode(
        &self,
        ssm_slot: usize,
        ring_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let flat = self.decode_flat_index(ssm_slot, ring_slot)?;
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.decode_h_snapshots[i].offset(flat * self.h_bytes),
                main_pool.h_state(i, ssm_slot),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.decode_conv_snapshots[i].offset(flat * self.conv_bytes),
                main_pool.conv_state(i, ssm_slot),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Flat index into the decode-rollback region, with bounds checks.
    fn decode_flat_index(&self, ssm_slot: usize, ring_slot: usize) -> Result<usize> {
        if !self.decode_rollback_enabled() {
            bail!("SSM decode-rollback region not allocated");
        }
        if ssm_slot >= self.decode_max_seqs {
            bail!(
                "SSM decode-rollback: ssm_slot {ssm_slot} >= reserved {} seqs",
                self.decode_max_seqs
            );
        }
        if ring_slot >= self.decode_ring_slots {
            bail!(
                "SSM decode-rollback: ring_slot {ring_slot} >= reserved {} slots",
                self.decode_ring_slots
            );
        }
        Ok(ssm_slot * self.decode_ring_slots + ring_slot)
    }

    /// Save SSM state from active pool slot into a snapshot slot.
    /// Returns `None` if no free snapshot slots are available.
    /// Tags the snapshot with `session_hash` for session-scoped isolation.
    pub(super) fn save(
        &self,
        ssm_slot: usize,
        session_hash: u64,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<usize>> {
        if !self.is_enabled() {
            return Ok(None);
        }
        let snap_slot = match self.free_slots.lock().pop() {
            Some(s) => s,
            None => return Ok(None),
        };
        // Reusing a freed slot: drop any stale last-token hidden tag. The
        // caller re-populates it via `save_hidden` for leaf snapshots only.
        self.slot_has_hidden.lock().remove(&snap_slot);
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                main_pool.h_state(i, ssm_slot),
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                main_pool.conv_state(i, ssm_slot),
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                self.conv_bytes,
                stream,
            )?;
        }
        if session_hash != 0 {
            self.session_tags.lock().insert(snap_slot, session_hash);
        }
        Ok(Some(snap_slot))
    }

    /// Check if a snapshot belongs to the given session.
    /// Returns true if: session tracking is disabled (hash=0), no tag exists, or tags match.
    pub(super) fn session_matches(&self, snap_slot: usize, session_hash: u64) -> bool {
        if session_hash == 0 {
            return true;
        } // Legacy: no session tracking
        let tags = self.session_tags.lock();
        match tags.get(&snap_slot) {
            None => true, // Untagged snapshot (pre-session-manager) — allow
            Some(&tag) => tag == session_hash,
        }
    }

    /// Restore SSM state from a snapshot slot into an active pool slot.
    pub(super) fn restore(
        &self,
        snap_slot: usize,
        ssm_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                main_pool.h_state(i, ssm_slot),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                main_pool.conv_state(i, ssm_slot),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Return a snapshot slot to the free list.
    pub(super) fn free(&self, snap_slot: usize) {
        self.slot_has_hidden.lock().remove(&snap_slot);
        self.free_slots.lock().push(snap_slot);
    }

    /// Claim an immediately-free Marconi slot (no eviction). `None` when the
    /// pool is full. The claimed slot must be `free`d if the caller doesn't use
    /// it (e.g. a fault-in miss).
    pub(super) fn try_pop_free_slot(&self) -> Option<usize> {
        if !self.is_enabled() {
            return None;
        }
        self.free_slots.lock().pop()
    }

    /// Acquire a Marconi slot for a **fault-in target** (Phase 1b), spilling a
    /// resident victim to make room when the pool is full. Under a small pool +
    /// heavy churn the free list is usually empty at warm-turn lookup time, so
    /// without this the fault-in silently degrades to recompute (measured: only
    /// 13 of 43 tiered hits completed a fault-in at `--ssm-cache-slots 4`).
    ///
    /// Order: pop a free slot; else spill the session-aware victim
    /// (`evict_snapshot_to_tier` keeps its entry findable → still faultable
    /// later) to free its slot, then pop that. The victim is always a RESIDENT
    /// entry (`skip_tiered`), never the tiered entry we're about to fault in.
    /// `None` only if nothing is resident to evict (every slot mid-flight).
    pub(super) fn acquire_or_spill_slot(
        &self,
        prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
        store: &dyn super::ssm_tier::SnapshotBlobStore,
        gpu: &dyn GpuBackend,
    ) -> Option<usize> {
        if let Some(s) = self.try_pop_free_slot() {
            return Some(s);
        }
        let (victim_slot, key) = prefix_cache.evict_snapshot_to_tier()?;
        let stream = gpu.default_stream();
        if let Err(e) = self.spill_slot(victim_slot, key, store, gpu, stream) {
            tracing::warn!("SSM spill during fault-in acquire failed ({e:#}); freeing slot anyway");
        }
        self.free(victim_slot);
        self.try_pop_free_slot()
    }

    /// Bytes in one slot's full spill blob: every SSM layer's `h` + `conv`
    /// state, laid out `[h_0 conv_0 h_1 conv_1 … h_{L-1} conv_{L-1}]`.
    pub(super) fn spill_blob_bytes(&self) -> usize {
        self.num_ssm_layers * (self.h_bytes + self.conv_bytes)
    }

    /// **Spill** Marconi slot `snap_slot` to the byte tier (Phase 1,
    /// spill-not-drop): gather the slot's scattered per-layer `(h,conv)` device
    /// chunks D2H into one host blob and `put` it under `key` (the snapshot's
    /// prefix hash). Returns whether the tier accepted the blob — `false` (tier
    /// full / disabled pool) means the caller should fall back to a plain drop.
    ///
    /// Ordering: a single `synchronize(stream)` first drains any in-flight D2D
    /// `save` into this slot (which the caller enqueued on `stream`) before the
    /// D2H read, so we never spill a half-written snapshot — the read-direction
    /// half of the cross-stream hazard the plan flags. (The caller still owns
    /// ordering the *slot reuse* after this spill; see Phase 1b.)
    pub(super) fn spill_slot(
        &self,
        snap_slot: usize,
        key: u64,
        store: &dyn super::ssm_tier::SnapshotBlobStore,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        if !self.is_enabled() {
            return Ok(false);
        }
        gpu.synchronize(stream)?; // drain the pending save into this slot
        let mut blob = vec![0u8; self.spill_blob_bytes()];
        let per_layer = self.h_bytes + self.conv_bytes;
        for i in 0..self.num_ssm_layers {
            let off = i * per_layer;
            gpu.copy_d2h(
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                &mut blob[off..off + self.h_bytes],
            )?;
            gpu.copy_d2h(
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                &mut blob[off + self.h_bytes..off + per_layer],
            )?;
        }
        store.put(key, &blob)
    }

    /// **Fault in** a spilled snapshot for `key` into Marconi slot `snap_slot`:
    /// fetch the host blob and scatter it H2D back into the slot's per-layer
    /// `(h,conv)` chunks. Returns `false` if the tier has no blob for `key`
    /// (caller recomputes) — the correct miss degradation.
    ///
    /// A trailing `synchronize(stream)` guarantees the H2D scatter has
    /// committed before the caller issues a `restore` (D2D slot→main pool) that
    /// reads this slot — the write-direction half of the ordering hazard.
    pub(super) fn fault_in_slot(
        &self,
        snap_slot: usize,
        key: u64,
        store: &dyn super::ssm_tier::SnapshotBlobStore,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        if !self.is_enabled() {
            return Ok(false);
        }
        let mut blob = vec![0u8; self.spill_blob_bytes()];
        if !store.get(key, &mut blob)? {
            return Ok(false);
        }
        let per_layer = self.h_bytes + self.conv_bytes;
        for i in 0..self.num_ssm_layers {
            let off = i * per_layer;
            gpu.copy_h2d_async(
                &blob[off..off + self.h_bytes],
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                stream,
            )?;
            gpu.copy_h2d_async(
                &blob[off + self.h_bytes..off + per_layer],
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                stream,
            )?;
        }
        gpu.synchronize(stream)?; // commit before caller's restore reads the slot
        Ok(true)
    }

    /// Stash the last-token post-final-norm hidden (`hidden_bytes`, BF16)
    /// for a leaf snapshot slot. Used so an exact full-prompt hit can emit
    /// the first token's logits via `lm_head` without re-running the last
    /// token through the SSM layers (which would double-advance the
    /// recurrent state). Only leaf saves call this; intermediate
    /// checkpoints leave the slot untagged.
    pub(super) fn save_hidden(
        &self,
        snap_slot: usize,
        last_hidden: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if !self.is_enabled() || self.hidden_snapshot.is_null() {
            return Ok(());
        }
        gpu.copy_d2d_async(
            last_hidden,
            self.hidden_snapshot.offset(snap_slot * self.hidden_bytes),
            self.hidden_bytes,
            stream,
        )?;
        self.slot_has_hidden.lock().insert(snap_slot);
        Ok(())
    }

    /// Whether `snap_slot` holds a valid last-token hidden (leaf snapshot).
    pub(super) fn has_hidden(&self, snap_slot: usize) -> bool {
        self.slot_has_hidden.lock().contains(&snap_slot)
    }

    /// Restore the stashed last-token hidden of `snap_slot` into `dst`
    /// (the `norm_output` buffer), ready for `lm_head`.
    pub(super) fn restore_hidden(
        &self,
        snap_slot: usize,
        dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if self.hidden_snapshot.is_null() {
            bail!("SSM hidden snapshot region not allocated");
        }
        gpu.copy_d2d_async(
            self.hidden_snapshot.offset(snap_slot * self.hidden_bytes),
            dst,
            self.hidden_bytes,
            stream,
        )?;
        Ok(())
    }

    /// Try to reclaim a snapshot slot by evicting a snapshot from the prefix
    /// cache's snapshot index. Snapshots are decoupled from tree nodes, so this
    /// directly frees a slot without evicting KV blocks.
    ///
    /// Phase 1b: when `tier` is `Some` (`ATLAS_SSM_TIER`), the victim is
    /// **spilled** — its bytes moved to the tier and its index entry kept
    /// (findable), so a warm turn faults it back instead of recomputing — before
    /// the slot is freed for reuse. When `tier` is `None` the victim is dropped
    /// exactly as before (byte-identical default path). Returns whether a slot
    /// was reclaimed.
    pub(super) fn reclaim_from_cache(
        &self,
        prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
        _kv_cache: &mut PagedKvCache,
        tier: Option<&dyn super::ssm_tier::SnapshotBlobStore>,
        gpu: &dyn GpuBackend,
    ) -> bool {
        if let Some(store) = tier {
            // Spill-not-drop. Marconi saves are enqueued on the default stream,
            // so draining it inside `spill_slot` guarantees we never D2H a
            // half-written victim slot (the read half of the ordering hazard).
            if let Some((slot, key)) = prefix_cache.evict_snapshot_to_tier() {
                let stream = gpu.default_stream();
                match self.spill_slot(slot, key, store, gpu, stream) {
                    Ok(true) => {}
                    Ok(false) => {
                        // Unbounded tier never rejects; a bounded one could. The
                        // entry is now marked tiered but holds no bytes → a later
                        // fault-in cleanly misses (recompute). Bounded-tier
                        // drop-on-reject is a follow-up.
                        tracing::warn!("SSM spill tier refused a blob; entry will miss on fault-in");
                    }
                    Err(e) => {
                        tracing::warn!("SSM spill failed ({e:#}); freeing slot, entry will miss");
                    }
                }
                self.free(slot); // slot reusable regardless; bytes are (or aren't) in the tier
                return true;
            }
            return false;
        }
        if let Some(snap) = prefix_cache.evict_snapshot_lru() {
            self.free(snap);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tier_tests {
    use super::*;
    use crate::model::ssm_tier::{MemBlobStore, SnapshotBlobStore};
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// Build a small Marconi-only pool (no decode-rollback region).
    fn pool(gpu: &dyn GpuBackend, slots: usize, layers: usize) -> SsmSnapshotPool {
        SsmSnapshotPool::new(
            slots, /*h_bytes*/ 32, /*conv_bytes*/ 16, layers, /*decode_ring*/ 0,
            /*decode_max_seqs*/ 0, /*hidden_bytes*/ 8, gpu,
        )
        .unwrap()
    }

    /// Fill slot `s`'s per-layer (h,conv) device chunks with a pattern unique
    /// per (layer, field) so a mis-scatter would be caught.
    fn write_pattern(p: &SsmSnapshotPool, gpu: &dyn GpuBackend, s: usize) {
        for i in 0..p.num_ssm_layers {
            let h = vec![(0x10 + i) as u8; p.h_bytes];
            let c = vec![(0x80 + i) as u8; p.conv_bytes];
            gpu.copy_h2d(&h, p.h_snapshots[i].offset(s * p.h_bytes)).unwrap();
            gpu.copy_h2d(&c, p.conv_snapshots[i].offset(s * p.conv_bytes)).unwrap();
        }
    }

    fn read_slot(p: &SsmSnapshotPool, gpu: &dyn GpuBackend, s: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut hs = Vec::new();
        let mut cs = Vec::new();
        for i in 0..p.num_ssm_layers {
            let mut h = vec![0u8; p.h_bytes];
            let mut c = vec![0u8; p.conv_bytes];
            gpu.copy_d2h(p.h_snapshots[i].offset(s * p.h_bytes), &mut h).unwrap();
            gpu.copy_d2h(p.conv_snapshots[i].offset(s * p.conv_bytes), &mut c).unwrap();
            hs.push(h);
            cs.push(c);
        }
        (hs, cs)
    }

    /// The headline invariant: spill a slot's scattered state to the tier, then
    /// fault it back into a DIFFERENT slot — the recurrent state is bit-for-bit
    /// preserved. This is "spill-not-drop" proven end-to-end at the pool layer.
    #[test]
    fn spill_then_fault_in_preserves_bytes() {
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, /*slots*/ 4, /*layers*/ 3);
        let store = MemBlobStore::new(0);
        let key = 0xABCD_1234;

        write_pattern(&p, &gpu, /*src*/ 1);
        let want = read_slot(&p, &gpu, 1);

        assert!(p.spill_slot(1, key, &store, &gpu, 0).unwrap());
        assert_eq!(store.len(), 1);
        assert_eq!(store.bytes_resident(), p.spill_blob_bytes());

        // Fault into slot 2 (which is still zeroed) and compare to slot 1.
        assert!(p.fault_in_slot(2, key, &store, &gpu, 0).unwrap());
        let got = read_slot(&p, &gpu, 2);
        assert_eq!(got, want, "faulted-in slot must equal the spilled slot bit-for-bit");
    }

    /// Faulting an absent key is a clean miss (caller recomputes), not an error.
    #[test]
    fn fault_in_absent_key_is_miss() {
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, 4, 2);
        let store = MemBlobStore::new(0);
        assert!(!p.fault_in_slot(0, /*absent*/ 999, &store, &gpu, 0).unwrap());
    }

    /// Blob size accounts for every layer's h+conv.
    #[test]
    fn spill_blob_bytes_matches_layout() {
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, 2, 5);
        assert_eq!(p.spill_blob_bytes(), 5 * (32 + 16));
    }

    /// Full-pool fault-in: when no slot is free, `acquire_or_spill_slot` spills a
    /// resident victim (to the tier, keeping it faultable) and hands back its
    /// freed slot — so a warm tiered hit isn't lost to a busy pool.
    #[test]
    fn acquire_or_spill_frees_a_slot_under_full_pool() {
        use spark_runtime::prefix_cache::PrefixCache;
        use spark_runtime::radix_tree::RadixTree;

        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, /*slots*/ 2, /*layers*/ 2);
        let store = MemBlobStore::new(0);
        let tree = RadixTree::new();

        // Register two resident snapshots (slots 0 and 1) for two prefixes, then
        // drain the free list so the pool is full.
        let toks_a: Vec<u32> = (0..16).collect();
        let toks_b: Vec<u32> = (100..116).collect();
        tree.insert_with_snapshot(&toks_a, &[10], &[], 16, /*slot*/ 0, /*sess*/ 7, 0, 0);
        tree.insert_with_snapshot(&toks_b, &[20], &[], 16, /*slot*/ 1, /*sess*/ 9, 0, 0);
        assert!(p.try_pop_free_slot().is_some());
        assert!(p.try_pop_free_slot().is_some());
        assert_eq!(p.try_pop_free_slot(), None, "pool is now full");

        // Acquire must spill a victim and return its slot.
        let slot = p
            .acquire_or_spill_slot(&tree, &store, &gpu)
            .expect("a resident victim exists to spill");
        assert!(slot == 0 || slot == 1);
        assert_eq!(store.len(), 1, "the evicted victim was spilled, not dropped");
        // The other snapshot stays resident (drop path can still free it).
        assert!(tree.evict_snapshot_lru().is_some());
    }

    /// The integration invariant: the tier is keyed by prefix, INDEPENDENT of
    /// HBM slot lifecycle. Spill snapshot A from slot 0, recycle slot 0 for a
    /// different snapshot B, spill B under its own key, then fault BOTH back —
    /// each must recover its own bytes. This is exactly what the Phase-1b
    /// serving wiring creates: `evict_to_tier` frees a slot that `save` then
    /// reuses, and a later warm turn faults the spilled key into a fresh slot.
    #[test]
    fn tier_survives_slot_recycling() {
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, /*slots*/ 3, /*layers*/ 2);
        let store = MemBlobStore::new(0);
        let (key_a, key_b) = (0xAAAA, 0xBBBB);

        // Snapshot A lives in slot 0; spill it.
        write_pattern(&p, &gpu, 0);
        let want_a = read_slot(&p, &gpu, 0);
        assert!(p.spill_slot(0, key_a, &store, &gpu, 0).unwrap());

        // Recycle slot 0 for a DIFFERENT snapshot B (distinct bytes), spill it.
        for i in 0..p.num_ssm_layers {
            let h = vec![0xEE; p.h_bytes];
            let c = vec![0xDD; p.conv_bytes];
            gpu.copy_h2d(&h, p.h_snapshots[i].offset(0)).unwrap();
            gpu.copy_h2d(&c, p.conv_snapshots[i].offset(0)).unwrap();
        }
        let want_b = read_slot(&p, &gpu, 0);
        assert_ne!(want_a, want_b, "B must differ from A for the test to bite");
        assert!(p.spill_slot(0, key_b, &store, &gpu, 0).unwrap());
        assert_eq!(store.len(), 2);

        // Fault each key into fresh slots — bytes recovered independently.
        assert!(p.fault_in_slot(1, key_a, &store, &gpu, 0).unwrap());
        assert!(p.fault_in_slot(2, key_b, &store, &gpu, 0).unwrap());
        assert_eq!(read_slot(&p, &gpu, 1), want_a, "key A recovered after slot recycle");
        assert_eq!(read_slot(&p, &gpu, 2), want_b, "key B recovered");
    }
}
