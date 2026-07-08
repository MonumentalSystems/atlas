// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::decode_ring_manager::{
    DecodeRingManager, RestoreDecision, SaveDecision, SpillCommit, SpillReq,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_tier::SnapshotBlobStore;
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
    /// ROLLING decode tier (`ATLAS_SSM_DECODE_RING_ROLL`): residency state
    /// machine shared with the async spill worker. `None` ⇒ the decode region is
    /// the flat HBM (or UMA) ring, byte-identical to the pre-rolling build.
    pub(super) decode_rolling: Option<Arc<Mutex<DecodeRingManager>>>,
    /// Physical HBM lanes per sequence: `hot_lanes + DECODE_SPILL_MARGIN` in
    /// rolling mode, else `decode_ring_slots`. Drives decode-region frame
    /// addressing and MUST equal what the preflight reservation mirrors.
    pub(super) decode_l_phys: usize,
    /// Cold tier for spilled decode boundaries (non-dropping). `Some` in rolling
    /// mode once `attach_decode_spiller` runs.
    decode_store: Option<Arc<dyn SnapshotBlobStore>>,
    /// Async spill worker handle (rolling mode). Drains aged hot lanes off the
    /// decode critical path.
    decode_spiller: Option<DecodeSpiller>,
}

/// A queued cold-tier spill for the async worker.
struct SpillJob {
    req: SpillReq,
    /// Event recorded on the save stream; the worker's stream waits on it so the
    /// D2H gather never races the boundary's D2D write into the same lane.
    wait_event: u64,
}

/// Owns the async decode-spill worker thread + its channel. Dropping it hangs up
/// the channel, ending the worker.
struct DecodeSpiller {
    /// `mpsc::Sender` is `!Sync`; the pool is shared `&self` across threads, so
    /// the sender is `Mutex`-guarded (a send is a cheap enqueue).
    tx: Mutex<mpsc::Sender<SpillJob>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for DecodeSpiller {
    fn drop(&mut self) {
        // Drop the sender first (hangs up the channel) so the worker's `recv`
        // returns `Err` and the loop exits, then join.
        {
            let (tx, _) = mpsc::channel();
            *self.tx.lock() = tx; // replace with a dangling sender → original dropped
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
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
                decode_rolling: None,
                decode_l_phys: 0,
                decode_store: None,
                decode_spiller: None,
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
        // ROLLING tier (ATLAS_SSM_DECODE_RING_ROLL): shrink the HBM decode region
        // to `hot_lanes + margin` physical lanes per seq (+ shared fault scratch)
        // and drive residency via the manager. Supersedes the flat UMA A/B path
        // when both are set (rolling keeps the hot write in HBM — the point).
        let rolling = decode_enabled && atlas_kernels::decode_ring_rolling_enabled();
        let hot_lanes = atlas_kernels::decode_hot_lanes_runtime();
        let rolling_mgr = if rolling {
            Some(DecodeRingManager::new(
                decode_ring_slots,
                hot_lanes,
                atlas_kernels::DECODE_SPILL_MARGIN,
                atlas_kernels::DECODE_FAULT_SCRATCH,
                decode_max_seqs,
                atlas_kernels::DECODE_DOMAIN,
            ))
        } else {
            None
        };
        let decode_l_phys =
            atlas_kernels::decode_hbm_lanes_per_seq(rolling, hot_lanes);
        let decode_region = if !decode_enabled {
            0
        } else if let Some(m) = &rolling_mgr {
            m.total_frames()
        } else {
            decode_max_seqs * decode_ring_slots
        };
        // The decode-rollback ring scales as `decode_max_seqs × ring_slots ×
        // layers × (h+conv)` — ~4 GB at C=8, ~32 GB at C=64 — yet it is touched
        // only at sentence boundaries (one D2D snapshot copy) and read only on a
        // rare watchdog rollback. `ATLAS_SSM_DECODE_RING_UMA=1` allocates it in
        // GB10 unified (managed) memory instead of the HBM/GPU pool, moving that
        // budget off the KV-competing pool onto the same physical LPDDR. The
        // device-side save/restore copies address a managed `DevicePtr`
        // identically, so nothing downstream changes.
        //
        // MEASURED COST (Holo-3.1-35B, C=8, decode-heavy 256-tok gens, 2026-07-08):
        // ~5.8% decode throughput (37.31 → 35.14 tok/s) — the per-boundary D2D
        // copy into managed memory is slower than HBM→HBM (cuMemAllocManaged is
        // not the pinned-UMA the KV zero-copy path uses). So this is OFF by
        // default: it's the escape valve for high-C, where the ring would
        // otherwise be tens of GB of HBM (32 GB at C=64) — trading ~6% decode for
        // the HBM to run that concurrency at all (or to give it to KV) is the win
        // there, NOT at low C where HBM isn't the binding constraint.
        let ring_uma = std::env::var_os("ATLAS_SSM_DECODE_RING_UMA").is_some();
        if decode_enabled {
            for _ in 0..num_ssm_layers {
                // Rolling ALWAYS lands in HBM (the hot per-boundary write must be
                // fast D2D — the whole reason UMA regressed 5.8%). UMA is the flat
                // A/B knob, honored only when rolling is off.
                let (h, c) = if ring_uma && !rolling {
                    (
                        gpu.alloc_managed(decode_region * h_bytes)?,
                        gpu.alloc_managed(decode_region * conv_bytes)?,
                    )
                } else {
                    (
                        gpu.alloc(decode_region * h_bytes)?,
                        gpu.alloc(decode_region * conv_bytes)?,
                    )
                };
                decode_h_snapshots.push(h);
                decode_conv_snapshots.push(c);
            }
        }

        let free_slots: Vec<usize> = if marconi_enabled {
            (0..num_slots).rev().collect()
        } else {
            Vec::new()
        };
        let marconi_mb = num_ssm_layers * num_slots * (h_bytes + conv_bytes) / (1024 * 1024);
        let decode_mb = num_ssm_layers * decode_region * (h_bytes + conv_bytes) / (1024 * 1024);
        let decode_mode = if rolling {
            "HBM ROLLING"
        } else if ring_uma {
            "UMA/managed"
        } else {
            "HBM"
        };
        tracing::info!(
            "SSM snapshot pool: Marconi {num_slots} slots ({marconi_mb} MB), \
             decode-rollback {decode_ring_slots} logical slots × {decode_max_seqs} seqs, \
             {decode_l_phys} HBM lanes/seq ({decode_mb} MB, {decode_mode}), \
             {num_ssm_layers} layers",
        );
        if rolling {
            tracing::info!(
                "SSM decode ROLLING tier: {hot_lanes} hot + {} margin lanes/seq, \
                 {} shared fault-scratch — HBM capped at {decode_max_seqs}×{decode_l_phys}×blob \
                 (cold residue spills to ATLAS_SSM_DECODE_TIER)",
                atlas_kernels::DECODE_SPILL_MARGIN,
                atlas_kernels::DECODE_FAULT_SCRATCH,
            );
        }

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
            decode_rolling: rolling_mgr.map(|m| Arc::new(Mutex::new(m))),
            decode_l_phys: if decode_enabled { decode_l_phys } else { 0 },
            decode_store: None,
            decode_spiller: None,
        })
    }

    /// Whether the rolling decode tier is active on this pool.
    pub(super) fn decode_rolling_enabled(&self) -> bool {
        self.decode_rolling.is_some()
    }

    /// Attach the cold tier + spawn the async spill worker for the rolling decode
    /// tier. Called once from model init (rolling mode only). `gpu`/`store` are
    /// owned `Arc`s the worker thread keeps. No-op if rolling is off.
    pub(super) fn attach_decode_spiller(
        &mut self,
        gpu: Arc<dyn GpuBackend>,
        store: Arc<dyn SnapshotBlobStore>,
    ) -> Result<()> {
        let Some(mgr) = self.decode_rolling.clone() else {
            return Ok(());
        };
        self.decode_store = Some(store.clone());
        let (tx, rx) = mpsc::channel::<SpillJob>();
        // Cloned handles the worker owns for the lifetime of the pool.
        let h_ptrs: Vec<DevicePtr> = self.decode_h_snapshots.clone();
        let c_ptrs: Vec<DevicePtr> = self.decode_conv_snapshots.clone();
        let h_bytes = self.h_bytes;
        let conv_bytes = self.conv_bytes;
        let num_layers = self.num_ssm_layers;
        let l_phys = self.decode_l_phys;
        let blob_bytes = self.spill_blob_bytes();
        let spill_stream = gpu.create_stream().unwrap_or(0);
        let handle = std::thread::Builder::new()
            .name("atlas-decode-spill".into())
            .spawn(move || {
                if let Err(e) = gpu.bind_to_thread() {
                    tracing::error!("decode-spill worker: bind_to_thread failed: {e:#}");
                    return;
                }
                while let Ok(job) = rx.recv() {
                    let req = job.req;
                    // Order the gather after the boundary write into this lane.
                    let _ = gpu.stream_wait_event(spill_stream, job.wait_event);
                    let frame = req.seq * l_phys + req.lane;
                    let mut blob = vec![0u8; blob_bytes];
                    let per_layer = h_bytes + conv_bytes;
                    let mut ok = true;
                    for i in 0..num_layers {
                        let off = i * per_layer;
                        if gpu
                            .copy_d2h_on_stream(
                                h_ptrs[i].offset(frame * h_bytes),
                                &mut blob[off..off + h_bytes],
                                spill_stream,
                            )
                            .is_err()
                            || gpu
                                .copy_d2h_on_stream(
                                    c_ptrs[i].offset(frame * conv_bytes),
                                    &mut blob[off + h_bytes..off + per_layer],
                                    spill_stream,
                                )
                                .is_err()
                        {
                            ok = false;
                            break;
                        }
                    }
                    let put_ok = ok && store.put(req.cold_key, &blob).unwrap_or(false);
                    if !put_ok {
                        // A non-dropping store (host-RAM / peer / a correctly-sized
                        // NVMe arena) must never reach here. If it does, the bytes
                        // did NOT land in the cold store — so we must NOT commit the
                        // slot Cold and free its lane (that would erase the only
                        // copy and corrupt a later rollback). Leave it Spilling: the
                        // hot lane stays pinned with the valid bytes, so a rollback
                        // to this boundary restores directly from the lane. The
                        // stranded lane is a bounded degradation, never corruption.
                        tracing::error!(
                            "decode-spill: store.put refused/failed for key {:#x} (seq {} slot {}) \
                             — rolling decode tier MUST be non-dropping; keeping slot resident \
                             (lane pinned) to preserve the rollback target",
                            req.cold_key,
                            req.seq,
                            req.logical
                        );
                        let _ = gpu.destroy_event(job.wait_event);
                        continue;
                    }
                    // Commit (or cancel on a superseding save/truncate) under the
                    // manager lock.
                    match mgr.lock().complete_spill(req.seq, req.logical, req.epoch) {
                        SpillCommit::Committed => {}
                        SpillCommit::Cancelled { remove_cold_key } => {
                            store.remove(remove_cold_key);
                        }
                    }
                    let _ = gpu.destroy_event(job.wait_event);
                }
            })?;
        self.decode_spiller = Some(DecodeSpiller {
            tx: Mutex::new(tx),
            handle: Some(handle),
        });
        Ok(())
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
        if self.decode_rolling.is_some() {
            return self.save_decode_managed(ssm_slot, ring_slot, main_pool, gpu, stream);
        }
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
        if self.decode_rolling.is_some() {
            return self.restore_decode_managed(ssm_slot, ring_slot, main_pool, gpu, stream);
        }
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

    // ── Rolling decode tier: managed save/restore + device gather/scatter ────

    /// Physical decode-frame index for a per-seq lane (rolling mode).
    #[inline]
    fn decode_lane_frame(&self, ssm_slot: usize, lane: usize) -> usize {
        ssm_slot * self.decode_l_phys + lane
    }

    /// D2D live pool slot → decode HBM lane frame (the fast per-boundary write).
    fn decode_write_lane(
        &self,
        ssm_slot: usize,
        lane: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let frame = self.decode_lane_frame(ssm_slot, lane);
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                main_pool.h_state(i, ssm_slot),
                self.decode_h_snapshots[i].offset(frame * self.h_bytes),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                main_pool.conv_state(i, ssm_slot),
                self.decode_conv_snapshots[i].offset(frame * self.conv_bytes),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// D2D decode HBM frame → live pool slot (restore from a resident/pinned
    /// lane, or from a just-faulted scratch frame).
    fn decode_restore_frame(
        &self,
        frame: usize,
        ssm_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.decode_h_snapshots[i].offset(frame * self.h_bytes),
                main_pool.h_state(i, ssm_slot),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.decode_conv_snapshots[i].offset(frame * self.conv_bytes),
                main_pool.conv_state(i, ssm_slot),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// The per-boundary save in rolling mode: plan against the residency manager,
    /// write the boundary D2D into an HBM lane (constraint 1), and async-spill the
    /// displaced LRU-hot lane off the decode path (constraint 2). Backpressure
    /// (all lanes busy) waits for the worker to free a lane rather than writing to
    /// a tier synchronously.
    fn save_decode_managed(
        &self,
        ssm_slot: usize,
        ring_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let mgr = self.decode_rolling.as_ref().expect("rolling");
        let decision = loop {
            let d = mgr.lock().plan_save(ssm_slot, ring_slot);
            match d {
                SaveDecision::Backpressure { drain } => {
                    if self.decode_spiller.is_some() {
                        // Async worker owns these in-flight spills — wait for it to
                        // complete one and free a lane, then re-plan. Never re-do
                        // them here (that would double-commit / double-remove).
                        std::thread::sleep(std::time::Duration::from_micros(50));
                    } else {
                        // No worker: safe to synchronously drain (nothing else
                        // owns these spills).
                        for req in drain {
                            self.decode_drain_spill_sync(req, gpu, stream)?;
                        }
                    }
                }
                other => break other,
            }
        };
        match decision {
            SaveDecision::InPlace { lane } => {
                self.decode_write_lane(ssm_slot, lane, main_pool, gpu, stream)?;
            }
            SaveDecision::Fresh { lane, spill } => {
                self.decode_write_lane(ssm_slot, lane, main_pool, gpu, stream)?;
                if let Some(req) = spill {
                    self.decode_enqueue_spill(req, gpu, stream)?;
                }
            }
            SaveDecision::Backpressure { .. } => unreachable!("drained above"),
        }
        Ok(())
    }

    /// The rare rollback restore in rolling mode: restore from the pinned HBM lane
    /// (Resident/Spilling) or fault the cold blob back into a scratch lane BEFORE
    /// the D2D restore reads it (constraint 3+4).
    fn restore_decode_managed(
        &self,
        ssm_slot: usize,
        ring_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let mgr = self.decode_rolling.as_ref().expect("rolling");
        let decision = mgr.lock().plan_restore(ssm_slot, ring_slot);
        match decision {
            RestoreDecision::FromLane { lane } => {
                let frame = self.decode_lane_frame(ssm_slot, lane);
                self.decode_restore_frame(frame, ssm_slot, main_pool, gpu, stream)
            }
            RestoreDecision::FaultThenRestore { scratch_lane, cold_key } => {
                let res =
                    self.decode_fault_scratch(ssm_slot, scratch_lane, cold_key, main_pool, gpu, stream);
                mgr.lock().release_scratch(scratch_lane);
                res
            }
            RestoreDecision::Decline => {
                bail!("SSM decode rolling: no live snapshot for ring_slot {ring_slot}")
            }
        }
    }

    /// Fault `cold_key` H2D into the shared scratch `scratch_lane`, synchronize so
    /// the scatter commits, then D2D scratch → live pool. A store MISS on a live
    /// rollback target is impossible with a non-dropping tier — treat it as a hard
    /// corruption bug, never a silent recompute.
    fn decode_fault_scratch(
        &self,
        ssm_slot: usize,
        scratch_lane: usize,
        cold_key: u64,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let store = self
            .decode_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("rolling decode tier: no cold store attached"))?;
        let mut blob = vec![0u8; self.spill_blob_bytes()];
        if !store.get(cold_key, &mut blob)? {
            bail!(
                "SSM decode rolling: cold MISS on live target key {cold_key:#x} — the decode \
                 tier MUST be non-dropping; this is corruption, not a recompute"
            );
        }
        let frame = self.decode_max_seqs * self.decode_l_phys + scratch_lane;
        let per_layer = self.h_bytes + self.conv_bytes;
        for i in 0..self.num_ssm_layers {
            let off = i * per_layer;
            gpu.copy_h2d_async(
                &blob[off..off + self.h_bytes],
                self.decode_h_snapshots[i].offset(frame * self.h_bytes),
                stream,
            )?;
            gpu.copy_h2d_async(
                &blob[off + self.h_bytes..off + per_layer],
                self.decode_conv_snapshots[i].offset(frame * self.conv_bytes),
                stream,
            )?;
        }
        gpu.synchronize(stream)?; // scatter committed before the D2D restore reads it
        self.decode_restore_frame(frame, ssm_slot, main_pool, gpu, stream)
    }

    /// Hand a displaced-lane spill to the async worker (records an ordering event
    /// on the save `stream`). Falls back to a synchronous drain if no worker is
    /// attached or the channel is closed.
    fn decode_enqueue_spill(&self, req: SpillReq, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        if let Some(sp) = &self.decode_spiller {
            let ev = gpu.create_event()?;
            gpu.record_event(ev, stream)?;
            match sp.tx.lock().send(SpillJob { req, wait_event: ev }) {
                Ok(()) => Ok(()),
                Err(_) => {
                    let _ = gpu.destroy_event(ev);
                    self.decode_drain_spill_sync(req, gpu, stream)
                }
            }
        } else {
            self.decode_drain_spill_sync(req, gpu, stream)
        }
    }

    /// Synchronous fallback spill (backpressure with no worker, or a closed
    /// channel): drain the save stream, gather the lane D2H, put, and commit the
    /// residency transition. Blocks the decode thread — only the uncommon path.
    fn decode_drain_spill_sync(
        &self,
        req: SpillReq,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let store = self
            .decode_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("rolling decode tier: no cold store attached"))?;
        gpu.synchronize(stream)?;
        let frame = req.seq * self.decode_l_phys + req.lane;
        let mut blob = vec![0u8; self.spill_blob_bytes()];
        let per_layer = self.h_bytes + self.conv_bytes;
        for i in 0..self.num_ssm_layers {
            let off = i * per_layer;
            gpu.copy_d2h(
                self.decode_h_snapshots[i].offset(frame * self.h_bytes),
                &mut blob[off..off + self.h_bytes],
            )?;
            gpu.copy_d2h(
                self.decode_conv_snapshots[i].offset(frame * self.conv_bytes),
                &mut blob[off + self.h_bytes..off + per_layer],
            )?;
        }
        if !store.put(req.cold_key, &blob)? {
            bail!("rolling decode tier refused a spill (must be non-dropping)");
        }
        if let SpillCommit::Cancelled { remove_cold_key } =
            self.decode_rolling.as_ref().expect("rolling").lock().complete_spill(
                req.seq,
                req.logical,
                req.epoch,
            )
        {
            store.remove(remove_cold_key);
        }
        Ok(())
    }

    /// Drop a logical decode slot (scheduler `truncate_after` / seq teardown): free
    /// its lane, cancel any in-flight spill (epoch bump), and remove its cold key.
    /// No-op when rolling is off.
    pub(super) fn drop_decode_slot(&self, ssm_slot: usize, ring_slot: usize) {
        let Some(mgr) = self.decode_rolling.as_ref() else {
            return;
        };
        if let Some(key) = mgr.lock().drop_slot(ssm_slot, ring_slot) {
            if let Some(store) = &self.decode_store {
                store.remove(key);
            }
        }
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
    ///
    /// Clears the slot's `session_tags` entry: a freed slot has no owner, and a
    /// spilled-then-reacquired slot must NOT carry the victim's stale session
    /// tag. Without this, `fault_in_slot` scatters the correct bytes into the
    /// reused slot but `session_matches` then compares the *previous* occupant's
    /// tag against the faulting session, rejects, and the restore silently
    /// degrades to a full recompute (measured: 0 completed tier restores at
    /// `--ssm-cache-slots 4` despite fault-ins firing).
    pub(super) fn free(&self, snap_slot: usize) {
        self.slot_has_hidden.lock().remove(&snap_slot);
        self.session_tags.lock().remove(&snap_slot);
        self.free_slots.lock().push(snap_slot);
    }

    /// Tag a slot with the session that now owns its state — used after a
    /// `fault_in_slot` re-homes a spilled snapshot into a fresh slot, so the
    /// subsequent `session_matches` gate (and any later lookup landing on this
    /// slot) sees the correct owner rather than an untagged/stale slot.
    pub(super) fn tag_session(&self, snap_slot: usize, session_hash: u64) {
        if session_hash != 0 {
            self.session_tags.lock().insert(snap_slot, session_hash);
        }
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
        let timing = std::env::var_os("ATLAS_SSM_TIER_TIMING").is_some();
        let t0 = std::time::Instant::now();
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
        let t_put = std::time::Instant::now();
        let r = store.put(key, &blob)?;
        if timing {
            tracing::info!(
                "SSM spill: {} B  gather+sync={}us  store.put={}us  total={}us",
                blob.len(),
                t_put.duration_since(t0).as_micros(),
                t_put.elapsed().as_micros(),
                t0.elapsed().as_micros(),
            );
        }
        Ok(r)
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
        let timing = std::env::var_os("ATLAS_SSM_TIER_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let mut blob = vec![0u8; self.spill_blob_bytes()];
        let hit = store.get(key, &mut blob)?;
        let get_us = t0.elapsed().as_micros();
        if !hit {
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
        if timing {
            tracing::info!(
                "SSM fault-in: {} B  store.get(RDMA read)={}us  scatter+sync={}us  total={}us",
                blob.len(),
                get_us,
                t0.elapsed().as_micros() - get_us,
                t0.elapsed().as_micros(),
            );
        }
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

    /// Regression: `free` must clear the slot's session tag, and a fault-in must
    /// re-tag it — otherwise a spilled-then-reacquired slot carries the victim's
    /// stale owner and `session_matches` wrongly rejects the just-faulted state,
    /// silently degrading every tier restore to a full recompute.
    #[test]
    fn free_clears_session_tag_then_faultin_retags() {
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, 4, 2);

        // Slot 1 owned by session A.
        p.tag_session(1, 0xAAAA);
        assert!(p.session_matches(1, 0xAAAA));
        assert!(!p.session_matches(1, 0xBBBB), "stale tag rejects a different session");

        // Spill frees the slot → owner must be cleared (untagged ⇒ allowed).
        p.free(1);
        assert!(
            p.session_matches(1, 0xBBBB),
            "freed slot must be untagged so a fault-in for any session is usable"
        );

        // Fault-in re-homes session B's state into this slot → now owned by B.
        p.tag_session(1, 0xBBBB);
        assert!(p.session_matches(1, 0xBBBB));
        assert!(!p.session_matches(1, 0xAAAA), "re-homed slot rejects the previous owner");
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
