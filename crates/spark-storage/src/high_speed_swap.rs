// SPDX-License-Identifier: AGPL-3.0-only
//
// `HighSpeedSwap` orchestrator: combines Predictor + ScratchPool +
// IoUringBackend + TiledAttention + EvictionPolicy behind a two-method API
// (`offload_block`, `attend_layer`). Designed to be the primitive a future
// scheduler integration in `spark-model` plugs into.

use anyhow::{Context, Result};

use crate::backend::IoUringBackend;
use crate::config::HighSpeedSwapConfig;
use crate::cuda_min::{CudaCtx, DeviceBuffer};
use crate::eviction::EvictionPolicy;
use crate::group::GroupLayout;
use crate::layout::Layout;
use crate::predictor::{Predictor, PredictorDims};
use crate::scratch_pool::{ScratchDims, ScratchPool};
use crate::tiled_attention::{TiledAttention, TiledAttentionDims};

// `ModelDims` lives in `crate::model_dims` so it stays available on
// non-cuda builds where the swap orchestrator below isn't compiled.
pub use crate::model_dims::ModelDims;

pub struct HighSpeedSwap {
    cfg: HighSpeedSwapConfig,
    model: ModelDims,
    predictor: Predictor,
    /// KV overflow tier: local NVMe (`IoUringBackend`) by default, or a remote
    /// RDMA RAM blade (`RdmaKvBackend`) when `$ATLAS_KV_PEER` is set — the same
    /// `StorageBackend` trait, so the offload/restore call sites are unchanged.
    ///
    /// Declared BEFORE `pool` so Rust's declaration-order drop tears the backend
    /// down first: with a UMA (pinned) `pool`, the RDMA backend caches
    /// destination MRs (`Rail::dst_lkeys`) registered INTO the pool's pages, and
    /// those MRs must be deregistered before the pool's `cuMemFreeHost` runs —
    /// otherwise it's a dereg-on-freed-pages use-after-free.
    backend: Box<dyn crate::backend::StorageBackend>,
    pool: ScratchPool,
    attn: TiledAttention,
    eviction: EvictionPolicy,
    // Reusable scratch buffers.
    q_proj: DeviceBuffer,
    block_scores_dev: DeviceBuffer, // [max_blocks] f32
    block_table_dev: DeviceBuffer,  // [tile_capacity] i32
    counts_dev: DeviceBuffer,       // [1] i32 (single seq)
    score_host_buf: Vec<f32>,
    // Tile pipeline (see impl_more::attend_layer_on_stream_with_q_pos): a
    // sibling copy stream carries tile[t+1]'s miss reads while tile[t]'s
    // attention runs on the compute stream. `ev_read[plane]` is the RAW gate
    // (compute waits for the plane's reads to land); `ev_war[plane]` is the
    // plane-reuse gate (the copy stream waits for the prior consumer of that
    // plane to finish before overwriting it). `war_armed` skips the very first
    // wait on each plane, before any consumer has recorded ev_war.
    copy_stream: u64,
    ev_read: [crate::cuda_min::CudaEvent; 2],
    ev_war: [crate::cuda_min::CudaEvent; 2],
    war_armed: [bool; 2],
    // Disk-block-ID allocator (Phase 6.1.a, refactored). One global
    // allocator: a `disk_block_id` indexes the SAME logical position
    // across every layer's file, so allocation, refcount, and free list
    // are layer-agnostic. Each layer's file independently stores its
    // K/V at `offset(layer, disk_block_id)`.
    disk_state: DiskState,
}

#[derive(Debug)]
struct DiskState {
    next_id: u32,
    free_list: Vec<u32>,
    refcount: Vec<u32>,
}

impl DiskState {
    fn new() -> Self {
        Self {
            next_id: 0,
            free_list: Vec::new(),
            refcount: Vec::new(),
        }
    }
}

impl HighSpeedSwap {
    pub fn new(ctx: &CudaCtx, cfg: HighSpeedSwapConfig, model: ModelDims) -> Result<Self> {
        Self::new_on_stream(ctx.stream, cfg, model)
    }

    /// Stream-only constructor for production callers that already own a
    /// CUDA context (spark-model). The provided `stream` is used only for
    /// init-time copies (uploading the projection matrix P); subsequent
    /// per-step calls take their own stream argument.
    pub fn new_on_stream(stream: u64, cfg: HighSpeedSwapConfig, model: ModelDims) -> Result<Self> {
        cfg.validate_and_prepare()?;
        let group_layout = GroupLayout::new(
            model.num_layers,
            model.max_blocks_per_layer,
            model.num_kv_heads,
            model.block_size as u32,
            model.head_dim as u32,
            2, // BF16
            4096,
        );
        // KV overflow tier selection: RDMA RAM blade when $ATLAS_KV_PEER is set,
        // else the local NVMe io_uring backend (default, unchanged).
        let kv_peer = std::env::var("ATLAS_KV_PEER").ok();
        let backing: Box<dyn crate::backend::StorageBackend> = {
            #[cfg(atlas_rdma_verbs)]
            {
                if let Some(peer) = kv_peer {
                    tracing::info!(
                        "high-speed-swap: KV overflow tier = RDMA peer {peer} (group_stride {})",
                        group_layout.group_stride
                    );
                    Box::new(crate::rdma_kv_backend::RdmaKvBackend::connect(
                        &peer,
                        group_layout,
                    )?)
                } else {
                    let layout = Layout::create(&cfg.dir, group_layout).context("create layout")?;
                    Box::new(IoUringBackend::new(layout, cfg.qd as usize)?)
                }
            }
            #[cfg(not(atlas_rdma_verbs))]
            {
                let _ = &kv_peer;
                let layout = Layout::create(&cfg.dir, group_layout).context("create layout")?;
                Box::new(IoUringBackend::new(layout, cfg.qd as usize)?)
            }
        };
        // Restore-destination pool. When zero-copy restore is requested
        // (ATLAS_KV_ZERO_COPY=1, the same flag the RDMA backend keys off), prefer
        // a UMA (pinned LPDDR) pool so the NIC can RDMA straight into the slots
        // with no bounce/copy_h2d. Default (flag unset) stays device-backed,
        // byte-for-byte the prior behavior. `new_preferring_uma` self-falls-back
        // to device memory on a non-UMA host, so this is always safe.
        let want_uma = std::env::var("ATLAS_KV_ZERO_COPY").ok().as_deref() == Some("1");
        let dims = ScratchDims {
            num_slots: cfg.resident_blocks,
            // Two extra streaming planes (S_A, S_B) of one tile each for the
            // tile pipeline's ping-pong double-buffer. The resident cache stays
            // exactly `resident_blocks` slots; these planes are never entered
            // into the cache maps (see ScratchDims::stream_slots).
            stream_slots: 2 * cfg.resident_blocks,
            num_kv_heads: model.num_kv_heads,
            group_stride: group_layout.group_stride,
        };
        let pool = if want_uma {
            ScratchPool::new_preferring_uma(dims)?
        } else {
            ScratchPool::new(dims)?
        };
        if want_uma {
            tracing::info!(
                "high-speed-swap: scratch pool UMA={} (zero-copy restore {})",
                pool.is_uma(),
                if pool.is_uma() {
                    "enabled"
                } else {
                    "unavailable — using bounce"
                },
            );
        }
        // T1 cascade: wrap `backing` in a local pinned-RAM write-back cache when
        // $ATLAS_KV_LOCAL_GB > 0 (hot groups stay local, evictions flush down to
        // the peer/SSD). 0 (default) is the passthrough — byte-identical to today.
        let local_gb: f64 = std::env::var("ATLAS_KV_LOCAL_GB")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|g: &f64| g.is_finite() && *g >= 0.0)
            .unwrap_or(0.0);
        let mut backend: Box<dyn crate::backend::StorageBackend> = if local_gb > 0.0 {
            let cap_slots =
                ((local_gb * 1024.0 * 1024.0 * 1024.0) / group_layout.group_bytes() as f64) as u32;
            Box::new(crate::cascade_backend::CascadeBackend::new(
                backing,
                group_layout,
                cap_slots.max(1),
            )?)
        } else {
            backing
        };
        // Register the whole UMA pool as one landing MR so zero-copy restore
        // reuses that lkey per slot (per-slot registration fails on GB10). When a
        // cascade wraps the backing, this forwards to the backing so RDMA restore
        // of MISSES still lands zero-copy. No-op for the file backends; on failure
        // the backend degrades to the bounce path at read() time (best-effort).
        if want_uma
            && pool.is_uma()
            && let Err(e) =
                backend.register_landing_region(pool.pool_dev_ptr(), pool.dims().pool_bytes())
        {
            tracing::warn!(
                "high-speed-swap: UMA landing-region registration failed ({e:#}); \
                 restore will use the bounce path"
            );
        }
        let predictor = Predictor::new_on_stream(
            stream,
            PredictorDims {
                num_layers: model.num_layers as usize,
                num_q_heads: model.num_q_heads as usize,
                num_kv_heads: model.num_kv_heads as usize,
                head_dim: model.head_dim as usize,
                r: cfg.rank as usize,
                block_size: model.block_size as usize,
                max_blocks: model.max_blocks_per_layer as usize,
            },
            cfg.projection_seed,
        )?;
        let attn = TiledAttention::new(TiledAttentionDims {
            max_seqs: 1, // single-seq for the orchestrator's first iteration
            num_q_heads: model.num_q_heads as usize,
            num_kv_heads: model.num_kv_heads as usize,
            head_dim: model.head_dim as usize,
            block_size: model.block_size as usize,
            tile_capacity: cfg.resident_blocks as usize,
        })?;
        let eviction = EvictionPolicy::new(cfg.resident_blocks);
        let q_proj = DeviceBuffer::new(model.num_q_heads as usize * cfg.rank as usize * 2)?;
        let block_scores_dev = DeviceBuffer::new(model.max_blocks_per_layer as usize * 4)?;
        let block_table_dev = DeviceBuffer::new(cfg.resident_blocks as usize * 4)?;
        let counts_dev = DeviceBuffer::new(4)?;
        let score_host_buf = vec![0.0_f32; model.max_blocks_per_layer as usize];
        let disk_state = DiskState::new();
        let copy_stream = crate::cuda_min::create_stream()?;
        let ev_read = [
            crate::cuda_min::CudaEvent::new()?,
            crate::cuda_min::CudaEvent::new()?,
        ];
        let ev_war = [
            crate::cuda_min::CudaEvent::new()?,
            crate::cuda_min::CudaEvent::new()?,
        ];
        Ok(Self {
            cfg,
            model,
            predictor,
            backend,
            pool,
            attn,
            eviction,
            q_proj,
            block_scores_dev,
            block_table_dev,
            counts_dev,
            score_host_buf,
            disk_state,
            copy_stream,
            ev_read,
            ev_war,
            war_armed: [false, false],
        })
    }

    // ── Disk-block-ID allocator (Phase 6.1.a) ─────────────────────────
    // Each layer has an independent ID space. Capacity == max_blocks_per_layer.
    // alloc / free list / refcount semantics:
    //   - alloc_disk_block_id(layer) -> Some(id) if room, else None
    //   - inc_disk_ref(layer, id) increments (panics if id is unallocated)
    //   - dec_disk_ref(layer, id) -> new refcount; on 0 returns id to free list

    pub fn alloc_disk_block_id(&mut self) -> Option<u32> {
        let st = &mut self.disk_state;
        if let Some(id) = st.free_list.pop() {
            st.refcount[id as usize] = 1;
            return Some(id);
        }
        if st.next_id >= self.model.max_blocks_per_layer {
            return None; // capacity exhausted
        }
        let id = st.next_id;
        st.next_id += 1;
        st.refcount.push(1);
        Some(id)
    }

    pub fn inc_disk_ref(&mut self, id: u32) {
        let rc = &mut self.disk_state.refcount[id as usize];
        if *rc == 0 {
            panic!("inc_disk_ref on freed disk_block_id {id}; caller must hold a live ref");
        }
        *rc += 1;
    }

    pub fn dec_disk_ref(&mut self, id: u32) -> u32 {
        let st = &mut self.disk_state;
        let rc = &mut st.refcount[id as usize];
        debug_assert!(*rc > 0, "dec_disk_ref on already-freed id {id}");
        *rc = rc.saturating_sub(1);
        let new_rc = *rc;
        if new_rc == 0 {
            st.free_list.push(id);
        }
        new_rc
    }

    pub fn disk_refcount(&self, id: u32) -> u32 {
        self.disk_state.refcount[id as usize]
    }

    pub fn disk_free_count(&self) -> usize {
        let st = &self.disk_state;
        st.free_list.len() + (self.model.max_blocks_per_layer - st.next_id) as usize
    }

    /// Aggregated diagnostic summary across all layers (Phase 6.1.j).
    /// Use to log periodic state during long-running decode loops; the
    /// scheduler can call this once per N steps to verify HBM-shrink
    /// behavior is on track.
    pub fn diagnostic_summary(&self) -> HighSpeedSwapDiagnostic {
        let st = &self.disk_state;
        let active = st.next_id.saturating_sub(st.free_list.len() as u32);
        HighSpeedSwapDiagnostic {
            num_layers: self.model.num_layers,
            active_disk_blocks: active,
            disk_block_capacity: self.model.max_blocks_per_layer,
            scratch_pool_resident: self.pool.dims().num_slots,
            scratch_pool_free: self.pool.free_count(),
        }
    }
}

impl Drop for HighSpeedSwap {
    fn drop(&mut self) {
        // Tear down the sibling copy stream. Declaration order already drops
        // `backend` before `pool`; the copy stream is independent of both.
        let _ = crate::cuda_min::destroy_stream(self.copy_stream);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HighSpeedSwapDiagnostic {
    pub num_layers: u32,
    pub active_disk_blocks: u32,
    pub disk_block_capacity: u32,
    pub scratch_pool_resident: u32,
    pub scratch_pool_free: u32,
}

#[cfg(test)]
mod disk_id_tests;

mod impl_more;

// ── Thread-local installation for production callers (spark-model) ──
//
// The scheduler thread, after `bind_gpu_to_thread`, calls `install_local`
// to register the orchestrator. Per-layer attention code in spark-model
// then accesses it via `with_local`. The orchestrator's HBM allocations
// live as long as the thread; cleanup happens on thread exit (or
// explicit drop via `take_local`).

use std::cell::RefCell;
thread_local! {
    static LOCAL: RefCell<Option<HighSpeedSwap>> = const { RefCell::new(None) };
}

/// Install the orchestrator on the current thread. Idempotent (overwrites
/// any prior installation, dropping it).
pub fn install_local(stream: u64, cfg: HighSpeedSwapConfig, model: ModelDims) -> Result<()> {
    let hss = HighSpeedSwap::new_on_stream(stream, cfg, model)?;
    LOCAL.with(|cell| {
        *cell.borrow_mut() = Some(hss);
    });
    Ok(())
}

/// True iff `install_local` has populated this thread's slot.
pub fn local_installed() -> bool {
    LOCAL.with(|cell| cell.borrow().is_some())
}

/// Run `f` with a `&mut HighSpeedSwap` if installed; returns `None` if not.
pub fn with_local<R>(f: impl FnOnce(&mut HighSpeedSwap) -> Result<R>) -> Option<Result<R>> {
    LOCAL.with(|cell| cell.borrow_mut().as_mut().map(f))
}
