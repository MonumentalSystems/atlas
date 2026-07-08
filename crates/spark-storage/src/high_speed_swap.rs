// SPDX-License-Identifier: AGPL-3.0-only
//
// `HighSpeedSwap` orchestrator: combines Predictor + ScratchPool +
// IoUringBackend + TiledAttention + EvictionPolicy behind a two-method API
// (`offload_block`, `attend_layer`). Designed to be the primitive a future
// scheduler integration in `spark-model` plugs into.

use anyhow::{Context, Result};

use crate::backend::IoUringBackend;
use crate::config::HighSpeedSwapConfig;
use crate::cuda_min::{CudaCtx, CudaEvent, DeviceBuffer};
use crate::eviction::EvictionPolicy;
use crate::group::GroupLayout;
use crate::layout::Layout;
use crate::predictor::{Predictor, PredictorDims};
use crate::scratch_pool::{ScratchDims, ScratchPool};
use crate::tiled_attention::{TiledAttention, TiledAttentionDims, TiledAttnPlanes};

// `ModelDims` lives in `crate::model_dims` so it stays available on
// non-cuda builds where the swap orchestrator below isn't compiled.
pub use crate::model_dims::ModelDims;

/// Which local-NVMe KV-overflow backend to build. io_uring is the fast default,
/// but it is not universally available: the default container seccomp profile
/// blocks the `io_uring_*` syscalls (post-2023 hardening) and pre-5.1 kernels
/// lack it entirely. `$ATLAS_KV_BACKEND` overrides the choice.
#[derive(Debug, PartialEq, Eq)]
enum KvBackendChoice {
    /// Default: try io_uring, transparently fall back to POSIX on failure.
    UringThenPosix,
    /// `ATLAS_KV_BACKEND=posix`: skip io_uring, use portable pread/pwrite.
    ForcePosix,
    /// `ATLAS_KV_BACKEND=io_uring`: require io_uring; fail loud if unavailable.
    RequireUring,
}

/// Pure `$ATLAS_KV_BACKEND` policy (unit-tested without CUDA/NVMe). Unknown
/// values fall through to the default (try-then-fall-back), so a typo never
/// silently forces the slow path.
fn kv_backend_choice(forced: Option<&str>) -> KvBackendChoice {
    match forced {
        Some("posix") => KvBackendChoice::ForcePosix,
        Some("io_uring") | Some("iouring") => KvBackendChoice::RequireUring,
        _ => KvBackendChoice::UringThenPosix,
    }
}

/// Build the local-NVMe KV-overflow backend (the default when no `$ATLAS_KV_PEER`
/// is set). io_uring is the fast path; on a host where it can't init — a
/// restrictive container seccomp profile (io_uring_* blocked → `EPERM`) or a
/// pre-5.1 kernel — degrade to the portable POSIX `pread`/`pwrite` backend with a
/// warning, so `--high-speed-swap` still works everywhere instead of failing
/// outright. RDMA and io_uring are both opt-in fast paths over this floor.
fn local_nvme_backend(
    dir: &std::path::Path,
    group_layout: GroupLayout,
    qd: usize,
) -> Result<Box<dyn crate::backend::StorageBackend>> {
    let choice = kv_backend_choice(std::env::var("ATLAS_KV_BACKEND").ok().as_deref());
    if choice != KvBackendChoice::ForcePosix {
        // `IoUringBackend::new` consumes `layout`; on failure it is dropped
        // (fds closed) and the POSIX path below recreates it cleanly.
        let layout = Layout::create(dir, group_layout).context("create layout")?;
        match IoUringBackend::new(layout, qd) {
            Ok(b) => {
                tracing::info!("high-speed-swap: KV overflow tier = local NVMe (io_uring, qd {qd})");
                return Ok(Box::new(b));
            }
            Err(e) if choice == KvBackendChoice::RequireUring => {
                return Err(e).context("ATLAS_KV_BACKEND=io_uring set but io_uring init failed");
            }
            Err(e) => {
                tracing::warn!(
                    "high-speed-swap: io_uring backend unavailable ({e:#}); falling back to \
                     the POSIX pread/pwrite backend. Expected under a restrictive container \
                     seccomp profile (io_uring_* syscalls blocked) or a pre-5.1 kernel. Run \
                     with --security-opt seccomp=unconfined (or set ATLAS_KV_BACKEND=io_uring \
                     to require io_uring and fail loud instead)."
                );
            }
        }
    }
    // ForcePosix, or io_uring init failed and fallback is allowed.
    let layout = Layout::create(dir, group_layout).context("create layout (posix)")?;
    tracing::info!("high-speed-swap: KV overflow tier = local NVMe (POSIX pread/pwrite)");
    Ok(Box::new(crate::backend::PosixBackend::new(layout)?))
}

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
    /// Shared, immutable tiled-attention kernel handles. The per-op accumulator
    /// planes live per-seq in `scratch` (Phase 2).
    attn: TiledAttention,
    eviction: EvictionPolicy,
    /// Phase 2: per-sequence transient orchestrator scratch — accumulator planes
    /// + the single-use projection/score/tile buffers. Indexed by `seq_slot`
    /// (the batch position) so two concurrent sequences never clobber each
    /// other's partial softmax / scratch (the C=8 race that killed the
    /// tile-pipeline). Lazily grown on first use of each slot; a serial pass
    /// touches slot `i` under a barrier, a future pipeline can overlap slots.
    /// The shared `pool`/`backend`/`predictor`/`disk_state` stay single-owner.
    scratch: Vec<SeqScratch>,
    /// Phase 5: C = max concurrent overflow-decode sequences one batched
    /// attend pass may carry (`$ATLAS_HSS_MAX_SEQS`, default 1). Sizes the
    /// scratch pool (`C × resident_blocks` slots), the shared
    /// `TiledAttentionDims::max_seqs`, and `batch` below. C=1 keeps every
    /// path byte-identical to the single-seq orchestrator.
    max_seqs: usize,
    /// Phase 5: shared batched-attend scratch, `None` when `max_seqs == 1`
    /// (gating the extra allocations out keeps the C=1 constructor
    /// byte-identical). Allocated in Inc 1 so the sizing lands with the
    /// plumbing; the num_seqs=C attend pass (Inc 3) consumes it.
    #[allow(dead_code)] // read by the batched attend (Phase 5 Inc 3)
    batch: Option<BatchScratch>,
    /// Phase 3: a side CUDA stream for prefetch H2D copies. Running the prefetch
    /// read on this stream (rather than the main compute stream) lets its H2D
    /// copies overlap the main stream's SSM/MoE kernels — the copy and compute
    /// engines run concurrently on GB10. The CPU still blocks in the io_uring
    /// drain, but that block overlaps the already-enqueued main-stream compute,
    /// which is where the read is hidden.
    prefetch_stream: u64,
    /// #11: orders the side-stream prefetch overwrite AFTER the main stream's
    /// KV-slot reads (the batched `step_tile`s) complete. Recorded on the main
    /// stream by the decode loop at each prefetch boundary
    /// (`record_kv_read_event`), before the prefetch fan-out; waited on
    /// `prefetch_stream` (`prefetch_layer_on_stream`) immediately before the
    /// overwriting evict-victim H2D. In-order main-stream execution ⇒ this
    /// event dominates every `step_tile` KV read enqueued so far this step
    /// (all prior attention layers), so the side-stream overwrite cannot land
    /// on a slot a pending `step_tile` still reads — the cross-stream WAR is
    /// closed without a full sync, letting batched sync-collapse and prefetch
    /// run together. A single event suffices: record and its waits issue from
    /// the same compute thread in fixed host order per boundary, and
    /// `cuStreamWaitEvent` captures the recorded state at enqueue time, so a
    /// later re-record cannot perturb an already-enqueued wait. RAII `Drop`
    /// destroys it. FOLLOW-UP (out of scope for #11): the mirror RAW —
    /// attend(L+1) reading prefetched slots before the H2D lands — stays
    /// closed today by the trailing host `stream_sync` inside
    /// `IoUringBackend::read`; when that host sync is dropped for genuinely
    /// async prefetch, a symmetric prefetch-completion event (record on
    /// `prefetch_stream` after the read, wait on the main stream before the
    /// next attend) becomes mandatory.
    kv_war_event: CudaEvent,
    /// #11-refinement: MIRROR-RAW twin of `kv_war_event`, exactly the follow-up
    /// pre-specified in the kv_war_event doc above. Recorded on `prefetch_stream`
    /// AFTER the prefetch H2D (in prefetch_layer_on_stream), waited on the MAIN
    /// stream at attend(L+1)'s tile-phase entry. Closes attend-reads-before-H2D-
    /// lands device-side so read_async can drop its terminal host stream_sync.
    /// One event suffices: at most one prefetch outstanding (issued at boundary L,
    /// consumed+unpinned by attend L+1 before boundary L+1); the C-seq fan-out is
    /// sequential on the single in-order prefetch_stream so the last record
    /// dominates every seq's H2D; cuStreamWaitEvent snapshots at enqueue so a
    /// later re-record can't perturb an enqueued wait — same argument as kv_war_event.
    kv_prefetch_done: CudaEvent,
    /// #11: mirrors the scheduler's `ATLAS_KV_PREFETCH` switch (read once at
    /// construction; decode_a2.rs gates the actual prefetch on the same var).
    /// Now used ONLY to gate the empty-read skip in the batched-attend union
    /// reads: `IoUringBackend::read` issues an unconditional trailing
    /// `stream_sync` even for zero requests, so on a fully-prefetched tile the
    /// union read collapses to a bare main-stream drain — an accidental
    /// WAR-narrowing barrier. That no-op read is skipped ONLY when prefetch is
    /// live (the run where the `kv_war_event` fence replaces it); prefetch-off
    /// keeps the unconditional read+sync for byte-for-byte op-identity.
    kv_prefetch_enabled: bool,
    // Disk-block-ID allocator (Phase 6.1.a, refactored). One global
    // allocator: a `disk_block_id` indexes the SAME logical position
    // across every layer's file, so allocation, refcount, and free list
    // are layer-agnostic. Each layer's file independently stores its
    // K/V at `offset(layer, disk_block_id)`.
    disk_state: DiskState,
}

/// Per-sequence transient orchestrator scratch (Phase 2). Everything here is
/// overwritten within a single `attend_layer` call, so one set per concurrent
/// sequence is required to make overlapping sequences race-free.
struct SeqScratch {
    /// Online-softmax accumulator planes carried across the tile loop.
    planes: TiledAttnPlanes,
    /// Projected-Q scratch `[num_q_heads × rank]`.
    q_proj: DeviceBuffer,
    /// Per-block scores `[max_blocks]` f32.
    block_scores_dev: DeviceBuffer,
    /// Per-tile block table `[tile_capacity]` i32.
    block_table_dev: DeviceBuffer,
    /// Per-tile seq counts `[1]` i32.
    counts_dev: DeviceBuffer,
    /// Host staging for the D2H score copy.
    score_host_buf: Vec<f32>,
}

/// Phase 5: one sequence's inputs for the batched overflow-decode attend
/// (`attend_layer_batch_on_stream`). The overflowed seqs are a possibly
/// SPARSE subset of the batch positions, so each entry carries its own
/// `seq_slot` and its own Q/output row pointers — the batched path must
/// never assume dense rows 0..C of the caller's buffers.
pub struct AttendSeqReq<'a> {
    /// Batch position — selects this seq's `SeqScratch` (Phase 2 semantics,
    /// identical to `attend_layer_on_stream`'s `seq_slot`).
    pub seq_slot: usize,
    /// This seq's full ordered disk-side block history.
    pub seq_block_ids: &'a [u32],
    /// Device ptr to this seq's `[num_q_heads × head_dim]` BF16 query row.
    pub q_dev: u64,
    /// Device ptr to this seq's `[num_q_heads × head_dim]` BF16 output row.
    pub output_dev: u64,
}

/// Phase 5: batched-attend scratch shared across the C sequences of one
/// `num_seqs = C` attend pass. The kernel reads per-seq state out of ONE
/// contiguous buffer each (`tile_blocks[seq × tile_capacity + b]`,
/// `counts[seq]`, m/l/o at `[seq × num_q_heads + qh]`), so C separate
/// per-`SeqScratch` buffers cannot serve a batched launch.
struct BatchScratch {
    /// C-sized online-softmax accumulator planes.
    planes: TiledAttnPlanes,
    /// Per-tile block table `[max_seqs × tile_capacity]` i32.
    block_table_dev: DeviceBuffer,
    /// Per-tile per-seq counts `[max_seqs]` i32.
    counts_dev: DeviceBuffer,
    /// Phase 5 Inc 3: contiguous `[max_seqs × num_q_heads × head_dim]` BF16
    /// Q-gather buffer. The kernel reads `Q[(seq×nq+qh)×hd]` with seq=0..C-1,
    /// but the overflowed seqs sit at their (possibly sparse) original batch
    /// positions — so each seq's Q is d2d-copied into row c here before the
    /// wide launch, keeping the kernel untouched.
    q_gather_dev: DeviceBuffer,
    /// Phase 5 Inc 3: contiguous `[max_seqs × num_q_heads × head_dim]` BF16
    /// output buffer. `finalize(num_seqs=C)` writes all C rows here; each row
    /// is then d2d-scattered back to its seq's `output_dev`.
    o_gather_dev: DeviceBuffer,
}

impl SeqScratch {
    fn new(
        attn: &TiledAttention,
        model: &ModelDims,
        cfg: &HighSpeedSwapConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            // Per-seq planes stay sized for ONE sequence even when the shared
            // `TiledAttention` dims carry `max_seqs = C` (Phase 5): each
            // SeqScratch is only ever bound to `num_seqs = 1` launches (the
            // kept single-seq/prefill path). The batched pass uses the
            // C-sized `BatchScratch::planes` instead.
            planes: TiledAttnPlanes::new(&TiledAttentionDims {
                max_seqs: 1,
                ..attn.dims()
            })?,
            q_proj: DeviceBuffer::new(model.num_q_heads as usize * cfg.rank as usize * 2)?,
            block_scores_dev: DeviceBuffer::new(model.max_blocks_per_layer as usize * 4)?,
            block_table_dev: DeviceBuffer::new(cfg.resident_blocks as usize * 4)?,
            counts_dev: DeviceBuffer::new(4)?,
            score_host_buf: vec![0.0_f32; model.max_blocks_per_layer as usize],
        })
    }
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
        // Phase 5: C = max concurrent overflow-decode seqs the orchestrator
        // sizes its batched buffers for. Env-driven like the other ATLAS_KV_*
        // switches; unset/1 = today's single-seq sizing, byte-identical.
        let max_seqs: usize = std::env::var("ATLAS_HSS_MAX_SEQS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|c: &usize| (1..=1024).contains(c))
            .unwrap_or(1);
        if max_seqs > 1 {
            tracing::info!("high-speed-swap: batched attend sized for max_seqs={max_seqs}");
        }
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
                    local_nvme_backend(&cfg.dir, group_layout, cfg.qd as usize)?
                }
            }
            #[cfg(not(atlas_rdma_verbs))]
            {
                let _ = &kv_peer;
                local_nvme_backend(&cfg.dir, group_layout, cfg.qd as usize)?
            }
        };
        // Restore-destination pool. When zero-copy restore is requested
        // (ATLAS_KV_ZERO_COPY=1, the same flag the RDMA backend keys off), prefer
        // a UMA (pinned LPDDR) pool so the NIC can RDMA straight into the slots
        // with no bounce/copy_h2d. Default (flag unset) stays device-backed,
        // byte-for-byte the prior behavior. `new_preferring_uma` self-falls-back
        // to device memory on a non-UMA host, so this is always safe.
        let want_uma = std::env::var("ATLAS_KV_ZERO_COPY").ok().as_deref() == Some("1");
        // Phase 5 Inc 1: decouple pool capacity from the per-seq resident
        // budget — C concurrent overflow seqs each keep a full tile budget
        // resident, so num_slots = C × resident_blocks. The per-seq tile
        // budget (`tile_capacity` below, and the tile loop's `tile_cap`)
        // stays `cfg.resident_blocks` — the C=1 tile geometry is untouched.
        let per_seq_budget = cfg.resident_blocks;
        let num_slots = per_seq_budget
            .checked_mul(max_seqs as u32)
            .context("ATLAS_HSS_MAX_SEQS × resident_blocks overflows u32")?;
        // The plan's invariant, spelled out: every concurrent seq must fit
        // its full per-seq tile budget in the pool. Trivially equal today
        // (num_slots is defined as the product); guards any future knob that
        // sizes num_slots independently.
        debug_assert!(max_seqs as u32 * per_seq_budget <= num_slots);
        let dims = ScratchDims {
            num_slots,
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
            max_seqs, // Phase 5: C-wide batched attend (1 = single-seq, as before)
            num_q_heads: model.num_q_heads as usize,
            num_kv_heads: model.num_kv_heads as usize,
            head_dim: model.head_dim as usize,
            block_size: model.block_size as usize,
            // DO NOT retarget to the grown `num_slots`: this is the PER-SEQ
            // tile budget baked into the kernel's tile_blocks[seq*tcap+b]
            // stride and the C=1 tile geometry.
            tile_capacity: cfg.resident_blocks as usize,
        })?;
        // Eviction ranks POOL slots, so its capacity follows num_slots (the
        // same resident_blocks==pool-capacity coupling decoupled above);
        // leaving it at the per-seq budget would exhaust the grown pool early.
        let eviction = EvictionPolicy::new(num_slots);
        // Phase 5 Inc 1: shared batched-attend scratch, sized for one
        // grid=(C, num_q_heads, 1) pass. Gated on C>1 so the C=1 constructor
        // performs today's exact allocation sequence.
        let batch = if max_seqs > 1 {
            Some(BatchScratch {
                planes: attn.new_planes()?, // dims.max_seqs == C ⇒ C-sized planes
                block_table_dev: DeviceBuffer::new(
                    max_seqs * attn.dims().tile_capacity * 4,
                )?,
                counts_dev: DeviceBuffer::new(max_seqs * 4)?,
                // [C × nq × hd] BF16 each (gather Q in / scatter O out).
                q_gather_dev: DeviceBuffer::new(
                    max_seqs * model.num_q_heads as usize * model.head_dim as usize * 2,
                )?,
                o_gather_dev: DeviceBuffer::new(
                    max_seqs * model.num_q_heads as usize * model.head_dim as usize * 2,
                )?,
            })
        } else {
            None
        };
        // One scratch slot to start; `attend_layer` grows the Vec on first use
        // of each higher batch position (lazy — avoids plumbing max_batch_size).
        let scratch = vec![SeqScratch::new(&attn, &model, &cfg)?];
        let prefetch_stream = crate::cuda_min::create_stream()?;
        // #11: WAR fence event, created unconditionally (like `prefetch_stream`).
        // Inert when prefetch is off (never recorded/waited), so it cannot
        // perturb bytes; unconditional creation avoids a second Option branch.
        // HSS is only built when a swap pool exists, so a CUDA context is live.
        let kv_war_event = CudaEvent::new()?;
        // #11-refinement: MIRROR-RAW completion event (see field doc), created
        // unconditionally like `kv_war_event`; inert (never recorded/waited) when
        // prefetch is off.
        let kv_prefetch_done = CudaEvent::new()?;
        // Same var decode_a2.rs gates prefetch on; read once here so the batched
        // attend can skip io_uring's unconditional empty-read drain only when the
        // event fence replaces it (see field docs).
        let kv_prefetch_enabled = std::env::var_os("ATLAS_KV_PREFETCH").is_some();
        let disk_state = DiskState::new();
        Ok(Self {
            cfg,
            model,
            predictor,
            backend,
            pool,
            attn,
            eviction,
            scratch,
            max_seqs,
            batch,
            prefetch_stream,
            kv_war_event,
            kv_prefetch_done,
            kv_prefetch_enabled,
            disk_state,
        })
    }

    /// Phase 5: C — the max concurrent overflow-decode sequences the batched
    /// buffers are sized for (`$ATLAS_HSS_MAX_SEQS`, default 1).
    pub fn max_seqs(&self) -> usize {
        self.max_seqs
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

#[cfg(test)]
mod kv_backend_choice_tests {
    use super::{KvBackendChoice, kv_backend_choice};

    #[test]
    fn default_when_unset_or_unknown() {
        // Unset, empty, and typos all keep the safe default: try io_uring, fall
        // back to POSIX. A misspelled value must never silently force the slow path.
        assert_eq!(kv_backend_choice(None), KvBackendChoice::UringThenPosix);
        assert_eq!(kv_backend_choice(Some("")), KvBackendChoice::UringThenPosix);
        assert_eq!(kv_backend_choice(Some("psoix")), KvBackendChoice::UringThenPosix);
    }

    #[test]
    fn posix_is_forced() {
        assert_eq!(kv_backend_choice(Some("posix")), KvBackendChoice::ForcePosix);
    }

    #[test]
    fn io_uring_is_required_both_spellings() {
        assert_eq!(kv_backend_choice(Some("io_uring")), KvBackendChoice::RequireUring);
        assert_eq!(kv_backend_choice(Some("iouring")), KvBackendChoice::RequireUring);
    }
}
