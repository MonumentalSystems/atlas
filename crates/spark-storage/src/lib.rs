// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

// Atlas spark-storage: high-speed NVMe-backed KV cache offload.
//
// Phase 0 of `--high-speed-swap` (see plan at
// /workspace/.claude/plans/i-want-to-ensure-valiant-bunny.md): runtime probe
// that decides whether the production backend should be cuFile/GDS or
// io_uring + pinned-host bounce. Later phases add the predictor, scratch
// pool, eviction, and I/O thread.
//
// Feature gating: every module that touches the CUDA driver (raw FFI in
// `cuda_min`, the module/event helpers in `cuda_module`, anything that
// holds a `DeviceBuffer`) is gated behind the `cuda` feature so the
// crate compiles on Apple Silicon (`--no-default-features --features
// metal`) where the high-speed-swap path won't be reachable anyway.

#[cfg(feature = "cuda")]
pub mod cuda_graph;
#[cfg(feature = "cuda")]
pub mod cuda_min;
#[cfg(feature = "cuda")]
pub mod cuda_module;

// Re-export the module/event/launch helpers from their new home so existing
// `use spark_storage::cuda_min::{CudaModule, CudaEvent, launch_kernel}` paths
// keep working.
#[cfg(feature = "cuda")]
pub use cuda_module::{CudaEvent, CudaModule, launch_kernel};

// Pure CPU-side modules — types, configs, references. Always compiled.
pub mod attention_ref;
pub mod config;
pub mod eviction;
pub mod expert;
pub mod expert_pack;
pub mod expert_peer;
// Process-global commit ledger shared by the RW/RO memory blades. Un-gated pure
// arithmetic (outside atlas_rdma_verbs) so the cap logic is unit-testable on the
// metal/skip build; consumed by the two unix `server_impl` modules.
#[cfg(unix)]
// Some accessors/paths are only exercised by tests or by the verbs-gated
// handshake, so allow dead_code across build configs (metal/skip vs verbs).
#[allow(dead_code)]
pub(crate) mod blade_cap;
// KV cache overflow blade: RW remote-RAM tier (wire types always available; the
// verbs server compiles under atlas_rdma_verbs). Faster-than-SSD KV overflow.
pub mod kv_peer;
// One-sided RDMA READ/WRITE verbs FFI (WS2 Phase B + KV overflow). CUDA-free so
// BOTH the non-cuda peer servers and the cuda client tiers can use it. Compiled
// only where the C shim is (build.rs emits `atlas_rdma_verbs` on Linux + rdma-core).
#[cfg(atlas_rdma_verbs)]
pub mod rdma_verbs;
pub mod group;
// T1 tier-cascade placement/eviction policy (pure LRU; unit-testable on
// metal/skip). The cuda composite backend is `cascade_backend`.
pub mod cascade_policy;
pub mod model_dims;
pub mod predictor_ref;
pub mod projection;

// `ModelDims` is a plain POD struct (no GPU state) that
// `spark-model`'s public surface threads through every layer's
// forward signature; it must stay reachable on metal builds even
// though the high-speed-swap orchestrator that consumes it is
// CUDA-gated.
pub use model_dims::ModelDims;

// `layout` opens disk files with `O_DIRECT` and pre-allocates via
// `posix_fallocate` — both Linux-specific. Only the cuda-side modules
// (high_speed_swap, backend/io_uring, backend/posix) consume it, so
// gating it on the cuda feature is sufficient.
#[cfg(feature = "cuda")]
pub mod layout;

// CUDA-only modules: each holds raw `cu*` FFI calls or a `DeviceBuffer`,
// or transitively imports from the cuda_* modules above. Gated together
// because separating them would just smear the boundary.
#[cfg(feature = "cuda")]
pub mod backend;
#[cfg(feature = "cuda")]
pub mod bench;
#[cfg(feature = "cuda")]
pub mod expert_arena;
#[cfg(feature = "cuda")]
pub mod expert_tier;
#[cfg(feature = "cuda")]
pub mod expert_tier_rdma;
// KV overflow tier (StorageBackend over RDMA). Needs both cuda (pinned bounce +
// copy_h2d) and the verbs shim.
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub mod rdma_kv_backend;
// T1 write-back cache composite (wraps any StorageBackend). cuda but not verbs.
#[cfg(feature = "cuda")]
pub mod cascade_backend;
#[cfg(feature = "cuda")]
pub mod high_speed_swap;
#[cfg(feature = "cuda")]
pub mod predictor;
#[cfg(feature = "cuda")]
pub mod probe;
#[cfg(feature = "cuda")]
pub mod scratch_pool;
#[cfg(feature = "cuda")]
pub mod tiled_attention;

#[cfg(feature = "cuda")]
pub use backend::{IoUringBackend, PosixBackend, ReadRequest, StorageBackend};
#[cfg(feature = "cuda")]
pub use expert_arena::ExpertArena;
#[cfg(feature = "cuda")]
pub use expert_tier::{
    ArenaSlot, ExpertResidency, ExpertTier, PosixTier, TierKind, UmaArenaTier, open_tier,
};
#[cfg(feature = "cuda")]
pub use expert_tier_rdma::RdmaTier;
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub use rdma_kv_backend::RdmaKvBackend;
pub use config::HighSpeedSwapConfig;
pub use eviction::EvictionPolicy;
pub use expert::{
    ExpertKey, ExpertLayout, ExpertRecordHeader, ExpertRecordId, ExpertRecordSpec, Proj, ProjBytes,
};
pub use expert_pack::{ExpertIndex, ProjData, ProjView, pack_record, unpack_record};
#[cfg(unix)]
pub use expert_pack::{ExpertFileReader, ExpertFileWriter};
#[cfg(feature = "cuda")]
pub use high_speed_swap::{HighSpeedSwap, install_local, local_installed, with_local};

// Non-cuda stub surface — same names as the real CUDA orchestrator
// above so spark-model's call sites compile unchanged. `with_local`
// always returns None (orchestrator absent), `local_installed` is
// false, and `install_local` bails — see `stubs.rs` for rationale.
#[cfg(not(feature = "cuda"))]
mod stubs;
#[cfg(not(feature = "cuda"))]
pub use stubs::{HighSpeedSwap, install_local, local_installed, with_local};

#[cfg(feature = "cuda")]
pub use predictor::{Predictor, PredictorDims};
#[cfg(feature = "cuda")]
pub use probe::{Backend, ProbeConfig, ProbeResult, run_probe};
pub use projection::{PredictorShape, build_projection};
#[cfg(feature = "cuda")]
pub use tiled_attention::{TiledAttention, TiledAttentionDims};
