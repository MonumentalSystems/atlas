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
// Weight-staging peer + wire protocol (RDMA weight tier for fast model swaps).
// Wire types are un-gated (unit-testable on metal/skip); the mmap+reg_mr server
// body is `cfg(unix)` and the verbs handshake compiles under atlas_rdma_verbs.
pub mod weight_peer;
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
pub mod cache_peer;
// One-sided RDMA READ/WRITE verbs FFI (WS2 Phase B + KV overflow). Moved to
// the CUDA-free `atlas-rdma` crate (RailSet extraction Step A); re-exported at
// its old path so every `use crate::rdma_verbs::{Verbs, MrKeys, Gid}` site
// compiles unchanged. Compiled only where the C shim is (atlas-rdma's build.rs
// emits `atlas_rdma_verbs` on Linux + rdma-core; ours re-emits it — see build.rs).
pub mod group;
#[cfg(atlas_rdma_verbs)]
pub use atlas_rdma::verbs as rdma_verbs;

/// Permanent cfg witness (RailSet extraction Step A): `true` iff the
/// `atlas_rdma_verbs` cfg was re-emitted for THIS crate by build.rs. The unit
/// test in `rdma_verbs_probe_tests.rs` asserts it, so a silent cfg
/// evaporation (e.g. the DEP_ATLAS_RDMA_SHIM_HAS_VERBS re-emit breaking)
/// fails `cargo test -p spark-storage --lib` on verbs hosts instead of
/// green-building with all nine gated modules compiled out.
pub const fn rdma_verbs_enabled() -> bool {
    cfg!(atlas_rdma_verbs)
}

#[cfg(test)]
mod rdma_verbs_probe_tests;
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
// RDMA weight loader (client of weight_peer). Needs cuda (pinned bounce +
// copy_h2d + GpuBackend from spark-runtime); the verbs data path is gated on
// atlas_rdma_verbs inside, with a runtime bail otherwise — the expert_tier_rdma
// precedent. Builds a WeightStore byte-identical to the disk loaders.
#[cfg(feature = "cuda")]
pub mod weight_tier_rdma;
// RDMA LoRA staging: land a named adapter's A/B into a resident pool SLOT for
// fast rotation. Same weight_peer wire + verbs stack as weight_tier_rdma;
// landing byte-identical to the disk pack (convert + B row-repack).
#[cfg(feature = "cuda")]
pub mod weight_lora_rdma;
// KV overflow tier (StorageBackend over RDMA). Needs both cuda (pinned bounce +
// copy_h2d) and the verbs shim.
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub mod rdma_kv_backend;

// Step B (tiered-cache consolidation): KV as a first-class paging kind behind
// the default-OFF ATLAS_KV_PAGING flag. The namespace/wire-key derivation and
// the strict env resolvers are pure and un-gated (unit-testable on the
// default-features=false peer path); the RDMA client backend is gated like
// rdma_kv_backend (cuda + verbs) inside the module.
pub mod kv_paging;
// Phase 4b: offset-addressed RDMA arena for the SSM-snapshot spill tier (reuses
// the same verbs + cache-peer blade protocol, keyed by byte offset not GroupKey).
// Always available — the module provides a `connect`-errors stub when the real
// transport (feature `cuda` + atlas_rdma_verbs) isn't built, so dependents can
// reference `RdmaSnapshotArena` unconditionally and degrade to host-RAM.
pub mod rdma_snapshot;
pub mod snapshot_swap;
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
pub use config::HighSpeedSwapConfig;
pub use eviction::EvictionPolicy;
pub use expert::{
    ExpertKey, ExpertLayout, ExpertRecordHeader, ExpertRecordId, ExpertRecordSpec, Proj, ProjBytes,
};
#[cfg(feature = "cuda")]
pub use expert_arena::ExpertArena;
#[cfg(unix)]
pub use expert_pack::{ExpertFileReader, ExpertFileWriter};
pub use expert_pack::{ExpertIndex, ProjData, ProjView, pack_record, unpack_record};
#[cfg(feature = "cuda")]
pub use expert_tier::{
    ArenaSlot, ExpertResidency, ExpertTier, PosixTier, TierKind, UmaArenaTier, open_tier,
};
#[cfg(feature = "cuda")]
pub use expert_tier_rdma::RdmaTier;
#[cfg(feature = "cuda")]
pub use high_speed_swap::{AttendSeqReq, HighSpeedSwap, install_local, local_installed, with_local};
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub use kv_paging::KvPagingBackend;
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub use rdma_kv_backend::RdmaKvBackend;
pub use rdma_snapshot::RdmaSnapshotArena;
#[cfg(feature = "cuda")]
pub use weight_lora_rdma::{LoraAbKind, LoraLandTarget, RdmaLoraLoader};
pub use weight_peer::{WeightManifest, WeightTensorRecord};
#[cfg(feature = "cuda")]
pub use weight_tier_rdma::RdmaWeightLoader;

// Non-cuda stub surface — same names as the real CUDA orchestrator
// above so spark-model's call sites compile unchanged. `with_local`
// always returns None (orchestrator absent), `local_installed` is
// false, and `install_local` bails — see `stubs.rs` for rationale.
#[cfg(not(feature = "cuda"))]
mod stubs;
#[cfg(not(feature = "cuda"))]
pub use stubs::{AttendSeqReq, HighSpeedSwap, install_local, local_installed, with_local};

#[cfg(feature = "cuda")]
pub use predictor::{Predictor, PredictorDims};
#[cfg(feature = "cuda")]
pub use probe::{Backend, ProbeConfig, ProbeResult, run_probe};
pub use projection::{PredictorShape, build_projection};
#[cfg(feature = "cuda")]
pub use tiled_attention::{TiledAttention, TiledAttentionDims};
