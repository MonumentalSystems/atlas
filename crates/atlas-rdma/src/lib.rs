// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

//! atlas-rdma: the one-sided RDMA verbs primitive shared by every Atlas RDMA
//! tier (experts / KV overflow / weight staging / LoRA / SSM snapshots).
//!
//! Step A of the RailSet extraction (tiered-cache consolidation, chunk 2):
//! this crate OWNS what used to be `spark-storage/src/rdma_verbs.rs` and
//! `spark-storage/src/rdma_shim.c`, moved verbatim — the shim's QP/RTR/RTS
//! attribute constants are interop-visible to the live gx10 peer running an
//! older binary, so the .c file is byte-identical. No client refactor here.
//!
//! CUDA-free by hard constraint (see Cargo.toml): both the non-cuda peer
//! daemons (`atlas-expert-pack`) and the cuda client tiers link this.
//!
//! `cfg(atlas_rdma_verbs)` is decided by build.rs (Linux + rdma-core, not
//! skipped) and published to direct dependents via the `links` metadata
//! `DEP_ATLAS_RDMA_SHIM_HAS_VERBS` — see build.rs for the full story.

/// Safe-ish wrapper over the C shim: one [`verbs::Verbs`] == one RC QP.
/// Compiled only where the shim is (`cfg(atlas_rdma_verbs)`).
#[cfg(atlas_rdma_verbs)]
pub mod verbs;

#[cfg(atlas_rdma_verbs)]
pub use verbs::{Gid, MrKeys, Verbs};

/// `true` iff this build of atlas-rdma compiled the verbs shim (i.e. the
/// `atlas_rdma_verbs` cfg was emitted by build.rs). A permanent, always-
/// compiled witness: tests assert it so a silent cfg evaporation fails
/// `cargo test` on verbs hosts instead of green-building an empty crate.
pub const fn verbs_enabled() -> bool {
    cfg!(atlas_rdma_verbs)
}
