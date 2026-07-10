// SPDX-License-Identifier: AGPL-3.0-only

//! The generic tiered-cache core — lifted VERBATIM from
//! `spark-storage/src/snapshot_swap.rs` (tiered-cache consolidation step 2,
//! docs/streaming-experts/TIERED-CACHE-CONSOLIDATION.md).
//!
//! One mechanism, three seams:
//!   * [`SlotArena`]  — the bounded HOT tier: `num_slots` fixed-size byte slots
//!     (the peer's mmap'd RDMA MR, a host-RAM `Vec`, …). No CUDA/HBM impl may
//!     live in this crate — that belongs to consumer crates.
//!   * [`SwapStore`]  — the unbounded COLD tier: a fixed-stride record store
//!     ([`DirectSwapFile`] on NVMe, [`MemSwapStore`] in RAM, …).
//!   * [`Residency`]  — the page table over both (was `SnapshotResidency`):
//!     opaque `u64` key → byte-agnostic fixed-size blob, two-level LRU (RAM
//!     `lru` above `disk_lru`), read-pins so a slot is never reused mid-read,
//!     and NEVER-reject puts (a full arena spills the coldest resident to
//!     disk; a capped disk drops its coldest key → clean later miss).
//!
//! This crate is CPU/disk-only: deps are `anyhow` + `libc` (unix). It is fully
//! unit-testable without RDMA or a GPU, and it is the reason the peer daemons
//! (`atlas-expert-pack`, `spark-storage` with `default-features = false`)
//! build CUDA-free. The peer wire protocol (PAGING_MAGIC, paging loops, client
//! codec) deliberately did NOT move — it stays in `spark-storage::snapshot_swap`,
//! which re-exports this core so existing consumers compile unchanged.

mod direct_swap;
mod mem;
mod residency;
mod traits;

pub use direct_swap::DirectSwapFile;
pub use mem::{MemSwapStore, VecSlotArena};
pub use residency::Residency;
pub use traits::{SlotArena, SwapStats, SwapStore};
