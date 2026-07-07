// SPDX-License-Identifier: AGPL-3.0-only
//
// Storage backend trait + impls for the high-speed-swap path.
//
// SBIO contract: tiled-attention / scratch-pool code never opens a file or
// issues a syscall. Every NVMe-touching operation flows through a
// `StorageBackend` impl, so the predictor / scratch / kernel layers can be
// tested with the deterministic POSIX backend and swap in the io_uring
// production backend transparently.

use anyhow::Result;

use crate::group::GroupKey;

pub mod io_uring;
pub mod posix;

pub use self::io_uring::IoUringBackend;
pub use posix::PosixBackend;

/// One read request: pull `group` from disk, land it at `dst_dev_ptr`.
#[derive(Clone, Copy, Debug)]
pub struct ReadRequest {
    pub group: GroupKey,
    pub dst_dev_ptr: u64,
}

pub trait StorageBackend: Send + Sync {
    /// Fulfil all `requests`, landing each into its HBM destination. The
    /// backend chooses how to schedule (blocking POSIX `pread`, batched
    /// `io_uring`, RDMA, etc.). At return, the reads are stream-ordered before
    /// any subsequent work issued on the SAME `stream` (the caller may enqueue
    /// dependent kernels on `stream` without extra sync), but the CPU is NOT
    /// guaranteed to have observed completion and a consumer on a DIFFERENT
    /// stream must synchronise itself. Bounce/staging buffers are recycled
    /// lazily via per-buffer CUDA events, so callers must keep issuing on the
    /// same stream.
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()>;

    /// One-shot sequential write — used at offload time to populate disk
    /// from a host-side K/V buffer.
    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()>;

    /// Optionally pre-register `[base, base+len)` as the read-landing region.
    /// The RDMA backend registers it as ONE MR (per rail) so zero-copy restore
    /// reuses that lkey for every slot within it — registering per-slot
    /// sub-regions of a CUDA-pinned pool fails on GB10, but the whole-allocation
    /// registration from its base succeeds. No-op for the file backends.
    fn register_landing_region(&mut self, base: u64, len: usize) -> Result<()> {
        let _ = (base, len);
        Ok(())
    }
}
