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

use crate::group::{GroupKey, GroupLayout, KvKind};

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

/// One block-granular read request (ATLAS_HSS_COALESCE_BLOCKS): pull the entire
/// block — all `num_kv_heads` × {K,V}, one contiguous on-disk span of
/// `block_bytes` — into the device slot based at `dst_dev_ptr`
/// (== `ScratchPool::slot_dev_ptr(slot)`). `base_key` carries `kv_head = 0,
/// kind = K` by convention; only its `layer` and `block` are load-bearing (the
/// block offset does NOT route through kv_head/kind — see `block_offset`).
#[derive(Clone, Copy, Debug)]
pub struct BlockReadRequest {
    pub base_key: GroupKey,
    pub dst_dev_ptr: u64,
}

/// Expand each `BlockReadRequest` into the exact `2·nkv` per-head `ReadRequest`s
/// the un-coalesced path issues, in the SAME order the caller loops emit
/// (interleaved `K(kh), V(kh)` for `kh` in `0..nkv`) with device destinations
/// at `dst + kh·gs` (K) and `dst + (nkv+kh)·gs` (V).
///
/// This is the SINGLE source of the per-head fan-out: the default `read_blocks`
/// / `write_block_from_host` trait impls AND the unit tests consume it, so the
/// RDMA/Cascade backends (which inherit the default) can never drift from the
/// caller-side per-head layout, and byte-identity is pinned host-side.
pub fn expand_blocks_to_groups(spec: &GroupLayout, reqs: &[BlockReadRequest]) -> Vec<ReadRequest> {
    let nkv = spec.num_kv_heads;
    let gs = spec.group_stride;
    let mut out = Vec::with_capacity(reqs.len() * 2 * nkv as usize);
    for r in reqs {
        let layer = r.base_key.layer;
        let block = r.base_key.block;
        for kh in 0..nkv {
            out.push(ReadRequest {
                group: GroupKey::new(layer, block, kh, KvKind::K),
                dst_dev_ptr: r.dst_dev_ptr + (kh as u64) * gs,
            });
            out.push(ReadRequest {
                group: GroupKey::new(layer, block, kh, KvKind::V),
                dst_dev_ptr: r.dst_dev_ptr + (nkv as u64 + kh as u64) * gs,
            });
        }
    }
    out
}

pub trait StorageBackend: Send + Sync {
    /// Synchronously fulfil all `requests`, returning when the corresponding
    /// HBM destinations are populated and visible on `stream`. The backend
    /// chooses how to schedule (blocking POSIX `pread`, batched `io_uring`,
    /// etc.). At return, the `stream` has been synchronised so the caller
    /// can issue subsequent kernels that depend on the data.
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()>;

    /// Async variant of `read` (#11-refinement): enqueue the tier read + H2D on
    /// `stream` and return WITHOUT a terminal host `stream_sync`. Mirror-RAW is
    /// the CALLER's job — HighSpeedSwap records `kv_prefetch_done` on this
    /// in-order stream right after this returns and the consumer waits it
    /// cross-stream, so the decode host thread never blocks on main compute at
    /// the prefetch boundary. Staging/bounce reuse MUST be made safe INTERNALLY
    /// (per-buffer completion events + FIFO reuse), NOT by a host sync.
    ///
    /// Default = the synchronous `read` (correct, just not async), so posix and
    /// any future backend need no change and the on-demand `read` path stays
    /// byte-identical.
    fn read_async(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read(requests, stream)
    }

    /// One-shot sequential write — used at offload time to populate disk
    /// from a host-side K/V buffer.
    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()>;

    /// Immutable disk/device geometry. The default block methods below use it to
    /// fan a block op back out to the per-head path; io_uring/posix return their
    /// layout spec, Cascade delegates to its backing, RDMA returns its layout.
    fn group_layout(&self) -> GroupLayout;

    /// Block-granular read (ATLAS_HSS_COALESCE_BLOCKS): fulfil each request with
    /// ONE contiguous `block_bytes` op instead of `2·nkv` per-head reads. Same
    /// stream contract as `read`. The DEFAULT fans out to `read` via
    /// `expand_blocks_to_groups`, so posix/RDMA/Cascade stay correct (just
    /// un-coalesced) with no change — and because the caller only calls this when
    /// the flag is ON, flag-OFF is byte- AND op-identical. io_uring and posix
    /// override it with a real single op.
    fn read_blocks(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        let groups = expand_blocks_to_groups(&self.group_layout(), requests);
        self.read(&groups, stream)
    }

    /// Async block-granular read — the coalesced twin of `read_async` for the
    /// prefetch path. DEFAULT fans out to `read_async`.
    fn read_blocks_async(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        let groups = expand_blocks_to_groups(&self.group_layout(), requests);
        self.read_async(&groups, stream)
    }

    /// Block-granular write: ONE contiguous `block_bytes` op. `src` is exactly
    /// `block_bytes` laid out `[K0,K1,…,K(nkv-1),V0,…,V(nkv-1)]` at `group_stride`
    /// pitch (see `assemble_block_write_buffer`). `base_key` carries the block
    /// identity (kv_head/kind ignored). DEFAULT splits `src` back into the
    /// `2·nkv` per-head `group_stride` stripes and calls `write_from_host` per
    /// head — byte-identical on-disk image to the un-coalesced path.
    fn write_block_from_host(&mut self, base_key: GroupKey, src: &[u8]) -> Result<()> {
        let spec = self.group_layout();
        let nkv = spec.num_kv_heads as usize;
        let gs = spec.group_stride as usize;
        let expect = 2 * nkv * gs;
        if src.len() != expect {
            anyhow::bail!(
                "write_block_from_host: src len {} != block bytes {expect}",
                src.len()
            );
        }
        let layer = base_key.layer;
        let block = base_key.block;
        for kh in 0..nkv {
            let k_off = kh * gs;
            let v_off = (nkv + kh) * gs;
            self.write_from_host(
                GroupKey::new(layer, block, kh as u16, KvKind::K),
                &src[k_off..k_off + gs],
            )?;
            self.write_from_host(
                GroupKey::new(layer, block, kh as u16, KvKind::V),
                &src[v_off..v_off + gs],
            )?;
        }
        Ok(())
    }

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

#[cfg(test)]
mod coalesce_tests {
    //! GPU-free tests for the ATLAS_HSS_COALESCE_BLOCKS block↔group mapping.
    //! These exercise only host-side pointer/offset arithmetic (no CUDA
    //! allocation, no I/O), so they RUN under the default cuda feature.
    use super::*;
    use crate::group::GroupLayout;
    use crate::scratch_pool::ScratchDims;

    fn spec() -> GroupLayout {
        // Holo-like: 8 kv_heads, block_size 16, head_dim 128, BF16 → gs 4096.
        GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096)
    }

    /// expand_blocks_to_groups yields exactly the interleaved per-head requests
    /// the caller loops would push, in order, with the right disk keys + device
    /// destinations relative to the slot base. This pins the default fan-out
    /// (and thus RDMA/Cascade correctness) byte-for-byte without a GPU.
    #[test]
    fn expand_matches_the_per_head_loop() {
        let s = spec();
        let nkv = s.num_kv_heads;
        let gs = s.group_stride;
        let base = 0xDEAD_0000u64;
        let (layer, block) = (7u32, 12u32);
        let br = BlockReadRequest {
            base_key: GroupKey::new(layer, block, 0, KvKind::K),
            dst_dev_ptr: base,
        };
        let got = expand_blocks_to_groups(&s, &[br]);
        assert_eq!(got.len(), 2 * nkv as usize);
        for kh in 0..nkv {
            let k = got[(2 * kh) as usize];
            let v = got[(2 * kh + 1) as usize];
            assert_eq!(k.group, GroupKey::new(layer, block, kh, KvKind::K));
            assert_eq!(k.dst_dev_ptr, base + (kh as u64) * gs);
            assert_eq!(v.group, GroupKey::new(layer, block, kh, KvKind::V));
            assert_eq!(v.dst_dev_ptr, base + (nkv as u64 + kh as u64) * gs);
        }
    }

    /// Multiple blocks in one call expand to concatenated per-block groups with
    /// independent bases (robustness / batching).
    #[test]
    fn expand_handles_multiple_blocks() {
        let s = spec();
        let nkv = s.num_kv_heads as usize;
        let reqs = [
            BlockReadRequest { base_key: GroupKey::new(1, 2, 0, KvKind::K), dst_dev_ptr: 0x1000 },
            BlockReadRequest { base_key: GroupKey::new(1, 9, 0, KvKind::K), dst_dev_ptr: 0x9000 },
        ];
        let got = expand_blocks_to_groups(&s, &reqs);
        assert_eq!(got.len(), 2 * (2 * nkv));
        assert_eq!(got[0].group, GroupKey::new(1, 2, 0, KvKind::K));
        assert_eq!(got[0].dst_dev_ptr, 0x1000);
        assert_eq!(got[2 * nkv].group, GroupKey::new(1, 9, 0, KvKind::K));
        assert_eq!(got[2 * nkv].dst_dev_ptr, 0x9000);
    }

    /// Cross-module disk↔device equality WITHOUT a GPU: block_bytes ==
    /// ScratchDims::slot_bytes, and the device index map (slot pointers relative
    /// to the slot base, computed via ScratchDims' pure arithmetic against a
    /// fake base) equals the disk index map (file_offset relative to
    /// block_offset) for every kv_head. This is the byte-identity proof.
    #[test]
    fn device_slot_map_equals_disk_block_map() {
        let s = spec();
        let dims = ScratchDims {
            num_slots: 4,
            num_kv_heads: s.num_kv_heads,
            group_stride: s.group_stride,
        };
        assert_eq!(s.block_bytes() as usize, dims.slot_bytes());
        let block = 3u32;
        let base = 0u64; // fake pool base; relative offsets are base-independent
        let slot = 2u32;
        let slot_base = dims.slot_base(base, slot);
        for kh in 0..s.num_kv_heads {
            assert_eq!(
                dims.k_ptr(base, slot, kh) - slot_base,
                s.file_offset(GroupKey::new(0, block, kh, KvKind::K)) - s.block_offset(block),
            );
            assert_eq!(
                dims.v_ptr(base, slot, kh) - slot_base,
                s.file_offset(GroupKey::new(0, block, kh, KvKind::V)) - s.block_offset(block),
            );
        }
    }

    /// A recording StorageBackend: the default `read_blocks` /
    /// `write_block_from_host` must emit the IDENTICAL ordered (layer, offset,
    /// bytes, dst) op stream as the hand-written per-head path. Confirms the
    /// default fan-out inherited by RDMA/Cascade is op-equivalent.
    struct Recorder {
        spec: GroupLayout,
        reads: Vec<(u32, u64, usize, u64)>,
        writes: Vec<(u32, u64, usize)>,
    }
    impl StorageBackend for Recorder {
        fn read(&mut self, requests: &[ReadRequest], _stream: u64) -> Result<()> {
            let gb = self.spec.group_bytes() as usize;
            for r in requests {
                self.reads
                    .push((r.group.layer, self.spec.file_offset(r.group), gb, r.dst_dev_ptr));
            }
            Ok(())
        }
        fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
            self.writes
                .push((key.layer, self.spec.file_offset(key), src.len()));
            Ok(())
        }
        fn group_layout(&self) -> GroupLayout {
            self.spec
        }
    }

    #[test]
    fn default_read_blocks_op_equivalent_to_per_head() {
        let s = spec();
        let base = 0x4000u64;
        let br = BlockReadRequest {
            base_key: GroupKey::new(4, 6, 0, KvKind::K),
            dst_dev_ptr: base,
        };
        // Oracle: what the per-head loop expansion records.
        let mut oracle = Recorder { spec: s, reads: vec![], writes: vec![] };
        let groups = expand_blocks_to_groups(&s, &[br]);
        oracle.read(&groups, 0).unwrap();
        // Subject: the default read_blocks.
        let mut subject = Recorder { spec: s, reads: vec![], writes: vec![] };
        subject.read_blocks(&[br], 0).unwrap();
        assert_eq!(subject.reads, oracle.reads);
    }

    #[test]
    fn default_write_block_op_equivalent_to_per_head() {
        let s = spec();
        let nkv = s.num_kv_heads as usize;
        let gs = s.group_stride as usize;
        let mut rec = Recorder { spec: s, reads: vec![], writes: vec![] };
        let buf = vec![0u8; 2 * nkv * gs];
        rec.write_block_from_host(GroupKey::new(2, 5, 0, KvKind::K), &buf)
            .unwrap();
        // 2*nkv per-head writes, each group_bytes, at the per-head offsets.
        assert_eq!(rec.writes.len(), 2 * nkv);
        let mut expected: Vec<(u32, u64, usize)> = Vec::new();
        for kh in 0..s.num_kv_heads {
            expected.push((2, s.file_offset(GroupKey::new(2, 5, kh, KvKind::K)), gs));
            expected.push((2, s.file_offset(GroupKey::new(2, 5, kh, KvKind::V)), gs));
        }
        assert_eq!(rec.writes, expected);
        // Length guard.
        let bad = vec![0u8; 2 * nkv * gs - 1];
        assert!(rec.write_block_from_host(GroupKey::new(2, 5, 0, KvKind::K), &bad).is_err());
    }
}
