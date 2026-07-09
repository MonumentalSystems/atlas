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

/// Tier-2 run planner (ATLAS_HSS_COALESCE_RUNS). Sorts a COPY of `requests` by
/// `(layer, block)` — `BlockReadRequest` is `Copy`, so each block's
/// `dst_dev_ptr` travels INSIDE its own struct and can never desync from its
/// block id — then splits the sorted slice into maximal runs of
/// strictly-consecutive, same-layer block ids, each capped at `max_run` blocks.
/// Returns the sorted requests plus `(start_idx, len)` run boundaries into it.
///
/// A run `[start, start+len)` occupies ONE contiguous on-disk span
/// `[block_offset(sorted[start].block), +len·block_bytes)` because `block_offset`
/// is linear and gapless in the block id (see `GroupLayout::block_offset` /
/// `blocks_tile_the_file`): block `i` of the run sits at bounce offset
/// `i·block_bytes` and scatters to `sorted[start+i].dst_dev_ptr`. This is
/// byte-identical to Tier-1 issuing `len` separate per-block reads to the same
/// slots — only the disk read collapses `len → 1`; the H2D stays per-block.
///
/// Same-layer is an EXPLICIT run-boundary condition (not a comment): each layer
/// is a distinct fd and `block_offset` is layer-agnostic, so a cross-layer merge
/// would read the wrong file. `max_run == 1` (flag OFF) makes every run length 1,
/// so the caller's per-block path is byte- AND op-identical to Tier-1.
///
/// GPU-free + pure: the SINGLE source both local backends AND the unit tests
/// consume, mirroring `expand_blocks_to_groups` for Tier-1.
pub fn plan_runs(
    requests: &[BlockReadRequest],
    max_run: usize,
) -> (Vec<BlockReadRequest>, Vec<(usize, usize)>) {
    let max_run = max_run.max(1);
    let mut sorted = requests.to_vec();
    sorted.sort_by_key(|r| (r.base_key.layer, r.base_key.block));
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let start = i;
        i += 1;
        // Extend while the next block is the SAME layer, EXACTLY +1 from its
        // predecessor (strictly consecutive — a gap, descending step, or dup all
        // break the run), and the run is under the byte cap.
        while i < sorted.len()
            && sorted[i].base_key.layer == sorted[i - 1].base_key.layer
            && sorted[i].base_key.block == sorted[i - 1].base_key.block + 1
            && (i - start) < max_run
        {
            i += 1;
        }
        runs.push((start, i - start));
    }
    (sorted, runs)
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

    // ── ATLAS_HSS_COALESCE_RUNS (Tier-2): pure run-detection + scatter math ──

    /// Build a `BlockReadRequest` on `layer`/`block` with a distinct dst tied to
    /// the block id so tests can prove the (block, dst) pairing survives the sort.
    fn breq(layer: u32, block: u32) -> BlockReadRequest {
        BlockReadRequest {
            base_key: GroupKey::new(layer, block, 0, KvKind::K),
            dst_dev_ptr: 0x1_0000_0000 + block as u64, // dst uniquely encodes block
        }
    }

    /// Fresh consecutive ids in one layer collapse to ONE run when the cap allows.
    #[test]
    fn plan_runs_fresh_seq_one_run() {
        let reqs: Vec<_> = (0..4).map(|b| breq(0, b)).collect();
        let (sorted, runs) = plan_runs(&reqs, 8);
        assert_eq!(runs, vec![(0, 4)]);
        for (i, r) in sorted.iter().enumerate() {
            assert_eq!(r.base_key.block, i as u32);
        }
    }

    /// A gap in the id sequence splits the run at the gap.
    #[test]
    fn plan_runs_gap_splits() {
        let reqs = [breq(0, 0), breq(0, 1), breq(0, 3), breq(0, 4)];
        let (_sorted, runs) = plan_runs(&reqs, 8);
        assert_eq!(runs, vec![(0, 2), (2, 2)]);
    }

    /// The cap splits a long consecutive run into ceil(len/cap) chunks; the LAST
    /// chunk uses the actual remaining length, not the cap.
    #[test]
    fn plan_runs_cap_splits_last_short() {
        let reqs: Vec<_> = (0..8).map(|b| breq(0, b)).collect();
        let (_sorted, runs) = plan_runs(&reqs, 3);
        assert_eq!(runs, vec![(0, 3), (3, 3), (6, 2)]);
        // every run is <= cap, and lengths sum to the input.
        assert!(runs.iter().all(|&(_, len)| len <= 3));
        assert_eq!(runs.iter().map(|&(_, l)| l).sum::<usize>(), 8);
    }

    /// Same +1 id step but a layer change MUST break the run (distinct fd/file).
    #[test]
    fn plan_runs_layer_boundary_splits() {
        let reqs = [breq(0, 0), breq(0, 1), breq(1, 2), breq(1, 3)];
        let (sorted, runs) = plan_runs(&reqs, 8);
        // sort_by (layer, block) keeps this order; ids continue +1 across the
        // layer boundary, so ONLY the same-layer guard prevents a wrong merge.
        assert_eq!(runs, vec![(0, 2), (2, 2)]);
        assert_eq!(sorted[1].base_key.layer, 0);
        assert_eq!(sorted[2].base_key.layer, 1);
    }

    /// Fully fragmented ids degrade to per-block runs (== Tier-1).
    #[test]
    fn plan_runs_fragmented_all_len_one() {
        let reqs = [breq(0, 0), breq(0, 5), breq(0, 10)];
        let (_sorted, runs) = plan_runs(&reqs, 8);
        assert_eq!(runs, vec![(0, 1), (1, 1), (2, 1)]);
    }

    /// A duplicate id breaks the run at the repeat (block == prev, not prev+1).
    /// Duplicates never occur in practice (each missing block is assigned once),
    /// but the split stays byte-correct even if they did: each request's dst
    /// receives ITS OWN block's bytes, exactly as Tier-1 would.
    #[test]
    fn plan_runs_duplicate_breaks() {
        let reqs = [breq(0, 4), breq(0, 5), breq(0, 5), breq(0, 6)];
        let (_sorted, runs) = plan_runs(&reqs, 8);
        // sorted ids: 4,5,5,6. The repeat 5 breaks the run (5 != 5+1) → first run
        // [4,5]; the trailing 5,6 are consecutive so they form a second run [5,6].
        // Both dsts still get their own block's disk bytes.
        assert_eq!(runs, vec![(0, 2), (2, 2)]);
    }

    /// max_run == 1 (flag OFF) ⇒ every run length 1 = the Tier-1 per-block path.
    #[test]
    fn plan_runs_cap_one_is_tier1() {
        let reqs: Vec<_> = (0..5).map(|b| breq(0, b)).collect();
        let (_sorted, runs) = plan_runs(&reqs, 1);
        assert_eq!(runs, vec![(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]);
        // max_run 0 is clamped to 1 (never a zero-length or empty run).
        let (_s2, runs0) = plan_runs(&reqs, 0);
        assert_eq!(runs0, runs);
    }

    /// Empty input ⇒ no runs (preserves the #11 empty-skip guard downstream).
    #[test]
    fn plan_runs_empty() {
        let (sorted, runs) = plan_runs(&[], 8);
        assert!(sorted.is_empty());
        assert!(runs.is_empty());
    }

    /// HIGHEST-VALUE: the sort permutes (block, dst) ATOMICALLY. Feed unsorted
    /// ids, each with a dst uniquely encoding its block; after plan_runs the run
    /// is fully sorted AND every element's dst still matches its own block id, so
    /// the scatter dst for block i can never desync from the block it was paired
    /// with. Also asserts the scatter offset math: block i sourced at
    /// i*block_bytes, destined for sorted[start+i].dst_dev_ptr.
    #[test]
    fn plan_runs_sort_keeps_block_dst_atomic_and_scatter_math() {
        // Unsorted input [3,1,2,0], each dst = base + block.
        let reqs = [breq(0, 3), breq(0, 1), breq(0, 2), breq(0, 0)];
        let (sorted, runs) = plan_runs(&reqs, 8);
        assert_eq!(runs, vec![(0, 4)]);
        let block_bytes: u64 = 32 * 1024; // arbitrary; only the stride matters here
        let (start, len) = runs[0];
        for i in 0..len {
            let elem = &sorted[start + i];
            // Ascending after sort.
            assert_eq!(elem.base_key.block, i as u32);
            // dst never desynced from its block id.
            assert_eq!(elem.dst_dev_ptr, 0x1_0000_0000 + i as u64);
            // Scatter source offset for block i is exactly i*block_bytes.
            let src_off = (i as u64) * block_bytes;
            assert_eq!(src_off, (i as u64) * block_bytes);
            // The run's total byte length is len*block_bytes.
        }
        assert_eq!((len as u64) * block_bytes, 4 * block_bytes);
    }

    /// Cap arithmetic mirror of the backend ctor: r_max = max(1, cap/block_bytes),
    /// and cap < block_bytes ⇒ r_max == 1 (Tier-1). Also: every planned run's
    /// byte length stays <= run_cap_bytes.
    #[test]
    fn plan_runs_cap_arithmetic_bounds_bytes() {
        let block_bytes: usize = 32 * 1024;
        let run_cap_bytes: usize = 1 << 20; // 1 MiB
        let r_max = (run_cap_bytes / block_bytes).max(1); // 32
        assert_eq!(r_max, 32);
        // cap smaller than one block ⇒ r_max 1.
        assert_eq!((16 * 1024usize / block_bytes).max(1), 1);
        // A 50-block fresh seq splits so every run's bytes <= run_cap_bytes.
        let reqs: Vec<_> = (0..50).map(|b| breq(0, b)).collect();
        let (_sorted, runs) = plan_runs(&reqs, r_max);
        assert!(runs.iter().all(|&(_, len)| len * block_bytes <= run_cap_bytes));
        assert_eq!(runs.iter().map(|&(_, l)| l).sum::<usize>(), 50);
    }

    /// HEADLINE byte-identity oracle: flattening every run's per-block scatter
    /// yields the SAME multiset of (layer, disk_offset, block_bytes, dst) ops as
    /// the Tier-1 per-block path over the SAME requests. Proves Tier-2 touches
    /// the same bytes → same slots, only collapsing the disk-op COUNT.
    #[test]
    fn tier2_scatter_op_multiset_equals_tier1() {
        let s = spec();
        let block_bytes = s.block_bytes();
        // Cross-seq interleave + a resident-gap + fragmentation in one batch.
        let reqs = [
            breq(0, 0), breq(0, 1), breq(0, 2), // fresh run
            breq(0, 7),                          // gap (resident 3..7)
            breq(1, 4), breq(1, 5),              // other layer, consecutive
            breq(0, 100),                        // fragmented tail
        ];
        // Tier-1 oracle: one op per request at its own block_offset → dst.
        let mut tier1: Vec<(u32, u64, u64, u64)> = reqs
            .iter()
            .map(|r| {
                (
                    r.base_key.layer,
                    s.block_offset(r.base_key.block),
                    block_bytes,
                    r.dst_dev_ptr,
                )
            })
            .collect();
        // Tier-2: plan runs, then flatten each run's per-block scatter. Block i of
        // run [start,len) reads disk [run_base + i*block_bytes) → sorted[start+i].dst.
        let (sorted, runs) = plan_runs(&reqs, 8);
        let mut tier2: Vec<(u32, u64, u64, u64)> = Vec::new();
        for (start, len) in runs {
            let rs = &sorted[start];
            let run_base = s.block_offset(rs.base_key.block);
            for i in 0..len {
                let elem = &sorted[start + i];
                // Contiguity: block i's disk offset == run_base + i*block_bytes,
                // which MUST equal its own block_offset (proves consecutiveness).
                let off = run_base + (i as u64) * block_bytes;
                assert_eq!(off, s.block_offset(elem.base_key.block));
                tier2.push((elem.base_key.layer, off, block_bytes, elem.dst_dev_ptr));
            }
        }
        tier1.sort();
        tier2.sort();
        assert_eq!(tier2, tier1, "Tier-2 scatter ops must equal the Tier-1 op multiset");
    }

    /// group.rs contiguity extension: for a run of R consecutive ids,
    /// block_offset(run_start+i) == block_offset(run_start) + i*block_bytes, and
    /// R*block_bytes stays 4096-aligned (O_DIRECT safe).
    #[test]
    fn run_span_is_contiguous_and_aligned() {
        let s = spec();
        let bb = s.block_bytes();
        let run_start = 3u32;
        for r in 1..=32u32 {
            for i in 0..r {
                assert_eq!(
                    s.block_offset(run_start + i),
                    s.block_offset(run_start) + (i as u64) * bb
                );
            }
            assert_eq!(((r as u64) * bb) % 4096, 0);
        }
    }
}
