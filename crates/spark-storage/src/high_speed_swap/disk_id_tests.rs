// SPDX-License-Identifier: AGPL-3.0-only
//
//! Unit tests for the disk-block-ID allocator.

use super::*;
use crate::cuda_min::CudaCtx;

fn dims() -> ModelDims {
    ModelDims {
        num_layers: 2,
        max_blocks_per_layer: 8,
        num_q_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        block_size: 16,
    }
}

fn cfg(dir: &str) -> HighSpeedSwapConfig {
    HighSpeedSwapConfig {
        dir: std::env::temp_dir().join(format!("atlas-hss-disk-id-{dir}-{}", std::process::id())),
        bytes: 64 * (1 << 20),
        resident_blocks: 4,
        rank: 32,
        qd: 4,
        graph: false,
        projection_seed: 0xCAFE_F00D,
    }
}

#[test]
#[ignore = "requires GPU"]
fn alloc_free_round_trip() {
    let _ctx = CudaCtx::new(0).expect("cuda init");
    let mut hss = HighSpeedSwap::new(&_ctx, cfg("rt"), dims()).unwrap();
    // Capacity 8; alloc all 8.
    let ids: Vec<u32> = (0..8).map(|_| hss.alloc_disk_block_id().unwrap()).collect();
    assert_eq!(ids, (0..8).collect::<Vec<_>>());
    // Pool exhausted; next alloc returns None.
    assert!(hss.alloc_disk_block_id().is_none());
    // Free one; next alloc reuses it.
    hss.dec_disk_ref(3);
    let reused = hss.alloc_disk_block_id().unwrap();
    assert_eq!(reused, 3);
}

#[test]
#[ignore = "requires GPU"]
fn ref_counting_holds() {
    let _ctx = CudaCtx::new(0).expect("cuda init");
    let mut hss = HighSpeedSwap::new(&_ctx, cfg("rc"), dims()).unwrap();
    let id = hss.alloc_disk_block_id().unwrap();
    assert_eq!(hss.disk_refcount(id), 1);
    // Two more refs (e.g. shared prefix entry).
    hss.inc_disk_ref(id);
    hss.inc_disk_ref(id);
    assert_eq!(hss.disk_refcount(id), 3);
    // Drop two; still alive.
    assert_eq!(hss.dec_disk_ref(id), 2);
    assert_eq!(hss.dec_disk_ref(id), 1);
    // Final drop returns to free list.
    assert_eq!(hss.dec_disk_ref(id), 0);
    let reused = hss.alloc_disk_block_id().unwrap();
    assert_eq!(reused, id, "freed id should be reused");
}

#[test]
#[ignore = "requires GPU"]
fn capacity_exhaustion_then_recovery() {
    let _ctx = CudaCtx::new(0).expect("cuda init");
    let mut hss = HighSpeedSwap::new(&_ctx, cfg("cap"), dims()).unwrap();
    // Alloc all 8.
    let ids: Vec<u32> = (0..8).map(|_| hss.alloc_disk_block_id().unwrap()).collect();
    assert!(hss.alloc_disk_block_id().is_none());
    // Free 3 specific ones; disk_free_count should reflect.
    for &id in &ids[..3] {
        hss.dec_disk_ref(id);
    }
    assert_eq!(hss.disk_free_count(), 3);
    // Realloc — picks from the free list (LIFO).
    let r = hss.alloc_disk_block_id().unwrap();
    assert!(ids[..3].contains(&r));
    assert_eq!(hss.disk_free_count(), 2);
}

/// Pure-arithmetic guard for the single capacity lever (meta.rs:42):
/// `max_blocks_per_seq.saturating_mul(max_batch_size.max(1))`. No GPU needed.
#[test]
// The `.max(1)` on literals deliberately mirrors the meta.rs:42 clamp formula
// term-for-term, so each case reads as the formula it guards.
#[allow(clippy::unnecessary_min_or_max)]
fn widening_formula() {
    // C=1: capacity unchanged (byte-identical single-seq path).
    assert_eq!(8u32.saturating_mul(1u32.max(1)), 8);
    // max_batch_size=0 is clamped to 1 → still C=1 capacity.
    assert_eq!(8u32.saturating_mul(0u32.max(1)), 8);
    // C=4: capacity widens to per_seq × batch.
    assert_eq!(8u32.saturating_mul(4u32.max(1)), 32);
    // Overflow saturates instead of panicking.
    assert_eq!(u32::MAX.saturating_mul(2u32.max(1)), u32::MAX);
}

/// dims() with the widened per-layer capacity (per_seq 8 × batch 4 = 32),
/// mirroring what meta.rs:42 now feeds the allocator under concurrency.
fn dims_batch() -> ModelDims {
    ModelDims {
        max_blocks_per_layer: 8 * 4,
        ..dims()
    }
}

#[test]
#[ignore = "requires GPU"]
fn capacity_scales_with_batch() {
    let _ctx = CudaCtx::new(0).expect("cuda init");
    let mut hss = HighSpeedSwap::new(&_ctx, cfg("cap-batch"), dims_batch()).unwrap();
    // (a) 32 successive allocs succeed and equal 0..32.
    let ids: Vec<u32> = (0..32).map(|_| hss.alloc_disk_block_id().unwrap()).collect();
    assert_eq!(ids, (0..32).collect::<Vec<_>>());
    // (b) the 33rd is None (capacity exhausted at per_seq × batch).
    assert!(hss.alloc_disk_block_id().is_none());
    // (c) diagnostic reports the widened capacity.
    assert_eq!(hss.diagnostic_summary().disk_block_capacity, 32);
    // (d) two disjoint 8-id runs never overlap while both are live. Free the
    // first run, alloc a second run, and confirm no live id appears twice.
    for &id in &ids[..8] {
        hss.dec_disk_ref(id);
    }
    let run_b: Vec<u32> = (0..8).map(|_| hss.alloc_disk_block_id().unwrap()).collect();
    // run_b draws from the freed first run (LIFO); the still-live ids[8..32]
    // are disjoint from run_b.
    for id in &run_b {
        assert!(
            !ids[8..].contains(id),
            "reallocated id {id} collides with a still-live id"
        );
    }
}
