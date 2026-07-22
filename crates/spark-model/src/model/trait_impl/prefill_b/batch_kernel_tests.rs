// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure-data eligibility predicate in
//! `batch_kernel.rs`. Kept in a sibling file to keep `batch_kernel.rs`
//! itself under the 500-LoC file-size-cap.

use spark_runtime::prefix_cache::PrefixMatch;

use super::batch_kernel::{cache_batch_matches_compatible, check_kernel_batched_eligible};

/// (chunk_len, chunk_start, is_last_chunk)
fn s(chunk_len: usize, chunk_start: usize, is_last: bool) -> (usize, usize, bool) {
    (chunk_len, chunk_start, is_last)
}

// Scratch capacity large enough that the #110 footprint check never trips for
// the structural-eligibility tests below (those assert the chunk_len/start/
// is_last/arena/model gates, not the scratch fit). 8 MiB ≫ any footprint here.
const BIG_SCRATCH: usize = 8 * 1024 * 1024;
const TOP_K: usize = 8;
const MROPE: bool = false;

fn cache_match(tokens: usize) -> PrefixMatch {
    PrefixMatch {
        matched_blocks: vec![7; tokens / 16],
        matched_disk_block_ids: Vec::new(),
        matched_tokens: tokens,
        ssm_snapshot: None,
        ssm_snapshot_tokens: 0,
        ssm_snapshot_tier_key: None,
        ssm_snapshot_tier_tokens: 0,
    }
}

#[test]
fn cache_batch_accepts_equal_partial_hits() {
    assert!(cache_batch_matches_compatible(
        &[cache_match(48), cache_match(48)],
        8192,
    ));
}

#[test]
fn cache_batch_rejects_mixed_hit_depths() {
    assert!(!cache_batch_matches_compatible(
        &[cache_match(0), cache_match(48)],
        8192,
    ));
}

#[test]
fn cache_batch_rejects_snapshot_or_disk_restore() {
    let mut snapshot = cache_match(48);
    snapshot.ssm_snapshot = Some(3);
    snapshot.ssm_snapshot_tokens = 48;
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), snapshot],
        8192,
    ));

    let mut disk = cache_match(48);
    disk.matched_disk_block_ids = vec![9; 3];
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), disk],
        8192,
    ));
}

#[test]
fn cache_batch_rejects_full_chunk_hit() {
    assert!(!cache_batch_matches_compatible(
        &[cache_match(8192), cache_match(8192)],
        8192,
    ));
}

#[test]
fn rejects_under_two_streams() {
    assert!(!check_kernel_batched_eligible(
        std::iter::empty(),
        0,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false)],
        1,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_chunk_zero() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, false)],
        2,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn accepts_chunk_zero_when_explicitly_allowed() {
    assert!(check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, false)],
        2,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false, // varlen
    ));
}

#[test]
fn accepts_uniform_paged_n_2() {
    assert!(check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_mismatched_chunk_len() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(2048, 4096, false)],
        2,
        16384,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_mismatched_chunk_start() {
    // Scheduler stream-desync case observed 2026-05-11:
    // stream 0 at chunk_start=12288, stream 1 at chunk_start=4096.
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 12288, false), s(4096, 4096, false)],
        2,
        16384,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_mismatched_is_last() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, true)],
        2,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_arena_overflow() {
    // N=2 × 4096 = 8192 > 4100 arena → reject.
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        4100,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_mla_model() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        8192,
        "mistral",
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_large_head_dim() {
    // Gemma-4 long-attention head_dim=512 → reject.
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        8192,
        "gemma4",
        512,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn accepts_n_4_uniform() {
    assert!(check_kernel_batched_eligible(
        vec![s(2048, 2048, false); 4],
        4,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn accepts_varlen_batch_when_packed_footprint_fits() {
    // Regression: the old preflight charged all four requests at 4,782 tokens
    // (19,128 tokens) instead of their packed cu_seqlens total (13,649). The
    // standard 16,388-token arena is deliberately provisioned for this
    // workload, so the oversized estimate silently serialized realistic
    // agentic/RAG traffic.
    let streams = [
        s(2051, 0, true),
        s(2953, 0, true),
        s(3863, 0, true),
        s(4782, 0, true),
    ];
    let arena: usize = 16_388;
    let scratch = spark_runtime::buffers::q12_batched_scratch_bytes(
        spark_runtime::buffers::Q12_SIZING_STREAMS,
        arena.div_ceil(spark_runtime::buffers::Q12_SIZING_STREAMS),
        TOP_K,
        MROPE,
    );
    assert!(check_kernel_batched_eligible(
        streams, 4, arena, "laguna", 128, scratch, TOP_K, MROPE, true, true,
    ));
}

#[test]
fn rejects_scratch_footprint_overflow() {
    // #110 regression lock: the staging footprint must fit in scratch even
    // when the token-arena check passes. The deterministic crash repro was
    // n=4, chunk_len=935, top_k=8, MRoPE → 374_352 B footprint vs a 348_840 B
    // scratch. With that exact (too-small) scratch the batch is INELIGIBLE
    // (routes to per-stream from clean state, no mid-Phase-A bail), but with
    // the #110 enlarged scratch sizing it becomes eligible again.
    let streams = [s(935, 4096, false); 4];
    let arena = 4096; // 4×935 = 3740 ≤ 4096 → arena check passes
    let too_small = 348_840;
    let enlarged = spark_runtime::buffers::q12_batched_scratch_bytes(4, 935, 8, true);
    assert!(
        !check_kernel_batched_eligible(
            streams.iter().copied(),
            4,
            arena,
            "qwen3_next",
            256,
            too_small,
            8,
            true,
            false,
            false, // varlen
        ),
        "footprint must NOT fit in the old 348_840 B scratch"
    );
    assert!(
        check_kernel_batched_eligible(
            streams.iter().copied(),
            4,
            arena,
            "qwen3_next",
            256,
            enlarged,
            8,
            true,
            false,
            false, // varlen
        ),
        "footprint must fit once scratch is sized to it"
    );
}
