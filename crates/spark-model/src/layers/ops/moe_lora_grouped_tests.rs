// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the pure per-expert table packer (`pack_expert_tables`) — the
//! correctness-critical layout that feeds the device `moe_lora_grouped_down`
//! fold. No GPU: verifies dense placement, unadapted `0` sentinels, table
//! length, and the router-only (empty) case.

use super::{ExpertTables, gather_bgmv_grids, gather_row_token, pack_expert_tables};

#[test]
fn empty_entries_none() {
    // A router-only adapter installs no expert route.
    assert!(pack_expert_tables(&[]).is_none());
}

#[test]
fn single_expert_zero_padded_prefix() {
    // Expert id 3 adapted; ids 0..3 are unadapted -> 0/0.0 sentinels, len = 4.
    let t = pack_expert_tables(&[(3u16, 0xAAAA, 0xBBBB, 2.0)]).unwrap();
    assert_eq!(
        t,
        ExpertTables {
            a: vec![0, 0, 0, 0xAAAA],
            b: vec![0, 0, 0, 0xBBBB],
            scale: vec![0.0, 0.0, 0.0, 2.0],
            n_experts: 4,
        }
    );
}

#[test]
fn sparse_experts_dense_table() {
    // Experts 0 and 2 adapted, 1 is a gap -> zero at index 1, len = 3.
    let t = pack_expert_tables(&[(0u16, 0x10, 0x20, 0.5), (2u16, 0x30, 0x40, 0.25)]).unwrap();
    assert_eq!(t.a, vec![0x10, 0, 0x30]);
    assert_eq!(t.b, vec![0x20, 0, 0x40]);
    assert_eq!(t.scale, vec![0.5, 0.0, 0.25]);
    assert_eq!(t.n_experts, 3);
}

#[test]
fn entry_order_independent() {
    // Descending input order lands the same dense table.
    let t = pack_expert_tables(&[(2u16, 0x30, 0x40, 0.25), (0u16, 0x10, 0x20, 0.5)]).unwrap();
    assert_eq!(t.a, vec![0x10, 0, 0x30]);
    assert_eq!(t.scale, vec![0.5, 0.0, 0.25]);
}

#[test]
fn table_length_is_max_id_plus_one() {
    // n_experts is the table length (max id + 1), never the layer's full count.
    let t = pack_expert_tables(&[(7u16, 1, 2, 1.0)]).unwrap();
    assert_eq!(t.n_experts, 8);
    assert_eq!(t.a.len(), 8);
    assert_eq!(t.a[7], 1);
    assert_eq!(t.a[..7], [0u64; 7]);
}

// ── SOLID Incr-4 decode gather-fold planner ──────────────────────────────────

#[test]
fn gather_grids_single_token_decode() {
    // Qwen3.6-A3B decode: top_k=8, max_rank=16, inter=768, hidden=4096.
    // n_slots = 1*top_k = 8. Shrink outputs max_rank, expand outputs hidden.
    let (shrink, expand) = gather_bgmv_grids(16, 4096, 8);
    assert_eq!(shrink, [4, 8, 1]); // ceil(16/4)=4 rank groups, 8 flat rows
    assert_eq!(expand, [1024, 8, 1]); // ceil(4096/4)=1024 hidden groups
}

#[test]
fn gather_grids_verify_flat_rows() {
    // K=3 verify: n_slots = 3*top_k. Grid.y scales with flat (token,slot) rows;
    // grid.x is unchanged (per-output, independent of row count).
    let (shrink, expand) = gather_bgmv_grids(32, 4096, 24);
    assert_eq!(shrink, [8, 24, 1]);
    assert_eq!(expand, [1024, 24, 1]);
}

#[test]
fn gather_grids_rank_not_multiple_of_four_rounds_up() {
    // A padded rank of 6 still covers all outputs (ceil, not floor).
    let (shrink, _) = gather_bgmv_grids(6, 512, 8);
    assert_eq!(shrink[0], 2); // ceil(6/4) = 2
}

#[test]
fn row_token_decomposition_matches_kernel() {
    // token = row / top_k, mirroring the kernel's per-token row_adapter gather.
    assert_eq!(gather_row_token(0, 8), 0);
    assert_eq!(gather_row_token(7, 8), 0); // last slot of token 0
    assert_eq!(gather_row_token(8, 8), 1); // first slot of token 1
    assert_eq!(gather_row_token(23, 8), 2); // K=3 verify, token 2
}
