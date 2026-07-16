// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure MoE-LoRA per-request routing primitives (no GPU).

use super::*;
use crate::layer::MoeLoraRoute;

#[test]
fn route_off_when_no_moe_lora() {
    // Value is inert (the fold hook no-ops on self.lora == None), but must stay
    // Fold so an off run is byte-identical.
    assert_eq!(resolve_moe_lora_route(-1, -1, false), MoeLoraRoute::Fold);
    assert_eq!(resolve_moe_lora_route(3, 0, false), MoeLoraRoute::Fold);
}

#[test]
fn route_base_request_skips() {
    // adapter_slot < 0 with an adapter installed => base pays nothing.
    assert_eq!(resolve_moe_lora_route(-1, 0, true), MoeLoraRoute::Skip);
    assert_eq!(resolve_moe_lora_route(-5, 2, true), MoeLoraRoute::Skip);
}

#[test]
fn route_active_adapter_folds() {
    assert_eq!(resolve_moe_lora_route(0, 0, true), MoeLoraRoute::Fold);
    assert_eq!(resolve_moe_lora_route(2, 2, true), MoeLoraRoute::Fold);
}

#[test]
fn route_non_active_adapter_refuses() {
    // Phase-1 installs one active MoE adapter; a request for a different slot
    // cannot be served correctly => refuse loudly, never fold the wrong one.
    assert_eq!(resolve_moe_lora_route(1, 0, true), MoeLoraRoute::Refuse);
    assert_eq!(resolve_moe_lora_route(0, 3, true), MoeLoraRoute::Refuse);
}

#[test]
fn route_short_prefill_single_active_folds() {
    // Load-bearing for the lifted short-prefill refuse (forward_prefill.rs): a
    // <=64-token prefill of ONE sequence owning the active adapter dispatches to
    // forward_batched, whose per-token decode fold hooks take the NULL-map
    // single-active path and consult this route. It MUST resolve Fold, or the
    // adapter delta would be silently dropped. Sibling routes the short prefill
    // relies on for its negative controls: a base sequence Skips (byte-identical
    // base output while an adapter is resident), a non-active adapter Refuses.
    assert_eq!(resolve_moe_lora_route(0, 0, true), MoeLoraRoute::Fold);
    assert_eq!(resolve_moe_lora_route(4, 4, true), MoeLoraRoute::Fold);
    assert_eq!(resolve_moe_lora_route(-1, 0, true), MoeLoraRoute::Skip);
    assert_eq!(resolve_moe_lora_route(1, 0, true), MoeLoraRoute::Refuse);
}

#[test]
fn row_adapter_uniform_single_stream() {
    // One stream of 4 tokens on slot 2 -> all rows 2.
    let map = build_moe_row_adapter_host(&[0, 4], &[2]).unwrap();
    assert_eq!(map, vec![2, 2, 2, 2]);
}

#[test]
fn row_adapter_varlen_mixed_streams() {
    // Three streams of unequal length: base, adapter 1, adapter 0.
    // cu_seqlens = [0, 2, 5, 6] -> rows: [base,base, 1,1,1, 0].
    let map = build_moe_row_adapter_host(&[0, 2, 5, 6], &[-1, 1, 0]).unwrap();
    assert_eq!(map, vec![-1, -1, 1, 1, 1, 0]);
}

#[test]
fn row_adapter_empty_stream_span() {
    // A zero-length stream (partial-prefix-cache hit: all tokens cached) leaves
    // no rows for that stream; neighbors still align.
    let map = build_moe_row_adapter_host(&[0, 2, 2, 5], &[7, 9, -1]).unwrap();
    assert_eq!(map, vec![7, 7, -1, -1, -1]);
}

#[test]
fn row_adapter_base_sentinel_is_minus_one() {
    let map = build_moe_row_adapter_host(&[0, 3], &[-1]).unwrap();
    assert_eq!(map, vec![-1, -1, -1]);
}

#[test]
fn row_adapter_rejects_malformed() {
    // Empty / single-element boundary.
    assert!(build_moe_row_adapter_host(&[], &[]).is_none());
    assert!(build_moe_row_adapter_host(&[0], &[]).is_none());
    // adapter_slots length mismatch.
    assert!(build_moe_row_adapter_host(&[0, 2, 4], &[0]).is_none());
    // Non-zero first boundary.
    assert!(build_moe_row_adapter_host(&[1, 3], &[0]).is_none());
    // Non-monotonic boundary.
    assert!(build_moe_row_adapter_host(&[0, 4, 2], &[0, 1]).is_none());
}

// ── SOLID Incr-4 batched-decode per-row map (build_moe_row_adapter_decode) ──

#[test]
fn decode_map_off_when_no_moe_lora() {
    // No adapter installed (active = -1, has = false): every row must skip so an
    // off run folds NOTHING even though resolve() returns the inert Fold.
    let map = build_moe_row_adapter_decode(&[-1, 0, 2], 4, -1, false);
    assert_eq!(map, vec![-1, -1, -1, -1]);
}

#[test]
fn decode_map_all_base_skips() {
    // Concurrent base-only batch with an adapter resident: all rows -1, pads -1.
    let map = build_moe_row_adapter_decode(&[-1, -1], 4, 0, true);
    assert_eq!(map, vec![-1, -1, -1, -1]);
}

#[test]
fn decode_map_mixed_base_and_active() {
    // active adapter = slot 3. Rows: [active, base, active] padded to 4.
    // Active rows carry `active` (>=0 => fold); base + pad carry -1 (skip).
    let map = build_moe_row_adapter_decode(&[3, -1, 3], 4, 3, true);
    assert_eq!(map, vec![3, -1, 3, -1]);
}

#[test]
fn decode_map_refuse_row_defensively_skips() {
    // A non-active adapter row (slot 1, active 0) resolves Refuse; the batch is
    // bailed host-side before upload, but if it leaked, the map skips (-1) rather
    // than folding the wrong adapter. Active-owning rows still fold.
    let map = build_moe_row_adapter_decode(&[0, 1], 2, 0, true);
    assert_eq!(map, vec![0, -1]);
}

#[test]
fn decode_map_padding_widths() {
    // The uploaded buffer is always [padded_n]; unused rows pad with -1.
    for padded_n in [2usize, 4, 8] {
        let map = build_moe_row_adapter_decode(&[0], padded_n, 0, true);
        assert_eq!(map.len(), padded_n);
        assert_eq!(map[0], 0);
        assert!(map[1..].iter().all(|&v| v == -1));
    }
}
