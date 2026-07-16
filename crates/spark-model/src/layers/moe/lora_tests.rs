// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the `delta` scratch sizing (`lora_delta_cols`) — the
//! correctness-critical column count that used to over-allocate 4-8x by taking
//! `max(all n_out, all k_in)`. `delta` has exactly two consumers (router expand
//! = num_experts, decode down x-recompute = moe_inter); gate/up's `k_in=hidden`
//! must NOT size it. GPU-free (pure function).

use super::lora_delta_cols;

// Holo-3.1-35B-A3B shapes: hidden=2048, moe_inter=512, num_experts=256.
const NUM_EXPERTS: u32 = 256;
const MOE_INTER: u32 = 512;
const HIDDEN: u32 = 2048;

#[test]
fn full_adapter_takes_max_of_router_and_down() {
    // down+gate+up+router: router n_out=num_experts, down k_in=moe_inter.
    // max(256, 512) = 512 — NOT hidden (2048), which gate/up's k_in would give.
    assert_eq!(
        lora_delta_cols(Some(NUM_EXPERTS), Some(MOE_INTER)),
        MOE_INTER as usize
    );
}

#[test]
fn router_only_is_num_experts() {
    // No Down pair -> only the router expand consumes delta.
    assert_eq!(lora_delta_cols(Some(NUM_EXPERTS), None), NUM_EXPERTS as usize);
}

#[test]
fn gateup_only_no_router_is_unit() {
    // gate/up-only with no router: delta has no real consumer -> 1 (never 0).
    assert_eq!(lora_delta_cols(None, None), 1);
}

#[test]
fn down_only_is_moe_inter() {
    // down-only (no router): decode x-recompute needs moe_inter cols.
    assert_eq!(lora_delta_cols(None, Some(MOE_INTER)), MOE_INTER as usize);
}

#[test]
fn router_wins_when_experts_exceed_inter() {
    // A model with num_experts > moe_inter routes the max the other way.
    assert_eq!(lora_delta_cols(Some(1024), Some(MOE_INTER)), 1024);
}

#[test]
fn gate_up_k_in_hidden_never_inflates() {
    // Regression: the old formula was max(n_out, k_in) over ALL pairs, so
    // gate/up k_in=hidden=2048 blew delta to 2048 (4x). The pure fn only sees
    // router n_out and down k_in, so hidden can never enter.
    let sized = lora_delta_cols(Some(NUM_EXPERTS), Some(MOE_INTER));
    assert!(
        sized < HIDDEN as usize,
        "delta_cols {sized} must not be inflated to hidden {HIDDEN}"
    );
}
