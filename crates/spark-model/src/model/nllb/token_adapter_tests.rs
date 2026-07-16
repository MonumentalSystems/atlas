// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side unit tests for the PEFT `trainable_tokens` overlay math: the
//! override-set union, the full-replacement (delta) vs baked (base_layer) row
//! selection, the row-diff threshold, and the `PeftCfg` parse (with and without
//! `trainable_token_indices`). No GPU — the device materialisation is exercised
//! by the `#[ignore]` kernel tests.

use super::*;

/// A trained (delta) index takes the FULL replacement row `delta[k]` — PEFT
/// `index_copy` semantics — NOT `base_layer[id] + delta[k]`.
#[test]
fn trainable_index_takes_full_delta_row_not_base_plus_delta() {
    let d = 3;
    // base_layer rows 0..2; delta row 0 maps to trainable id 1.
    let base = vec![10.0, 11.0, 12.0, /*id1*/ 20.0, 21.0, 22.0, 30.0, 31.0, 32.0];
    let delta = vec![100.0, 200.0, 300.0]; // the trained replacement for id 1
    let trainable = [1u32];
    let row = effective_row(1, &trainable, &base, &delta, d);
    assert_eq!(row, &[100.0, 200.0, 300.0], "trainable id must be replaced by delta");
    // Explicitly NOT base+delta (that would be [120,221,322]).
    assert_ne!(row, &[120.0, 221.0, 322.0]);
}

/// A non-trainable mismatch row (the gvn_Latn "baked into base_layer" quirk)
/// takes `base_layer[id]`, unchanged.
#[test]
fn non_trainable_mismatch_takes_base_layer_row() {
    let d = 2;
    let base = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let delta = vec![9.0, 9.0]; // delta position 0 -> trainable id 2
    let trainable = [2u32];
    // id 1 is a baked mismatch (not trainable) -> base_layer[1].
    assert_eq!(effective_row(1, &trainable, &base, &delta, d), &[3.0, 4.0]);
    // id 2 is trainable -> delta[0].
    assert_eq!(effective_row(2, &trainable, &base, &delta, d), &[9.0, 9.0]);
}

/// `override_source`: `Some(k)` for a trainable id at position k, `None` else.
/// An id that is BOTH trainable and a mismatch resolves to delta (delta wins).
#[test]
fn override_source_delta_wins_for_trainable() {
    let trainable = [256205u32, 42u32];
    assert_eq!(override_source(256205, &trainable), Some(0));
    assert_eq!(override_source(42, &trainable), Some(1));
    assert_eq!(override_source(256204, &trainable), None);
}

/// The override set is the union of differing rows and trainable ids, ascending
/// and deduped; a matching (non-differing) row is omitted.
#[test]
fn override_set_is_sorted_deduped_union() {
    // rows 1 and 4 differ; trainable = {4, 2} (4 also differs -> dedup).
    let row_diff = [false, true, false, false, true];
    let trainable = [4u32, 2u32];
    assert_eq!(build_override_set(&row_diff, &trainable), vec![1, 2, 4]);
}

/// The Kuku shapes: vanilla base -> {256204 baked, 256205 delta} = 2 overrides;
/// merged base (256204 already correct) -> {256205} = 1 override.
#[test]
fn kuku_override_counts_match_study() {
    let vocab = 8; // stand-in; ids scaled down to 6 (baked) and 7 (delta)
    let trainable = [7u32];
    // vanilla: both the new language row (6) and the lexeme row (7) differ.
    let mut vanilla = vec![false; vocab];
    vanilla[6] = true;
    vanilla[7] = true;
    assert_eq!(build_override_set(&vanilla, &trainable), vec![6, 7]);
    // merged: row 6 already matches the served base; only the lexeme (7) differs.
    let mut merged = vec![false; vocab];
    merged[7] = true;
    assert_eq!(build_override_set(&merged, &trainable), vec![7]);
}

/// Row-diff threshold: bf16 noise (≤0.05) is not flagged; a real differing row
/// (≥1.3) is.
#[test]
fn row_diff_threshold_clears_noise_flags_real_diff() {
    let d = 2;
    // row0: within noise; row1: a real 1.35 diff.
    let a = vec![1.00, 2.00, 5.00, 6.00];
    let b = vec![1.03, 2.02, 6.35, 6.01];
    let flags = row_diff_flags(&a, &b, 2, d, super::ROWDIFF_THRESH);
    assert_eq!(flags, vec![false, true]);
}

/// A standard adapter (no `token_adapter`) yields no overrides: empty diff +
/// empty trainable -> empty set (=> `overlay = None` in `build_overlay`).
#[test]
fn standard_adapter_yields_empty_override_set() {
    let row_diff = vec![false; 16];
    assert!(build_override_set(&row_diff, &[]).is_empty());
}

/// `PeftCfg` parses `trainable_token_indices` when present and defaults to empty
/// when absent (backward compatible: absent => no overlay).
#[test]
fn peft_cfg_parses_trainable_token_indices() {
    use super::super::lora::PeftCfg;
    let with = r#"{"r":32,"lora_alpha":64,"trainable_token_indices":[256205]}"#;
    let cfg: PeftCfg = serde_json::from_str(with).unwrap();
    assert_eq!(cfg.trainable_token_indices, vec![256205]);

    let without = r#"{"r":32,"lora_alpha":64}"#;
    let cfg: PeftCfg = serde_json::from_str(without).unwrap();
    assert!(cfg.trainable_token_indices.is_empty());
}

/// The Kuku v24.3 case: a resized adapter (R=256206) served on the merged base
/// (vocab=256205). The single `<lexeme>` extension index (256205) is skipped so
/// the overlay stays inside the served embedding, leaving no in-vocab trainables.
#[test]
fn clamp_drops_extension_index_on_smaller_base() {
    let (kept, skipped) = clamp_trainable_to_vocab(&[256205], 256206, 256205).unwrap();
    assert!(kept.is_empty());
    assert_eq!(skipped, 1);
}

/// In-vocab trainables are kept and their tail extension siblings dropped; the
/// kept prefix preserves positional (delta-row) order.
#[test]
fn clamp_keeps_in_vocab_and_drops_tail_extension() {
    let (kept, skipped) = clamp_trainable_to_vocab(&[100, 200, 256205], 256206, 256205).unwrap();
    assert_eq!(kept, vec![100, 200]);
    assert_eq!(skipped, 1);
}

/// A resized base (vocab >= R) keeps every index — no clamping.
#[test]
fn clamp_noop_when_base_covers_adapter() {
    let (kept, skipped) = clamp_trainable_to_vocab(&[100, 256205], 256206, 256206).unwrap();
    assert_eq!(kept, vec![100, 256205]);
    assert_eq!(skipped, 0);
}

/// A kept index after a larger kept index would mis-align delta rows once the
/// tail is dropped — rejected rather than silently corrupting the overlay.
#[test]
fn clamp_rejects_non_tail_ordered_kept_indices() {
    assert!(clamp_trainable_to_vocab(&[200, 100, 256205], 256206, 256205).is_err());
}

/// An index outside the adapter embedding itself (>= R) is a hard error.
#[test]
fn clamp_rejects_index_beyond_adapter() {
    assert!(clamp_trainable_to_vocab(&[999], 500, 500).is_err());
}
