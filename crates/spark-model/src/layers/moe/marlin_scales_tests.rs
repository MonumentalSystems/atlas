// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn transpose_w13_is_gate_then_up() {
    let gate = [0, 1, 2, 3, 4, 5];
    let up = [10, 11, 12, 13, 14, 15];
    assert_eq!(
        transpose_w13(&gate, &up, 3, 2),
        [0, 2, 4, 10, 12, 14, 1, 3, 5, 11, 13, 15]
    );
}

#[test]
fn scale_factor_is_shared_power_of_two() {
    assert_eq!(combined_scale_factor(&[vec![0x38; 64]]), 256.0);
    assert_eq!(combined_scale_factor(&[vec![0x7e; 64]]), 1.0);
}

#[test]
fn global_scale_uses_bf16_marlin_bias() {
    assert_eq!(process_global(2.0f32.powi(-119), 1.0), 1.0);
    assert_eq!(process_global(2.0f32.powi(-118), 2.0), 1.0);
}

#[test]
fn processing_matches_vllm_marlin_reference() {
    let raw = [0x20, 0x28, 0x30, 0x38, 0x40, 0x48, 0x50, 0x58].repeat(8);
    let factor = combined_scale_factor(std::slice::from_ref(&raw));
    assert_eq!(factor, 16.0);
    assert_eq!(
        process(&raw, factor),
        [
            184, 184, 184, 184, 184, 184, 184, 184, 192, 192, 192, 192, 192, 192, 192, 192, 200,
            200, 200, 200, 200, 200, 200, 200, 208, 208, 208, 208, 208, 208, 208, 208, 216, 216,
            216, 216, 216, 216, 216, 216, 224, 224, 224, 224, 224, 224, 224, 224, 232, 232, 232,
            232, 232, 232, 232, 232, 240, 240, 240, 240, 240, 240, 240, 240,
        ]
    );
}
