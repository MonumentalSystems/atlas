// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the FlashInfer GDN F32-output gating logic (GPU-free).

use super::want_f32_output;

#[test]
fn f32_output_requires_symbol() {
    // No F32 symbol in the loaded lib → never route F32, whatever the env says.
    assert!(!want_f32_output(false, None));
    assert!(!want_f32_output(false, Some("1")));
    assert!(!want_f32_output(false, Some("0")));
}

#[test]
fn f32_output_default_on_when_capable() {
    // Symbol present, env unset → default ON.
    assert!(want_f32_output(true, None));
    // Any value other than the literal "0" keeps it on.
    assert!(want_f32_output(true, Some("1")));
}

#[test]
fn f32_output_kill_switch() {
    // ATLAS_GDN_FI_F32_OUT=0 disables the F32 path even when capable.
    assert!(!want_f32_output(true, Some("0")));
}
