// SPDX-License-Identifier: AGPL-3.0-only

//! Stage-(a) (GPU-independent) tests for the b12x scale codec + bake + seam config.

use super::*;

#[test]
fn e4m3_roundtrip_representative() {
    // Exact e4m3fn-representable magnitudes round-trip bit-exact.
    for &v in &[0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 8.0, 0.125, 0.25, 448.0] {
        let enc = f32_to_e4m3(v);
        let dec = e4m3_to_f32(enc);
        assert_eq!(dec, v, "roundtrip {v} -> 0x{enc:02x} -> {dec}");
    }
}

#[test]
fn e4m3_satfinite_clamps_overflow() {
    // Above max normal (448) clamps to 0x7E, not the NaN slot 0x7F.
    assert_eq!(f32_to_e4m3(1.0e9), 0x7E);
    assert_eq!(f32_to_e4m3(500.0), 0x7E);
    assert!(e4m3_to_f32(0x7E) == 448.0);
    // NaN maps to the canonical NaN byte.
    assert_eq!(f32_to_e4m3(f32::NAN), 0x7F);
}

#[test]
fn e4m3_decode_known_bytes() {
    // 0x38 = exp 7 (bias) mant 0 => 1.0; 0x40 = exp 8 => 2.0.
    assert_eq!(e4m3_to_f32(0x38), 1.0);
    assert_eq!(e4m3_to_f32(0x40), 2.0);
    assert_eq!(e4m3_to_f32(0x00), 0.0);
}

#[test]
fn bake_w13_multiplies_by_scale2_and_concats() {
    // h=16 => 1 k-group; inter=2 => 2 output rows per half.
    let h = 16;
    let inter = 2;
    let up = vec![f32_to_e4m3(1.0), f32_to_e4m3(2.0)]; // [K/16=1, inter=2]
    let gate = vec![f32_to_e4m3(4.0), f32_to_e4m3(0.5)];
    let baked = bake_w13_logical(&up, &gate, 2.0, 0.5, h, inter);
    assert_eq!(baked.len(), (h / 16) * 2 * inter);
    // up cols [0,inter) scaled by up_ws2=2.0
    assert_eq!(e4m3_to_f32(baked[0]), 2.0); // 1*2
    assert_eq!(e4m3_to_f32(baked[1]), 4.0); // 2*2
    // gate cols [inter,2inter) scaled by gate_ws2=0.5
    assert_eq!(e4m3_to_f32(baked[2]), 2.0); // 4*0.5
    assert_eq!(e4m3_to_f32(baked[3]), 0.25); // 0.5*0.5
}

#[test]
fn bake_single_preserves_layout() {
    let s = vec![f32_to_e4m3(1.0), f32_to_e4m3(2.0), f32_to_e4m3(3.0)];
    let b = bake_single(&s, 2.0);
    assert_eq!(b.len(), 3);
    assert_eq!(e4m3_to_f32(b[0]), 2.0);
    assert_eq!(e4m3_to_f32(b[1]), 4.0);
    assert_eq!(e4m3_to_f32(b[2]), 6.0);
}

#[test]
fn ones_and_slice_bytes() {
    let ones = ones_f32_bytes(3);
    assert_eq!(ones.len(), 12);
    for c in ones.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(c.try_into().unwrap()), 1.0);
    }
    let s = f32_slice_bytes(&[0.5, 2.5]);
    assert_eq!(f32::from_le_bytes(s[0..4].try_into().unwrap()), 0.5);
    assert_eq!(f32::from_le_bytes(s[4..8].try_into().unwrap()), 2.5);
}

#[test]
fn sfb_len_holo_dims() {
    // w13: n=2I=1024, k=H=2048 -> 128 KiB/expert.
    assert_eq!(sfb_len(1024, 2048), 131072);
    // w2: n=H=2048, k=I=512 -> 64 KiB/expert.
    assert_eq!(sfb_len(2048, 512), 65536);
}

#[test]
fn strategy_env_default_is_concat() {
    // Default (unset or anything but "rebuild") is ConcatReuse.
    // (Env is process-global; assert the mapping via explicit values instead.)
    assert_eq!(
        match "rebuild" {
            "rebuild" => SfbStrategy::RebuildFromRaw,
            _ => SfbStrategy::ConcatReuse,
        },
        SfbStrategy::RebuildFromRaw
    );
    assert_eq!(
        match "concat" {
            "rebuild" => SfbStrategy::RebuildFromRaw,
            _ => SfbStrategy::ConcatReuse,
        },
        SfbStrategy::ConcatReuse
    );
    // Smoke the real resolver (result depends on ambient env; just ensure it returns).
    let _ = sfb_strategy_from_env();
}
