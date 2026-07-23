// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-independent tests: eligibility truth table + b12x buffer-size arithmetic at the
//! Laguna-S-2.1 E=256, H=3072, I=1024 geometry.

use super::*;

#[test]
fn eligibility_truth_table() {
    // Streamer + flag is ALWAYS a hard error, regardless of the other flags.
    assert_eq!(eligibility(true, 1, false), B12xEligibility::ErrStreamer);
    assert_eq!(eligibility(true, 4, true), B12xEligibility::ErrStreamer);
    // EP shard disables b12x.
    assert_eq!(eligibility(false, 2, false), B12xEligibility::SkipEp);
    // Null/placeholder expert disables b12x.
    assert_eq!(eligibility(false, 1, true), B12xEligibility::SkipNullExpert);
    // All experts resident, single rank => build. b12x sources n-major original
    // scales, so there is no transposed-`_t`-tables precondition.
    assert_eq!(eligibility(false, 1, false), B12xEligibility::Build);
}

#[test]
fn eligibility_precedence_streamer_over_all() {
    // Even with EP + nulls, the streamer error takes precedence.
    assert_eq!(eligibility(true, 8, true), B12xEligibility::ErrStreamer);
}

#[test]
fn laguna_fp4_buffer_geometry() {
    // Laguna-S-2.1: E=256, H=3072, I=1024, NVFP4 (2 vals/byte).
    let (e, h, inter) = (256usize, 3072usize, 1024usize);
    let half_h = h / 2;
    let half_i = inter / 2;
    // w13 [E, 2I, H/2] = 256 * 2048 * 1536 = 768 MiB.
    let w13_stride = 2 * inter * half_h;
    assert_eq!(w13_stride, 3 * 1024 * 1024);
    assert_eq!(e * w13_stride, 768 * 1024 * 1024);
    // w2 [E, H, I/2] = 256 * 3072 * 512 = 384 MiB.
    let w2_stride = h * half_i;
    assert_eq!(w2_stride, 3072 * 512);
    assert_eq!(e * w2_stride, 384 * 1024 * 1024);
    // Per-expert UP block size = GATE block size = I * H/2 = 1.5 MiB; UP first.
    let up_bytes = inter * half_h;
    assert_eq!(up_bytes, 1024 * 1536);
    // GATE starts exactly one UP block into the expert's w13 slab.
    assert_eq!(w13_stride, up_bytes * 2);
}

#[test]
fn laguna_sf_buffer_geometry() {
    // Swizzled SFB: w13 384 KiB/expert (96 MiB total), w2 192 KiB/expert (48 MiB).
    let (e, h, inter) = (256usize, 3072usize, 1024usize);
    assert_eq!(b12x_scales::sfb_len(2 * inter, h), 384 * 1024);
    assert_eq!(b12x_scales::sfb_len(h, inter), 192 * 1024);
    assert_eq!(e * b12x_scales::sfb_len(2 * inter, h), 96 * 1024 * 1024);
    assert_eq!(e * b12x_scales::sfb_len(h, inter), 48 * 1024 * 1024);
}
