// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-independent tests: eligibility truth table + b12x buffer-size arithmetic at the
//! Holo E=256, H=2048, I=512 geometry.

use super::*;

#[test]
fn eligibility_truth_table() {
    // Streamer + flag is ALWAYS a hard error, regardless of the other flags.
    assert_eq!(
        eligibility(true, 1, false, true),
        B12xEligibility::ErrStreamer
    );
    assert_eq!(
        eligibility(true, 4, true, false),
        B12xEligibility::ErrStreamer
    );
    // EP shard disables b12x.
    assert_eq!(eligibility(false, 2, false, true), B12xEligibility::SkipEp);
    // Null/placeholder expert disables b12x.
    assert_eq!(
        eligibility(false, 1, true, true),
        B12xEligibility::SkipNullExpert
    );
    // Missing transposed tables disable b12x.
    assert_eq!(
        eligibility(false, 1, false, false),
        B12xEligibility::SkipNoTables
    );
    // All experts resident, tables present, single rank => build.
    assert_eq!(eligibility(false, 1, false, true), B12xEligibility::Build);
}

#[test]
fn eligibility_precedence_streamer_over_all() {
    // Even with EP + nulls + missing tables, the streamer error takes precedence.
    assert_eq!(
        eligibility(true, 8, true, true),
        B12xEligibility::ErrStreamer
    );
}

#[test]
fn holo_fp4_buffer_geometry() {
    // Holo-3.1-35B-A3B: E=256, H=2048, I=512, NVFP4 (2 vals/byte).
    let (e, h, inter) = (256usize, 2048usize, 512usize);
    let half_h = h / 2;
    let half_i = inter / 2;
    // w13 [E, 2I, H/2] = 256 * 1024 * 1024 = 256 MiB.
    let w13_stride = 2 * inter * half_h;
    assert_eq!(w13_stride, 1024 * 1024);
    assert_eq!(e * w13_stride, 256 * 1024 * 1024);
    // w2 [E, H, I/2] = 256 * 2048 * 256 = 128 MiB.
    let w2_stride = h * half_i;
    assert_eq!(w2_stride, 2048 * 256);
    assert_eq!(e * w2_stride, 128 * 1024 * 1024);
    // Per-expert UP block size = GATE block size = I * H/2 = 512 KiB; UP first.
    let up_bytes = inter * half_h;
    assert_eq!(up_bytes, 512 * 1024);
    // GATE starts exactly one UP block into the expert's w13 slab.
    assert_eq!(w13_stride, up_bytes * 2);
}

#[test]
fn holo_sf_buffer_geometry() {
    // Swizzled SFB: w13 128 KiB/expert (32 MiB total), w2 64 KiB/expert (16 MiB).
    let (e, h, inter) = (256usize, 2048usize, 512usize);
    assert_eq!(b12x_scales::sfb_len(2 * inter, h), 128 * 1024);
    assert_eq!(b12x_scales::sfb_len(h, inter), 64 * 1024);
    assert_eq!(e * b12x_scales::sfb_len(2 * inter, h), 32 * 1024 * 1024);
    assert_eq!(e * b12x_scales::sfb_len(h, inter), 16 * 1024 * 1024);
}
