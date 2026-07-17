// SPDX-License-Identifier: AGPL-3.0-only

//! vLLM-compatible NVFP4 scale preprocessing for the Marlin W4A16 kernel.

use half::{bf16, f16};

use super::b12x_scales::e4m3_to_f32;

const SCALE_PERM: [usize; 64] = [
    0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3,
    11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6,
    14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
];

/// Convert checkpoint row-major scales `[N,K/16]` to the logical Marlin
/// `[K/16,N]` layout. W13 concatenates GATE then UP, matching vLLM.
pub(crate) fn transpose_w13(gate: &[u8], up: &[u8], n: usize, k_groups: usize) -> Vec<u8> {
    let mut out = vec![0; k_groups * n * 2];
    for kg in 0..k_groups {
        for col in 0..n {
            out[kg * n * 2 + col] = gate[col * k_groups + kg];
            out[kg * n * 2 + n + col] = up[col * k_groups + kg];
        }
    }
    out
}

/// Convert checkpoint row-major scales `[N,K/16]` to `[K/16,N]`.
pub(crate) fn transpose_single(src: &[u8], n: usize, k_groups: usize) -> Vec<u8> {
    let mut out = vec![0; k_groups * n];
    for kg in 0..k_groups {
        for col in 0..n {
            out[kg * n + col] = src[col * k_groups + kg];
        }
    }
    out
}

/// One power-of-two factor is shared by every expert in a projection.
pub(crate) fn combined_scale_factor(scales: &[Vec<u8>]) -> f32 {
    let max_scale = scales
        .iter()
        .flat_map(|s| s.iter())
        .map(|&x| bf16::from_f32(e4m3_to_f32(x)).to_f32())
        .filter(|x| *x > 0.0)
        .fold(0.0f32, f32::max);
    if max_scale > 0.0 && max_scale < 448.0 {
        2.0f32.powf((448.0 / max_scale).log2().floor()).max(1.0)
    } else {
        1.0
    }
}

/// Apply `marlin_permute_scales` followed by vLLM's special S0E5M3 encoding.
pub(crate) fn process(logical: &[u8], scale_factor: f32) -> Vec<u8> {
    assert_eq!(logical.len() % 64, 0);
    let mut permuted = vec![0u8; logical.len()];
    for (src, dst) in logical.chunks_exact(64).zip(permuted.chunks_exact_mut(64)) {
        for (j, &source) in SCALE_PERM.iter().enumerate() {
            dst[j] = src[source];
        }
    }

    let mut ordered = vec![0u8; logical.len()];
    for (src, dst) in permuted.chunks_exact(4).zip(ordered.chunks_exact_mut(4)) {
        dst.copy_from_slice(&[src[0], src[2], src[1], src[3]]);
    }

    ordered
        .into_iter()
        .map(|byte| {
            let bf = bf16::from_f32(e4m3_to_f32(byte)).to_f32();
            let mut value = f16::from_f32(bf);
            if scale_factor > 1.0 {
                value = f16::from_f32(value.to_f32() * scale_factor);
            }
            value = f16::from_f32(value.to_f32() * 128.0);
            if value.to_f32() < 2.0 {
                0
            } else {
                (value.to_bits() >> 7) as u8
            }
        })
        .collect()
}

/// Marlin BF16 global-scale bias, compensated for the shared scale factor.
pub(crate) fn process_global(scale: f32, scale_factor: f32) -> f32 {
    scale * 2.0f32.powi(119) / scale_factor
}

#[cfg(test)]
#[path = "marlin_scales_tests.rs"]
mod tests;
