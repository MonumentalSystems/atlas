// SPDX-License-Identifier: AGPL-3.0-only

//! b12x scale assembly — logical bake (GPU-independent host math) + the swizzle-atom
//! seam (the #1 numeric risk, isolated behind one call gated on the parent P0 proof).
//!
//! Two stages, deliberately separated so a wrong atom throws away ONE function, not the
//! module:
//!  - Stage (a): host e4m3 codec + w13 scale2-bake + ones-vectors — fully testable now.
//!  - Stage (b): `swizzle_sfb` — `ConcatReuse` reuses the proven `pack_weight_sfb` atom;
//!    `RebuildFromRaw` bails until the parent supplies an FI-matching layout (only this
//!    arm changes if P0 fails).
//!
//! ## Frozen scale decisions (see 3rdparty_patches/b12x_aot/STATUS.md)
//!  - `w1_alpha = ones`, up/gate `weight_scale_2` BAKED into the w13 block scales
//!    (MANDATORY — the kernel applies one per-expert alpha to both FC1 halves AND reuses
//!    `w1_alpha` as the FC1 input-quant scale, applied quadratically; it cannot represent
//!    a per-projection scale2).
//!  - `w2_alpha = down.weight_scale_2` (lossless default, unbaked). `ATLAS_B12X_BAKE_W2=1`
//!    bakes it and sets `w2_alpha = ones` (vLLM parity / bisection only).
//!  - `fc2_input_scale = 1.0` always (the kernel does dynamic per-block FC2-input quant;
//!    a static calibrated scale saturates FP4 — vLLM PR 40082 finding).
//!
//! Geometry (Laguna-S-2.1): the bake is dimension-parametric — `h`/`inter` flow in from
//! `ModelConfig`, so H=3072/I=1024 is handled with no code change here. The frozen dims
//! only matter at the AOT kernel (see `3rdparty_patches/b12x_aot/`).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Swizzle strategy for the SFB atom (chosen by `ATLAS_B12X_SFB_STRATEGY`, default
/// `concat`). Only [`SfbStrategy::RebuildFromRaw`] changes if the parent P0 proof shows
/// Atlas's `pack_weight_sfb` atom differs from FlashInfer's `convert_sf_to_mma_layout`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SfbStrategy {
    /// Reuse the proven CUTLASS SM120 SFB atom (`pack_weight_sfb`). Expected/PASS path.
    ConcatReuse,
    /// Parent-supplied FI-matching swizzle (stubbed until P0 fails). Isolated seam.
    RebuildFromRaw,
}

/// Resolve the swizzle strategy from `ATLAS_B12X_SFB_STRATEGY` (default `concat`).
pub(crate) fn sfb_strategy_from_env() -> SfbStrategy {
    match std::env::var("ATLAS_B12X_SFB_STRATEGY").as_deref() {
        Ok("rebuild") => SfbStrategy::RebuildFromRaw,
        _ => SfbStrategy::ConcatReuse,
    }
}

/// Swizzled SFB atom size in bytes for a projection of GEMM dims `(n, k)`:
/// `round_up(n,128) * round_up(k/16,4) * 4` — identical to `build_cutlass_grouped_sfb`.
pub(crate) fn sfb_len(n: usize, k: usize) -> usize {
    n.div_ceil(128) * 128 * (k / 16).div_ceil(4) * 4
}

/// `[n]` f32 ones, little-endian device-upload bytes.
pub(crate) fn ones_f32_bytes(n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    for _ in 0..n {
        v.extend_from_slice(&1.0f32.to_le_bytes());
    }
    v
}

/// `[vals]` f32 little-endian device-upload bytes.
pub(crate) fn f32_slice_bytes(vals: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(vals.len() * 4);
    for &x in vals {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

// ── e4m3fn codec (OCP float8_e4m3: 1-4-3, bias 7, no inf, max 448) ──────────

/// Decode an e4m3fn byte to f32. Block scales are positive so the sign path is unused
/// in the bake, but the codec is full for round-trip correctness tests.
pub(crate) fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as i32;
    let mag = if exp == 0 {
        // subnormal: mant * 2^-9
        (mant as f32) * 2f32.powi(-9)
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    };
    sign * mag
}

/// Round half-to-even (matches `cvt.rn`), returning the nearest integer as f32.
fn round_ne(x: f32) -> f32 {
    let f = x.floor();
    let diff = x - f;
    if (diff - 0.5).abs() < 1e-6 {
        // exactly halfway → to even
        if (f as i64) % 2 == 0 { f } else { f + 1.0 }
    } else {
        x.round()
    }
}

/// Encode f32 to e4m3fn with round-to-nearest-even + satfinite clamp (matches
/// `cvt.rn.satfinite.e4m3x2.f32`). Out-of-range magnitudes clamp to ±448.
pub(crate) fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs().min(448.0);
    if a == 0.0 {
        return sign;
    }
    if a < 2f32.powi(-6) {
        // subnormal: val = mant * 2^-9, mant in [0,7]
        let q = a / 2f32.powi(-9);
        let m = (round_ne(q) as i32).clamp(0, 7) as u8;
        return sign | m;
    }
    let mut e = a.log2().floor() as i32;
    e = e.clamp(-6, 8);
    let frac = a / 2f32.powi(e) - 1.0;
    let mut mant = round_ne(frac * 8.0) as i32;
    let mut ef = e + 7;
    if mant == 8 {
        mant = 0;
        ef += 1;
    }
    if ef >= 15 {
        // would overflow into the NaN slot → satfinite to 448 (0x7E)
        return sign | 0x7E;
    }
    sign | ((ef as u8) << 3) | (mant as u8)
}

/// Bake the concatenated w13 logical scale for one expert. `up`/`gate` are that
/// expert's transposed `[K/16, I]` e4m3 scale bytes (`[H/16, inter]` row-major,
/// row = k-group, col = expert-output-row). The output is the concatenated
/// `[K/16, 2I]` logical (up cols `[0,inter)`, gate cols `[inter,2*inter)`), each
/// block scale multiplied by its projection's `weight_scale_2` and re-encoded. This
/// is exactly the `pack_weight_sfb(n=2I, k=H)` input orientation.
pub(crate) fn bake_w13_logical(
    up: &[u8],
    gate: &[u8],
    up_ws2: f32,
    gate_ws2: f32,
    h: usize,
    inter: usize,
) -> Vec<u8> {
    let kg = h / 16; // number of k-groups (rows)
    let mut out = vec![0u8; kg * 2 * inter];
    for row in 0..kg {
        let obase = row * 2 * inter;
        let sbase = row * inter;
        for col in 0..inter {
            let u = e4m3_to_f32(up[sbase + col]) * up_ws2;
            out[obase + col] = f32_to_e4m3(u);
            let g = e4m3_to_f32(gate[sbase + col]) * gate_ws2;
            out[obase + inter + col] = f32_to_e4m3(g);
        }
    }
    out
}

/// Bake a single-projection logical scale (used for the optional w2 bake): decode
/// `[K/16, N]` bytes, multiply by `ws2`, re-encode in place. Layout-preserving.
pub(crate) fn bake_single(scale: &[u8], ws2: f32) -> Vec<u8> {
    scale
        .iter()
        .map(|&b| f32_to_e4m3(e4m3_to_f32(b) * ws2))
        .collect()
}

/// Swizzle a logical `[K/16, N]` scale (already on device at `logical_dev`) into the
/// SFB atom at `out`. THE isolated atom seam. `ConcatReuse` reuses `pack_weight_sfb`;
/// `RebuildFromRaw` bails until the parent supplies an FI-matching layout.
///
/// The baked logical buffers this port feeds in are K-major `[K/16, N]` (the bake
/// preserves the transposed `_t`-table orientation), so `src_n_major = false`.
pub(crate) fn swizzle_sfb(
    logical_dev: u64,
    out: DevicePtr,
    n: u32,
    k: u32,
    strat: SfbStrategy,
    stream: u64,
) -> Result<()> {
    match strat {
        SfbStrategy::ConcatReuse => {
            // `src_n_major = false`: the bake keeps the transposed `[K/16, N]` layout.
            spark_runtime::cutlass::pack_weight_sfb(logical_dev, out.0, n, k, false, stream)
        }
        SfbStrategy::RebuildFromRaw => anyhow::bail!(
            "b12x SFB atom mismatch — gated on parent P0 proof; RebuildFromRaw path not yet \
             supplied. Run 3rdparty_patches/b12x_aot/proof_sfb_atom.py, then implement the \
             FI-matching swizzle here (Stage-(a) assembly/bake is untouched)."
        ),
    }
}

/// One expert's device scale pointers + per-projection `weight_scale_2` values.
pub(crate) struct ExpertScaleSrc {
    /// Transposed `[K/16, I]` e4m3 up-proj scale (device).
    pub up: u64,
    /// Transposed `[K/16, I]` e4m3 gate-proj scale (device).
    pub gate: u64,
    /// Transposed `[K/16, H]` e4m3 down-proj scale (device).
    pub down: u64,
    pub up_ws2: f32,
    pub gate_ws2: f32,
    pub down_ws2: f32,
}

/// Build the contiguous swizzled SFB tables for all experts:
/// `w13_sf` (`[E]` × `sfb_len(2I, H)`) and `w2_sf` (`[E]` × `sfb_len(H, I)`). Returns
/// the two device buffers plus the `w2_alpha` f32 host bytes (down `weight_scale_2`
/// unless `bake_w2`, else ones). Temporary per-expert logical uploads are freed here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_sf_tables(
    gpu: &dyn GpuBackend,
    experts: &[ExpertScaleSrc],
    h: usize,
    inter: usize,
    bake_w2: bool,
    strat: SfbStrategy,
    stream: u64,
) -> Result<(DevicePtr, DevicePtr, Vec<f32>)> {
    let e_count = experts.len();
    let sfb13 = sfb_len(2 * inter, h);
    let sfb2 = sfb_len(h, inter);
    let w13_sf = gpu.alloc(e_count * sfb13)?;
    let w2_sf = gpu.alloc(e_count * sfb2)?;
    let up_bytes = (h / 16) * inter; // transposed [K/16, I]
    let down_bytes = (inter / 16) * h; // transposed [K/16, H]
    let mut w2_alpha = Vec::with_capacity(e_count);

    for (e, s) in experts.iter().enumerate() {
        // ── w13: read up/gate transposed scales, bake, upload, swizzle ──
        let mut up_h = vec![0u8; up_bytes];
        let mut gate_h = vec![0u8; up_bytes];
        gpu.copy_d2h(DevicePtr(s.up), &mut up_h)?;
        gpu.copy_d2h(DevicePtr(s.gate), &mut gate_h)?;
        let baked = bake_w13_logical(&up_h, &gate_h, s.up_ws2, s.gate_ws2, h, inter);
        let tmp = gpu.alloc(baked.len())?;
        gpu.copy_h2d(&baked, tmp)?;
        swizzle_sfb(
            tmp.0,
            w13_sf.offset(e * sfb13),
            (2 * inter) as u32,
            h as u32,
            strat,
            stream,
        )?;
        gpu.synchronize(stream)?;
        gpu.free(tmp)?;

        // ── w2: default lossless (pass raw down scale straight to the swizzle) ──
        let down_src = if bake_w2 {
            let mut down_h = vec![0u8; down_bytes];
            gpu.copy_d2h(DevicePtr(s.down), &mut down_h)?;
            let baked = bake_single(&down_h, s.down_ws2);
            let tmp = gpu.alloc(baked.len())?;
            gpu.copy_h2d(&baked, tmp)?;
            w2_alpha.push(1.0);
            Some(tmp)
        } else {
            w2_alpha.push(s.down_ws2);
            None
        };
        let down_dev = down_src.map(|t| t.0).unwrap_or(s.down);
        swizzle_sfb(
            down_dev,
            w2_sf.offset(e * sfb2),
            h as u32,
            inter as u32,
            strat,
            stream,
        )?;
        gpu.synchronize(stream)?;
        if let Some(t) = down_src {
            gpu.free(t)?;
        }
    }
    Ok((w13_sf, w2_sf, w2_alpha))
}

#[cfg(test)]
#[path = "b12x_scales_tests.rs"]
mod tests;
