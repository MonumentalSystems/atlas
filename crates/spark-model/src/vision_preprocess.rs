// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side image preprocessing for Qwen3-VL vision inputs.
//!
//! Decodes base64 JPEG/PNG images, resizes to a grid snapped to
//! `patch_size × spatial_merge_size`, normalizes with ImageNet stats,
//! and produces a flat `f32` tensor ready for the GPU vision encoder.

use anyhow::{Context, Result};
use atlas_core::config::VisionConfig;
use image::{DynamicImage, ImageFormat};

/// SigLIP normalization — matches HF's Qwen2VLImageProcessor
/// (`image_mean = image_std = (0.5, 0.5, 0.5)` → pixels mapped to [-1, 1]).
const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Maximum allowed image dimension in pixels (longer side).
const MAX_DIM: u32 = 1280;

/// Decode a base64 data URI or raw base64 string into a `DynamicImage`.
fn decode_image(data_uri: &str) -> Result<DynamicImage> {
    // Strip optional "data:image/<fmt>;base64," prefix.
    let b64 = if let Some(pos) = data_uri.find(",base64,") {
        &data_uri[pos + 8..]
    } else if data_uri.starts_with("data:") {
        // "data:image/jpeg;base64,..."
        data_uri
            .find(',')
            .map(|p| &data_uri[p + 1..])
            .unwrap_or(data_uri)
    } else {
        data_uri
    };

    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64.trim())
        .context("base64 decode failed")?;

    // Probe format from magic bytes.
    let fmt = image::guess_format(&bytes).unwrap_or(ImageFormat::Jpeg);
    image::load_from_memory_with_format(&bytes, fmt).context("image decode failed")
}

/// Compute the target (H, W) so that:
/// - Neither side exceeds `MAX_DIM`.
/// - Both sides are multiples of `grid_unit = patch_size × spatial_merge_size`.
/// - Aspect ratio is preserved (rounded to nearest grid_unit).
fn target_size_with_max_pixels(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    let dim_scale = (MAX_DIM as f32) / (orig_h.max(orig_w) as f32);
    let pixel_scale = max_pixels
        .filter(|&p| p > 0)
        .map(|p| ((p as f32) / ((orig_h as f32) * (orig_w as f32))).sqrt())
        .unwrap_or(1.0);
    let scale = dim_scale.min(pixel_scale).min(1.0); // never upscale
    let gu = grid_unit as f32;
    let sh = orig_h as f32 * scale;
    let sw = orig_w as f32 * scale;
    let mut target_h = ((sh / gu).round() as u32).max(1) * grid_unit;
    let mut target_w = ((sw / gu).round() as u32).max(1) * grid_unit;
    // Round-to-NEAREST grid_unit can push the area OVER max_pixels even when the
    // source was under it (e.g. 400 → 416: 416×640 = 266240 > 262144). The patch
    // buffers (buf_rope/buf_pos/…) are sized for max_pixels/patch² patches, so an
    // over-cap target overflows them → CUDA_ERROR_ILLEGAL_ADDRESS (700) in the
    // vision prefill on NON-SQUARE images. Mirror Qwen smart_resize: if the
    // rounded area exceeds the cap, FLOOR each side to grid_unit instead (the
    // floored area is ≤ the scaled source area ≤ max_pixels). Both sides stay
    // multiples of grid_unit and ≥ 1.
    if let Some(mp) = max_pixels.filter(|&p| p > 0) {
        if (target_h as usize) * (target_w as usize) > mp {
            target_h = ((sh / gu).floor() as u32).max(1) * grid_unit;
            target_w = ((sw / gu).floor() as u32).max(1) * grid_unit;
        }
    }
    (target_h, target_w)
}

/// Preprocess a single base64-encoded image for the Qwen3-VL encoder.
///
/// Returns:
/// - `pixels`: flat `f32` tensor shaped `[P, C × T × H_p × W_p]` where:
///   - `P = (H/patch_size) × (W/patch_size)` — number of patches
///   - `C = 3` channels, `T = temporal_patch_size` (image duplicated), `H_p = W_p = patch_size`
/// - `grid_h`: number of patches along height
/// - `grid_w`: number of patches along width
pub fn preprocess_image(data_uri: &str, vcfg: &VisionConfig) -> Result<(Vec<f32>, usize, usize)> {
    preprocess_image_with_max_pixels(data_uri, vcfg, None)
}

/// Preprocess an image with an optional max-pixels cap, matching vLLM-style
/// multimodal processor controls. `None` preserves Atlas' historical 1280px
/// long-side cap.
pub fn preprocess_image_with_max_pixels(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Result<(Vec<f32>, usize, usize)> {
    let img = decode_image(data_uri)?;
    let img = img.to_rgb8();
    let (orig_w, orig_h) = (img.width(), img.height());

    let grid_unit = (vcfg.patch_size * vcfg.spatial_merge_size) as u32;
    let (th, tw) = target_size_with_max_pixels(orig_h, orig_w, grid_unit, max_pixels);

    // Resize with CatmullRom — closest BICUBIC match in the `image` crate,
    // matching HF's `Qwen2VLImageProcessor` which uses PIL resample=3 (BICUBIC).
    let img = image::imageops::resize(&img, tw, th, image::imageops::FilterType::CatmullRom);

    let ps = vcfg.patch_size;
    let tp = vcfg.temporal_patch_size;
    let grid_h = (th as usize) / ps;
    let grid_w = (tw as usize) / ps;
    let num_patches = grid_h * grid_w;
    // Flattened patch dim: C × temporal_patch_size × patch_size × patch_size
    let patch_dim = 3 * tp * ps * ps;
    let mut pixels = vec![0.0f32; num_patches * patch_dim];

    // Build patches. The temporal dimension is handled by duplicating the image `tp` times.
    // Layout: [P, C, T, Hp, Wp] → stored as [P, C*T*Hp*Wp] in row-major order.
    for ph in 0..grid_h {
        for pw in 0..grid_w {
            let patch_idx = ph * grid_w + pw;
            for c in 0..3usize {
                for t in 0..tp {
                    for py in 0..ps {
                        for px in 0..ps {
                            let pixel_y = ph * ps + py;
                            let pixel_x = pw * ps + px;
                            let raw =
                                img.get_pixel(pixel_x as u32, pixel_y as u32)[c] as f32 / 255.0;
                            let norm = (raw - MEAN[c]) / STD[c];
                            // Offset into patch_dim: c*(T*Hp*Wp) + t*(Hp*Wp) + py*Wp + px
                            let off = c * (tp * ps * ps) + t * (ps * ps) + py * ps + px;
                            pixels[patch_idx * patch_dim + off] = norm;
                        }
                    }
                }
            }
        }
    }

    Ok((pixels, grid_h, grid_w))
}

/// Number of image pad tokens produced per image after the vision
/// encoder's spatial merger. Qwen3-VL / Qwen3.6 fold a 2×2 patch block
/// into a single token, so the embedding stream has
/// `(grid_h / sms) * (grid_w / sms)` rows — not `grid_h * grid_w`.
pub fn image_pad_count(grid_h: usize, grid_w: usize, spatial_merge_size: usize) -> usize {
    let sms = spatial_merge_size.max(1);
    (grid_h / sms) * (grid_w / sms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_size_no_upscale() {
        // Small image: grid_unit=32, no upscale needed.
        let (h, w) = target_size_with_max_pixels(100, 150, 32, None);
        assert!(h <= 1280 && w <= 1280);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn test_target_size_downscale() {
        // Large image: should be downscaled.
        let (h, w) = target_size_with_max_pixels(2000, 3000, 32, None);
        assert!(h.max(w) <= 1280);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn test_target_size_max_pixels() {
        let (h, w) = target_size_with_max_pixels(1254, 1254, 32, Some(512 * 512));
        assert_eq!((h, w), (512, 512));
    }

    #[test]
    fn test_target_size_nonsquare_never_exceeds_max_pixels() {
        // Regression: a NON-SQUARE source UNDER max_pixels whose round-to-nearest
        // overshoots the cap (640×400: round 400→416 → 416×640=266240 > 262144)
        // must NOT exceed max_pixels — else the patch buffers (sized for
        // max_pixels/patch²) overflow → CUDA 700 in the vision prefill.
        let mp = 262144usize; // 512×512, the deployed ATLAS_VISION_MAX_PIXELS
        for &(oh, ow) in &[(400u32, 640u32), (640, 400), (401, 1280), (720, 480), (333, 999)] {
            let (h, w) = target_size_with_max_pixels(oh, ow, 32, Some(mp));
            assert_eq!(h % 32, 0, "{oh}x{ow} -> {h}x{w}: h not grid-aligned");
            assert_eq!(w % 32, 0, "{oh}x{ow} -> {h}x{w}: w not grid-aligned");
            assert!(h >= 32 && w >= 32, "{oh}x{ow} -> {h}x{w}: degenerate");
            assert!(
                (h as usize) * (w as usize) <= mp,
                "{oh}x{ow} -> {h}x{w} = {} exceeds max_pixels {mp} (would OOB patch buffers)",
                (h as usize) * (w as usize)
            );
        }
    }

    #[test]
    fn test_image_pad_count_2x2_merge() {
        // Standard Qwen3-VL: 2×2 spatial merger folds a patch block
        // into one embedding token.
        assert_eq!(image_pad_count(64, 64, 2), 32 * 32);
        assert_eq!(image_pad_count(40, 80, 2), 20 * 40);
    }

    #[test]
    fn test_image_pad_count_no_merge() {
        // spatial_merge_size=1 → identity (each patch → one token).
        assert_eq!(image_pad_count(64, 64, 1), 64 * 64);
        assert_eq!(image_pad_count(8, 12, 1), 96);
    }

    #[test]
    fn test_image_pad_count_zero_sms_clamps_to_one() {
        // sms=0 is invalid; clamps to 1 so we never divide by zero.
        assert_eq!(image_pad_count(64, 64, 0), 64 * 64);
    }

    #[test]
    fn test_image_pad_count_non_divisible_floors() {
        // Integer division truncates: 65/2 = 32 (not 33).
        assert_eq!(image_pad_count(65, 64, 2), 32 * 32);
    }
}
