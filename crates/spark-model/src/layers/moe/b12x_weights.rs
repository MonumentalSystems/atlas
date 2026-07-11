// SPDX-License-Identifier: AGPL-3.0-only

//! b12x fused-MoE repacked weights: struct, eligibility gate, and the load-time
//! NVFP4 repack (Stage-(a) logical assembly). All D2D copies — no requant (the same
//! NVFP4 bytes the grouped path already reads, just concatenated into b12x's contiguous
//! `[E, 2I, H/2]` (UP rows first, then GATE) + `[E, H, I/2]` layout). The scale-atom
//! swizzle + scale2-bake live in `b12x_scales.rs`.
//!
//! HARD CONSTRAINT: b12x enforces `num_local_experts == num_experts` — it is ONLY for the
//! FULLY-RESIDENT expert path. `--stream-experts` + this flag is a load-time hard error;
//! EP / partial-residency configs silently disable b12x (WARN) and run the grouped path.

use super::*;
use crate::layers::moe::b12x_scales::{
    self, ExpertScaleSrc, SfbStrategy, f32_slice_bytes, ones_f32_bytes, sfb_strategy_from_env,
};

/// b12x fused-MoE repacked weights (`ATLAS_HOLO_MOE_B12X`). Device buffers are process-
/// lifetime (never freed); `DevicePtr` is a bare handle so there is no `Drop` concern.
pub(crate) struct B12xMoeWeights {
    /// `[E, 2I, H/2]` u8 — UP rows `[0,I)`, then GATE rows `[I,2I)`.
    pub(crate) w13_fp4: DevicePtr,
    /// Swizzled ue4m3 SFB, `sfb_len(2I,H)` bytes/expert (scale2 baked in).
    pub(crate) w13_sf: DevicePtr,
    /// `[E, H, I/2]` u8 down-proj.
    pub(crate) w2_fp4: DevicePtr,
    /// Swizzled ue4m3 SFB, `sfb_len(H,I)` bytes/expert.
    pub(crate) w2_sf: DevicePtr,
    /// `[E]` f32 = ONES (scale2 baked into `w13_sf`).
    pub(crate) w1_alpha: DevicePtr,
    /// `[E]` f32 = down `weight_scale_2` (lossless default) or ONES (`ATLAS_B12X_BAKE_W2`).
    pub(crate) w2_alpha: DevicePtr,
    /// `[E]` f32 = ONES (`fc2_input_scale = 1.0`).
    pub(crate) fc2_gs: DevicePtr,
    /// Shim workspace token capacity — prefills beyond this fall back to grouped.
    pub(crate) max_tokens: u32,
}

/// Result of the pure eligibility check (unit-testable over bools; mirrors the
/// `gdn_flashinfer` `want_f32_output` test style).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum B12xEligibility {
    /// All experts resident — build the fused weights.
    Build,
    /// `--stream-experts` + `ATLAS_HOLO_MOE_B12X=1` — user misconfiguration, hard error.
    ErrStreamer,
    /// `ep_world_size > 1` (EP shard, `local_expert_range != (0,E)`) — skip, run grouped.
    SkipEp,
    /// Some expert is a NULL placeholder (partial residency) — skip, run grouped.
    SkipNullExpert,
    /// Transposed `_t` scale tables absent (not FAST_MOE=full) — skip, run grouped.
    SkipNoTables,
}

/// Pure eligibility decision. b12x's `num_local_experts == num_experts` enforcement makes
/// the streamed/EP/null-expert configs fundamentally incompatible: streamer is a HARD
/// error (never silently fall back on that combo), the rest silently disable b12x.
pub(crate) fn eligibility(
    has_streamer: bool,
    ep_world_size: usize,
    any_null_expert: bool,
    have_t_tables: bool,
) -> B12xEligibility {
    if has_streamer {
        return B12xEligibility::ErrStreamer;
    }
    if ep_world_size > 1 {
        return B12xEligibility::SkipEp;
    }
    if any_null_expert {
        return B12xEligibility::SkipNullExpert;
    }
    if !have_t_tables {
        return B12xEligibility::SkipNoTables;
    }
    B12xEligibility::Build
}

impl MoeLayer {
    /// Read a device `[num]` u64 pointer array into a host vec (mirrors
    /// `build_cutlass_grouped_sfb`).
    fn read_ptr_array(gpu: &dyn GpuBackend, arr: DevicePtr, num: usize) -> Result<Vec<u64>> {
        let mut raw = vec![0u8; num * 8];
        gpu.copy_d2h(arr, &mut raw)?;
        Ok(raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect())
    }

    /// Build b12x fused-MoE weights at load (behind `ATLAS_HOLO_MOE_B12X`). Sets
    /// `self.b12x = Some(..)` only when every expert is resident and the shim lib is
    /// loaded; otherwise leaves it `None` (grouped path). Hard-errors on the streamer
    /// combo.
    pub(crate) fn build_b12x_weights(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let has_streamer = self.streamer.is_some();
        let any_null = self.weights.experts.iter().any(|e| e.gate_proj.is_null());
        let have_t =
            self.gate_ptrs_t.is_some() && self.up_ptrs_t.is_some() && self.down_ptrs_t.is_some();
        match eligibility(has_streamer, config.ep_world_size, any_null, have_t) {
            B12xEligibility::ErrStreamer => anyhow::bail!(
                "ATLAS_HOLO_MOE_B12X=1 is incompatible with --stream-experts: b12x enforces \
                 num_local_experts == num_experts (fully-resident experts only). Unset one."
            ),
            B12xEligibility::SkipEp => {
                tracing::warn!(
                    "ATLAS_HOLO_MOE_B12X: ep_world_size={} > 1 — b12x disabled, grouped path runs",
                    config.ep_world_size
                );
                return Ok(());
            }
            B12xEligibility::SkipNullExpert => {
                tracing::warn!(
                    "ATLAS_HOLO_MOE_B12X: null/placeholder expert(s) present — b12x disabled, grouped"
                );
                return Ok(());
            }
            B12xEligibility::SkipNoTables => {
                tracing::warn!(
                    "ATLAS_HOLO_MOE_B12X: transposed _t tables absent (need FAST_MOE=full) — grouped"
                );
                return Ok(());
            }
            B12xEligibility::Build => {}
        }

        // Shim must be loaded to size the workspace; if absent, skip (dispatch's
        // `available()` would refuse anyway) rather than duplicate ~ the fp4 weights.
        let max_tokens = match ops::b12x_flashinfer::max_tokens() {
            Some(c) => c,
            None => {
                tracing::warn!(
                    "ATLAS_HOLO_MOE_B12X: libatlasb12x.so not loaded (max_tokens unavailable) — \
                     b12x disabled, grouped path runs"
                );
                return Ok(());
            }
        };

        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let e_count = self.weights.experts.len();
        let half_h = h / 2;
        let half_i = inter / 2;

        // ── fp4 repack: concat UP‖GATE into [E,2I,H/2] and DOWN into [E,H,I/2] ──
        let w13_stride = 2 * inter * half_h;
        let w2_stride = h * half_i;
        let w13_fp4 = gpu.alloc(e_count * w13_stride)?;
        let w2_fp4 = gpu.alloc(e_count * w2_stride)?;
        for (e, expert) in self.weights.experts.iter().enumerate() {
            let up_bytes = inter * half_h;
            gpu.copy_d2d(
                expert.up_proj.weight,
                w13_fp4.offset(e * w13_stride),
                up_bytes,
            )?;
            gpu.copy_d2d(
                expert.gate_proj.weight,
                w13_fp4.offset(e * w13_stride + up_bytes),
                up_bytes,
            )?;
            gpu.copy_d2d(
                expert.down_proj.weight,
                w2_fp4.offset(e * w2_stride),
                w2_stride,
            )?;
        }

        // ── scale sources: transposed [K/16, N] scales + per-projection scale2 ──
        let up_sf =
            Self::read_ptr_array(gpu, self.up_ptrs_t.as_ref().unwrap().scale_ptrs, e_count)?;
        let gate_sf =
            Self::read_ptr_array(gpu, self.gate_ptrs_t.as_ref().unwrap().scale_ptrs, e_count)?;
        let down_sf =
            Self::read_ptr_array(gpu, self.down_ptrs_t.as_ref().unwrap().scale_ptrs, e_count)?;
        let srcs: Vec<ExpertScaleSrc> = self
            .weights
            .experts
            .iter()
            .enumerate()
            .map(|(e, expert)| ExpertScaleSrc {
                up: up_sf[e],
                gate: gate_sf[e],
                down: down_sf[e],
                up_ws2: expert.up_proj.weight_scale_2,
                gate_ws2: expert.gate_proj.weight_scale_2,
                down_ws2: expert.down_proj.weight_scale_2,
            })
            .collect();

        let bake_w2 = std::env::var("ATLAS_B12X_BAKE_W2").as_deref() == Ok("1");
        let strat: SfbStrategy = sfb_strategy_from_env();
        let (w13_sf, w2_sf, w2_alpha_vals) =
            b12x_scales::build_sf_tables(gpu, &srcs, h, inter, bake_w2, strat, stream)?;

        // ── alpha vectors: w1_alpha=ones (scale2 baked), fc2_gs=ones, w2_alpha ──
        let w1_alpha = gpu.alloc(e_count * 4)?;
        gpu.copy_h2d(&ones_f32_bytes(e_count), w1_alpha)?;
        let fc2_gs = gpu.alloc(e_count * 4)?;
        gpu.copy_h2d(&ones_f32_bytes(e_count), fc2_gs)?;
        let w2_alpha = gpu.alloc(e_count * 4)?;
        gpu.copy_h2d(&f32_slice_bytes(&w2_alpha_vals), w2_alpha)?;
        gpu.synchronize(stream)?;

        tracing::info!(
            "ATLAS_HOLO_MOE_B12X: built fused weights for {e_count} experts (H={h} I={inter}, \
             strat={strat:?}, bake_w2={bake_w2}, max_tokens={max_tokens}); scatter is atomic-add \
             (non-deterministic vs grouped unpermute) — A/B tolerance-based"
        );
        self.b12x = Some(B12xMoeWeights {
            w13_fp4,
            w13_sf,
            w2_fp4,
            w2_sf,
            w1_alpha,
            w2_alpha,
            fc2_gs,
            max_tokens,
        });
        Ok(())
    }
}

#[cfg(test)]
#[path = "b12x_weights_tests.rs"]
mod tests;
