// SPDX-License-Identifier: AGPL-3.0-only

//! Apply-time hooks for the vocab-extending [`EmbedOverlay`](super::token_adapter)
//! — the two gated forward operations that make a PEFT `trainable_tokens`
//! adapter's new/changed embedding rows take effect at inference. Both mirror
//! the projection-delta `apply_lora` pattern in [`super::compute`]: guarded by
//! `lora_is_active()` + an active overlay, they NEVER mutate the shared
//! `embed_table` in place, so base (un-adapted) requests interleaving with
//! adapter requests stay correct.
//!
//! Kept in this sibling file (not `compute.rs`) purely to hold that file under
//! the 500-LoC cap; the call-sites there are one line each.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use super::NllbGpuModel;

impl NllbGpuModel {
    /// Override the input-embedding rows of the `rows` just-embedded tokens
    /// (`ids`, device `u32[rows]`) in `out` (`[rows,d]` bf16), for any token the
    /// active adapter changes. No-op for the base model / an A/B-only adapter.
    /// Must run AFTER the embed gather and BEFORE the sqrt(d) scale.
    pub(super) fn apply_embed_overlay(
        &self,
        ids: DevicePtr,
        out: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        if !self.lora_is_active() {
            return Ok(());
        }
        let Some(ov) = self.lora.as_ref().and_then(|l| l.overlay()) else {
            return Ok(());
        };
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.embed_overlay)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ids)
            .arg_ptr(ov.slot_map)
            .arg_ptr(ov.rows)
            .arg_ptr(out)
            .arg_u32(self.d as u32)
            .launch(self.stream())
    }

    /// Recompute the overridden vocab columns of `logits` (`[m,vocab]` bf16)
    /// from `hidden` (`[m,d]` bf16): the tied lm_head gemv/gemm used the OLD
    /// embed row for those columns. No-op for the base model / an A/B-only
    /// adapter. Must run AFTER the lm_head gemv/gemm.
    pub(super) fn apply_lmhead_overlay(
        &self,
        hidden: DevicePtr,
        logits: DevicePtr,
        m: usize,
    ) -> Result<()> {
        if !self.lora_is_active() {
            return Ok(());
        }
        let Some(ov) = self.lora.as_ref().and_then(|l| l.overlay()) else {
            return Ok(());
        };
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.lmhead_overlay)
            .grid([m as u32 * ov.n_override, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(hidden)
            .arg_ptr(ov.rows)
            .arg_ptr(ov.ids_dev)
            .arg_ptr(logits)
            .arg_u32(ov.n_override)
            .arg_u32(self.d as u32)
            .arg_u32(self.vocab as u32)
            .launch(self.stream())
    }
}
