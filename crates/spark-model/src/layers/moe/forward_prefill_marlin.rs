// SPDX-License-Identifier: AGPL-3.0-only

//! Small-batch grouped Marlin MoE dispatch for Qwen3.6-35B NVFP4.

use super::*;

impl MoeLayer {
    pub fn try_marlin_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !(4..=8).contains(&num_tokens)
            || self.marlin.is_none()
            || self.lora.is_some()
            || ctx.comm.is_some_and(|comm| comm.world_size() > 1)
        {
            return Ok(false);
        }
        self.forward_prefill(input, num_tokens, ctx, stream)?;
        Ok(true)
    }

    /// Returns true after running both routed Marlin GEMMs and the final
    /// routed reduction/shared-expert blend.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_marlin_prefill(
        &self,
        input: DevicePtr,
        indices: DevicePtr,
        weights: DevicePtr,
        output: DevicePtr,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(marlin) = self.marlin.as_ref() else {
            return Ok(false);
        };
        if !(4..=8).contains(&n)
            || self.pre_expert_norm.is_some()
            || self.lora.is_some()
            || ctx.config.shared_expert_intermediate_size == 0
            || ctx.comm.is_some_and(|comm| comm.world_size() > 1)
        {
            return Ok(false);
        }

        // Worst case is 64 routes selecting 64 distinct experts: 64 padded
        // blocks x 8 rows = 512 sorted ids and 64 expert ids.
        let metadata = ctx.buffers.gate_logits();
        let sorted = metadata;
        let experts = metadata.offset(512 * 4);
        let padded = experts.offset(64 * 4);
        ops::marlin_moe::align(indices, sorted, experts, padded, n, stream)?;

        let w13_out = marlin.w13_out;
        ops::marlin_moe::gemm(
            input,
            marlin.w13,
            w13_out,
            marlin.reduce_tmp,
            marlin.w13_scales,
            marlin.w13_global,
            sorted,
            experts,
            padded,
            weights,
            8,
            false,
            n,
            1024,
            2048,
            marlin.workspace,
            stream,
        )?;

        let routes = n * 8;
        let activation = ctx.buffers.expert_gate_out();
        ops::marlin_moe::silu_mul(w13_out, activation, routes, 512, stream)?;
        let expert_out = ctx.buffers.expert_down_out();
        ops::marlin_moe::gemm(
            activation,
            marlin.w2,
            expert_out,
            marlin.reduce_tmp,
            marlin.w2_scales,
            marlin.w2_global,
            sorted,
            experts,
            padded,
            weights,
            1,
            false,
            routes,
            2048,
            512,
            marlin.workspace,
            stream,
        )?;

        // A shared expert scheduled on `prefill_stream` must join the main
        // capture before the fused blend consumes its output. Besides being
        // the data dependency, this closes CUDA's fork/join graph capture.
        if !ctx.profile
            && std::env::var("ATLAS_MOE_MARLIN_SHARED_OVERLAP")
                .ok()
                .as_deref()
                != Some("0")
        {
            ctx.gpu.stream_wait_event(stream, self.event_b)?;
        }

        ops::moe_weighted_sum_blend_prefill(
            ctx.gpu,
            self.moe_weighted_sum_blend_token_major,
            output,
            expert_out,
            weights,
            ctx.buffers.attn_output(),
            input,
            self.weights.shared_expert_gate.weight,
            2048,
            8,
            2048,
            n,
            stream,
        )?;
        tracing::trace!("ATLAS_MOE_MARLIN: dispatched N={n} through two grouped GEMMs");
        Ok(true)
    }
}
