// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in compact NVFP4 routed-MoE decode.
//!
//! The work-list is built and consumed on one stream, making the path safe for
//! CUDA graph capture without host-visible routing metadata.

use super::*;

pub(super) fn compact_nvfp4_decode_enabled() -> bool {
    std::env::var("ATLAS_MOE_NVFP4_COMPACT_DECODE")
        .ok()
        .as_deref()
        == Some("1")
}

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_compact_nvfp4_decode(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !compact_nvfp4_decode_enabled()
            || !(4..=8).contains(&n)
            || n.saturating_mul(top_k) > 64
            || self.lora.is_some()
            || self.pre_expert_norm.is_some()
            || ctx.comm.is_some()
            || self.experts_scale_kind != crate::weight_map::WeightQuantFormat::Nvfp4
            || self.moe_grouped_gemm_t_k64_worklist.0 == 0
            || self.moe_fused_gate_up_t_k64_worklist.0 == 0
        {
            return Ok(false);
        }
        let (Some(gate), Some(up), Some(down)) =
            (&self.gate_ptrs_t, &self.up_ptrs_t, &self.down_ptrs_t)
        else {
            return Ok(false);
        };

        let worklist = ctx.buffers.moe_decode_worklist();
        let total_tiles = ctx.buffers.moe_decode_worklist_count();
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let down_out = ctx.buffers.expert_down_out();

        // Gate/up uses a single combined [0, 2*inter) N-space. The existing
        // builder's NULL-pointer filtering uses gate weights; NVFP4 gate and
        // up tables are local together for the single-GPU decode path.
        ops::moe_build_tile_worklist(
            ctx.gpu,
            self.moe_build_tile_worklist_k,
            expert_offsets,
            gate.packed_ptrs,
            worklist,
            total_tiles,
            num_experts,
            (2 * inter).div_ceil(128),
            64,
            stream,
        )?;
        ops::moe_w4a16_fused_gate_up_k64_worklist(
            ctx.gpu,
            self.moe_fused_gate_up_t_k64_worklist,
            expert_input,
            gate.packed_ptrs,
            gate.scale_ptrs,
            gate.scale2_vals,
            up.packed_ptrs,
            up.scale_ptrs,
            up.scale2_vals,
            gate_out,
            up_out,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            inter,
            h,
            worklist,
            total_tiles,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate_out,
            up_out,
            gate_out,
            n * top_k * inter,
            stream,
        )?;

        ops::moe_build_tile_worklist(
            ctx.gpu,
            self.moe_build_tile_worklist_k,
            expert_offsets,
            down.packed_ptrs,
            worklist,
            total_tiles,
            num_experts,
            h.div_ceil(128),
            64,
            stream,
        )?;
        ops::moe_w4a16_grouped_gemm_k64_worklist(
            ctx.gpu,
            self.moe_grouped_gemm_t_k64_worklist,
            gate_out,
            down.packed_ptrs,
            down.scale_ptrs,
            down.scale2_vals,
            down_out,
            expert_offsets,
            DevicePtr::NULL,
            num_experts,
            h,
            inter,
            worklist,
            total_tiles,
            stream,
        )?;
        Ok(true)
    }
}
