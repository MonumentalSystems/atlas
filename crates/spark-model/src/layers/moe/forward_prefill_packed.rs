// SPDX-License-Identifier: AGPL-3.0-only

//! Keep-packed GGUF Q4_K_M MoE prefill compute (Laguna-S-2.1).
//!
//! Drop-in replacement for [`MoeLayer::run_routed_grouped_gemm`] when the layer
//! holds keep-packed experts ([`MoeWeights::packed_experts`]). It reuses the
//! caller's routing (gate GEMM + top-k + `moe_sort_by_expert` → `expert_offsets`
//! / `sorted_token_ids`) and the caller's post-blend (`moe_unpermute_reduce_
//! indexed`); only the per-expert COMPUTE differs: native W4A8 `q4k_mmq` on the
//! packed Q4_K gate/up blocks (weights never dequant to BF16 — mirroring the
//! NVFP4 grouped path), and a per-expert Q6_K dequant-scratch + dense GEMM for
//! `down`. It writes `expert_down_out` in the SAME sorted layout the blend
//! expects, so the surrounding forward_prefill body is unchanged.
//!
//! First-correct version: scratch is allocated/freed per call and experts run
//! in a host-offset loop (≈num_active_experts launches/layer). A fused grouped
//! Q4_K kernel + Q6_K MMQ are the perf follow-ons.

use super::*;

impl MoeLayer {
    /// Native keep-packed Q4_K/Q6_K routed compute. Writes routed expert outputs
    /// into `ctx.buffers.expert_down_out()` in sorted (permuted) order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_routed_grouped_gemm_packed(
        &self,
        expert_input: DevicePtr,     // [n, h] BF16 (normed MoE input)
        expert_offsets: DevicePtr,   // [ne+1] i32, device — sorted cumulative counts
        sorted_token_ids: DevicePtr, // [n*top_k] i32, device
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let packed = self.weights.packed_experts.as_ref().ok_or_else(|| {
            anyhow::anyhow!("run_routed_grouped_gemm_packed: layer has no packed_experts")
        })?;
        let total_expanded = n * top_k;

        // All scratch is persistent arena — NO per-call alloc/free, so the whole
        // arm is CUDA-graph-capture-legal (decode). `permuted` aliases
        // `expert_down_out`: the gathered activations are consumed (quantized to
        // q8) before the grouped down projection overwrites it with the real
        // output, so the two uses never overlap in time. `q8` is a dedicated
        // arena buffer sized for the whole sorted [k_max*top_k, h] tile.
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let permuted = expert_down_out;
        let q8 = ctx.buffers.moe_grouped_q8();

        // 1. Gather rows into expert-contiguous (sorted) order.
        ops::moe_permute_tokens(
            ctx.gpu,
            self.moe_permute_tokens_k,
            expert_input,
            permuted,
            sorted_token_ids,
            h,
            total_expanded,
            stream,
        )?;

        // 3. Gate/up: quantize the whole sorted activation buffer ONCE, then run
        // two DEVICE-SIDE GROUPED GEMMs (one launch each, grid.z=num_experts) that
        // read per-expert row ranges from `expert_offsets` on-device and write
        // sorted [total_expanded, inter] output. Weights are the contiguous
        // per-proj expert stacks — pass expert 0's base pointer.
        ops::quantize_act_q8_1(
            ctx.gpu,
            self.q4k_quant_act_k,
            permuted,
            q8,
            total_expanded,
            h,
            stream,
        )?;
        let gate_base = packed[0].gate.weight;
        let up_base = packed[0].up.weight;
        // FUSED gate+up: ONE grouped launch; each CTA computes both projections
        // (shared empty-expert early-return + ids setup → half the scheduled CTAs
        // vs two separate launches). Numerically identical to the two-call path.
        ops::q4k_grouped_gemm_gate_up(
            ctx.gpu,
            self.q4k_grouped_gate_up_nc_k,
            self.q4k_grouped_gate_up_wc_k,
            gate_base,
            up_base,
            q8,
            expert_offsets,
            expert_gate_out,
            expert_up_out,
            inter,
            h,
            num_experts,
            total_expanded,
            stream,
        )?;
        // SiLU(gate) * up over the whole sorted buffer (one launch).
        ops::silu_mul(
            ctx.gpu,
            self.moe_silu_mul,
            expert_gate_out,
            expert_up_out,
            expert_gate_out,
            total_expanded * inter,
            stream,
        )?;

        // 4. Down: one DEVICE-SIDE GROUPED GEMM over the post-silu buffer. Q4_K_M
        // mixes the down projection Q4_K vs Q6_K PER LAYER (all experts in a layer
        // share one ggml type — the GGUF stores down_exps as a single tensor), so
        // the whole layer takes one grouped launch of the matching type. Q6_K stays
        // packed (native Q6_K MMQ) — no BF16 dequant. Activations quantize once:
        // Q4_K wants the DS4 q8_1 layout, Q6_K wants D4.
        match &packed[0].down {
            crate::weight_map::QuantWeight::PackedQ4(w4) => {
                ops::quantize_act_q8_1(
                    ctx.gpu,
                    self.q4k_quant_act_k,
                    expert_gate_out, // [total_expanded, inter] post-silu
                    q8,
                    total_expanded,
                    inter,
                    stream,
                )?;
                ops::q4k_grouped_gemm(
                    ctx.gpu,
                    self.q4k_grouped_nc_k,
                    self.q4k_grouped_wc_k,
                    w4.weight,
                    q8,
                    expert_offsets,
                    expert_down_out,
                    h,
                    inter,
                    num_experts,
                    total_expanded,
                    stream,
                )?;
            }
            crate::weight_map::QuantWeight::PackedQ6(w6) => {
                ops::quantize_act_q8_1(
                    ctx.gpu,
                    self.q4k_quant_act_d4_k,
                    expert_gate_out, // [total_expanded, inter] post-silu
                    q8,
                    total_expanded,
                    inter,
                    stream,
                )?;
                ops::q4k_grouped_gemm(
                    ctx.gpu,
                    self.q6k_grouped_nc_k,
                    self.q6k_grouped_wc_k,
                    w6.weight,
                    q8,
                    expert_offsets,
                    expert_down_out,
                    h,
                    inter,
                    num_experts,
                    total_expanded,
                    stream,
                )?;
            }
            other => anyhow::bail!("packed MoE down_proj: unexpected variant {other:?}"),
        }

        Ok(())
    }
}
