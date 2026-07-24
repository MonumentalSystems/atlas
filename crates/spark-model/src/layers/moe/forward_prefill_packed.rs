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
        let ne = num_experts as usize;
        let total_expanded = n * top_k;

        // Per-call scratch (first-correct; move to the arena later):
        //  - permuted:     [total_expanded, h] BF16 (expert_input gathered by
        //                  sorted_token_ids into contiguous per-expert rows)
        //  - q8:           q8_1 activations for the largest single-expert tile
        //  - down_scratch: one expert's dequant'd Q6_K down weight [h, inter] BF16
        let permuted = ctx.gpu.alloc(total_expanded as usize * h as usize * 2)?;
        let q8 = ctx.gpu.alloc(ops::q8_1_scratch_bytes(total_expanded, h))?;
        let down_scratch = ctx.gpu.alloc(h as usize * inter as usize * 2)?;

        // 1. Gather rows into expert-contiguous order.
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

        // 2. Host the per-expert row boundaries (i32 cumulative, ne+1 entries).
        let mut off_bytes = vec![0u8; (ne + 1) * 4];
        ctx.gpu
            .copy_d2h_on_stream(expert_offsets, &mut off_bytes, stream)?;
        ctx.gpu.synchronize(stream)?;
        let offs: Vec<u32> = off_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Defensive: the sort emits an exact prefix sum (offs[ne] == total_expanded,
        // monotonic). A violation ⇒ a stale/aliased offsets buffer, and per-expert
        // slicing below would run OOB (async CUDA_ERROR_ILLEGAL_ADDRESS). Fail loud.
        let last = *offs.last().unwrap_or(&0);
        let monotonic = offs.windows(2).all(|w| w[0] <= w[1]);
        if last != total_expanded || !monotonic || offs.len() != ne + 1 {
            anyhow::bail!(
                "packed MoE: bad expert_offsets (len={}, last={last}, total_expanded={total_expanded}, \
                 monotonic={monotonic}); first8={:?} last8={:?}",
                offs.len(),
                &offs[..offs.len().min(8)],
                &offs[offs.len().saturating_sub(8)..],
            );
        }

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let n_down_blocks = (h * inter) / 256; // Q6_K super-blocks in one [h,inter] down weight

        // 3. Per-expert native compute.
        for e in 0..ne {
            let start = offs[e];
            let rows = offs[e + 1].saturating_sub(start);
            if rows == 0 {
                continue;
            }
            let pe = &packed[e];
            // Row offsets into the permuted/sorted buffers (BF16 element strides).
            let in_off = permuted.offset(start as usize * h as usize * 2);
            let g_off = expert_gate_out.offset(start as usize * inter as usize * 2);
            let u_off = expert_up_out.offset(start as usize * inter as usize * 2);
            let d_off = expert_down_out.offset(start as usize * h as usize * 2);

            // Quantize this expert's activations to q8_1 (fresh, rows-sized).
            ops::quantize_act_q8_1(ctx.gpu, self.q4k_quant_act_k, in_off, q8, rows, h, stream)?;
            // Native W4A8 gate/up on the packed Q4_K blocks (weights stay packed).
            ops::q4k_mmq_gemm(
                ctx.gpu,
                self.q4k_mmq_nc_k,
                self.q4k_mmq_wc_k,
                q8,
                pe.gate.weight,
                g_off,
                rows,
                inter,
                h,
                stream,
            )?;
            ops::q4k_mmq_gemm(
                ctx.gpu,
                self.q4k_mmq_nc_k,
                self.q4k_mmq_wc_k,
                q8,
                pe.up.weight,
                u_off,
                rows,
                inter,
                h,
                stream,
            )?;
            // SiLU(gate) * up, in place into gate_out.
            ops::silu_mul(
                ctx.gpu,
                self.moe_silu_mul,
                g_off,
                u_off,
                g_off,
                rows * inter,
                stream,
            )?;
            // down: Q4_K_M mixes Q4_K and Q6_K here per layer.
            //  - Q4_K: native q4k_mmq (q8_1-quantize the post-silu [rows,inter]
            //    activation, reusing the `q8` scratch now that gate/up are done).
            //  - Q6_K: dequant this expert's weight to BF16 scratch → dense GEMM.
            match &pe.down {
                crate::weight_map::QuantWeight::PackedQ4(w4) => {
                    ops::quantize_act_q8_1(
                        ctx.gpu,
                        self.q4k_quant_act_k,
                        g_off, // [rows, inter] post-silu
                        q8,
                        rows,
                        inter,
                        stream,
                    )?;
                    ops::q4k_mmq_gemm(
                        ctx.gpu,
                        self.q4k_mmq_nc_k,
                        self.q4k_mmq_wc_k,
                        q8,
                        w4.weight,
                        d_off,
                        rows,
                        h,
                        inter,
                        stream,
                    )?;
                }
                crate::weight_map::QuantWeight::PackedQ6(w6) => {
                    ops::dequant_q6k_into(
                        ctx.gpu,
                        self.q6k_dequant_k,
                        w6.weight,
                        down_scratch,
                        n_down_blocks,
                        stream,
                    )?;
                    let down_w = crate::weight_map::DenseWeight {
                        weight: down_scratch,
                    };
                    ops::dense_gemm(
                        ctx.gpu,
                        self.dense_gemm,
                        g_off, // [rows, inter] post-silu
                        &down_w,
                        d_off, // [rows, h]
                        rows,
                        h,
                        inter,
                        stream,
                    )?;
                }
                other => anyhow::bail!("packed MoE down_proj: unexpected variant {other:?}"),
            }
        }

        ctx.gpu.free(permuted)?;
        ctx.gpu.free(q8)?;
        ctx.gpu.free(down_scratch)?;
        Ok(())
    }
}
