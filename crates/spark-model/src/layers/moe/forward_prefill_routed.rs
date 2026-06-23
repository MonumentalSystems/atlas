// SPDX-License-Identifier: AGPL-3.0-only

//! Routed grouped-GEMM phase of `MoeLayer::forward_prefill`.
//!
//! Hoisted from `forward_prefill.rs` to keep that file under the 500 LoC
//! cap. The single entry point [`MoeLayer::run_routed_grouped_gemm`]
//! mirrors the original block 1:1 — same control flow, same kernel
//! launches, same buffer wiring. Covers steps 4-6 of the prefill
//! pipeline: grid sizing, grouped gate+up GEMM, SiLU, grouped down GEMM.

use super::*;

impl MoeLayer {
    /// Routed-expert grouped-GEMM path: upper-bound grid sizing → grouped
    /// gate+up GEMM → SiLU+mul → grouped down GEMM.
    ///
    /// Writes the routed expert outputs into `ctx.buffers.expert_down_out()`.
    /// `t0` carries the running profile timer so per-step timing output
    /// matches the original inline pipeline exactly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_routed_grouped_gemm(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        num_tokens: usize,
        ne: usize,
        t0: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        macro_rules! prof_step {
            ($label:expr) => {
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    *t0 = Some(std::time::Instant::now());
                }
            };
        }

        let avg_per_expert = (num_tokens * top_k as usize).div_ceil(ne);
        // Default to the absolute worst case (one expert receives every routed
        // token) to prevent silent truncation. An opt-in load-factor cap lets
        // Holo experiments trade that safety margin for fewer empty expert
        // tiles after validating the router histogram.
        let worst_case_m_tiles = (num_tokens * top_k as usize).div_ceil(64).max(1) as u32;
        let exact_tiles = std::env::var("ATLAS_MOE_PREFILL_EXACT_TILES")
            .ok()
            .as_deref()
            == Some("1")
            && !ctx.graph_capture;
        let max_m_tiles = if exact_tiles {
            let mut offsets = vec![0u8; (ne + 1) * 4];
            ctx.gpu
                .copy_d2h_on_stream(expert_offsets, &mut offsets, stream)?;
            let mut prev = 0u32;
            let mut max_rows = 0u32;
            for raw in offsets.chunks_exact(4).skip(1) {
                let cur = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                max_rows = max_rows.max(cur.saturating_sub(prev));
                prev = cur;
            }
            max_rows.div_ceil(64).max(1).min(worst_case_m_tiles)
        } else {
            std::env::var("ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&factor| factor > 0)
                .map(|factor| {
                    let capped_rows = avg_per_expert.saturating_mul(factor);
                    worst_case_m_tiles.min(capped_rows.div_ceil(64).max(1) as u32)
                })
                .unwrap_or(worst_case_m_tiles)
        };
        super::dump::dump_expert_load(
            ctx.gpu,
            stream,
            expert_offsets,
            ne,
            num_tokens,
            avg_per_expert,
            max_m_tiles,
        );
        prof_step!("grid_setup");

        let total_expanded = n * top_k;

        // 5. Grouped gate+up GEMM — cp.async pipelined FP8-MMA K64 (transposed).
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        // EP remote experts return without writing, so the destination must be
        // zeroed before dispatch. In non-EP, `moe_sort_by_expert` produces a
        // dense token_to_perm over exactly [0, total_expanded), and grouped
        // kernels write every row that can be referenced by unpermute_reduce.
        // Skipping the memset removes ~138 MB/layer of scratch clears on Holo.
        let force_zero = std::env::var("ATLAS_MOE_PREFILL_ZERO").ok().as_deref() == Some("1");
        if ctx.comm.is_some() || force_zero {
            let gate_bytes = total_expanded as usize * inter as usize * 2;
            let up_bytes = gate_bytes;
            let down_bytes = total_expanded as usize * h as usize * 2;
            ctx.gpu
                .memset_async(expert_gate_out, 0, gate_bytes, stream)?;
            ctx.gpu.memset_async(expert_up_out, 0, up_bytes, stream)?;
            ctx.gpu
                .memset_async(ctx.buffers.expert_down_out(), 0, down_bytes, stream)?;
        }
        if max_m_tiles > 0 {
            if let Some(fp4) = self.fp4_gate_up.as_ref().filter(|_| {
                self.moe_permute_tokens_k.0 != 0
            }) {
                // ── FP4 gate_up escape-hatch (ATLAS_HOLO_MOE_GATEUP_FP4) ──
                // The CUTLASS grouped collective consumes contiguous, expert-
                // sorted A rows (unlike the FP8 fused kernel, which gathers via
                // sorted_token_ids in-kernel). So: (1) gather expert_input into
                // [total_expanded, h] sorted order, (2) copy expert_offsets D2H
                // (the grouped wrapper takes host offsets), (3) run the per-
                // expert NVFP4 collective. Outputs land in expert_gate_out /
                // expert_up_out in the SAME sorted order the FP8 path produces,
                // so silu+down+unpermute downstream are unchanged.
                let te = total_expanded as usize;
                let permuted = ctx.gpu.alloc(te * h as usize * 2)?;
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
                // expert_offsets is [ne+1] i32 on device; the grouped wrapper
                // needs it on the host. Sync so the gather + offsets are ready.
                let mut off_bytes = vec![0u8; (ne + 1) * 4];
                ctx.gpu
                    .copy_d2h_on_stream(expert_offsets, &mut off_bytes, stream)?;
                ctx.gpu.synchronize(stream)?;
                let offsets_host: Vec<i32> = off_bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let res = ops::moe_nvfp4_fused_gate_up_grouped(
                    fp4,
                    permuted,
                    expert_gate_out,
                    expert_up_out,
                    &offsets_host,
                    inter,
                    h,
                    stream,
                );
                ctx.gpu.synchronize(stream)?;
                ctx.gpu.free(permuted)?;
                res?;
            } else if let (Some(gp), Some(up)) = (&self.gate_ptrs_t, &self.up_ptrs_t) {
                // Block D #3 dispatch: M=128 path needs the env var on AND
                // the kernel actually loaded (try_kernel returns 0 on
                // models that don't ship it). max_m_tiles_m128 = ceil(...
                // /128) instead of /64; reuse the same upper bound by
                // halving (each m128 tile covers 2 m64 tiles).
                let use_m128 = self.nvfp4_gate_up_m128 && self.moe_fused_gate_up_t_k64_m128.0 != 0;
                if use_m128 {
                    let max_m_tiles_m128 = max_m_tiles.div_ceil(2).max(1);
                    ops::moe_w4a16_fused_gate_up_k64_m128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64_m128,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles_m128,
                        stream,
                    )?;
                } else {
                    ops::moe_w4a16_fused_gate_up_k64_n128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles,
                        stream,
                    )?;
                }
            } else {
                let (gp, up) = (&self.gate_ptrs, &self.up_ptrs);
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_input,
                    gp.packed_ptrs,
                    gp.scale_ptrs,
                    gp.scale2_vals,
                    expert_gate_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_input,
                    up.packed_ptrs,
                    up.scale_ptrs,
                    up.scale2_vals,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_gate_up");

        // 6. Activation+mul for routed experts + grouped down GEMM (K64 pipelined).
        let expert_down_out = ctx.buffers.expert_down_out();
        if max_m_tiles > 0 {
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            if let Some(dp) = &self.down_ptrs_t {
                let fp8_down = std::env::var("ATLAS_MOE_PREFILL_FP8_DOWN").ok().as_deref()
                    == Some("1")
                    && self.moe_fp8_grouped_gemm_t.0 != 0
                    && self.bf16_to_fp8_k.0 != 0;
                if fp8_down {
                    ops::bf16_to_fp8(
                        ctx.gpu,
                        self.bf16_to_fp8_k,
                        expert_gate_out,
                        expert_up_out,
                        total_expanded * inter,
                        stream,
                    )?;
                    ops::moe_fp8_grouped_gemm_ptrtable_n128(
                        ctx.gpu,
                        self.moe_fp8_grouped_gemm_t,
                        expert_up_out,
                        dp.packed_ptrs,
                        dp.scale_ptrs,
                        dp.scale2_vals,
                        expert_down_out,
                        expert_offsets,
                        DevicePtr(0),
                        num_experts,
                        h,
                        inter,
                        max_m_tiles,
                        stream,
                    )?;
                } else {
                    ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                        ctx.gpu,
                        self.moe_grouped_gemm_t_k64,
                        expert_gate_out,
                        dp.packed_ptrs,
                        dp.scale_ptrs,
                        dp.scale2_vals,
                        expert_down_out,
                        expert_offsets,
                        DevicePtr(0),
                        num_experts,
                        h,
                        inter,
                        max_m_tiles,
                        stream,
                    )?;
                }
            } else {
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_gate_out,
                    self.down_ptrs.packed_ptrs,
                    self.down_ptrs.scale_ptrs,
                    self.down_ptrs.scale2_vals,
                    expert_down_out,
                    expert_offsets,
                    DevicePtr(0),
                    num_experts,
                    h,
                    inter,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_silu_down");

        Ok(())
    }
}
