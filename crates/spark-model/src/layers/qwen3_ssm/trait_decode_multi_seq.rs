// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    /// Multi-sequence decode for SSM (gated-delta-net) layers.
    ///
    /// The SSM mixer (conv1d + GDN recurrence + in/out projections) carries
    /// independent per-sequence recurrent state, so it runs in a per-seq loop
    /// using the SAME single-token kernels as `decode()` (proven correct). The
    /// MoE sublayer is stateless and shared across sequences, so it is hoisted
    /// OUT of the loop and run ONCE as a batched grouped-GEMM over all N
    /// tokens — the same `forward_prefill` path the prefill scheduler and the
    /// attention layers' multi-seq path already use.
    ///
    /// This supersedes the earlier "delegate every sequence to the full
    /// single-token `decode()`" fallback, which ran N separate single-token
    /// MoE forwards (N × top_k expert GEMVs + N per-token all_reduces under
    /// EP). Phase B collapses those to one grouped gate+up+down GEMM and one
    /// batched all_reduce.
    ///
    /// Buffer safety (the old bug #6): each per-seq mixer writes its MoE input
    /// to `norm_output[i]` — a distinct per-seq offset. `ssm_forward` never
    /// touches `norm_output` (verified: 0 references) and its returned
    /// `ssm_out` (in `moe_output[0]`) is consumed by the same iteration's
    /// `residual_add_rms_norm` before the next iteration runs, so nothing
    /// needs to survive across sequences and no aliasing is possible.
    /// `forward_prefill` then reads the assembled `norm_output[0..n]` and
    /// writes `moe_output[0..n]`.
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_seqs;
        let ssm_ms_profile = std::env::var("ATLAS_SSM_MS_PROFILE").ok().as_deref() == Some("1")
            && !ctx.graph_capture;
        let phase_a_t0 = if ssm_ms_profile {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Per-seq hidden/residual stride: the residual stream is always
        // BF16 (2 bytes), so hardcode the per-seq stride.
        let residual_elem = 2usize;

        // ── Phase A: SSM mixer ──
        // Pre-norm, SSM mixer (recurrent, per-seq state), post-attn-norm.
        // Lays out `norm_output[0..n]` as the contiguous [N, h] BF16 MoE
        // input. The MoE is deferred to Phase B.
        //
        // Fast path (batched projections): when the layer uses the
        // sequential-QKVZ dense/NVFP4 weights with the FP32 conv+GDN
        // recurrent kernels (the GB10 Holo serving config), the big
        // QKVZ and out_proj GEMMs are batched into a single [N, ...] GEMM
        // each — reading the ~50 MB QKVZ / out_proj weights ONCE instead
        // of N times. On bandwidth-bound LPDDR5X this is the dominant
        // decode cost, so it is the lever that makes C=N decode scale.
        // The recurrent inner (BA/gates, conv1d, GDN, gated-norm) stays a
        // per-seq loop with byte-identical kernels to `decode()`/`ssm_forward`.
        if !self.try_decode_multi_seq_ssm_batched(hidden, residual, n, states, ctx, stream)? {
            for i in 0..n {
                let hidden_i = hidden.offset(i * h * residual_elem);
                let residual_i = residual.offset(i * h * residual_elem);
                let normed_i = ctx.buffers.norm_output().offset(i * h * bf16);

                let ssm_state = states[i]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

                // normed_i = rms_norm(hidden_i); residual_i = hidden_i
                ops::rms_norm_residual(
                    ctx.gpu,
                    self.rms_norm_residual_k,
                    hidden_i,
                    &self.input_norm,
                    normed_i,
                    residual_i,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;

                // SSM mixer: consumes normed_i, returns ssm_out (in moe_output[0]).
                let ssm_out = self.ssm_forward(normed_i, ssm_state, ctx, stream, false)?;

                // hidden_i += ssm_out; normed_i = rms_norm(hidden_i); residual_i = hidden_i
                ops::residual_add_rms_norm(
                    ctx.gpu,
                    self.residual_add_rms_norm_k,
                    hidden_i,
                    ssm_out,
                    &self.post_attn_norm,
                    normed_i,
                    residual_i,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
        }
        let phase_a_us = if let Some(t0) = phase_a_t0 {
            ctx.gpu.synchronize(stream).ok();
            t0.elapsed().as_micros()
        } else {
            0
        };
        let phase_b_t0 = if ssm_ms_profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── Phase B+C: MoE + residual, dispatched by batch size ──
        // Measured on GB10 (qwen3.5-122b, 256-expert MoE, EP=2):
        //   N=2/3: the FUSED batch-2/3 expert kernels (forward_k2/k3) win —
        //          SSM step 44->36.5ms at N=2 (one batched all_reduce, no
        //          per-token launch overhead).
        //   N>=4:  the generic grouped-GEMM (forward_prefill) is a NET LOSS
        //          here — per-expert M ~1, and the expert sort/permute/ptr-
        //          table overhead (paid once per layer, x36 SSM layers)
        //          dominates (SSM step ~88ms per-token vs ~140ms grouped).
        //          So fall back to the per-token MoE loop, identical to
        //          decode()'s MoE — the fastest option at these sizes until
        //          a true batched-EP MoE kernel exists.
        // Mirrors the attention layers' forward_k2/k3 dispatch
        // (qwen3_attention/.../multi_seq/ffn.rs); diverges only in declining
        // forward_prefill at N>=4, which that path uses but which loses for
        // the 36-layer SSM stack.
        let normed_base = ctx.buffers.norm_output();
        match n {
            2 | 3 => {
                if n == 2 {
                    self.ffn.forward_k2(normed_base, ctx, stream)?;
                } else {
                    self.ffn.forward_k3(normed_base, ctx, stream)?;
                }
                // Batched output lives in moe_output[0..n].
                for i in 0..n {
                    let hidden_i = hidden.offset(i * h * residual_elem);
                    let moe_out_i = ctx.buffers.moe_output().offset(i * h * bf16);
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden_i,
                        moe_out_i,
                        h as u32,
                        stream,
                    )?;
                }
            }
            _ => {
                // Per-token MoE: each seq's forward() writes moe_output[0];
                // consume it immediately with a per-seq residual add before
                // the next iteration overwrites it. NOTE: the batched
                // grouped-GEMM (forward_prefill) was measured SLOWER here on
                // Holo (c4 31 vs 56 tok/s) — the expert sort/permute fixed
                // overhead per layer dominates at small N. The real fix for
                // this launch overhead is CUDA graphs for n>=2, not MoE
                // batching (graphs capture these per-token launches for free).
                if std::env::var("ATLAS_MOE_GROUPED_DECODE").ok().as_deref() == Some("1") {
                    // Grouped-GEMM MoE over all N tokens (each expert read once).
                    // Only sensible under CUDA graphs, where the sort/permute
                    // launch overhead that made this a loss is captured for free.
                    self.ffn.forward_prefill(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else if n == 4
                    && std::env::var("ATLAS_MOE_ATOMIC_C4_DECODE")
                        .ok()
                        .as_deref()
                        == Some("1")
                {
                    // Purpose-built C=4 routed MoE decode: batched routing,
                    // token-major gate/up, FP32 atomicAdd routed down
                    // accumulation, then BF16 finalize/blend.
                    self.ffn.forward_atomic_c4_decode(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else if std::env::var("ATLAS_MOE_TOKEN_MAJOR_DECODE")
                    .ok()
                    .as_deref()
                    == Some("1")
                {
                    // Token-major N-token MoE decode: batched gate/top-k plus
                    // generic fused routed/shared kernels, no grouped-GEMM sort.
                    self.ffn
                        .forward_token_major_decode(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else if std::env::var("ATLAS_MOE_BATCHED_DECODE").ok().as_deref() == Some("1") {
                    // Batched gate GEMM over all N tokens, but keep the proven
                    // per-token expert kernels. This avoids the grouped path's
                    // sort/GEMM overhead while testing whether reading router
                    // weights once helps C=4 decode.
                    self.ffn.forward_batched(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else {
                    for i in 0..n {
                        let hidden_i = hidden.offset(i * h * residual_elem);
                        let normed_i = normed_base.offset(i * h * bf16);
                        let moe_out = self.ffn.forward(normed_i, ctx, stream)?;
                        ops::residual_add(
                            ctx.gpu,
                            self.residual_add_k,
                            hidden_i,
                            moe_out,
                            h as u32,
                            stream,
                        )?;
                    }
                }
            }
        }
        if let Some(t0) = phase_b_t0 {
            ctx.gpu.synchronize(stream).ok();
            tracing::info!(
                "ATLAS_SSM_MS_PROFILE n={n}: mixer={}us moe_residual={}us",
                phase_a_us,
                t0.elapsed().as_micros(),
            );
        }

        Ok(())
    }

    /// Batched-projection SSM mixer for N concurrent decode sequences.
    ///
    /// Returns `Ok(false)` (caller falls back to the per-seq loop) unless the
    /// layer is in the GB10 Holo serving config: sequential-QKVZ dense/NVFP4
    /// weights + FP32 conv/GDN recurrent kernels. When eligible, the big QKVZ
    /// and out_proj projections run as a single `[N, ...]` GEMM each (weights
    /// read ONCE, not N times — the dominant bandwidth cost on LPDDR5X), while
    /// the recurrent inner (BA/gates → conv1d → GDN → gated-norm) stays a
    /// per-seq loop using the SAME single-token kernels as `ssm_forward`, so
    /// the recurrence is byte-identical to the proven path. The per-seq states
    /// are read straight from each `SsmLayerState`, so no contiguous-slot
    /// assumption is required.
    #[allow(clippy::too_many_arguments)]
    fn try_decode_multi_seq_ssm_batched<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let use_f32_conv = self.conv1d_l2norm_f32_k.0 != 0;
        let use_f32_gdn = self.gdn_f32_k.0 != 0 && self.gated_rms_norm_f32_k.0 != 0;
        // QKVZ via dense BF16 GEMM or block-scaled FP8 GEMM (w8a16). NVFP4 and
        // interleaved-QKVZ layouts take the proven per-seq loop.
        let qkvz_ok = self.qkvz_nvfp4.is_none() && self.w8a16_gemm_k.0 != 0;
        let out_ok = self.out_proj_fp8w.is_some() || self.out_proj_dense.is_some();
        if n < 2 || !self.sequential_qkvz || !use_f32_conv || !use_f32_gdn || !qkvz_ok || !out_ok {
            return Ok(false);
        }

        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let eps = ctx.config.rms_norm_eps as f32;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = (key_dim * 2 + value_dim) as u32;
        let qk_channels = (key_dim * 2) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim as u32;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size() as u32;

        let normed_base = ctx.buffers.norm_output();
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        // normed_out[0..n] (post gated-norm, [N, value_dim] BF16) parks in the
        // QKVZ scratch — free here because QKVZ projects into `deinterleaved`
        // and the FP32 conv path uses `ssm_conv_out_f32`, not `ssm_qkvz`.
        let normed_out_base = ctx.buffers.ssm_qkvz();
        let ssm_out_base = ctx.buffers.moe_output();
        let detail_profile = std::env::var("ATLAS_SSM_DETAIL_PROFILE").ok().as_deref() == Some("1")
            && !ctx.graph_capture;
        let mut detail_parts: Vec<(&'static str, u128)> = Vec::new();
        let mut detail_t0 = if detail_profile {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut rec_ba_us = 0u128;
        let mut rec_conv_us = 0u128;
        let mut rec_gdn_us = 0u128;
        let mut rec_norm_us = 0u128;
        macro_rules! detail_step {
            ($label:expr) => {
                if let Some(t0) = detail_t0.take() {
                    ctx.gpu.synchronize(stream).ok();
                    detail_parts.push(($label, t0.elapsed().as_micros()));
                    detail_t0 = Some(std::time::Instant::now());
                }
            };
            ($label:expr, final) => {
                if let Some(t0) = detail_t0.take() {
                    ctx.gpu.synchronize(stream).ok();
                    detail_parts.push(($label, t0.elapsed().as_micros()));
                }
            };
        }

        // ── 1. Batched input RMS norm: hidden[0..n] → normed[0..n], residual ──
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed_base,
            residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        detail_step!("input_norm");

        // ── 2. Batched QKVZ projection: ONE [N,h]→[N,qkvz] GEMM (weights ×1) ──
        // FP8 (w8a16) when the decode overlay is installed, else BF16 dense.
        // Prefer the pipelined (cp.async) w8a16 kernel — bit-identical, ~4.6×
        // faster than the base w8a16_gemm, which nsys showed as 44.6% of the
        // C>1 decode step. `.0 == 0` → fall back to the base kernel.
        let w8a16_pipe = self.w8a16_gemm_pipelined_k.0 != 0;
        // Weight-streaming block-scaled GEMV for batched decode: avoids the
        // pipelined kernel's M->128 MMA pad (issue-bound). batch4 (M<=4) for the
        // common path, batch16 (M<=16) for high-concurrency C=8/16. Bit-identical
        // per row to w8a16_gemv. Disable with ATLAS_SSM_GEMV_BATCH4=0.
        let gemv_batch_k = if n <= 4 {
            self.w8a16_gemv_batch4_k
        } else {
            self.w8a16_gemv_batch16_k
        };
        let use_batch4 = gemv_batch_k.0 != 0
            && n <= 16
            && std::env::var("ATLAS_SSM_GEMV_BATCH4").ok().as_deref() != Some("0");
        if let Some(ref fp8) = self.qkvz_fp8w {
            if use_batch4 {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    gemv_batch_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else if w8a16_pipe {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed_base,
                &self.ssm.in_proj_qkvz,
                deinterleaved,
                n as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        detail_step!("qkvz");

        // ── 3. Recurrent inner ──
        // Default: per-seq, byte-identical to ssm_forward. Experimental path:
        // use existing batch dimensions for BA/gates, conv, GDN, and gated norm
        // when the SSM pool states are contiguous slots [0..n).
        let batched_recurrent = if std::env::var("ATLAS_SSM_BATCHED_RECURRENT").ok().as_deref()
            == Some("1")
            && self.gdn_f32_strided_k.0 != 0
            && n > 1
        {
            let mut h_base = DevicePtr::NULL;
            let mut conv_base = DevicePtr::NULL;
            let mut contiguous = true;
            for i in 0..n {
                let ssm_state = states[i]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                if i == 0 {
                    h_base = ssm_state.h_state;
                    conv_base = ssm_state.conv_state;
                } else {
                    contiguous &= ssm_state.h_state.0 == h_base.0 + (i * self.h_state_bytes) as u64;
                    contiguous &=
                        ssm_state.conv_state.0 == conv_base.0 + (i * self.conv_state_bytes) as u64;
                }
            }
            if contiguous {
                Some((h_base, conv_base))
            } else {
                None
            }
        } else {
            None
        };

        if let Some((h_state_base, conv_state_base)) = batched_recurrent {
            let gates = ctx.buffers.ssm_gates();
            let beta_fp32 = gates.offset(nv * 4);
            let gate_stride = (nv * 2) as u32;
            ops::dense_gemm_ba_gates_prefill(
                ctx.gpu,
                self.ba_gates_prefill_k,
                normed_base,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates,
                n as u32,
                ba_size,
                h as u32,
                h as u32,
                gate_stride,
                nv as u32,
                vpg as u32,
                stream,
            )?;
            detail_step!("recurrent_batched_ba");

            let conv_out = ctx.buffers.ssm_conv_out_f32();
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_f32_k,
                conv_state_base,
                deinterleaved,
                &self.ssm.conv1d,
                conv_out,
                conv_dim,
                d_conv,
                n as u32,
                qk_channels,
                kd as u32,
                1e-6,
                stream,
            )?;
            detail_step!("recurrent_batched_conv");

            if self.gdn_f32_strided_norm_k.0 != 0
                && std::env::var("ATLAS_GDN_FUSED_NORM").ok().as_deref() == Some("1")
            {
                let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
                ops::gdn_decode_f32_strided_norm(
                    ctx.gpu,
                    self.gdn_f32_strided_norm_k,
                    h_state_base,
                    conv_out,
                    conv_out.offset(key_dim * 4),
                    conv_out.offset(key_dim * 2 * 4),
                    gates,
                    beta_fp32,
                    z_base,
                    self.ssm.norm.weight,
                    normed_out_base,
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim,
                    conv_dim,
                    gate_stride,
                    qkvz_size as u32,
                    value_dim as u32,
                    eps,
                    stream,
                )?;
                detail_step!("recurrent_batched_gdn_norm");
            } else {
                let gdn_out = conv_out.offset(n * conv_dim as usize * 4);
                ops::gdn_decode_f32_strided(
                    ctx.gpu,
                    self.gdn_f32_strided_k,
                    h_state_base,
                    conv_out,
                    conv_out.offset(key_dim * 4),
                    conv_out.offset(key_dim * 2 * 4),
                    gates,
                    beta_fp32,
                    gdn_out,
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim,
                    conv_dim,
                    gate_stride,
                    value_dim as u32,
                    stream,
                )?;
                detail_step!("recurrent_batched_gdn");

                for i in 0..n {
                    let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
                    let z_i = deint_i.offset((key_dim * 2 + value_dim) * bf16);
                    let gdn_out_i = gdn_out.offset(i * value_dim * 4);
                    let normed_out_i = normed_out_base.offset(i * value_dim * bf16);
                    ops::gated_rms_norm(
                        ctx.gpu,
                        self.gated_rms_norm_f32_k,
                        gdn_out_i,
                        z_i,
                        &self.ssm.norm,
                        normed_out_i,
                        nv as u32,
                        vd as u32,
                        vd as u32,
                        eps,
                        vd as u32,
                        stream,
                    )?;
                }
                detail_step!("recurrent_batched_norm");
            }
        } else {
            for i in 0..n {
                let normed_i = normed_base.offset(i * h * bf16);
                let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
                let z_i = deint_i.offset((key_dim * 2 + value_dim) * bf16);
                let normed_out_i = normed_out_base.offset(i * value_dim * bf16);

                let ssm_state = states[i]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

                let gates = ctx.buffers.ssm_gates();
                let beta_fp32 = gates.offset(nv * 4);
                let sub_t0 = if detail_profile {
                    ctx.gpu.synchronize(stream).ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                ops::dense_gemv_ba_gates(
                    ctx.gpu,
                    self.ba_gates_k,
                    normed_i,
                    &self.ssm.in_proj_ba,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gates,
                    beta_fp32,
                    ba_size,
                    h as u32,
                    vpg as u32,
                    stream,
                )?;
                if let Some(t0) = sub_t0 {
                    ctx.gpu.synchronize(stream).ok();
                    rec_ba_us += t0.elapsed().as_micros();
                }

                let conv_out = ctx.buffers.ssm_conv_out_f32();
                let sub_t0 = if detail_profile {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_k,
                    ssm_state.conv_state,
                    deint_i,
                    &self.ssm.conv1d,
                    conv_out,
                    conv_dim,
                    d_conv,
                    1,
                    qk_channels,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                if let Some(t0) = sub_t0 {
                    ctx.gpu.synchronize(stream).ok();
                    rec_conv_us += t0.elapsed().as_micros();
                }

                let gdn_out = conv_out.offset((key_dim * 2 + value_dim) * 4);
                let q_conv = conv_out;
                let k_conv = conv_out.offset(key_dim * 4);
                let v_conv = conv_out.offset(key_dim * 2 * 4);
                let sub_t0 = if detail_profile {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                if self.gdn_f32_norm_k.0 != 0
                    && std::env::var("ATLAS_GDN_FUSED_NORM").ok().as_deref() == Some("1")
                {
                    ops::gdn_decode_f32_norm(
                        ctx.gpu,
                        self.gdn_f32_norm_k,
                        ssm_state.h_state,
                        q_conv,
                        k_conv,
                        v_conv,
                        gates,
                        beta_fp32,
                        z_i,
                        self.ssm.norm.weight,
                        normed_out_i,
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        eps,
                        stream,
                    )?;
                    if let Some(t0) = sub_t0 {
                        ctx.gpu.synchronize(stream).ok();
                        rec_gdn_us += t0.elapsed().as_micros();
                    }
                } else {
                    ops::gdn_decode(
                        ctx.gpu,
                        self.gdn_f32_k,
                        ssm_state.h_state,
                        q_conv,
                        k_conv,
                        v_conv,
                        gates,
                        beta_fp32,
                        gdn_out,
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        stream,
                    )?;
                    if let Some(t0) = sub_t0 {
                        ctx.gpu.synchronize(stream).ok();
                        rec_gdn_us += t0.elapsed().as_micros();
                    }

                    let sub_t0 = if detail_profile {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    ops::gated_rms_norm(
                        ctx.gpu,
                        self.gated_rms_norm_f32_k,
                        gdn_out,
                        z_i,
                        &self.ssm.norm,
                        normed_out_i,
                        nv as u32,
                        vd as u32,
                        vd as u32,
                        eps,
                        vd as u32,
                        stream,
                    )?;
                    if let Some(t0) = sub_t0 {
                        ctx.gpu.synchronize(stream).ok();
                        rec_norm_us += t0.elapsed().as_micros();
                    }
                }
            }
            if detail_profile {
                detail_parts.push(("recurrent_ba", rec_ba_us));
                detail_parts.push(("recurrent_conv", rec_conv_us));
                detail_parts.push(("recurrent_gdn", rec_gdn_us));
                if rec_norm_us > 0 {
                    detail_parts.push(("recurrent_norm", rec_norm_us));
                }
                detail_t0 = Some(std::time::Instant::now());
            }
        }
        detail_step!("recurrent_total_tail");

        // ── 4. Batched out_proj: ONE [N,value_dim]→[N,h] GEMM (weights ×1) ──
        // FP8 (w8a16) when the decode overlay is installed, else BF16 dense.
        if let Some(ref fp8) = self.out_proj_fp8w {
            if use_batch4 {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    gemv_batch_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else if w8a16_pipe {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else if let Some(ref out_proj_dense) = self.out_proj_dense {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed_out_base,
                out_proj_dense,
                ssm_out_base,
                n as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        }
        detail_step!("out_proj");

        // ── 5. Batched residual add + post-attn RMS norm → norm_output[0..n] ──
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            ssm_out_base,
            &self.post_attn_norm,
            normed_base,
            residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        detail_step!("post_norm", final);
        if detail_profile {
            let summary = detail_parts
                .iter()
                .map(|(label, us)| format!("{label}={us}us"))
                .collect::<Vec<_>>()
                .join(" ");
            tracing::info!("ATLAS_SSM_DETAIL n={n}: {summary}");
        }

        Ok(true)
    }
}
