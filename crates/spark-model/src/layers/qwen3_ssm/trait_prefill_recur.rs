// SPDX-License-Identifier: AGPL-3.0-only

//! GDN recurrence kernel dispatch for `Qwen3SsmLayer::prefill_inner`.
//!
//! Hoisted from `trait_prefill.rs` to keep that file under the 500 LoC
//! cap. [`Qwen3SsmLayer::prefill_gdn_recurrence`] mirrors the original
//! step 8 block 1:1 — same WY4-persistent / single-token persistent /
//! split4 dispatch, same env overrides, same kernel launches.

use super::*;

impl Qwen3SsmLayer {
    /// GDN prefill recurrence via the WY4-persistent kernel.
    ///
    /// Dispatch: FLA chunked prefill (baked default, 128-dim linear heads) →
    /// WY4-persistent (4 tokens/iter, H in shared memory) → single-token persistent
    /// (256..=4096) → split4 for unsupported configurations.
    ///
    /// Returns `true` when the GDN output was written in FP32 to
    /// `ctx.buffers.ssm_conv_out_f32()` (FlashInfer F32-output path) instead of
    /// BF16 to `gdn_out_buf` — the caller must then use the F32-input gated
    /// norm ([`Self::prefill_gdn_gated_norm`] dispatches on this flag).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_gdn_recurrence(
        &self,
        h_state: DevicePtr,
        q_ptr: DevicePtr,
        k_ptr: DevicePtr,
        v_ptr: DevicePtr,
        gates_buf: DevicePtr,
        gdn_out_buf: DevicePtr,
        k: u32,
        nk: usize,
        nv: usize,
        kd: usize,
        vd: usize,
        conv_dim: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let fp32 = 4usize;
        let gb_stride = (nv * 2) as u32;

        // gfx1151/SCALE (atlas_scale): every H-in-shared-memory GDN prefill
        // kernel exceeds RDNA3.5's 64KB LDS cap — FLA (C=64) ≈96KB, WY4 =69688,
        // persistent =67584. Only split4 keeps the kd*vd H-state in global
        // memory (~2KB smem) and handles arbitrary length, so route there for
        // all sizes. Correctness-equivalent, lower throughput; the smem-H fast
        // paths (and a future C=32 FLA variant) are Blackwell-only. NVIDIA
        // (cfg unset) takes the full FLA/WY ladder below unchanged.
        if cfg!(atlas_scale) {
            ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
            return Ok(false);
        }

        // FlashInfer GDN (opt-in, ATLAS_GDN_FLASHINFER=1): tensor-core chunked delta-rule
        // scan, ~11× the scalar FLA chunk_delta_h at the Holo shape. This is the live
        // single-stream prefill path (trait_prefill.rs -> prefill_gdn_recurrence). q_ptr is
        // the packed-QKV base, gates_buf the gate base — handed straight to the bit-exact
        // shim (ops::gdn_flashinfer). FLA ladder below is the fallback when flag/lib absent.
        if !ctx.gdn_exact_replay && kd == 128 && vd == 128 && ops::gdn_flashinfer::available() {
            let scale = 1.0f32 / (kd as f32).sqrt();
            // F32 GDN output (avarok #248/#290): when the lib exports the F32
            // entry AND the F32-input prefill norm kernel is loaded, write the
            // recurrence output in FP32 to ssm_conv_out_f32 (idle during the
            // GDN step; m*qkvz_size*4 B ≥ the m*value_dim*4 B needed) so the
            // recurrence→gated-norm handoff skips the BF16 truncation — the
            // long-context (200K needle) coherence lever for the FlashInfer
            // path. Kill switch: ATLAS_GDN_FI_F32_OUT=0.
            let f32_out = ops::gdn_flashinfer::f32_output_available()
                && self.gated_rms_norm_prefill_f32_k.0 != 0
                && ctx.buffers.ssm_conv_out_f32().0 != 0;
            let out = if f32_out {
                ctx.buffers.ssm_conv_out_f32()
            } else {
                gdn_out_buf
            };
            ops::gdn_flashinfer::flashinfer_gdn_prefill(
                ctx.gpu,
                q_ptr,
                gates_buf,
                out,
                h_state,
                scale,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                gb_stride,
                1,
                f32_out,
                stream,
            )?;
            return Ok(f32_out);
        }

        // 2026-06-06: removed the concluded GDN-prefill experiment env flags
        // (ATLAS_GDN_CHUNK64 / ATLAS_FORCE_PERSISTENT / ATLAS_DISABLE_WY4) and their
        // dispatch branches. FLA is the baked default for 128-dim linear heads; the
        // WY4-persistent kernel is the unconditional fallback below.
        // FLA multi-kernel chunked prefill (recompute_wu → chunk_delta_h_ksplit →
        // chunk_fwd_o): 1.75x vs wy4 @16k, token-equal (cos=1.0 vs scalar). BAKED
        // DEFAULT 2026-06-06 (was gated behind ATLAS_GDN_FLA=1 — the env var is gone):
        // always taken for 128-dim linear-head GDN models when the FLA kernels & scratch
        // are present (scratch is allocated for exactly those models, sizes.rs). The wy4
        // branch below remains the fallback for other head dims / a guard miss.
        // Warm-hit replay (Marconi SSM snapshot restored): force the WY4
        // recurrence. FLA's chunked algebra is only token-equal when its
        // 64-token grid matches the pass that originally produced the cached
        // K/V; a replay anchored at an arbitrary snapshot offset regroups the
        // recurrence and its bf16 W/U/uc/S_c intermediates drift. The replay
        // range is rewritten into SHARED prefix-cache blocks, so non-exact
        // recompute poisons them and the drift ratchets across agentic turns
        // (token-stutter corruption, 2026-06-10). WY4 keeps H in FP32 SMEM
        // token-sequentially — same family as the decode kernel — and is the
        // path the clean pre-FLA baseline used. Replay segments are short
        // (suffix after a ≥10k skipped prefix), so the FLA speed loss is nil.
        let fla_scratch = ctx.buffers.gdn_fla_scratch();
        if !ctx.gdn_exact_replay
            && kd == 128
            && vd == 128
            && fla_scratch.0 != 0
            && self.gdn_prefill_fla_recompute_wu_k.0 != 0
            && self.gdn_prefill_fla_chunk_delta_h_k.0 != 0
            && self.gdn_prefill_fla_chunk_fwd_o_k.0 != 0
        {
            // One-time positive signal that the FLA path is live (vs silently
            // falling through to wy4 on a guard miss) — greppable in the server log.
            static FLA_LOG: std::sync::Once = std::sync::Once::new();
            FLA_LOG.call_once(|| {
                tracing::info!(
                    "GDN prefill: FLA chunked path ACTIVE (baked default: recompute_wu → chunk_delta_h_ksplit → chunk_fwd_o)"
                );
            });
            let num_chunks = k.div_ceil(64);
            let nt = num_chunks as usize;
            let w_out = fla_scratch;
            let u_out = w_out.offset(nt * nv * 64 * kd * 2);
            let s_out = u_out.offset(nt * nv * 64 * vd * 2);
            let uc_out = s_out.offset(nt * nv * kd * vd * 2);
            let gc_out = uc_out.offset(nt * nv * 64 * vd * 2);
            ops::gdn_prefill_fla(
                ctx.gpu,
                self.gdn_prefill_fla_recompute_wu_k,
                self.gdn_prefill_fla_chunk_delta_h_k,
                self.gdn_prefill_fla_chunk_delta_h_tc_vblock_k,
                self.gdn_prefill_fla_chunk_fwd_o_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                w_out,
                u_out,
                s_out,
                uc_out,
                gc_out,
                1,
                k,
                num_chunks,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                false, // single-stream: contiguous h_state (not a pointer table)
                spark_runtime::gpu::DevicePtr::NULL, // cu_seqlens (unused)
                spark_runtime::gpu::DevicePtr::NULL, // cu_chunks (unused)
                false, // not varlen
                ctx.profile,
                stream,
            )?;
        } else if std::env::var_os("ATLAS_GDN_REGRESIDENT").is_some()
            && kd == 128
            && vd == 128
            && self.gdn_prefill_regresident_k.0 != 0
        {
            // Register-resident token-sequential recurrence — drop-in for WY4 on
            // the warm Marconi-replay path (this branch is only reached when the
            // FLA `if` above fell through, i.e. gdn_exact_replay). H lives in
            // registers (one warp per v-column, 4 k-rows/lane) instead of 64KB
            // smem, so >=2 CTA/SM and no per-token barriers. Token-equal to WY4
            // (cosine 1.0, max|dH|~1e-8 — same acceptance class) and ~2.9x faster
            // in isolation. Gated by ATLAS_GDN_REGRESIDENT until serve-validated.
            static RR_LOG: std::sync::Once = std::sync::Once::new();
            RR_LOG.call_once(|| {
                tracing::info!(
                    "GDN prefill: REGISTER-RESIDENT warm-replay path ACTIVE (ATLAS_GDN_REGRESIDENT; H in regs, no smem-H)"
                );
            });
            ops::gdn_prefill_regresident(
                ctx.gpu,
                self.gdn_prefill_regresident_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if self.gdn_prefill_persistent_wy4_k.0 != 0 {
            // WY4-persistent: H in shared memory, 4 tokens per iteration
            // smem = H[K_DIM*V_DIM] + 8*k/q buffers + warp sums + WY scalars
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&k) && self.gdn_prefill_persistent_k.0 != 0 {
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        }
        Ok(false)
    }

    /// Step-9 gated RMS norm dispatch for the monolithic prefill: reads the GDN
    /// output from the FP32 scratch via the F32-input kernel when
    /// [`Self::prefill_gdn_recurrence`] returned `true`, else the BF16 buffer
    /// via the baked BF16 kernel. Hoisted here to keep `trait_prefill.rs` under
    /// the 500-LoC cap.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_gdn_gated_norm(
        &self,
        gdn_out_f32: bool,
        gdn_out_buf: DevicePtr,
        z_base: DevicePtr,
        normed_out_buf: DevicePtr,
        nv: usize,
        vd: usize,
        eps: f32,
        k: u32,
        value_dim: usize,
        qkvz_size: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if gdn_out_f32 {
            ops::gated_rms_norm_prefill_f32_input(
                ctx.gpu,
                self.gated_rms_norm_prefill_f32_k,
                ctx.buffers.ssm_conv_out_f32(),
                z_base,
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                vd as u32,
                eps,
                k,
                value_dim as u32,
                qkvz_size as u32,
                stream,
            )
        } else {
            ops::gated_rms_norm_prefill(
                ctx.gpu,
                self.gated_rms_norm_prefill_k,
                gdn_out_buf,
                z_base,
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                vd as u32,
                eps,
                k,
                value_dim as u32,
                qkvz_size as u32,
                stream,
            )
        }
    }
}
