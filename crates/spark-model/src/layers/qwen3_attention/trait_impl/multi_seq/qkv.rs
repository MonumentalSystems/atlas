// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2: per-token Q/K/V projection. Four branches:
//! - n=3 + NVFP4  → batch3 GEMV path
//! - n=2 + NVFP4  → batch2 GEMV path
//! - n>=4 + NVFP4 → tiled batch3/batch2 GEMV path (greedy 3s + 2/1 remainder)
//! - else         → sequential per-token GEMV (FP8/NVFP4/BF16 fallback)
//!
//! The batch paths read each weight once per tile and then scatter into the
//! per-seq QKV layout. The tiled path extends that to n>=4 so decode
//! concurrency doesn't collapse back to one weight read per token. The
//! sequential path repeats the GEMV per token but supports every weight
//! encoding.

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    pub(super) fn ms_phase_qkv(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            bf16,
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;

        if n == 3
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batch3(c)?;
        } else if n == 2
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batch2(c)?;
        } else if n >= 4
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            // n>=4 NVFP4: tile the proven batch3/batch2 GEMV kernels so the
            // q/k/v projection weights are read once per tile (n=8 -> 3+3+2 =
            // 3 weight-matrix reads instead of 8 per-seq) — amortizes the
            // attention-projection bandwidth that otherwise caps decode scaling.
            self.ms_qkv_batch_tiled(c)?;
        } else {
            // Projection only — the q/k RMS norms are deferred to the shared
            // tail so the per-request K/V LoRA delta lands BEFORE k_norm
            // (HF: k_norm(k_proj(x)+Δ); the single-seq oracle does the same).
            for i in 0..n {
                let normed_i = normed.offset(i * h * bf16);
                let q_out_i = qkv_buf.offset(i * per_seq_qkv);
                let k_out_i = q_out_i.offset(q_proj_bytes);
                let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);

                self.ms_qkv_seq_q(fwd, normed_i, q_out_i, q_proj_dim, q_dim, nq, hd, h, stream)?;
                self.ms_qkv_seq_kv(fwd, normed_i, k_out_i, v_out_i, nkv, hd, h, stream)?;
            }
        }

        // ── Per-request K/V LoRA delta (batched bgmv), pre-norm. No-op when no
        // routing table is installed or `seq_slot` is null (base model / n==1).
        self.ms_qkv_apply_lora(c)?;

        // ── Shared q/k RMS-norm pass (all three projection branches).
        let _ = (q_dim, q_proj_dim); // consumed by the projection branches
        self.ms_qkv_norms(c)?;
        Ok(())
    }

    /// Per-request K/V LoRA routing on the batched decode path. Applies each
    /// sequence's own adapter delta to the strided `qkv_buf` K and V regions
    /// via the fused bgmv (byte-identical to N single-seq `apply_lora_delta`).
    /// No-op unless a routing table is installed AND `seq_slot` is non-null.
    fn ms_qkv_apply_lora(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let Some(ref lw) = self.lora else {
            return Ok(());
        };
        if c.seq_slot.0 == 0 {
            return Ok(());
        }
        let bf16 = c.bf16;
        let out_row_stride = (c.per_seq_qkv / bf16) as u32; // strided [Q|K|V] layout
        let x_row_stride = c.h as u32; // normed rows are contiguous [n, h]
        let kv_bytes = (c.nkv * c.hd) as usize * bf16;
        // K delta: base_out = k_out region (after Q), fold in place.
        if let Some(ref route) = lw.k_route {
            let k_out0 = c.qkv_buf.offset(c.q_proj_bytes);
            ops::lora_delta::apply_lora_bgmv(
                c.fwd.gpu,
                &lw.kernels,
                route,
                c.normed,
                k_out0,
                c.seq_slot,
                c.n as u32,
                x_row_stride,
                out_row_stride,
                c.fwd.buffers.lora_xa(),
                c.stream,
            )?;
        }
        // V delta: base_out = v_out region (after Q and K).
        if let Some(ref route) = lw.v_route {
            let v_out0 = c.qkv_buf.offset(c.q_proj_bytes + kv_bytes);
            ops::lora_delta::apply_lora_bgmv(
                c.fwd.gpu,
                &lw.kernels,
                route,
                c.normed,
                v_out0,
                c.seq_slot,
                c.n as u32,
                x_row_stride,
                out_row_stride,
                c.fwd.buffers.lora_xa(),
                c.stream,
            )?;
        }
        Ok(())
    }

    /// Shared q/k RMS-norm pass over the per-seq `qkv_buf` regions. Extracted
    /// so all three projection branches (seq / batch2 / batch3) defer norms to
    /// one place, after the pre-norm K/V LoRA delta.
    fn ms_qkv_norms(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let _ = bf16;
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// n=3 NVFP4 batched path.
    fn ms_qkv_batch3(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let q_scratch = fwd.buffers.ssm_qkvz();
        if self.gated {
            ops::w4a16_gemv_qg_batch3(
                fwd.gpu,
                self.w4a16_gemv_qg_batch3_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_batch3(
                fwd.gpu,
                self.w4a16_gemv_batch3_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                stream,
            )?;
        }

        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(3 * kv_bytes);
        ops::w4a16_gemv_dual_batch3(
            fwd.gpu,
            self.w4a16_gemv_dual_batch3_k,
            normed,
            k_nvfp4,
            k_scratch,
            v_nvfp4,
            v_scratch,
            kv_dim,
            h as u32,
            stream,
        )?;

        for i in 0..3usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }
        // q/k norms deferred to ms_qkv_norms (after the pre-norm K/V LoRA delta).
        Ok(())
    }

    /// n=2 NVFP4 batched path.
    fn ms_qkv_batch2(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let q_scratch = fwd.buffers.ssm_qkvz();
        if self.gated {
            ops::w4a16_gemv_qg_batch2(
                fwd.gpu,
                self.w4a16_gemv_qg_batch2_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_batch2(
                fwd.gpu,
                self.w4a16_gemv_batch2_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                stream,
            )?;
        }

        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(2 * kv_bytes);
        ops::w4a16_gemv_dual_batch2(
            fwd.gpu,
            self.w4a16_gemv_dual_batch2_k,
            normed,
            k_nvfp4,
            k_scratch,
            v_nvfp4,
            v_scratch,
            kv_dim,
            h as u32,
            stream,
        )?;

        for i in 0..2usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }
        // q/k norms deferred to ms_qkv_norms (after the pre-norm K/V LoRA delta).
        Ok(())
    }

    /// n>=4 NVFP4 batched path: tile the proven `batch3`/`batch2` GEMV kernels
    /// across the n sequences (greedy 3s, then a 2 or 1 remainder). Each tile
    /// reads the q/k/v projection weights ONCE for its 2-3 tokens, so n=8 costs
    /// 3 weight-matrix reads instead of 8 — amortizing attention-projection
    /// bandwidth at decode. Layout is byte-identical to `ms_qkv_batch3` (same
    /// kernels, same q_proj_bytes/kv_bytes/per_seq_qkv offsets), so it inherits
    /// that path's correctness; the only new logic is the tiling loop + base
    /// offsets. The k==1 remainder falls back to the per-seq projection.
    fn ms_qkv_batch_tiled(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;

        let mut s = 0usize;
        while s < n {
            let k = (n - s).min(3);
            let normed_base = normed.offset(s * h * bf16);
            if k >= 2 {
                let q_scratch = fwd.buffers.ssm_qkvz();
                let k_scratch = fwd.buffers.attn_output();
                let v_scratch = k_scratch.offset(k * kv_bytes);
                if k == 3 {
                    if self.gated {
                        ops::w4a16_gemv_qg_batch3(
                            fwd.gpu,
                            self.w4a16_gemv_qg_batch3_k,
                            normed_base,
                            q_nvfp4,
                            q_scratch,
                            q_proj_dim,
                            h as u32,
                            nq,
                            hd,
                            stream,
                        )?;
                    } else {
                        ops::w4a16_gemv_batch3(
                            fwd.gpu,
                            self.w4a16_gemv_batch3_k,
                            normed_base,
                            q_nvfp4,
                            q_scratch,
                            q_proj_dim,
                            h as u32,
                            stream,
                        )?;
                    }
                    ops::w4a16_gemv_dual_batch3(
                        fwd.gpu,
                        self.w4a16_gemv_dual_batch3_k,
                        normed_base,
                        k_nvfp4,
                        k_scratch,
                        v_nvfp4,
                        v_scratch,
                        kv_dim,
                        h as u32,
                        stream,
                    )?;
                } else {
                    if self.gated {
                        ops::w4a16_gemv_qg_batch2(
                            fwd.gpu,
                            self.w4a16_gemv_qg_batch2_k,
                            normed_base,
                            q_nvfp4,
                            q_scratch,
                            q_proj_dim,
                            h as u32,
                            nq,
                            hd,
                            stream,
                        )?;
                    } else {
                        ops::w4a16_gemv_batch2(
                            fwd.gpu,
                            self.w4a16_gemv_batch2_k,
                            normed_base,
                            q_nvfp4,
                            q_scratch,
                            q_proj_dim,
                            h as u32,
                            stream,
                        )?;
                    }
                    ops::w4a16_gemv_dual_batch2(
                        fwd.gpu,
                        self.w4a16_gemv_dual_batch2_k,
                        normed_base,
                        k_nvfp4,
                        k_scratch,
                        v_nvfp4,
                        v_scratch,
                        kv_dim,
                        h as u32,
                        stream,
                    )?;
                }
                for j in 0..k {
                    let q_out = qkv_buf.offset((s + j) * per_seq_qkv);
                    let k_out = q_out.offset(q_proj_bytes);
                    let v_out = k_out.offset(kv_bytes);
                    fwd.gpu.copy_d2d_async(
                        q_scratch.offset(j * q_proj_bytes),
                        q_out,
                        q_proj_bytes,
                        stream,
                    )?;
                    fwd.gpu.copy_d2d_async(
                        k_scratch.offset(j * kv_bytes),
                        k_out,
                        kv_bytes,
                        stream,
                    )?;
                    fwd.gpu.copy_d2d_async(
                        v_scratch.offset(j * kv_bytes),
                        v_out,
                        kv_bytes,
                        stream,
                    )?;
                    if !self.attn.q_norm.weight.is_null() {
                        ops::rms_norm(
                            fwd.gpu,
                            self.rms_norm_k,
                            q_out,
                            &self.attn.q_norm,
                            q_out,
                            nq,
                            hd,
                            eps,
                            stream,
                        )?;
                    }
                    if !self.attn.k_norm.weight.is_null() {
                        ops::rms_norm(
                            fwd.gpu,
                            self.rms_norm_k,
                            k_out,
                            &self.attn.k_norm,
                            k_out,
                            nkv,
                            hd,
                            eps,
                            stream,
                        )?;
                    }
                }
                s += k;
            } else {
                // k == 1 remainder: per-seq projection (byte-identical to the
                // n<4 fallback loop).
                let q_out = qkv_buf.offset(s * per_seq_qkv);
                let k_out = q_out.offset(q_proj_bytes);
                let v_out = k_out.offset(kv_bytes);
                self.ms_qkv_seq_q(
                    fwd,
                    normed_base,
                    q_out,
                    q_proj_dim,
                    q_dim,
                    nq,
                    hd,
                    h,
                    stream,
                )?;
                self.ms_qkv_seq_kv(fwd, normed_base, k_out, v_out, nkv, hd, h, stream)?;
                if !self.attn.q_norm.weight.is_null() {
                    ops::rms_norm(
                        fwd.gpu,
                        self.rms_norm_k,
                        q_out,
                        &self.attn.q_norm,
                        q_out,
                        nq,
                        hd,
                        eps,
                        stream,
                    )?;
                }
                if !self.attn.k_norm.weight.is_null() {
                    ops::rms_norm(
                        fwd.gpu,
                        self.rms_norm_k,
                        k_out,
                        &self.attn.k_norm,
                        k_out,
                        nkv,
                        hd,
                        eps,
                        stream,
                    )?;
                }
                s += 1;
            }
        }
        Ok(())
    }

    /// Sequential per-token Q projection (handles gated and ungated).
    #[allow(clippy::too_many_arguments)]
    fn ms_qkv_seq_q(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        normed_i: spark_runtime::gpu::DevicePtr,
        q_out_i: spark_runtime::gpu::DevicePtr,
        q_proj_dim: u32,
        q_dim: u32,
        nq: u32,
        hd: u32,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        if self.gated {
            if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                ops::w8a16_gemv(
                    fwd.gpu,
                    self.w8a16_gemv_k,
                    normed_i,
                    fp8.weight,
                    fp8.row_scale,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    stream,
                )?;
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv_qg(
                    fwd.gpu,
                    self.w4a16_gemv_qg_k,
                    normed_i,
                    nvfp4,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    nq,
                    hd,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.q_proj,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    stream,
                )?;
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            }
        } else if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                fp8.weight,
                fp8.row_scale,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            ops::w4a16_gemv(
                fwd.gpu,
                self.w4a16_gemv_k,
                normed_i,
                nvfp4,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                fwd.gpu,
                self.dense_gemv_k,
                normed_i,
                &self.attn.q_proj,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }

    /// Sequential per-token K + V projections.
    #[allow(clippy::too_many_arguments)]
    fn ms_qkv_seq_kv(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        normed_i: spark_runtime::gpu::DevicePtr,
        k_out_i: spark_runtime::gpu::DevicePtr,
        v_out_i: spark_runtime::gpu::DevicePtr,
        nkv: u32,
        hd: u32,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        if let (Some(k_fp8), Some(v_fp8)) = (
            self.k_weight.as_ref().and_then(|w| w.as_fp8()),
            self.v_weight.as_ref().and_then(|w| w.as_fp8()),
        ) {
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                k_fp8.weight,
                k_fp8.row_scale,
                k_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                v_fp8.weight,
                v_fp8.row_scale,
                v_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
        } else if let (Some(k_fp4), Some(v_fp4)) = (
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) {
            ops::w4a16_gemv_dual(
                fwd.gpu,
                self.w4a16_gemv_dual_k,
                normed_i,
                k_fp4,
                k_out_i,
                v_fp4,
                v_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
        } else {
            if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
                    normed_i,
                    nvfp4,
                    k_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.k_proj,
                    k_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            }
            if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
                    normed_i,
                    nvfp4,
                    v_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.v_proj,
                    v_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
