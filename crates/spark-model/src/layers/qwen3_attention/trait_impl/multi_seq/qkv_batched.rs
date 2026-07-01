// SPDX-License-Identifier: AGPL-3.0-only

//! Batched Q/K/V projection helpers (batch2 / batch3 / n>=4 tiled) for the
//! multi-seq decode path — split from qkv.rs to keep files <=500 LoC.

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// n=3 NVFP4 batched path.
    pub(super) fn ms_qkv_batch3(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
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

        for i in 0..3usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
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

    /// n=2 NVFP4 batched path.
    pub(super) fn ms_qkv_batch2(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
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

        for i in 0..2usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
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

    /// n>=4 NVFP4 batched path: tile the proven `batch3`/`batch2` GEMV kernels
    /// across the n sequences (greedy 3s, then a 2 or 1 remainder). Each tile
    /// reads the q/k/v projection weights ONCE for its 2-3 tokens, so n=8 costs
    /// 3 weight-matrix reads instead of 8 — amortizing attention-projection
    /// bandwidth at decode. Layout is byte-identical to `ms_qkv_batch3` (same
    /// kernels, same q_proj_bytes/kv_bytes/per_seq_qkv offsets), so it inherits
    /// that path's correctness; the only new logic is the tiling loop + base
    /// offsets. The k==1 remainder falls back to the per-seq projection.
    pub(super) fn ms_qkv_batch_tiled(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
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
}
