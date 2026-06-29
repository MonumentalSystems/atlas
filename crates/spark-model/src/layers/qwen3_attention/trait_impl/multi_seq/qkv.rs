// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2: per-token Q/K/V projection. Three branches:
//! - n=3 + NVFP4 → batch3 GEMV path
//! - n=2 + NVFP4 → batch2 GEMV path
//! - else        → sequential per-token GEMV (FP8/NVFP4/BF16 fallback)
//!
//! Both batch paths read each weight once for N tokens and then scatter
//! into the per-seq QKV layout. The sequential path repeats the GEMV per
//! token but supports every weight encoding.

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
            for i in 0..n {
                let normed_i = normed.offset(i * h * bf16);
                let q_out_i = qkv_buf.offset(i * per_seq_qkv);
                let k_out_i = q_out_i.offset(q_proj_bytes);
                let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);

                self.ms_qkv_seq_q(fwd, normed_i, q_out_i, q_proj_dim, q_dim, nq, hd, h, stream)?;
                self.ms_qkv_seq_kv(fwd, normed_i, k_out_i, v_out_i, nkv, hd, h, stream)?;

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
        }
        Ok(())
    }
}
