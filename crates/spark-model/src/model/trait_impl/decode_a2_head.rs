// SPDX-License-Identifier: AGPL-3.0-only

//! `TransformerModel::decode_final_norm_and_head` — final RMS norm + LM-head
//! GEMM (FP8 / NVFP4 / dense) + the `ATLAS_MS_PROFILE` emit, hoisted out of
//! `decode_a2.rs` to keep that file under the 500 LoC cap. Invoked at exactly
//! one site inside the decode graph-capture region, so the kernel launch order
//! (critical for CUDA-graph capture correctness) is byte-for-byte unchanged.

#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    /// Final norm over the `[eff_n, H]` decode hidden state, then the LM-head
    /// projection into `logits`, then (when `lmhead_t0` is set) the per-step
    /// `ATLAS_MS_PROFILE` component-timing line. `ssm_us`/`attn_us` are the
    /// accumulated layer timings from the caller's decode loop.
    pub(super) fn decode_final_norm_and_head(
        &self,
        hidden: DevicePtr,
        n: usize,
        padded_n: usize,
        eff_n: usize,
        h: usize,
        bf16: usize,
        ssm_us: u128,
        attn_us: u128,
        lmhead_t0: Option<std::time::Instant>,
        stream: u64,
    ) -> Result<()> {
        // Final norm [eff_n, H]
        let normed = self.buffers.norm_output();
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            eff_n as u32,
            h as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )?;

        // LM head: ONE batched [eff_n, vocab] GEMM so the ~254 MB
        // vocab weight is read ONCE per step instead of once per sequence
        // (the per-row GEMV loop re-read it N times — a major C>=2 cost:
        // ~N×254 MB/step). nvfp4/dense are batched here; FP8 single-scale
        // keeps the per-row path (no batched single-scale FP8 GEMM handle
        // on the model, and Holo's lm_head is NVFP4 anyway).
        let logits = self.buffers.logits();
        let v = self.config.vocab_size;
        if let Some(ref fp8) = self.lm_head_fp8 {
            for i in 0..eff_n {
                ops::dense_gemv_fp8w(
                    self.gpu.as_ref(),
                    self.dense_gemv_fp8w_kernel,
                    normed.offset(i * h * bf16),
                    fp8,
                    logits.offset(i * v * bf16),
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            // Batched M<=16 GEMV reads the ~254 MB vocab weight ONCE for all
            // rows; the tiled w4a16_gemm re-tiles it (14% of the C=8 step).
            // Opt-in: reduction order differs from the tiled MMA → ULP logit
            // shift (greedy re-baselined + soaked before default-ON).
            if self.lmhead_batch_gemv && eff_n <= 16 && self.w4a16_gemv_batch16_kernel.0 != 0 {
                ops::w4a16_gemv_batchm(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_batch16_kernel,
                    normed,
                    nvfp4,
                    logits,
                    eff_n as u32,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_kernel,
                    normed,
                    nvfp4,
                    logits,
                    eff_n as u32,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else {
            ops::dense_gemm(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                normed,
                &self.lm_head_weight,
                logits,
                eff_n as u32,
                v as u32,
                h as u32,
                stream,
            )?;
        }
        if let Some(t0) = lmhead_t0 {
            self.gpu.synchronize(stream).ok();
            let head_us = t0.elapsed().as_micros();
            let total = ssm_us + attn_us + head_us;
            tracing::info!(
                "ATLAS_MS_PROFILE n={n} padded_n={padded_n} eff_n={eff_n}: total={}us  ssm={}us({}L)  attn={}us({}L)  head={}us  [per-tok {:.2}ms]",
                total,
                ssm_us,
                self.config.num_ssm_layers(),
                attn_us,
                self.layers.len() - self.config.num_ssm_layers(),
                head_us,
                total as f64 / 1000.0 / eff_n as f64,
            );
        }
        Ok(())
    }
}
