// SPDX-License-Identifier: AGPL-3.0-only

//! Batched verifier-logit finalization for Q12 continuation forwards.
//!
//! Unlike prompt finalization this intentionally has no prefix-cache or SSM
//! snapshot side effects: speculative verification has already advanced the
//! candidate state and needs logits for every candidate row before deciding
//! which state to commit.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::model) fn prefill_b_finalize_batch_stream(
        &self,
        tokens: &[u32],
        seq: &mut crate::traits::SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        hidden_offset_tokens: usize,
        logits_row: usize,
        logits_rows: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        if !is_last_chunk {
            self.prefill_b_save_checkpoint(tokens, seq, kv_cache, chunk_start, chunk_len, stream)?;
            return Ok(DevicePtr::NULL);
        }
        if logits_rows == 1 {
            return self.prefill_b_finalize_last_at(
                tokens,
                seq,
                kv_cache,
                chunk_start,
                chunk_len,
                proc_count,
                hidden_offset_tokens,
                logits_row,
                stream,
            );
        }
        anyhow::ensure!(
            logits_rows <= proc_count,
            "verifier requested {logits_rows} logits from {proc_count} computed rows"
        );
        self.prefill_b_finalize_rows_at(
            hidden_offset_tokens + proc_count - logits_rows,
            logits_rows,
            logits_row,
            stream,
        )
    }

    /// Final-norm and project `rows` contiguous hidden rows from the Q12 packed
    /// buffer. `logits_row` selects a non-overlapping region of the shared
    /// logits arena and the returned pointer is its first row.
    pub(in crate::model) fn prefill_b_finalize_rows_at(
        &self,
        hidden_offset_tokens: usize,
        rows: usize,
        logits_row: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        assert!(rows > 0, "verifier finalization requires at least one row");
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let hidden = self
            .buffers
            .hidden_states()
            .offset(hidden_offset_tokens * h * bf16);
        let normed = self.buffers.norm_output();
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            rows as u32,
            h as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )?;
        let logits = self
            .buffers
            .logits()
            .offset(logits_row * self.config.vocab_size * bf16);
        self.lm_head_batched(normed, rows as u32, logits, stream)
    }
}
