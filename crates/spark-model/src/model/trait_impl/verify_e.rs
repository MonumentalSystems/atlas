// SPDX-License-Identifier: AGPL-3.0-only

//! C×4 speculative verification through the Q12 batched continuation path.

use anyhow::{Result, bail, ensure};

use super::super::types::TransformerModel;
use crate::traits::{Model, PrefillSlice, SequenceState};

impl TransformerModel {
    pub(super) fn decode_verify_batch_k4_dispatch(
        &self,
        candidates: &[[u32; 4]],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<Vec<[u32; 4]>> {
        ensure!(
            candidates.len() == seqs.len(),
            "verify batch token/state mismatch"
        );
        if candidates.len() < 2 || self.multi_rank_protocol_active() {
            return Self::decode_verify_k4_serial(self, candidates, seqs, stream);
        }
        if seqs.iter().any(|seq| seq.tokens.len() != seq.seq_len) {
            bail!("batched verify requires token history to match seq_len");
        }
        for seq in seqs.iter_mut() {
            self.pre_verify_copy_async_dispatch(seq)?;
        }

        let mut histories = Vec::with_capacity(candidates.len());
        for (seq, candidate) in seqs.iter().zip(candidates) {
            let mut tokens = seq.tokens.clone();
            tokens.extend_from_slice(candidate);
            histories.push(tokens);
        }
        let mut slices: Vec<PrefillSlice<'_>> = histories
            .iter()
            .zip(seqs.iter_mut())
            .map(|(tokens, seq)| PrefillSlice {
                prompt_tokens: tokens,
                chunk_start: seq.seq_len,
                chunk_len: 4,
                is_last_chunk: true,
                logits_rows: 4,
                seq: &mut **seq,
            })
            .collect();
        if !self.kernel_batched_eligible(&slices) {
            tracing::info!(
                lanes = candidates.len(),
                seq_len = slices[0].chunk_start,
                marconi_skips = ?slices.iter().map(|s| s.seq.marconi_skip_to).collect::<Vec<_>>(),
                "C×4 verifier ineligible; falling back to serial K4"
            );
            drop(slices);
            return Self::decode_verify_k4_serial(self, candidates, seqs, stream);
        }
        self.prefill_batch_chunk_kernel_batched(&mut slices, stream)?;
        tracing::info!(lanes = candidates.len(), "C×4 verifier completed");
        drop(slices);

        let ids =
            self.argmax_batch_dispatch(self.buffers.logits(), candidates.len() * 4, stream)?;
        Ok(ids
            .chunks_exact(4)
            .map(|ids| [ids[0], ids[1], ids[2], ids[3]])
            .collect())
    }

    fn decode_verify_k4_serial(
        &self,
        candidates: &[[u32; 4]],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<Vec<[u32; 4]>> {
        candidates
            .iter()
            .zip(seqs.iter_mut())
            .map(|(tokens, seq)| self.decode_verify_graphed_k4_dispatch(tokens, seq, stream))
            .collect()
    }
}
