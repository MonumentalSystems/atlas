// SPDX-License-Identifier: AGPL-3.0-only

//! Eligibility gating for the Q12 Path B kernel-batched prefill.
//!
//! Extracted from `batch_kernel.rs` to keep each file under the 500-LoC
//! file-size cap. Holds the env-flag predicates (`first_chunk_batched_enabled`,
//! `varlen_prefill_enabled`), the pure-data eligibility check
//! (`check_kernel_batched_eligible`, unit-tested in `batch_kernel_tests.rs`),
//! and the `TransformerModel::kernel_batched_eligible` wrapper the dispatcher
//! calls upfront.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use super::super::super::super::types::TransformerModel;
use crate::traits::PrefillSlice;
use spark_runtime::prefix_cache::PrefixMatch;

impl TransformerModel {
    /// Returns true when the batched-kernel path is viable for these
    /// streams. Cheap upfront check — caller (dispatch) falls back to
    /// per-stream when false.
    pub(in crate::model) fn kernel_batched_eligible(&self, streams: &[PrefillSlice<'_>]) -> bool {
        // Routed-MoE metadata still assumes a uniform per-stream layout.
        // Keep the experimental ragged path on dense models until its MoE
        // indexing is made cu_seqlens-aware; otherwise C=4 heterogeneous
        // Laguna traffic can reach a CUDA illegal access after admission.
        let varlen = varlen_prefill_enabled() && self.config.num_experts == 0;
        check_kernel_batched_eligible(
            streams
                .iter()
                .map(|s| (s.chunk_len, s.chunk_start, s.is_last_chunk)),
            streams.len(),
            self.buffers.max_batch_tokens(),
            &self.config.model_type,
            self.config.head_dim,
            self.buffers.scratch_bytes(),
            self.config.num_experts_per_tok,
            self.config.mrope_interleaved,
            // VARLEN v1 batches chunk-0 (fresh K/V) through FlashInfer ragged.
            crate::layers::ops::prefill_batched_first_chunk_enabled() || varlen,
            varlen,
        )
    }
}

/// Whether reserved prefix matches can share one stacked attention forward.
///
/// This is deliberately narrower than the single-stream prefix-cache path.
/// A batched cache hit is admitted only when every request has identical
/// processing geometry and needs neither an SSM restore nor disk-cache work.
/// The caller owns the reservation/rollback protocol; this predicate is pure
/// so the safety envelope remains unit-testable.
pub(in crate::model) fn cache_batch_matches_compatible(
    matches: &[PrefixMatch],
    chunk_len: usize,
) -> bool {
    let Some(first) = matches.first() else {
        return false;
    };
    let matched = first.matched_tokens;
    // Full-chunk hits use the single-token logits/early-return special cases;
    // keep those on the established sequential path in v1.
    if matched >= chunk_len {
        return false;
    }
    matches.iter().all(|m| {
        m.matched_tokens == matched
            && m.matched_blocks.len() == first.matched_blocks.len()
            && m.matched_disk_block_ids.is_empty()
            && m.ssm_snapshot.is_none()
            && m.ssm_snapshot_tokens == 0
            && m.ssm_snapshot_tier_key.is_none()
            && m.ssm_snapshot_tier_tokens == 0
    })
}

impl TransformerModel {
    /// DIAG: detect cross-stream physical-block sharing (co-dispatch KV
    /// double-issue hypothesis for the n>=5 decode-bleed bug). Gated behind
    /// `ATLAS_CODISPATCH_BTCHECK=1`; no-op otherwise.
    pub(super) fn codispatch_btcheck(&self, streams: &[PrefillSlice<'_>], n: usize) {
        if std::env::var("ATLAS_CODISPATCH_BTCHECK").ok().as_deref() != Some("1") {
            return;
        }
        let mut owner: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut slot_owner: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut dump: Vec<(usize, usize, Option<usize>, usize, u32)> = Vec::new();
        for (b, slice) in streams.iter().enumerate() {
            let bt = slice.seq.block_table.clone();
            let slot = slice.seq.slot_idx;
            // Authoritative owned slot from the RAII guard (slot_idx may be
            // stale post-compaction); plus prompt length + first token to
            // prove two DIFFERENT prompts share a slot.
            let guard_slot = slice.seq.ssm_slot.as_ref().and_then(|g| g.idx());
            let ptoks = slice.prompt_tokens.len();
            let tok0 = slice.prompt_tokens.first().copied().unwrap_or(0);
            if let Some(gs) = guard_slot {
                if let Some(&prev) = slot_owner.get(&gs) {
                    tracing::warn!(
                        "ATLAS_GUARDSHARE n={n}: GUARD slot {gs} SHARED by stream {prev} and {b}"
                    );
                } else {
                    slot_owner.insert(gs, b);
                }
            }
            for &blk in &bt {
                if let Some(&prev) = owner.get(&blk) {
                    tracing::warn!(
                        "ATLAS_BTSHARE n={n}: KV block {blk} SHARED by stream {prev} and {b}"
                    );
                } else {
                    owner.insert(blk, b);
                }
            }
            dump.push((b, slot, guard_slot, ptoks, tok0));
        }
        tracing::warn!("ATLAS_BTDUMP n={n} (stream,slot_idx,guard_slot,ptoks,tok0): {dump:?}");
    }
}

/// VARLEN batched prefill enabled? (`ATLAS_PREFILL_VARLEN=1`). Co-admits
/// varied-length concurrent prefills into one forward (cu_seqlens geometry,
/// FlashInfer ragged attention). Requires a FLASHINFER_HOME build.
pub(in crate::model) fn varlen_prefill_enabled() -> bool {
    std::env::var("ATLAS_PREFILL_VARLEN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Pure-data predicate extracted from [`TransformerModel::kernel_batched_eligible`]
/// so the gating rules are unit-testable without a real `TransformerModel`.
/// Caller materialises per-stream tuples `(chunk_len, chunk_start, is_last_chunk)`.
#[allow(clippy::too_many_arguments)]
pub(in crate::model) fn check_kernel_batched_eligible<I>(
    streams: I,
    n: usize,
    arena_cap: usize,
    model_type: &str,
    head_dim: usize,
    scratch_cap: usize,
    top_k: usize,
    mrope: bool,
    allow_chunk_zero: bool,
    varlen: bool,
) -> bool
where
    I: IntoIterator<Item = (usize, usize, bool)>,
{
    if n < 2 {
        return false;
    }
    // No MLA layers in stack (batched attention doesn't support MLA).
    // Conservatively check via model_type — mistral is the only MLA
    // model in Atlas today.
    if model_type == "mistral" {
        return false;
    }
    // No HDIM=512 layers (Gemma-4 long-attention).
    if head_dim > 256 {
        return false;
    }
    let mut first: Option<(usize, usize, bool)> = None;
    let mut total = 0usize;
    let mut max_chunk_len = 0usize;
    for (chunk_len, chunk_start, is_last) in streams {
        // `chunk_start` and `is_last_chunk` must match across streams (different
        // `chunk_start` → different `effective_seq_len_start`; mixing `is_last`
        // can't dispatch finalize_last + save_checkpoint together). `chunk_len`
        // must ALSO match in the legacy path; the VARLEN path allows differing
        // lengths (cu_seqlens geometry + FlashInfer ragged attention).
        match first {
            None => first = Some((chunk_len, chunk_start, is_last)),
            Some((cl, cs, il)) => {
                if (!varlen && chunk_len != cl) || chunk_start != cs || is_last != il {
                    return false;
                }
            }
        }
        total += chunk_len;
        max_chunk_len = max_chunk_len.max(chunk_len);
    }
    let Some((_chunk_len, chunk_start, _)) = first else {
        return false;
    };
    // Batched attention is paged-only today; chunk 0 uses the non-paged
    // cache-skip path and must stay on the single-stream dispatcher.
    if chunk_start == 0 && !allow_chunk_zero {
        return false;
    }
    // Total stacked tokens fit in the token arena (hidden_states buffer).
    if total > arena_cap {
        return false;
    }
    // #110: the kernel-batched staging footprint must fit in scratch. PURE
    // pre-flight — runs before any stream mutation, so a false routes to the
    // per-stream path from a clean state (a mid-dispatch overrun would leave
    // streams dirty and the fallback would re-run setup → corruption).
    // VARLEN: size the scratch pre-flight by the worst-case per-stream length.
    let scratch_needed = if varlen {
        spark_runtime::buffers::q12_batched_scratch_bytes_varlen(
            n,
            total,
            max_chunk_len,
            top_k,
            mrope,
        )
    } else {
        spark_runtime::buffers::q12_batched_scratch_bytes(n, max_chunk_len, top_k, mrope)
    };
    scratch_needed <= scratch_cap
}
