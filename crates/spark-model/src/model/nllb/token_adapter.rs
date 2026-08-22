// SPDX-License-Identifier: AGPL-3.0-only

//! PEFT `trainable_tokens` / `token_adapter` support for the served NLLB model
//! — the piece that makes a *vocab-extending* adapter actually change the token
//! embeddings (new low-resource-language token + a mode token) instead of being
//! silently dropped by the A/B filter.
//!
//! A vocab-extending adapter ships, per tied embedding, a `token_adapter`:
//! `base_layer.weight` `[R,d]` (the adapter's OWN resized base embedding) and
//! `trainable_tokens_delta` `[T,d]` — the **full replacement rows** for the `T`
//! trainable ids (`trainable_token_indices` in `adapter_config.json`). This is
//! PEFT's `index_copy` semantics (`peft/tuners/trainable_tokens/layer.py`
//! `get_merged_weights`): `E[idx[k]] = delta[k]`, NOT `base_layer[idx[k]] +
//! delta[k]`. The delta was initialised FROM the base row and trained in place,
//! so it already carries the whole row.
//!
//! Rather than swap the full `[R,d]` (~512 MB) embedding per adapter, we build a
//! compact [`EmbedOverlay`]: only the rows that differ from the *served* embed
//! table (found by a load-time row-diff) plus the trainable rows. For the Kuku
//! Yalanji adapter that is 1–2 rows. This keeps per-adapter resident cost at the
//! `slot_map` (`vocab·4` B) plus a few KB of override rows, which is what makes
//! the future many-`xxx_Latn`-adapter vision affordable. The four big
//! `token_adapter` device buffers (~1 GB) are freed once the overlay is built.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::WeightStore;

use super::util::{i32_bytes, u32_bytes};

/// PEFT `token_adapter` tensor keys (raw safetensors names). Only `model.shared`
/// is materialised; `lm_head` is tied (byte-identical) and unused beyond an
/// optional sanity check, but is still freed to reclaim its ~512 MB.
const SHARED_BASE: &str = "base_model.model.model.shared.token_adapter.base_layer.weight";
const SHARED_DELTA: &str = "base_model.model.model.shared.token_adapter.trainable_tokens_delta";
const LMHEAD_BASE: &str = "base_model.model.lm_head.token_adapter.base_layer.weight";
const LMHEAD_DELTA: &str = "base_model.model.lm_head.token_adapter.trainable_tokens_delta";

/// Row-diff threshold: `max|base_layer[r]-embed[r]|` above this marks row `r` as
/// an override. Clears bf16 rounding noise (matching rows measure ≤0.05; a real
/// differing token row is ≥1.3).
const ROWDIFF_THRESH: f32 = 0.1;

/// A compact embedding overlay: the effective rows for the handful of token ids
/// whose embedding the active adapter changes, plus the addressing needed to
/// apply them at input-embed (`slot_map`) and lm_head (`ids_dev`) time.
pub(super) struct EmbedOverlay {
    /// `[n_override, d]` bf16 — the effective (replacement) embedding rows.
    pub(super) rows: DevicePtr,
    /// `u32[n_override]` — the overridden token ids, ascending (lm_head scatter).
    pub(super) ids_dev: DevicePtr,
    /// `i32[vocab]` — token id -> row index in `rows`, else -1 (embed lookup).
    pub(super) slot_map: DevicePtr,
    pub(super) n_override: u32,
}

/// Union of the differing-row ids and the trainable ids, ascending + deduped.
/// Pure (host) — the override-set math, unit-tested without a GPU.
pub(super) fn build_override_set(row_diff: &[bool], trainable: &[u32]) -> Vec<u32> {
    let mut ids: Vec<u32> = row_diff
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| d.then_some(i as u32))
        .collect();
    ids.extend_from_slice(trainable);
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Source of the effective row for override `id`: `Some(k)` -> replacement row
/// `delta[k]` (id is trainable, position `k`); `None` -> `base_layer[id]` (a
/// baked non-trainable mismatch, e.g. the gvn_Latn language token). Delta wins
/// when an id is both trainable and a mismatch. Pure — unit-tested.
pub(super) fn override_source(id: u32, trainable: &[u32]) -> Option<usize> {
    trainable.iter().position(|&t| t == id)
}

/// Keep only the trainable indices addressable in the served `vocab`, returning
/// `(kept, skipped_extension_count)`. Indices in `[vocab, r)` are vocab-extension
/// tokens (a resized adapter served on a smaller base) — dropped so the overlay
/// stays inside the served embedding. PEFT appends extension tokens as the
/// largest indices with delta rows in the same order, so dropping the tail
/// preserves the positional delta mapping for the kept ones; a non-tail-ordered
/// index (a kept index after a larger kept index) would break that alignment and
/// is a hard error. An index `>= r` is out of the adapter itself and also errors.
/// Pure — unit-tested.
pub(super) fn clamp_trainable_to_vocab(
    trainable: &[u32],
    r: usize,
    vocab: usize,
) -> Result<(Vec<u32>, usize)> {
    let mut kept: Vec<u32> = Vec::with_capacity(trainable.len());
    let mut skipped = 0usize;
    let mut max_kept: i64 = -1;
    for &idx in trainable {
        let u = idx as usize;
        if u >= r {
            bail!("NLLB token_adapter: trainable index {idx} out of range (R={r})");
        } else if u >= vocab {
            skipped += 1;
        } else {
            if (idx as i64) < max_kept {
                bail!(
                    "NLLB token_adapter: trainable_token_indices not tail-ordered \
                     ({idx} after {max_kept}); cannot align delta rows after clamping"
                );
            }
            max_kept = idx as i64;
            kept.push(idx);
        }
    }
    Ok((kept, skipped))
}

/// Host mirror of the row-diff kernel: `max|a[r]-b[r]| > thresh`. Used by the
/// CPU tests (and documents the exact device predicate).
#[cfg(test)]
pub(super) fn row_diff_flags(a: &[f32], b: &[f32], rows: usize, d: usize, thresh: f32) -> Vec<bool> {
    (0..rows)
        .map(|r| {
            (0..d)
                .map(|i| (a[r * d + i] - b[r * d + i]).abs())
                .fold(0.0f32, f32::max)
                > thresh
        })
        .collect()
}

/// Host mirror of the overlay materialisation: the effective row for override
/// `id`. Pure — the CPU test asserts full-replacement (delta) vs baked
/// (base_layer) selection on synthetic rows.
#[cfg(test)]
pub(super) fn effective_row<'a>(
    id: u32,
    trainable: &[u32],
    base_layer: &'a [f32],
    delta: &'a [f32],
    d: usize,
) -> &'a [f32] {
    match override_source(id, trainable) {
        Some(k) => &delta[k * d..(k + 1) * d],
        None => &base_layer[id as usize * d..(id as usize + 1) * d],
    }
}

/// Build the embedding overlay from an already-loaded adapter `store`, or
/// `Ok(None)` when the adapter carries no `token_adapter.*` tensors (standard
/// A/B-only adapters — fully backward compatible). Materialises the compact
/// override rows on device and frees the ~1 GB of raw `token_adapter` buffers.
///
/// `embed_table` is the served tied embedding `[vocab, d]` bf16; `vocab`/`d` are
/// the served dims. Correct on EITHER base (vanilla or the merged v21.2): the
/// load-time row-diff overrides exactly the rows that differ from whatever base
/// is served, plus the trainable replacement rows.
pub(super) fn build_overlay(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    embed_table: DevicePtr,
    vocab: usize,
    d: usize,
    trainable: &[u32],
) -> Result<Option<EmbedOverlay>> {
    if !store.contains(SHARED_BASE) {
        return Ok(None); // A/B-only adapter: no vocab extension.
    }
    let base = store.get(SHARED_BASE)?;
    let delta = store.get(SHARED_DELTA)?;
    if base.shape.len() != 2 || base.shape[1] != d {
        bail!(
            "NLLB token_adapter: base_layer shape {:?} incompatible with served d={d}",
            base.shape
        );
    }
    let r = base.shape[0];
    let t = delta.shape[0];
    if delta.shape.len() != 2 || delta.shape[1] != d {
        bail!(
            "NLLB token_adapter: trainable_tokens_delta shape {:?} incompatible with d={d}",
            delta.shape
        );
    }
    // Rows beyond the served vocab are vocab-extension tokens (e.g. a new
    // `<lexeme>` control token appended past the base). The served tokenizer
    // cannot emit them and the lm_head gemv N / embed addressing are bounded by
    // `vocab`, so they can never participate in this deployment. Clamp the
    // overlay to the served vocab and report exactly what is skipped — never
    // silently drop coverage. To use the extension tokens, serve the resized
    // tokenizer/config so `vocab` grows to cover them.
    let r_eff = r.min(vocab);
    if r > vocab {
        tracing::warn!(
            "NLLB token_adapter: adapter embedding {r} > served vocab {vocab}; \
             {} vocab-extension row(s) beyond the served vocab are skipped \
             (serve the resized tokenizer/config to use them)",
            r - vocab
        );
    }
    // Keep only trainable indices addressable in the served vocab; the (tail)
    // extension indices and their delta rows are dropped together.
    let (trainable_kept, skipped_ext) = clamp_trainable_to_vocab(trainable, r, vocab)?;
    if skipped_ext > 0 {
        tracing::warn!(
            "NLLB token_adapter: {skipped_ext} trainable token(s) address extension rows \
             beyond served vocab {vocab} — skipped"
        );
    }
    let trainable: &[u32] = &trainable_kept;
    if trainable.len() != t && skipped_ext == 0 {
        // Not fatal: the delta rows map positionally to trainable_token_indices.
        tracing::warn!(
            "NLLB token_adapter: {t} delta rows but {} trainable indices — using min",
            trainable.len()
        );
    }

    // 1) differing rows over the served vocab only: max|base_layer[r]-embed[r]| > thresh.
    let flags_dev = gpu.alloc(r_eff)?;
    let rowdiff = gpu.kernel("nllb_encoder", "nllb_embed_rowdiff_bf16")?;
    KernelLaunch::new(gpu, rowdiff)
        .grid([div_ceil(r_eff as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(base.ptr)
        .arg_ptr(embed_table)
        .arg_ptr(flags_dev)
        .arg_u32(r_eff as u32)
        .arg_u32(d as u32)
        .arg_f32(ROWDIFF_THRESH)
        .launch(gpu.default_stream())?;
    gpu.synchronize(gpu.default_stream())?;
    let mut flags = vec![0u8; r_eff];
    gpu.copy_d2h(flags_dev, &mut flags)?;
    gpu.free(flags_dev)?;
    let row_diff: Vec<bool> = flags.iter().map(|&f| f != 0).collect();

    // 2) override id set = mismatches ∪ trainable.
    let ids = build_override_set(&row_diff, trainable);
    let n = ids.len();
    if n == 0 {
        // Adapter's base_layer matches the served embed and no trainable rows:
        // nothing to override. Free the big buffers and skip the overlay.
        free_token_adapter_buffers(store, gpu)?;
        return Ok(None);
    }

    // 3) materialise the compact [n,d] bf16 override rows.
    let rows = gpu.alloc(n * d * 2)?;
    for (slot, &id) in ids.iter().enumerate() {
        let dst = rows.offset(slot * d * 2);
        match override_source(id, trainable) {
            Some(k) => gpu.copy_d2d(delta.ptr.offset(k * d * 2), dst, d * 2)?,
            None => gpu.copy_d2d(base.ptr.offset(id as usize * d * 2), dst, d * 2)?,
        }
    }

    // 4) slot_map[vocab] (id -> row, else -1) and ascending ids_dev[n].
    let mut slot_map_host = vec![-1i32; vocab];
    for (slot, &id) in ids.iter().enumerate() {
        slot_map_host[id as usize] = slot as i32;
    }
    let slot_map = gpu.alloc(vocab * 4)?;
    gpu.copy_h2d(i32_bytes(&slot_map_host), slot_map)?;
    let ids_dev = gpu.alloc(n * 4)?;
    gpu.copy_h2d(u32_bytes(&ids), ids_dev)?;

    // 5) reclaim the ~1 GB of raw token_adapter buffers (base_layer×2 + delta×2)
    //    — nothing references them past this point (they are not in `pairs`).
    free_token_adapter_buffers(store, gpu)?;

    let baked = n - trainable.len().min(n);
    tracing::info!("NLLB token overlay: n_override={n} (trainable={}, baked={baked})", trainable.len());
    Ok(Some(EmbedOverlay {
        rows,
        ids_dev,
        slot_map,
        n_override: n as u32,
    }))
}

/// Free the four large `token_adapter` device buffers (shared + lm_head, each a
/// `base_layer` + `delta`). Safe: `WeightStore` has no `Drop` (Atlas never frees
/// weights) and nothing else references these ptrs after the overlay is built.
fn free_token_adapter_buffers(store: &WeightStore, gpu: &dyn GpuBackend) -> Result<()> {
    for key in [SHARED_BASE, SHARED_DELTA, LMHEAD_BASE, LMHEAD_DELTA] {
        if let Ok(t) = store.get(key) {
            gpu.free(t.ptr)?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "token_adapter_tests.rs"]
mod tests;
