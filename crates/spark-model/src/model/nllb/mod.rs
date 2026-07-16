// SPDX-License-Identifier: AGPL-3.0-only

//! Served NLLB-200 / M2M-100 encoder-decoder translation model.
//!
//! Atlas's production engine is decoder-only + GPU-only; NLLB is seq2seq
//! (bidirectional encoder + decoder cross-attention + sinusoidal positions +
//! ReLU FFN + biased LayerNorm). This module promotes the validated bf16 CUDA
//! runtime (`examples/nllb_cuda_bf16`) into a first-class [`crate::traits::Model`]
//! so `spark serve --model <nllb-dir>` translates through the SAME scheduler,
//! OpenAI API, sampling and (later) LoRA path as every other model — no parallel
//! server, no hardcoded paths (weights come from the standard `--model` store).
//!
//! The model owns ALL its KV (the scheduler's paged block cache is unused): a
//! per-sequence decoder self-attn cache that grows one row per token plus a
//! fixed cross-attn cache computed once from the encoder. See the `kv` module. Logits are
//! bf16 — the scheduler's default sampling path — so no fp32 overrides are
//! needed.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

mod beam;
mod beam_compute;
mod beam_multi;
mod compute;
mod kernels;
mod kv;
mod lang;
mod lora;
mod model_impl;
mod token_adapter;
mod token_overlay;
mod util;

pub use lang::NllbLang;

use compute::DecScratch;
use kernels::NllbKernels;
use kv::NllbSeqKv;
use lora::NllbLora;

/// Decoder self-attn KV cache depth (max generated tokens per translation).
/// NLLB `max_length` defaults to 200; 512 is a comfortable cap that keeps
/// per-sequence KV small (`cache_rows·d·2·dec_layers·2` bytes).
const DEFAULT_CACHE_ROWS: usize = 512;

/// The served NLLB encoder-decoder model. `Send + Sync`: every field is either
/// immutable after construction or behind a `Mutex`; `DevicePtr` is a `Copy`
/// device handle and the GPU-side buffers are driven only through `&self`.
pub struct NllbGpuModel {
    gpu: Box<dyn GpuBackend>,
    kernels: NllbKernels,
    weights: HashMap<String, DevicePtr>,
    embed_table: DevicePtr,
    // dims
    d: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    enc_layers: usize,
    dec_layers: usize,
    vocab: usize,
    embed_scale: f32,
    attn_scale: f32,
    cache_rows: usize,
    max_batch: usize,
    lang: NllbLang,
    // persistent device scratch (single-token decode) + outputs
    dec: DecScratch,
    /// bf16 decode logits `[max_batch, vocab]` — `decode_batch` writes CONTIGUOUS
    /// rows `0..n` (batch position `i` ↔ `seqs[i]`), the scheduler's contract.
    decode_logits: DevicePtr,
    /// bf16 prefill logits `[max_batch, vocab]` — each concurrent prefill writes
    /// its OWN row (indexed by `slot_idx`) so overlapping prefills don't collide.
    prefill_logits: DevicePtr,
    /// Decoder sinusoidal position table `[cache_rows, d]` bf16.
    pos_table: DevicePtr,
    // per-sequence KV + slot allocator
    kv: Mutex<HashMap<usize, NllbSeqKv>>,
    slots: Mutex<SlotAlloc>,
    /// Optional PEFT LoRA adapter applied on every projection (runtime delta).
    lora: Option<NllbLora>,
    /// Per-request LoRA gate: set from `SequenceState.adapter_slot` before each
    /// sequence's forward (`>=0` → apply the adapter, `-1` → base). Sound because
    /// the scheduler drives prefill/decode serially on one thread.
    lora_active: AtomicBool,
}

/// Trivial monotonic slot allocator with a free-list for reuse.
#[derive(Default)]
struct SlotAlloc {
    next: usize,
    free: Vec<usize>,
}

impl SlotAlloc {
    fn claim(&mut self) -> usize {
        self.free.pop().unwrap_or_else(|| {
            let s = self.next;
            self.next += 1;
            s
        })
    }
    fn release(&mut self, slot: usize) {
        self.free.push(slot);
    }
    /// Remove `slot` from the free-list so it is owned exclusively by the caller
    /// (the two-phase retire compaction migrates a survivor onto a slot freed by
    /// a retiring sequence — that target is on the free-list and MUST be claimed
    /// or a later `claim` would hand the same slot to a new sequence). Also keep
    /// the monotonic `next` ahead of any claimed index so it is never re-minted.
    /// Returns whether the slot was actually on the free-list.
    fn claim_specific(&mut self, slot: usize) -> bool {
        let found = self.free.iter().position(|&s| s == slot);
        if let Some(pos) = found {
            self.free.swap_remove(pos);
        }
        if slot >= self.next {
            self.next = slot + 1;
        }
        found.is_some()
    }
}

impl NllbGpuModel {
    /// Build from the standard `--model` weight store + GPU backend. `lang`
    /// carries the tokenizer-resolved source/target language ids (resolved
    /// server-side, where the tokenizer lives). `max_seq_len` caps the decoder
    /// KV depth.
    pub fn new(
        config: &ModelConfig,
        store: &WeightStore,
        gpu: Box<dyn GpuBackend>,
        lang: NllbLang,
        max_seq_len: usize,
        max_batch: usize,
        lora_dir: Option<&std::path::Path>,
    ) -> Result<Self> {
        let d = config.hidden_size;
        let heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let ffn = config.intermediate_size;
        let dec_layers = config.num_hidden_layers;
        // NLLB / M2M-100 are architecturally symmetric: encoder and decoder
        // share layer count, head count and FFN width.
        let enc_layers = dec_layers;
        let vocab = config.vocab_size;
        // `scale_embedding` is `true` for the NLLB family (√d_model embed scale).
        let embed_scale = (d as f32).sqrt();
        let attn_scale = (head_dim as f32).powf(-0.5);
        let cache_rows = DEFAULT_CACHE_ROWS.max(max_seq_len.min(2048));

        ensure!(
            store.get("model.shared.weight")?.dtype == WeightDtype::BF16,
            "nllb serving requires a bf16 checkpoint; convert with \
             scripts/convert-safetensors-to-bf16.py"
        );
        let weights: HashMap<String, DevicePtr> = store
            .names()
            .map(|n| Ok((n.to_string(), store.get(n)?.ptr)))
            .collect::<Result<_>>()?;
        let embed_table = *weights
            .get("model.shared.weight")
            .context("nllb: missing tied embedding model.shared.weight")?;

        // Make the CUDA context current on the construction thread before any
        // kernel resolve / device alloc.
        gpu.bind_to_thread()?;
        let kernels = NllbKernels::new(gpu.as_ref())?;
        let dec = DecScratch::new(gpu.as_ref(), d, ffn, vocab)?;
        let max_batch = max_batch.max(1);
        let decode_logits = gpu.alloc(max_batch * vocab * 2)?;
        let prefill_logits = gpu.alloc(max_batch * vocab * 2)?;
        let pos_table = gpu.alloc(cache_rows * d * 2)?;
        let pos_host = util::decoder_pos_table_bf16(cache_rows, d);
        gpu.copy_h2d(util::bf16_bytes(&pos_host), pos_table)?;

        let lora = match lora_dir {
            Some(dir) => Some(NllbLora::load(
                dir,
                gpu.as_ref(),
                cache_rows,
                embed_table,
                vocab,
                d,
            )?),
            None => None,
        };

        tracing::info!(
            "NLLB served model ready: d={d} heads={heads} enc={enc_layers} dec={dec_layers} \
             vocab={vocab} src_lang_id={} tgt_lang_id={} cache_rows={cache_rows}",
            lang.src_lang_id,
            lang.tgt_lang_id,
        );

        Ok(Self {
            gpu,
            kernels,
            weights,
            embed_table,
            d,
            heads,
            head_dim,
            ffn,
            enc_layers,
            dec_layers,
            vocab,
            embed_scale,
            attn_scale,
            cache_rows,
            max_batch,
            lang,
            dec,
            decode_logits,
            prefill_logits,
            pos_table,
            kv: Mutex::new(HashMap::new()),
            slots: Mutex::new(SlotAlloc::default()),
            lora,
            lora_active: AtomicBool::new(false),
        })
    }

    /// Device pointer for weight `name` (panics if absent — construction
    /// resolves the full store, so a miss is a checkpoint/format bug).
    #[inline]
    pub(super) fn w(&self, name: &str) -> DevicePtr {
        self.weights[name]
    }

    /// Arm/disarm the LoRA delta for the sequence about to be forwarded
    /// (`adapter_slot >= 0` → apply). No-op when no adapter is loaded.
    #[inline]
    pub(super) fn set_lora_active(&self, adapter_slot: i32) {
        self.lora_active
            .store(self.lora.is_some() && adapter_slot >= 0, Ordering::Relaxed);
    }

    #[inline]
    pub(super) fn lora_is_active(&self) -> bool {
        self.lora_active.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod slot_tests {
    use super::SlotAlloc;

    /// `claim_specific` must remove a free slot so it is never handed out again —
    /// the exact invariant the retire-compaction relies on (a survivor migrated
    /// onto a freed slot must own it exclusively).
    #[test]
    fn claim_specific_removes_from_free_list() {
        let mut a = SlotAlloc::default();
        let (s0, s1, s2) = (a.claim(), a.claim(), a.claim()); // 0,1,2
        assert_eq!((s0, s1, s2), (0, 1, 2));
        a.release(1); // free-list: [1]
        assert!(a.claim_specific(1)); // migrate a survivor onto slot 1
        assert!(!a.claim_specific(1)); // already owned — no longer free
        // The next fresh claim must NOT reuse slot 1 (it is live now).
        assert_eq!(a.claim(), 3);
    }

    /// The compaction bookkeeping: release(old) + claim_specific(new) keeps the
    /// free-list consistent so no two live sequences ever share a slot.
    #[test]
    fn compaction_bookkeeping_keeps_slots_disjoint() {
        let mut a = SlotAlloc::default();
        for _ in 0..4 {
            a.claim();
        } // 0,1,2,3 live
        a.release(1); // seq at slot 1 retired → free-list [1]
        // Survivor at slot 3 compacts into the hole at slot 1.
        a.claim_specific(1); // claim target
        a.release(3); // release vacated source
        // Now live = {0,1,2}; a new sequence must get slot 3 (the only free one),
        // never slot 1 (owned by the migrated survivor).
        assert_eq!(a.claim(), 3);
    }

    /// `claim_specific` for a slot beyond `next` reserves the range so a later
    /// monotonic `claim` cannot re-mint it.
    #[test]
    fn claim_specific_beyond_next_advances_monotonic() {
        let mut a = SlotAlloc::default();
        assert!(!a.claim_specific(5)); // not on free-list, but reserve up to 5
        assert_eq!(a.claim(), 6);
    }
}
