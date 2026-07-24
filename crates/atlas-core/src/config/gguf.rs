// SPDX-License-Identifier: AGPL-3.0-only

//! Build a [`ModelConfig`] from GGUF file metadata.
//!
//! GGUF carries its model config inline as metadata key/values
//! (`llama.block_count`, `qwen3.attention.head_count`, …) rather than a
//! sibling `config.json`. This module reads those keys through the
//! [`GgufMeta`] accessor (implemented by the GGUF parser in spark-runtime, so
//! atlas-core keeps no GGUF dependency) and produces a validated
//! [`ModelConfig`] for the llama / qwen2 / qwen3 / gemma decoder families and
//! Laguna-S-2.1's heterogeneous full/sliding-attention MoE architecture.
//!
//! Strategy: synthesize an HF-config-shaped JSON object from the GGUF keys and
//! deserialize it into `ModelConfig` (serde `#[serde(default)]` fills the many
//! fields GGUF has no analog for), then set the architecture flags
//! (`attn_gated`, `weight_prefix`, gemma `embed_scale` /
//! `final_logit_softcapping`) explicitly, then run the shared
//! [`super::finalize_config`]. No silent production defaults: every value GGUF
//! omits is either derived by an explicit documented rule or is an error.

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};

use super::{ModelConfig, finalize_config, parse_laguna};

/// Typed read access to GGUF metadata. Implemented by the spark-runtime GGUF
/// parser over its parsed key/value table. All getters return `None` when the
/// key is absent or holds a different value type — the builder decides whether
/// absence is fatal or has a derivation rule.
pub trait GgufMeta {
    /// Any unsigned/signed integer metadata value, widened to u64.
    fn get_u64(&self, key: &str) -> Option<u64>;
    /// An array of non-negative integer metadata values, widened to u64.
    fn get_u64_array(&self, key: &str) -> Option<Vec<u64>>;
    /// Any float metadata value (f32/f64), widened to f64.
    fn get_f64(&self, key: &str) -> Option<f64>;
    /// A string metadata value.
    fn get_str(&self, key: &str) -> Option<&str>;
    /// Length of an array metadata value (e.g. `tokenizer.ggml.tokens`).
    fn get_arr_len(&self, key: &str) -> Option<usize>;
}

/// Inputs to [`config_from_gguf`]: the metadata accessor plus two facts the
/// builder needs from the tensor section (not the metadata KV block).
pub struct GgufConfigInputs<'a> {
    pub meta: &'a dyn GgufMeta,
    /// Rows of `token_embd.weight` — the authoritative vocab size when the
    /// `{arch}.vocab_size` key is absent. `None` if the loader could not read
    /// the tensor shape before building the config.
    pub token_embd_vocab: Option<usize>,
    /// Whether the file contains an `output.weight` tensor. Its presence means
    /// an untied LM head; its absence means the LM head ties to the input
    /// embeddings. GGUF has no explicit `tie_word_embeddings` key, so this is
    /// the only reliable signal.
    pub has_output_weight: bool,
}

/// Map a GGUF `general.architecture` string to an Atlas `model_type` (must be
/// a supported loader string) and whether attention Q is gated.
///
/// Plain-decoder GGUFs (llama/qwen2) have no dedicated Atlas arch loader; the
/// closest dense GQA path is the Mistral loader. qwen3 dense maps to `qwen3_5`
/// with `num_experts == 0` (dense qwen3.5 loader). Returns an error for
/// unmapped architectures rather than guessing.
fn arch_to_model_type(arch: &str) -> Result<(&'static str, bool)> {
    // (model_type, attn_gated)
    Ok(match arch {
        "llama" => ("mistral", false),
        // qwen2 ships QKV biases; the Mistral GQA path is the closest dense
        // loader. (Bias handling is a known caveat — see module notes.)
        "qwen2" => ("mistral", false),
        // qwen3 dense: q_norm/k_norm, ungated Q. num_experts==0 → dense loader.
        "qwen3" => ("qwen3_5", false),
        "qwen3moe" => ("qwen3_5_moe", false),
        // gemma family: GeGLU, ungated Q, embedding scale + logit softcap.
        "gemma" | "gemma2" | "gemma3" | "gemma4" => ("gemma4", false),
        other => bail!(
            "GGUF general.architecture '{other}' has no Atlas model_type mapping. \
             Supported GGUF arches: llama, qwen2, qwen3, qwen3moe, gemma/gemma2/gemma3/gemma4."
        ),
    })
}

/// Build a validated [`ModelConfig`] from GGUF metadata.
pub fn config_from_gguf(inputs: &GgufConfigInputs) -> Result<ModelConfig> {
    let meta = inputs.meta;

    let arch = meta
        .get_str("general.architecture")
        .context("GGUF metadata missing required key 'general.architecture'")?
        .to_string();
    if arch == "laguna" {
        return config_from_laguna_gguf(inputs);
    }
    let (model_type, attn_gated) = arch_to_model_type(&arch)?;

    // Namespaced key helper: `{arch}.<suffix>`.
    let k = |suffix: &str| format!("{arch}.{suffix}");
    let req_u64 = |suffix: &str| -> Result<u64> {
        meta.get_u64(&k(suffix))
            .with_context(|| format!("GGUF metadata missing required key '{arch}.{suffix}'"))
    };

    // ── Core dimensions (required) ──
    let hidden_size = req_u64("embedding_length")? as usize;
    let num_hidden_layers = req_u64("block_count")? as usize;
    let intermediate_size = req_u64("feed_forward_length")? as usize;
    let num_attention_heads = req_u64("attention.head_count")? as usize;

    // GQA: kv head count defaults to full MHA (== attention heads) when the key
    // is absent, which is the ggml convention.
    let num_key_value_heads = meta
        .get_u64(&k("attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(num_attention_heads);

    // head_dim: explicit key_length if present, else hidden_size / head_count.
    // Erroring on a non-divisible fallback avoids a silently-wrong head_dim.
    let head_dim = match meta.get_u64(&k("attention.key_length")) {
        Some(v) => v as usize,
        None => {
            if num_attention_heads == 0 || !hidden_size.is_multiple_of(num_attention_heads) {
                bail!(
                    "GGUF: cannot derive head_dim — '{arch}.attention.key_length' absent and \
                     hidden_size ({hidden_size}) not divisible by head_count ({num_attention_heads})"
                );
            }
            hidden_size / num_attention_heads
        }
    };

    // vocab_size: explicit key → token_embd rows → tokenizer token list length.
    let vocab_size = meta
        .get_u64(&k("vocab_size"))
        .map(|v| v as usize)
        .or(inputs.token_embd_vocab)
        .or_else(|| meta.get_arr_len("tokenizer.ggml.tokens"))
        .context(
            "GGUF: could not determine vocab_size (no '{arch}.vocab_size', no token_embd rows, \
             no 'tokenizer.ggml.tokens')",
        )?;

    // ── Normalization / RoPE / context (documented explicit defaults) ──
    // rms_norm_eps: ggml default is 1e-5 when the key is absent (differs from
    // Atlas's 1e-6 default — we set it explicitly rather than inherit).
    let rms_norm_eps = meta
        .get_f64(&k("attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);
    // rope_theta: ggml default 10000.0.
    let rope_theta = meta.get_f64(&k("rope.freq_base")).unwrap_or(10_000.0);
    // context_length is required for a usable KV cache upper bound.
    let max_position_embeddings = req_u64("context_length")? as usize;

    // Tokenizer special tokens (0 when unset is acceptable).
    let bos_token_id = meta.get_u64("tokenizer.ggml.bos_token_id").unwrap_or(0);
    let eos_token_id = meta.get_u64("tokenizer.ggml.eos_token_id").unwrap_or(0);

    // Tied embeddings: no `output.weight` tensor ⇒ tied.
    let tie_word_embeddings = !inputs.has_output_weight;

    // ── MoE (only for MoE arches) ──
    let num_experts = meta
        .get_u64(&k("expert_count"))
        .map(|v| v as usize)
        .unwrap_or(0);

    let mut body: Map<String, Value> = json!({
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "max_position_embeddings": max_position_embeddings,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "tie_word_embeddings": tie_word_embeddings,
        "model_type": model_type,
    })
    .as_object()
    .expect("json! object literal")
    .clone();

    if num_experts > 0 {
        let experts_per_tok = req_u64("expert_used_count").with_context(|| {
            format!("GGUF: MoE arch '{arch}' has expert_count>0 but no '{arch}.expert_used_count'")
        })? as usize;
        let moe_ffn = req_u64("expert_feed_forward_length").with_context(|| {
            format!("GGUF: MoE arch '{arch}' missing '{arch}.expert_feed_forward_length'")
        })? as usize;
        body.insert("num_experts".into(), json!(num_experts));
        body.insert("num_experts_per_tok".into(), json!(experts_per_tok));
        body.insert("moe_intermediate_size".into(), json!(moe_ffn));
    }

    // sliding_window (gemma hybrid attention); 0/absent ⇒ full attention.
    if let Some(sw) = meta.get_u64(&k("attention.sliding_window")) {
        body.insert("sliding_window".into(), json!(sw));
    }

    // ── Deserialize numeric fields, then set arch fields explicitly ──
    let raw = Value::Object(body);
    let json_str = serde_json::to_string(&raw).context("serialize synthesized GGUF config")?;
    let mut config: ModelConfig =
        serde_json::from_str(&json_str).context("deserialize synthesized GGUF config")?;

    config.model_type = model_type.to_string();
    config.attn_gated = attn_gated;
    // The GGUF name map emits HF names under the `model.` prefix
    // (`model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`).
    // `layer_prefix()` yields `model.layers.N` for both "" and "model", but the
    // embed/norm/lm_head lookups use the raw prefix — so it must be "model", not
    // "" (else they resolve to `.embed_tokens.weight` and fail).
    config.weight_prefix = "model".to_string();

    // Gemma-specific post-parse fixups.
    if model_type == "gemma4" {
        config.embed_scale = (hidden_size as f32).sqrt();
        // Logit softcap: honor the GGUF key if present (gemma2), else 0.0
        // (disabled). gemma3+ dropped softcapping.
        config.final_logit_softcapping = meta
            .get_f64(&k("final_logit_softcapping"))
            .map(|v| v as f32)
            .unwrap_or(0.0);
    }

    // Reuse the shared quantization-config + validation pass.
    finalize_config(&mut config, &raw)?;
    Ok(config)
}

/// Build Laguna-S-2.1's heterogeneous HF-shaped config from its native GGUF
/// metadata. Unlike homogeneous GGUF decoders, Laguna encodes Q-head counts as
/// one integer per layer and carries separate full/SWA RoPE parameters.
fn config_from_laguna_gguf(inputs: &GgufConfigInputs) -> Result<ModelConfig> {
    let meta = inputs.meta;
    let req_u64 = |suffix: &str| -> Result<u64> {
        meta.get_u64(&format!("laguna.{suffix}"))
            .with_context(|| format!("GGUF metadata missing required key 'laguna.{suffix}'"))
    };
    let req_f64 = |suffix: &str| -> Result<f64> {
        meta.get_f64(&format!("laguna.{suffix}"))
            .with_context(|| format!("GGUF metadata missing required key 'laguna.{suffix}'"))
    };

    let num_hidden_layers = req_u64("block_count")? as usize;
    ensure!(
        num_hidden_layers == 48,
        "Laguna-S-2.1 GGUF must contain 48 layers, found {num_hidden_layers}"
    );
    let heads_u64 = meta
        .get_u64_array("laguna.attention.head_count")
        .context("GGUF metadata missing required integer array 'laguna.attention.head_count'")?;
    ensure!(
        heads_u64.len() == num_hidden_layers,
        "laguna.attention.head_count length ({}) does not match block_count ({num_hidden_layers})",
        heads_u64.len()
    );
    let heads: Vec<usize> = heads_u64
        .into_iter()
        .map(|v| usize::try_from(v).context("laguna attention head count does not fit usize"))
        .collect::<Result<_>>()?;
    let full_heads = heads[0];
    let sliding_heads = heads
        .iter()
        .copied()
        .max()
        .context("empty Laguna head array")?;
    ensure!(
        full_heads > 0 && sliding_heads > full_heads,
        "Laguna GGUF must declare distinct non-zero full and sliding Q-head counts"
    );
    ensure!(
        heads
            .iter()
            .enumerate()
            .all(|(i, &h)| if i.is_multiple_of(4) {
                h == full_heads
            } else {
                h == sliding_heads
            }),
        "Laguna GGUF attention heads must follow one full layer then three sliding layers"
    );
    let layer_types: Vec<&str> = heads
        .iter()
        .map(|&h| {
            if h == full_heads {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();

    let head_dim = req_u64("attention.key_length")?;
    ensure!(head_dim > 0, "laguna.attention.key_length must be non-zero");
    ensure!(
        req_u64("attention.value_length")? == head_dim,
        "Laguna GGUF key/value head dimensions must match"
    );
    let full_rope_dim = req_u64("rope.dimension_count")?;
    let sliding_rope_dim = req_u64("rope.dimension_count_swa")?;
    ensure!(
        full_rope_dim > 0 && full_rope_dim <= head_dim,
        "laguna.rope.dimension_count must be in 1..=head_dim"
    );
    ensure!(
        sliding_rope_dim == head_dim,
        "laguna.rope.dimension_count_swa must equal head_dim"
    );
    ensure!(
        meta.get_str("laguna.rope.scaling.type") == Some("yarn"),
        "laguna.rope.scaling.type must be 'yarn'"
    );
    ensure!(
        req_u64("expert_gating_func")? == 2,
        "Laguna GGUF expert_gating_func must be sigmoid (2)"
    );

    let leading_dense = req_u64("leading_dense_block_count")? as usize;
    ensure!(
        leading_dense <= num_hidden_layers,
        "laguna.leading_dense_block_count exceeds block_count"
    );
    let mlp_only_layers: Vec<usize> = (0..leading_dense).collect();
    let bos_token_id = meta.get_u64("tokenizer.ggml.bos_token_id").unwrap_or(0);
    let eos_token_id = meta.get_u64("tokenizer.ggml.eos_token_id").unwrap_or(0);

    let raw = json!({
        "model_type": "laguna",
        "hidden_size": req_u64("embedding_length")?,
        "intermediate_size": req_u64("feed_forward_length")?,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": req_u64("vocab_size")?,
        "num_attention_heads": sliding_heads,
        "num_attention_heads_per_layer": heads,
        "num_key_value_heads": req_u64("attention.head_count_kv")?,
        "head_dim": head_dim,
        "num_experts": req_u64("expert_count")?,
        "num_experts_per_tok": req_u64("expert_used_count")?,
        "moe_intermediate_size": req_u64("expert_feed_forward_length")?,
        "shared_expert_intermediate_size": req_u64("expert_shared_feed_forward_length")?,
        "norm_topk_prob": req_u64("expert_weights_norm")? != 0,
        "decoder_sparse_step": 1,
        "mlp_only_layers": mlp_only_layers,
        "layer_types": layer_types,
        "sliding_window": req_u64("attention.sliding_window")?,
        "max_position_embeddings": req_u64("context_length")?,
        "rms_norm_eps": req_f64("attention.layer_norm_rms_epsilon")?,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "tie_word_embeddings": !inputs.has_output_weight,
        "gating": "per-head",
        "moe_routed_scaling_factor": req_f64("expert_weights_scale")?,
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": req_f64("rope.freq_base")?,
                "factor": req_f64("rope.scaling.factor")?,
                "original_max_position_embeddings": req_u64("rope.scaling.original_context_length")?,
                "beta_slow": req_f64("rope.scaling.yarn_beta_slow")?,
                "beta_fast": req_f64("rope.scaling.yarn_beta_fast")?,
                "attention_factor": req_f64("rope.scaling.yarn_attn_factor")?,
                "partial_rotary_factor": full_rope_dim as f64 / head_dim as f64
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": req_f64("rope.freq_base_swa")?,
                "partial_rotary_factor": sliding_rope_dim as f64 / head_dim as f64
            }
        }
    });
    parse_laguna(&raw).context("failed to build Laguna ModelConfig from GGUF metadata")
}

// ── Fields GGUF does NOT provide, and how they are set (explicit, no silent
//    prod defaults) ──
//   * partial_rotary_factor / rotary_dim: left at struct default 1.0 (full
//     RoPE). `{arch}.rope.dimension_count` could refine this for partial-rotary
//     models; deliberately NOT auto-applied here until a target arch needs it.
//   * layer_types / hybrid fields: left empty for homogeneous decoders;
//     Laguna constructs them from its per-layer head-count array.
//   * All SSM/MLA/DeepSeek/MiniMax/vision fields: 0 / empty — not applicable to
//     the llama/qwen/gemma decoder families this builder targets.
//   * ep_rank/ep_world_size/tp_*: set at runtime by the caller, not here.

#[cfg(test)]
mod tests;
