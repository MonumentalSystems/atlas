// SPDX-License-Identifier: AGPL-3.0-only

//! Build a [`ModelConfig`] from GGUF file metadata.
//!
//! GGUF carries its model config inline as metadata key/values
//! (`llama.block_count`, `qwen3.attention.head_count`, …) rather than a
//! sibling `config.json`. This module reads those keys through the
//! [`GgufMeta`] accessor (implemented by the GGUF parser in spark-runtime, so
//! atlas-core keeps no GGUF dependency) and produces a validated
//! [`ModelConfig`] for the llama / qwen2 / qwen3 / gemma decoder families.
//!
//! Strategy: synthesize an HF-config-shaped JSON object from the GGUF keys and
//! deserialize it into `ModelConfig` (serde `#[serde(default)]` fills the many
//! fields GGUF has no analog for), then set the architecture flags
//! (`attn_gated`, `weight_prefix`, gemma `embed_scale` /
//! `final_logit_softcapping`) explicitly, then run the shared
//! [`super::finalize_config`]. No silent production defaults: every value GGUF
//! omits is either derived by an explicit documented rule or is an error.

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use super::{ModelConfig, finalize_config};

/// Typed read access to GGUF metadata. Implemented by the spark-runtime GGUF
/// parser over its parsed key/value table. All getters return `None` when the
/// key is absent or holds a different value type — the builder decides whether
/// absence is fatal or has a derivation rule.
pub trait GgufMeta {
    /// Any unsigned/signed integer metadata value, widened to u64.
    fn get_u64(&self, key: &str) -> Option<u64>;
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

// ── Fields GGUF does NOT provide, and how they are set (explicit, no silent
//    prod defaults) ──
//   * partial_rotary_factor / rotary_dim: left at struct default 1.0 (full
//     RoPE). `{arch}.rope.dimension_count` could refine this for partial-rotary
//     models; deliberately NOT auto-applied here until a target arch needs it.
//   * layer_types / hybrid fields: left empty (homogeneous decoder).
//   * All SSM/MLA/DeepSeek/MiniMax/vision fields: 0 / empty — not applicable to
//     the llama/qwen/gemma decoder families this builder targets.
//   * ep_rank/ep_world_size/tp_*: set at runtime by the caller, not here.

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Minimal in-memory GgufMeta for tests.
    #[derive(Default)]
    struct Meta {
        u: HashMap<String, u64>,
        f: HashMap<String, f64>,
        s: HashMap<String, String>,
        arr: HashMap<String, usize>,
    }
    impl Meta {
        fn u(mut self, k: &str, v: u64) -> Self {
            self.u.insert(k.into(), v);
            self
        }
        fn f(mut self, k: &str, v: f64) -> Self {
            self.f.insert(k.into(), v);
            self
        }
        fn s(mut self, k: &str, v: &str) -> Self {
            self.s.insert(k.into(), v.into());
            self
        }
    }
    impl GgufMeta for Meta {
        fn get_u64(&self, k: &str) -> Option<u64> {
            self.u.get(k).copied()
        }
        fn get_f64(&self, k: &str) -> Option<f64> {
            self.f.get(k).copied()
        }
        fn get_str(&self, k: &str) -> Option<&str> {
            self.s.get(k).map(String::as_str)
        }
        fn get_arr_len(&self, k: &str) -> Option<usize> {
            self.arr.get(k).copied()
        }
    }

    fn llama_meta() -> Meta {
        Meta::default()
            .s("general.architecture", "llama")
            .u("llama.embedding_length", 4096)
            .u("llama.block_count", 32)
            .u("llama.feed_forward_length", 11008)
            .u("llama.attention.head_count", 32)
            .u("llama.attention.head_count_kv", 8)
            .u("llama.context_length", 4096)
            .u("llama.vocab_size", 32000)
            .f("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .f("llama.rope.freq_base", 10000.0)
    }

    #[test]
    fn llama_dense_maps_to_mistral() {
        let m = llama_meta();
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: true };
        let c = config_from_gguf(&inp).unwrap();
        assert_eq!(c.model_type, "mistral");
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_hidden_layers, 32);
        assert_eq!(c.intermediate_size, 11008);
        assert_eq!(c.num_attention_heads, 32);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.head_dim, 128); // 4096/32
        assert_eq!(c.vocab_size, 32000);
        assert_eq!(c.max_position_embeddings, 4096);
        assert!((c.rms_norm_eps - 1e-5).abs() < 1e-12);
        assert!(!c.attn_gated);
        assert!(!c.tie_word_embeddings); // has_output_weight = true
        assert_eq!(c.weight_prefix, "model"); // HF-name prefix the GGUF loader emits
        assert_eq!(c.num_experts, 0);
    }

    #[test]
    fn head_dim_derived_when_key_absent() {
        let m = llama_meta();
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: true };
        let c = config_from_gguf(&inp).unwrap();
        assert_eq!(c.head_dim, 128);
    }

    #[test]
    fn explicit_key_length_wins() {
        let m = llama_meta().u("llama.attention.key_length", 96);
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: true };
        let c = config_from_gguf(&inp).unwrap();
        assert_eq!(c.head_dim, 96);
    }

    #[test]
    fn kv_heads_default_to_mha() {
        let mut m = llama_meta();
        m.u.remove("llama.attention.head_count_kv");
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: true };
        let c = config_from_gguf(&inp).unwrap();
        assert_eq!(c.num_key_value_heads, c.num_attention_heads);
    }

    #[test]
    fn vocab_from_token_embd_rows() {
        let mut m = llama_meta();
        m.u.remove("llama.vocab_size");
        let inp =
            GgufConfigInputs { meta: &m, token_embd_vocab: Some(128256), has_output_weight: true };
        let c = config_from_gguf(&inp).unwrap();
        assert_eq!(c.vocab_size, 128256);
    }

    #[test]
    fn tied_embeddings_when_no_output_tensor() {
        let m = llama_meta();
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: false };
        let c = config_from_gguf(&inp).unwrap();
        assert!(c.tie_word_embeddings);
    }

    #[test]
    fn qwen3_dense_routes_to_qwen35_dense() {
        let m = Meta::default()
            .s("general.architecture", "qwen3")
            .u("qwen3.embedding_length", 2048)
            .u("qwen3.block_count", 28)
            .u("qwen3.feed_forward_length", 6144)
            .u("qwen3.attention.head_count", 16)
            .u("qwen3.attention.head_count_kv", 8)
            .u("qwen3.attention.key_length", 128)
            .u("qwen3.context_length", 40960)
            .u("qwen3.vocab_size", 151936)
            .f("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
            .f("qwen3.rope.freq_base", 1000000.0);
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: true };
        let c = config_from_gguf(&inp).unwrap();
        assert_eq!(c.model_type, "qwen3_5");
        assert_eq!(c.num_experts, 0);
        assert!(c.is_qwen35_dense());
        assert_eq!(c.head_dim, 128);
        assert!((c.rope_theta - 1_000_000.0).abs() < 1e-3);
    }

    #[test]
    fn qwen3_moe_populates_expert_fields() {
        let m = Meta::default()
            .s("general.architecture", "qwen3moe")
            .u("qwen3moe.embedding_length", 2048)
            .u("qwen3moe.block_count", 48)
            .u("qwen3moe.feed_forward_length", 768)
            .u("qwen3moe.attention.head_count", 32)
            .u("qwen3moe.attention.head_count_kv", 4)
            .u("qwen3moe.attention.key_length", 128)
            .u("qwen3moe.context_length", 32768)
            .u("qwen3moe.vocab_size", 151936)
            .u("qwen3moe.expert_count", 128)
            .u("qwen3moe.expert_used_count", 8)
            .u("qwen3moe.expert_feed_forward_length", 768)
            .f("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
            .f("qwen3moe.rope.freq_base", 1000000.0);
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: true };
        let c = config_from_gguf(&inp).unwrap();
        assert_eq!(c.model_type, "qwen3_5_moe");
        assert_eq!(c.num_experts, 128);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_intermediate_size, 768);
    }

    #[test]
    fn gemma_sets_embed_scale_and_softcap() {
        let m = Meta::default()
            .s("general.architecture", "gemma2")
            .u("gemma2.embedding_length", 2304)
            .u("gemma2.block_count", 26)
            .u("gemma2.feed_forward_length", 9216)
            .u("gemma2.attention.head_count", 8)
            .u("gemma2.attention.head_count_kv", 4)
            .u("gemma2.attention.key_length", 256)
            .u("gemma2.context_length", 8192)
            .u("gemma2.vocab_size", 256000)
            .u("gemma2.attention.sliding_window", 4096)
            .f("gemma2.attention.layer_norm_rms_epsilon", 1e-6)
            .f("gemma2.final_logit_softcapping", 30.0);
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: false };
        let c = config_from_gguf(&inp).unwrap();
        assert_eq!(c.model_type, "gemma4");
        assert!(!c.attn_gated);
        assert_eq!(c.sliding_window, 4096);
        assert!((c.embed_scale - (2304f32).sqrt()).abs() < 1e-3);
        assert!((c.final_logit_softcapping - 30.0).abs() < 1e-3);
    }

    #[test]
    fn missing_architecture_errors() {
        let m = Meta::default().u("llama.embedding_length", 4096);
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: true };
        assert!(config_from_gguf(&inp).is_err());
    }

    #[test]
    fn missing_required_dim_errors() {
        let mut m = llama_meta();
        m.u.remove("llama.embedding_length");
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: true };
        let err = config_from_gguf(&inp).unwrap_err().to_string();
        assert!(err.contains("embedding_length"), "unexpected: {err}");
    }

    #[test]
    fn moe_without_used_count_errors() {
        let m = Meta::default()
            .s("general.architecture", "qwen3moe")
            .u("qwen3moe.embedding_length", 2048)
            .u("qwen3moe.block_count", 48)
            .u("qwen3moe.feed_forward_length", 768)
            .u("qwen3moe.attention.head_count", 32)
            .u("qwen3moe.context_length", 32768)
            .u("qwen3moe.vocab_size", 151936)
            .u("qwen3moe.expert_count", 128);
        // expert_used_count / expert_feed_forward_length intentionally absent.
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: true };
        assert!(config_from_gguf(&inp).is_err());
    }

    #[test]
    fn unmapped_arch_errors() {
        let m = Meta::default()
            .s("general.architecture", "mamba")
            .u("mamba.embedding_length", 4096);
        let inp = GgufConfigInputs { meta: &m, token_embd_vocab: None, has_output_weight: true };
        assert!(config_from_gguf(&inp).is_err());
    }
}
