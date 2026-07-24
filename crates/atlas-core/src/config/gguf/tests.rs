// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::collections::HashMap;

/// Minimal in-memory GgufMeta for tests.
#[derive(Default)]
struct Meta {
    u: HashMap<String, u64>,
    ua: HashMap<String, Vec<u64>>,
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
    fn ua(mut self, k: &str, v: &[u64]) -> Self {
        self.ua.insert(k.into(), v.to_vec());
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
    fn get_u64_array(&self, k: &str) -> Option<Vec<u64>> {
        self.ua.get(k).cloned()
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

#[test]
fn laguna_populates_heterogeneous_attention_moe_and_yarn() {
    let heads = [
        48, 72, 72, 72, 48, 72, 72, 72, 48, 72, 72, 72, 48, 72, 72, 72, 48, 72, 72, 72, 48, 72, 72,
        72, 48, 72, 72, 72, 48, 72, 72, 72, 48, 72, 72, 72, 48, 72, 72, 72, 48, 72, 72, 72, 48, 72,
        72, 72,
    ];
    let mut m = Meta::default()
        .s("general.architecture", "laguna")
        .u("laguna.block_count", 48)
        .u("laguna.context_length", 262_144)
        .u("laguna.embedding_length", 3072)
        .u("laguna.feed_forward_length", 12_288)
        .ua("laguna.attention.head_count", &heads)
        .u("laguna.attention.head_count_kv", 8)
        .s("laguna.rope.scaling.type", "yarn")
        .f("laguna.rope.scaling.factor", 32.0)
        .u("laguna.rope.scaling.original_context_length", 8192)
        .f("laguna.rope.scaling.yarn_attn_factor", 1.0)
        .f("laguna.rope.scaling.yarn_beta_fast", 32.0)
        .f("laguna.rope.scaling.yarn_beta_slow", 1.0)
        .f("laguna.rope.freq_base", 500_000.0)
        .f("laguna.rope.freq_base_swa", 10_000.0)
        .f("laguna.attention.layer_norm_rms_epsilon", 1e-6)
        .u("laguna.expert_count", 256)
        .u("laguna.expert_used_count", 10)
        .u("laguna.attention.key_length", 128)
        .u("laguna.attention.value_length", 128)
        .u("laguna.vocab_size", 100_352)
        .u("laguna.attention.sliding_window", 512)
        .u("laguna.expert_feed_forward_length", 1024)
        .u("laguna.expert_shared_feed_forward_length", 1024)
        .u("laguna.expert_weights_norm", 1)
        .f("laguna.expert_weights_scale", 2.5)
        .u("laguna.expert_gating_func", 2)
        .u("laguna.leading_dense_block_count", 1)
        .u("laguna.rope.dimension_count", 64)
        .u("laguna.rope.dimension_count_swa", 128)
        .u("tokenizer.ggml.bos_token_id", 2)
        .u("tokenizer.ggml.eos_token_id", 2);
    let c = config_from_gguf(&GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    })
    .expect("parse official Laguna metadata shape");

    assert_eq!(c.model_type, "laguna");
    assert_eq!(c.num_hidden_layers, 48);
    assert_eq!(c.num_attention_heads, 72);
    assert_eq!(c.num_attention_heads_per_layer, heads.map(|v| v as usize));
    assert_eq!(
        c.layer_types
            .iter()
            .filter(|t| matches!(t, super::super::LayerType::FullAttention))
            .count(),
        12
    );
    assert_eq!(
        c.layer_types
            .iter()
            .filter(|t| matches!(t, super::super::LayerType::SlidingAttention))
            .count(),
        36
    );
    assert_eq!(c.mlp_only_layers, [0]);
    assert_eq!(c.num_experts, 256);
    assert_eq!(c.num_experts_per_tok, 10);
    assert_eq!(c.shared_expert_intermediate_size, 1024);
    assert_eq!(c.sliding_window, 512);
    assert_eq!(c.max_position_embeddings, 262_144);
    assert_eq!(c.rotary_dim(), 64);
    assert_eq!(c.yarn_factor, 32.0);
    assert_eq!(c.yarn_original_max_position_embeddings, 8192);
    assert_eq!(c.yarn_attention_factor, 1.0);
    assert_eq!(c.scoring_func, "sigmoid");
    assert!(c.use_routing_bias);
    assert_eq!(c.routed_scaling_factor, 2.5);

    m.ua.get_mut("laguna.attention.head_count").unwrap()[1] = 48;
    let error = config_from_gguf(&GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    })
    .expect_err("reject invalid full/sliding layer pattern");
    assert!(
        error
            .to_string()
            .contains("one full layer then three sliding")
    );
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
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
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
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.head_dim, 128);
}

#[test]
fn explicit_key_length_wins() {
    let m = llama_meta().u("llama.attention.key_length", 96);
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.head_dim, 96);
}

#[test]
fn kv_heads_default_to_mha() {
    let mut m = llama_meta();
    m.u.remove("llama.attention.head_count_kv");
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.num_key_value_heads, c.num_attention_heads);
}

#[test]
fn vocab_from_token_embd_rows() {
    let mut m = llama_meta();
    m.u.remove("llama.vocab_size");
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: Some(128256),
        has_output_weight: true,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.vocab_size, 128256);
}

#[test]
fn tied_embeddings_when_no_output_tensor() {
    let m = llama_meta();
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: false,
    };
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
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
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
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
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
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: false,
    };
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
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    assert!(config_from_gguf(&inp).is_err());
}

#[test]
fn missing_required_dim_errors() {
    let mut m = llama_meta();
    m.u.remove("llama.embedding_length");
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
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
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    assert!(config_from_gguf(&inp).is_err());
}

#[test]
fn unmapped_arch_errors() {
    let m = Meta::default()
        .s("general.architecture", "mamba")
        .u("mamba.embedding_length", 4096);
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    assert!(config_from_gguf(&inp).is_err());
}
