// SPDX-License-Identifier: AGPL-3.0-only

//! Tests split out of `config.rs` for file-size budget.

#![allow(unused_imports)]

use super::*;

mod tests_b;

#[test]
fn test_qwen3_default_config() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(cfg.num_experts, 512);
    assert_eq!(cfg.num_attention_layers(), 12);
    assert_eq!(cfg.num_ssm_layers(), 36);
    assert_eq!(cfg.gqa_ratio(), 8);
    assert_eq!(cfg.rotary_dim(), 64);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.layer_type(2), LayerType::LinearAttention);
    assert_eq!(cfg.layer_type(3), LayerType::FullAttention);
    assert_eq!(cfg.layer_type(47), LayerType::FullAttention);
    assert_eq!(cfg.ssm_qkvz_size(), 2048 + 2048 + 4096 + 4096);
    assert_eq!(cfg.ssm_ba_size(), 64);
}

#[test]
fn test_parse_actual_config() {
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/qwen3_config.json"
    ));
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_experts, 512);
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(cfg.layer_types.len(), 48);
    assert_eq!(cfg.layer_types[0], LayerType::LinearAttention);
    assert_eq!(cfg.layer_types[3], LayerType::FullAttention);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert!(cfg.norm_topk_prob);
    assert!(!cfg.tie_word_embeddings);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
    assert_eq!(cfg.model_type, "qwen3_next");
    assert!(cfg.weight_prefix.is_empty());
}

#[test]
fn test_parse_qwen35_nested_config() {
    let json = r#"{
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "rope_parameters": {
                "rope_theta": 10000000,
                "rope_type": "default"
            },
            "mtp_num_hidden_layers": 1
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5_moe");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_hidden_layers, 40);
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.num_attention_layers(), 10);
    assert_eq!(cfg.num_ssm_layers(), 30);
    assert_eq!(cfg.layer_types.len(), 40);
    assert_eq!(cfg.eos_token_id, 248044);
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert!(cfg.is_qwen35());
    assert!(cfg.norm_topk_prob); // Qwen3.5 unconditionally normalizes
    assert_eq!(cfg.ssm_qkv_size(), 2048 + 2048 + 4096); // 8192
    assert_eq!(cfg.ssm_z_size(), 4096);
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
}

#[test]
fn test_parse_holo31_vlm_config() {
    let json = r#"{
        "model_type": "qwen3_5_moe",
        "image_token_id": 248056,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000,
                "rope_type": "default"
            }
        },
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "holo3_1_moe");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_attention_layers(), 10);
    assert_eq!(cfg.num_ssm_layers(), 30);
    assert_eq!(cfg.mrope_section, [11, 11, 10]);
    assert!(cfg.mrope_interleaved);

    let vision = cfg.vision.expect("Holo3.1 must parse vision_config");
    assert_eq!(vision.depth, 27);
    assert_eq!(vision.hidden_size, 1152);
    assert_eq!(vision.out_hidden_size, 2048);
    assert!(vision.deepstack_visual_indexes.is_empty());
    assert_eq!(vision.image_pad_token_id, 248056);
}

#[test]
fn test_parse_qwen3_vl_config() {
    let json = r#"{
        "model_type": "qwen3_vl_moe",
        "text_config": {
            "model_type": "qwen3_vl_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "vocab_size": 151936,
            "rope_theta": 5000000,
            "norm_topk_prob": true
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_vl_moe");
    assert!(cfg.is_qwen3_vl());
    assert!(!cfg.is_qwen35());
    assert!(cfg.capabilities().has_nested_config);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.num_hidden_layers, 48);
    // Pure attention: all layers are FullAttention (full_attention_interval defaults to 1)
    assert_eq!(cfg.num_attention_layers(), 48);
    assert_eq!(cfg.num_ssm_layers(), 0);
    assert_eq!(cfg.gqa_ratio(), 8);
    // Full rotary: partial_rotary_factor defaults to 1.0
    assert_eq!(cfg.rotary_dim(), 128);
    assert_eq!(cfg.rope_theta, 5_000_000.0);
    assert!(cfg.norm_topk_prob);
}

/// Qwen3.5-VL detection: the trunk `model_type` stays `qwen3_5`
/// (same as the text-only variant) but the upstream config ships a
/// `vision_config` block plus `architectures =
/// ["Qwen3_5ForConditionalGeneration"]`. `is_qwen3_vl()` must
/// distinguish via the parsed `config.vision` so the factory routes
/// the checkpoint to the VL weight loader instead of the dense LLM
/// loader.
#[test]
fn test_parse_qwen3_5_vl_config() {
    let json = r#"{
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 9216,
            "vocab_size": 248320,
            "rope_theta": 10000000.0
        },
        "vision_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "patch_size": 16,
            "spatial_merge_size": 2
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5");
    assert!(
        cfg.is_qwen3_vl(),
        "Qwen3.5-VL detected via model_type=qwen3_5 + vision_config presence"
    );
    assert!(cfg.vision.is_some());
}

/// Counter-test: a text-only `qwen3_5` config WITHOUT `vision_config`
/// must NOT be misclassified as VL. Pins the gate condition is
/// actually using `vision.is_some()`, not just model_type.
#[test]
fn test_qwen3_5_text_only_not_vl() {
    let json = r#"{
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 151936,
            "rope_theta": 10000000.0
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5");
    assert!(
        !cfg.is_qwen3_vl(),
        "qwen3_5 without vision_config must not be classified as VL"
    );
}

/// Regression for the alpha-2.99 dispatch bug:
/// Kbenkhaled/Qwen3.5-27B-NVFP4 is a *dense* hybrid (top model_type
/// "qwen3_5", num_experts=0) that nonetheless enables MRoPE in
/// text_config.rope_parameters. Pre-c0cde18, the MRoPE detector
/// rewrote model_type → "qwen3_6_moe" unconditionally, then the
/// kernel dispatcher couldn't find a target for
/// (qwen3_6_moe, hidden_size=5120) — only (qwen3_6_moe, 2048) for
/// qwen3.6-35b-a3b exists. The fix gates the rewrite on
/// is_moe(top_model_type). This test pins that contract.
#[test]
fn test_kbenkhaled_qwen35_27b_dense_mrope_no_rewrite() {
    let json = r#"{
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 17408,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "rope_parameters": {
                "rope_theta": 10000000,
                "rope_type": "default",
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            }
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    // Critical: dense + MRoPE must NOT be rewritten to qwen3_6_moe,
    // or the dispatcher won't find the qwen3.5-27b kernel target.
    assert_eq!(cfg.model_type, "qwen3_5");
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_experts, 0);
    // MRoPE flags still parsed so the kernel uses the right rope path.
    assert!(cfg.mrope_interleaved);
    assert_eq!(cfg.mrope_section, [11, 11, 10]);
}

#[test]
fn test_layer_prefix() {
    let cfg80b = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(cfg80b.layer_prefix(3), "model.layers.3");

    let mut cfg35 = ModelConfig::qwen3_next_80b_nvfp4();
    cfg35.weight_prefix = "model.language_model".to_string();
    assert_eq!(cfg35.layer_prefix(3), "model.language_model.layers.3");
}

