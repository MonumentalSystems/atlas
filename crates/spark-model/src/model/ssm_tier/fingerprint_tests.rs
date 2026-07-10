// SPDX-License-Identifier: AGPL-3.0-only

//! Fingerprint golden pins + determinism + override precedence.
//!
//! L1 pins the hash primitive to external FNV reference vectors, L2 pins two
//! full fingerprints to frozen literals, L3 proves per-field sensitivity (a
//! refactor cannot silently drop a field), L4 proves encoding injectivity.

use std::num::NonZeroU64;

use atlas_core::config::{LayerType, ModelConfig, QuantizationConfig};

use super::*;

const BLOB: usize = 4096;

fn hybrid() -> ModelConfig {
    ModelConfig::qwen3_next_80b_nvfp4()
}

fn dense() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3".to_string();
    c.num_hidden_layers = 28;
    c.layer_types = vec![LayerType::FullAttention; 28];
    c.num_experts = 0;
    c.linear_num_key_heads = 0;
    c.linear_key_head_dim = 0;
    c.linear_num_value_heads = 0;
    c.linear_value_head_dim = 0;
    c
}

fn fp(cfg: &ModelConfig) -> u64 {
    ModelFingerprint::derive_with_id(cfg, BLOB, "")
        .unwrap()
        .get()
}

// ── L1: pin the primitive to published FNV-1a/64 reference vectors ────
// A swapped or "optimized" implementation cannot pass someone else's
// constants.
#[test]
fn fnv1a_64_matches_reference_vectors() {
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_64(b"foobar"), 0x85944171f73967e8);
}

// ── L2: golden pins ────────────────────────────────────────────────────
// DO NOT update these literals to make a test pass — changing them
// invalidates every persisted cache key on every shared paging peer. An
// encoding change is a deliberate fleet-wide cache flush and bumps
// FP_VERSION (and gets a changelog entry).
#[test]
fn golden_fingerprint_hybrid_moe_is_pinned() {
    assert_eq!(fp(&hybrid()), 0x7a19_08d7_ca78_e45b);
}

#[test]
fn golden_fingerprint_dense_is_pinned() {
    assert_eq!(fp(&dense()), 0xe0b6_f920_07eb_d0a6);
}

// ── Determinism: same config → same u64, every time ───────────────────
#[test]
fn fingerprint_is_deterministic() {
    assert_eq!(fp(&hybrid()), fp(&hybrid()));
    assert_eq!(fp(&dense()), fp(&dense()));
    assert_ne!(fp(&hybrid()), fp(&dense()));
}

// ── L3: per-field sensitivity ──────────────────────────────────────────
// Mutating any single fingerprint field must change the value — this is
// what prevents a refactor from silently DROPPING a field from the
// encoding (the golden pin alone would still pass whenever the dropped
// field is unchanged in the fixture).
#[test]
fn every_fingerprint_field_is_load_bearing() {
    let base = fp(&hybrid());
    let muts: Vec<(&str, Box<dyn Fn(&mut ModelConfig)>)> = vec![
        ("model_type", Box::new(|c| c.model_type = "other".into())),
        (
            "num_hidden_layers",
            Box::new(|c| {
                c.num_hidden_layers += 1;
                // keep layer_types-driven counts moving too
                c.layer_types.push(LayerType::FullAttention);
            }),
        ),
        (
            "layer mix (ssm/attn split)",
            Box::new(|c| {
                c.layer_types[0] = LayerType::FullAttention;
            }),
        ),
        ("head_dim", Box::new(|c| c.head_dim += 1)),
        (
            "num_key_value_heads",
            Box::new(|c| c.num_key_value_heads += 1),
        ),
        ("num_experts", Box::new(|c| c.num_experts = 0)),
        (
            "linear_num_key_heads",
            Box::new(|c| c.linear_num_key_heads += 1),
        ),
        (
            "linear_key_head_dim",
            Box::new(|c| c.linear_key_head_dim += 1),
        ),
        (
            "linear_num_value_heads",
            Box::new(|c| c.linear_num_value_heads += 1),
        ),
        (
            "linear_value_head_dim",
            Box::new(|c| c.linear_value_head_dim += 1),
        ),
        (
            "linear_conv_kernel_dim",
            Box::new(|c| c.linear_conv_kernel_dim += 1),
        ),
        ("mamba_num_heads", Box::new(|c| c.mamba_num_heads = 8)),
        ("mamba_head_dim", Box::new(|c| c.mamba_head_dim = 64)),
        ("ssm_state_size", Box::new(|c| c.ssm_state_size = 128)),
        ("n_groups", Box::new(|c| c.n_groups = 8)),
        (
            "quantization_config",
            Box::new(|c| {
                c.quantization_config = Some(QuantizationConfig {
                    quant_method: "modelopt".into(),
                    quant_algo: "NVFP4".into(),
                    format: String::new(),
                    ignore_modules: Vec::new(),
                });
            }),
        ),
        (
            "kv_layer_dims",
            Box::new(|c| c.kv_layer_dims = vec![(2, 256), (4, 128)]),
        ),
    ];
    for (name, m) in muts {
        let mut c = hybrid();
        m(&mut c);
        assert_ne!(fp(&c), base, "field {name} dropped from the fingerprint");
    }
    // blob_bytes is a fingerprint input too (task hard requirement).
    let b2 = ModelFingerprint::derive_with_id(&hybrid(), BLOB + 1, "")
        .unwrap()
        .get();
    assert_ne!(b2, base, "blob_bytes dropped from the fingerprint");
    // Optional ATLAS_MODEL_ID salt (fine-tune with identical geometry).
    let salted = ModelFingerprint::derive_with_id(&hybrid(), BLOB, "ft-v2")
        .unwrap()
        .get();
    assert_ne!(salted, base, "model_id salt dropped from the fingerprint");
}

// kv_layer_dims order is canonical (layer order) — a reorder must rotate.
#[test]
fn kv_layer_dims_order_is_canonical() {
    let mut a = hybrid();
    a.kv_layer_dims = vec![(2, 256), (4, 128)];
    let mut b = hybrid();
    b.kv_layer_dims = vec![(4, 128), (2, 256)];
    assert_ne!(fp(&a), fp(&b));
}

// ── L4: encoding injectivity (length-prefixed strings) ─────────────────
// Under naive concatenation ("ab" + "c") == ("a" + "bc"); the tagged,
// length-prefixed encoding must keep them distinct.
#[test]
fn string_encoding_is_injective() {
    let mut a = hybrid();
    a.model_type = "ab".into();
    a.quantization_config = Some(QuantizationConfig {
        quant_method: "c".into(),
        quant_algo: String::new(),
        format: String::new(),
        ignore_modules: Vec::new(),
    });
    let mut b = hybrid();
    b.model_type = "a".into();
    b.quantization_config = Some(QuantizationConfig {
        quant_method: "bc".into(),
        quant_algo: String::new(),
        format: String::new(),
        ignore_modules: Vec::new(),
    });
    assert_ne!(fp(&a), fp(&b));
}

// ── Fail-fast: an underivable config is a hard error (PCND) ───────────
#[test]
fn underivable_config_fails_fast() {
    let mut c = hybrid();
    c.model_type = String::new();
    c.num_hidden_layers = 0;
    c.layer_types = Vec::new();
    let err = ModelFingerprint::derive_with_id(&c, BLOB, "").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("ATLAS_SSM_SWAP_NS"),
        "actionable message: {msg}"
    );
}

// ── Decode namespace: domain-separated mix, never the bare constant ────
#[test]
fn decode_ns_mixes_fingerprint_and_domain() {
    let fa = ModelFingerprint::derive_with_id(&hybrid(), BLOB, "").unwrap();
    let fb = ModelFingerprint::derive_with_id(&dense(), BLOB, "").unwrap();
    let da = mix64(fa.get(), atlas_kernels::DECODE_DOMAIN);
    let db = mix64(fb.get(), atlas_kernels::DECODE_DOMAIN);
    // Separated from the same model's Marconi keys (shared peer residency)…
    assert_ne!(da, fa.get());
    // …never the bare constant (the old cross-model decode collision)…
    assert_ne!(da, atlas_kernels::DECODE_DOMAIN);
    // …and distinct across models.
    assert_ne!(da, db);
}

// ── Override precedence + strict parsing ───────────────────────────────
#[test]
fn override_precedence_and_strict_parse() {
    let derived = NonZeroU64::new(0xFEED).unwrap();
    // No override → derived fingerprint namespace.
    assert_eq!(resolve_ns_from(None, "V", derived).unwrap(), derived);
    // Explicit decimal and 0x-hex overrides win.
    assert_eq!(resolve_ns_from(Some("42"), "V", derived).unwrap().get(), 42);
    assert_eq!(
        resolve_ns_from(Some("0xD3C0"), "V", derived).unwrap().get(),
        0xD3C0
    );
    // Unparseable is a hard error, not a silent fallthrough (PCND).
    assert!(resolve_ns_from(Some("banana"), "V", derived).is_err());
    assert!(resolve_ns_from(Some("-1"), "V", derived).is_err());
    assert!(resolve_ns_from(Some("18446744073709551616"), "V", derived).is_err());
    // 0 is a hard error: the passthrough is removed.
    let err = resolve_ns_from(Some("0"), "V", derived).unwrap_err();
    assert!(format!("{err:#}").contains("passthrough"));
}
