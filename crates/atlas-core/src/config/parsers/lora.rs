// SPDX-License-Identifier: AGPL-3.0-only

//! PEFT `adapter_config.json` parser for runtime LoRA adapters.
//!
//! Split out of `config.rs` for file-size budget, mirroring
//! [`super::quantization`]. Unlike that parser (which returns `Option` so
//! callers fall through to tensor-name heuristics), this one is **hard-fail**:
//! the adapter is explicitly requested via `--lora-adapter`, so anything
//! Atlas cannot faithfully apply must error with a named reason — never be
//! silently skipped (wrong output).
//!
//! NAMING DISCIPLINE: everything here is `peft_*` / `adapter_*`.
//! `kv_lora_rank` / `q_lora_rank` / `o_lora_rank` (`config.rs:182-207`) are
//! MLA low-rank *attention compression*, unrelated to adapters — never reuse
//! those names.

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// v0 target-module allow-list. Deltas apply on full-attention layers
/// (holo-3.1-0.8b: layer indices 3,7,11,15,19,23) plus the dense SwiGLU FFN.
/// `q_proj` is deliberately absent: `attn_output_gate=true` models emit an
/// interleaved Q+gate `q_proj` output (`ModelConfig::attn_gated`,
/// `config.rs:289`) that a PEFT delta maps onto only partially — rejected in
/// v0 rather than mis-sliced. GDN/linear-attention modules are likewise
/// rejected (no exact-replay parity harness for the recurrence yet).
pub const PEFT_SUPPORTED_TARGET_MODULES: &[&str] = &[
    "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
];

/// Parsed subset of a PEFT `adapter_config.json` that Atlas consumes.
///
/// `lora_dropout` is intentionally ignored (train-time only, inference
/// no-op). Everything else PEFT can emit that would change inference
/// output is validated in [`parse_peft_adapter_config`] and rejected by
/// name if unsupported.
#[derive(Debug, Clone)]
pub struct PeftAdapterConfig {
    /// LoRA rank. Must be > 0.
    pub r: usize,
    /// LoRA alpha. PEFT serializes int or float; both accepted.
    pub lora_alpha: f64,
    /// Verbatim `target_modules` entries (bare module names, or full paths
    /// which are validated on their final `.`-segment). The weight loader's
    /// bidirectional audit is the authority on actual per-layer matching.
    pub target_modules: Vec<String>,
    /// rsLoRA flag: switches scaling from `alpha/r` to `alpha/sqrt(r)`.
    /// Hard-required in the on-disk config (never defaulted — a wrong scale
    /// is silent quality loss).
    pub use_rslora: bool,
    /// Informational: the `layers_to_transform` restriction if present.
    /// The weight loader's per-`LayerType` gate is the real authority on
    /// which layers receive deltas; this is kept only for the startup log.
    pub layers_to_transform: Option<Vec<usize>>,
}

impl PeftAdapterConfig {
    /// Delta scale applied at merge: `y += scaling() * (x @ Aᵀ) @ Bᵀ`.
    ///
    /// `alpha/r`, or `alpha/sqrt(r)` when `use_rslora` — read from the
    /// adapter's own config, NEVER defaulted (a wrong scale is silent
    /// quality loss, not an error).
    pub fn scaling(&self) -> f32 {
        debug_assert!(self.r > 0, "validated at parse");
        if self.use_rslora {
            (self.lora_alpha / (self.r as f64).sqrt()) as f32
        } else {
            (self.lora_alpha / self.r as f64) as f32
        }
    }
}

/// Raw deserialization target mirroring PEFT's on-disk field names verbatim
/// (same approach as `DflashConfig`, `dflash_loader.rs:40`). No
/// `deny_unknown_fields`: PEFT emits many irrelevant keys (`task_type`,
/// `revision`, `loftq_config`, `lora_dropout`, ...).
#[derive(Deserialize)]
struct RawPeftAdapterConfig {
    /// "LORA" for LoRA adapters; ADALORA/LOHA/LOKR/IA3 etc. rejected.
    #[serde(default)]
    peft_type: Option<String>,
    r: usize,
    lora_alpha: f64,
    /// Array of strings, or the string "all-linear" (rejected — Atlas
    /// cannot enumerate "all linear" against fused/quantized layouts).
    target_modules: serde_json::Value,
    /// Hard-required: scaling inputs are never defaulted. `None` (field
    /// absent) is a REJECT, not a `false` default.
    #[serde(default)]
    use_rslora: Option<bool>,
    #[serde(default)]
    use_dora: bool,
    /// "none" (default) is the only supported value.
    #[serde(default)]
    bias: Option<String>,
    #[serde(default)]
    rank_pattern: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    alpha_pattern: Option<serde_json::Map<String, serde_json::Value>>,
    /// Full (non-low-rank) modules saved alongside the adapter — rejected.
    #[serde(default)]
    modules_to_save: Option<Vec<String>>,
    /// Layer-subset restriction. ACCEPTED (array form) and kept for logging;
    /// the loader's per-`LayerType` gate is the authority. A non-null,
    /// non-array form is rejected as malformed.
    #[serde(default)]
    layers_to_transform: Option<serde_json::Value>,
}

/// Parse a PEFT `adapter_config.json` payload.
///
/// Hard-fails with a `REJECT(<feature>)`-prefixed message on every PEFT
/// feature v0 does not support. The caller supplies file-path context.
pub fn parse_peft_adapter_config(json: &str) -> Result<PeftAdapterConfig> {
    let raw: RawPeftAdapterConfig = serde_json::from_str(json)
        .context("Parsing PEFT adapter_config.json (r / lora_alpha / target_modules required)")?;

    if let Some(ref pt) = raw.peft_type
        && !pt.eq_ignore_ascii_case("LORA")
    {
        bail!("REJECT(peft_type): adapter declares peft_type='{pt}'; only LORA is supported");
    }
    if raw.use_dora {
        bail!("REJECT(use_dora): DoRA adapters are unsupported (magnitude decomposition has no runtime-delta form)");
    }
    if let Some(ref b) = raw.bias
        && b != "none"
    {
        bail!("REJECT(bias): bias='{b}' ships trained bias deltas; only bias='none' is supported");
    }
    if raw.rank_pattern.as_ref().is_some_and(|m| !m.is_empty()) {
        bail!("REJECT(rank_pattern): per-module rank overrides are unsupported in v0 (uniform r only)");
    }
    if raw.alpha_pattern.as_ref().is_some_and(|m| !m.is_empty()) {
        bail!("REJECT(alpha_pattern): per-module alpha overrides are unsupported in v0 (uniform lora_alpha only)");
    }
    if raw.modules_to_save.as_ref().is_some_and(|v| !v.is_empty()) {
        bail!(
            "REJECT(modules_to_save): adapter saves full modules {:?}; full-weight replacement is unsupported",
            raw.modules_to_save.as_deref().unwrap_or_default()
        );
    }

    // rsLoRA flag is a scaling input — never defaulted (locked decision).
    let use_rslora = raw.use_rslora.ok_or_else(|| {
        anyhow::anyhow!(
            "REJECT(use_rslora): field absent — scaling inputs are never defaulted \
             (PEFT <0.7 config; re-export the adapter with peft>=0.7)"
        )
    })?;

    // layers_to_transform: accept the array form (kept for logging), reject a
    // malformed non-array form. The loader's per-LayerType gate is authority.
    let layers_to_transform = parse_layers_to_transform(&raw.layers_to_transform)?;

    if raw.r == 0 {
        bail!("REJECT(r): LoRA rank must be > 0");
    }
    if !(raw.lora_alpha.is_finite() && raw.lora_alpha > 0.0) {
        bail!("REJECT(lora_alpha): must be a finite positive number, got {}", raw.lora_alpha);
    }

    let target_modules = parse_target_modules(&raw.target_modules)?;
    for entry in &target_modules {
        validate_target_module(entry)?;
    }

    Ok(PeftAdapterConfig {
        r: raw.r,
        lora_alpha: raw.lora_alpha,
        target_modules,
        use_rslora,
        layers_to_transform,
    })
}

fn parse_layers_to_transform(v: &Option<serde_json::Value>) -> Result<Option<Vec<usize>>> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Array(arr)) => {
            let layers = arr
                .iter()
                .map(|e| {
                    e.as_u64()
                        .map(|n| n as usize)
                        .context("REJECT(layers_to_transform): entries must be non-negative integers")
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Some(layers))
        }
        Some(other) => bail!(
            "REJECT(layers_to_transform): expected null or an array of layer indices, got {other}"
        ),
    }
}

fn parse_target_modules(v: &serde_json::Value) -> Result<Vec<String>> {
    match v {
        serde_json::Value::String(s) => bail!(
            "REJECT(target_modules): string form '{s}' (e.g. 'all-linear') is unsupported — \
             re-export the adapter with an explicit module list"
        ),
        serde_json::Value::Array(arr) => {
            let mods: Vec<String> = arr
                .iter()
                .map(|e| {
                    e.as_str()
                        .map(str::to_string)
                        .context("REJECT(target_modules): entries must be strings")
                })
                .collect::<Result<_>>()?;
            if mods.is_empty() {
                bail!("REJECT(target_modules): empty list — adapter targets nothing");
            }
            Ok(mods)
        }
        other => bail!("REJECT(target_modules): expected an array of module names, got {other}"),
    }
}

/// Per-module-name allow-list gate. PEFT entries may be bare names
/// (`"k_proj"`) or full paths (`"model.layers.3.self_attn.k_proj"`); both
/// validate on the final `.`-segment. Per-`LayerType` enforcement (deltas
/// land on full-attention layers only) is the weight loader's job — this is
/// the name-level gate.
fn validate_target_module(entry: &str) -> Result<()> {
    let leaf = entry.rsplit('.').next().unwrap_or(entry);
    match leaf {
        "q_proj" => bail!(
            "REJECT(q_proj): base model uses attn_output_gate (gated/interleaved Q+gate q_proj \
             output); q_proj adapters are unsupported in v0"
        ),
        // GDN / linear-attention projections — reject both the fused
        // (`in_proj_qkvz`/`in_proj_ba`) and split (`in_proj_qkv`/`in_proj_z`/
        // `in_proj_a`/`in_proj_b`) spellings, plus `out_proj`/`conv1d`.
        "in_proj_qkvz" | "in_proj_ba" | "in_proj_qkv" | "in_proj_z" | "in_proj_a"
        | "in_proj_b" | "out_proj" | "conv1d" => bail!(
            "REJECT(gdn): target module '{leaf}' is a GDN/linear-attention projection; GDN \
             layers are unsupported in v0 (full-attention layers only)"
        ),
        "embed_tokens" | "lm_head" => bail!(
            "REJECT(embedding): target module '{leaf}' is unsupported in v0"
        ),
        m if PEFT_SUPPORTED_TARGET_MODULES.contains(&m) => Ok(()),
        other => bail!(
            "REJECT(unknown_module): target module '{other}' is not in the v0 allow-list \
             {PEFT_SUPPORTED_TARGET_MODULES:?}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_json() -> serde_json::Value {
        serde_json::json!({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Hcompany/Holo-3.1-0.8B",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "use_rslora": false,
            "use_dora": false,
            "rank_pattern": {},
            "alpha_pattern": {},
            "modules_to_save": null,
            "layers_to_transform": null,
            "target_modules": ["k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        })
    }

    #[test]
    fn happy_path_scaling_alpha_over_r() {
        let cfg = parse_peft_adapter_config(&base_json().to_string()).unwrap();
        assert_eq!(cfg.r, 16);
        assert_eq!(cfg.lora_alpha, 32.0);
        assert!(!cfg.use_rslora);
        assert_eq!(cfg.scaling(), 2.0);
        assert_eq!(cfg.target_modules.len(), 6);
    }

    #[test]
    fn rslora_scaling_alpha_over_sqrt_r() {
        let mut j = base_json();
        j["use_rslora"] = serde_json::json!(true);
        let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
        assert_eq!(cfg.scaling(), 8.0); // 32 / sqrt(16)
    }

    #[test]
    fn float_alpha_accepted() {
        let mut j = base_json();
        j["lora_alpha"] = serde_json::json!(16.5);
        let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
        assert_eq!(cfg.lora_alpha, 16.5);
    }

    #[test]
    fn layers_to_transform_array_accepted() {
        // The generated Holo fixture carries layers_to_transform=[3,7,...];
        // it must be ACCEPTED (kept for logging), not rejected.
        let mut j = base_json();
        j["layers_to_transform"] = serde_json::json!([3, 7, 11, 15, 19, 23]);
        let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
        assert_eq!(cfg.layers_to_transform, Some(vec![3, 7, 11, 15, 19, 23]));
    }

    #[test]
    fn missing_use_rslora_rejected_named() {
        let mut j = base_json();
        j.as_object_mut().unwrap().remove("use_rslora");
        let err = parse_peft_adapter_config(&j.to_string()).unwrap_err().to_string();
        assert!(err.contains("REJECT(use_rslora)"), "{err}");
    }

    #[test]
    fn q_proj_rejected_named() {
        let mut j = base_json();
        j["target_modules"] = serde_json::json!(["q_proj", "v_proj"]);
        let err = parse_peft_adapter_config(&j.to_string()).unwrap_err().to_string();
        assert!(err.contains("REJECT(q_proj)"), "{err}");
        assert!(err.contains("attn_output_gate"), "{err}");
    }

    #[test]
    fn gdn_module_rejected_named() {
        for m in ["in_proj_qkvz", "in_proj_qkv", "in_proj_z", "out_proj", "conv1d"] {
            let mut j = base_json();
            j["target_modules"] = serde_json::json!([m]);
            let err = parse_peft_adapter_config(&j.to_string()).unwrap_err().to_string();
            assert!(err.contains("REJECT(gdn)"), "{m}: {err}");
        }
    }

    #[test]
    fn all_linear_rejected_named() {
        let mut j = base_json();
        j["target_modules"] = serde_json::json!("all-linear");
        let err = parse_peft_adapter_config(&j.to_string()).unwrap_err().to_string();
        assert!(err.contains("REJECT(target_modules)"), "{err}");
    }

    #[test]
    fn dora_bias_rank_pattern_rejected_named() {
        for (key, val, tag) in [
            ("use_dora", serde_json::json!(true), "REJECT(use_dora)"),
            ("bias", serde_json::json!("lora_only"), "REJECT(bias)"),
            ("rank_pattern", serde_json::json!({"k_proj": 8}), "REJECT(rank_pattern)"),
            ("modules_to_save", serde_json::json!(["lm_head"]), "REJECT(modules_to_save)"),
            ("peft_type", serde_json::json!("ADALORA"), "REJECT(peft_type)"),
        ] {
            let mut j = base_json();
            j[key] = val;
            let err = parse_peft_adapter_config(&j.to_string()).unwrap_err().to_string();
            assert!(err.contains(tag), "{key}: {err}");
        }
    }

    #[test]
    fn full_path_target_validates_on_leaf() {
        let mut j = base_json();
        j["target_modules"] = serde_json::json!(["model.layers.3.self_attn.k_proj"]);
        let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
        assert_eq!(cfg.target_modules, vec!["model.layers.3.self_attn.k_proj"]);
    }

    #[test]
    fn zero_rank_rejected() {
        let mut j = base_json();
        j["r"] = serde_json::json!(0);
        assert!(parse_peft_adapter_config(&j.to_string()).is_err());
    }
}
