// SPDX-License-Identifier: AGPL-3.0-only

//! GGUF tensor-name → Atlas HF tensor-name translation.
//!
//! GGUF names decoder weights as `blk.N.<sub>` plus a handful of top-level
//! tensors (`token_embd`, `output_norm`, `output`). Atlas per-arch loaders ask
//! the [`crate::weights::WeightStore`] for HuggingFace names
//! (`model.layers.N.self_attn.q_proj.weight`, …). This module is the pure,
//! side-effect-free bridge between the two. It emits standard HF names for the
//! `weight_prefix = "model"` convention (see `ModelConfig::layer_prefix`).
//!
//! Expert-stacked GGUF tensors (`blk.N.ffn_{gate,up,down}_exps.weight`) are a
//! single `[n_expert, …]` tensor that Atlas expects as `num_experts` separate
//! `…experts.{E}.*` tensors. A 1:1 name map cannot express that fan-out, so
//! those names resolve to [`GgufName::ExpertStack`] and the loader is
//! responsible for slicing + naming each expert. Everything else resolves to
//! [`GgufName::Direct`] (a single HF name) or [`GgufName::Drop`] (metadata-only
//! tensors like `rope_freqs.weight` that carry no learnable weight).

/// Result of translating one GGUF tensor name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufName {
    /// A single HF tensor name to store the (dequantized) weight under.
    Direct(String),
    /// A stacked MoE expert tensor `blk.N.ffn_<which>_exps.weight`. The loader
    /// must split the leading `[n_expert]` dimension and emit, per expert `E`,
    /// `model.layers.N.mlp.experts.{E}.{proj}_proj.weight`.
    ExpertStack {
        layer: usize,
        /// `"gate"`, `"up"`, or `"down"` — the projection to suffix as
        /// `{proj}_proj`.
        proj: &'static str,
    },
    /// Tensor carries no learnable weight for Atlas (e.g. precomputed rope
    /// frequencies); the loader should skip it.
    Drop,
}

/// The HF weight prefix Atlas defaults to when `weight_prefix` is empty
/// (`ModelConfig::layer_prefix` → `model.layers.N`). Kept as a constant so the
/// non-layer names below stay in sync with the per-layer names.
const HF_PREFIX: &str = "model";

/// Translate a GGUF tensor name to its Atlas HF equivalent for architecture
/// `arch` (the value of GGUF metadata key `general.architecture`, already
/// lower-cased by the caller — e.g. `"llama"`, `"qwen2"`, `"qwen3"`,
/// `"gemma2"`). Returns `None` for names this translator does not recognize, so
/// the loader can surface an explicit error rather than silently mis-storing a
/// tensor.
pub fn translate(gguf_name: &str, arch: &str) -> Option<GgufName> {
    // Per-arch override hook. Only `default` is populated today; add arms here
    // as GGUF variants diverge (e.g. a future arch that renames `ffn_norm`).
    if let Some(overridden) = arch_override(gguf_name, arch) {
        return Some(overridden);
    }
    translate_default(gguf_name)
}

/// Architecture-specific overrides. Return `Some(_)` to short-circuit
/// [`translate_default`], `None` to fall through so [`translate_default`] runs.
fn arch_override(gguf_name: &str, arch: &str) -> Option<GgufName> {
    match arch {
        // Qwen3.5/3.6 GDN-hybrid (`general.architecture = qwen35`). llama.cpp
        // emits the linear-attention (Gated DeltaNet) projections under `ssm_*`
        // / `attn_{qkv,gate}` names and the second RMSNorm as
        // `post_attention_norm`; everything else (attn_q/k/v/output, q/k norms,
        // ffn_*, top-level tensors) matches the default translation.
        "qwen35" | "qwen3_5" => translate_qwen35_layer(gguf_name),
        _ => None,
    }
}

/// Qwen3.5/3.6-specific per-layer name remaps. Returns `None` for names the
/// default translator already handles (so they fall through), and for anything
/// unrecognized.
///
/// Layer roles are positional (every `full_attention_interval`-th layer is full
/// attention, the rest are GDN/linear-attention), but the tensor NAMES are
/// unambiguous per role: full-attention layers carry `attn_q/k/v/output`, GDN
/// layers carry `attn_qkv` (fused Q|K|V), `attn_gate` (the Z gate) and the
/// `ssm_*` family. So a pure name map suffices — no layer-index arithmetic.
///
/// GDN → Atlas HF mapping (`model.layers.N.linear_attn.*`, consumed by
/// `Qwen35DenseWeightLoader`'s `LinearAttention` arm):
///   * `attn_qkv`   → `in_proj_qkv`   (fused Q|K|V, rows = ssm_qkv_size)
///   * `attn_gate`  → `in_proj_z`     (Z gate, rows = ssm_z_size)
///   * `ssm_alpha`  → `in_proj_a`     ([num_v_heads, hidden])
///   * `ssm_beta`   → `in_proj_b`     ([num_v_heads, hidden])
///   * `ssm_conv1d` → `conv1d`
///   * `ssm_a`      → `A_log`         (bare — no `.weight`; F32 decay, per v-head)
///   * `ssm_dt.bias`→ `dt_bias`       (bare — loader keys it without `.bias`)
///   * `ssm_norm`   → `norm`
///   * `ssm_out`    → `out_proj`
fn translate_qwen35_layer(gguf_name: &str) -> Option<GgufName> {
    let rest = gguf_name.strip_prefix("blk.")?;
    let (n_str, sub) = rest.split_once('.')?;
    let layer: usize = n_str.parse().ok()?;
    let la = |suffix: &str| format!("{HF_PREFIX}.layers.{layer}.linear_attn.{suffix}");

    let hf = match sub {
        // Second RMSNorm (both layer roles). GGUF names it `post_attention_norm`
        // rather than the default translator's `ffn_norm`.
        "post_attention_norm.weight" => {
            format!("{HF_PREFIX}.layers.{layer}.post_attention_layernorm.weight")
        }
        // GDN / linear-attention (Gated DeltaNet) projections.
        "attn_qkv.weight" => la("in_proj_qkv.weight"),
        "attn_gate.weight" => la("in_proj_z.weight"),
        "ssm_alpha.weight" => la("in_proj_a.weight"),
        "ssm_beta.weight" => la("in_proj_b.weight"),
        "ssm_conv1d.weight" => la("conv1d.weight"),
        "ssm_norm.weight" => la("norm.weight"),
        "ssm_out.weight" => la("out_proj.weight"),
        // Bare (extension-less) SSM gate params — the loader keys A_log / dt_bias
        // without a `.weight` / `.bias` suffix, and loads them as F32.
        "ssm_a" => la("A_log"),
        "ssm_dt.bias" => la("dt_bias"),
        // Everything else (attn_norm, attn_q/k/v/output, attn_q/k_norm, ffn_*)
        // is handled by the default translator.
        _ => return None,
    };
    Some(GgufName::Direct(hf))
}

/// The default (llama/qwen2/qwen3/gemma-family) name translation.
fn translate_default(gguf_name: &str) -> Option<GgufName> {
    // ── Top-level (non-layer) tensors ──
    match gguf_name {
        "token_embd.weight" => {
            return Some(GgufName::Direct(format!("{HF_PREFIX}.embed_tokens.weight")));
        }
        "output_norm.weight" => {
            return Some(GgufName::Direct(format!("{HF_PREFIX}.norm.weight")));
        }
        // Untied LM head. (Tied models omit this tensor and reuse token_embd.)
        "output.weight" => return Some(GgufName::Direct("lm_head.weight".to_string())),
        // Precomputed rope frequency table — Atlas builds rope itself.
        "rope_freqs.weight" => return Some(GgufName::Drop),
        _ => {}
    }

    // ── Per-layer tensors: `blk.<N>.<sub>` ──
    let rest = gguf_name.strip_prefix("blk.")?;
    let (n_str, sub) = rest.split_once('.')?;
    let layer: usize = n_str.parse().ok()?;

    translate_layer_sub(layer, sub)
}

/// Translate the portion of a per-layer name after `blk.N.`.
fn translate_layer_sub(layer: usize, sub: &str) -> Option<GgufName> {
    // Stacked MoE experts fan out to many HF tensors — handled by the loader.
    match sub {
        "ffn_gate_exps.weight" => return Some(GgufName::ExpertStack { layer, proj: "gate" }),
        "ffn_up_exps.weight" => return Some(GgufName::ExpertStack { layer, proj: "up" }),
        "ffn_down_exps.weight" => return Some(GgufName::ExpertStack { layer, proj: "down" }),
        _ => {}
    }

    // Split off the `.weight` / `.bias` extension so biases (qwen2 QKV) map for
    // free alongside their weight.
    let (stem, ext) = match sub.rsplit_once('.') {
        Some((stem, ext @ ("weight" | "bias"))) => (stem, ext),
        // Unknown extension → unrecognized; let the caller error explicitly.
        _ => return None,
    };

    // Map the GGUF sub-stem to the HF sub-path (without extension).
    let hf_sub = match stem {
        // Norms
        "attn_norm" => "input_layernorm",
        "ffn_norm" => "post_attention_layernorm",
        // Attention projections
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        // Qwen3 per-head QK RMSNorm
        "attn_q_norm" => "self_attn.q_norm",
        "attn_k_norm" => "self_attn.k_norm",
        // Dense MLP
        "ffn_gate" => "mlp.gate_proj",
        "ffn_up" => "mlp.up_proj",
        "ffn_down" => "mlp.down_proj",
        // MoE router
        "ffn_gate_inp" => "mlp.gate",
        _ => return None,
    };

    Some(GgufName::Direct(format!(
        "{HF_PREFIX}.layers.{layer}.{hf_sub}.{ext}"
    )))
}

/// Expand an [`GgufName::ExpertStack`] into the concrete per-expert HF name for
/// expert `e`. The loader calls this while slicing the stacked tensor so the
/// naming convention lives next to the translation table it mirrors.
pub fn expert_name(layer: usize, proj: &str, e: usize) -> String {
    format!("{HF_PREFIX}.layers.{layer}.mlp.experts.{e}.{proj}_proj.weight")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(name: &str) -> Option<GgufName> {
        Some(GgufName::Direct(name.to_string()))
    }

    #[test]
    fn top_level_tensors() {
        assert_eq!(
            translate("token_embd.weight", "llama"),
            direct("model.embed_tokens.weight")
        );
        assert_eq!(
            translate("output_norm.weight", "llama"),
            direct("model.norm.weight")
        );
        assert_eq!(translate("output.weight", "llama"), direct("lm_head.weight"));
        assert_eq!(translate("rope_freqs.weight", "llama"), Some(GgufName::Drop));
    }

    #[test]
    fn attention_projections() {
        assert_eq!(
            translate("blk.0.attn_q.weight", "qwen3"),
            direct("model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            translate("blk.7.attn_k.weight", "qwen3"),
            direct("model.layers.7.self_attn.k_proj.weight")
        );
        assert_eq!(
            translate("blk.7.attn_v.weight", "qwen3"),
            direct("model.layers.7.self_attn.v_proj.weight")
        );
        assert_eq!(
            translate("blk.7.attn_output.weight", "qwen3"),
            direct("model.layers.7.self_attn.o_proj.weight")
        );
    }

    #[test]
    fn norms_and_mlp() {
        assert_eq!(
            translate("blk.3.attn_norm.weight", "llama"),
            direct("model.layers.3.input_layernorm.weight")
        );
        assert_eq!(
            translate("blk.3.ffn_norm.weight", "llama"),
            direct("model.layers.3.post_attention_layernorm.weight")
        );
        assert_eq!(
            translate("blk.3.ffn_gate.weight", "llama"),
            direct("model.layers.3.mlp.gate_proj.weight")
        );
        assert_eq!(
            translate("blk.3.ffn_up.weight", "llama"),
            direct("model.layers.3.mlp.up_proj.weight")
        );
        assert_eq!(
            translate("blk.3.ffn_down.weight", "llama"),
            direct("model.layers.3.mlp.down_proj.weight")
        );
    }

    #[test]
    fn qwen3_qk_norms() {
        assert_eq!(
            translate("blk.11.attn_q_norm.weight", "qwen3"),
            direct("model.layers.11.self_attn.q_norm.weight")
        );
        assert_eq!(
            translate("blk.11.attn_k_norm.weight", "qwen3"),
            direct("model.layers.11.self_attn.k_norm.weight")
        );
    }

    #[test]
    fn qwen2_qkv_biases() {
        assert_eq!(
            translate("blk.2.attn_q.bias", "qwen2"),
            direct("model.layers.2.self_attn.q_proj.bias")
        );
        assert_eq!(
            translate("blk.2.attn_k.bias", "qwen2"),
            direct("model.layers.2.self_attn.k_proj.bias")
        );
        assert_eq!(
            translate("blk.2.attn_v.bias", "qwen2"),
            direct("model.layers.2.self_attn.v_proj.bias")
        );
    }

    #[test]
    fn moe_router_and_expert_stack() {
        assert_eq!(
            translate("blk.4.ffn_gate_inp.weight", "qwen3"),
            direct("model.layers.4.mlp.gate.weight")
        );
        assert_eq!(
            translate("blk.4.ffn_gate_exps.weight", "qwen3"),
            Some(GgufName::ExpertStack { layer: 4, proj: "gate" })
        );
        assert_eq!(
            translate("blk.4.ffn_up_exps.weight", "qwen3"),
            Some(GgufName::ExpertStack { layer: 4, proj: "up" })
        );
        assert_eq!(
            translate("blk.4.ffn_down_exps.weight", "qwen3"),
            Some(GgufName::ExpertStack { layer: 4, proj: "down" })
        );
    }

    #[test]
    fn expert_name_expansion() {
        assert_eq!(
            expert_name(4, "gate", 17),
            "model.layers.4.mlp.experts.17.gate_proj.weight"
        );
        assert_eq!(
            expert_name(0, "down", 0),
            "model.layers.0.mlp.experts.0.down_proj.weight"
        );
    }

    #[test]
    fn robust_layer_index_parsing() {
        assert_eq!(
            translate("blk.123.attn_q.weight", "llama"),
            direct("model.layers.123.self_attn.q_proj.weight")
        );
        assert_eq!(
            translate("blk.5.attn_q_norm.weight", "qwen3"),
            direct("model.layers.5.self_attn.q_norm.weight")
        );
    }

    #[test]
    fn unrecognized_names_return_none() {
        assert_eq!(translate("blk.x.attn_q.weight", "llama"), None);
        assert_eq!(translate("blk.0.something_new.weight", "llama"), None);
        assert_eq!(translate("blk.0.attn_q.scale", "llama"), None);
        assert_eq!(translate("garbage", "llama"), None);
    }

    #[test]
    fn qwen35_gdn_and_norm_mappings() {
        // Second RMSNorm renamed vs. the default `ffn_norm`.
        assert_eq!(
            translate("blk.0.post_attention_norm.weight", "qwen35"),
            direct("model.layers.0.post_attention_layernorm.weight")
        );
        // GDN / linear-attention projections → `linear_attn.*`.
        assert_eq!(
            translate("blk.0.attn_qkv.weight", "qwen35"),
            direct("model.layers.0.linear_attn.in_proj_qkv.weight")
        );
        assert_eq!(
            translate("blk.0.attn_gate.weight", "qwen35"),
            direct("model.layers.0.linear_attn.in_proj_z.weight")
        );
        assert_eq!(
            translate("blk.2.ssm_alpha.weight", "qwen35"),
            direct("model.layers.2.linear_attn.in_proj_a.weight")
        );
        assert_eq!(
            translate("blk.2.ssm_beta.weight", "qwen35"),
            direct("model.layers.2.linear_attn.in_proj_b.weight")
        );
        assert_eq!(
            translate("blk.4.ssm_conv1d.weight", "qwen35"),
            direct("model.layers.4.linear_attn.conv1d.weight")
        );
        assert_eq!(
            translate("blk.4.ssm_norm.weight", "qwen35"),
            direct("model.layers.4.linear_attn.norm.weight")
        );
        assert_eq!(
            translate("blk.4.ssm_out.weight", "qwen35"),
            direct("model.layers.4.linear_attn.out_proj.weight")
        );
        // Bare (extension-less) F32 gate params keyed without a suffix.
        assert_eq!(
            translate("blk.1.ssm_a", "qwen35"),
            direct("model.layers.1.linear_attn.A_log")
        );
        assert_eq!(
            translate("blk.1.ssm_dt.bias", "qwen35"),
            direct("model.layers.1.linear_attn.dt_bias")
        );
    }

    #[test]
    fn qwen35_shared_names_fall_through_to_default() {
        // Full-attention + FFN + norms shared with the default translator.
        assert_eq!(
            translate("blk.3.attn_q.weight", "qwen35"),
            direct("model.layers.3.self_attn.q_proj.weight")
        );
        assert_eq!(
            translate("blk.3.attn_output.weight", "qwen35"),
            direct("model.layers.3.self_attn.o_proj.weight")
        );
        assert_eq!(
            translate("blk.3.attn_q_norm.weight", "qwen35"),
            direct("model.layers.3.self_attn.q_norm.weight")
        );
        assert_eq!(
            translate("blk.0.attn_norm.weight", "qwen35"),
            direct("model.layers.0.input_layernorm.weight")
        );
        assert_eq!(
            translate("blk.0.ffn_down.weight", "qwen35"),
            direct("model.layers.0.mlp.down_proj.weight")
        );
        assert_eq!(
            translate("token_embd.weight", "qwen35"),
            direct("model.embed_tokens.weight")
        );
        assert_eq!(translate("output.weight", "qwen35"), direct("lm_head.weight"));
    }

    #[test]
    fn unknown_arch_falls_through_to_default() {
        assert_eq!(
            translate("blk.0.attn_q.weight", "some_future_arch"),
            direct("model.layers.0.self_attn.q_proj.weight")
        );
    }
}
