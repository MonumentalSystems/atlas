// SPDX-License-Identifier: AGPL-3.0-only

//! Stable, config-derived model fingerprint for the SHARED paging peer.
//!
//! The paging peer (atlas-cache-peer) owns ONE residency map shared across
//! every fleet client, keyed purely by the u64 the client sends — so the
//! per-model namespace folded into each key is the ONLY thing preventing two
//! models from silently serving each other's recurrent state. That makes the
//! namespace a **durable on-disk contract**: the peer's NVMe swap file
//! outlives client rebuilds and toolchains, so the same model config must
//! derive the same u64 forever.
//!
//! ## Why FNV-1a/64, vendored
//!
//! `std::hash::DefaultHasher` documents its algorithm as unspecified and "not
//! to be relied upon over releases" — a toolchain bump may silently rotate
//! every persisted key (total cache miss) or collide with a stale namespace
//! (silent wrong-state corruption). FNV-1a/64 is fully specified (offset
//! basis `0xcbf29ce484222325`, prime `0x100000001b3`), consumes a byte stream
//! (endianness-free), and is vendored here (~10 lines) so no crate version
//! bump can ever change the value. The test file pins the primitive to the
//! published FNV reference vectors and the full fingerprint of known configs
//! to frozen literals. Collision quality is irrelevant at this input size (a
//! handful of fleet model configs, not an adversarial keyspace).
//!
//! ## Encoding contract
//!
//! The hash input is an injective canonical encoding: tagged records, u64s as
//! `[tag][8-byte LE]`, strings length-prefixed `[tag][4-byte LE len][bytes]`
//! (so `("ab","c")` never collides with `("a","bc")`). Field set and order
//! are FROZEN behind [`FP_VERSION`]; any change to either is a deliberate
//! fleet-wide cache flush and must bump the version. `kv_layer_dims` is
//! loader-populated and order-sensitive — derive() must run after loader
//! post-processing (it does: the only call sites are in
//! `TransformerModel::new`).
//!
//! The runtime KV-cache dtype (`--kv-cache-dtype`) is deliberately EXCLUDED:
//! SSM h/conv state is FP32 by construction regardless of KV dtype, so two
//! serves of one model with different KV dtypes produce byte-identical SSM
//! blobs and SHOULD share the warm cache. A future KV paging tier must fold
//! its own dtype/block_size mix-in at its own call site.
//!
//! Note: `wire()`'s splitmix fold is bijective per namespace, but two
//! DIFFERENT namespaces can map two (key, ns) pairs to one wire key with
//! ~2^-64 probability — acceptable, and NOT to be "strengthened" into a wire
//! -format change.

use std::num::NonZeroU64;

use anyhow::{Result, anyhow, bail};
use atlas_core::config::ModelConfig;

/// Bump = deliberate fleet-wide cache-key rotation (document it).
// v2 (2026-07-10): added hidden_size / num_attention_heads / intermediate_size /
// moe_intermediate_size / num_experts_per_tok (tags 0x16-0x1a). v1 omitted them, so
// two distinct models differing only in residual/FFN width collided. Bumping the
// version is a DELIBERATE, one-time fleet cache-key rotation (greenfield: no shim).
pub(crate) const FP_VERSION: u64 = 2;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a/64 over a byte stream. Deterministic across toolchains, platforms
/// and rebuilds — see the module header for why this is load-bearing.
pub(crate) const fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        h = (h ^ bytes[i] as u64).wrapping_mul(FNV_PRIME);
        i += 1;
    }
    h
}

/// splitmix64 finalizer over `a ^ b·GOLDEN` — domain-separated mixing (the
/// decode namespace is `mix64(fingerprint, DECODE_DOMAIN)`). Same finalizer
/// family as `PagingSnapshotStore::wire()`; deterministic forever.
pub(crate) fn mix64(a: u64, b: u64) -> u64 {
    let mut h = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

fn put_u64(buf: &mut Vec<u8>, tag: u8, v: u64) {
    buf.push(tag);
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_str(buf: &mut Vec<u8>, tag: u8, s: &str) {
    buf.push(tag);
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Stable identity of "the bytes this model's SSM tier produces", derived
/// from the loaded [`ModelConfig`] geometry + quantization identity +
/// `blob_bytes`. Non-zero by construction (a 0 namespace — the old silent
/// passthrough — is unrepresentable downstream).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ModelFingerprint(NonZeroU64);

impl ModelFingerprint {
    /// Derive the fingerprint and log it at INFO so operators can see and pin
    /// it. `ATLAS_MODEL_ID` (optional) is an extra salt for the one case
    /// geometry cannot distinguish: a fine-tune with byte-identical config —
    /// unset (the common case) it contributes an empty record and never
    /// rotates keys.
    pub(crate) fn derive(cfg: &ModelConfig, blob_bytes: usize) -> Result<Self> {
        let model_id = std::env::var("ATLAS_MODEL_ID").unwrap_or_default();
        let fp = Self::derive_with_id(cfg, blob_bytes, &model_id)?;
        tracing::info!(
            "SSM tier model fingerprint = {:#018x} (model_type={}, blob_bytes={blob_bytes}, \
             ATLAS_MODEL_ID={model_id:?}); pin with ATLAS_SSM_SWAP_NS / ATLAS_SSM_DECODE_NS",
            fp.get(),
            cfg.model_type,
        );
        // The fingerprint is GEOMETRY-ONLY. It cannot distinguish two checkpoints
        // with byte-identical config — a fine-tune, an RL variant, a continued
        // pre-train of one base. Those derive the same namespace and would share a
        // shared peer's cache. We cannot fail fast (config carries no weight
        // identity), so warn loudly exactly when it matters: a SHARED paging peer
        // is selected and the operator gave neither a salt nor an explicit ns.
        let shared_peer = std::env::var("ATLAS_SSM_SWAP").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_SSM_DECODE_TIER").ok().as_deref() == Some("peer");
        let overridden = std::env::var_os("ATLAS_SSM_SWAP_NS").is_some()
            || std::env::var_os("ATLAS_SSM_DECODE_NS").is_some();
        if shared_peer && model_id.is_empty() && !overridden {
            tracing::warn!(
                "shared paging peer selected but ATLAS_MODEL_ID is unset: the fingerprint is \
                 derived from config GEOMETRY ONLY, so two checkpoints with identical config \
                 (fine-tunes, RL variants, continued pre-train of one base) will SHARE cache \
                 keys and silently cross-serve recurrent state. Set ATLAS_MODEL_ID to a stable \
                 per-checkpoint string (or set ATLAS_SSM_SWAP_NS / ATLAS_SSM_DECODE_NS \
                 explicitly) when co-locating such models on one peer."
            );
        }
        Ok(fp)
    }

    /// KV-paging fingerprint (Step B, `ATLAS_KV_PAGING`): the SAME canonical
    /// per-model encoding as the SSM tier, derived with the KV CONVENTION
    /// `blob_bytes = 0` (tag 0x40 = 0 marks the KV instance; SSM instances
    /// put their real, non-zero blob size there — so the two never share a
    /// value, and attention-only models with no SSM tier still get an fp).
    /// The KV tier folds its full block geometry + dtype + a per-client salt
    /// DOWNSTREAM at its own call site (`spark_storage::kv_paging::ns`),
    /// exactly as the module doc above prescribes — do NOT add KV fields
    /// here (that would rotate SSM keys).
    pub(crate) fn derive_kv(cfg: &ModelConfig) -> Result<Self> {
        let model_id = std::env::var("ATLAS_MODEL_ID").unwrap_or_default();
        let fp = Self::derive_with_id(cfg, 0, &model_id)?;
        tracing::info!(
            "KV paging model fingerprint = {:#018x} (model_type={}, \
             ATLAS_MODEL_ID={model_id:?}); folded into the ATLAS_KV_PAGING namespace",
            fp.get(),
            cfg.model_type,
        );
        Ok(fp)
    }

    /// Pure derivation (no env, no logging) — the canonical encoding.
    /// FROZEN: field set + order changes require an FP_VERSION bump.
    pub(crate) fn derive_with_id(
        cfg: &ModelConfig,
        blob_bytes: usize,
        model_id: &str,
    ) -> Result<Self> {
        // Defensive PCND bail: never legitimate after parse_config. Without a
        // real fingerprint a shared peer would need an explicit override.
        if cfg.model_type.is_empty() && cfg.num_hidden_layers == 0 {
            bail!(
                "cannot derive a model fingerprint: empty model_type and zero geometry; \
                 fix the model config or set ATLAS_SSM_SWAP_NS / ATLAS_SSM_DECODE_NS \
                 to explicit non-zero u64 namespaces"
            );
        }
        let (qm, qa, qf) = cfg.quantization_config.as_ref().map_or(("", "", ""), |q| {
            (
                q.quant_method.as_str(),
                q.quant_algo.as_str(),
                q.format.as_str(),
            )
        });
        let mut buf = Vec::with_capacity(256);
        put_u64(&mut buf, 0x00, FP_VERSION);
        put_str(&mut buf, 0x01, &cfg.model_type);
        // Quant identity: an NVFP4 build and a bf16 build of one checkpoint
        // share geometry but produce different recurrent state values.
        put_str(&mut buf, 0x02, qm);
        put_str(&mut buf, 0x03, qa);
        put_str(&mut buf, 0x04, qf);
        put_str(&mut buf, 0x05, model_id);
        for (tag, v) in [
            (0x10, cfg.num_hidden_layers),
            (0x11, cfg.num_ssm_layers()),
            (0x12, cfg.num_attention_layers()),
            (0x13, cfg.head_dim),
            (0x14, cfg.num_key_value_heads),
            (0x15, cfg.num_experts),
            // FP_VERSION 2: residual-stream and FFN widths. Omitting these was a
            // BLOCKING defect — two distinct models differing ONLY in
            // `hidden_size` / `num_attention_heads` hashed identically and would
            // have silently cross-served recurrent state on a shared paging peer
            // (the exact failure this fingerprint exists to prevent). `blob_bytes`
            // does NOT capture them: it is `num_ssm_layers * (h + conv)` bytes,
            // which is SSM-state-only and independent of the residual width.
            (0x16, cfg.hidden_size),
            (0x17, cfg.num_attention_heads),
            (0x18, cfg.intermediate_size),
            (0x19, cfg.moe_intermediate_size),
            (0x1a, cfg.num_experts_per_tok),
            // SSM state geometry — included alongside blob_bytes because the
            // blob size is a lossy product of these.
            (0x20, cfg.linear_num_key_heads),
            (0x21, cfg.linear_key_head_dim),
            (0x22, cfg.linear_num_value_heads),
            (0x23, cfg.linear_value_head_dim),
            (0x24, cfg.linear_conv_kernel_dim),
            (0x25, cfg.mamba_num_heads),
            (0x26, cfg.mamba_head_dim),
            (0x27, cfg.ssm_state_size),
            (0x28, cfg.n_groups),
            // SSM h/conv state element size: FP32 today (the ×4 hardcoded in
            // ssm_h_state_bytes/ssm_conv_state_bytes). Fingerprinted so a
            // future non-FP32 SSM state rotates keys instead of colliding.
            (0x30, 4),
            (0x40, blob_bytes),
        ] {
            put_u64(&mut buf, tag, v as u64);
        }
        // Heterogeneous-attention per-layer (kv_heads, head_dim) overrides
        // (Gemma-4). Loader-populated; canonical order = layer order.
        put_u64(&mut buf, 0x50, cfg.kv_layer_dims.len() as u64);
        for &(kvh, hd) in &cfg.kv_layer_dims {
            put_u64(&mut buf, 0x51, kvh as u64);
            put_u64(&mut buf, 0x52, hd as u64);
        }
        let h = fnv1a_64(&buf);
        // Zero-avoidance (p = 2^-64): keep NonZeroU64 total + deterministic.
        Ok(Self(
            NonZeroU64::new(h).unwrap_or(NonZeroU64::new(FNV_OFFSET).unwrap()),
        ))
    }

    pub(crate) fn get(self) -> u64 {
        self.0.get()
    }

    pub(crate) fn nonzero(self) -> NonZeroU64 {
        self.0
    }
}

/// Marconi swap namespace: `ATLAS_SSM_SWAP_NS` override (strict — junk or 0
/// is a startup ERROR, never a silent fallthrough) else the fingerprint.
pub(crate) fn resolve_swap_ns(fp: ModelFingerprint) -> Result<NonZeroU64> {
    resolve_ns_from(
        std::env::var("ATLAS_SSM_SWAP_NS").ok().as_deref(),
        "ATLAS_SSM_SWAP_NS",
        fp.nonzero(),
    )
}

/// Decode namespace: `ATLAS_SSM_DECODE_NS` override (same strictness) else
/// `mix64(fingerprint, DECODE_DOMAIN)` — the DOMAIN SEPARATOR must survive
/// (decode + Marconi share ONE peer residency whenever blob_bytes match) and
/// so must model identity (the bare constant collided two models' decode
/// spills). Never DECODE_DOMAIN alone, never the fingerprint alone.
pub(crate) fn resolve_decode_ns(fp: ModelFingerprint) -> Result<NonZeroU64> {
    let mixed = mix64(fp.get(), atlas_kernels::DECODE_DOMAIN);
    // Zero-avoidance (p = 2^-64). It must NOT fall back to `fp.nonzero()`: that is
    // exactly the Marconi swap namespace, so the decode tier would alias onto it
    // and the two tiers of one model would cross-serve. Fall back to the domain
    // constant instead — distinct from the fingerprint by construction.
    let derived = NonZeroU64::new(mixed).unwrap_or_else(|| {
        NonZeroU64::new(atlas_kernels::DECODE_DOMAIN).expect("DECODE_DOMAIN is a non-zero constant")
    });
    resolve_ns_from(
        std::env::var("ATLAS_SSM_DECODE_NS").ok().as_deref(),
        "ATLAS_SSM_DECODE_NS",
        derived,
    )
}

/// Env-free core (unit-testable without process-global setenv races).
pub(crate) fn resolve_ns_from(
    override_raw: Option<&str>,
    var: &str,
    derived: NonZeroU64,
) -> Result<NonZeroU64> {
    match override_raw {
        Some(raw) => parse_ns(var, raw),
        None => Ok(derived),
    }
}

/// Strict override parser: decimal or `0x`-hex u64. Unparseable values are a
/// hard error (PCND — the old code silently `.ok()`-swallowed a mistyped
/// override into the model-blind default). 0 is a hard error: the ns=0
/// passthrough is removed, a shared peer must always be namespaced.
pub(crate) fn parse_ns(var: &str, raw: &str) -> Result<NonZeroU64> {
    let s = raw.trim();
    let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => s.parse::<u64>(),
    };
    let v = parsed.map_err(|e| {
        anyhow!("{var}={raw:?} is not a valid u64 namespace (decimal or 0x-hex): {e}")
    })?;
    NonZeroU64::new(v).ok_or_else(|| {
        anyhow!(
            "{var}=0 is invalid: the ns=0 passthrough is removed (it silently \
             cross-served state between models on a shared peer); unset {var} \
             to use the derived model fingerprint (logged at INFO on startup)"
        )
    })
}

#[cfg(test)]
#[path = "fingerprint_tests.rs"]
mod tests;
