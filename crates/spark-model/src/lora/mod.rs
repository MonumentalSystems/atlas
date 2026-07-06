// SPDX-License-Identifier: AGPL-3.0-only

//! Startup-static PEFT LoRA adapter: remap/validate/pack into the
//! fixed-address rank-padded pool. v0 = one adapter, slot 0, always on.
//!
//! NAMING: everything here is `Peft*`/`adapter_*`/`Lora*` (adapter sense) —
//! `kv_lora_rank`/`q_lora_rank` (atlas-core/src/config.rs:182-207) are MLA
//! vocabulary, not this.
//!
//! NOTE on leaks: the intermediate `WeightStore` device copies of the
//! unpadded A/B tensors become garbage after pool packing and are never
//! freed (no dealloc on weight structs anywhere in Atlas). Accepted at
//! holo adapter scale (~tens of MiB).

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use atlas_core::config::{LayerType, ModelConfig, PeftAdapterConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::layers::ops::lora_delta::LoraPair;
use crate::weight_map::DenseWeight;

pub mod rdma_stage;

const BF16_BYTES: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoraModule {
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

impl LoraModule {
    pub const ALL: [LoraModule; 6] = [
        Self::KProj,
        Self::VProj,
        Self::OProj,
        Self::GateProj,
        Self::UpProj,
        Self::DownProj,
    ];

    /// PEFT suffix name (target_modules vocabulary).
    pub fn peft_name(&self) -> &'static str {
        match self {
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
        }
    }

    /// (out_dim, in_dim) of the base projection. Holo-3.1-0.8B (verified
    /// against the checkpoint header): k/v `[512,1024]`, o `[1024,2048]`,
    /// gate/up `[3584,1024]`, down `[1024,3584]`.
    pub fn dims(&self, cfg: &ModelConfig) -> (usize, usize) {
        let h = cfg.hidden_size;
        match self {
            Self::KProj | Self::VProj => (cfg.num_key_value_heads * cfg.head_dim, h),
            Self::OProj => (h, cfg.num_attention_heads * cfg.head_dim),
            Self::GateProj | Self::UpProj => (cfg.intermediate_size, h),
            Self::DownProj => (h, cfg.intermediate_size),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdapterAb {
    A = 0,
    B = 1,
}

/// One full-attention layer's adapted modules. `None` = module not adapted.
/// Pairs are the CANONICAL [`LoraPair`] from `layers::ops::lora_delta`
/// (Copy — installed by copy into the layer structs at model build).
///
/// `Clone` (LoraPair is Copy) so a slot's layers can be re-installed onto the
/// layer structs on a runtime rotation (`set_active_lora`).
#[derive(Clone)]
pub struct LoraLayerWeights {
    pub layer_idx: usize,
    pub k_proj: Option<LoraPair>,
    pub v_proj: Option<LoraPair>,
    pub o_proj: Option<LoraPair>,
    pub gate_proj: Option<LoraPair>,
    pub up_proj: Option<LoraPair>,
    pub down_proj: Option<LoraPair>,
}

/// One packed pool slot: a resident adapter's own name/config + its per-layer
/// pairs (a/b DevicePtrs into that slot's byte sub-region of the shared pool).
/// `layers` is GLOBAL-layer-indexed (len = num_hidden_layers), the same index
/// the install walk uses.
#[derive(Clone)]
pub struct AdapterSlot {
    pub name: String,
    pub adapter_config: PeftAdapterConfig,
    pub layers: Vec<Option<LoraLayerWeights>>,
}

/// One adapter to pack, for the multi-adapter entry point. `store` is the
/// adapter's on-device BF16 `WeightStore` (host F16/F32→BF16 already done by
/// `spark_runtime::weights::adapter::load_adapter_safetensors`).
pub struct LoraAdapterInput<'a> {
    pub name: String,
    pub store: &'a WeightStore,
    pub peft: PeftAdapterConfig,
}

/// The loaded adapter set: one fixed-address rank-padded pool holding up to
/// `max_loras` equal-size slots, one [`AdapterSlot`] per resident adapter, and
/// per-module `[max_loras]` device u64 pointer tables (the frozen M2 BGMV
/// contract — filled index k for each packed slot, NULL for the rest).
///
/// Single-adapter runs pack exactly one slot (`slots.len() == 1`, `active == 0`)
/// — byte-identical to the pre-multi-adapter path. `name`/`adapter_config` mirror
/// the ACTIVE slot for logs/status; the install walk reads [`Self::active_layers`].
pub struct LoraWeights {
    /// Name of the ACTIVE adapter (mirrors `slots[active].name`).
    pub name: String,
    /// Config of the ACTIVE adapter (mirrors `slots[active].adapter_config`).
    pub adapter_config: PeftAdapterConfig,
    pub max_rank: usize,
    pub max_loras: usize,
    /// One fixed-address allocation holding every padded A/B for every slot.
    pub pool: DevicePtr,
    pub pool_bytes: usize,
    /// The resident adapters, slot-indexed (`slots[k]` lives at pool byte
    /// offset `k * pool_slot_bytes`). `len() <= max_loras`.
    pub slots: Vec<AdapterSlot>,
    /// Index into `slots` of the currently-active adapter (0 at load).
    pub active: usize,
    /// key = (global_layer_idx, module) → (a_table, b_table); each table is
    /// a device `[max_loras]` u64 array, NULL (0) = base-only slot.
    pub tables: BTreeMap<(usize, LoraModule), (DevicePtr, DevicePtr)>,
    /// The parallel `[max_loras]` device f32 SCALE table the bgmv reads,
    /// indexed by slot: `scale_table[k]` = `slots[k].adapter_config.scaling()`
    /// (alpha/r, or alpha/√r under rsLoRA — the same per-adapter scale that
    /// rides each [`LoraPair`]), 0.0 for unpacked slots. Scale is per-ADAPTER
    /// (not per-module), so ONE table suffices. Built once at pool pack time
    /// alongside the a/b tables (load-time-fixed → graph-safe kernel arg).
    pub scale_table: DevicePtr,
}

/// Build the per-step `seq_slot[padded_n]` host buffer the batched bgmv reads,
/// from each real sequence's `adapter_slot`. Resolution rules (graph-safe:
/// contents vary per step, buffer address is fixed):
///   real row i (< n): `adapter_slots[i]` if `>= 0`, else `active` — a request
///     with no `adapter` field carries `-1` and DEFERS to the installed active
///     adapter, so a single global adapter (or a rotate re-point) applies to
///     every default row exactly like the n==1 path.
///   pad row i (n..padded_n): `-1` — base / no delta (bgmv early-returns).
/// A row that explicitly names the base model (some future `-1`-means-base
/// convention) is out of scope here; `-1` uniformly means "defer to active".
pub fn build_seq_slot_host(adapter_slots: &[i32], padded_n: usize, active: i32) -> Vec<i32> {
    let n = adapter_slots.len();
    (0..padded_n)
        .map(|i| {
            if i < n {
                let s = adapter_slots[i];
                if s >= 0 { s } else { active }
            } else {
                -1
            }
        })
        .collect()
}

/// Pure per-slot scale vector for the `[max_loras]` f32 scale table: entry `k`
/// = adapter `k`'s `scaling()` (alpha/r, or alpha/√r under rsLoRA — read per
/// adapter, never defaulted), 0.0 for unpacked slots `k >= adapters.len()`.
/// Split out for unit testing (the device upload is a thin wrapper).
pub(crate) fn scale_table_values(adapters: &[LoraAdapterInput<'_>], max_loras: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; max_loras];
    for (k, a) in adapters.iter().enumerate() {
        v[k] = a.peft.scaling();
    }
    v
}

impl LoraWeights {
    /// The active slot's per-layer pairs (GLOBAL-layer-indexed) — what the
    /// install walk copies onto the layer structs.
    pub fn active_layers(&self) -> &[Option<LoraLayerWeights>] {
        &self.slots[self.active].layers
    }

    /// Resolve an adapter NAME to its slot index (for runtime rotation).
    pub fn slot_of(&self, name: &str) -> Option<usize> {
        self.slots.iter().position(|s| s.name == name)
    }

    /// All resident adapter names in slot order (for `/v1/models`).
    pub fn adapter_names(&self) -> Vec<String> {
        self.slots.iter().map(|s| s.name.clone()).collect()
    }
}

/// PEFT key → (layer, module, A|B). Every unsupported shape is a NAMED
/// hard rejection — never a skip. Prefix-agnostic on purpose: the Holo
/// base checkpoint keys are `model.language_model.layers.{i}.*`
/// (weight_prefix auto-detected server-side), but a PEFT trainer wrapping
/// the text trunk emits `model.layers.{i}.*`; both carry the layer index
/// right after ".layers.".
pub fn classify_key(key: &str, cfg: &ModelConfig) -> Result<(usize, LoraModule, AdapterAb)> {
    let stripped = key.strip_prefix("base_model.model.").ok_or_else(|| {
        anyhow!("REJECT[not-peft-key]: '{key}' lacks the 'base_model.model.' PEFT prefix")
    })?;
    if stripped.contains("lora_embedding_") {
        bail!("REJECT[embedding-lora]: '{key}' — embed_tokens/lm_head LoRA is out of v0 scope");
    }
    let (module_path, ab) = if let Some(p) = stripped.strip_suffix(".lora_A.weight") {
        (p, AdapterAb::A)
    } else if let Some(p) = stripped.strip_suffix(".lora_B.weight") {
        (p, AdapterAb::B)
    } else {
        bail!(
            "REJECT[unrecognized-tensor]: '{key}' is not a lora_A/lora_B weight \
             (modules_to_save exports and old '.lora_A.<adapter>.weight' layouts \
             are not supported in v0)"
        );
    };
    let (_prefix, rest) = module_path.split_once(".layers.").ok_or_else(|| {
        anyhow!("REJECT[non-layer-module]: '{key}' targets '{module_path}' outside the layer stack")
    })?;
    let (idx_str, tail) = rest
        .split_once('.')
        .ok_or_else(|| anyhow!("REJECT[malformed-key]: '{key}'"))?;
    let layer_idx: usize = idx_str
        .parse()
        .map_err(|_| anyhow!("REJECT[malformed-layer-index]: '{key}'"))?;
    if layer_idx >= cfg.num_hidden_layers {
        bail!(
            "REJECT[layer-out-of-range]: '{key}' targets layer {layer_idx} \
             (model has {})",
            cfg.num_hidden_layers
        );
    }
    let module = match tail {
        "self_attn.q_proj" => bail!(
            "REJECT[gated-q-proj]: '{key}' — q_proj is excluded in v0: \
             attn_output_gate=true makes q_proj emit interleaved [Q|gate] \
             (out = 2·q_heads·head_dim = {}); a PEFT q_proj delta maps only to \
             the Q half and needs segment-offset expand support (M3+)",
            2 * cfg.num_attention_heads * cfg.head_dim
        ),
        "self_attn.k_proj" => LoraModule::KProj,
        "self_attn.v_proj" => LoraModule::VProj,
        "self_attn.o_proj" => LoraModule::OProj,
        "mlp.gate_proj" => LoraModule::GateProj,
        "mlp.up_proj" => LoraModule::UpProj,
        "mlp.down_proj" => LoraModule::DownProj,
        t if t.starts_with("linear_attn.") => bail!(
            "REJECT[gdn-target]: '{key}' — GDN/linear-attention projections \
             (in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj) are \
             rejected until an exact-replay parity harness exists"
        ),
        other => bail!("REJECT[unsupported-module]: '{key}' targets '{other}'"),
    };
    match cfg.layer_type(layer_idx) {
        LayerType::FullAttention => {}
        lt => bail!(
            "REJECT[non-full-attention-layer]: '{key}' targets layer {layer_idx} \
             ({lt:?}); v0 applies LoRA only on the full-attention layers \
             {:?}. NOTE: dense mlp.* exists on the GDN layers too — train with \
             layers_to_transform=[3,7,11,15,19,23] to produce a loadable adapter",
            full_attention_layers(cfg)
        ),
    }
    Ok((layer_idx, module, ab))
}

/// Permanent LoRA debugging hatch: `ATLAS_LORA_EAGER=1` (or `true`) forces
/// eager decode (no CUDA-graph capture) when an adapter is active, so
/// graph-vs-eager output parity can be compared in the field. Read ONCE —
/// the decode graph gate runs per token.
pub fn lora_eager_env() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ATLAS_LORA_EAGER").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// `ATLAS_LORA_ROTATE=1` (or `true`) ARMS runtime adapter rotation: it forces
/// eager decode (no CUDA-graph capture) so a `set_active_lora` re-point is
/// immediately live (eager-on-rotate — the graph would otherwise replay the
/// previously-captured slot pointers). A pool with >1 resident adapter arms
/// this automatically (see `TransformerModel::lora_rotatable`), so this env is
/// only needed to arm rotation on a SINGLE resident adapter (e.g. RDMA
/// slot-swap-in-place). Unset + a single startup adapter = today's behaviour
/// exactly (graphs ON, slot-0 pointers baked).
pub fn lora_rotate_env() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ATLAS_LORA_ROTATE").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// `$ATLAS_LORA_PEER` (host:port of an `atlas-weight-peer` staging a rotation
/// set) — when set, arms rotation (eager decode) even for a single resident
/// slot, because an RDMA swap re-points that slot in place. Unset = disk path
/// only, byte-identical to today.
pub fn lora_peer_env() -> Option<String> {
    std::env::var("ATLAS_LORA_PEER")
        .ok()
        .filter(|s| !s.is_empty())
}

pub fn full_attention_layers(cfg: &ModelConfig) -> Vec<usize> {
    (0..cfg.num_hidden_layers)
        .filter(|&i| cfg.layer_type(i) == LayerType::FullAttention)
        .collect()
}

/// Adapter-config gates that need build-time context (`--max-lora-rank`).
/// Parse-time gates (peft_type/DoRA/bias/regex target_modules/…) already
/// ran in `atlas_core::config::parse_peft_adapter_config`.
pub fn validate_peft_config(peft: &PeftAdapterConfig, max_lora_rank: usize) -> Result<()> {
    if peft.r > max_lora_rank {
        bail!(
            "REJECT[rank-exceeds-pool]: r={} > --max-lora-rank={}",
            peft.r,
            max_lora_rank
        );
    }
    for t in &peft.target_modules {
        let last = t.rsplit('.').next().unwrap_or(t);
        if last == "q_proj" {
            bail!(
                "REJECT[gated-q-proj]: target_modules includes q_proj — \
                 excluded in v0 (attn_output_gate interleaved [Q|gate])"
            );
        }
        if !LoraModule::ALL.iter().any(|m| m.peft_name() == last) {
            bail!(
                "REJECT[unsupported-target]: target_modules entry '{t}' \
                 (allowed: k_proj v_proj o_proj gate_proj up_proj down_proj)"
            );
        }
    }
    Ok(())
}

/// Padded per-slot bytes: Σ over (full-attn layers × 6 modules) of
/// (max_rank·in + out·max_rank)·2. Holo @ max_rank=64: ≈ 2.44 MiB/layer
/// × 6 = ~14.6 MiB/slot; × max_loras=8 ≈ 117 MiB total.
pub(crate) fn pool_slot_bytes(cfg: &ModelConfig, max_rank: usize) -> usize {
    full_attention_layers(cfg)
        .iter()
        .map(|_| {
            LoraModule::ALL
                .iter()
                .map(|m| {
                    let (out, inp) = m.dims(cfg);
                    (max_rank * inp + out * max_rank) * BF16_BYTES
                })
                .sum::<usize>()
        })
        .sum()
}

/// Byte offset of slot `k`'s base within the pool. Slots are equal fixed size,
/// so slot `k` starts at `k * pool_slot_bytes`. Slot 0 → 0 (byte-identical to
/// the single-adapter path).
pub(crate) fn slot_base_offset(slot: usize, cfg: &ModelConfig, max_rank: usize) -> usize {
    slot * pool_slot_bytes(cfg, max_rank)
}

/// The (a_off, b_off) of a given (layer, module) WITHIN a slot — the exact
/// running offsets the pack loop computes (layer asc × [`LoraModule::ALL`] ×
/// A-then-B). `None` if `target_layer` is not a full-attention layer. Used by
/// the pack loop, the RDMA landing path, and the offset unit tests so all three
/// agree on the one frozen layout.
pub(crate) fn module_slot_offsets(
    cfg: &ModelConfig,
    max_rank: usize,
    target_layer: usize,
    target_module: LoraModule,
) -> Option<(usize, usize)> {
    let mut off = 0usize;
    for layer_idx in full_attention_layers(cfg) {
        for module in LoraModule::ALL {
            let (out_dim, in_dim) = module.dims(cfg);
            let a_off = off;
            let b_off = off + max_rank * in_dim * BF16_BYTES;
            off = b_off + out_dim * max_rank * BF16_BYTES;
            if layer_idx == target_layer && module == target_module {
                return Some((a_off, b_off));
            }
        }
    }
    None
}

/// The v0 family allow-list, checked once per load. v0 is validated on the
/// Qwen3.5-family attention trunk — qwen3_5 DENSE (holo-3.1-0.8b) and
/// holo3_1_moe (holo-3.1-35b-a3b, MoE). Both route to `Qwen35WeightLoader`, so
/// their full-attention layers are `Qwen3AttentionLayer` — what the install
/// walk downcasts to. Other families stay rejected (no validated mapping).
fn check_family(cfg: &ModelConfig) -> Result<()> {
    if !(cfg.is_qwen35_dense() || cfg.model_type == "holo3_1_moe") {
        bail!(
            "REJECT[unvalidated-family]: LoRA v0 is validated on qwen3_5 dense \
             (holo-3.1-0.8b) and holo3_1_moe (holo-3.1-35b-a3b) only; \
             model_type='{}', num_experts={}",
            cfg.model_type,
            cfg.num_experts
        );
    }
    Ok(())
}

/// Classify + audit one adapter's tensors (unconsumed key = fatal; pair
/// completeness; A=[r,in]/B=[out,r] shapes; every `target_modules` entry
/// matched). Returns the (layer, module) → [a_key, b_key] map used to pack.
fn audit_adapter(
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    max_lora_rank: usize,
) -> Result<BTreeMap<(usize, LoraModule), [Option<String>; 2]>> {
    validate_peft_config(peft, max_lora_rank)?;

    // 1) classify EVERY adapter tensor — any unclassifiable/unsupported key
    //    is a hard error, which IS the "unconsumed adapter tensors fatal"
    //    audit direction.
    let mut found: BTreeMap<(usize, LoraModule), [Option<String>; 2]> = BTreeMap::new();
    for name in adapter_store.names() {
        let (layer, module, ab) = classify_key(name, cfg)?;
        let entry = found.entry((layer, module)).or_default();
        let slot = &mut entry[ab as usize];
        if slot.is_some() {
            bail!(
                "REJECT[duplicate-tensor]: two tensors map to layer {layer} \
                 {module:?} lora_{ab:?}"
            );
        }
        *slot = Some(name.to_string());
    }
    if found.is_empty() {
        bail!("REJECT[empty-adapter]: no lora_A/lora_B tensors in adapter");
    }

    // 2) pair completeness + shape audit. PEFT shapes: A=[r, in], B=[out, r].
    for ((layer, module), pair) in &found {
        let [Some(a_key), Some(b_key)] = pair else {
            bail!(
                "REJECT[unpaired-tensor]: layer {layer} {module:?} has only \
                 one of lora_A/lora_B"
            );
        };
        let (out_dim, in_dim) = module.dims(cfg);
        let a = adapter_store.get(a_key)?; // hard-fail get
        let b = adapter_store.get(b_key)?;
        if a.shape != vec![peft.r, in_dim] {
            bail!(
                "REJECT[shape-mismatch]: '{a_key}' is {:?}, expected [{}, {}] \
                 (r, in_dim)",
                a.shape,
                peft.r,
                in_dim
            );
        }
        if b.shape != vec![out_dim, peft.r] {
            bail!(
                "REJECT[shape-mismatch]: '{b_key}' is {:?}, expected [{}, {}] \
                 (out_dim, r)",
                b.shape,
                out_dim,
                peft.r
            );
        }
    }

    // 3) other audit direction: every target_modules entry matched ≥1 pair.
    for t in &peft.target_modules {
        let last = t.rsplit('.').next().unwrap_or(t);
        if !found.keys().any(|(_, m)| m.peft_name() == last) {
            bail!(
                "REJECT[unmatched-target]: target_modules entry '{t}' matched \
                 no adapter tensor on any full-attention layer"
            );
        }
    }
    Ok(found)
}

/// Pack one already-audited adapter into pool `slot` (byte sub-region at base
/// `slot * pool_slot_bytes`). The intra-slot walk (layer asc ×
/// [`LoraModule::ALL`] × A-then-B, A contiguous, B row-repacked stride r →
/// max_rank) is IDENTICAL for every slot — slot 0 is byte-for-byte the
/// pre-multi-adapter path. Returns this slot's GLOBAL-layer-indexed pairs and,
/// per (layer, module), the packed (a_ptr, b_ptr) as raw u64 ((0,0) where the
/// adapter omits the module) for the post-pass pointer-table build.
#[allow(clippy::type_complexity)]
fn pack_slot(
    slot: usize,
    name: &str,
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    found: &BTreeMap<(usize, LoraModule), [Option<String>; 2]>,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    pool: DevicePtr,
    max_lora_rank: usize,
) -> Result<(
    Vec<Option<LoraLayerWeights>>,
    BTreeMap<(usize, LoraModule), (u64, u64)>,
)> {
    let scale = peft.scaling();
    let slot_bytes = pool_slot_bytes(cfg, max_lora_rank);
    let mut layers: Vec<Option<LoraLayerWeights>> =
        (0..cfg.num_hidden_layers).map(|_| None).collect();
    let mut slot_ptrs: BTreeMap<(usize, LoraModule), (u64, u64)> = BTreeMap::new();
    let mut off = slot * slot_bytes; // slot base offset
    for layer_idx in full_attention_layers(cfg) {
        let mut lw = LoraLayerWeights {
            layer_idx,
            k_proj: None,
            v_proj: None,
            o_proj: None,
            gate_proj: None,
            up_proj: None,
            down_proj: None,
        };
        let mut any = false;
        for module in LoraModule::ALL {
            let (out_dim, in_dim) = module.dims(cfg);
            let a_off = off;
            let b_off = off + max_lora_rank * in_dim * BF16_BYTES;
            off = b_off + out_dim * max_lora_rank * BF16_BYTES;
            let a_ptr = DevicePtr(pool.0 + a_off as u64);
            let b_ptr = DevicePtr(pool.0 + b_off as u64);

            let mut this = (0u64, 0u64); // NULL = base-only
            if let Some([Some(a_key), Some(b_key)]) = found.get(&(layer_idx, module)) {
                // A: contiguous [r, in] → head of the padded [max_rank, in] region.
                let a_t = adapter_store.get(a_key)?;
                let mut a_host = vec![0u8; peft.r * in_dim * BF16_BYTES];
                gpu.copy_d2h(a_t.ptr, &mut a_host)?;
                gpu.copy_h2d(&a_host, a_ptr)?;
                // B: [out, r] → row-stride pad to [out, max_rank].
                let b_t = adapter_store.get(b_key)?;
                let mut b_src = vec![0u8; out_dim * peft.r * BF16_BYTES];
                gpu.copy_d2h(b_t.ptr, &mut b_src)?;
                let mut b_host = vec![0u8; out_dim * max_lora_rank * BF16_BYTES];
                for row in 0..out_dim {
                    let d = row * max_lora_rank * BF16_BYTES;
                    let s = row * peft.r * BF16_BYTES;
                    b_host[d..d + peft.r * BF16_BYTES]
                        .copy_from_slice(&b_src[s..s + peft.r * BF16_BYTES]);
                }
                gpu.copy_h2d(&b_host, b_ptr)?;

                let pair = LoraPair {
                    a: DenseWeight { weight: a_ptr },
                    b: DenseWeight { weight: b_ptr },
                    rank: peft.r as u32,
                    k_in: in_dim as u32,
                    n_out: out_dim as u32,
                    scale,
                    // Kernel contraction dim: B's packed row stride (and A's
                    // padded row count) — see LoraPair docs in lora_delta.rs.
                    max_rank: max_lora_rank as u32,
                };
                tracing::info!(
                    "LoRA: slot {slot} '{name}' layer {layer_idx} {module:?} r={} \
                     scale={:.6} A=[{},{}] B=[{},{}] (padded to max_rank={})",
                    peft.r,
                    scale,
                    peft.r,
                    in_dim,
                    out_dim,
                    peft.r,
                    max_lora_rank
                );
                match module {
                    LoraModule::KProj => lw.k_proj = Some(pair),
                    LoraModule::VProj => lw.v_proj = Some(pair),
                    LoraModule::OProj => lw.o_proj = Some(pair),
                    LoraModule::GateProj => lw.gate_proj = Some(pair),
                    LoraModule::UpProj => lw.up_proj = Some(pair),
                    LoraModule::DownProj => lw.down_proj = Some(pair),
                }
                this = (a_ptr.0, b_ptr.0);
                any = true;
            }
            slot_ptrs.insert((layer_idx, module), this);
        }
        if any {
            layers[layer_idx] = Some(lw);
        }
    }
    debug_assert_eq!(off, (slot + 1) * slot_bytes); // one slot filled exactly
    Ok((layers, slot_ptrs))
}

/// Model-agnostic MULTI-adapter PEFT load: audit every adapter, VRAM-preflight
/// the N-slot pool, pack each adapter into its slot (0..N-1), and build the
/// per-module `[max_loras]` pointer tables (index k filled per packed slot,
/// rest NULL). One resident adapter is byte-identical to the single-adapter
/// path (slot 0, `off` starts at 0).
///
/// Called (via the `ModelWeightLoader::load_lora_adapters` hook) from
/// `build_model` BEFORE `BufferArena::new` and the free-memory snapshot, so
/// the pool bytes land in `used_so_far` and the KV budget shrinks
/// automatically. Do NOT move the call later.
pub fn load_lora_adapters_multi(
    adapters: &[LoraAdapterInput<'_>],
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_loras: usize,
    max_lora_rank: usize,
) -> Result<LoraWeights> {
    check_family(cfg)?;
    if adapters.is_empty() {
        bail!("REJECT[no-adapters]: load_lora_adapters_multi called with an empty set");
    }
    if adapters.len() > max_loras {
        bail!(
            "REJECT[too-many-adapters]: {} --lora-adapter given but --max-loras={} \
             (pool has {} slots); raise --max-loras or stage the extras on an \
             $ATLAS_LORA_PEER for on-demand RDMA swap",
            adapters.len(),
            max_loras,
            max_loras
        );
    }

    // Audit every adapter up front (each gets its own classify/shape/target
    // audit + rank<=max_lora_rank check) before touching VRAM.
    let mut audited: Vec<BTreeMap<(usize, LoraModule), [Option<String>; 2]>> =
        Vec::with_capacity(adapters.len());
    for a in adapters {
        audited.push(audit_adapter(a.store, &a.peft, cfg, max_lora_rank)?);
    }

    // VRAM preflight, then one fixed-address pool alloc for ALL slots, zeroed
    // once (pad rows/cols and unpacked slots stay 0 = padded-K correctness).
    let pool_bytes = pool_slot_bytes(cfg, max_lora_rank) * max_loras;
    let free = gpu.free_memory()?;
    if pool_bytes * 2 > free {
        bail!(
            "OOM pre-flight (LoRA pool): {:.1} MiB pool ({} slots × padded A/B) \
             would leave < 1× headroom of {:.1} MiB free; every pool byte comes \
             directly out of the KV-cache budget on GB10 unified memory",
            pool_bytes as f64 / (1024.0 * 1024.0),
            max_loras,
            free as f64 / (1024.0 * 1024.0),
        );
    }
    let pool = gpu.alloc(pool_bytes)?;
    gpu.memset(pool, 0, pool_bytes)?;

    // Pack each adapter into its slot; accumulate per-(layer,module) [max_loras]
    // pointer arrays for the post-pass table build.
    let mut slots: Vec<AdapterSlot> = Vec::with_capacity(adapters.len());
    let mut a_tabs: BTreeMap<(usize, LoraModule), Vec<u64>> = BTreeMap::new();
    let mut b_tabs: BTreeMap<(usize, LoraModule), Vec<u64>> = BTreeMap::new();
    for (k, a) in adapters.iter().enumerate() {
        let (layers, slot_ptrs) = pack_slot(
            k,
            &a.name,
            a.store,
            &a.peft,
            &audited[k],
            cfg,
            gpu,
            pool,
            max_lora_rank,
        )?;
        for ((layer, module), (a_ptr, b_ptr)) in slot_ptrs {
            a_tabs
                .entry((layer, module))
                .or_insert_with(|| vec![0u64; max_loras])[k] = a_ptr;
            b_tabs
                .entry((layer, module))
                .or_insert_with(|| vec![0u64; max_loras])[k] = b_ptr;
        }
        slots.push(AdapterSlot {
            name: a.name.clone(),
            adapter_config: a.peft.clone(),
            layers,
        });
    }

    // Post-pass: materialize the per-module [max_loras] u64 pointer tables (the
    // frozen M2 BGMV contract; currently dormant — no compute site reads them).
    // build_ptr_table pattern (nemotron_moe.rs:414): pack le bytes → alloc → h2d.
    let mk = |tab: &[u64]| -> Result<DevicePtr> {
        let bytes: Vec<u8> = tab.iter().flat_map(|p| p.to_le_bytes()).collect();
        let d = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, d)?;
        Ok(d)
    };
    let mut tables = BTreeMap::new();
    for (key, a_tab) in &a_tabs {
        let b_tab = &b_tabs[key];
        tables.insert(*key, (mk(a_tab)?, mk(b_tab)?));
    }

    // Parallel [max_loras] f32 scale table (per-slot scale, 0.0 for unpacked
    // slots) — the bgmv fold reads scale_table[seq_slot] in fp32. Same
    // load-time-fixed pattern as the a/b tables.
    let scale_vals = scale_table_values(adapters, max_loras);
    let scale_bytes: Vec<u8> = scale_vals.iter().flat_map(|s| s.to_le_bytes()).collect();
    let scale_table = gpu.alloc(scale_bytes.len())?;
    gpu.copy_h2d(&scale_bytes, scale_table)?;

    Ok(LoraWeights {
        name: slots[0].name.clone(),
        adapter_config: slots[0].adapter_config.clone(),
        max_rank: max_lora_rank,
        max_loras,
        pool,
        pool_bytes,
        slots,
        active: 0,
        tables,
        scale_table,
    })
}

/// Runtime disk swap: audit + pack an already-loaded adapter `store` into an
/// EXISTING pool `slot` of `lw`, in place, and stamp that slot's
/// name/config/layers. Byte-identical to a startup pack of the same adapter into
/// that slot — same audit, A-contiguous copy, and B row-repack via [`pack_slot`].
/// The slot sub-region is re-zeroed first (a reused slot still holds the prior
/// adapter's bytes, and pad rows/cols must stay 0 for padded-K correctness).
/// Returns the rebuilt per-layer pairs so the caller can re-install them if the
/// slot is currently active. Like the startup pack, the intermediate `store`'s
/// device copies leak (small, one-off per swap). Used for the pool-size-1
/// dynamic-load demo (load a different adapter into the single slot at runtime).
pub fn pack_store_into_slot(
    lw: &mut LoraWeights,
    slot: usize,
    name: &str,
    store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Vec<Option<LoraLayerWeights>>> {
    if slot >= lw.max_loras {
        bail!(
            "LoRA disk swap: slot {slot} >= max_loras {} (pool has {} slots)",
            lw.max_loras,
            lw.max_loras
        );
    }
    validate_peft_config(peft, lw.max_rank)?;
    let found = audit_adapter(store, peft, cfg, lw.max_rank)?;
    let slot_bytes = pool_slot_bytes(cfg, lw.max_rank);
    gpu.memset(
        DevicePtr(lw.pool.0 + (slot * slot_bytes) as u64),
        0,
        slot_bytes,
    )?;
    let (layers, _slot_ptrs) = pack_slot(
        slot,
        name,
        store,
        peft,
        &found,
        cfg,
        gpu,
        lw.pool,
        lw.max_rank,
    )?;
    lw.slots[slot].name = name.to_string();
    lw.slots[slot].adapter_config = peft.clone();
    lw.slots[slot].layers = layers.clone();
    Ok(layers)
}

/// Single-adapter convenience wrapper (packs slot 0 only) — byte-identical to
/// the pre-multi-adapter path. Kept for the unit tests and any single-adapter
/// caller. The `name` is stamped onto the sole slot.
pub fn load_lora_adapters_generic(
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_loras: usize,
    max_lora_rank: usize,
) -> Result<LoraWeights> {
    let inputs = [LoraAdapterInput {
        name: String::new(),
        store: adapter_store,
        peft: peft.clone(),
    }];
    load_lora_adapters_multi(&inputs, cfg, gpu, max_loras, max_lora_rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::config::{LayerType, ModelConfig};

    // Real factory config: layers 3,7,…,47 are FullAttention. The pack offset
    // math depends only on layer_type + projection dims.
    fn cfg() -> ModelConfig {
        ModelConfig::qwen3_next_80b_nvfp4()
    }

    #[test]
    fn slot_base_is_k_times_slot_bytes() {
        let cfg = cfg();
        let mr = 16;
        let sb = pool_slot_bytes(&cfg, mr);
        for k in 0..8 {
            assert_eq!(slot_base_offset(k, &cfg, mr), k * sb);
        }
    }

    #[test]
    fn module_offsets_walk_matches_pack_loop_and_fill_exactly_one_slot() {
        // Reproduce the pack loop's cumulative A-then-B walk (the frozen
        // layout) and assert module_slot_offsets agrees at every step, and the
        // running end lands exactly on pool_slot_bytes (one full slot).
        let cfg = cfg();
        let mr = 16;
        let mut off = 0usize;
        for layer in full_attention_layers(&cfg) {
            for module in LoraModule::ALL {
                let (out, inp) = module.dims(&cfg);
                let a_off = off;
                let b_off = off + mr * inp * BF16_BYTES;
                off = b_off + out * mr * BF16_BYTES;
                assert_eq!(
                    module_slot_offsets(&cfg, mr, layer, module),
                    Some((a_off, b_off)),
                    "layer {layer} {module:?}"
                );
                assert!(a_off < b_off, "A precedes B within a module region");
            }
        }
        assert_eq!(
            off,
            pool_slot_bytes(&cfg, mr),
            "one pass fills exactly one slot"
        );
    }

    #[test]
    fn module_offsets_none_for_non_full_attention_layer() {
        let cfg = cfg();
        assert_eq!(cfg.layer_type(0), LayerType::LinearAttention);
        assert_eq!(module_slot_offsets(&cfg, 16, 0, LoraModule::KProj), None);
    }

    #[test]
    fn slot_boundaries_do_not_overlap() {
        let cfg = cfg();
        let mr = 16;
        let sb = pool_slot_bytes(&cfg, mr);
        // Last module (down_proj on the last full-attn layer) ends exactly at
        // slot_bytes, i.e. flush against slot 1's base.
        let last = *full_attention_layers(&cfg).last().unwrap();
        let (_, b_off) = module_slot_offsets(&cfg, mr, last, LoraModule::DownProj).unwrap();
        let (out, _) = LoraModule::DownProj.dims(&cfg);
        assert_eq!(b_off + out * mr * BF16_BYTES, sb);
        assert_eq!(slot_base_offset(1, &cfg, mr), sb);
    }

    #[test]
    fn adapter_names_and_slot_resolve() {
        // A hand-built LoraWeights (no GPU) exercising the name→slot resolver
        // and the active-slot mirror the rotation control path relies on.
        let peft = PeftAdapterConfig {
            r: 4,
            lora_alpha: 8.0,
            target_modules: vec!["k_proj".into()],
            use_rslora: false,
            layers_to_transform: None,
        };
        let mk_slot = |name: &str| AdapterSlot {
            name: name.to_string(),
            adapter_config: peft.clone(),
            layers: Vec::new(),
        };
        let lw = LoraWeights {
            name: "alpha".into(),
            adapter_config: peft.clone(),
            max_rank: 4,
            max_loras: 8,
            pool: DevicePtr(0),
            pool_bytes: 0,
            slots: vec![mk_slot("alpha"), mk_slot("beta")],
            active: 0,
            tables: BTreeMap::new(),
            scale_table: DevicePtr(0),
        };
        assert_eq!(lw.adapter_names(), vec!["alpha", "beta"]);
        assert_eq!(lw.slot_of("beta"), Some(1));
        assert_eq!(lw.slot_of("missing"), None);
    }

    #[test]
    fn scale_table_values_per_slot_and_padded() {
        // scaling() = alpha/r (no rslora); alpha/sqrt(r) under rslora. The
        // scale table carries one f32 per slot, 0.0 for unpacked slots, in
        // slot order — exactly what bgmv indexes by seq_slot.
        let store = WeightStore::empty();
        let mk = |alpha: f64, r: usize, rslora: bool| LoraAdapterInput {
            name: String::new(),
            store: &store,
            peft: PeftAdapterConfig {
                r,
                lora_alpha: alpha,
                target_modules: vec!["k_proj".into()],
                use_rslora: rslora,
                layers_to_transform: None,
            },
        };
        let adapters = [mk(16.0, 8, false), mk(16.0, 4, true)];
        let v = scale_table_values(&adapters, 8);
        assert_eq!(v.len(), 8);
        assert_eq!(v[0], (16.0_f64 / 8.0) as f32); // alpha/r
        assert_eq!(v[1], (16.0_f64 / (4.0_f64).sqrt()) as f32); // rslora: alpha/sqrt(r)
        assert!(v[2..].iter().all(|&s| s == 0.0)); // unpacked slots
        // Table order matches the a/b table slot order (slot k = adapters[k]).
        for (k, a) in adapters.iter().enumerate() {
            assert_eq!(v[k], a.peft.scaling());
        }
    }

    #[test]
    fn seq_slot_host_defers_negatives_and_pads() {
        // Two real seqs on explicit slots 1 and 0, one defaulting (-1 -> active=2),
        // padded to 4 (pad rows -1 = base/no delta).
        let slots = [1i32, -1, 0];
        let v = build_seq_slot_host(&slots, 4, 2);
        assert_eq!(v, vec![1, 2, 0, -1]);
    }

    #[test]
    fn seq_slot_host_single_global_adapter_all_active() {
        // All requests default (-1) → all real rows resolve to the active slot,
        // so a single global adapter applies to every row (matches n==1).
        let slots = [-1i32, -1, -1, -1];
        let v = build_seq_slot_host(&slots, 4, 0);
        assert_eq!(v, vec![0, 0, 0, 0]);
    }

    #[test]
    fn seq_slot_host_no_pad_when_full() {
        let slots = [3i32, 1];
        assert_eq!(build_seq_slot_host(&slots, 2, 0), vec![3, 1]);
    }
}
