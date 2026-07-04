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
    /// against the checkpoint header): k/v [512,1024], o [1024,2048],
    /// gate/up [3584,1024], down [1024,3584].
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
pub struct LoraLayerWeights {
    pub layer_idx: usize,
    pub k_proj: Option<LoraPair>,
    pub v_proj: Option<LoraPair>,
    pub o_proj: Option<LoraPair>,
    pub gate_proj: Option<LoraPair>,
    pub up_proj: Option<LoraPair>,
    pub down_proj: Option<LoraPair>,
}

/// The loaded adapter: one fixed-address rank-padded pool, per-layer pairs,
/// and per-module `[max_loras]` device u64 pointer tables (the frozen M2
/// BGMV contract — v0 fills slot 0, slots 1.. are NULL = base-only).
pub struct LoraWeights {
    /// Adapter name from `--lora-adapter NAME=PATH` (stamped by the caller).
    pub name: String,
    pub adapter_config: PeftAdapterConfig,
    pub max_rank: usize,
    pub max_loras: usize,
    /// One fixed-address allocation holding every padded A/B for every slot.
    pub pool: DevicePtr,
    pub pool_bytes: usize,
    /// Indexed by GLOBAL layer index (len = num_hidden_layers). `None` on
    /// GDN layers and on full-attention layers with no adapted module. The
    /// install walk iterates model layers by this same global index.
    pub layers: Vec<Option<LoraLayerWeights>>,
    /// key = (global_layer_idx, module) → (a_table, b_table); each table is
    /// a device `[max_loras]` u64 array, NULL (0) = base-only slot.
    pub tables: BTreeMap<(usize, LoraModule), (DevicePtr, DevicePtr)>,
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
        std::env::var("ATLAS_LORA_EAGER")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
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
fn pool_slot_bytes(cfg: &ModelConfig, max_rank: usize) -> usize {
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

/// Model-agnostic PEFT adapter load: classify EVERY tensor (unconsumed =
/// fatal), audit pairs + shapes bidirectionally against `target_modules`,
/// VRAM-preflight, then pack slot 0 of the fixed-address rank-padded pool
/// and build the per-module `[max_loras]` pointer tables.
///
/// Called (via the `ModelWeightLoader::load_lora_adapters` hook) from
/// `build_model` BEFORE `BufferArena::new` and the free-memory snapshot, so
/// the pool bytes land in `used_so_far` and the KV budget shrinks
/// automatically. Do NOT move the call later.
pub fn load_lora_adapters_generic(
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_loras: usize,
    max_lora_rank: usize,
) -> Result<LoraWeights> {
    // 0) family allow-list: v0 validated on qwen3_5 dense (holo) only.
    //    `is_qwen35_dense()` is the same predicate factory.rs uses to route
    //    to Qwen35DenseWeightLoader (model_type == "qwen3_5" && num_experts == 0).
    if !cfg.is_qwen35_dense() {
        bail!(
            "REJECT[unvalidated-family]: LoRA v0 is validated on qwen3_5 dense \
             (holo-3.1-0.8b) only; model_type='{}', num_experts={}",
            cfg.model_type,
            cfg.num_experts
        );
    }
    validate_peft_config(peft, max_lora_rank)?;
    let scale = peft.scaling();

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

    // 4) VRAM preflight, then one fixed-address pool alloc, zeroed.
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
    gpu.memset(pool, 0, pool_bytes)?; // zero pad rows/cols = padded-K correctness

    // 5) pack slot 0. Fixed layout order: layer asc × LoraModule::ALL × (A,B).
    //    A copies contiguously (r·in real bytes; pad rows trail, already 0).
    //    B repacks row-by-row from stride r to stride max_rank
    //    (d2h → host repack → h2d).
    let slot_bytes = pool_slot_bytes(cfg, max_lora_rank);
    let mut layers: Vec<Option<LoraLayerWeights>> = (0..cfg.num_hidden_layers).map(|_| None).collect();
    let mut tables = BTreeMap::new();
    let mut off = 0usize; // slot 0 base offset
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

            let mut slot0 = (0u64, 0u64); // NULL = base-only
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
                    "LoRA: layer {layer_idx} {module:?} r={} scale={:.6} \
                     A=[{},{}] B=[{},{}] (padded to max_rank={})",
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
                slot0 = (a_ptr.0, b_ptr.0);
                any = true;
            }
            // Frozen v1 contract: per-module [max_loras] u64 tables, slot 0
            // filled (or NULL), slots 1.. NULL. build_ptr_table pattern
            // (nemotron_moe.rs:414): pack le bytes → alloc → copy_h2d.
            let mut a_tab = vec![0u64; max_loras];
            let mut b_tab = vec![0u64; max_loras];
            (a_tab[0], b_tab[0]) = slot0;
            let mk = |tab: &[u64]| -> Result<DevicePtr> {
                let bytes: Vec<u8> = tab.iter().flat_map(|p| p.to_le_bytes()).collect();
                let d = gpu.alloc(bytes.len())?;
                gpu.copy_h2d(&bytes, d)?;
                Ok(d)
            };
            tables.insert((layer_idx, module), (mk(&a_tab)?, mk(&b_tab)?));
        }
        if any {
            layers[layer_idx] = Some(lw);
        }
    }
    debug_assert_eq!(off, slot_bytes); // slot-0 layout fills exactly one slot

    Ok(LoraWeights {
        name: String::new(), // caller stamps the --lora-adapter NAME
        adapter_config: peft.clone(),
        max_rank: max_lora_rank,
        max_loras,
        pool,
        pool_bytes,
        layers,
        tables,
    })
}
