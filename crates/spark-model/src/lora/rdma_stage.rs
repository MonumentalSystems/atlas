// SPDX-License-Identifier: AGPL-3.0-only

//! RDMA LoRA staging (spark-model half): turn a peer-staged adapter's manifest
//! into a set of pool-slot LANDING TARGETS (the only place `classify_key` +
//! the per-slot offset math live), then drive `spark_storage::RdmaLoraLoader`
//! to RDMA-load the adapter's A/B straight into a resident slot for fast
//! rotation. Landing is byte-identical to the disk pack (the loader does the
//! same F16/F32→BF16 convert + B row-repack).
//!
//! Gated behind `$ATLAS_LORA_PEER` at the call site; when unset the disk
//! rotation path is unchanged.

use anyhow::{Result, anyhow, bail};
use atlas_core::config::{ModelConfig, PeftAdapterConfig};
use spark_runtime::gpu::DevicePtr;
use spark_storage::weight_peer::WeightManifest;
use spark_storage::{LoraAbKind, LoraLandTarget};

use super::{
    AdapterAb, LoraLayerWeights, LoraModule, classify_key, module_slot_offsets, pool_slot_bytes,
    slot_base_offset,
};
use crate::layers::ops::lora_delta::LoraPair;
use crate::weight_map::DenseWeight;

/// Build the landing targets for one adapter's manifest into pool `slot`. Each
/// `lora_A/lora_B` tensor is classified to (layer, module, A|B) and mapped to
/// its byte sub-region `pool + slot*slot_bytes + a_off|b_off`. The adapter's
/// real rank r is read from the tensor shape (A=[r,in], B=[out,r]). Rejections
/// from `classify_key` (q_proj / GDN / wrong-layer / non-PEFT key) fire here
/// too — never a silent skip.
pub fn build_land_targets(
    manifest: &WeightManifest,
    cfg: &ModelConfig,
    pool: DevicePtr,
    slot: usize,
    max_rank: usize,
) -> Result<Vec<LoraLandTarget>> {
    let base = pool.0 + slot_base_offset(slot, cfg, max_rank) as u64;
    let mut targets = Vec::with_capacity(manifest.tensors.len());
    for rec in &manifest.tensors {
        let (layer, module, ab) = classify_key(&rec.name, cfg)?;
        let (a_off, b_off) = module_slot_offsets(cfg, max_rank, layer, module)
            .ok_or_else(|| anyhow!("lora rdma: layer {layer} not a full-attention slot layer"))?;
        let (out_dim, in_dim) = module.dims(cfg);
        // r from the on-wire shape: A=[r,in] → shape[0]; B=[out,r] → shape[1].
        let rank = match ab {
            AdapterAb::A => *rec
                .shape
                .first()
                .ok_or_else(|| anyhow!("A tensor {} has no shape", rec.name))?
                as usize,
            AdapterAb::B => *rec
                .shape
                .get(1)
                .ok_or_else(|| anyhow!("B tensor {} shape < 2", rec.name))?
                as usize,
        };
        if rank > max_rank {
            bail!(
                "lora rdma: adapter rank {rank} for {} exceeds pool max_rank {max_rank}",
                rec.name
            );
        }
        let (kind, off) = match ab {
            AdapterAb::A => (LoraAbKind::A, a_off),
            AdapterAb::B => (LoraAbKind::B, b_off),
        };
        targets.push(LoraLandTarget {
            tensor_name: rec.name.clone(),
            kind,
            dst: base + off as u64,
            out_dim,
            in_dim,
            rank,
            max_rank,
        });
    }
    if targets.is_empty() {
        bail!("lora rdma: adapter manifest has no lora_A/lora_B tensors");
    }
    Ok(targets)
}

/// Rebuild a slot's per-layer [`LoraLayerWeights`] after an in-place RDMA
/// reload — the A/B bytes changed AND the adapter's r/scale may differ, so the
/// `LoraPair`s (which bake rank + scale) must be rebuilt, not just re-pointed.
/// Pointers are deterministic (`pool + slot*slot_bytes + off`); this does NOT
/// touch the GPU. Modules present are those with a target of the matching kind.
pub fn rebuild_slot_layers(
    targets: &[LoraLandTarget],
    cfg: &ModelConfig,
    peft: &PeftAdapterConfig,
    pool: DevicePtr,
    slot: usize,
    max_rank: usize,
) -> Result<Vec<Option<LoraLayerWeights>>> {
    let scale = peft.scaling();
    let base = pool.0 + slot_base_offset(slot, cfg, max_rank) as u64;
    let mut layers: Vec<Option<LoraLayerWeights>> =
        (0..cfg.num_hidden_layers).map(|_| None).collect();
    // Group targets by (layer, module): we need both A and B present to build a
    // pair. Re-derive from classify (targets carry only geometry, not keys' layer).
    // Simpler: walk the pool layout and, for each (layer, module), find whether a
    // target lands there (by matching dst).
    for rec_layer in super::full_attention_layers(cfg) {
        let mut lw = LoraLayerWeights {
            layer_idx: rec_layer,
            k_proj: None,
            v_proj: None,
            o_proj: None,
            gate_proj: None,
            up_proj: None,
            down_proj: None,
        };
        let mut any = false;
        for module in LoraModule::ALL {
            let (a_off, b_off) =
                module_slot_offsets(cfg, max_rank, rec_layer, module).expect("full-attn layer");
            let a_dst = base + a_off as u64;
            let b_dst = base + b_off as u64;
            let a_t = targets
                .iter()
                .find(|t| t.kind == LoraAbKind::A && t.dst == a_dst);
            let b_t = targets
                .iter()
                .find(|t| t.kind == LoraAbKind::B && t.dst == b_dst);
            if let (Some(a), Some(b)) = (a_t, b_t) {
                let (out_dim, in_dim) = module.dims(cfg);
                let pair = LoraPair {
                    a: DenseWeight {
                        weight: DevicePtr(a_dst),
                    },
                    b: DenseWeight {
                        weight: DevicePtr(b_dst),
                    },
                    rank: a.rank as u32,
                    k_in: in_dim as u32,
                    n_out: out_dim as u32,
                    scale,
                    max_rank: max_rank as u32,
                };
                let _ = b; // b geometry equals a's rank; both audited upstream
                match module {
                    LoraModule::KProj => lw.k_proj = Some(pair),
                    LoraModule::VProj => lw.v_proj = Some(pair),
                    LoraModule::OProj => lw.o_proj = Some(pair),
                    LoraModule::GateProj => lw.gate_proj = Some(pair),
                    LoraModule::UpProj => lw.up_proj = Some(pair),
                    LoraModule::DownProj => lw.down_proj = Some(pair),
                }
                any = true;
            }
        }
        if any {
            layers[rec_layer] = Some(lw);
        }
    }
    Ok(layers)
}

/// The per-slot byte length (re-exported so the swap path can re-zero exactly
/// one slot's sub-region before an in-place reload).
pub fn slot_bytes(cfg: &ModelConfig, max_rank: usize) -> usize {
    pool_slot_bytes(cfg, max_rank)
}

/// Fetch a peer-staged adapter's manifest over the `weight_peer` control
/// channel (connect → request → read manifest, then drop the connection).
/// Needed to build landing targets before the loader's own verbs handshake.
#[cfg(feature = "cuda")]
pub fn fetch_adapter_manifest(peer_addr: &str, adapter_id: &str) -> Result<WeightManifest> {
    use std::net::TcpStream;

    use anyhow::Context;
    use spark_storage::weight_peer::{read_weight_manifest, write_model_request};

    let mut stream =
        TcpStream::connect(peer_addr).with_context(|| format!("connect lora peer {peer_addr}"))?;
    stream.set_nodelay(true).ok();
    write_model_request(&mut stream, adapter_id).context("send adapter request")?;
    let manifest = read_weight_manifest(&mut stream).context("read adapter manifest")?;
    // Drop the connection without a transport handshake; the loader reconnects
    // for the actual one-sided read.
    let _ = std::io::Write::write_all(&mut stream, &[]);
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_storage::weight_peer::{WeightManifest, WeightTensorRecord};

    // Real factory config (layer 3,7,… are FullAttention). Offset math only
    // needs layer_type + projection dims, so the family (MoE here) is irrelevant.
    fn cfg() -> ModelConfig {
        ModelConfig::qwen3_next_80b_nvfp4()
    }

    fn rec(name: &str, shape: Vec<u64>) -> WeightTensorRecord {
        WeightTensorRecord {
            name: name.into(),
            dtype: "F32".into(),
            shape,
            offset_in_shard: 0,
            len: 0,
            shard_index: 0,
            extra: false,
        }
    }

    #[test]
    fn land_targets_map_to_slot_subregions() {
        let cfg = cfg();
        // Layer 3 is FullAttention in the factory config.
        let layer = 3usize;
        assert_eq!(
            cfg.layer_type(layer),
            atlas_core::config::LayerType::FullAttention
        );
        let (out_dim, in_dim) = LoraModule::KProj.dims(&cfg);
        let max_rank = 8;
        let r = 4u64;
        let pool = DevicePtr(0x1_0000);
        let manifest = WeightManifest {
            version: WeightManifest::VERSION,
            model_id: "adp".into(),
            shard_files: vec!["adapter_model.safetensors".into()],
            shard_lens: vec![0],
            tensors: vec![
                rec(
                    &format!(
                        "base_model.model.model.layers.{layer}.self_attn.k_proj.lora_A.weight"
                    ),
                    vec![r, in_dim as u64],
                ),
                rec(
                    &format!(
                        "base_model.model.model.layers.{layer}.self_attn.k_proj.lora_B.weight"
                    ),
                    vec![out_dim as u64, r],
                ),
            ],
        };
        let targets = build_land_targets(&manifest, &cfg, pool, 1, max_rank).unwrap();
        assert_eq!(targets.len(), 2);
        // Slot 1 base = pool + 1*slot_bytes.
        let base = pool.0 + slot_base_offset(1, &cfg, max_rank) as u64;
        let (a_off, b_off) = module_slot_offsets(&cfg, max_rank, layer, LoraModule::KProj).unwrap();
        let a = targets.iter().find(|t| t.kind == LoraAbKind::A).unwrap();
        let b = targets.iter().find(|t| t.kind == LoraAbKind::B).unwrap();
        assert_eq!(a.dst, base + a_off as u64);
        assert_eq!(b.dst, base + b_off as u64);
        assert_eq!(a.rank, r as usize);
        assert_eq!(b.rank, r as usize);
        assert_eq!(a.max_rank, max_rank);
    }

    #[test]
    fn rebuild_slot_layers_sets_rank_and_pointers() {
        let cfg = cfg();
        let layer = 3usize;
        let (out_dim, in_dim) = LoraModule::KProj.dims(&cfg);
        let max_rank = 8;
        let pool = DevicePtr(0x2_0000);
        let base = pool.0 + slot_base_offset(2, &cfg, max_rank) as u64;
        let (a_off, b_off) = module_slot_offsets(&cfg, max_rank, layer, LoraModule::KProj).unwrap();
        let targets = vec![
            LoraLandTarget {
                tensor_name: "a".into(),
                kind: LoraAbKind::A,
                dst: base + a_off as u64,
                out_dim,
                in_dim,
                rank: 4,
                max_rank,
            },
            LoraLandTarget {
                tensor_name: "b".into(),
                kind: LoraAbKind::B,
                dst: base + b_off as u64,
                out_dim,
                in_dim,
                rank: 4,
                max_rank,
            },
        ];
        let peft = PeftAdapterConfig {
            r: 4,
            lora_alpha: 8.0,
            target_modules: vec!["k_proj".into()],
            use_rslora: false,
            layers_to_transform: None,
        };
        let layers = rebuild_slot_layers(&targets, &cfg, &peft, pool, 2, max_rank).unwrap();
        let lw = layers[layer].as_ref().expect("layer 3 rebuilt");
        let pair = lw.k_proj.expect("k_proj pair");
        assert_eq!(pair.rank, 4);
        assert_eq!(pair.max_rank, max_rank as u32);
        assert_eq!(pair.a.weight.0, base + a_off as u64);
        assert_eq!(pair.b.weight.0, base + b_off as u64);
    }
}
