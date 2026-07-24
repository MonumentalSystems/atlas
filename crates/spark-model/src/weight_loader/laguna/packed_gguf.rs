// SPDX-License-Identifier: AGPL-3.0-only

//! Laguna's keep-packed GGUF weight assembly.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::qwen3_attention::HeadGateActivation;
use crate::layers::{DenseFfnLayer, FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::weight_map::{
    DenseWeight, PackedExpertWeights, PackedQ4Weight, PackedQ6Weight, PackedQ8Weight, QuantWeight,
    QuantizedWeight, dense_auto,
};

/// Load Laguna's dense-only layer from packed Q8_0 or BF16 matrices.
pub(super) fn load_dense_ffn(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    layer_prefix: &str,
) -> Result<FfnComponent> {
    let mut layer = DenseFfnLayer::new(null_dense_ffn_weights(), gpu)?;
    let gate_name = format!("{layer_prefix}.mlp.gate_proj.weight");
    if store.get(&gate_name)?.is_packed_q8_0() {
        layer.set_q8_weights(
            PackedQ8Weight::from_store(store, &gate_name)?,
            PackedQ8Weight::from_store(store, &format!("{layer_prefix}.mlp.up_proj.weight"))?,
            PackedQ8Weight::from_store(store, &format!("{layer_prefix}.mlp.down_proj.weight"))?,
        );
    } else {
        layer.set_bf16_weights(
            dense_auto(store, &gate_name, gpu)?,
            dense_auto(store, &format!("{layer_prefix}.mlp.up_proj.weight"), gpu)?,
            dense_auto(store, &format!("{layer_prefix}.mlp.down_proj.weight"), gpu)?,
        );
    }
    Ok(FfnComponent::Dense(layer))
}

/// Build per-expert views over the contiguous routed Q4_K/Q6_K GGUF stacks.
pub(super) fn load_routed_experts(
    store: &WeightStore,
    config: &ModelConfig,
    mlp: &str,
) -> Result<Vec<PackedExpertWeights>> {
    let mut packed = Vec::with_capacity(config.num_experts);
    for expert in 0..config.num_experts {
        if !config.is_local_expert(expert) {
            packed.push(PackedExpertWeights {
                gate: PackedQ4Weight::null_view(),
                up: PackedQ4Weight::null_view(),
                down: QuantWeight::PackedQ6(PackedQ6Weight::null_view()),
            });
            continue;
        }
        let expert_prefix = format!("{mlp}.experts.{expert}");
        let down_prefix = format!("{expert_prefix}.down_proj");
        let down = if store.get(&format!("{down_prefix}.weight"))?.is_packed_q4k() {
            QuantWeight::PackedQ4(packed_q4_from_store(store, &down_prefix)?)
        } else {
            QuantWeight::PackedQ6(packed_q6_from_store(store, &down_prefix)?)
        };
        packed.push(PackedExpertWeights {
            gate: packed_q4_from_store(store, &format!("{expert_prefix}.gate_proj"))?,
            up: packed_q4_from_store(store, &format!("{expert_prefix}.up_proj"))?,
            down,
        });
    }
    Ok(packed)
}

/// Materialize the GGUF correction bias as F32 for the sigmoid router kernel.
pub(super) fn load_correction_bias(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let tensor = store
        .get(name)
        .with_context(|| format!("keep-packed bias {name} missing"))?;
    let num_elements = tensor.num_elements();
    let source_bytes = tensor.byte_size();
    let mut host = vec![0u8; source_bytes];
    gpu.copy_d2h(tensor.ptr, &mut host)
        .with_context(|| format!("copy_d2h correction_bias {name} ({source_bytes}B)"))?;
    let f32_bytes: Vec<u8> = match tensor.dtype {
        WeightDtype::BF16 => host
            .chunks_exact(2)
            .flat_map(|chunk| {
                let value = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits((value as u32) << 16).to_le_bytes()
            })
            .collect(),
        WeightDtype::FP32 => host,
        dtype => anyhow::bail!("unexpected correction_bias dtype {dtype:?} for {name}"),
    };
    let ptr = gpu
        .alloc(num_elements * 4)
        .with_context(|| format!("allocate F32 device copy of {name}"))?;
    gpu.copy_h2d(&f32_bytes, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

pub(super) fn set_q8_shared_expert(
    layer: &mut MoeLayer,
    store: &WeightStore,
    shared: &str,
) -> Result<()> {
    layer.set_q8_shared_expert(
        PackedQ8Weight::from_store(store, &format!("{shared}.gate_proj.weight"))?,
        PackedQ8Weight::from_store(store, &format!("{shared}.up_proj.weight"))?,
        PackedQ8Weight::from_store(store, &format!("{shared}.down_proj.weight"))?,
    );
    Ok(())
}

pub(super) fn set_q8_attention(
    layer: &mut Qwen3AttentionLayer,
    store: &WeightStore,
    prefix: &str,
) -> Result<()> {
    layer.set_packed_q8_weights(
        PackedQ8Weight::from_store(store, &format!("{prefix}.q_proj.weight"))?,
        PackedQ8Weight::from_store(store, &format!("{prefix}.k_proj.weight"))?,
        PackedQ8Weight::from_store(store, &format!("{prefix}.v_proj.weight"))?,
        PackedQ8Weight::from_store(store, &format!("{prefix}.o_proj.weight"))?,
    );
    layer.set_packed_q8_head_gate_weight(
        PackedQ8Weight::from_store(store, &format!("{prefix}.g_proj.weight"))?,
        HeadGateActivation::Softplus,
    );
    Ok(())
}

fn null_dense_ffn_weights() -> DenseFfnWeights {
    DenseFfnWeights {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    }
}

fn packed_q4_from_store(store: &WeightStore, prefix: &str) -> Result<PackedQ4Weight> {
    let tensor = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        tensor.is_packed_q4k(),
        "{prefix}.weight is not keep-packed Q4_K"
    );
    ensure!(
        tensor.shape.len() == 2,
        "{prefix}.weight is not 2D ({:?})",
        tensor.shape
    );
    Ok(PackedQ4Weight {
        weight: tensor.ptr,
        n: tensor.shape[0] as u32,
        k: tensor.shape[1] as u32,
    })
}

fn packed_q6_from_store(store: &WeightStore, prefix: &str) -> Result<PackedQ6Weight> {
    let tensor = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        tensor.is_packed_q6k(),
        "{prefix}.weight is not keep-packed Q6_K"
    );
    ensure!(
        tensor.shape.len() == 2,
        "{prefix}.weight is not 2D ({:?})",
        tensor.shape
    );
    Ok(PackedQ6Weight {
        weight: tensor.ptr,
        n: tensor.shape[0] as u32,
        k: tensor.shape[1] as u32,
    })
}
