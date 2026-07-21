// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::qwen3_attention::HeadGateActivation;
use crate::layers::{DenseFfnLayer, FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense, dense_auto,
    quantize_to_nvfp4, quantized_v2,
};

pub(super) fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    ensure!(
        layer_kv_dtypes.len() == config.num_hidden_layers,
        "laguna requires one KV dtype per attention layer"
    );
    ensure!(
        config.shared_expert_intermediate_size == config.moe_intermediate_size,
        "laguna fused shared-expert path requires equal shared/routed widths"
    );

    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let yarn_inv_freq = compute_yarn_inv_freq(config, gpu)?;
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);

    for i in 0..config.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
        let ffn = if config.mlp_only_layers.contains(&i) {
            load_dense_ffn(store, gpu, &lp)?
        } else {
            load_moe_ffn(store, config, gpu, &lp, absmax_k, quantize_k, stream)?
        };
        let layer = load_attention(
            store,
            config,
            gpu,
            &lp,
            input_norm,
            post_attn_norm,
            ffn,
            layer_kv_dtypes[i],
            yarn_inv_freq,
            i,
        )?;
        layers.push(Box::new(layer));
    }
    Ok(layers)
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

fn load_dense_ffn(store: &WeightStore, gpu: &dyn GpuBackend, lp: &str) -> Result<FfnComponent> {
    let mut layer = DenseFfnLayer::new(null_dense_ffn_weights(), gpu)?;
    layer.set_bf16_weights(
        dense_auto(store, &format!("{lp}.mlp.gate_proj.weight"), gpu)?,
        dense_auto(store, &format!("{lp}.mlp.up_proj.weight"), gpu)?,
        dense_auto(store, &format!("{lp}.mlp.down_proj.weight"), gpu)?,
    );
    Ok(FfnComponent::Dense(layer))
}

#[allow(clippy::too_many_arguments)]
fn load_moe_ffn(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<FfnComponent> {
    let mlp = format!("{lp}.mlp");
    let gate = dense(store, &format!("{mlp}.gate.weight"))?;
    let correction_bias = dense(store, &format!("{mlp}.experts.e_score_correction_bias"))?;
    let experts = (0..config.num_experts)
        .map(|e| {
            if !config.is_local_expert(e) {
                return Ok(ExpertWeight::null());
            }
            let ep = format!("{mlp}.experts.{e}");
            Ok(ExpertWeight {
                gate_proj: quantized_v2(store, &format!("{ep}.gate_proj"), gpu)?,
                up_proj: quantized_v2(store, &format!("{ep}.up_proj"), gpu)?,
                down_proj: quantized_v2(store, &format!("{ep}.down_proj"), gpu)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let shared = format!("{mlp}.shared_expert");
    let shared_gate = dense_auto(store, &format!("{shared}.gate_proj.weight"), gpu)?;
    let shared_up = dense_auto(store, &format!("{shared}.up_proj.weight"), gpu)?;
    let shared_down = dense_auto(store, &format!("{shared}.down_proj.weight"), gpu)?;
    let si = config.shared_expert_intermediate_size;
    let h = config.hidden_size;
    let shared_expert = ExpertWeight {
        gate_proj: quantize_to_nvfp4(&shared_gate, si, h, gpu, absmax_k, quantize_k, stream)?,
        up_proj: quantize_to_nvfp4(&shared_up, si, h, gpu, absmax_k, quantize_k, stream)?,
        down_proj: quantize_to_nvfp4(&shared_down, h, si, gpu, absmax_k, quantize_k, stream)?,
    };
    let weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate: DenseWeight {
            weight: DevicePtr::NULL,
        },
        experts,
        router_pre_norm: None,
        correction_bias: Some(correction_bias),
    };
    let mut layer = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    layer.predequant_for_prefill(gpu, config, stream)?;
    Ok(FfnComponent::Moe(layer))
}

#[allow(clippy::too_many_arguments)]
fn load_attention(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    kv_dtype: KvCacheDtype,
    yarn_inv_freq: DevicePtr,
    i: usize,
) -> Result<Qwen3AttentionLayer> {
    let p = format!("{lp}.self_attn");
    let heads = config.num_attention_heads_per_layer[i];
    let q_width = heads * config.head_dim;
    validate_matrix(
        store,
        &format!("{p}.q_proj.weight"),
        q_width,
        config.hidden_size,
    )?;
    validate_matrix(
        store,
        &format!("{p}.g_proj.weight"),
        heads,
        config.hidden_size,
    )?;
    validate_matrix(
        store,
        &format!("{p}.o_proj.weight"),
        config.hidden_size,
        q_width,
    )?;

    let q_proj = dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?;
    let k_proj = dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?;
    let v_proj = dense_auto(store, &format!("{p}.v_proj.weight"), gpu)?;
    let o_proj = dense_auto(store, &format!("{p}.o_proj.weight"), gpu)?;
    let (k_scale, v_scale) = load_kv_scales(store, gpu, &p)?;
    let attn = AttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj: QuantizedWeight::null(),
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        i,
        None,
        None,
        None,
        gpu,
        kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    layer.set_dimension_overrides(config.head_dim, heads, config.num_key_value_heads);
    layer.set_o_dense_bf16(o_proj);
    layer.set_head_gate_weight(
        dense_auto(store, &format!("{p}.g_proj.weight"), gpu)?,
        HeadGateActivation::Softplus,
    );
    match config.layer_types[i] {
        LayerType::SlidingAttention => {
            layer.set_sliding_window(Some(config.sliding_window));
            layer.set_rope_overrides(10_000.0, config.head_dim as u32);
        }
        LayerType::FullAttention => {
            layer.set_sliding_window(None);
            layer.set_rope_overrides(config.rope_theta as f32, config.rotary_dim() as u32);
            layer.set_yarn_rope(yarn_inv_freq, config.yarn_attention_factor);
        }
        other => anyhow::bail!("laguna layer {i} is not attention: {other:?}"),
    }
    Ok(layer)
}

fn validate_matrix(store: &WeightStore, key: &str, rows: usize, cols: usize) -> Result<()> {
    let tensor = store.get(key)?;
    ensure!(
        tensor.shape == [rows, cols],
        "{key} shape {:?}, expected [{rows}, {cols}]",
        tensor.shape
    );
    Ok(())
}

fn load_kv_scales(store: &WeightStore, gpu: &dyn GpuBackend, prefix: &str) -> Result<(f32, f32)> {
    Ok((
        load_scalar(store, gpu, &format!("{prefix}.k_scale"))?,
        load_scalar(store, gpu, &format!("{prefix}.v_scale"))?,
    ))
}

fn load_scalar(store: &WeightStore, gpu: &dyn GpuBackend, key: &str) -> Result<f32> {
    let tensor = store.get(key)?;
    ensure!(
        tensor.shape.iter().product::<usize>() == 1,
        "{key} must be scalar"
    );
    match tensor.dtype {
        WeightDtype::BF16 => {
            let mut bytes = [0u8; 2];
            gpu.copy_d2h(tensor.ptr, &mut bytes)?;
            Ok(f32::from_bits((u16::from_le_bytes(bytes) as u32) << 16))
        }
        WeightDtype::FP32 => {
            let mut bytes = [0u8; 4];
            gpu.copy_d2h(tensor.ptr, &mut bytes)?;
            Ok(f32::from_le_bytes(bytes))
        }
        dtype => anyhow::bail!("{key} must be BF16 or F32, got {dtype:?}"),
    }
}

fn compute_yarn_inv_freq(config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let dim = config.rotary_dim();
    let dim_f = dim as f32;
    let theta = config.rope_theta as f32;
    let max_pos = config.yarn_original_max_position_embeddings as f32;
    let correction = |rotations: f32| {
        (dim_f * (max_pos / (rotations * 2.0 * std::f32::consts::PI)).ln()) / (2.0 * theta.ln())
    };
    let low = correction(config.yarn_beta_fast).floor().max(0.0);
    let high = correction(config.yarn_beta_slow)
        .ceil()
        .min((dim - 1) as f32);
    let denominator = if (high - low).abs() < 1e-6 {
        0.001
    } else {
        high - low
    };
    let values = (0..dim / 2)
        .map(|j| {
            let base = theta.powf((2 * j) as f32 / dim_f);
            let ramp = ((j as f32 - low) / denominator).clamp(0.0, 1.0);
            (1.0 - ramp) / base + ramp / (config.yarn_factor * base)
        })
        .collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let ptr = gpu
        .alloc(bytes.len())
        .context("allocate laguna YaRN table")?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}
