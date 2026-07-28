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
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, PackedExpertWeights, PackedQ4Weight,
    PackedQ6Weight, QuantWeight, QuantizedWeight, dense, dense_auto, quantize_to_nvfp4,
    quantized_v2,
};

/// Wrap a keep-packed Q4_K store tensor (`{prefix}.weight`, tagged
/// [`WeightDtype::PackedQ4K`] by the GGUF loader) into a [`PackedQ4Weight`]
/// layer view. The pointer aliases the store's block buffer (no copy).
fn packed_q4_from_store(store: &WeightStore, prefix: &str) -> Result<PackedQ4Weight> {
    let t = store.get(&format!("{prefix}.weight"))?;
    ensure!(t.is_packed_q4k(), "{prefix}.weight is not keep-packed Q4_K");
    ensure!(
        t.shape.len() == 2,
        "{prefix}.weight is not 2D ({:?})",
        t.shape
    );
    Ok(PackedQ4Weight {
        weight: t.ptr,
        n: t.shape[0] as u32,
        k: t.shape[1] as u32,
    })
}

/// Wrap a keep-packed Q6_K store tensor into a [`PackedQ6Weight`] layer view.
fn packed_q6_from_store(store: &WeightStore, prefix: &str) -> Result<PackedQ6Weight> {
    let t = store.get(&format!("{prefix}.weight"))?;
    ensure!(t.is_packed_q6k(), "{prefix}.weight is not keep-packed Q6_K");
    ensure!(
        t.shape.len() == 2,
        "{prefix}.weight is not 2D ({:?})",
        t.shape
    );
    Ok(PackedQ6Weight {
        weight: t.ptr,
        n: t.shape[0] as u32,
        k: t.shape[1] as u32,
    })
}

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
    // Sliding layers: theta=10000 over the full head_dim, no YaRN ramp.
    let sliding_inv_freq = if sliding_rope_table_enabled() {
        compute_plain_inv_freq(10_000.0, config.head_dim, gpu)?
    } else {
        DevicePtr::NULL
    };
    let unified_moe_layout =
        unified_moe_layout_enabled(std::env::var("ATLAS_UNIFIED_MOE_LAYOUT").ok().as_deref());
    if unified_moe_layout {
        tracing::info!(
            "Laguna: using unified transposed MoE layout; prefill uses fused K64 kernels and decode uses transposed experts"
        );
    }
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);

    for i in 0..config.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
        let ffn = if config.mlp_only_layers.contains(&i) {
            load_dense_ffn(store, gpu, &lp)?
        } else {
            load_moe_ffn(
                store,
                config,
                gpu,
                &lp,
                absmax_k,
                quantize_k,
                stream,
                unified_moe_layout,
            )?
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
            sliding_inv_freq,
            i,
            absmax_k,
            quantize_k,
            stream,
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
    unified_moe_layout: bool,
) -> Result<FfnComponent> {
    let mlp = format!("{lp}.mlp");
    let mi = config.moe_intermediate_size;
    let h0 = config.hidden_size;

    // Keep-packed GGUF experts (Laguna Q4_K_M): the loader stored the routed
    // experts as raw Q4_K (gate/up) / Q6_K (down) blocks — detect via the store
    // dtype tag and build PackedExpertWeights so the MoE keep-packed prefill arm
    // computes NATIVELY on the packed blocks (q4k_mmq W4A8; weights never
    // dequant to a BF16 buffer, mirroring how the NVFP4 path computes on packed
    // weights). `experts` is left null and packed_experts carries the layer.
    let experts_keep_packed = store
        .get(&format!("{mlp}.experts.0.gate_proj.weight"))
        .map(|t| t.is_packed_q4k())
        .unwrap_or(false);

    let gate = dense(store, &format!("{mlp}.gate.weight"))?;
    // e_score_correction_bias: the GGUF loader dequants this originally-F32 tensor
    // to a BF16 device buffer, but `moe_topk_sigmoid_batched` reads `bias` as
    // `const float*` — handing it the BF16 buffer over-reads one buffer-length
    // past the end (CUDA_ERROR_ILLEGAL_ADDRESS, surfacing at whichever layer's
    // trailing bytes hit an unmapped page). Widen BF16→F32 into a correctly-sized
    // device buffer. Safetensors already delivers F32 on-device, so keep `dense`.
    let correction_bias = if experts_keep_packed {
        dense_bias_to_device(
            store,
            &format!("{mlp}.experts.e_score_correction_bias"),
            gpu,
        )?
    } else {
        dense(store, &format!("{mlp}.experts.e_score_correction_bias"))?
    };

    // ── Native full-BF16 MoE ──
    // A BF16 checkpoint (poolside/Laguna-XS-2.1 base) ships routed + shared
    // experts as plain BF16 `.weight` (no `.weight_packed`, no packed Q4K). The
    // NVFP4 `expert_proj` else-branch below would quantize them to NVFP4 at load
    // (Atlas min-max, no imatrix) — LOWER fidelity than the pre-calibrated NVFP4
    // checkpoint (factual recall degrades). Instead keep them BF16 and compute
    // via `moe_bf16_grouped_gemm` (the same stack qwen35 uses for FP8→BF16): load
    // DenseWeight arrays + `set_bf16_experts` (which installs the BF16 routed +
    // shared pointer tables and flips the forward onto the BF16 path).
    //
    // EXPERIMENTAL / opt-in (`ATLAS_LAGUNA_NATIVE_BF16=1`): the path LOADS + runs
    // + generates coherently, but a correctness bug remains — factual recall
    // degrades vs the NVFP4 checkpoint even though the BF16 base and the NVFP4
    // checkpoint are the SAME model (attention/embed/norms verified bit-identical;
    // only the experts are NVFP4-quantized in the checkpoint). The kernel +
    // set_bf16_experts are proven by qwen35, so the fault is in this laguna
    // BF16-MoE integration (suspects: down-proj transpose scratch, the routed-BF16
    // + shared-BF16 combo, or the BF16 decode path). Until fixed, the NVFP4
    // checkpoint is the recommended artifact; default is the (also-lossy)
    // quantize-at-load so a BF16 checkpoint at least serves without this bug.
    let experts_are_bf16 = !experts_keep_packed
        && store.contains(&format!("{mlp}.experts.0.gate_proj.weight"))
        && !store.contains(&format!("{mlp}.experts.0.gate_proj.weight_packed"));
    if experts_are_bf16 && std::env::var("ATLAS_LAGUNA_NATIVE_BF16").ok().as_deref() == Some("1") {
        let load_experts = |proj: &str| -> Result<Vec<DenseWeight>> {
            (0..config.num_experts)
                .map(|e| {
                    if !config.is_local_expert(e) {
                        return Ok(DenseWeight {
                            weight: DevicePtr::NULL,
                        });
                    }
                    dense_auto(store, &format!("{mlp}.experts.{e}.{proj}.weight"), gpu)
                })
                .collect()
        };
        let gate_bf16 = load_experts("gate_proj")?;
        let up_bf16 = load_experts("up_proj")?;
        let down_bf16 = load_experts("down_proj")?;
        let sh = format!("{mlp}.shared_expert");
        let sh_g = dense_auto(store, &format!("{sh}.gate_proj.weight"), gpu)?;
        let sh_u = dense_auto(store, &format!("{sh}.up_proj.weight"), gpu)?;
        let sh_d = dense_auto(store, &format!("{sh}.down_proj.weight"), gpu)?;
        let weights = MoeWeights {
            gate,
            shared_expert: ExpertWeight::null(),
            shared_expert_gate: DenseWeight {
                weight: DevicePtr::NULL,
            },
            experts: (0..config.num_experts)
                .map(|_| ExpertWeight::null())
                .collect(),
            packed_experts: None,
            router_pre_norm: None,
            correction_bias: Some(correction_bias),
        };
        let mut layer = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
        layer.set_bf16_experts(
            &gate_bf16,
            &up_bf16,
            &down_bf16,
            sh_g.weight,
            sh_u.weight,
            sh_d.weight,
            gpu,
        )?;
        tracing::info!(
            "Laguna MoE: NATIVE BF16 ({} routed + 1 shared experts kept BF16, no quant-at-load)",
            config.num_experts
        );
        return Ok(FfnComponent::Moe(layer));
    }

    let (experts, packed_experts) = if experts_keep_packed {
        let mut packed = Vec::with_capacity(config.num_experts);
        for e in 0..config.num_experts {
            if !config.is_local_expert(e) {
                packed.push(PackedExpertWeights {
                    gate: PackedQ4Weight::null_view(),
                    up: PackedQ4Weight::null_view(),
                    down: QuantWeight::PackedQ6(PackedQ6Weight::null_view()),
                });
                continue;
            }
            let ep = format!("{mlp}.experts.{e}");
            // down_proj is Q4_K on some layers, Q6_K on others (Q4_K_M mixed).
            let down_prefix = format!("{ep}.down_proj");
            let down = if store.get(&format!("{down_prefix}.weight"))?.is_packed_q4k() {
                QuantWeight::PackedQ4(packed_q4_from_store(store, &down_prefix)?)
            } else {
                QuantWeight::PackedQ6(packed_q6_from_store(store, &down_prefix)?)
            };
            packed.push(PackedExpertWeights {
                gate: packed_q4_from_store(store, &format!("{ep}.gate_proj"))?,
                up: packed_q4_from_store(store, &format!("{ep}.up_proj"))?,
                down,
            });
        }
        let null_experts = (0..config.num_experts)
            .map(|_| ExpertWeight::null())
            .collect();
        (null_experts, Some(packed))
    } else {
        // Existing NVFP4/safetensors path: pre-packed NVFP4 (`.weight_packed`)
        // or a BF16 GGUF (`.weight`) requantized to NVFP4 at load. Computed
        // natively by the grouped NVFP4 GEMM — no dequant-to-BF16 buffer.
        let expert_proj = |proj: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
            if store.contains(&format!("{proj}.weight_packed")) {
                quantized_v2(store, proj, gpu)
            } else {
                let bf16 = dense_auto(store, &format!("{proj}.weight"), gpu)?;
                quantize_to_nvfp4(&bf16, n, k, gpu, absmax_k, quantize_k, stream)
            }
        };
        let experts = (0..config.num_experts)
            .map(|e| {
                if !config.is_local_expert(e) {
                    return Ok(ExpertWeight::null());
                }
                let ep = format!("{mlp}.experts.{e}");
                Ok(ExpertWeight {
                    gate_proj: expert_proj(&format!("{ep}.gate_proj"), mi, h0)?,
                    up_proj: expert_proj(&format!("{ep}.up_proj"), mi, h0)?,
                    down_proj: expert_proj(&format!("{ep}.down_proj"), h0, mi)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        (experts, None)
    };

    let shared = format!("{mlp}.shared_expert");
    let si = config.shared_expert_intermediate_size;
    let h = config.hidden_size;
    // Shared-expert precision differs across Laguna variants: S-2.1 ships it
    // BF16 (in the quant ignore list) and keeps BF16 authoritative; XS-2.1 ships
    // it NVFP4-packed (`.weight_packed`, NOT ignored) like the routed experts.
    // Detect and load accordingly — mirrors the routed `expert_proj` above.
    let shared_packed = store.contains(&format!("{shared}.gate_proj.weight_packed"));
    let shared_proj = |proj: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        if shared_packed {
            quantized_v2(store, proj, gpu)
        } else {
            let bf16 = dense_auto(store, &format!("{proj}.weight"), gpu)?;
            quantize_to_nvfp4(&bf16, n, k, gpu, absmax_k, quantize_k, stream)
        }
    };
    let shared_expert = ExpertWeight {
        gate_proj: shared_proj(&format!("{shared}.gate_proj"), si, h)?,
        up_proj: shared_proj(&format!("{shared}.up_proj"), si, h)?,
        down_proj: shared_proj(&format!("{shared}.down_proj"), h, si)?,
    };
    // BF16 variant (S-2.1): also keep the raw BF16 tensors to set authoritative
    // (the NVFP4 copies above are placeholders overwritten before blending).
    // NVFP4-packed variant (XS-2.1): the NVFP4 shared_expert IS authoritative.
    let bf16_shared = if shared_packed {
        None
    } else {
        Some((
            dense_auto(store, &format!("{shared}.gate_proj.weight"), gpu)?,
            dense_auto(store, &format!("{shared}.up_proj.weight"), gpu)?,
            dense_auto(store, &format!("{shared}.down_proj.weight"), gpu)?,
        ))
    };
    let weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate: DenseWeight {
            weight: DevicePtr::NULL,
        },
        experts,
        packed_experts,
        router_pre_norm: None,
        correction_bias: Some(correction_bias),
    };
    let mut layer = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    // S-2.1 excludes the shared expert from NVFP4 compression: keep its BF16
    // weights authoritative for both prefill and decode; the quantized copies
    // above are placeholders for fused routed kernels and their shared
    // contribution is overwritten before blending. XS-2.1 ships the shared
    // expert NVFP4-packed → no BF16 override, so the NVFP4 `weights.shared_expert`
    // (same machinery as the routed NVFP4 experts) is the authoritative path.
    if let Some((sg, su, sd)) = bf16_shared {
        layer.set_bf16_shared_expert(sg, su, sd)?;
    }
    // Keep-packed GGUF experts: the routed experts are raw Q4_K/Q6_K blocks and
    // carry NO NVFP4 scale tables, so the NVFP4-specific transpose and CUTLASS
    // SFB swizzle below (which read the null NVFP4 expert scales) must be
    // skipped — the keep-packed MoE prefill arm consumes the packed blocks via
    // q4k_mmq instead.
    let experts_keep_packed = layer.weights.packed_experts.is_some();
    if unified_moe_layout && !experts_keep_packed {
        layer.transpose_for_prefill_unified(gpu, config)?;
    }
    // Native NVFP4 CUTLASS grouped MoE (ATLAS_HOLO_MOE_GROUPED_CUTLASS=1).
    // The routed grouped GEMMs are ~47% of Laguna's C=1 prefill GPU time and
    // otherwise run on the w4a16 kernels, which LUT-dequant NVFP4 to FP8 per
    // tile. The SFB swizzle is built from whichever scale tables exist —
    // transposed [K/16,N] under the unified layout, else the checkpoint's own
    // [N,K/16] via the src_n_major packer path.
    if cutlass_grouped_moe_enabled() && !experts_keep_packed {
        layer.build_cutlass_grouped_sfb(gpu, config, gpu.default_stream())?;
    }
    Ok(FfnComponent::Moe(layer))
}

fn unified_moe_layout_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
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
    sliding_inv_freq: DevicePtr,
    i: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
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

    // Lever A (ATLAS_LAGUNA_ATTN_NVFP4=1): the NVFP4 checkpoint keeps attention
    // BF16 (excluded from compression), which is ~33% of the decode weight-byte
    // traffic and the entire concurrency gap to GGUF (roofline: CUTLASS MoE is
    // already at the DRAM floor). Quantizing q/k/v/o to NVFP4 at load halves
    // those bytes → projected C4 43.6→~57.6 (beats vLLM 51 + GGUF 56). The
    // decode/prefill NVFP4 attention GEMV kernels already exist (fire on
    // as_nvfp4().is_some()). Opt-in; quality is uncalibrated static min-max so
    // gate behind the coherence check. Default OFF = BF16 attention unchanged.
    let attn_nvfp4 = std::env::var("ATLAS_LAGUNA_ATTN_NVFP4")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let kv_width = config.num_key_value_heads * config.head_dim;
    // Quantize q/k/v to NVFP4 (multi-seq ms_qkv_batch2/3/n read them batched-ONCE).
    // o_proj is kept BF16: MEASURED, o-NVFP4's w4a16_gemv_batch4/batch8 arms are
    // SLOWER than the BF16 dense_gemv_batchm at C=4/8 (full qkvo-NVFP4 tanked C4
    // 45.9→33.1, C8 55.2→49), so o stays on its batched BF16 read-once path.
    // Net win: single-stream C1 19.3→~30 (+55%, beats vLLM/GGUF) + C2/C4 gains;
    // a faster batched-NVFP4 o (and levers B/C: shared-expert/lm_head) is the
    // follow-up for concurrency parity with GGUF. o_proj is [N=hidden, K=q_width].
    let (q_nvfp4, k_nvfp4, v_nvfp4) = if attn_nvfp4 {
        (
            Some(quantize_to_nvfp4(
                &q_proj,
                q_width,
                config.hidden_size,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?),
            Some(quantize_to_nvfp4(
                &k_proj,
                kv_width,
                config.hidden_size,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?),
            Some(quantize_to_nvfp4(
                &v_proj,
                kv_width,
                config.hidden_size,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?),
        )
    } else {
        (None, None, None)
    };

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
        q_nvfp4,
        k_nvfp4,
        v_nvfp4,
        gpu,
        kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    layer.set_dimension_overrides(config.head_dim, heads, config.num_key_value_heads);
    // Lever A off (default): BF16 o_proj keeps the batched read-once
    // dense_gemv_batchm arm at n=2..8 — unchanged default path. Lever A on:
    // o_proj was installed as NVFP4 in `attn` above; o_dense_bf16 must stay
    // o_proj is always BF16 (batched read-once at n=2..8) — o-NVFP4 measured
    // slower at concurrency, so lever A quantizes only q/k/v.
    layer.set_o_dense_bf16(o_proj);
    layer.set_head_gate_weight(
        dense_auto(store, &format!("{p}.g_proj.weight"), gpu)?,
        HeadGateActivation::Softplus,
    );
    match config.layer_types[i] {
        LayerType::SlidingAttention => {
            layer.set_sliding_window(Some(config.sliding_window));
            layer.set_rope_overrides(10_000.0, config.head_dim as u32);
            if !sliding_inv_freq.is_null() {
                // attention_factor = 1.0 => cos/sin unscaled, i.e. plain RoPE.
                layer.set_yarn_rope(sliding_inv_freq, 1.0);
            }
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
    // GGUF checkpoints carry no calibrated FP8 KV scales; default to 1.0 (no
    // scaling) when absent so they load. The NVFP4 safetensors checkpoint ships
    // real k_scale/v_scale and is unchanged. (FP8 KV without calibration clips —
    // serve GGUF with --kv-cache-dtype bf16; the scales are then inert anyway.)
    let load_opt = |name: String| -> Result<f32> {
        if store.contains(&name) {
            load_scalar(store, gpu, &name)
        } else {
            Ok(1.0)
        }
    };
    Ok((
        load_opt(format!("{prefix}.k_scale"))?,
        load_opt(format!("{prefix}.v_scale"))?,
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

/// Precomputed plain RoPE inv_freq table for the sliding-attention layers.
///
/// Those layers use theta=10000 over the full head_dim with no YaRN ramp, and
/// the default rope kernel recomputes `1/theta^(2j/dim)` on the GPU with an
/// FP64 `pow` per pair index per block (kernels/gb10/common/rope.cu). For
/// Laguna's sliding layers rotary_dim == head_dim == 128, so a block covers
/// only 2 positions and pays 64 doubles to produce them — measured at 6.3% of
/// C=1 prefill GPU time. The table-based `rope_yarn_scaled` kernel is already
/// wired for this model (it serves the full-attention YaRN layers); feeding it
/// a plain table with attention_factor = 1.0 is the same math without the
/// per-block transcendentals.
///
/// Computed in f64 and narrowed once, so the stored values are at least as
/// accurate as the kernel's own FP64 `pow` followed by an f32 store.
/// Build the CUTLASS grouped-NVFP4 SFB tables at load
/// (`ATLAS_HOLO_MOE_GROUPED_CUTLASS=1`). Costs ~7.1 GB of device memory for
/// Laguna (256 experts x 47 layers x 3 projections), so it is opt-in.
fn cutlass_grouped_moe_enabled() -> bool {
    matches!(
        std::env::var("ATLAS_HOLO_MOE_GROUPED_CUTLASS").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Materialise the sigmoid router's `e_score_correction_bias` as an F32 device
/// buffer for the keep-packed GGUF path.
///
/// The GGUF loader dequants this originally-F32 tensor to **BF16** on load (2
/// bytes/elem), but `moe_topk_sigmoid_batched` reads `bias` as `const float*`
/// (4 bytes/elem). Handing it the BF16 buffer makes the kernel over-read one
/// buffer-length past the end — a CUDA_ERROR_ILLEGAL_ADDRESS that surfaces at
/// whichever layer's trailing bytes land on an unmapped page (seen drifting
/// across layers with prompt length). Widen BF16→F32 here into a correctly
/// sized [num_experts] F32 device allocation. Safetensors already ships an F32
/// device pointer and keeps `dense()`, so this runs on the keep-packed path only.
fn dense_bias_to_device(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<crate::weight_map::DenseWeight> {
    let t = store
        .get(name)
        .with_context(|| format!("keep-packed bias {name} missing"))?;
    let n = t.num_elements();
    let src_bytes = t.byte_size();
    // `t.ptr` is a DEVICE buffer — the GGUF loader dequants this bias to BF16 on
    // GPU. Copy it down at its true (BF16) size, widen on the host, upload as F32.
    let mut host = vec![0u8; src_bytes];
    gpu.copy_d2h(t.ptr, &mut host)
        .with_context(|| format!("copy_d2h correction_bias {name} ({src_bytes}B)"))?;
    let f32_bytes: Vec<u8> = match t.dtype {
        WeightDtype::BF16 => host
            .chunks_exact(2)
            .flat_map(|c| {
                let bf = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bf as u32) << 16).to_le_bytes()
            })
            .collect(),
        WeightDtype::FP32 => host,
        d => anyhow::bail!("unexpected correction_bias dtype {d:?} for {name}"),
    };
    let ptr = gpu
        .alloc(n * 4)
        .with_context(|| format!("allocate F32 device copy of {name}"))?;
    gpu.copy_h2d(&f32_bytes, ptr)?;
    Ok(crate::weight_map::DenseWeight { weight: ptr })
}

fn compute_plain_inv_freq(theta: f64, dim: usize, gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let bytes = (0..dim / 2)
        .map(|j| (1.0f64 / theta.powf((2 * j) as f64 / dim as f64)) as f32)
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let ptr = gpu
        .alloc(bytes.len())
        .context("allocate laguna sliding-layer RoPE table")?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

/// Opt out of the precomputed sliding-layer RoPE table with
/// `ATLAS_LAGUNA_ROPE_TABLE=0` (falls back to the on-the-fly rope kernel).
fn sliding_rope_table_enabled() -> bool {
    std::env::var("ATLAS_LAGUNA_ROPE_TABLE").as_deref() != Ok("0")
}

#[cfg(test)]
mod tests {
    use super::unified_moe_layout_enabled;

    #[test]
    fn unified_moe_layout_is_explicitly_opt_in() {
        assert!(unified_moe_layout_enabled(Some("1")));
        assert!(unified_moe_layout_enabled(Some("true")));
        assert!(unified_moe_layout_enabled(Some("TRUE")));
        assert!(!unified_moe_layout_enabled(None));
        assert!(!unified_moe_layout_enabled(Some("0")));
        assert!(!unified_moe_layout_enabled(Some("full")));
    }
}
