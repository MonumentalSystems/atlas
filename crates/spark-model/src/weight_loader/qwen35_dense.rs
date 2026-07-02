// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::{ModelWeightLoader, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::{DenseFfnLayer, FfnComponent, Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8Weight, MtpWeights, Nvfp4Variant, SsmWeights, dense,
    dense_auto, dense_f32_safe, dense_keep_f32, dequant_nvfp4_to_bf16, detect_nvfp4_variant,
    gpu_concat_rows, interleave_ba, load_dense_ffn, load_fp8_block_scaled_as_fp8weight,
    load_kv_scales, load_mtp, quantize_to_nvfp4, quantized_auto,
};

/// True when `{prefix}.weight` is FP8 E4M3 on disk with a 2D block scale
/// (`weight_scale_inv` or 2D `weight_scale`) — i.e. a native FP8 checkpoint
/// projection that should load as `Fp8Weight` rather than be requantized to
/// NVFP4. Mirrors `qwen35::load_layers::proj_is_native_fp8`.
fn proj_is_native_fp8(store: &WeightStore, prefix: &str) -> bool {
    let is_fp8_weight = store
        .get(&format!("{prefix}.weight"))
        .map(|w| w.dtype == WeightDtype::FP8E4M3)
        .unwrap_or(false);
    // Any FP8 dequant scale qualifies: 2D block scale (`weight_scale_inv` or a
    // 2D `weight_scale`) OR a scalar per-tensor `weight_scale` (ModelOpt
    // MIXED_PRECISION, e.g. nvidia/Qwen3.6-27B-NVFP4's per-tensor-FP8 attention).
    // `load_fp8_block_scaled_as_fp8weight` broadcasts a scalar into the
    // [N/128,K/128] block matrix, so all three are consumable by the W8A16 kernels.
    let has_fp8_scale = store.contains(&format!("{prefix}.weight_scale_inv"))
        || store.contains(&format!("{prefix}.weight_scale"));
    is_fp8_weight && has_fp8_scale
}

/// Opt-in gate for native dense-FP8 attention + FFN dispatch (Qwythos / dense
/// Ornith-FP8). Default OFF.
///
/// VERIFIED 2026-06-29 on Qwythos-9B-FP8 (gb10/ornith-1.0-9b): with the flag
/// on, the FP8 arms fire for all 32 FFN + 8 full-attn layers and text is
/// correct (coherence/fib/tools 3/3). BUT it is NOT a perf win — ~30 tok/s vs
/// ~40 for the NVFP4 fallback — because this target's NVFP4 W4A16 kernels
/// (fused dual-GEMV decode, transposed m128 prefill) are more optimized than
/// its FP8 W8A16 kernels (unfused per-projection GEMV, non-transposed
/// `w8a16_gemm` prefill; the attention FP8 prefill transpose also does not
/// engage). Vision prefill additionally hits a CUDA-700. Making FP8 pay off
/// here needs dedicated dense-FP8 kernels (fused FP8 dual-GEMV + fast
/// transposed FP8 prefill GEMM), not loader wiring. Until then NVFP4 autoquant
/// is the better dense runtime. `ATLAS_DENSE_FP8=1` opts in for that kernel work.
fn dense_fp8_enabled() -> bool {
    std::env::var("ATLAS_DENSE_FP8").as_deref() == Ok("1")
}

mod loaders_b;

pub struct Qwen35DenseWeightLoader;

impl ModelWeightLoader for Qwen35DenseWeightLoader {
    fn supports_tp(&self) -> bool {
        // FullAttention layers are TP-sharded (NVFP4-from-disk and BF16
        // → NVFP4 paths). LinearAttention (GDN SSM) layers run
        // full-replica per rank — see qwen35.rs for the rationale.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types = if config.layer_types.is_empty() {
            (0..config.num_hidden_layers)
                .map(|i| config.layer_type(i))
                .collect::<Vec<_>>()
        } else {
            config.layer_types.clone()
        };

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;

        let variant = detect_nvfp4_variant(store, config);
        let weight_format = WeightFormat::detect(store, config);
        tracing::info!(
            "Weight format: {:?}, NVFP4 variant: {:?}",
            weight_format,
            variant
        );

        // Native FP8 SSM prefill GEMM (Qwen3.6-27B-FP8 root-cause fix,
        // commit 3ebc08a). Atlas's prior SSM in_proj_qkv path was
        // FP8 → BF16 → NVFP4 → BF16 (in `w4a16_gemm` dequant) → MMA — a
        // double-quant chain whose NVFP4 hop's ~4-bit per-group precision
        // is dominated by signal at q/v but attenuated into a k-direction
        // error (HF conv-k ‖6.3‖ vs conv-v ‖117.2‖, ~18× smaller). For
        // every FP8-on-disk checkpoint we install a single-scale FP8 copy
        // of the stacked `[QKV|Z]` and `out_proj` weights for prefill,
        // bypassing the NVFP4 intermediate. Prefill dispatches via the
        // existing `fp8_gemm_n128` (BF16 act × FP8 weight) — same path
        // the MoE shared-expert FP8 prefill uses. Decode/GEMV unchanged.
        // Originally env-gated `ATLAS_FP8_SSM_PREFILL=1`; promoted to
        // unconditional 2026-05-20 after live verification (commit
        // dfb4e8a era, tokens_to_first_degeneration 1,196 → 16,968).
        let fp8_ssm_prefill = matches!(variant, Nvfp4Variant::Fp8Dequanted);
        let bf16_to_fp8_k = if fp8_ssm_prefill {
            tracing::info!(
                "SSM in_proj_qkv + out_proj via native FP8 prefill GEMM \
                 (BF16 act × FP8 weight via fp8_gemm_n128); NVFP4 kept as \
                 structural fallback for decode batch paths"
            );
            Some(gpu.kernel("w4a16", "bf16_to_fp8")?)
        } else {
            None
        };

        // ATLAS_MEM_PROFILE: per-phase GPU-free trace to pin the strix/APU
        // load-time footprint (FP8-source persistence vs NVFP4 steady-state vs
        // BF16 requant transients). Gated env so it's a no-op in production.
        let mem_profile = std::env::var("ATLAS_MEM_PROFILE").is_ok();
        let log_free = |tag: &str| {
            if mem_profile && let Ok(free) = gpu.free_memory() {
                tracing::info!("MEM_PROFILE[{tag}]: {:.2} GB GPU-free", free as f64 / 1e9);
            }
        };
        log_free("dense-load-start");

        for (i, lt) in layer_types.iter().enumerate() {
            if i % 8 == 0 {
                log_free(&format!("layer-{i}"));
            }
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            // Dense FFN instead of MoE. Native FP8 checkpoints (single-GPU)
            // load gate/up/down directly as block-scaled `Fp8Weight` and
            // dispatch w8a16 — no NVFP4 requant. TP>1 still uses the NVFP4
            // path (FP8 FFN sharding is a follow-up).
            let ffn_fp8 = dense_fp8_enabled()
                && config.tp_world_size.max(1) == 1
                && matches!(variant, Nvfp4Variant::Fp8Dequanted)
                && proj_is_native_fp8(store, &format!("{lp}.mlp.gate_proj"));
            // Always load the NVFP4 weights so every dispatch path (incl. the
            // batched spec-decode forward_k2/k3 paths that have no FP8 branch)
            // has a valid weight to fall back to — a null NVFP4 weight under a
            // w4a16 dispatch is the CUDA-700-at-concurrency bug. When native
            // FP8 is enabled we overlay the block-scaled FP8 weights on top;
            // the hot forward / forward_prefill paths then use FP8, the rare
            // batched paths fall back to real NVFP4.
            let ffn_weights = load_dense_ffn(
                store, &lp, gpu, variant, absmax_k, quantize_k, stream, config,
            )?;
            let mut ffn_layer = DenseFfnLayer::new(ffn_weights, gpu)?;
            if ffn_fp8 {
                let load_ffn_fp8 = |name: &str| {
                    load_fp8_block_scaled_as_fp8weight(store, &format!("{lp}.mlp.{name}"), gpu)
                };
                ffn_layer.set_fp8_weights(
                    load_ffn_fp8("gate_proj")?,
                    load_ffn_fp8("up_proj")?,
                    load_ffn_fp8("down_proj")?,
                );
            }
            let ffn = FfnComponent::Dense(ffn_layer);

            match lt {
                LayerType::FullAttention => {
                    let p = format!("{lp}.self_attn");
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let (attn, q_nvfp4, k_nvfp4, v_nvfp4) = match variant {
                        Nvfp4Variant::CompressedTensors => {
                            // NVFP4-from-disk path: column-parallel Q/K/V, row-parallel O.
                            let group_size = 16usize;
                            let load_nvfp4 = |name: &str,
                                              full_n: usize,
                                              full_k: usize,
                                              kind: TpShardKind|
                             -> Result<crate::weight_map::QuantizedWeight> {
                                let src = quantized_auto(store, &format!("{p}.{name}"), gpu, variant)?;
                                if tp_size == 1 {
                                    return Ok(src);
                                }
                                let sharded = shard_quantized_nvfp4(
                                    &src, full_n, full_k, kind, tp_rank, tp_size, group_size, gpu,
                                )?;
                                gpu.free(src.weight)?;
                                gpu.free(src.weight_scale)?;
                                Ok(sharded)
                            };
                            let [q, k, v, o] = load_qkvo_tp(config, load_nvfp4)?;
                            let dummy = DenseWeight {
                                weight: spark_runtime::gpu::DevicePtr::NULL,
                            };
                            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
                            let attn = AttentionWeights {
                                q_proj: dummy,
                                k_proj: dummy,
                                v_proj: dummy,
                                o_proj: o,
                                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                                q_norm_full: None,
                                k_norm_full: None,
                                k_scale,
                                v_scale,
                            };
                            (attn, Some(q), Some(k), Some(v))
                        }
                        Nvfp4Variant::Standard
                        | Nvfp4Variant::Fp8Dequanted
                        | Nvfp4Variant::Bf16Raw => {
                            // BF16 → NVFP4 path: shard BF16 then quantize per-rank.
                            let load_bf16_then_nvfp4 = |name: &str,
                                                        full_n: usize,
                                                        full_k: usize,
                                                        kind: TpShardKind|
                             -> Result<(
                                DenseWeight,
                                crate::weight_map::QuantizedWeight,
                            )> {
                                let src = dense_auto(store, &format!("{p}.{name}.weight"), gpu)?;
                                let (sharded_ptr, local_n, local_k) = shard_dense_bf16(
                                    src.weight, full_n, full_k, kind, tp_rank, tp_size, gpu,
                                )?;
                                let sharded = DenseWeight {
                                    weight: sharded_ptr,
                                };
                                let q = quantize_to_nvfp4(
                                    &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
                                )?;
                                if sharded_ptr != src.weight {
                                    gpu.free(sharded_ptr)?;
                                }
                                Ok((src, q))
                            };
                            let [
                                (q_dense, q_nvfp4),
                                (k_dense, k_nvfp4),
                                (v_dense, v_nvfp4),
                                (o_dense, o_nvfp4),
                            ] = load_qkvo_tp(config, load_bf16_then_nvfp4)?;

                            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

                            // The BF16 q/k/v/o dense tensors are only the intermediate
                            // fed to the GPU quantize_to_nvfp4 above. Prefill AND decode
                            // always dispatch the NVFP4 weights, so the BF16 copies are
                            // dead once quantized. Free them instead of retaining a full
                            // second copy of every projection (Atlas issue #A1).
                            gpu.free(q_dense.weight)?;
                            gpu.free(k_dense.weight)?;
                            gpu.free(v_dense.weight)?;
                            gpu.free(o_dense.weight)?;

                            let attn = AttentionWeights {
                                q_proj: DenseWeight {
                                    weight: spark_runtime::gpu::DevicePtr::NULL,
                                },
                                k_proj: DenseWeight {
                                    weight: spark_runtime::gpu::DevicePtr::NULL,
                                },
                                v_proj: DenseWeight {
                                    weight: spark_runtime::gpu::DevicePtr::NULL,
                                },
                                o_proj: o_nvfp4,
                                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                                q_norm_full: None,
                                k_norm_full: None,
                                k_scale,
                                v_scale,
                            };
                            (attn, Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4))
                        }
                    };

                    let mut layer = Qwen3AttentionLayer::new(
                        input_norm,
                        attn,
                        post_attn_norm,
                        ffn,
                        attn_idx,
                        q_nvfp4,
                        k_nvfp4,
                        v_nvfp4,
                        gpu,
                        layer_kv_dtypes[attn_idx],
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;
                    // Overlay native FP8 q/k/v/o on top of the NVFP4 weights when
                    // enabled (single-GPU FP8 checkpoint). Hot decode/prefill paths
                    // dispatch FP8 (w8a16); any path without an FP8 branch falls back
                    // to the real NVFP4 weights above (never a null → no CUDA-700).
                    // Overlay fires whenever attention is natively FP8 on disk —
                    // NOT gated on the whole-model `Fp8Dequanted` variant, so
                    // mixed-precision NVFP4-FFN + FP8-attn dense checkpoints
                    // (nvidia/Qwen3.6-27B-NVFP4, `Standard` variant, per-tensor-FP8
                    // attn) keep attention native FP8 instead of the lossy
                    // FP8->BF16->NVFP4 requant. Still opt-in via ATLAS_DENSE_FP8=1
                    // (dense FP8 W8A16 attn kernels are ~lossless but slower than the
                    // NVFP4 W4A16 fallback — a quality/speed tradeoff).
                    if dense_fp8_enabled()
                        && config.tp_world_size.max(1) == 1
                        && proj_is_native_fp8(store, &format!("{p}.q_proj"))
                    {
                        let load_fp8_proj = |name: &str,
                                             _n: usize,
                                             _k: usize,
                                             _kind: TpShardKind|
                         -> Result<Fp8Weight> {
                            load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.{name}"), gpu)
                        };
                        let [q_fp8, k_fp8, v_fp8, o_fp8] = load_qkvo_tp(config, load_fp8_proj)?;
                        layer.set_fp8_weights(Some(q_fp8), Some(k_fp8), Some(v_fp8), Some(o_fp8));
                        if i < 2 {
                            tracing::info!(
                                "Layer {i}: attention native FP8 overlay ACTIVE (W8A16, no NVFP4 requant)"
                            );
                        }
                        if let Err(e) = layer.transpose_fp8_for_prefill(gpu, stream) {
                            tracing::warn!("Layer {i}: dense FP8 transpose failed: {e}");
                        }
                    }
                    layers.push(Box::new(layer));
                    attn_idx += 1;
                }
                LayerType::LinearAttention => {
                    let nv = config.linear_num_value_heads;
                    let nk = config.linear_num_key_heads;
                    let qkv_rows = config.ssm_qkv_size();
                    let z_rows = config.ssm_z_size();
                    let value_dim = nv * config.linear_value_head_dim;
                    let la = format!("{lp}.linear_attn");

                    // SSM projections are loaded per-projection by on-disk dtype:
                    // each of in_proj_qkv / in_proj_z / out_proj may independently
                    // be NVFP4-packed (`weight_packed`) or plain (`weight`, routed
                    // by `dense_auto` → BF16/FP32/FP8). The unsloth NVFP4 re-quant
                    // of Qwen3.6-27B quantizes ONLY out_proj while keeping the
                    // in_proj_* in BF16; the old all-or-nothing gate (keyed on
                    // in_proj_qkv.weight_packed) then looked for a non-existent
                    // out_proj.weight and failed to build. `dense_auto` is dequant-
                    // to-BF16 for the concat pipeline regardless of source dtype.
                    let load_ssm_proj =
                        |name: &str, rows: usize, cols: usize| -> Result<DenseWeight> {
                            if store.contains(&format!("{name}.weight_packed")) {
                                dequant_nvfp4_to_bf16(store, name, rows, cols, gpu)
                            } else {
                                dense_auto(store, &format!("{name}.weight"), gpu)
                            }
                        };
                    let qkv_dense = load_ssm_proj(&format!("{la}.in_proj_qkv"), qkv_rows, h)?;
                    let z_dense = load_ssm_proj(&format!("{la}.in_proj_z"), z_rows, h)?;
                    let out_proj_dense = load_ssm_proj(&format!("{la}.out_proj"), h, value_dim)?;

                    // A, B are always BF16
                    let in_proj_a = dense(store, &format!("{la}.in_proj_a.weight"))?;
                    let in_proj_b = dense(store, &format!("{la}.in_proj_b.weight"))?;
                    let conv1d = dense(store, &format!("{la}.conv1d.weight"))?;
                    // A_log and dt_bias MUST be FP32 — consumer kernels in
                    // `ssm_preprocess.cu` and `mamba2_ssm_decode.cu` declare
                    // them `const float*`. Loading via `dense()` kept BF16
                    // storage, reinterpreting 48-elt BF16 (96B) as 48-elt
                    // FP32 → per-head scrambled decay gates and exponential
                    // error amplification through GDR recurrence at long
                    // context. The MoE sister loader (`ssm_qwen35.rs`)
                    // already promotes these; dense was missing the mirror.
                    let a_log = dense_keep_f32(store, &format!("{la}.A_log"), gpu)?;
                    let dt_bias = dense_keep_f32(store, &format!("{la}.dt_bias"), gpu)?;
                    // norm.weight: use `dense_f32_safe` (FP32-aware: detects
                    // a fp32 checkpoint and truncates to BF16 with logging;
                    // bf16 passes through). Mirrors `weight_map/ssm_qwen35.rs`
                    // MoE sister loader (backported here 2026-05-20).
                    let norm = dense_f32_safe(store, &format!("{la}.norm.weight"), gpu)?;

                    let qkvz_dense =
                        gpu_concat_rows(&qkv_dense, qkv_rows, &z_dense, z_rows, h, gpu)?;
                    // qkv/z BF16 are only inputs to the concat above; free them now
                    // rather than leaking them for the layer's lifetime (Atlas issue #A1).
                    gpu.free(qkv_dense.weight)?;
                    gpu.free(z_dense.weight)?;

                    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;

                    let qkvz_size = config.ssm_qkvz_size();
                    let qkvz_nvfp4 = quantize_to_nvfp4(
                        &qkvz_dense,
                        qkvz_size,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;

                    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

                    let out_proj_nvfp4 = quantize_to_nvfp4(
                        &out_proj_dense,
                        h,
                        value_dim,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;

                    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

                    // Native FP8 SSM prefill GEMM: build a single-scale FP8
                    // copy of `qkvz_dense` [qkvz_size, h] and `out_proj_dense`
                    // [h, value_dim] by direct BF16→FP8 truncation. SSM weight
                    // magnitudes fit in FP8 E4M3 range (|w| ≤ 448), so no
                    // separate scalar dequant is needed at GEMM time — the
                    // `fp8_gemm_n128` kernel interprets the FP8 bytes as
                    // values directly (mirrors how `predequant_nvfp4_to_fp8`
                    // bakes `scale2` into the FP8 stream). PCND: gated.
                    let (qkvz_fp8_prefill, out_proj_fp8_prefill) =
                        if let Some(b2f_k) = bf16_to_fp8_k {
                            let qkvz_total = (qkvz_size * h) as u32;
                            let qkvz_fp8 = gpu.alloc(qkvz_size * h)?;
                            crate::layers::ops::bf16_to_fp8(
                                gpu,
                                b2f_k,
                                qkvz_dense.weight,
                                qkvz_fp8,
                                qkvz_total,
                                stream,
                            )?;
                            let out_total = (h * value_dim) as u32;
                            let out_fp8 = gpu.alloc(h * value_dim)?;
                            crate::layers::ops::bf16_to_fp8(
                                gpu,
                                b2f_k,
                                out_proj_dense.weight,
                                out_fp8,
                                out_total,
                                stream,
                            )?;
                            gpu.synchronize(stream)?;
                            (Some(qkvz_fp8), Some(out_fp8))
                        } else {
                            (None, None)
                        };

                    // SSM prefill/decode always dispatch qkvz_nvfp4/_t and the NVFP4
                    // out_proj; the BF16 qkvz_dense / out_proj_dense were only quantize
                    // inputs. Free them rather than keep a third full-precision copy of
                    // the largest SSM tensor across every layer (Atlas issue #A1).
                    gpu.free(qkvz_dense.weight)?;
                    gpu.free(out_proj_dense.weight)?;

                    let ssm = SsmWeights {
                        in_proj_qkvz: DenseWeight {
                            weight: spark_runtime::gpu::DevicePtr::NULL,
                        },
                        in_proj_ba: ba_dense,
                        conv1d,
                        a_log,
                        dt_bias,
                        norm,
                        out_proj: out_proj_nvfp4,
                    };

                    let mut layer = Qwen3SsmLayer::new_sequential(
                        input_norm,
                        ssm,
                        post_attn_norm,
                        ffn,
                        Some(qkvz_nvfp4),
                        Some(qkvz_nvfp4_t),
                        Some(out_proj_nvfp4_t),
                        config,
                        gpu,
                    )?;
                    layer.predequant_for_prefill(gpu, config, stream)?;
                    // Install the FP8 prefill weights AFTER `predequant_for_prefill`
                    // (which sets `out_proj_fp8` from NVFP4 + scale2). The
                    // native-FP8 path overrides both pointers when active,
                    // routing prefill through `fp8_gemm_n128` instead of
                    // `w4a16_gemm_t`. Decode batch paths keep their NVFP4
                    // fallback (the `qkvz_nvfp4*` fields above).
                    if qkvz_fp8_prefill.is_some() || out_proj_fp8_prefill.is_some() {
                        layer.set_fp8_prefill_only_weights(qkvz_fp8_prefill, out_proj_fp8_prefill);
                    }
                    layers.push(Box::new(layer));
                }
                LayerType::SlidingAttention => {
                    unreachable!("unexpected SlidingAttention in this loader")
                }
                LayerType::Moe => unreachable!("Qwen3.5 dense has no standalone MoE layers"),
            }

            if (i + 1) % 10 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Qwen3.5 dense weight loader: {} layers ({} attention, {} SSM, dense FFN)",
            layers.len(),
            attn_idx,
            layers.len() - attn_idx,
        );

        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loaders_b::load_embedding(store, config)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loaders_b::load_final_norm(store, config)
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        loaders_b::load_lm_head(store, config)
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "Loading dense MTP weights (variant={:?}, hidden={}, inter={})",
            variant,
            config.hidden_size,
            config.intermediate_size,
        );
        // `load_mtp` auto-detects MoE vs dense FFN by inspecting the weight
        // names. For dense Qwen3.6-27B-FP8 it returns a MtpWeights with
        // `dense_ffn = Some(...)` and NULL placeholders for the MoE fields.
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        if mtp.dense_ffn.is_some() {
            tracing::info!("Dense MTP head ready (FP8 e4m3 projections + dense gate/up/down MLP)");
        } else {
            tracing::info!(
                "MoE MTP head ready ({} experts) — dense loader sees MoE bundle",
                mtp.experts.len(),
            );
        }
        Ok(Some(mtp))
    }

    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionEncoder>> {
        // Dense Qwen3.5 / Holo VL checkpoints (e.g. Holo-3.1-0.8B, Ornith-1.0-9B)
        // ship the SAME Qwen3-VL ViT tower as their MoE siblings. The MoE
        // loader's `load_vision_encoder` reads only `store` + `config.vision`
        // (no MoE-specific state), so reuse it verbatim. The shared model
        // forward (`model/trait_impl/*`, gated on `vision_encoder.is_some()`)
        // then merges image embeddings — no dense-specific forward changes.
        super::qwen35::Qwen35WeightLoader.load_vision_encoder(store, config, gpu)
    }
}
