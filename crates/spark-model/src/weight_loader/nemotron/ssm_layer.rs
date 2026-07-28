// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H Mamba-2 SSM layer construction: mixed-quant weight load plus the
//! FP8 / transposed-NVFP4 prefill weight copies. Split from `nemotron.rs`
//! (500-LoC cap).

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use super::NemotronHWeightLoader;
use crate::layers::NemotronMamba2Layer;
use crate::weight_map::{
    DenseWeight, NemotronSsmQuant, dense, dequant_fp8_to_bf16_into,
    load_fp8_block_scaled_as_fp8weight, load_nemotron_ssm, quantize_to_nvfp4,
};

impl NemotronHWeightLoader {
    /// Build one Mamba-2 SSM layer (the `LayerType::LinearAttention` arm).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_ssm_layer(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        config: &ModelConfig,
        i: usize,
        h: usize,
        lp: &str,
        norm: DenseWeight,
        quantize_k: KernelHandle,
        absmax_k: KernelHandle,
        scratch: DevicePtr,
        stream: u64,
    ) -> Result<NemotronMamba2Layer> {
        // Mamba-2 SSM layer (mixed quant: NVFP4, FP8, or BF16)
        let (mut ssm, quant_kind) = load_nemotron_ssm(store, i, gpu, lp)?;
        let p = format!("{lp}.mixer");
        // ModelOpt ships this checkpoint's SSM projections as FP8 inside an
        // otherwise-NVFP4 repo. The legacy path dequantized them to BF16 and
        // re-quantized to NVFP4 under ONE global scale spanning the whole
        // [18048, 4096] tensor (measured at load: global_max=0.394531,
        // scale2=0.00014678 over 73.9M elements). Puzzle is Mamba-dominant —
        // only 8 attention layers — so these two projections carry the entire
        // sequence state, and the model answers noun-retrieval questions from
        // its training prior instead of from context. Native FP8 keeps the
        // checkpoint's own per-block scales end to end.
        //
        // The old "FP8 direct load causes CUDA 700 / mmap pointers may be
        // invalidated" TODO here was wrong: `WeightStore` holds *device*
        // pointers (the default fast loader uses O_DIRECT and never mmaps), and
        // the NVFP4 arm below has always aliased those pointers for the process
        // lifetime. The two mechanisms that genuinely fault are (1) the NULL
        // `QuantizedWeight`s `load_nemotron_ssm` returns on the FP8 arm being
        // dereferenced by the prefill weight copies, and (2) the checkpoint's
        // 4-byte scalar `weight_scale` being indexed as a [N/128, K/128] matrix
        // by w8a16. Both are closed below: the prefill copies are hard-gated off
        // under native, and `load_fp8_block_scaled_as_fp8weight` materializes a
        // real block-scale buffer. `ATLAS_NEMOTRON_NATIVE_FP8_SSM=0` restores
        // the legacy path exactly (same-binary A/B).
        //
        // DEFAULT OFF — this path is a MEASURED REGRESSION, not yet a fix.
        // Enabled on Puzzle-75B (2026-07-27, `native_fp8=true` confirmed in the
        // load log) it made output strictly worse than the double-quant it
        // replaces: "Alice is 30. Bob is 25. Who is older?" went from a correct
        // "Alice" to inventing a Carol/Charlie never present in the prompt;
        // "my dog is named Rufus" degenerated into a `The user says: "My".`
        // repeat loop; a 3-item colour question returned empty. Arithmetic
        // (17*23=391) still passed, so the model loads and runs — the numerics
        // of this path are wrong somewhere between the scale materialisation and
        // the w8a16 dispatch, not the wiring.
        //
        // Kept, default-off, because the wiring is the valuable part and was
        // non-obvious: `set_fp8_weights` (layers/nemotron_mamba2.rs) and the
        // decode `w8a16_gemv` preference (trait_impl.rs) were already written
        // and had ZERO callers — a finished, dead path. All three w8a16 kernels
        // are already compiled into the Puzzle target; no CUDA authoring is
        // needed to revive it. Suspects for the bad numerics, in order: the
        // scalar-`weight_scale` broadcast into a [ceil(N/128), ceil(K/128)]
        // block matrix, and the k_blocks floor-vs-ceil disagreement between
        // w8a16_gemv.cu:129 `(K+127)/128` and w8a16_gemm{,_pipelined}.cu
        // `K/128` (benign at K=4096/8192, but it shows the two kernels do not
        // share one scale-layout contract).
        //
        // THREE modes, so decode and prefill can be bisected independently (the
        // native path changed BOTH at once — `w8a16_gemv` for decode and
        // `w8a16_gemm[_pipelined]` for prefill — so a single on/off gate cannot
        // say which one corrupts):
        //   unset / "0"  — legacy FP8→BF16→NVFP4 everywhere (baseline)
        //   "decode"     — native FP8 weights installed AND used by decode, but
        //                  the legacy NVFP4 requant + prefill weight copies are
        //                  still built and prefill keeps using them
        //   "1" / "both" — native FP8 for decode and prefill (no NVFP4 copies)
        let mode = std::env::var("ATLAS_NEMOTRON_NATIVE_FP8_SSM").unwrap_or_default();
        let native_fp8_enabled = matches!(mode.as_str(), "1" | "both" | "decode");
        // Decode-only mode keeps the legacy NVFP4 weights resident for prefill.
        let native_prefill_wanted = matches!(mode.as_str(), "1" | "both");
        // Probe the kernels BEFORE committing to the path: `w8a16_gemm_pipelined`
        // has silently resolved to handle 0 in a shipped binary before (that is
        // why `kernel_audit` exists), and a 0 handle must degrade to the legacy
        // path rather than fault at the first token. Both projections must be
        // FP8 — a half-native layer would leave one NULL `QuantizedWeight` live.
        let out_is_fp8 = store.contains(&format!("{p}.out_proj.weight_scale"));
        let gemv_k = crate::layers::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv");
        let gemm_pipe_k =
            crate::layers::try_kernel(gpu, "w8a16_gemm_pipelined", "w8a16_gemm_pipelined");
        let gemm_k = crate::layers::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm");
        let native_fp8 = native_fp8_enabled
            && quant_kind == NemotronSsmQuant::Fp8
            && out_is_fp8
            && gemv_k.0 != 0
            && (gemm_pipe_k.0 != 0 || gemm_k.0 != 0);
        if native_fp8_enabled && quant_kind == NemotronSsmQuant::Fp8 && !native_fp8 {
            static NATIVE_FALLBACK_WARN: std::sync::Once = std::sync::Once::new();
            NATIVE_FALLBACK_WARN.call_once(|| {
                tracing::warn!(
                    "L{i} SSM: native FP8 unavailable (out_proj_fp8={out_is_fp8} \
                     w8a16_gemv={} w8a16_gemm_pipelined={} w8a16_gemm={}) — falling back \
                     to the FP8→BF16→NVFP4 double-quant path",
                    gemv_k.0 != 0,
                    gemm_pipe_k.0 != 0,
                    gemm_k.0 != 0,
                );
            });
        }
        // Native FP8 for prefill only when the native path is live AND the mode
        // asked for it. In "decode" mode the NVFP4 copies below are built as
        // usual and prefill keeps using them.
        let native_prefill = native_fp8 && native_prefill_wanted;
        tracing::info!(
            "L{i} SSM quant={quant_kind:?} native_fp8={native_fp8} native_prefill={native_prefill} \
             in_proj_size={} d_inner={} h={h}",
            config.mamba2_in_proj_size(),
            config.mamba2_d_inner(),
        );
        let native = if native_fp8 {
            let mut in_fp8 = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.in_proj"), gpu)?;
            let mut out_fp8 =
                load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.out_proj"), gpu)?;
            // OWN the FP8 weight bytes. `load_fp8_block_scaled_as_fp8weight`
            // ALIASES the `WeightStore` pointer (loaders_fp8.rs `let weight_ptr =
            // w.ptr`), which is fine for a loader that consumes the bytes during
            // load — the legacy arm below dequantizes them into a fresh NVFP4
            // buffer and never looks again. This arm is different: it keeps the
            // pointer and dereferences it on every token, for the process
            // lifetime. That is the real content of the old "FP8 direct load
            // causes CUDA 700 / mmap pointers may be invalidated" TODO.
            //
            // Measured: with the alias, `w8a16_gemv` is fed stale bytes and the
            // model stays fluent while losing its prompt ("Alice is 30. Bob is
            // 25. Who is older?" -> "Charlie is the oldest."). The kernel itself
            // is NOT at fault — `examples/w8a16_gemv_numeric` checks it at this
            // exact shape (N=18048, K=4096) under both a constant and a varying
            // per-block scale and it is correct to BF16 tolerance.
            //
            // Copying costs ~108 MB per SSM layer (in_proj 74 MB + out_proj
            // 34 MB), ~2.2 GB over the 20 SSM layers — well under the ~6.4 GB of
            // derived prefill copies this path no longer builds.
            for w in [&mut in_fp8, &mut out_fp8] {
                let bytes = (w.n as usize) * (w.k as usize);
                let owned = gpu.alloc(bytes)?;
                gpu.copy_d2d(w.weight, owned, bytes)?;
                w.weight = owned;
            }
            if i == 0 {
                // Prove the bytes the kernel will read ARE the checkpoint's. The
                // numeric test covers the kernel with synthetic weights, so this
                // is the one link it cannot check.
                let mut head = [0u8; 16];
                gpu.copy_d2h(in_fp8.weight, &mut head)?;
                tracing::info!(
                    "L0 in_proj FP8 head: {}",
                    head.iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            // Shape + alignment contract. `w8a16_gemv.cu` computes K16 = K/16 with
            // no tail and issues 16-byte uint4 loads of `B + n*K`, so a K that is
            // not a multiple of 16 reads garbage silently — fail loudly at load
            // instead (Puzzle is K=4096 / K=8192, both fine).
            ensure!(
                in_fp8.n as usize == config.mamba2_in_proj_size() && in_fp8.k as usize == h,
                "L{i} SSM in_proj FP8 shape [{},{}] != [{},{h}]",
                in_fp8.n,
                in_fp8.k,
                config.mamba2_in_proj_size(),
            );
            ensure!(
                out_fp8.n as usize == h && out_fp8.k as usize == config.mamba2_d_inner(),
                "L{i} SSM out_proj FP8 shape [{},{}] != [{h},{}]",
                out_fp8.n,
                out_fp8.k,
                config.mamba2_d_inner(),
            );
            ensure!(
                in_fp8.k % 16 == 0 && out_fp8.k % 16 == 0,
                "L{i} SSM: w8a16 requires K%16==0, got in_proj K={} out_proj K={}",
                in_fp8.k,
                out_fp8.k,
            );
            Some((in_fp8, out_fp8))
        } else {
            None
        };
        // Legacy FP8→BF16→NVFP4 requant. Runs whenever prefill is NOT native
        // (gate unset, or `decode` mode): the prefill arms and every derived
        // weight copy below read `ssm.in_proj`/`ssm.out_proj`, which
        // `load_nemotron_ssm` returns as `QuantizedWeight::null()` on the FP8
        // arm, so they must be materialized here or prefill derefs NULL.
        if !native_prefill && quant_kind != NemotronSsmQuant::Nvfp4 {
            let in_proj_dense = if quant_kind == NemotronSsmQuant::Fp8 {
                dequant_fp8_to_bf16_into(store, &format!("{p}.in_proj"), gpu, scratch)?
            } else {
                dense(store, &format!("{p}.in_proj.weight"))?
            };
            ssm.in_proj = quantize_to_nvfp4(
                &in_proj_dense,
                config.mamba2_in_proj_size(),
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let out_proj_dense = if out_is_fp8 {
                dequant_fp8_to_bf16_into(store, &format!("{p}.out_proj"), gpu, scratch)?
            } else {
                dense(store, &format!("{p}.out_proj.weight"))?
            };
            ssm.out_proj = quantize_to_nvfp4(
                &out_proj_dense,
                h,
                config.mamba2_d_inner(),
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
        }
        // Transposed NVFP4 copies of the two SSM projections. These
        // switch prefill from the base `w4a16_gemm` (M64/N64/K16, no
        // pipelining) to `w4a16_gemm_t` (N128/K32, FP8 MMA, 2-stage
        // cp.async) — see NemotronMamba2Layer::set_prefill_weights.
        // The 40 SSM layers are ~46% of prefill time on Puzzle, and
        // without this the fast kernel is compiled but unreachable.
        // Cost: ~2.1 GB extra weights. ATLAS_NO_SSM_PREFILL_T=1 keeps
        // the base GEMM (same-binary A/B + escape hatch).
        //
        // Two mutually exclusive prefill weight representations:
        //
        //   FP8  (default) — pre-dequantized E4M3 [N, K], consumed by
        //     `fp8_gemm_t`, which has NO dequant phase. The NVFP4
        //     path re-derives its B tile from FP4 on every K step of
        //     every M-block (cost N*K*(M/M_TILE), i.e. 8x over at 1k
        //     tokens); ablating that ALU alone cut a 1k prefill from
        //     557 ms to 424 ms, so it is worth removing outright.
        //     Cost: ~4.3 GB (vs ~2.1 GB for the transposed copies).
        //
        //   Transposed NVFP4 — `w4a16_gemm_t`/`_m128`. Kept as the
        //     escape hatch via ATLAS_NO_SSM_FP8_PREFILL=1.
        //
        // NVFP4 stays resident either way: decode uses w4a16_gemv.
        //
        // Both copies are derived FROM `ssm.in_proj`/`ssm.out_proj`, which are
        // `QuantizedWeight::null()` on the native-FP8 arm — `transpose_for_gemm`
        // would `copy_d2h` from NULL and `predequant_to_fp8` would launch on a
        // NULL B. Native prefill goes through w8a16_gemm instead, so neither
        // copy is built (and ~6.4 GB of derived weights is not allocated).
        let fp8_prefill = !native_prefill && std::env::var("ATLAS_NO_SSM_FP8_PREFILL").is_err();
        let prefill_t = !native_prefill
            && !fp8_prefill
            && std::env::var("ATLAS_NO_SSM_PREFILL_T").is_err();
        let proj_t = if prefill_t {
            let in_t = ssm
                .in_proj
                .transpose_for_gemm(gpu, config.mamba2_in_proj_size(), h)?;
            let out_t = ssm
                .out_proj
                .transpose_for_gemm(gpu, h, config.mamba2_d_inner())?;
            Some((in_t, out_t))
        } else {
            None
        };
        let proj_fp8 = if fp8_prefill {
            let pdq_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
            let in_fp8 = ssm.in_proj.predequant_to_fp8(
                gpu,
                pdq_k,
                config.mamba2_in_proj_size(),
                h,
                stream,
            )?;
            let out_fp8 =
                ssm.out_proj
                    .predequant_to_fp8(gpu, pdq_k, h, config.mamba2_d_inner(), stream)?;
            Some((in_fp8, out_fp8))
        } else {
            None
        };
        let mut layer = NemotronMamba2Layer::new(norm, ssm, config, gpu, i)?;
        if let Some((in_fp8, out_fp8)) = native {
            layer.set_fp8_weights(Some(in_fp8), Some(out_fp8), native_prefill)?;
        }
        if let Some((in_t, out_t)) = proj_t {
            layer.set_prefill_weights(Some(in_t), Some(out_t));
        }
        if let Some((in_fp8, out_fp8)) = proj_fp8 {
            layer.set_fp8_prefill_weights(in_fp8, out_fp8);
        }
        Ok(layer)
    }
}
