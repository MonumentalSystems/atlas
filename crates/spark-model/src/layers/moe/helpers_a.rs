// SPDX-License-Identifier: AGPL-3.0-only

//! Setters + transposes + transpose_for_prefill_unified_inner.

use super::*;

impl MoeLayer {
    /// Transpose MoE weights for coalesced prefill GEMM reads.
    ///
    /// Transposes per-expert routed weights [N, K/2] → [K/2, N] to enable
    /// the cp.async pipelined FP8-MMA K64 kernels. This doubles expert
    /// memory (~17 GB for 35B, ~30 GB for 122B) but eliminates the
    /// catastrophic uncoalesced B reads in the fallback grouped GEMM,
    /// cutting MoE prefill time by ~2x.
    /// Set pre-expert norm (Gemma-4 26B: pre_feedforward_layernorm_2).
    /// Applied to input AFTER routing but BEFORE expert dispatch.
    pub fn set_pre_expert_norm(&mut self, norm: crate::weight_map::DenseWeight) {
        self.pre_expert_norm = Some(norm);
    }

    /// Set GeGLU activation for MoE experts (Gemma-4 26B).
    /// Replaces SiLU with GELU in the sorted/unfused path and forces decode
    /// to use the sorted path (avoiding fused SiLU kernels).
    pub fn set_gelu_activation(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.moe_act_mul = gpu.kernel("gelu", "gelu_mul")?;
        self.gelu_activation = true;
        Ok(())
    }

    pub fn transpose_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, true)
    }

    /// Transpose only the gate+up routed weights, leaving the down projection
    /// in its original layout. Cuts the transpose memory cost from ~3×
    /// (gate+up+down) to ~2× per expert. Used by MiniMax M2.7-NVFP4 EP=2
    /// when the full transpose doesn't fit but gate+up does — the fused
    /// `moe_w4a16_fused_gate_up_k64_n128` kernel still runs (capturing the
    /// dominant gate+up bandwidth savings), while down stays on the
    /// uncoalesced grouped-GEMM path.
    pub fn transpose_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, false)
    }

    pub(super) fn transpose_for_prefill_impl(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        include_down: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;

        // Transpose per-expert routed weights for coalesced prefill GEMM reads.
        let num_experts = self.weights.experts.len();
        let mut gate_t = Vec::with_capacity(num_experts);
        let mut up_t = Vec::with_capacity(num_experts);
        let mut down_t = Vec::with_capacity(num_experts);

        for expert in &self.weights.experts {
            if expert.gate_proj.is_null() {
                gate_t.push(QuantizedWeight::null());
                up_t.push(QuantizedWeight::null());
                if include_down {
                    down_t.push(QuantizedWeight::null());
                }
            } else {
                gate_t.push(expert.gate_proj.transpose_for_gemm(gpu, inter, h)?);
                up_t.push(expert.up_proj.transpose_for_gemm(gpu, inter, h)?);
                if include_down {
                    down_t.push(expert.down_proj.transpose_for_gemm(gpu, h, inter)?);
                }
            }
        }

        self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
        self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        if include_down {
            self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        }

        // Transpose shared expert weights (tiny: ~5 MB per layer).
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            if include_down {
                self.shared_down_t =
                    Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                        gpu,
                        h,
                        shared_inter,
                    )?);
            }
        }

        Ok(())
    }

    /// Phase 8a unified-layout transpose pass: build persistent transposed
    /// gate/up/down for all experts, freeing the untransposed copies between
    /// phases so the entire pass fits in tight memory budgets that the
    /// non-unified `transpose_for_prefill_impl(true)` would reject.
    ///
    /// Phased flow (memory math for MiniMax M2.7-NVFP4 EP=2 ≈ 47 GB free):
    ///   A. Transpose gate+up               (allocs +39 GB; free ≈ 8 GB)
    ///   B. Free gate+up untransposed       (frees 39 GB; free ≈ 47 GB)
    ///   C. Transpose down                  (allocs +20 GB; free ≈ 27 GB)
    ///   D. Free down untransposed          (frees 20 GB; free ≈ 47 GB)
    ///
    /// Net memory: same as starting point, but layout is now unified
    /// (transposed-only) — the `[N, K/2]` decode kernels can no longer
    /// run; dispatch must use the `_t` decode kernels (which do).
    ///
    /// Caller responsibilities:
    ///   1. Set `ATLAS_UNIFIED_MOE_LAYOUT=1` so `MoeLayer::use_t_layout_for_decode()`
    ///      returns true at dispatch time.
    ///   2. Call this method INSTEAD of `transpose_for_prefill` /
    ///      `transpose_gate_up_for_prefill`.
    pub fn transpose_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, false)
    }

    /// Hybrid-layout transpose pass — analogue of `transpose_for_prefill_unified`
    /// that **keeps** the untransposed originals so decode + MTP verify dispatch
    /// can continue using the warp-reduction kernels. Allocates ~58 GB
    /// transposed alongside the existing ~58 GB originals on MiniMax M2.7-NVFP4
    /// EP=2; fits in 122 GB GB10 with KV-cache headroom up to ~32K context.
    /// Caller is responsible for memory-fit gating (factory checks free memory
    /// before invoking this).
    pub fn transpose_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, true)
    }

    /// Phased build of the transposed weight set. When `keep_originals` is true
    /// (hybrid-layout mode), Phase B and Phase D frees are skipped so decode
    /// paths still find the untransposed weights. When false (unified-layout
    /// mode), the originals are freed between phases — current Phase 8a
    /// behavior.
    pub(super) fn transpose_for_prefill_unified_inner(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        keep_originals: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;
        let num_experts = self.weights.experts.len();

        // ── Phase A: transpose gate+up routed experts ──
        let mut gate_t = Vec::with_capacity(num_experts);
        let mut up_t = Vec::with_capacity(num_experts);
        for expert in &self.weights.experts {
            if expert.gate_proj.is_null() {
                gate_t.push(QuantizedWeight::null());
                up_t.push(QuantizedWeight::null());
            } else {
                gate_t.push(expert.gate_proj.transpose_for_gemm(gpu, inter, h)?);
                up_t.push(expert.up_proj.transpose_for_gemm(gpu, inter, h)?);
            }
        }
        self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
        self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        // Shared expert (tiny, do unconditionally — fits regardless).
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
        }

        if !keep_originals {
            // ── Phase B: free gate+up untransposed ──
            // The previous gate_ptrs / up_ptrs device-side pointer tables now
            // contain stale addresses, but the unified dispatch never reads
            // them (gated by `use_t_layout_for_decode()`).
            for expert in &mut self.weights.experts {
                if !expert.gate_proj.weight.is_null() {
                    gpu.free(expert.gate_proj.weight)?;
                    gpu.free(expert.gate_proj.weight_scale)?;
                    expert.gate_proj.weight = DevicePtr::NULL;
                    expert.gate_proj.weight_scale = DevicePtr::NULL;
                }
                if !expert.up_proj.weight.is_null() {
                    gpu.free(expert.up_proj.weight)?;
                    gpu.free(expert.up_proj.weight_scale)?;
                    expert.up_proj.weight = DevicePtr::NULL;
                    expert.up_proj.weight_scale = DevicePtr::NULL;
                }
            }
            if !self.weights.shared_expert.gate_proj.weight.is_null() && shared_inter > 0 {
                gpu.free(self.weights.shared_expert.gate_proj.weight)?;
                gpu.free(self.weights.shared_expert.gate_proj.weight_scale)?;
                self.weights.shared_expert.gate_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.gate_proj.weight_scale = DevicePtr::NULL;
                gpu.free(self.weights.shared_expert.up_proj.weight)?;
                gpu.free(self.weights.shared_expert.up_proj.weight_scale)?;
                self.weights.shared_expert.up_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.up_proj.weight_scale = DevicePtr::NULL;
            }
        }

        // ── Phase C: transpose down routed experts ──
        let mut down_t = Vec::with_capacity(num_experts);
        for expert in &self.weights.experts {
            if expert.down_proj.is_null() {
                down_t.push(QuantizedWeight::null());
            } else {
                down_t.push(expert.down_proj.transpose_for_gemm(gpu, h, inter)?);
            }
        }
        self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        if !self.weights.shared_expert.down_proj.is_null() && shared_inter > 0 {
            self.shared_down_t = Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                gpu,
                h,
                shared_inter,
            )?);
        }

        if !keep_originals {
            // ── Phase D: free down untransposed ──
            for expert in &mut self.weights.experts {
                if !expert.down_proj.weight.is_null() {
                    gpu.free(expert.down_proj.weight)?;
                    gpu.free(expert.down_proj.weight_scale)?;
                    expert.down_proj.weight = DevicePtr::NULL;
                    expert.down_proj.weight_scale = DevicePtr::NULL;
                }
            }
            if !self.weights.shared_expert.down_proj.weight.is_null() && shared_inter > 0 {
                gpu.free(self.weights.shared_expert.down_proj.weight)?;
                gpu.free(self.weights.shared_expert.down_proj.weight_scale)?;
                self.weights.shared_expert.down_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.down_proj.weight_scale = DevicePtr::NULL;
            }
        }

        Ok(())
    }

    /// Build the per-expert NVFP4 gate_up tables for the FP4 prefill path
    /// (`ATLAS_HOLO_MOE_GATEUP_FP4`). For each non-null expert, dequant the
    /// stored NVFP4 `gate_proj`/`up_proj` (`[N=inter, K=h]`) to BF16, then
    /// re-pack via `pack_bf16_weight_to_nvfp4_t` into the CUTLASS escape-hatch
    /// layout (`[N,K/2]` packed + `[K/16,N]` E4M3 scale, scale2 = 1.0).
    ///
    /// Additive: only invoked when the env flag is on. Leaves all existing
    /// weight tables untouched so the FP8 path stays bit-identical when off.
    pub fn build_fp4_gate_up(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let h = config.hidden_size; // K
        let inter = config.moe_intermediate_size; // N (per gate/up proj)
        let n = inter;
        let k = h;
        let packed_len = (k / 2) * n; // [K/2, N] bytes (== [N,K/2] elems packed)
        let scale_len = (k / 16) * n; // [K/16, N] E4M3 bytes

        let mut t = MoeFp4GateUp {
            gate_packed_ptrs: Vec::new(),
            gate_scale_ptrs: Vec::new(),
            gate_scale2_vals: Vec::new(),
            up_packed_ptrs: Vec::new(),
            up_scale_ptrs: Vec::new(),
            up_scale2_vals: Vec::new(),
            _owned: Vec::new(),
        };

        // Pack one NVFP4 expert projection: dequant -> BF16 -> CUTLASS NVFP4.
        // Returns (packed_ptr, scale_ptr); both are tracked in `_owned`.
        let mut pack_one = |qw: &QuantizedWeight| -> Result<(u64, u64)> {
            let bf16 = dequant_nvfp4_qw_to_bf16(gpu, qw, n, k)?;
            let packed = gpu.alloc(packed_len)?;
            let scale = gpu.alloc(scale_len)?;
            spark_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
                bf16.0,
                packed.0,
                scale.0,
                n as u32,
                k as u32,
                stream,
            )?;
            gpu.synchronize(stream)?;
            gpu.free(bf16)?; // BF16 staging buffer no longer needed
            t._owned.push(packed);
            t._owned.push(scale);
            Ok((packed.0, scale.0))
        };

        for expert in &self.weights.experts {
            if expert.gate_proj.is_null() {
                // Remote/placeholder expert: zero pointers (escape-hatch skips
                // experts with empty token ranges, so these are never read).
                t.gate_packed_ptrs.push(0);
                t.gate_scale_ptrs.push(0);
                t.gate_scale2_vals.push(1.0);
                t.up_packed_ptrs.push(0);
                t.up_scale_ptrs.push(0);
                t.up_scale2_vals.push(1.0);
                continue;
            }
            let (gp, gs) = pack_one(&expert.gate_proj)?;
            t.gate_packed_ptrs.push(gp);
            t.gate_scale_ptrs.push(gs);
            t.gate_scale2_vals.push(1.0);
            let (up, us) = pack_one(&expert.up_proj)?;
            t.up_packed_ptrs.push(up);
            t.up_scale_ptrs.push(us);
            t.up_scale2_vals.push(1.0);
        }

        tracing::info!(
            "FP4 MoE gate_up: packed {} experts (N={n} K={k}) -> CUTLASS NVFP4 layout",
            t.gate_packed_ptrs.len(),
        );
        self.fp4_gate_up = Some(t);
        Ok(())
    }
}

/// Host dequant of an NVFP4 `QuantizedWeight` (`[N,K/2]` packed + `[N,K/16]`
/// E4M3 block scales + f32 `weight_scale_2`) to a fresh BF16 `[N,K]` GPU
/// buffer. Mirrors `weight_map::dequant_nvfp4_to_bf16`'s modelopt math
/// (`val = e2m1[nibble] * fp8_e4m3(group_scale) * weight_scale_2`) but reads
/// from already-loaded device pointers rather than the weight store.
fn dequant_nvfp4_qw_to_bf16(
    gpu: &dyn GpuBackend,
    qw: &QuantizedWeight,
    n: usize,
    k: usize,
) -> Result<DevicePtr> {
    let total = n * k;
    let packed_bytes = total / 2;
    let num_groups = total / 16;

    let mut packed = vec![0u8; packed_bytes];
    let mut scales = vec![0u8; num_groups];
    gpu.copy_d2h(qw.weight, &mut packed)?;
    gpu.copy_d2h(qw.weight_scale, &mut scales)?;
    let global_scale = qw.weight_scale_2;

    // E2M1 nibble -> float (sign|exp2|mant1).
    const E2M1: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    let mut bf16_out = vec![0u16; total];
    for group in 0..num_groups {
        let block_scale = e4m3_byte_to_f32(scales[group]);
        let combined = block_scale * global_scale;
        for elem in 0..16 {
            let flat = group * 16 + elem;
            let byte = packed[flat / 2];
            let nibble = if flat % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
            bf16_out[flat] = f32_to_bf16_bits(E2M1[nibble as usize] * combined);
        }
    }

    let buf = gpu.alloc(total * 2)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(bf16_out.as_ptr() as *const u8, total * 2) };
    gpu.copy_h2d(bytes, buf)?;
    Ok(buf)
}

/// OCP FP8 E4M3 (1-4-3, bias 7) byte -> f32. NaN (0x7F/0xFF) -> 0.
fn e4m3_byte_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        0.0
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

/// f32 -> BF16 bits with round-to-nearest-even (matches weight_map::f32_to_bf16).
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
