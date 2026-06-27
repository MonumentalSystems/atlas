// SPDX-License-Identifier: AGPL-3.0-only

//! Dense SwiGLU FFN component for non-MoE models.
//!
//! Forward: gate = gate_proj(x), up = up_proj(x), out = down_proj(SiLU(gate) * up)
//! 2 fused kernel launches per decode token (dual GEMV + SiLU-fused down GEMV).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};

pub struct DenseFfnWeights {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    /// Transposed ([K/2, N]) copies for the fast `w4a16_gemm_t_m128` prefill
    /// kernel. `None` → prefill falls back to the slow M64xN64 base kernel.
    /// The non-transposed copies above are kept for the decode gemv path.
    pub gate_proj_t: Option<QuantizedWeight>,
    pub up_proj_t: Option<QuantizedWeight>,
    pub down_proj_t: Option<QuantizedWeight>,
}

/// BF16 dense MLP weights — alternative to NVFP4 for precision-sensitive
/// models (Gemma-4-31B). Each is `[N, K]` row-major BF16. When installed
/// on a `DenseFfnLayer` via `set_bf16_weights`, the forward paths
/// dispatch to `dense_gemv_bf16` / `dense_gemm_bf16` instead of the
/// w4a16 NVFP4 kernels. Costs ~3.4 GB extra GPU memory on Gemma-4-31B
/// (3 × hidden×intermediate × 2 bytes) vs NVFP4's 0.5 bytes/weight.
pub struct DenseFfnWeightsBf16 {
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// Activation function for gated FFN (SiLU for Qwen/Llama, GELU for Gemma-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    SiLU,
    GeLU,
}

/// A per-projection int8 W4A8 weight, built lazily from the NVFP4 weight on the
/// first `ATLAS_INT8_PREFILL` prefill (see `DenseFfnLayer::ensure_int8_weight`).
/// `w_i8` is `[N, K]` signed int8; `w_scale` is `[N, K/32]` F32. Cached for the
/// process lifetime in a `OnceLock`, so the requant kernel runs once per weight.
#[derive(Debug, Clone, Copy)]
struct Int8Weight {
    w_i8: DevicePtr,
    w_scale: DevicePtr,
}

/// Caller-owned activation-requant scratch for the int8 prefill path. Sized to
/// the largest `(M, K)` seen so far (`M*K` int8 bytes + `M*(K/32)*4` F32 bytes)
/// and reused across calls. Grown (with a stream sync before freeing the old
/// buffers) only when a larger prefill arrives.
#[derive(Debug, Clone, Copy)]
struct Int8Scratch {
    a_i8: DevicePtr,
    a_scale: DevicePtr,
    i8_bytes: usize,
    scale_bytes: usize,
}

pub struct DenseFfnLayer {
    pub weights: DenseFfnWeights,
    activation: FfnActivation,
    w4a16_gemv: KernelHandle,
    w4a16_gemv_dual: KernelHandle,
    w4a16_gemv_silu_input: KernelHandle,
    // LOSSLESS single-warp-per-output decode variants (8 outputs/block, no smem
    // cross-warp reduce). Bit-identical to the 64-thread kernels (proven by the
    // w4a16_gemv_sw microtest). Opt-in via ATLAS_DECODE_OPT (default off →
    // dispatch unchanged). KernelHandle(0) on miss → fall back to base kernels.
    w4a16_gemv_dual_sw: KernelHandle,
    w4a16_gemv_silu_input_sw: KernelHandle,
    decode_opt: bool,
    w4a16_gemv_dual_batch2: KernelHandle,
    w4a16_gemv_dual_batch3: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    w4a16_gemm: KernelHandle,
    // 128x128 2-stage cp.async pipelined w4a16 GEMM — the fast prefill kernel
    // attention/SSM already use. The base `w4a16_gemm` (M64xN64) only hits
    // ~10 TFLOPS at M=8k and was the flat ~155 tok/s dense-FFN prefill
    // bottleneck on Qwen3.6-27B. KernelHandle(0) on miss → scalar-tile fallback.
    w4a16_gemm_t_m128_k: KernelHandle,
    // v2: 8-warp (256-thread) variant of t_m128 — parallel chunk MMAs, 3 CTAs/SM.
    // Preferred over t_m128 for dense-FFN prefill when present. KernelHandle(0) → use t_m128.
    w4a16_gemm_t_m128_v2_k: KernelHandle,
    // LOSSLESS BF16 variant of t_m128: same 128x128 cp.async tiling, but FP4→BF16
    // dequant + BF16 m16n8k16 MMA (FP32 accum) instead of the FP8-E4M3 crush the
    // default NVIDIA t_m128 uses. The FP8 path perturbs generation (measured
    // length-truncations / accuracy risk on Qwen3.6-27B); this kernel keeps prefill
    // outputs bit-for-bit vs the base `w4a16_gemm`. OPT-IN only, gated by
    // ATLAS_BF16_TC_PREFILL (default off → dispatch unchanged). KernelHandle(0) on miss.
    w4a16_gemm_t_m128_bf16_k: KernelHandle,
    // v2 of the LOSSLESS BF16 128x128 prefill kernel: same MMA instruction order
    // (so BIT-IDENTICAL to bf16_k, proven by w4a16_bf16_v2_microtest) but a
    // smaller A-tile smem pad lifts occupancy from 2→3 CTAs/SM (~+50% resident
    // warps), giving a measured ~3-8% faster prefill GEMM on this latency-bound
    // kernel. Preferred over bf16_k when present. KernelHandle(0) on miss → bf16_k.
    w4a16_gemm_t_m128_bf16_v2_k: KernelHandle,
    // FP8 M64 prefill (w4a16_gemm_t): m16n8k32 e4m3 MMA + M_TILE=64. Packed 1-byte
    // operands cut shared-memory load instructions ~4x (the v2 BF16 path is
    // smem-bandwidth-bound, L1/TEX 90% per ncu), and M64's lower register pressure
    // lifts occupancy → measured ~44 TFLOP/s vs ~30 for v2 (~1.47x prefill) on dgx1.
    // LOSSY (FP8 E4M3, cosine ~0.9997) — OPT-IN via ATLAS_FP8_M64_PREFILL, gated on
    // quality. KernelHandle(0) on miss → dispatch unchanged.
    w4a16_gemm_t_k: KernelHandle,
    // int8 W4A8 prefill (ATLAS_INT8_PREFILL): the validated requant→faith2
    // pipeline (cosine 0.999978). `int8_gemm_faith2` is an int8×int8 MMA with
    // per-32 block scales, so BOTH operands must be int8 — unlike the FP8 path
    // (mixed BF16×FP8). At first int8 prefill we requant the NVFP4 gate/up/down
    // weights to int8 once (`requant_w_nvfp4_int8`, cached in the OnceLocks
    // below) and requant the BF16 activations every call (`requant_a_bf16_int8`,
    // into `int8_a_scratch`). KernelHandle(0) on miss → arm never taken.
    int8_faith2_k: KernelHandle,
    requant_w_int8_k: KernelHandle,
    requant_a_int8_k: KernelHandle,
    // Lazily-built, process-lifetime int8 weight copies (one per projection),
    // requanted from `self.weights.{gate,up,down}_proj`. Only ever touched when
    // ATLAS_INT8_PREFILL is set → default-off path is byte-identical.
    int8_gate: std::sync::OnceLock<Int8Weight>,
    int8_up: std::sync::OnceLock<Int8Weight>,
    int8_down: std::sync::OnceLock<Int8Weight>,
    // Grow-on-demand activation requant scratch, shared by all three int8 GEMMs.
    int8_a_scratch: std::sync::Mutex<Option<Int8Scratch>>,
    /// SiLU(gate)*up or GELU(gate)*up depending on activation.
    act_mul: KernelHandle,
    /// BF16 dense MLP weights — when `Some`, all forward paths use the
    /// `dense_gemv_bf16` / `dense_gemm_bf16` kernels instead of w4a16
    /// NVFP4. Falls back to the NVFP4 weights when `None`. Set via
    /// `set_bf16_weights`. Used by Gemma-4 dense to avoid the structural
    /// NVFP4 attention drift on greedy code generation (the fib test's
    /// broken-indentation pattern).
    bf16_weights: Option<DenseFfnWeightsBf16>,
    dense_gemv_bf16_k: KernelHandle,
    dense_gemm_bf16_k: KernelHandle,
    // Tensor-core BF16 GEMM (m16n8k16 MMA) for the dense-FFN PREFILL path.
    // The scalar `dense_gemm_bf16` is ~10x too slow on long prefills (it was
    // the flat ~155 tok/s prefill bottleneck on Qwen3.6-27B dense NVFP4).
    // KernelHandle(0) on miss → forward_prefill falls back to the scalar path.
    // Decode (gemv, M=1) is untouched, so TPOT is unaffected.
    dense_gemm_tc_k: KernelHandle,
}

impl DenseFfnLayer {
    pub fn new(weights: DenseFfnWeights, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::new_with_activation(weights, FfnActivation::SiLU, gpu)
    }

    pub fn new_with_activation(
        weights: DenseFfnWeights,
        activation: FfnActivation,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let act_mul = match activation {
            FfnActivation::SiLU => gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            FfnActivation::GeLU => gpu.kernel("gelu", "gelu_mul")?,
        };
        // BF16 path kernels — optional (only loaded if available; gemma4
        // is the only consumer today). `try_kernel` returns
        // `KernelHandle(0)` on miss so we don't break NVFP4-only models
        // that were built without these kernels. Module names per
        // `kernels/gb10/{target}/nvfp4/KERNEL.toml`:
        //   `dense_gemv_bf16 = "gemv"`, `dense_gemm_bf16 = "gemm"`.
        let dense_gemv_bf16_k = super::try_kernel(gpu, "gemv", "dense_gemv_bf16");
        let dense_gemm_bf16_k = super::try_kernel(gpu, "gemm", "dense_gemm_bf16");
        let dense_gemm_tc_k = super::try_kernel(gpu, "gemm_tc", "dense_gemm_tc");

        Ok(Self {
            weights,
            activation,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_dual: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            w4a16_gemv_silu_input: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input")?,
            w4a16_gemv_dual_sw: super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_sw"),
            w4a16_gemv_silu_input_sw: super::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_silu_input_sw",
            ),
            decode_opt: std::env::var_os("ATLAS_DECODE_OPT").is_some(),
            w4a16_gemv_dual_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_dual_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_m128_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            w4a16_gemm_t_m128_v2_k: super::try_kernel(gpu, "w4a16_v2", "w4a16_gemm_t_m128_v2"),
            w4a16_gemm_t_m128_bf16_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128_bf16"),
            w4a16_gemm_t_m128_bf16_v2_k: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m128_bf16_v2",
            ),
            w4a16_gemm_t_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t"),
            int8_faith2_k: super::try_kernel(gpu, "w4a16", "int8_gemm_faith2"),
            requant_w_int8_k: super::try_kernel(gpu, "w4a16", "requant_w_nvfp4_int8"),
            requant_a_int8_k: super::try_kernel(gpu, "w4a16", "requant_a_bf16_int8"),
            int8_gate: std::sync::OnceLock::new(),
            int8_up: std::sync::OnceLock::new(),
            int8_down: std::sync::OnceLock::new(),
            int8_a_scratch: std::sync::Mutex::new(None),
            act_mul,
            bf16_weights: None,
            dense_gemv_bf16_k,
            dense_gemm_bf16_k,
            dense_gemm_tc_k,
        })
    }

    /// Install BF16 dense MLP weights. After this call, the forward paths
    /// dispatch to the BF16 GEMV/GEMM kernels instead of w4a16. The
    /// caller must ensure the BF16 kernels are loaded (see
    /// `dense_gemv_bf16_k` / `dense_gemm_bf16_k` checks). Spec-decode
    /// batched paths (`forward_k2`, `forward_k3`) are NOT supported on
    /// the BF16 path — Gemma-4 dense has no MTP so they're never called.
    pub fn set_bf16_weights(&mut self, gate: DenseWeight, up: DenseWeight, down: DenseWeight) {
        self.bf16_weights = Some(DenseFfnWeightsBf16 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
    }

    /// Ensure the int8 W4A8 copy of one NVFP4 projection weight exists, building
    /// it once via `requant_w_nvfp4_int8` and caching it in `cell`. Reads the
    /// NON-transposed NVFP4 layout (`weight` = packed E2M1 `[N, K/2]`,
    /// `weight_scale` = per-16 E4M3 `[N, K/16]`, `weight_scale_2` = per-tensor
    /// F32) — so it is independent of the `*_proj_t` transposed copies. The
    /// requant launches on `stream`; the subsequent faith2 read is stream-ordered
    /// after it, so no host sync is needed.
    fn ensure_int8_weight(
        &self,
        cell: &std::sync::OnceLock<Int8Weight>,
        gpu: &dyn GpuBackend,
        src: &QuantizedWeight,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<Int8Weight> {
        if let Some(w) = cell.get() {
            return Ok(*w);
        }
        let (nn, kk) = (n as usize, k as usize);
        let w_i8 = gpu.alloc(nn * kk)?; // [N, K] int8
        let w_scale = gpu.alloc(nn * (kk / 32) * 4)?; // [N, K/32] F32
        ops::requant_w_nvfp4_int8(
            gpu,
            self.requant_w_int8_k,
            src.weight,
            src.weight_scale,
            src.weight_scale_2,
            w_i8,
            w_scale,
            n,
            k,
            stream,
        )?;
        let built = Int8Weight { w_i8, w_scale };
        // Lost a race (another thread built first): free our duplicate buffers.
        if let Err(dup) = cell.set(built) {
            let _ = gpu.free(dup.w_i8);
            let _ = gpu.free(dup.w_scale);
        }
        Ok(*cell.get().expect("int8 weight cell set above"))
    }

    /// Ensure the shared activation-requant scratch holds at least `M*K` int8
    /// bytes + `M*(K/32)*4` F32 bytes, growing (and stream-syncing before freeing
    /// the old buffers) only when a larger prefill arrives. Returns
    /// `(a_i8, a_scale)` device pointers reused across all three int8 GEMMs.
    fn ensure_int8_scratch(
        &self,
        gpu: &dyn GpuBackend,
        m: u32,
        k: u32,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let need_i8 = (m as usize) * (k as usize);
        let need_scale = (m as usize) * ((k as usize) / 32) * 4;
        let mut guard = self
            .int8_a_scratch
            .lock()
            .expect("int8 scratch mutex poisoned");
        let grow = match guard.as_ref() {
            Some(s) => s.i8_bytes < need_i8 || s.scale_bytes < need_scale,
            None => true,
        };
        if grow {
            if let Some(old) = guard.take() {
                // Old buffers may still be referenced by in-flight kernels on
                // this stream; sync before freeing to avoid a use-after-free.
                gpu.synchronize(stream)?;
                let _ = gpu.free(old.a_i8);
                let _ = gpu.free(old.a_scale);
            }
            let a_i8 = gpu.alloc(need_i8.max(1))?;
            let a_scale = gpu.alloc(need_scale.max(1))?;
            *guard = Some(Int8Scratch {
                a_i8,
                a_scale,
                i8_bytes: need_i8,
                scale_bytes: need_scale,
            });
        }
        let s = guard.as_ref().expect("int8 scratch set above");
        Ok((s.a_i8, s.a_scale))
    }

    /// Single-token decode: 2-3 kernel launches depending on activation.
    /// SiLU: dual GEMV + SiLU-fused down GEMV (2 launches).
    /// GELU: dual GEMV + gelu_mul + down GEMV (3 launches, no fused GELU down kernel).
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // BF16 dispatch: per-projection GEMV via `dense_gemv_bf16`. We
        // don't have a fused dual-BF16-GEMV kernel today; two sequential
        // launches are still BF16-precision-correct and only ~10% slower
        // than the fused w4a16 path on Gemma-4-31B (the cost is dominated
        // by the bigger BF16 weight reads, not launch overhead).
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        // Fused gate_proj + up_proj: [1, H] → [1, inter] × 2.
        // Single-warp variant (lossless) when ATLAS_DECODE_OPT is on and the
        // _sw kernel is present; otherwise the proven 64-thread kernel.
        let use_sw = self.decode_opt
            && self.w4a16_gemv_dual_sw.0 != 0
            && self.w4a16_gemv_silu_input_sw.0 != 0;
        if use_sw {
            ops::w4a16_gemv_dual_sw(
                ctx.gpu,
                self.w4a16_gemv_dual_sw,
                input,
                &self.weights.gate_proj,
                gate_out,
                &self.weights.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_dual(
                ctx.gpu,
                self.w4a16_gemv_dual,
                input,
                &self.weights.gate_proj,
                gate_out,
                &self.weights.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
        }

        let output = ctx.buffers.moe_output();
        match self.activation {
            FfnActivation::SiLU => {
                // Fused SiLU(gate)*up + down_proj: [1, inter] → [1, H]
                if use_sw {
                    ops::w4a16_gemv_silu_input_sw(
                        ctx.gpu,
                        self.w4a16_gemv_silu_input_sw,
                        gate_out,
                        up_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemv_silu_input(
                        ctx.gpu,
                        self.w4a16_gemv_silu_input,
                        gate_out,
                        up_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                }
            }
            FfnActivation::GeLU => {
                // GELU(gate)*up → gate_out, then down_proj GEMV
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    inter,
                    stream,
                )?;
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv,
                    gate_out,
                    &self.weights.down_proj,
                    output,
                    h,
                    inter,
                    stream,
                )?;
            }
        }

        Ok(output)
    }

    /// K=2 speculative: batched GEMV for 2 tokens.
    /// 3 launches: dual batch2 (gate+up) + silu_mul + batch2 (down).
    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 2 tokens
        ops::w4a16_gemv_dual_batch2(
            ctx.gpu,
            self.w4a16_gemv_dual_batch2,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            2 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch2(
            ctx.gpu,
            self.w4a16_gemv_batch2,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// K=3 speculative: batched GEMV for 3 tokens.
    /// 3 launches: dual batch3 (gate+up) + silu_mul + batch3 (down).
    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 3 tokens
        ops::w4a16_gemv_dual_batch3(
            ctx.gpu,
            self.w4a16_gemv_dual_batch3,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            3 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch3(
            ctx.gpu,
            self.w4a16_gemv_batch3,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// N-token prefill: GEMM for all projections.
    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let m = num_tokens as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // BF16 prefill dispatch. Prefer the tensor-core m16n8k16 MMA kernel
        // (`dense_gemm_tc`, 3-5x+ over scalar) — the scalar `dense_gemm_bf16`
        // was the flat ~155 tok/s prefill bottleneck on Qwen3.6-27B dense
        // NVFP4 (FFN = ~83% of prefill). Falls back to scalar if the TC
        // kernel isn't loaded for this target. Decode (gemv, M=1) is a
        // separate path, so TPOT is unaffected; BF16 MMA preserves coherence.
        if let Some(ref bf16w) = self.bf16_weights {
            let tc = self.dense_gemm_tc_k.0 != 0;
            // helper: tensor-core GEMM when available, else scalar
            macro_rules! ffn_gemm {
                ($a:expr, $b:expr, $c:expr, $n:expr, $k:expr) => {
                    if tc {
                        ops::dense_gemm_tc(
                            ctx.gpu,
                            self.dense_gemm_tc_k,
                            $a,
                            $b,
                            $c,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    } else {
                        ops::dense_gemm(
                            ctx.gpu,
                            self.dense_gemm_bf16_k,
                            $a,
                            $b,
                            $c,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    }
                };
            }
            ffn_gemm!(input, &bf16w.gate_proj, gate_out, inter, h);
            ffn_gemm!(input, &bf16w.up_proj, up_out, inter, h);
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ffn_gemm!(gate_out, &bf16w.down_proj, output, h, inter);
            return Ok(());
        }

        // Prefill: prefer the 128x128 cp.async-pipelined `w4a16_gemm_t_m128`
        // (the kernel attention/SSM use) over the M64xN64 base `w4a16_gemm`
        // (~10 TFLOPS, the flat ~155 tok/s bottleneck). That kernel needs the
        // TRANSPOSED weight layout, so we use the `*_proj_t` copies built at
        // load (decode keeps the non-transposed weights via gemv → TPOT/
        // coherence unaffected). Falls back to base when no transposed copy /
        // kernel is present.
        // LOSSLESS prefill opt-in: when ATLAS_BF16_TC_PREFILL is set AND the
        // BF16 128x128 kernel is present, route prefill GEMMs through the
        // bit-equivalent BF16 tensor-core path instead of the default FP8-E4M3
        // `t_m128`. The FP8 crush is fast but perturbs generation (measured
        // length-truncations / accuracy risk on Qwen3.6-27B); the BF16 variant
        // keeps the same 128x128 cp.async speed at base-kernel precision.
        // Unset (default) → every arm below is byte-for-byte the prior behavior
        // (PCND: explicit opt-in, no silent default change). Read once per call.
        let bf16_tc_prefill = self.w4a16_gemm_t_m128_bf16_k.0 != 0
            && std::env::var_os("ATLAS_BF16_TC_PREFILL").is_some();
        // FP8 M64 fast-prefill opt-in: route prefill GEMMs through the m16n8k32
        // e4m3 M64 kernel (~1.47x vs v2 BF16, smem-relieved). Lossy (cosine 0.9997)
        // → highest priority when set, so it overrides the BF16/FP8 t_m128 arms.
        // PCND: explicit opt-in, default off = byte-for-byte prior behavior.
        let fp8_m64_prefill = self.w4a16_gemm_t_k.0 != 0
            && std::env::var_os("ATLAS_FP8_M64_PREFILL").is_some();
        // int8 W4A8 fast-prefill opt-in (ATLAS_INT8_PREFILL): route prefill GEMMs
        // through the validated requant→`int8_gemm_faith2` pipeline (cosine
        // 0.999978 vs the host full-precision dequant GEMM). HIGHEST priority when
        // set, so it overrides every other prefill arm. Needs both operands int8:
        // the NVFP4 weights are requanted to int8 once (cached, see
        // `ensure_int8_weight`) and the BF16 activations are requanted every call
        // into the shared scratch (`ensure_int8_scratch`). LOSSY (perf gate, not
        // bit-identical) — the _2.5h IoU gate is the final arbiter.
        // PCND: explicit opt-in, default off = byte-for-byte prior behavior; the
        // arm is a no-op (and no buffers are built) unless the kernels are loaded.
        let int8_prefill =
            self.int8_faith2_k.0 != 0 && std::env::var_os("ATLAS_INT8_PREFILL").is_some();
        if int8_prefill {
            static INT8_LOG: std::sync::Once = std::sync::Once::new();
            INT8_LOG.call_once(|| {
                eprintln!(
                    "[atlas] ATLAS_INT8_PREFILL=1: dense-FFN prefill via int8_gemm_faith2 (W4A8 requant→int8 MMA, lossy ~0.99998 cosine)"
                );
            });
        }
        // Pre-allocate (or reuse) the activation-requant scratch once per call,
        // sized to the largest projection K (= max(h, inter)) so the per-GEMM
        // arms never trigger a mid-call grow/sync. NULL when the int8 path is off.
        let (int8_a_i8, int8_a_scale) = if int8_prefill {
            self.ensure_int8_scratch(ctx.gpu, m, h.max(inter), stream)?
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        // A/B escape hatch (benchmark only): force the proven v1 BF16 kernel even
        // when v2 is loaded, so v1-vs-v2 prefill TTFT can be compared in one
        // binary. Default unset → prefer v2 (the faster, bit-identical variant).
        let use_v2 = self.w4a16_gemm_t_m128_bf16_v2_k.0 != 0
            && std::env::var_os("ATLAS_DISABLE_PREFILL_V2").is_none();
        let bf16_kernel = if use_v2 {
            self.w4a16_gemm_t_m128_bf16_v2_k
        } else {
            self.w4a16_gemm_t_m128_bf16_k
        };

        macro_rules! w4_gemm {
            ($w:expr, $wt:expr, $cell:expr, $in:expr, $out:expr, $n:expr, $k:expr) => {
                match $wt {
                    // int8 W4A8 fast prefill (ATLAS_INT8_PREFILL) — HIGHEST priority.
                    // Independent of `$wt`/the transposed copies: requant reads the
                    // non-transposed NVFP4 `$w` directly. Builds (once) + caches the
                    // int8 weight in `$cell`, then requant_a + faith2 via the shared
                    // scratch. Lossy (cosine ~0.99998).
                    _ if int8_prefill => {
                        let iw = self.ensure_int8_weight(
                            $cell, ctx.gpu, $w, $n, $k, stream,
                        )?;
                        ops::int8_gemm_faith2_prefill(
                            ctx.gpu,
                            self.int8_faith2_k,
                            self.requant_a_int8_k,
                            $in,
                            iw.w_i8,
                            iw.w_scale,
                            int8_a_i8,
                            int8_a_scale,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    }
                    // Lossless opt-in: BF16 128x128 tensor-core prefill (bit-equivalent
                    // to base `w4a16_gemm`). Preferred over the FP8 t_m128/v2 paths only
                    // when ATLAS_BF16_TC_PREFILL is set and the kernel is loaded. Within
                    // the lossless path, prefer the higher-occupancy v2 kernel (3 CTAs/SM,
                    // bit-identical to v1) when it is loaded; else the proven v1 kernel.
                    // Both go through the same launch helper (identical grid/block/args).
                    // FP8 M64 fast prefill (ATLAS_FP8_M64_PREFILL) — highest priority,
                    // M64 grid via the w4a16_gemm_n128 launcher.
                    Some(wt) if fp8_m64_prefill => ops::w4a16_gemm_n128(
                        ctx.gpu,
                        self.w4a16_gemm_t_k,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    Some(wt) if bf16_tc_prefill => ops::w4a16_gemm_n128_m128_bf16(
                        ctx.gpu,
                        bf16_kernel,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    // Prefer v2 (8-warp) > t_m128 (4-warp) > scalar-tile base.
                    Some(wt) if self.w4a16_gemm_t_m128_v2_k.0 != 0 => ops::w4a16_gemm_n128_m128_v2(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128_v2_k,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    Some(wt) if self.w4a16_gemm_t_m128_k.0 != 0 => ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128_k,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    _ => {
                        ops::w4a16_gemm(ctx.gpu, self.w4a16_gemm, $in, $w, $out, m, $n, $k, stream)?
                    }
                }
            };
        }

        // gate_proj GEMM: [M, H] → [M, inter]
        w4_gemm!(
            &self.weights.gate_proj,
            self.weights.gate_proj_t,
            &self.int8_gate,
            input,
            gate_out,
            inter,
            h
        );
        // up_proj GEMM: [M, H] → [M, inter]
        w4_gemm!(
            &self.weights.up_proj,
            self.weights.up_proj_t,
            &self.int8_up,
            input,
            up_out,
            inter,
            h
        );

        // activation(gate) * up for all M tokens (SiLU or GELU)
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;

        // down_proj GEMM: [M, inter] → [M, H]
        let output = ctx.buffers.moe_output();
        w4_gemm!(
            &self.weights.down_proj,
            self.weights.down_proj_t,
            &self.int8_down,
            gate_out,
            output,
            h,
            inter
        );

        Ok(())
    }

    /// Batched forward (per-token loop). Used by forward_batched in model loop.
    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }
}
