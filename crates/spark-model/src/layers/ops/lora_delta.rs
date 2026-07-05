// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime LoRA delta: y += scale * (x @ A^T) @ B^T, BF16 side-path.
//! Zero new CUDA kernels — reuses dense_gemv_bf16 / dense_gemm_tc /
//! dense_gemm_bf16 / bf16_scaled_add, all shipped in kernels/gb10/common/.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Resolved once at adapter load (module names per common/KERNEL.toml:
/// gemv=dense_gemv_bf16.cu, gemm_tc=dense_gemm_tc.cu, gemm=dense_gemm_bf16.cu,
/// residual_add=residual_add.cu — stem, no override).
#[derive(Clone, Copy)]
pub struct LoraKernels {
    pub gemv_k: KernelHandle,
    pub gemm_tc_k: KernelHandle, // KernelHandle(0) on miss -> gemm_k fallback
    pub gemm_k: KernelHandle,
    pub scaled_add_k: KernelHandle,
}

impl LoraKernels {
    pub fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gemm_tc_k: crate::layers::try_kernel(gpu, "gemm_tc", "dense_gemm_tc"),
            gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            scaled_add_k: gpu.kernel("residual_add", "bf16_scaled_add")?,
        })
    }
}

/// One adapted module. A/B are PEFT tensors VERBATIM (host F16->BF16 at load):
///   a: [rank, k_in]  row-major BF16  (PEFT lora_A [r, in_features] — already
///                                     the B-operand `[N,K]` layout dense_* expect)
///   b: [n_out, rank] row-major BF16  (PEFT lora_B [out_features, r] — likewise)
/// Both are rank-padded to the pool's max_rank (zero rows/cols beyond `rank`),
/// so kernels may uniformly run at the pool rank — bit-identical to true rank.
/// scale = lora_alpha/r, or lora_alpha/sqrt(r) under use_rslora — read per
/// adapter at load, never defaulted. Do NOT pre-fold into B (keeps tensors
/// verbatim for the M0 offline parity test); it rides the scaled_add for free.
#[derive(Debug, Clone, Copy)]
pub struct LoraPair {
    pub a: DenseWeight,
    pub b: DenseWeight,
    pub rank: u32,
    pub k_in: u32,
    pub n_out: u32,
    pub scale: f32,
    /// The pool's padded rank — the ROW STRIDE of `b` (and row count of `a`).
    /// Kernels MUST contract/produce at this dim, not `rank`: B rows are
    /// `max_rank` elements apart in the pool, so a `k = rank` expand would
    /// misread every row past the first when `rank < max_rank`. Pad rows of
    /// A and pad cols of B are zeroed at pack time, so running the shrink at
    /// `n = max_rank` and the expand at `k = max_rank` is bit-identical to
    /// the true-rank product.
    pub max_rank: u32,
}

/// Per-layer attention-side LoRA weights, installed by copy onto
/// `Qwen3AttentionLayer`.
///
/// v0: NO q field — q_proj is rejected at load (attn_output_gate makes the
/// projection 2x q_dim Q+gate interleaved; a PEFT q delta maps only to the
/// Q half). The named rejection lives in the loader (`crate::lora`).
#[derive(Clone, Copy)]
pub struct LoraAttnWeights {
    pub k: Option<LoraPair>,
    pub v: Option<LoraPair>,
    pub o: Option<LoraPair>,
    pub kernels: LoraKernels,
}

/// Per-layer dense-FFN LoRA weights, installed by copy onto `DenseFfnLayer`.
#[derive(Clone, Copy)]
pub struct LoraFfnWeights {
    pub gate: Option<LoraPair>,
    pub up: Option<LoraPair>,
    pub down: Option<LoraPair>,
    pub kernels: LoraKernels,
}

/// base_out[m, n_out] += scale * (x[m, k_in] @ a^T) @ b^T.
///
/// CONTIGUITY CONTRACT: x rows contiguous with stride k_in*2 bytes, base_out
/// rows contiguous with stride n_out*2 bytes. Every v0 site satisfies this
/// (k/v/o/gate/up/down all land in dedicated contiguous buffers/regions);
/// strided cases (multi-seq per-seq qkv_buf) must loop with m=1 on offset ptrs.
///
/// GRAPH-SAFE: pure kernel launches, no alloc/sync; a/b (load-time device
/// weights), lora_xa/lora_delta (BufferArena, fixed address), and scale
/// (baked kernel arg, constant for a startup-static adapter) are all
/// pointer/value-stable across capture and replay — identical status to base
/// weights. m==1 -> GEMV; m>1 -> tensor-core GEMM (scalar fallback).
///
/// POOL LAYOUT (lora/mod.rs pack): A is [max_rank, k_in] (real rows at the
/// head, pad rows zero), B is [n_out, max_rank] row-major (pad COLS zero,
/// row stride = max_rank). Both stages therefore run at `pair.max_rank`:
/// shrink n = max_rank (xa pad cols come out zero), expand k = max_rank
/// (matches B's row stride; zero pads contribute nothing) — bit-identical
/// to a true-rank product.
#[allow(clippy::too_many_arguments)]
pub fn apply_lora_delta(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    pair: &LoraPair,
    x: DevicePtr,        // [m, pair.k_in] BF16
    base_out: DevicePtr, // [m, pair.n_out] BF16, modified in place
    m: u32,
    lora_xa: DevicePtr,    // arena scratch >= m * max_rank BF16
    lora_delta: DevicePtr, // arena scratch >= m * n_out BF16
    stream: u64,
) -> Result<()> {
    if m == 1 {
        // shrink: [1,k_in] @ A[max_rank,k_in]^T -> xa[1,max_rank]
        ops::dense_gemv(
            gpu,
            kernels.gemv_k,
            x,
            &pair.a,
            lora_xa,
            pair.max_rank,
            pair.k_in,
            stream,
        )?;
        // expand: [1,max_rank] @ B[n_out,max_rank]^T -> delta[1,n_out]
        ops::dense_gemv(
            gpu,
            kernels.gemv_k,
            lora_xa,
            &pair.b,
            lora_delta,
            pair.n_out,
            pair.max_rank,
            stream,
        )?;
    } else if kernels.gemm_tc_k.0 != 0 {
        ops::dense_gemm_tc(
            gpu,
            kernels.gemm_tc_k,
            x,
            &pair.a,
            lora_xa,
            m,
            pair.max_rank,
            pair.k_in,
            stream,
        )?;
        ops::dense_gemm_tc(
            gpu,
            kernels.gemm_tc_k,
            lora_xa,
            &pair.b,
            lora_delta,
            m,
            pair.n_out,
            pair.max_rank,
            stream,
        )?;
    } else {
        ops::dense_gemm(
            gpu,
            kernels.gemm_k,
            x,
            &pair.a,
            lora_xa,
            m,
            pair.max_rank,
            pair.k_in,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            kernels.gemm_k,
            lora_xa,
            &pair.b,
            lora_delta,
            m,
            pair.n_out,
            pair.max_rank,
            stream,
        )?;
    }
    // fold: base_out += scale * delta   (kernels/gb10/common/residual_add.cu:60)
    ops::scaled_add(
        gpu,
        kernels.scaled_add_k,
        base_out,
        lora_delta,
        pair.scale,
        m * pair.n_out,
        stream,
    )
}
