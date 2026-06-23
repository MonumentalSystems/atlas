// SPDX-License-Identifier: AGPL-3.0-only

//! Shared kernel dispatch operations.
//!
//! Freestanding functions wrapping CUDA kernel launches via `KernelLaunch`.
//! Layer implementations compose these to build forward passes.
//!
//! Each function's parameters exactly match the corresponding CUDA kernel
//! signature. Grid/block dimensions are computed from the problem size.
//!
//! Refactor wave 4a (2026-05-03): split into `ops/` sub-modules with thematic
//! groupings. All public functions remain available at this path via re-export.

#[path = "ops/activations.rs"]
mod activations;
#[path = "ops/embeddings.rs"]
mod embeddings;
#[path = "ops/fp8_gemv_batch.rs"]
mod fp8_gemv_batch;
#[path = "ops/fp8_moe.rs"]
mod fp8_moe;
#[path = "ops/fp8_moe_batch_a.rs"]
mod fp8_moe_batch_a;
#[path = "ops/fp8_moe_batch_b.rs"]
mod fp8_moe_batch_b;
#[path = "ops/gemm_dense.rs"]
mod gemm_dense;
#[path = "ops/gemm_quant.rs"]
mod gemm_quant;
#[path = "ops/kv_cache.rs"]
mod kv_cache;
#[path = "ops/kv_cache_fp8k.rs"]
mod kv_cache_fp8k;
#[path = "ops/kv_cache_turbok.rs"]
mod kv_cache_turbok;
#[path = "ops/moe_expert.rs"]
mod moe_expert;
#[path = "ops/moe_expert_more.rs"]
mod moe_expert_more;
#[path = "ops/moe_atomic_c4.rs"]
mod moe_atomic_c4;
#[path = "ops/moe_gate.rs"]
mod moe_gate;
#[path = "ops/moe_grouped_a.rs"]
mod moe_grouped_a;
#[path = "ops/moe_grouped_b.rs"]
mod moe_grouped_b;
#[path = "ops/moe_prefill.rs"]
mod moe_prefill;
#[path = "ops/norm.rs"]
mod norm;
#[path = "ops/prefill_attn_a.rs"]
mod prefill_attn_a;
#[path = "ops/prefill_attn_b.rs"]
mod prefill_attn_b;
#[path = "ops/prefill_attn_batched.rs"]
mod prefill_attn_batched;
#[path = "ops/prefill_attn_fp8k.rs"]
mod prefill_attn_fp8k;
#[path = "ops/prefill_attn_main_a.rs"]
mod prefill_attn_main_a;
#[path = "ops/prefill_attn_main_b.rs"]
mod prefill_attn_main_b;
#[path = "ops/prefill_attn_turbok.rs"]
mod prefill_attn_turbok;
#[path = "ops/quant_dispatch.rs"]
mod quant_dispatch;
#[path = "ops/sampling.rs"]
mod sampling;
#[path = "ops/ssm_gdn_a.rs"]
mod ssm_gdn_a;
#[path = "ops/ssm_gdn_b.rs"]
mod ssm_gdn_b;
#[path = "ops/ssm_gdn_batched.rs"]
mod ssm_gdn_batched;
#[path = "ops/ssm_mamba.rs"]
mod ssm_mamba;
#[path = "ops/ssm_preproc.rs"]
mod ssm_preproc;

pub use activations::*;
pub use embeddings::*;
pub use fp8_gemv_batch::*;
pub use fp8_moe::*;
pub use fp8_moe_batch_a::*;
pub use fp8_moe_batch_b::*;
pub use gemm_dense::*;
pub use gemm_quant::*;
pub use kv_cache::*;
pub use kv_cache_fp8k::*;
pub use kv_cache_turbok::*;
pub use moe_expert::*;
pub use moe_expert_more::*;
pub use moe_atomic_c4::*;
pub use moe_gate::*;
pub use moe_grouped_a::*;
#[allow(unused_imports)]
pub(crate) use moe_grouped_b::*;
pub use moe_prefill::*;
pub use norm::*;
pub use prefill_attn_a::*;
pub use prefill_attn_b::*;
pub use prefill_attn_batched::*;
pub use prefill_attn_fp8k::*;
pub use prefill_attn_main_a::*;
pub use prefill_attn_main_b::*;
pub use prefill_attn_turbok::*;
pub use quant_dispatch::*;
pub use sampling::*;
pub use ssm_gdn_a::*;
pub use ssm_gdn_b::*;
pub use ssm_gdn_batched::*;
pub use ssm_mamba::*;
pub use ssm_preproc::*;

/// Whether block-scaled FP8 prefill (per-128-block weight scales + per-token
/// activation scales via `fp8_gemm_t_blockscaled` / `moe_w8a8_grouped_gemm`)
/// is enabled. This is the DEFAULT for block-scaled FP8 checkpoints as of
/// 2026-06-17: it matches vLLM's per-block precision and avoids the
/// single-scale `fp8_gemm_n128` path, whose collapse of per-block dynamic
/// range pushed long-context tool-arg decode into the FP8 argmax-flip regime
/// (B1 drift gauge ~1400 → ~100 once block-scaled prefill is on).
///
/// Opt out with `ATLAS_FP8_SINGLE_SCALE=1` to restore the old single-scale
/// prefill (diagnostic / fallback only). Call sites still guard on the
/// presence of block-scaled weights + kernel handles, so builds/models
/// without those fall back automatically regardless of this flag.
pub fn fp8_blockscaled_prefill_enabled() -> bool {
    !matches!(
        std::env::var("ATLAS_FP8_SINGLE_SCALE").ok().as_deref(),
        Some("1")
    )
}

/// cuBLASLt GEMM path enabled? (`ATLAS_CUBLAS_GEMM=1`), cached. The hand-written
/// mma.sync projection GEMMs hit only ~30% of the cuBLAS bf16 ceiling on GB10.
pub fn cublas_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUBLAS_GEMM").ok().as_deref() == Some("1"))
}

/// Native-FP8 cuBLASLt GEMM path enabled? (`ATLAS_CUBLAS_FP8=1`), cached.
pub fn cublas_fp8_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUBLAS_FP8").ok().as_deref() == Some("1"))
}

/// CUTLASS GEMM path enabled? (`ATLAS_CUTLASS_GEMM=1`), cached. M0 is scoped to
/// dense BF16 projections using the same FP8→BF16 cached dequant as cuBLASLt.
pub fn cutlass_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUTLASS_GEMM").ok().as_deref() == Some("1"))
}

/// Native CUTLASS NVFP4 GEMM path enabled? (`ATLAS_CUTLASS_NVFP4_GEMM=1`).
/// This path quantizes activations to CUTLASS NVFP4 and consumes transposed
/// Atlas NVFP4 weights after repacking scales into CUTLASS SM120 layout.
pub fn cutlass_nvfp4_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        std::env::var("ATLAS_CUTLASS_NVFP4_GEMM").ok().as_deref() == Some("1")
    })
}

fn cutlass_nvfp4_flag_enabled(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

/// Native CUTLASS NVFP4 SSM QKVZ path enabled.
pub fn cutlass_nvfp4_qkvz_enabled() -> bool {
    cutlass_nvfp4_gemm_enabled() || cutlass_nvfp4_flag_enabled("ATLAS_CUTLASS_NVFP4_QKVZ")
}

/// Native CUTLASS NVFP4 attention Q/K/V path enabled for the named projection.
pub fn cutlass_nvfp4_attn_qkv_enabled(label: &str) -> bool {
    cutlass_nvfp4_gemm_enabled()
        || match label {
            "q_proj" => cutlass_nvfp4_flag_enabled("ATLAS_CUTLASS_NVFP4_ATTN_Q"),
            "k_proj" | "v_proj" => cutlass_nvfp4_flag_enabled("ATLAS_CUTLASS_NVFP4_ATTN_KV"),
            _ => false,
        }
}

/// Native CUTLASS NVFP4 attention O path enabled.
pub fn cutlass_nvfp4_attn_o_enabled() -> bool {
    cutlass_nvfp4_gemm_enabled() || cutlass_nvfp4_flag_enabled("ATLAS_CUTLASS_NVFP4_ATTN_O")
}

/// Native CUTLASS NVFP4 SSM out-projection path enabled.
pub fn cutlass_nvfp4_ssm_out_enabled() -> bool {
    cutlass_nvfp4_flag_enabled("ATLAS_CUTLASS_NVFP4_SSM_OUT")
}

pub fn log_cutlass_nvfp4_route(name: &str, m: u32, n: u32, k: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, u32, u32, u32)>>> = OnceLock::new();
    let mut h: u64 = 1469598103934665603;
    for b in name.bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen.lock().unwrap().insert((h, m, n, k)) {
        tracing::warn!("CUTLASS_NVFP4_ROUTE {name} M={m} N={n} K={k}");
    }
}

/// Route a projection through native-FP8 cuBLASLt block-scaled matmul: quantize
/// the activation to FP8 + per-[token,128-of-K] VEC128 scales (the existing
/// `per_token_group_quant_fp8` kernel), feed the FP8 weight + its per-128×128
/// block scales directly (zero dequant, zero extra weight memory). Both operands
/// 128-block-scaled (cuBLASLt requires it). ~1.8× the bf16 path (152 vs 85 TF).
///
/// `act_fp8_scratch`/`act_scale_scratch` must hold the padded extents (the
/// `buffers.fp8_act`/`fp8_act_scale` arena buffers, sized for max_batch_tokens).
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    ptg_quant_k: spark_runtime::gpu::KernelHandle,
    act_bf16: spark_runtime::gpu::DevicePtr,
    act_fp8_scratch: spark_runtime::gpu::DevicePtr,
    act_scale_scratch: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    // Quantize the real M tokens → fp8 bytes + VEC128 scales [M, K/128].
    per_token_group_quant_fp8(
        gpu,
        ptg_quant_k,
        act_bf16,
        act_fp8_scratch,
        act_scale_scratch,
        m,
        k,
        stream,
    )?;
    // cuBLASLt requires the scale-tensor M extent to be a multiple of 4; pad to
    // 16 (TC-friendly) and zero the padding scale rows so the phantom output
    // columns (ignored by the caller) are well-defined.
    let m_pad = m.div_ceil(16) * 16;
    if m_pad > m {
        let kg = (k / 128) as usize;
        let pad_off = m as usize * kg * 4;
        let pad_bytes = (m_pad - m) as usize * kg * 4;
        gpu.memset_async(act_scale_scratch.offset(pad_off), 0, pad_bytes, stream)?;
    }
    spark_runtime::cublaslt::fp8_gemm_act_weight_t_blkscaled(
        act_fp8_scratch.0,
        act_scale_scratch.0,
        fp8w.weight.0,
        fp8w.row_scale.0,
        out.0,
        m_pad,
        n,
        k,
        stream,
    )
}

/// Re-quantize a block-scaled FP8 weight `[N,K]` → ROW-WISE FP8 (E4M3 + per-row
/// FP32 scale `[N]`) on-GPU once, cached by the FP8 weight pointer. Path:
/// block-fp8 → BF16 (transient) → row-wise fp8. Backs the GB10-supported
/// `cublas_fp8_rowwise_proj`. Returns `(fp8_weight_ptr, per_row_scale_ptr)`.
fn requant_weight_rowwise_fp8_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<(u64, u64)> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u64, (u64, u64)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&p) = cache.lock().unwrap().get(&fp8w.weight.0) {
        return Ok(p);
    }
    let (n, k) = (fp8w.n, fp8w.k);
    // 1. block-fp8 → BF16 (transient scratch, freed after re-quant).
    let bf16 = gpu.alloc(n as usize * k as usize * 2)?;
    let block = 128u32;
    let sk = k / block;
    let dq = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, dq)
        .grid([div_ceil(k, 64), div_ceil(n, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(fp8w.weight)
        .arg_ptr(fp8w.row_scale)
        .arg_ptr(bf16)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(block)
        .arg_u32(block)
        .arg_u32(sk)
        .arg_u32(1)
        .launch(stream)?;
    // 2. BF16 → row-wise fp8 [N,K] + per-row scale [N].
    let w_fp8 = gpu.alloc(n as usize * k as usize)?;
    let w_scale = gpu.alloc(n as usize * 4)?;
    let qk = gpu.kernel("quant_rowwise_fp8", "quant_rowwise_fp8")?;
    KernelLaunch::new(gpu, qk)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16)
        .arg_ptr(w_fp8)
        .arg_ptr(w_scale)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    gpu.synchronize(stream)?; // re-quant must finish before the transient bf16 is freed
    gpu.free(bf16)?;
    cache
        .lock()
        .unwrap()
        .insert(fp8w.weight.0, (w_fp8.0, w_scale.0));
    Ok((w_fp8.0, w_scale.0))
}

/// Route a projection through ROW-WISE native-FP8 cuBLASLt (the fp8 path GB10
/// supports). Weight is re-quantized once to per-row fp8 (cached); the activation
/// is quantized per-token each call. ~1.8× the bf16 path (152 vs 85 TF), and
/// frees the bf16-dequant memory the bf16 path holds.
/// `act_fp8_scratch` ≥ m*k fp8 bytes; `act_scale_scratch` ≥ m f32 (e.g. the
/// `buffers.fp8_act` / `fp8_act_scale` arena buffers).
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_rowwise_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act_bf16: spark_runtime::gpu::DevicePtr,
    act_fp8_scratch: spark_runtime::gpu::DevicePtr,
    act_scale_scratch: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    use spark_runtime::kernel_args::KernelLaunch;
    let (w_fp8, w_scale) = requant_weight_rowwise_fp8_cached(gpu, fp8w, stream)?;
    // Per-token row-wise quant of the activation → fp8 [M,K] + scale [M].
    let qk = gpu.kernel("quant_rowwise_fp8", "quant_rowwise_fp8")?;
    KernelLaunch::new(gpu, qk)
        .grid([m, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(act_bf16)
        .arg_ptr(act_fp8_scratch)
        .arg_ptr(act_scale_scratch)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)?;
    spark_runtime::cublaslt::fp8_gemm_act_weight_t_rowwise(
        act_fp8_scratch.0,
        act_scale_scratch.0,
        w_fp8,
        w_scale,
        out.0,
        m,
        n,
        k,
        stream,
    )
}

/// Dequantize a block-scaled FP8 weight `[N,K]` → BF16 on-GPU once, cached by the
/// FP8 weight pointer (weights are immutable after load). 128×128 blocks + FP32
/// scales (the holo layout). Backs [`cublas_bf16_proj`].
fn dequant_fp8_bf16_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<u64> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u64, u64>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&p) = cache.lock().unwrap().get(&fp8w.weight.0) {
        return Ok(p);
    }
    let (n, kk) = (fp8w.n, fp8w.k);
    let out = gpu.alloc(n as usize * kk as usize * 2)?; // BF16 [N,K]
    let block = 128u32;
    let sk = kk / block;
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(kk, 64), div_ceil(n, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(fp8w.weight)
        .arg_ptr(fp8w.row_scale)
        .arg_ptr(out)
        .arg_u32(n)
        .arg_u32(kk)
        .arg_u32(block)
        .arg_u32(block)
        .arg_u32(sk)
        .arg_u32(1) // scale_is_fp32
        .launch(stream)?;
    cache.lock().unwrap().insert(fp8w.weight.0, out.0);
    Ok(out.0)
}

fn dequant_fp8_bf16_uncached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<spark_runtime::gpu::DevicePtr> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let (n, kk) = (fp8w.n, fp8w.k);
    let out = gpu.alloc(n as usize * kk as usize * 2)?;
    let block = 128u32;
    let sk = kk / block;
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(kk, 64), div_ceil(n, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(fp8w.weight)
        .arg_ptr(fp8w.row_scale)
        .arg_ptr(out)
        .arg_u32(n)
        .arg_u32(kk)
        .arg_u32(block)
        .arg_u32(block)
        .arg_u32(sk)
        .arg_u32(1)
        .launch(stream)?;
    Ok(out)
}

/// Route a projection `out[M,N] = act[M,K] @ weightᵀ` through cuBLASLt BF16.
/// The FP8 weight is dequantized to BF16 once (cached); W16A16 here is strictly
/// more accurate than the blockscaled W8A8 path it replaces.
#[allow(clippy::too_many_arguments)]
pub fn cublas_bf16_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let w_bf16 = dequant_fp8_bf16_cached(gpu, fp8w, stream)?;
    spark_runtime::cublaslt::bf16_gemm_act_weight_t(act.0, w_bf16, out.0, m, n, k, stream)
}

/// Route a projection `out[M,N] = act[M,K] @ weightᵀ` through CUTLASS BF16.
/// This is the M0 de-risk path for replacing Atlas/cuBLAS GEMMs with
/// CUTLASS-backed kernels on GB10; keep it behind `ATLAS_CUTLASS_GEMM=1`.
#[allow(clippy::too_many_arguments)]
pub fn cutlass_bf16_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let w_bf16 = dequant_fp8_bf16_cached(gpu, fp8w, stream)?;
    spark_runtime::cutlass::bf16_gemm_act_weight_t(act.0, w_bf16, out.0, m, n, k, stream)
}

/// Route a projection `out[M,N] = act[M,K] @ weightᵀ` through native CUTLASS
/// NVFP4. The activation is packed to CUTLASS NVFP4 inside the runtime wrapper.
/// `weight_t` must be Atlas's transposed NVFP4 layout `[K/2,N]` plus
/// `[K/16,N]` scales, as produced by `QuantizedWeight::transpose_for_gemm`.
#[allow(clippy::too_many_arguments)]
/// Transpose a native NVFP4 checkpoint weight from Atlas `[K/2,N]` into the
/// CUTLASS `[N,K/2]` byte layout the GEMM consumes, caching the result by
/// source weight ptr. Without this the ColumnMajor B operand is read
/// transposed and the GEMM produces garbage (cos≈0 vs reference).
fn cutlass_nvfp4_weight_transposed_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    weight_t: &crate::weight_map::QuantizedWeight,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<u64> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u64, u64>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&p) = cache.lock().unwrap().get(&weight_t.weight.0) {
        return Ok(p);
    }
    let dst = gpu.alloc((n as usize) * (k as usize) / 2)?;
    spark_runtime::cutlass::transpose_nvfp4_packed_kton(weight_t.weight.0, dst.0, n, k, stream)?;
    gpu.synchronize(stream)?;
    cache.lock().unwrap().insert(weight_t.weight.0, dst.0);
    Ok(dst.0)
}

#[allow(clippy::too_many_arguments)]
pub fn cutlass_nvfp4_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act: spark_runtime::gpu::DevicePtr,
    weight_t: &crate::weight_map::QuantizedWeight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let packed = cutlass_nvfp4_weight_transposed_cached(gpu, weight_t, n, k, stream)?;
    spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
        act.0,
        packed,
        weight_t.weight_scale.0,
        weight_t.weight_scale_2,
        out.0,
        m,
        n,
        k,
        stream,
    )
}

fn cutlass_nvfp4_weight_from_fp8_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<(u64, u64)> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u64, (u64, u64)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&p) = cache.lock().unwrap().get(&fp8w.weight.0) {
        return Ok(p);
    }

    let n = fp8w.n as usize;
    let k = fp8w.k as usize;
    let w_bf16 = dequant_fp8_bf16_uncached(gpu, fp8w, stream)?;
    let packed_t = gpu.alloc(n * k / 2)?;
    let scale_t = gpu.alloc(n * k / 16)?;
    spark_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
        w_bf16.0,
        packed_t.0,
        scale_t.0,
        fp8w.n,
        fp8w.k,
        stream,
    )?;
    gpu.synchronize(stream)?;
    gpu.free(w_bf16)?;
    cache
        .lock()
        .unwrap()
        .insert(fp8w.weight.0, (packed_t.0, scale_t.0));
    Ok((packed_t.0, scale_t.0))
}

/// Native CUTLASS NVFP4 projection for FP8 checkpoint weights. The FP8 weight
/// is dequantized to BF16 using the existing cache, then packed once into
/// Atlas-transposed NVFP4 data/scales and reused for future calls.
#[allow(clippy::too_many_arguments)]
pub fn cutlass_nvfp4_proj_from_fp8(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let (packed_t, scale_t) = cutlass_nvfp4_weight_from_fp8_cached(gpu, fp8w, stream)?;
    spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
        act.0, packed_t, scale_t, 1.0, out.0, m, n, k, stream,
    )
}

/// Roofline instrumentation: log each unique (kernel, M, N, K) GEMM shape once,
/// gated by `ATLAS_GEMM_SHAPE_LOG=1`. Used to cross-reference nsys per-call
/// durations → achieved TFLOPS/bandwidth vs GB10 peak.
pub fn log_gemm_shape(name: &str, m: u32, n: u32, k: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    if std::env::var("ATLAS_GEMM_SHAPE_LOG").ok().as_deref() != Some("1") {
        return;
    }
    static SEEN: OnceLock<Mutex<HashSet<(u64, u32, u32, u32)>>> = OnceLock::new();
    let mut h: u64 = 1469598103934665603;
    for b in name.bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    let key = (h, m, n, k);
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen.lock().unwrap().insert(key) {
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        tracing::warn!("GEMM_SHAPE {name} M={m} N={n} K={k} FLOP={flop:.3e}");
    }
}
