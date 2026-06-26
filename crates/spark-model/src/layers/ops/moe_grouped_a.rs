// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// MoE grouped GEMM: per-expert W4A16 matrix multiply.
pub fn moe_w4a16_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed: DevicePtr,
    b_scale: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    num_experts: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_experts, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_packed)
        .arg_ptr(b_scale)
        .arg_f32(scale2)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

// ── Grouped MoE prefill ops ─────────────────────────────────────

/// Batched top-K softmax: N tokens in parallel.
///
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_softmax_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .launch(stream)
}

/// Batched sigmoid + correction-bias top-K MoE routing.
///
/// Kernel: `moe_topk_sigmoid_batched(gate_logits, bias, expert_indices,
///         expert_weights, num_experts, top_k, normalize, scaling_factor)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sigmoid_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    bias: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(bias)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .launch(stream)
}

/// Pointer-table grouped GEMM: one launch covers all experts.
///
/// Grid: (ceil(n_out/64), max_m_tiles, num_experts)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 64), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// Pointer-table grouped GEMM with N_TILE=128 (transposed weights).
///
/// Grid: (ceil(n_out/128), max_m_tiles, num_experts)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// FP8-A pointer-table grouped GEMM with transposed NVFP4 weights.
///
/// A must already be converted to FP8 E4M3. The launch shape mirrors
/// `moe_w4a16_grouped_gemm_ptrtable_n128`.
#[allow(clippy::too_many_arguments)]
pub fn moe_fp8_grouped_gemm_ptrtable_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// K64 down GEMM: K_STEP_T=64 eliminates pipeline stall (compute=128 cycles > load ~100 cycles).
/// Use when K=inter (512 for 35B) — 8 K-steps vs 16 with K32.
///
/// Grid: (ceil(n_out/128), max_m_tiles, num_experts)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable_k64_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// K64 fused gate+up GEMM — zero pipeline stall for K=h (2048 for 35B), 32 K-steps vs 64.
///
/// Grid: (ceil(2*n_out/128), max_m_tiles, num_experts)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_k64_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(2 * n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// Gather token rows into expert-sorted order: `permuted[i] = hidden[sorted_token_ids[i]]`.
/// `permuted` is `[total_expanded, hidden]`. One block per output row, threads
/// stride over `hidden`. Used by the FP4 grouped gate_up path (the CUTLASS
/// escape-hatch needs contiguous per-expert rows; the FP8 fused kernel gathers
/// internally so it doesn't need this).
///
/// Retained for the legacy FP4 escape-hatch + potential reuse; the live FP4
/// path now uses the fused kernel (in-kernel gather), so this is currently
/// uncalled.
#[allow(clippy::too_many_arguments, dead_code)]
pub fn moe_permute_tokens(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden_states: DevicePtr,
    permuted: DevicePtr,
    sorted_token_ids: DevicePtr,
    hidden: u32,
    total_expanded: u32,
    stream: u64,
) -> Result<()> {
    let threads = hidden.min(256).max(1);
    KernelLaunch::new(gpu, kernel)
        .grid([total_expanded, 1, 1])
        .block([threads, 1, 1])
        .arg_ptr(hidden_states)
        .arg_ptr(permuted)
        .arg_ptr(sorted_token_ids)
        .arg_u32(hidden)
        .arg_u32(total_expanded)
        .launch(stream)
}

/// K64 fused gate+up GEMM — M=128 variant (Block D #3 — Avarok pattern).
///
/// Doubles M_TILE from 64 → 128. Caller must compute `max_m_tiles_m128`
/// using divisor 128 (vs 64 for the M=64 variant). Grid covers the same
/// total work but with half the blocks (and twice the work per block).
///
/// Grid: (ceil(2*n_out/128), max_m_tiles_m128, num_experts)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_k64_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles_m128: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(2 * n_out, 128), max_m_tiles_m128, num_experts])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// Fused gate+up grouped GEMM — single launch for both projections.
///
/// Grid: (ceil(2*n_out/128), max_m_tiles, num_experts)  Block: (128, 1, 1)
/// First N cols → gate weights/output, last N cols → up weights/output.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(2 * n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// Element-wise SiLU activation + multiply: `output[i] = silu(gate[i]) * up[i]`.
///
/// Grid: (ceil(total_elements/256), 1, 1)  Block: (256, 1, 1)
pub fn moe_silu_mul(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    total_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(output)
        .arg_u32(total_elements)
        .launch(stream)
}

/// Counting sort tokens by expert assignment.
///
/// Produces sorted_token_ids (grouped by expert), expert_offsets (prefix sum),
/// and token_to_perm (reverse map for unpermute).
///
/// Grid: (1, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_sort_by_expert(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    topk_ids: DevicePtr,
    sorted_token_ids: DevicePtr,
    sorted_expert_ids: DevicePtr,
    expert_offsets: DevicePtr,
    token_to_perm: DevicePtr,
    total_expanded: u32,
    num_experts: u32,
    topk: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(topk_ids)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(sorted_expert_ids)
        .arg_ptr(expert_offsets)
        .arg_ptr(token_to_perm)
        .arg_u32(total_expanded)
        .arg_u32(num_experts)
        .arg_u32(topk)
        .launch(stream)
}

/// Unpermute + weighted reduce with pre-built reverse map.
///
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_unpermute_reduce_indexed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_output: DevicePtr,
    output: DevicePtr,
    token_to_perm: DevicePtr,
    topk_weights: DevicePtr,
    hidden_size: u32,
    num_tokens: u32,
    topk: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_output)
        .arg_ptr(output)
        .arg_ptr(token_to_perm)
        .arg_ptr(topk_weights)
        .arg_u32(hidden_size)
        .arg_u32(num_tokens)
        .arg_u32(topk)
        .launch(stream)
}

/// Batched sigmoid blend: output += sigmoid(dot(normed, gate_weight)) * shared_out.
///
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
pub fn moe_batched_blend(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    shared_out: DevicePtr,
    normed: DevicePtr,
    gate_weight: DevicePtr,
    hidden_size: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(shared_out)
        .arg_ptr(normed)
        .arg_ptr(gate_weight)
        .arg_u32(hidden_size)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// Single-launch CUTLASS grouped NVFP4 fused gate_up GEMM (Phase-2).
///
/// Bridges the device-resident per-expert pointer/scale tables to the host-side
/// [`spark_runtime::cutlass::nvfp4_grouped_gate_up_fused`] entry. `a` is the
/// expert-contiguous bf16 activation `[total_expanded, k]`. `gate_packed`/
/// `gate_sfb`/`up_packed`/`up_sfb` are device `[num_experts]` u64 pointer arrays
/// (into the CUTLASS-layout weight + swizzled-SFB tables); `gate_scale2`/
/// `up_scale2` are device `[num_experts]` f32 arrays; `expert_offsets` is the
/// device i32 `[num_experts+1]` prefix sum. This op copies all of those host-side
/// (the C entry indexes offsets/scale2 on the host and needs the pointer values),
/// syncs the stream so they are valid, then dispatches the single grouped launch.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gate_up_cutlass(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    gate_packed: DevicePtr,
    gate_sfb: DevicePtr,
    gate_scale2: DevicePtr,
    up_packed: DevicePtr,
    up_sfb: DevicePtr,
    up_scale2: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    num_experts: usize,
    inter: u32,
    hidden: u32,
    stream: u64,
) -> Result<()> {
    let read_u64 = |p: DevicePtr| -> Result<Vec<u64>> {
        let mut raw = vec![0u8; num_experts * 8];
        gpu.copy_d2h_on_stream(p, &mut raw, stream)?;
        Ok(raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    };
    let read_f32 = |p: DevicePtr| -> Result<Vec<f32>> {
        let mut raw = vec![0u8; num_experts * 4];
        gpu.copy_d2h_on_stream(p, &mut raw, stream)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };

    let gate_packed_h = read_u64(gate_packed)?;
    let gate_sfb_h = read_u64(gate_sfb)?;
    let up_packed_h = read_u64(up_packed)?;
    let up_sfb_h = read_u64(up_sfb)?;
    let gate_scale2_h = read_f32(gate_scale2)?;
    let up_scale2_h = read_f32(up_scale2)?;

    let mut off_raw = vec![0u8; (num_experts + 1) * 4];
    gpu.copy_d2h_on_stream(expert_offsets, &mut off_raw, stream)?;
    // The pointer/scale/offset host snapshots above are needed by the C entry
    // before it can launch — make sure the async D2H copies have landed.
    gpu.synchronize(stream)?;
    let eoff: Vec<i32> = off_raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    spark_runtime::cutlass::nvfp4_grouped_gate_up_fused(
        a.0,
        &gate_packed_h,
        &gate_sfb_h,
        &gate_scale2_h,
        &up_packed_h,
        &up_sfb_h,
        &up_scale2_h,
        c_gate.0,
        c_up.0,
        &eoff,
        inter,
        hidden,
        stream,
    )
}
