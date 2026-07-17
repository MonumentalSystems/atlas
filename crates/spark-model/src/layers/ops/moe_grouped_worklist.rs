// SPDX-License-Identifier: AGPL-3.0-only

//! Device-worklist launchers for compact NVFP4 MoE decode.
//!
//! The preceding `moe_build_tile_worklist` kernel writes both the tile list and
//! count on the same CUDA stream. These launchers deliberately use a fixed
//! persistent grid, so decode capture never needs a D2H count read.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const COMPACT_GRID_CTAS: u32 = 256;
const COMPACT_BLOCK_THREADS: u32 = 128;

#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_k64_worklist(
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
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([COMPACT_GRID_CTAS, 1, 1])
        .block([COMPACT_BLOCK_THREADS, 1, 1])
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
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_k64_worklist(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([COMPACT_GRID_CTAS, 1, 1])
        .block([COMPACT_BLOCK_THREADS, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .launch(stream)
}
