// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-grid launchers for unpadded NVFP4 routed-MoE decode.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const PERSISTENT_GRID_CTAS: u32 = 512;
const PERSISTENT_BLOCK_THREADS: u32 = 256;

pub fn moe_build_decode_worklist_c8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_offsets: DevicePtr,
    worklist: DevicePtr,
    total_groups: DevicePtr,
    num_experts: u32,
    routes: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_offsets)
        .arg_ptr(worklist)
        .arg_ptr(total_groups)
        .arg_u32(num_experts)
        .arg_u32(routes)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_decode_persistent_gate_up_c8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    sorted_token_ids: DevicePtr,
    worklist: DevicePtr,
    total_groups: DevicePtr,
    n_out: u32,
    k_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([PERSISTENT_GRID_CTAS, 1, 1])
        .block([PERSISTENT_BLOCK_THREADS, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(worklist)
        .arg_ptr(total_groups)
        .arg_u32(n_out)
        .arg_u32(k_size)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_decode_persistent_down_c8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_packed_ptrs: DevicePtr,
    down_scale_ptrs: DevicePtr,
    down_scale2: DevicePtr,
    down_out: DevicePtr,
    worklist: DevicePtr,
    total_groups: DevicePtr,
    n_out: u32,
    k_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([PERSISTENT_GRID_CTAS, 1, 1])
        .block([PERSISTENT_BLOCK_THREADS, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(down_packed_ptrs)
        .arg_ptr(down_scale_ptrs)
        .arg_ptr(down_scale2)
        .arg_ptr(down_out)
        .arg_ptr(worklist)
        .arg_ptr(total_groups)
        .arg_u32(n_out)
        .arg_u32(k_size)
        .launch(stream)
}
