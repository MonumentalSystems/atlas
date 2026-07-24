// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for Metal kernels that consume native GGUF Q4_K blocks.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::PackedQ4Weight;

const Q4_K_VALUES: u32 = 256;
const Q4_K_BLOCK_BYTES: u64 = 144;

/// Decode projection: `y[N] = dequant(weight[N,K]) @ x[K]`.
pub fn gguf_q4_k_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ4Weight,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        weight.k.is_multiple_of(Q4_K_VALUES),
        "Q4_K K must be a multiple of {Q4_K_VALUES}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(weight.n, 4), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(weight.n)
        .arg_u32(weight.k)
        .launch(stream)
}

/// Project sorted token rows through a contiguous packed expert stack.
///
/// `expert_ids[slot]` chooses one `[N,K]` matrix at
/// `expert_base + expert_id * expert_stride_bytes`. A negative id writes a
/// zero output row. Gate, up, and down projections use the same primitive.
#[allow(clippy::too_many_arguments)]
pub fn gguf_q4_k_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    expert_base: DevicePtr,
    expert_stride_bytes: u64,
    expert_ids: DevicePtr,
    output: DevicePtr,
    total_slots: u32,
    n: u32,
    k: u32,
    num_experts: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(Q4_K_VALUES),
        "Q4_K K must be a multiple of {Q4_K_VALUES}"
    );
    let matrix_bytes = u64::from(n)
        .checked_mul(u64::from(k / Q4_K_VALUES))
        .and_then(|blocks| blocks.checked_mul(Q4_K_BLOCK_BYTES))
        .ok_or_else(|| anyhow::anyhow!("Q4_K expert matrix byte size overflow"))?;
    ensure!(
        expert_stride_bytes >= matrix_bytes,
        "Q4_K expert stride {expert_stride_bytes} is smaller than matrix size {matrix_bytes}"
    );
    ensure!(num_experts > 0, "Q4_K expert stack must not be empty");
    KernelLaunch::new(gpu, kernel)
        .grid([total_slots, div_ceil(n, 4), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(expert_base)
        .arg_ptr(expert_ids)
        .arg_ptr(output)
        .arg_u32(total_slots)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_experts)
        .arg_u64(expert_stride_bytes)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn wrappers_validate_q4_k_blocks_and_expert_stride() {
        let gpu = MockGpuBackend::new();
        let invalid = PackedQ4Weight {
            weight: DevicePtr(1),
            n: 2,
            k: 255,
        };
        assert!(
            gguf_q4_k_gemv(
                &gpu,
                KernelHandle(1),
                DevicePtr(2),
                &invalid,
                DevicePtr(3),
                0
            )
            .is_err()
        );
        assert!(
            gguf_q4_k_grouped_gemm(
                &gpu,
                KernelHandle(1),
                DevicePtr(1),
                DevicePtr(2),
                143,
                DevicePtr(3),
                DevicePtr(4),
                1,
                1,
                256,
                1,
                0,
            )
            .is_err()
        );
    }
}
