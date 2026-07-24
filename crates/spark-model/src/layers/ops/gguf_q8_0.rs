// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for Metal kernels that consume native GGUF Q8_0 blocks.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::PackedQ8Weight;

/// Decode projection: `y[N] = dequant(weight[N,K]) @ x[K]`.
pub fn gguf_q8_0_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ8Weight,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        weight.k.is_multiple_of(32),
        "Q8_0 K must be a multiple of 32"
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

/// Prefill projection: `Y[M,N] = X[M,K] @ dequant(weight[N,K])^T`.
pub fn gguf_q8_0_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ8Weight,
    output: DevicePtr,
    m: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        weight.k.is_multiple_of(32),
        "Q8_0 K must be a multiple of 32"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(weight.n, 16), div_ceil(m, 16), 1])
        .block([16, 16, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(weight.n)
        .arg_u32(weight.k)
        .launch(stream)
}

/// Gather `num_tokens` embedding rows directly from a packed Q8_0 table.
pub fn gguf_q8_0_embedding(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    token_ids: DevicePtr,
    table: &PackedQ8Weight,
    output: DevicePtr,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        table.k.is_multiple_of(32),
        "Q8_0 hidden size must be a multiple of 32"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(table.k, 16), div_ceil(num_tokens, 16), 1])
        .block([16, 16, 1])
        .arg_ptr(token_ids)
        .arg_ptr(table.weight)
        .arg_ptr(output)
        .arg_u32(num_tokens)
        .arg_u32(table.k)
        .arg_u32(table.n)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn wrappers_reject_partial_q8_blocks() {
        let gpu = MockGpuBackend::new();
        let w = PackedQ8Weight {
            weight: DevicePtr(1),
            n: 2,
            k: 33,
        };
        assert!(gguf_q8_0_gemv(&gpu, KernelHandle(1), DevicePtr(2), &w, DevicePtr(3), 0).is_err());
        assert!(
            gguf_q8_0_gemm(&gpu, KernelHandle(1), DevicePtr(2), &w, DevicePtr(3), 2, 0).is_err()
        );
    }
}
