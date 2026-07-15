// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for CANDIDATE B of the native keep-packed ternary Q2_0
//! decode GEMV (`kernels/gb10/common/q2_0_gemv_vec.cu`, module stem
//! `q2_0_gemv_vec`).
//!
//! Same call surface as [`super::gemv_q2`] — the Q2_0 weight carries its fp16
//! scale INLINE in each `block_q2_0`, so there is no separate scale-pointer
//! argument. The only launch-geometry difference is the CANDIDATE-B thread map:
//! ONE warp per output row, EIGHT rows per 256-thread CTA, so the grid is
//! `(ceil(N/8),1,1)` (vs the 2-warp/4-row `ceil(N/4)` of the baseline kernel).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::PackedQ2Weight;

/// Q2_0 GEMV (M=1 decode), CANDIDATE B: `C[1,N] = A[1,K] @ dequant(B)`.
///
/// Vectorized code loads (one `uint32` = 16 ternary codes per lane) + shared-
/// memory activation staging. `A` BF16 `[1,K]`, `B` raw `block_q2_0`, `C` BF16
/// `[1,N]`. Dequant `(code-1)*d` happens inside the dot-product.
///
/// Kernel: `q2_0_gemv_vec(A, B, C, N, K, group)`  Grid: (ceil(N/8),1,1) Block: (256,1,1)
pub fn q2_0_gemv_vec(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ2Weight,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(weight.n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(weight.n)
        .arg_u32(weight.k)
        .arg_u32(weight.group as u32)
        .launch(stream)
}

/// Q2_0 batched GEMV (M=1..8 decode), CANDIDATE B: `C[M,N] = A[M,K] @ dequant(B)`.
///
/// Reads each weight word once and MAC's it into all `m` accumulators (all `m`
/// activation rows staged in smem). `A` BF16 `[M,K]` row-major, `C` BF16
/// `[M,N]` row-major. Bit-consistent with running the M=1 kernel `M` times.
///
/// Kernel: `q2_0_gemv_vec_batchm(A, B, C, N, K, group, M)`.
#[allow(clippy::too_many_arguments)]
pub fn q2_0_gemv_vec_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ2Weight,
    output: DevicePtr,
    m: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(weight.n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(weight.n)
        .arg_u32(weight.k)
        .arg_u32(weight.group as u32)
        .arg_u32(m)
        .launch(stream)
}
