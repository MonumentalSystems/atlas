// SPDX-License-Identifier: AGPL-3.0-only
//! Optional CUTLASS host-wrapper FFI for de-risking GB10 GEMM replacements.

use anyhow::{Result, bail};

#[cfg(atlas_cutlass)]
use std::ffi::c_void;
#[cfg(atlas_cutlass)]
use std::sync::OnceLock;

#[cfg(atlas_cutlass)]
unsafe extern "C" {
    fn atlas_cutlass_bf16_gemm_act_weight_t(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    fn atlas_cutlass_nvfp4_gemm_bf16_act_weight_t(
        act: *const c_void,
        weight_packed_t: *const c_void,
        weight_scale_t: *const c_void,
        weight_scale_2: f32,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    fn atlas_cutlass_pack_bf16_weight_to_nvfp4_t(
        weight_bf16: *const c_void,
        packed_t: *mut c_void,
        scale_t: *mut c_void,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    fn atlas_cutlass_transpose_nvfp4_packed_kton(
        src_packed_t: *const c_void,
        dst_packed: *mut c_void,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    fn atlas_cutlass_bf16_gemm_act_weight_t_128x256(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    fn atlas_cutlass_bf16_gemm_act_weight_t_256x128(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    fn atlas_cutlass_bf16_gemm_act_weight_t_64x128(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    fn atlas_cutlass_bf16_gemm_act_weight_t_128x64(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    fn atlas_cutlass_bf16_gemm_act_weight_t_64x64(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    fn atlas_cublaslt_bf16_gemm_act_weight_t_algo(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
        algo_index: i32,
        returned_count: *mut i32,
    ) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
}

#[cfg(atlas_cutlass)]
struct Ctx {
    workspace: u64,
    ws_size: usize,
}

#[cfg(atlas_cutlass)]
unsafe impl Send for Ctx {}
#[cfg(atlas_cutlass)]
unsafe impl Sync for Ctx {}

#[cfg(atlas_cutlass)]
static CTX: OnceLock<Ctx> = OnceLock::new();

#[cfg(atlas_cutlass)]
fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    let ws_size = 64 * 1024 * 1024;
    let mut workspace = 0u64;
    let status = unsafe { cuMemAlloc_v2(&mut workspace, ws_size) };
    if status != 0 {
        bail!("cuMemAlloc CUTLASS workspace failed: {status}");
    }
    let _ = CTX.set(Ctx { workspace, ws_size });
    Ok(CTX.get().unwrap())
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]^T`, all BF16.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            atlas_cutlass_bf16_gemm_act_weight_t(
                act as *const c_void,
                weight as *const c_void,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS bf16 GEMM failed: status {status} for {m}x{n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (act, weight, out, m, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// Native CUTLASS NVFP4 dense projection:
/// `out[M,N] = quant_nvfp4(act[M,K]) @ weight_t[N,K]^T -> BF16`.
///
/// `weight_packed_t` and `weight_scale_t` are Atlas's transposed NVFP4
/// prefill layout: packed data `[K/2,N]`, scales `[K/16,N]`. The wrapper
/// repacks activation and scale tensors into CUTLASS's SM120 blockscaled
/// layouts in the shared CUTLASS workspace before dispatch.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_bf16_act_weight_t(
    act: u64,
    weight_packed_t: u64,
    weight_scale_t: u64,
    weight_scale_2: f32,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            atlas_cutlass_nvfp4_gemm_bf16_act_weight_t(
                act as *const c_void,
                weight_packed_t as *const c_void,
                weight_scale_t as *const c_void,
                weight_scale_2,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS nvfp4 GEMM failed: status {status} for {m}x{n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (act, weight_packed_t, weight_scale_t, weight_scale_2, out, m, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// Pack BF16 row-major weight `[N,K]` into Atlas transposed NVFP4 layout:
/// packed `[K/2,N]` and E4M3 scales `[K/16,N]`. `weight_scale_2` is assumed
/// to be 1.0 by the caller when feeding this into the native CUTLASS wrapper.
pub fn pack_bf16_weight_to_nvfp4_t(
    weight_bf16: u64,
    packed_t: u64,
    scale_t: u64,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let status = unsafe {
            atlas_cutlass_pack_bf16_weight_to_nvfp4_t(
                weight_bf16 as *const c_void,
                packed_t as *mut c_void,
                scale_t as *mut c_void,
                n as i32,
                k as i32,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS BF16->NVFP4 weight pack failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (weight_bf16, packed_t, scale_t, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// Transpose an Atlas-packed NVFP4 weight from the checkpoint/hand-kernel
/// `[K/2, N]` layout into CUTLASS's `[N, K/2]` layout (the byte order the
/// native NVFP4 GEMM consumes for the ColumnMajor B operand). Pure byte
/// transpose; nibble pairing within each byte is preserved. `dst_packed` must
/// have `N * K/2` bytes.
pub fn transpose_nvfp4_packed_kton(
    src_packed_t: u64,
    dst_packed: u64,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let status = unsafe {
            atlas_cutlass_transpose_nvfp4_packed_kton(
                src_packed_t as *const c_void,
                dst_packed as *mut c_void,
                n as i32,
                k as i32,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS NVFP4 weight transpose failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (src_packed_t, dst_packed, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

#[cfg(all(test, atlas_cutlass))]
mod tests {
    use super::*;
    use std::ffi::c_void;

    const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

    unsafe extern "C" {
        fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFree(ptr: *mut c_void) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
        fn cudaDeviceSynchronize() -> i32;
    }

    fn f32_to_bf16(x: f32) -> u16 {
        let bits = x.to_bits();
        let lsb = (bits >> 16) & 1;
        ((bits + 0x7fff + lsb) >> 16) as u16
    }

    fn bf16_to_f32(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }

    fn cuda_check(status: i32, what: &str) {
        assert_eq!(status, 0, "CUDA {what} failed: {status}");
    }

    unsafe fn device_alloc(bytes: usize) -> *mut c_void {
        let mut ptr = std::ptr::null_mut();
        cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) }, "malloc");
        ptr
    }

    unsafe fn copy_h2d<T>(dst: *mut c_void, src: &[T]) {
        cuda_check(
            unsafe {
                cudaMemcpy(
                    dst,
                    src.as_ptr() as *const c_void,
                    std::mem::size_of_val(src),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "copy h2d",
        );
    }

    unsafe fn copy_d2h<T>(dst: &mut [T], src: *const c_void) {
        cuda_check(
            unsafe {
                cudaMemcpy(
                    dst.as_mut_ptr() as *mut c_void,
                    src,
                    std::mem::size_of_val(dst),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "copy d2h",
        );
    }

    type CutlassVariant = unsafe extern "C" fn(
        *const c_void,
        *const c_void,
        *mut c_void,
        i32,
        i32,
        i32,
        *mut c_void,
        usize,
        *mut c_void,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn run_cutlass_variant(
        name: &str,
        f: CutlassVariant,
        act: *mut c_void,
        weight: *mut c_void,
        out: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let ctx = ctx()?;
        let status = unsafe {
            f(
                act,
                weight,
                out,
                m as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                std::ptr::null_mut(),
            )
        };
        if status != 0 {
            bail!("CUTLASS variant {name} failed: status {status} for {m}x{n}x{k}");
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_cublaslt_algo(
        algo_index: i32,
        act: *mut c_void,
        weight: *mut c_void,
        out: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<i32> {
        let ctx = ctx()?;
        let mut returned = 0i32;
        let status = unsafe {
            atlas_cublaslt_bf16_gemm_act_weight_t_algo(
                act,
                weight,
                out,
                m as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                std::ptr::null_mut(),
                algo_index,
                &mut returned,
            )
        };
        if status != 0 {
            bail!(
                "cuBLASLt algo {algo_index} failed: status {status} returned={returned} for {m}x{n}x{k}"
            );
        }
        Ok(returned)
    }

    #[test]
    #[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
    fn cutlass_bf16_ffi_smoke_computes_row_major_act_by_weight_t() {
        const M: usize = 128;
        const N: usize = 128;
        const K: usize = 32;

        let act_f32: Vec<f32> = (0..M * K)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.125)
            .collect();
        let weight_f32: Vec<f32> = (0..N * K).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let act: Vec<u16> = act_f32.iter().copied().map(f32_to_bf16).collect();
        let weight: Vec<u16> = weight_f32.iter().copied().map(f32_to_bf16).collect();
        let mut out = vec![0u16; M * N];

        let act_dev;
        let weight_dev;
        let out_dev;
        unsafe {
            act_dev = device_alloc(act.len() * 2);
            weight_dev = device_alloc(weight.len() * 2);
            out_dev = device_alloc(out.len() * 2);
            copy_h2d(act_dev, &act);
            copy_h2d(weight_dev, &weight);
        }

        let result = bf16_gemm_act_weight_t(
            act_dev as u64,
            weight_dev as u64,
            out_dev as u64,
            M as u32,
            N as u32,
            K as u32,
            0,
        );
        assert!(result.is_ok(), "{result:?}");

        unsafe {
            cuda_check(cudaDeviceSynchronize(), "device synchronize");
            copy_d2h(&mut out, out_dev);
            cuda_check(cudaFree(act_dev), "free act");
            cuda_check(cudaFree(weight_dev), "free weight");
            cuda_check(cudaFree(out_dev), "free out");
        }

        for m in 0..M {
            for n in 0..N {
                let mut expected = 0.0f32;
                for k in 0..K {
                    expected += bf16_to_f32(act[m * K + k]) * bf16_to_f32(weight[n * K + k]);
                }
                let actual = bf16_to_f32(out[m * N + n]);
                assert!(
                    (actual - expected).abs() < 0.025,
                    "m={m} n={n} actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
    fn cutlass_bf16_ffi_computes_holo_ssm_qkvz_shape() {
        const M: usize = 3537;
        const N: usize = 12288;
        const K: usize = 2048;

        let mut act = vec![0u16; M * K];
        for m in 0..M {
            act[m * K + (m % K)] = f32_to_bf16(1.0);
        }
        let weight: Vec<u16> = (0..N * K)
            .map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.0625))
            .collect();
        let mut out = vec![0u16; M * N];

        let act_dev;
        let weight_dev;
        let out_dev;
        unsafe {
            act_dev = device_alloc(act.len() * 2);
            weight_dev = device_alloc(weight.len() * 2);
            out_dev = device_alloc(out.len() * 2);
            copy_h2d(act_dev, &act);
            copy_h2d(weight_dev, &weight);
        }

        let result = bf16_gemm_act_weight_t(
            act_dev as u64,
            weight_dev as u64,
            out_dev as u64,
            M as u32,
            N as u32,
            K as u32,
            0,
        );
        assert!(result.is_ok(), "{result:?}");

        unsafe {
            cuda_check(cudaDeviceSynchronize(), "device synchronize");
            copy_d2h(&mut out, out_dev);
            cuda_check(cudaFree(act_dev), "free act");
            cuda_check(cudaFree(weight_dev), "free weight");
            cuda_check(cudaFree(out_dev), "free out");
        }

        for m in 0..M {
            let selected_k = m % K;
            for n in 0..N {
                let actual = bf16_to_f32(out[m * N + n]);
                let expected = bf16_to_f32(weight[n * K + selected_k]);
                assert_eq!(
                    actual, expected,
                    "m={m} n={n} selected_k={selected_k} actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
    fn cutlass_bf16_holo_qkvz_bench_against_cublaslt() {
        const ITERS: usize = 100;

        let shapes = [
            ("ssm_qkvz", 3537usize, 12288usize, 2048usize),
            ("ssm_out", 3537, 2048, 4096),
            ("attn_q", 3537, 8192, 2048),
            ("attn_k", 3537, 512, 2048),
            ("attn_v", 3537, 512, 2048),
            ("attn_o", 3537, 2048, 4096),
            ("moe_gate_up_dense", 28296, 1024, 2048),
            ("moe_down_dense", 28296, 2048, 512),
        ];

        let variants: [(&str, CutlassVariant); 6] = [
            ("128x128", atlas_cutlass_bf16_gemm_act_weight_t),
            ("128x256", atlas_cutlass_bf16_gemm_act_weight_t_128x256),
            ("256x128", atlas_cutlass_bf16_gemm_act_weight_t_256x128),
            ("64x128", atlas_cutlass_bf16_gemm_act_weight_t_64x128),
            ("128x64", atlas_cutlass_bf16_gemm_act_weight_t_128x64),
            ("64x64", atlas_cutlass_bf16_gemm_act_weight_t_64x64),
        ];

        for (name, m, n, k) in shapes {
            let act: Vec<u16> = (0..m * k)
                .map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.03125))
                .collect();
            let weight: Vec<u16> = (0..n * k)
                .map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.0625))
                .collect();

            let act_dev;
            let weight_dev;
            let cutlass_out;
            let cublas_out;
            unsafe {
                act_dev = device_alloc(act.len() * 2);
                weight_dev = device_alloc(weight.len() * 2);
                cutlass_out = device_alloc(m * n * 2);
                cublas_out = device_alloc(m * n * 2);
                copy_h2d(act_dev, &act);
                copy_h2d(weight_dev, &weight);
            }

            crate::cublaslt::bf16_gemm_act_weight_t(
                act_dev as u64,
                weight_dev as u64,
                cublas_out as u64,
                m as u32,
                n as u32,
                k as u32,
                0,
            )
            .unwrap();
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "warmup synchronize");
            }

            let mut best_variant = "";
            let mut best_cutlass_ms = f64::INFINITY;
            for (variant, f) in variants {
                run_cutlass_variant(variant, f, act_dev, weight_dev, cutlass_out, m, n, k).unwrap();
                unsafe {
                    cuda_check(cudaDeviceSynchronize(), "cutlass warmup synchronize");
                }
                let t0 = std::time::Instant::now();
                for _ in 0..ITERS {
                    run_cutlass_variant(variant, f, act_dev, weight_dev, cutlass_out, m, n, k)
                        .unwrap();
                }
                unsafe {
                    cuda_check(cudaDeviceSynchronize(), "cutlass synchronize");
                }
                let cutlass_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
                if cutlass_ms < best_cutlass_ms {
                    best_cutlass_ms = cutlass_ms;
                    best_variant = variant;
                }
            }

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                crate::cublaslt::bf16_gemm_act_weight_t(
                    act_dev as u64,
                    weight_dev as u64,
                    cublas_out as u64,
                    m as u32,
                    n as u32,
                    k as u32,
                    0,
                )
                .unwrap();
            }
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "cublas synchronize");
            }
            let cublas_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

            let flop = 2.0 * m as f64 * n as f64 * k as f64;
            eprintln!(
                "HOLO_DENSE_BENCH {name} M={m} N={n} K={k} iters={ITERS} best_cutlass={best_variant} cutlass_ms={best_cutlass_ms:.3} cutlass_tflops={:.1} cublaslt_ms={cublas_ms:.3} cublaslt_tflops={:.1} speedup_vs_cublas={:.3}",
                flop / (best_cutlass_ms / 1000.0) / 1.0e12,
                flop / (cublas_ms / 1000.0) / 1.0e12,
                cublas_ms / best_cutlass_ms
            );

            unsafe {
                cuda_check(cudaFree(act_dev), "free act");
                cuda_check(cudaFree(weight_dev), "free weight");
                cuda_check(cudaFree(cutlass_out), "free cutlass out");
                cuda_check(cudaFree(cublas_out), "free cublas out");
            }
        }
    }

    #[test]
    #[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
    fn cutlass_bf16_holo_decode_route_batch_shapes() {
        const ITERS: usize = 2000;
        let shapes = [
            ("moe_gate_up_routes_c1", 8usize, 1024usize, 2048usize),
            ("moe_gate_up_routes_c2", 16, 1024, 2048),
            ("moe_gate_up_routes_c4", 32, 1024, 2048),
            ("moe_gate_up_routes_c8", 64, 1024, 2048),
            ("moe_gate_up_routes_c16", 128, 1024, 2048),
            ("moe_down_routes_c1", 8, 2048, 512),
            ("moe_down_routes_c2", 16, 2048, 512),
            ("moe_down_routes_c4", 32, 2048, 512),
            ("moe_down_routes_c8", 64, 2048, 512),
            ("moe_down_routes_c16", 128, 2048, 512),
        ];
        let variants: [(&str, CutlassVariant); 6] = [
            ("128x128", atlas_cutlass_bf16_gemm_act_weight_t),
            ("128x256", atlas_cutlass_bf16_gemm_act_weight_t_128x256),
            ("256x128", atlas_cutlass_bf16_gemm_act_weight_t_256x128),
            ("64x128", atlas_cutlass_bf16_gemm_act_weight_t_64x128),
            ("128x64", atlas_cutlass_bf16_gemm_act_weight_t_128x64),
            ("64x64", atlas_cutlass_bf16_gemm_act_weight_t_64x64),
        ];

        for (name, m, n, k) in shapes {
            let act: Vec<u16> = (0..m * k)
                .map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.03125))
                .collect();
            let weight: Vec<u16> = (0..n * k)
                .map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.0625))
                .collect();

            let act_dev;
            let weight_dev;
            let cutlass_out;
            let cublas_out;
            unsafe {
                act_dev = device_alloc(act.len() * 2);
                weight_dev = device_alloc(weight.len() * 2);
                cutlass_out = device_alloc(m * n * 2);
                cublas_out = device_alloc(m * n * 2);
                copy_h2d(act_dev, &act);
                copy_h2d(weight_dev, &weight);
            }

            crate::cublaslt::bf16_gemm_act_weight_t(
                act_dev as u64,
                weight_dev as u64,
                cublas_out as u64,
                m as u32,
                n as u32,
                k as u32,
                0,
            )
            .unwrap();
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "warmup synchronize");
            }

            let mut best_variant = "";
            let mut best_cutlass_ms = f64::INFINITY;
            for (variant, f) in variants {
                run_cutlass_variant(variant, f, act_dev, weight_dev, cutlass_out, m, n, k).unwrap();
                unsafe {
                    cuda_check(cudaDeviceSynchronize(), "cutlass warmup synchronize");
                }
                let t0 = std::time::Instant::now();
                for _ in 0..ITERS {
                    run_cutlass_variant(variant, f, act_dev, weight_dev, cutlass_out, m, n, k)
                        .unwrap();
                }
                unsafe {
                    cuda_check(cudaDeviceSynchronize(), "cutlass synchronize");
                }
                let cutlass_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
                if cutlass_ms < best_cutlass_ms {
                    best_cutlass_ms = cutlass_ms;
                    best_variant = variant;
                }
            }

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                crate::cublaslt::bf16_gemm_act_weight_t(
                    act_dev as u64,
                    weight_dev as u64,
                    cublas_out as u64,
                    m as u32,
                    n as u32,
                    k as u32,
                    0,
                )
                .unwrap();
            }
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "cublas synchronize");
            }
            let cublas_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

            let flop = 2.0 * m as f64 * n as f64 * k as f64;
            eprintln!(
                "HOLO_ROUTE_BATCH_BENCH {name} M={m} N={n} K={k} iters={ITERS} best_cutlass={best_variant} cutlass_us={:.3} cutlass_tflops={:.1} cublaslt_us={:.3} cublaslt_tflops={:.1} speedup_vs_cublas={:.3}",
                best_cutlass_ms * 1000.0,
                flop / (best_cutlass_ms / 1000.0) / 1.0e12,
                cublas_ms * 1000.0,
                flop / (cublas_ms / 1000.0) / 1.0e12,
                cublas_ms / best_cutlass_ms
            );

            unsafe {
                cuda_check(cudaFree(act_dev), "free act");
                cuda_check(cudaFree(weight_dev), "free weight");
                cuda_check(cudaFree(cutlass_out), "free cutlass out");
                cuda_check(cudaFree(cublas_out), "free cublas out");
            }
        }
    }

    #[test]
    #[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
    fn cublaslt_bf16_holo_route_batch_algo_sweep() {
        const ITERS: usize = 2000;
        let shapes = [
            ("gate_up_c1", 8usize, 1024usize, 2048usize),
            ("gate_up_c2", 16, 1024, 2048),
            ("gate_up_c4", 32, 1024, 2048),
            ("gate_up_c8", 64, 1024, 2048),
            ("gate_up_c16", 128, 1024, 2048),
            ("down_c1", 8, 2048, 512),
            ("down_c2", 16, 2048, 512),
            ("down_c4", 32, 2048, 512),
            ("down_c8", 64, 2048, 512),
            ("down_c16", 128, 2048, 512),
        ];

        for (name, m, n, k) in shapes {
            let act: Vec<u16> = (0..m * k)
                .map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.03125))
                .collect();
            let weight: Vec<u16> = (0..n * k)
                .map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.0625))
                .collect();
            let act_dev;
            let weight_dev;
            let out_dev;
            unsafe {
                act_dev = device_alloc(act.len() * 2);
                weight_dev = device_alloc(weight.len() * 2);
                out_dev = device_alloc(m * n * 2);
                copy_h2d(act_dev, &act);
                copy_h2d(weight_dev, &weight);
            }

            let returned = run_cublaslt_algo(0, act_dev, weight_dev, out_dev, m, n, k).unwrap();
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "algo warmup synchronize");
            }
            let mut best_algo = 0;
            let mut best_ms = f64::INFINITY;
            for algo in 0..returned.min(16) {
                if run_cublaslt_algo(algo, act_dev, weight_dev, out_dev, m, n, k).is_err() {
                    continue;
                }
                unsafe {
                    cuda_check(cudaDeviceSynchronize(), "algo warmup synchronize");
                }
                let t0 = std::time::Instant::now();
                let mut ok = true;
                for _ in 0..ITERS {
                    if run_cublaslt_algo(algo, act_dev, weight_dev, out_dev, m, n, k).is_err() {
                        ok = false;
                        break;
                    }
                }
                unsafe {
                    cuda_check(cudaDeviceSynchronize(), "algo synchronize");
                }
                if ok {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
                    if ms < best_ms {
                        best_ms = ms;
                        best_algo = algo;
                    }
                }
            }
            let flop = 2.0 * m as f64 * n as f64 * k as f64;
            eprintln!(
                "HOLO_CUBLASLT_ALGO_BENCH {name} M={m} N={n} K={k} returned={returned} best_algo={best_algo} best_us={:.3} best_tflops={:.1}",
                best_ms * 1000.0,
                flop / (best_ms / 1000.0) / 1.0e12
            );
            unsafe {
                cuda_check(cudaFree(act_dev), "free act");
                cuda_check(cudaFree(weight_dev), "free weight");
                cuda_check(cudaFree(out_dev), "free out");
            }
        }
    }

    // ---- NVFP4 op-level numeric comparator ---------------------------------
    //
    // Goal: decide whether the corrupt-output native-NVFP4 prefill path is a
    // wrapper layout/scale BUG (fixable) or inherent W4A4 quantization LOSS
    // (abandon on the sensitive projections). For each Holo projection shape we
    // compare three GEMM results over the SAME inputs:
    //   - out_cutlass: the native CUTLASS NVFP4 kernel (act packed in-wrapper,
    //     weight = our packed_t/scale_t).
    //   - out_ref:     a host W4A4 dequant reference. The weight side reads the
    //     EXACT packed_t nibbles + scale_t (e4m3) bytes the kernel consumes, so
    //     it is bit-faithful to the kernel's weight operand; the activation side
    //     replicates the wrapper's per-16-group max/6 -> e2m1 quantizer.
    //   - out_true:    the full unquantized BF16 GEMM.
    // Interpretation:
    //   cos(cutlass,ref) ~ 1  -> kernel + layouts are CORRECT; any divergence
    //       from out_true is inherent W4A4 loss (see cos(ref,true)).
    //   cos(cutlass,ref) low  -> wrapper layout/scale BUG; the kernel is not
    //       computing what its packed operands imply.

    /// E2M1 (FP4) level magnitudes, indexed by the low 3 bits; bit 3 is sign.
    const E2M1_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    fn decode_e2m1(nib: u8) -> f32 {
        let mag = E2M1_LEVELS[(nib & 0x7) as usize];
        if nib & 0x8 != 0 { -mag } else { mag }
    }

    /// Matches the device `float_to_e2m1` round-to-nearest-bin in the wrapper.
    fn f32_to_e2m1(x: f32) -> u8 {
        let sign = if x < 0.0 { 0x8u8 } else { 0 };
        let ax = x.abs();
        let mag = if ax <= 0.25 {
            0
        } else if ax <= 0.75 {
            1
        } else if ax <= 1.25 {
            2
        } else if ax <= 1.75 {
            3
        } else if ax <= 2.5 {
            4
        } else if ax <= 3.5 {
            5
        } else if ax <= 5.0 {
            6
        } else {
            7
        };
        sign | mag
    }

    /// Decode an OCP E4M3 byte (the format `__nv_fp8_e4m3` stores the weight
    /// per-group scale in) to f32. Scales are always positive here.
    fn e4m3_to_f32(byte: u8) -> f32 {
        let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = ((byte >> 3) & 0x0f) as i32;
        let m = (byte & 0x07) as i32;
        let val = if e == 0 {
            // subnormal: m/8 * 2^(1-7)
            (m as f32 / 8.0) * 2f32.powi(1 - 7)
        } else {
            // normal: (1 + m/8) * 2^(e-7)
            (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
        };
        sign * val
    }

    /// Deterministic, full-rank-ish pseudo-random value in roughly [-0.5, 0.5].
    fn gen_val(seed: u64) -> f32 {
        let mut x = seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0x1234_5678_9ABC_DEF0);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        ((x >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    #[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
    fn cutlass_nvfp4_projection_numeric_comparator() {
        // Small M faithfully exercises the N/K-dependent weight scale-factor
        // layout (the suspected bug correlates with large N) while keeping the
        // host reference GEMM cheap.
        const M: usize = 128;
        let shapes = [
            ("ssm_qkvz", 12288usize, 2048usize),
            ("attn_q", 8192, 2048),
            ("attn_kv", 512, 2048),
            ("attn_o", 2048, 4096),
        ];

        for (name, n, k) in shapes {
            assert_eq!(k % 16, 0, "{name}: K must be a multiple of 16");

            // Host inputs (bf16, as the device sees them).
            let weight_bf16: Vec<u16> = (0..n * k)
                .map(|i| f32_to_bf16(gen_val(i as u64) * 0.2))
                .collect();
            let act_bf16: Vec<u16> = (0..M * k)
                .map(|i| f32_to_bf16(gen_val((i as u64) ^ 0xA5A5_0000_0000) * 2.0))
                .collect();

            let packed_len = (k / 2) * n; // [K/2, N] u8
            let scale_len = (k / 16) * n; // [K/16, N] u8

            let weight_dev;
            let act_dev;
            let packed_dev;
            let scale_dev;
            let out_dev;
            unsafe {
                weight_dev = device_alloc(weight_bf16.len() * 2);
                act_dev = device_alloc(act_bf16.len() * 2);
                packed_dev = device_alloc(packed_len);
                scale_dev = device_alloc(scale_len);
                out_dev = device_alloc(M * n * 2);
                copy_h2d(weight_dev, &weight_bf16);
                copy_h2d(act_dev, &act_bf16);
            }

            // Pack the weight to Atlas transposed NVFP4 exactly as the runtime does.
            pack_bf16_weight_to_nvfp4_t(
                weight_dev as u64,
                packed_dev as u64,
                scale_dev as u64,
                n as u32,
                k as u32,
                0,
            )
            .unwrap();
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "pack synchronize");
            }

            // Native CUTLASS NVFP4 GEMM (weight_scale_2 = 1.0 for this pack).
            nvfp4_gemm_bf16_act_weight_t(
                act_dev as u64,
                packed_dev as u64,
                scale_dev as u64,
                1.0,
                out_dev as u64,
                M as u32,
                n as u32,
                k as u32,
                0,
            )
            .unwrap();
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "cutlass gemm synchronize");
            }

            let mut out_cutlass_bf16 = vec![0u16; M * n];
            let mut packed = vec![0u8; packed_len];
            let mut scale = vec![0u8; scale_len];
            unsafe {
                copy_d2h(&mut out_cutlass_bf16, out_dev);
                copy_d2h(&mut packed, packed_dev);
                copy_d2h(&mut scale, scale_dev);
                cuda_check(cudaFree(weight_dev), "free weight");
                cuda_check(cudaFree(act_dev), "free act");
                cuda_check(cudaFree(packed_dev), "free packed");
                cuda_check(cudaFree(scale_dev), "free scale");
                cuda_check(cudaFree(out_dev), "free out");
            }

            // Weight dequant, bit-faithful to the kernel's operand: read the same
            // packed nibbles + e4m3 group scales the kernel consumes. The pack
            // kernel emits CUTLASS [N,K/2] layout (K-contiguous): byte for
            // (n=col,k) is col*(K/2) + k/2, nibble = k&1. Scales stay [K/16,N].
            let mut w_q = vec![0f32; n * k];
            let mut w_true = vec![0f32; n * k];
            for col in 0..n {
                for kk in 0..k {
                    let g = kk / 16;
                    let byte = packed[col * (k / 2) + kk / 2];
                    let nib = if kk % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    let s = e4m3_to_f32(scale[g * n + col]);
                    w_q[col * k + kk] = decode_e2m1(nib) * s;
                    w_true[col * k + kk] = bf16_to_f32(weight_bf16[col * k + kk]);
                }
            }

            // Activation dequant, replicating the wrapper's per-16-group quantizer.
            let mut a_q = vec![0f32; M * k];
            let mut a_true = vec![0f32; M * k];
            for m in 0..M {
                for g in 0..(k / 16) {
                    let base = g * 16;
                    let mut max_abs = 0.0f32;
                    for i in 0..16 {
                        let v = bf16_to_f32(act_bf16[m * k + base + i]);
                        max_abs = max_abs.max(v.abs());
                    }
                    let s = if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 };
                    let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                    for i in 0..16 {
                        let v = bf16_to_f32(act_bf16[m * k + base + i]);
                        let nib = f32_to_e2m1(v * inv);
                        a_q[m * k + base + i] = decode_e2m1(nib) * s;
                        a_true[m * k + base + i] = v;
                    }
                }
            }

            // Reference GEMMs: out[m,n] = sum_k a[m,k] * w[n,k].
            let mut out_ref = vec![0f32; M * n];
            let mut out_true = vec![0f32; M * n];
            let mut out_cutlass = vec![0f32; M * n];
            for m in 0..M {
                for col in 0..n {
                    let mut acc_ref = 0.0f32;
                    let mut acc_true = 0.0f32;
                    for kk in 0..k {
                        acc_ref += a_q[m * k + kk] * w_q[col * k + kk];
                        acc_true += a_true[m * k + kk] * w_true[col * k + kk];
                    }
                    out_ref[m * n + col] = acc_ref;
                    out_true[m * n + col] = acc_true;
                    out_cutlass[m * n + col] = bf16_to_f32(out_cutlass_bf16[m * n + col]);
                }
            }

            let cos_cr = cosine(&out_cutlass, &out_ref);
            let cos_ct = cosine(&out_cutlass, &out_true);
            let cos_rt = cosine(&out_ref, &out_true);
            let max_abs_cr = out_cutlass
                .iter()
                .zip(&out_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let ref_rms =
                (out_ref.iter().map(|x| (x * x) as f64).sum::<f64>() / out_ref.len() as f64).sqrt();

            let verdict = if cos_cr > 0.999 {
                "KERNEL OK (cutlass matches W4A4 ref) -> divergence from true is inherent W4A4 loss"
            } else if cos_cr > 0.95 {
                "SUSPECT (minor cutlass<->ref drift; check scale rounding)"
            } else {
                "BUG (cutlass does NOT match its own packed operands -> layout/scale wrong)"
            };

            eprintln!(
                "NVFP4_COMPARATOR {name} M={M} N={n} K={k} \
                 cos(cutlass,ref)={cos_cr:.6} cos(cutlass,true)={cos_ct:.6} \
                 cos(ref,true)={cos_rt:.6} max_abs(cutlass-ref)={max_abs_cr:.5} \
                 ref_rms={ref_rms:.5} => {verdict}"
            );
        }
    }

    #[test]
    #[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
    fn cutlass_nvfp4_transpose_is_bit_exact() {
        // Validate the [K/2,N] -> [N,K/2] packed-weight transpose used by the
        // native-checkpoint path (cutlass_nvfp4_proj). Build a golden [N,K/2]
        // pack from a bf16 weight, derive the [K/2,N] "checkpoint" form by
        // transposing on host, run the device transpose, and require it to
        // reproduce the golden pack byte-for-byte.
        const N: usize = 512;
        const K: usize = 256;
        let half = K / 2;

        let weight_bf16: Vec<u16> = (0..N * K)
            .map(|i| f32_to_bf16(gen_val(i as u64) * 0.2))
            .collect();

        // Golden [N,K/2] pack via the (fixed) pack kernel.
        let weight_dev;
        let golden_dev;
        let scale_dev;
        unsafe {
            weight_dev = device_alloc(weight_bf16.len() * 2);
            golden_dev = device_alloc(N * half);
            scale_dev = device_alloc((K / 16) * N);
            copy_h2d(weight_dev, &weight_bf16);
        }
        pack_bf16_weight_to_nvfp4_t(
            weight_dev as u64,
            golden_dev as u64,
            scale_dev as u64,
            N as u32,
            K as u32,
            0,
        )
        .unwrap();
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "pack synchronize");
        }
        let mut golden = vec![0u8; N * half];
        unsafe {
            copy_d2h(&mut golden, golden_dev);
        }

        // Host-derived [K/2,N] checkpoint form: src[h*N + c] = golden[c*half + h].
        let mut checkpoint = vec![0u8; half * N];
        for c in 0..N {
            for h in 0..half {
                checkpoint[h * N + c] = golden[c * half + h];
            }
        }

        let src_dev;
        let dst_dev;
        unsafe {
            src_dev = device_alloc(checkpoint.len());
            dst_dev = device_alloc(N * half);
            copy_h2d(src_dev, &checkpoint);
        }
        transpose_nvfp4_packed_kton(src_dev as u64, dst_dev as u64, N as u32, K as u32, 0).unwrap();
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "transpose synchronize");
        }
        let mut got = vec![0u8; N * half];
        unsafe {
            copy_d2h(&mut got, dst_dev);
            cuda_check(cudaFree(weight_dev), "free weight");
            cuda_check(cudaFree(golden_dev), "free golden");
            cuda_check(cudaFree(scale_dev), "free scale");
            cuda_check(cudaFree(src_dev), "free src");
            cuda_check(cudaFree(dst_dev), "free dst");
        }

        assert_eq!(got, golden, "device transpose must reproduce the golden [N,K/2] pack");
        eprintln!("NVFP4_TRANSPOSE bit-exact over N={N} K={K} ({} bytes) OK", N * half);
    }
}
