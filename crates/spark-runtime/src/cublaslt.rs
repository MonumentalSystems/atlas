// SPDX-License-Identifier: AGPL-3.0-only
//! Minimal cuBLASLt FFI for the high-efficiency GEMM path (`ATLAS_CUBLAS_GEMM`).
//!
//! The hand-written mma.sync projection/MoE GEMMs reach only ~30% of the cuBLAS
//! ceiling on GB10 (measured: 32 vs 85 TFLOPS bf16, 152 fp8, on the SSM-qkvz
//! shape 3537×12288×2048). This routes those GEMMs through cuBLASLt instead.
//! BF16 only for now — correctness-clean (no scale-format issues); native fp8
//! block-scaled is the follow-up once the end-to-end win is proven.

use anyhow::{Result, bail};
use std::ffi::c_void;
use std::sync::OnceLock;

#[allow(non_camel_case_types)]
type cublasLtHandle_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulDesc_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatrixLayout_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulPreference_t = *mut c_void;

const CUDA_R_16BF: i32 = 14;
const CUDA_R_32F: i32 = 0;
const CUDA_R_8F_E4M3: i32 = 28;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const DESC_TRANSA: u32 = 3;
const DESC_TRANSB: u32 = 4;
const DESC_A_SCALE_POINTER: u32 = 17;
const DESC_B_SCALE_POINTER: u32 = 18;
const DESC_A_SCALE_MODE: u32 = 31;
const DESC_B_SCALE_MODE: u32 = 32;
const SCALE_MODE_OUTER_VEC_32F: i32 = 3;
const SCALE_MODE_VEC128_32F: i32 = 4;
const SCALE_MODE_BLK128X128_32F: i32 = 5;
const PREF_MAX_WORKSPACE_BYTES: u32 = 1;

unsafe extern "C" {
    fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> i32;
    fn cublasLtMatmulDescCreate(
        desc: *mut cublasLtMatmulDesc_t,
        compute_type: i32,
        scale_type: i32,
    ) -> i32;
    fn cublasLtMatmulDescSetAttribute(
        desc: cublasLtMatmulDesc_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> i32;
    fn cublasLtMatrixLayoutCreate(
        layout: *mut cublasLtMatrixLayout_t,
        dtype: i32,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> i32;
    fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> i32;
    fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> i32;
    fn cublasLtMatmulPreferenceSetAttribute(
        pref: cublasLtMatmulPreference_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoGetHeuristic(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        pref: cublasLtMatmulPreference_t,
        requested: i32,
        results: *mut c_void,
        returned: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmul(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        layout_a: cublasLtMatrixLayout_t,
        b: *const c_void,
        layout_b: cublasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        layout_c: cublasLtMatrixLayout_t,
        d: *mut c_void,
        layout_d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
}

struct Ctx {
    handle: cublasLtHandle_t,
    workspace: u64,
    ws_size: usize,
}
// cuBLASLt handle + device workspace are process-global; matmul is invoked
// serially from the single-threaded scheduler forward.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

static CTX: OnceLock<Ctx> = OnceLock::new();

fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    let mut handle: cublasLtHandle_t = std::ptr::null_mut();
    let st = unsafe { cublasLtCreate(&mut handle) };
    if st != 0 {
        bail!("cublasLtCreate failed: {st}");
    }
    let ws_size = 64 * 1024 * 1024;
    let mut ws: u64 = 0;
    let st = unsafe { cuMemAlloc_v2(&mut ws, ws_size) };
    if st != 0 {
        bail!("cuMemAlloc cuBLASLt workspace failed: {st}");
    }
    let _ = CTX.set(Ctx {
        handle,
        workspace: ws,
        ws_size,
    });
    Ok(CTX.get().unwrap())
}

fn chk(status: i32, what: &str) -> Result<()> {
    if status != 0 {
        bail!("cuBLASLt {what} failed: status {status}");
    }
    Ok(())
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all BF16 — the standard
/// projection GEMM (activation × transposed weight). Maps to cuBLASLt's
/// column-major convention as `D[N,M] = opT(weightᶜ[K,N]) · opN(actᶜ[K,M])`.
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ),
            "TRANSB",
        )?;
        // A = weight stored row-major [N,K] == col-major [K,N], ld=K, opT → [N,K]
        // B = act    stored row-major [M,K] == col-major [K,M], ld=K, opN → [K,M]
        // D = out    row-major [M,N]        == col-major [N,M], ld=N
        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_16BF, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_16BF, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        // cublasLtMatmulHeuristicResult_t = { algo[64B], workspaceSize, state,
        // wavesCount, reserved[4] } ≈ 96B; algo at offset 0. 128B for margin.
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc,
                la,
                lb,
                ld_,
                ld_,
                pref,
                1,
                result.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            la,
            act as *const c_void,
            lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_,
            out as *mut c_void,
            ld_,
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(())
}

/// Native FP8 (E4M3) `out[M,N] = act[M,K] @ weight[N,K]ᵀ` → BF16 with ROW-WISE
/// scaling (OUTER_VEC): per-output-row weight scale `weight_scale[N]` and
/// per-token activation scale `act_scale[M]`. This is the fp8 path GB10/sm_121
/// actually supports (128-block fp8 is B200-only). ~1.8× the bf16 path.
/// cuBLAS folds `A_scale[i]·B_scale[j]` into the FP32 epilogue; with D=[N,M],
/// i indexes weight rows (N) and j indexes tokens (M) — exactly row-wise.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_rowwise(
    act_fp8: u64,
    act_scale: u64,
    weight_fp8: u64,
    weight_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        let set = |attr: u32, val: *const c_void, sz: usize, what: &str| -> Result<()> {
            chk(cublasLtMatmulDescSetAttribute(desc, attr, val, sz), what)
        };
        set(DESC_TRANSA, &ta as *const i32 as *const c_void, 4, "TRANSA")?;
        set(DESC_TRANSB, &tb as *const i32 as *const c_void, 4, "TRANSB")?;
        let mode = SCALE_MODE_OUTER_VEC_32F;
        set(DESC_A_SCALE_MODE, &mode as *const i32 as *const c_void, 4, "A_SCALE_MODE")?;
        set(DESC_B_SCALE_MODE, &mode as *const i32 as *const c_void, 4, "B_SCALE_MODE")?;
        set(
            DESC_A_SCALE_POINTER,
            &weight_scale as *const u64 as *const c_void,
            8,
            "A_SCALE_POINTER",
        )?;
        set(DESC_B_SCALE_POINTER, &act_scale as *const u64 as *const c_void, 8, "B_SCALE_POINTER")?;

        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_8F_E4M3, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_8F_E4M3, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle, desc, la, lb, ld_, ld_, pref, 1,
                result.as_mut_ptr() as *mut c_void, &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt fp8 rowwise: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle, desc,
            &alpha as *const f32 as *const c_void,
            weight_fp8 as *const c_void, la,
            act_fp8 as *const c_void, lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void, ld_,
            out as *mut c_void, ld_,
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void, ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(())
}

/// Native FP8 (E4M3) `out[M,N] = act[M,K] @ weight[N,K]ᵀ` → BF16, with the
/// weight per-128×128-block FP32-scaled (matches Atlas's `Fp8Weight.row_scale`
/// layout exactly) and the activation cast at unit scale (post-RMSNorm acts sit
/// in fp8's range). ~1.8× the bf16 path (152 vs 85 TFLOPS on GB10).
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_blkscaled(
    act_fp8: u64,
    act_scale: u64,
    weight_fp8: u64,
    weight_block_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        let set = |attr: u32, val: *const c_void, sz: usize, what: &str| -> Result<()> {
            chk(cublasLtMatmulDescSetAttribute(desc, attr, val, sz), what)
        };
        set(DESC_TRANSA, &ta as *const i32 as *const c_void, 4, "TRANSA")?;
        set(DESC_TRANSB, &tb as *const i32 as *const c_void, 4, "TRANSB")?;
        // FP8 block scaling requires BOTH operands use a 128-block mode (SCALAR
        // is rejected → status 7). Weight = per-128×128 block, activation =
        // per-[token,128-of-K] VEC128 (DeepSeek block-fp8 scheme).
        let a_mode = SCALE_MODE_BLK128X128_32F;
        let b_mode = SCALE_MODE_VEC128_32F;
        set(DESC_A_SCALE_MODE, &a_mode as *const i32 as *const c_void, 4, "A_SCALE_MODE")?;
        set(DESC_B_SCALE_MODE, &b_mode as *const i32 as *const c_void, 4, "B_SCALE_MODE")?;
        set(
            DESC_A_SCALE_POINTER,
            &weight_block_scale as *const u64 as *const c_void,
            8,
            "A_SCALE_POINTER",
        )?;
        set(DESC_B_SCALE_POINTER, &act_scale as *const u64 as *const c_void, 8, "B_SCALE_POINTER")?;

        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_8F_E4M3, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_8F_E4M3, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle, desc, la, lb, ld_, ld_, pref, 1,
                result.as_mut_ptr() as *mut c_void, &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt fp8: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle, desc,
            &alpha as *const f32 as *const c_void,
            weight_fp8 as *const c_void, la,
            act_fp8 as *const c_void, lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void, ld_,
            out as *mut c_void, ld_,
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void, ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(())
}
