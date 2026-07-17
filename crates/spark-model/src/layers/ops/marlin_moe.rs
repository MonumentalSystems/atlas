// SPDX-License-Identifier: AGPL-3.0-only

//! Raw-pointer CUDA 13.2 Marlin NVFP4 MoE bridge.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::OnceLock;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;

type InitFn = unsafe extern "C" fn() -> c_int;
type RepackFn =
    unsafe extern "C" fn(*const c_void, *mut c_void, c_int, c_int, *mut c_void) -> c_int;
type AlignFn = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    c_int,
    *mut c_void,
) -> c_int;
type SiluMulFn =
    unsafe extern "C" fn(*const c_void, *mut c_void, c_int, c_int, *mut c_void) -> c_int;
type GemmFn = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *mut c_void,
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *mut c_void,
    *mut c_void,
) -> c_int;

struct Lib {
    repack: RepackFn,
    align: AlignFn,
    silu_mul: SiluMulFn,
    gemm: GemmFn,
}

// SAFETY: the dynamic library stays loaded for process lifetime and its function
// pointers are immutable after initialization.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

static LIB: OnceLock<Option<Lib>> = OnceLock::new();

fn lib() -> Option<&'static Lib> {
    LIB.get_or_init(|| unsafe {
        let path = std::env::var("ATLAS_MARLIN_MOE_LIB")
            .unwrap_or_else(|_| "libatlas_marlin_moe.so".to_string());
        let path_c = std::ffi::CString::new(path.clone()).ok()?;
        let handle = dlopen(path_c.as_ptr(), RTLD_NOW);
        if handle.is_null() {
            tracing::warn!("ATLAS_MOE_MARLIN: dlopen('{path}') failed; path disabled");
            return None;
        }
        let init = dlsym(handle, c"atlas_marlin_moe_init".as_ptr());
        let repack = dlsym(handle, c"atlas_marlin_moe_repack".as_ptr());
        let align = dlsym(handle, c"atlas_marlin_moe_align".as_ptr());
        let silu_mul = dlsym(handle, c"atlas_marlin_moe_silu_mul".as_ptr());
        let gemm = dlsym(handle, c"atlas_marlin_moe_gemm".as_ptr());
        if init.is_null()
            || repack.is_null()
            || align.is_null()
            || silu_mul.is_null()
            || gemm.is_null()
        {
            tracing::warn!("ATLAS_MOE_MARLIN: required symbols absent; path disabled");
            return None;
        }
        let init: InitFn = std::mem::transmute(init);
        let status = init();
        if status != 0 {
            tracing::warn!("ATLAS_MOE_MARLIN: CUDA initialization returned {status}");
            return None;
        }
        tracing::info!("ATLAS_MOE_MARLIN: CUDA 13.2 Marlin NVFP4 MoE bridge loaded from {path}");
        Some(Lib {
            repack: std::mem::transmute::<*mut c_void, RepackFn>(repack),
            align: std::mem::transmute::<*mut c_void, AlignFn>(align),
            silu_mul: std::mem::transmute::<*mut c_void, SiluMulFn>(silu_mul),
            gemm: std::mem::transmute::<*mut c_void, GemmFn>(gemm),
        })
    })
    .as_ref()
}

pub(crate) fn available() -> bool {
    std::env::var("ATLAS_MOE_MARLIN").as_deref() == Ok("1") && lib().is_some()
}

pub(crate) fn repack(src: DevicePtr, dst: DevicePtr, k: u32, n: u32, stream: u64) -> Result<()> {
    let Some(lib) = lib() else {
        bail!("ATLAS_MOE_MARLIN requested but library is unavailable");
    };
    let status = unsafe {
        (lib.repack)(
            src.0 as *const c_void,
            dst.0 as *mut c_void,
            k as c_int,
            n as c_int,
            stream as *mut c_void,
        )
    };
    if status != 0 {
        bail!("atlas_marlin_moe_repack returned {status}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn align(
    ids: DevicePtr,
    sorted: DevicePtr,
    experts: DevicePtr,
    padded: DevicePtr,
    tokens: u32,
    stream: u64,
) -> Result<()> {
    let Some(lib) = lib() else {
        bail!("ATLAS_MOE_MARLIN requested but library is unavailable");
    };
    let status = unsafe {
        (lib.align)(
            ids.0 as *const c_void,
            sorted.0 as *mut c_void,
            experts.0 as *mut c_void,
            padded.0 as *mut c_void,
            tokens as c_int,
            stream as *mut c_void,
        )
    };
    if status != 0 {
        bail!("atlas_marlin_moe_align returned {status}");
    }
    Ok(())
}

pub(crate) fn silu_mul(
    gate_up: DevicePtr,
    output: DevicePtr,
    routes: u32,
    intermediate: u32,
    stream: u64,
) -> Result<()> {
    let Some(lib) = lib() else {
        bail!("ATLAS_MOE_MARLIN requested but library is unavailable");
    };
    let status = unsafe {
        (lib.silu_mul)(
            gate_up.0 as *const c_void,
            output.0 as *mut c_void,
            routes as c_int,
            intermediate as c_int,
            stream as *mut c_void,
        )
    };
    if status != 0 {
        bail!("atlas_marlin_moe_silu_mul returned {status}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    a: DevicePtr,
    weights: DevicePtr,
    output: DevicePtr,
    reduce_tmp: DevicePtr,
    scales: DevicePtr,
    global_scales: DevicePtr,
    sorted: DevicePtr,
    experts: DevicePtr,
    padded: DevicePtr,
    topk_weights: DevicePtr,
    top_k: u32,
    mul_topk_weights: bool,
    m: u32,
    n: u32,
    k: u32,
    workspace: DevicePtr,
    stream: u64,
) -> Result<()> {
    let Some(lib) = lib() else {
        bail!("ATLAS_MOE_MARLIN requested but library is unavailable");
    };
    let status = unsafe {
        (lib.gemm)(
            a.0 as *const c_void,
            weights.0 as *const c_void,
            output.0 as *mut c_void,
            reduce_tmp.0 as *mut c_void,
            scales.0 as *const c_void,
            global_scales.0 as *const c_void,
            sorted.0 as *const c_void,
            experts.0 as *const c_void,
            padded.0 as *const c_void,
            topk_weights.0 as *const c_void,
            top_k as c_int,
            i32::from(mul_topk_weights),
            m as c_int,
            n as c_int,
            k as c_int,
            workspace.0 as *mut c_void,
            stream as *mut c_void,
        )
    };
    if status != 0 {
        bail!("atlas_marlin_moe_gemm returned {status}");
    }
    Ok(())
}
