// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in FlashInfer b12x fused-MoE prefill via `dlopen(libatlasb12x.so)` — behind
//! `ATLAS_LAGUNA_MOE_B12X=1`. Mirrors `gdn_flashinfer.rs` exactly (raw `dlopen`/`dlsym`,
//! `OnceLock<Option<Lib>>`, warn-and-fallback on any failure).
//!
//! The b12x CuTe-DSL kernel (SM120/SM121, NVFP4) replaces Atlas's ~6-launch grouped
//! MoE path (sort → gather → grouped gate_up GEMM → SwiGLU → grouped down GEMM →
//! unpermute_reduce) with ONE resident launch that fuses route/pack, FC1, SwiGLU,
//! FP4-requant, FC2, and scatter. The AOT export and C shim live in
//! `3rdparty_patches/b12x_aot/`; the shim owns a cached one-time workspace sized for
//! `atlas_b12x_max_tokens()` tokens.
//!
//! Geometry note (Laguna-S-2.1): the AOT kernel is frozen at H=3072, I=1024, E=256,
//! top_k=10. The FFI surface itself is geometry-agnostic (it marshals device pointers +
//! `num_tokens`); the frozen dims live in `b12x_dyn_0.geom.txt` / `libatlasb12x.so`.
//!
//! ## DETERMINISM WARNING (true under every parent-proof outcome — ships now)
//! b12x's scatter is a bf16×2 **atomic add** → the routed-expert summation order is
//! NON-DETERMINISTIC across runs, unlike Atlas's deterministic `moe_unpermute_reduce`.
//! Its SwiGLU uses an `rcp.approx` sigmoid + fast-math `exp`. Expect ~1e-3 elementwise
//! diffs vs the grouped path AND run-to-run wobble. Every A/B validation of this path is
//! TOLERANCE-based (cos ≥ 0.999, rel-L2 ≤ 2e-3) — NEVER bit-exact.
//!
//! dlopen (not link-time) keeps this fully opt-in: the binary builds and runs without the
//! library; it is only loaded when the flag is set. `ATLAS_B12X_LIB` overrides the path.
use anyhow::{Result, anyhow, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::OnceLock;

use crate::layers::moe::B12xMoeWeights;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

type LoadFn = unsafe extern "C" fn();
type MaxTokensFn = unsafe extern "C" fn() -> c_int;
// ABI frozen at parent P3 header inspection (see 3rdparty_patches/b12x_aot/b12x_shim.cpp).
// The dynamic export marshals via `make_ptr` pointer-fakes with NO GDN precedent — the
// 19-ptr/16-memref/5-i32 layout in the shim is the SSOT; this order must match it.
type PrefillFn = unsafe extern "C" fn(
    *mut c_void, // x_bf16          [num_tokens, hidden]
    *mut c_void, // topk_ids_i32    [num_tokens, top_k]
    *mut c_void, // topk_w_f32      [num_tokens, top_k]
    *mut c_void, // out_bf16        [num_tokens, hidden]
    *mut c_void, // w13_fp4         [E, 2I, H/2]
    *mut c_void, // w13_sf          swizzled
    *mut c_void, // w2_fp4          [E, H, I/2]
    *mut c_void, // w2_sf           swizzled
    *mut c_void, // w1_alpha        [E] f32
    *mut c_void, // w2_alpha        [E] f32
    *mut c_void, // fc2_gs          [E] f32
    c_int,       // num_tokens
    *mut c_void, // stream
) -> c_int;

struct Lib {
    prefill: PrefillFn,
    max_tokens: u32,
}
// SAFETY: the resolved fn pointers are process-global and immutable after load.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

static LIB: OnceLock<Option<Lib>> = OnceLock::new();

fn lib() -> Option<&'static Lib> {
    LIB.get_or_init(|| unsafe {
        let path =
            std::env::var("ATLAS_B12X_LIB").unwrap_or_else(|_| "libatlasb12x.so".to_string());
        let cpath = std::ffi::CString::new(path.clone()).ok()?;
        let h = dlopen(cpath.as_ptr(), RTLD_NOW);
        if h.is_null() {
            tracing::warn!(
                "ATLAS_LAGUNA_MOE_B12X: dlopen('{path}') failed — falling back to grouped"
            );
            return None;
        }
        let load = dlsym(h, c"atlas_b12x_load".as_ptr());
        let prefill = dlsym(h, c"atlas_b12x_moe_prefill".as_ptr());
        let max_tokens = dlsym(h, c"atlas_b12x_max_tokens".as_ptr());
        if load.is_null() || prefill.is_null() || max_tokens.is_null() {
            tracing::warn!(
                "ATLAS_LAGUNA_MOE_B12X: symbols not found in lib — falling back to grouped"
            );
            return None;
        }
        let load: LoadFn = std::mem::transmute(load);
        let max_tokens: MaxTokensFn = std::mem::transmute(max_tokens);
        load(); // load the cubin module onto the device(s) once
        let cap = max_tokens();
        if cap <= 0 {
            tracing::warn!(
                "ATLAS_LAGUNA_MOE_B12X: atlas_b12x_max_tokens()={cap} (export absent?) — grouped"
            );
            return None;
        }
        tracing::info!(
            "ATLAS_LAGUNA_MOE_B12X: FlashInfer b12x fused-MoE loaded (opt-in, max_tokens={cap}); \
             scatter is atomic-add (non-deterministic) — A/B tolerance-based, never bit-exact"
        );
        Some(Lib {
            prefill: std::mem::transmute::<*mut c_void, PrefillFn>(prefill),
            max_tokens: cap as u32,
        })
    })
    .as_ref()
}

/// True when `ATLAS_LAGUNA_MOE_B12X=1` AND the library + symbols loaded successfully.
pub(crate) fn available() -> bool {
    std::env::var("ATLAS_LAGUNA_MOE_B12X").as_deref() == Ok("1") && lib().is_some()
}

/// Token capacity the shim workspace was sized for (from `atlas_b12x_max_tokens()`).
/// `None` when the lib is unavailable. Prefills with `num_tokens` beyond this must
/// fall back to the grouped path (the shim returns nonzero above capacity).
pub(crate) fn max_tokens() -> Option<u32> {
    lib().map(|l| l.max_tokens)
}

// CUDA runtime: returns AND clears the last runtime error. Used to reset a stale
// pending error before the b12x MLIR launcher reads it (cudart is linked into the
// final `spark` binary).
unsafe extern "C" {
    fn cudaGetLastError() -> c_int;
}

/// Run one prefill through the b12x fused-MoE kernel, writing the routed-expert
/// result into `out` (bf16 `[num_tokens, hidden]`). Routing (`ids`/`weights`) is
/// Atlas's own renormalized `moe_topk_softmax_batched` output. Caller MUST gate on
/// [`available`] + all-experts-resident + `n <= max_tokens` (see `try_b12x_prefill`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn b12x_moe_prefill(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    ids: DevicePtr,
    weights: DevicePtr,
    out: DevicePtr,
    w: &B12xMoeWeights,
    n: u32,
    stream: u64,
) -> Result<()> {
    let l = lib().ok_or_else(|| anyhow!("FlashInfer b12x lib unavailable"))?;
    if n > l.max_tokens {
        bail!(
            "b12x_moe_prefill: n={n} exceeds shim capacity {}",
            l.max_tokens
        );
    }
    // The b12x kernel's MLIR launcher captures its own status via the CUDA
    // RUNTIME `cudaGetLastError()`. Atlas drives CUDA via the DRIVER API, so a
    // benign STALE runtime error left pending by a prior op is never cleared on
    // our side and the launcher returns it as a bogus `1` (cudaErrorInvalidValue)
    // even when the launch is fine. Drain the stream, then read+clear the pending
    // runtime error so the launcher starts from a clean slate.
    gpu.synchronize(stream)?;
    let pending = unsafe { cudaGetLastError() };
    if pending != 0 {
        tracing::warn!(
            "ATLAS_LAGUNA_MOE_B12X: cleared stale CUDA runtime error {pending} before b12x launch"
        );
    }
    let ret = unsafe {
        (l.prefill)(
            x.0 as *mut c_void,
            ids.0 as *mut c_void,
            weights.0 as *mut c_void,
            out.0 as *mut c_void,
            w.w13_fp4.0 as *mut c_void,
            w.w13_sf.0 as *mut c_void,
            w.w2_fp4.0 as *mut c_void,
            w.w2_sf.0 as *mut c_void,
            w.w1_alpha.0 as *mut c_void,
            w.w2_alpha.0 as *mut c_void,
            w.fc2_gs.0 as *mut c_void,
            n as c_int,
            stream as *mut c_void,
        )
    };
    if ret != 0 {
        bail!("atlas_b12x_moe_prefill returned {ret}");
    }
    Ok(())
}

#[cfg(test)]
#[path = "b12x_flashinfer_tests.rs"]
mod tests;
