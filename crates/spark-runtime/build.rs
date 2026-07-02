// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    println!("cargo:rerun-if-env-changed=ATLAS_CUDA_ARCH");
    // Register the `atlas_scale` cfg so `#[cfg(atlas_scale)]` does not trip
    // the `unexpected_cfgs` lint. `atlas_scale` selects SCALE/AMD (gfx1151)
    // codepaths over NVIDIA ones where the CUDA driver ABI differs — e.g.
    // SCALE's libcuda exports `cuGraphInstantiate` (not the NVIDIA-only
    // `cuGraphInstantiateWithFlags`). Driven by the same `ATLAS_TARGET_HW`
    // signal the atlas-kernels build uses; covers both the SCALE (`strix`)
    // and native-HIP (`strix-hip`) AMD targets.
    println!("cargo:rustc-check-cfg=cfg(atlas_scale)");
    if std::env::var("ATLAS_TARGET_HW")
        .as_deref()
        .map(|hw| hw.starts_with("strix"))
        .unwrap_or(false)
    {
        println!("cargo:rustc-cfg=atlas_scale");
    }

    if matches!(
        std::env::var("ATLAS_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    ) {
        // Even under ATLAS_SKIP_BUILD (CI no-GPU test build, no nvcc), the
        // cublaslt.rs FFI references cublasLt symbols that must resolve at LINK
        // time. cudarc emits -lcuda for us, but -lcublasLt is our own; emit it
        // here so the no-GPU `cargo test` build links (CI provides a stub
        // libcublasLt.so, same treatment as libcuda/libnccl).
        if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
            // -lcuda for the raw cu* driver FFI in cuda_backend/gpu_impl.rs
            // (cudarc is dlopen-mode here, so it doesn't emit it for us).
            println!("cargo:rustc-link-lib=dylib=cuda");
            println!("cargo:rustc-link-lib=dylib=cublasLt");
            // cudart: copy_d2d_2d_async uses cudaMemcpy2DAsync (a runtime, not
            // driver, symbol — CI's libcuda stub only has cu* driver symbols).
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
            println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
        }
        return;
    }

    // libcuda is only needed when the cuda feature is on (i.e. when
    // AtlasCudaBackend is compiled in). The metal feature build on
    // Apple Silicon must not request -lcuda.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    // Link libcuda for AtlasCudaBackend's raw CUDA driver API calls.
    // The actual CUDA driver is a stub at compile time; at runtime
    // it resolves to the NVIDIA driver installed on the system.
    println!("cargo:rustc-link-lib=dylib=cuda");
    // cuBLASLt for the high-efficiency GEMM path (ATLAS_CUBLAS_GEMM=1). The
    // hand-written mma.sync projection/MoE GEMMs hit only ~30% of the cuBLAS
    // ceiling on GB10; cuBLASLt is a measured 2.7-4.8x lever on those shapes.
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    // cudart: copy_d2d_2d_async uses cudaMemcpy2DAsync (a runtime, not driver,
    // symbol). Previously only emitted in the ATLAS_SKIP_BUILD stub path, so a
    // real GPU build fails to link with "undefined reference to
    // cudaMemcpy2DAsync" even though CI (which builds under ATLAS_SKIP_BUILD)
    // is green.
    println!("cargo:rustc-link-lib=dylib=cudart");

    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
    }
    // Standard CUDA locations
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
}
