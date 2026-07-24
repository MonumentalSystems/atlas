// SPDX-License-Identifier: AGPL-3.0-only

//! Exact GGUF loading admission accounting.

use anyhow::{Context, Result, bail};

use crate::gpu::GpuBackend;

const LAGUNA_MIN_GUARD_BYTES: usize = 5 * 1024 * 1024 * 1024;

pub(super) fn effective_guard_bytes(arch: &str, configured_bytes: usize) -> usize {
    if arch == "laguna" {
        configured_bytes.max(LAGUNA_MIN_GUARD_BYTES)
    } else {
        configured_bytes
    }
}

/// Pre-flight admission using exact GGUF loading components.
///
/// `free_memory()` is the relevant allocation budget for the active backend:
/// CUDA free device memory or Metal's recommended working-set headroom. The
/// configured guard reserves the model arena, KV cache, pipelines, and safety
/// headroom that the caller allocates after weights.
pub(super) fn preflight_oom(
    gpu: &dyn GpuBackend,
    resident_bytes: usize,
    max_tensor_transient_bytes: usize,
    guard_bytes: usize,
) -> Result<()> {
    let peak = resident_bytes
        .checked_add(max_tensor_transient_bytes)
        .context("GGUF peak byte size overflow")?;
    let required = peak
        .checked_add(guard_bytes)
        .context("GGUF required byte size overflow")?;
    let free = gpu.free_memory()?;
    let capacity = gpu.total_memory().ok();
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    tracing::info!(
        "GGUF pre-flight: {:.2} GiB resident weights + {:.2} GiB max tensor transient \
         = {:.2} GiB loading peak; {:.2} GiB configured/runtime guard; {:.2} GiB \
         backend allocation headroom; physical/device capacity: {}",
        gib(resident_bytes),
        gib(max_tensor_transient_bytes),
        gib(peak),
        gib(guard_bytes),
        gib(free),
        capacity
            .map(|bytes| format!("{:.2} GiB", gib(bytes)))
            .unwrap_or_else(|| "unavailable".to_string()),
    );
    if required > free {
        bail!(
            "Pre-flight OOM: GGUF needs {:.2} GiB resident weights + {:.2} GiB \
             max tensor transient + {:.2} GiB configured/runtime guard = {:.2} GiB, \
             but the backend has only {:.2} GiB allocation headroom. Use a smaller \
             quantization or reduce the requested context/arena size.",
            gib(resident_bytes),
            gib(max_tensor_transient_bytes),
            gib(guard_bytes),
            gib(required),
            gib(free),
        );
    }
    Ok(())
}
