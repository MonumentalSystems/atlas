// SPDX-License-Identifier: AGPL-3.0-only

//! PEFT adapter loader: `adapter_model.safetensors` → [`WeightStore`].
//!
//! Not `SafetensorsLoader` because (a) that loader only probes
//! `model.safetensors*` names (weights/loader.rs) and (b)
//! `WeightDtype::from_safetensors` rejects F16 (weights.rs), the PEFT
//! default save dtype. F16 is converted to BF16 on the host here so no
//! F16 ever reaches a kernel or the WeightDtype whitelist.
//!
//! NOTE: the device copies made here become garbage once the adapter is
//! packed into the fixed-address LoRA pool and are never freed (no weight
//! dealloc anywhere in Atlas). Accepted leak at adapter scale (~MBs).

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::{bf16, f16};

use super::{WeightDtype, WeightStore, WeightTensor, evict_page_cache};
use crate::gpu::GpuBackend;

/// Load a PEFT adapter's `adapter_model.safetensors` from `adapter_dir`
/// onto the GPU. Mirrors the single-file path of `SafetensorsLoader`
/// (mmap → per-tensor alloc + copy_h2d → page-cache evict) with a
/// host-side F16→BF16 conversion branch added.
pub fn load_adapter_safetensors(
    adapter_dir: &Path,
    gpu: &dyn GpuBackend,
    oom_reserve_bytes: usize,
) -> Result<WeightStore> {
    let path = adapter_dir.join("adapter_model.safetensors");
    if !path.exists() {
        if adapter_dir.join("adapter_model.bin").exists() {
            bail!(
                "REJECT[pickle-adapter]: {} ships adapter_model.bin (torch pickle); \
                 re-save with safe_serialization=True",
                adapter_dir.display()
            );
        }
        bail!("No adapter_model.safetensors in {}", adapter_dir.display());
    }

    // Header-only preflight (no mmap): F16 counts 2 B/elem — identical to
    // its post-conversion BF16 footprint.
    let estimated = super::estimate_load_bytes(&[path.clone()], &|_| false)?;
    let free = gpu.free_memory()?;
    if estimated + oom_reserve_bytes > free {
        bail!(
            "OOM pre-flight (LoRA adapter): {estimated} B adapter tensors + \
             {oom_reserve_bytes} B reserve exceeds {free} B free"
        );
    }

    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;

    let mut weights = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();
        let (bytes, dtype): (Cow<'_, [u8]>, WeightDtype) = match view.dtype() {
            safetensors::Dtype::F16 => {
                // Host-side F16 -> BF16 (locked decision; `half = "2"` is
                // already a spark-runtime dep).
                let conv: Vec<u8> = data
                    .chunks_exact(2)
                    .flat_map(|c| {
                        bf16::from_f32(f16::from_le_bytes([c[0], c[1]]).to_f32()).to_le_bytes()
                    })
                    .collect();
                (Cow::Owned(conv), WeightDtype::BF16)
            }
            other => (
                Cow::Borrowed(data),
                WeightDtype::from_safetensors(other)
                    .with_context(|| format!("LoRA adapter tensor '{name}'"))?,
            ),
        };
        let ptr = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, ptr)?;
        weights.insert(name, WeightTensor { ptr, shape, dtype });
    }

    // Drop mmap before evicting page cache (GB10 unified memory).
    drop(tensors);
    drop(mmap);
    evict_page_cache(&file);

    Ok(WeightStore::from_map(weights))
}
