// SPDX-License-Identifier: AGPL-3.0-only

//! Generic GGUF weight loader for Atlas.
//!
//! Loads any GGUF checkpoint the same way [`super::SafetensorsLoader`] loads
//! safetensors: mmap the file, walk its tensors, land each one GPU-resident in
//! a [`WeightStore`] keyed by HuggingFace name. Unlike safetensors, GGUF tensors
//! are quantized in-file, so each tensor is *dequantized to BF16* on the way in.
//!
//! Per the project design directive we **prefer GPU dequant**: upload the raw
//! quantized GGUF block bytes h2d, run a device dequant kernel that writes BF16
//! into fresh device memory, and hand that BF16 [`WeightTensor`] to the store.
//! The per-arch model loaders do the downstream NVFP4 requantize — this loader's
//! job ends at clean BF16. A pure-CPU reference dequant is the fallback for ggml
//! types lacking a GPU kernel and the correctness oracle under `MockGpuBackend`
//! (which cannot execute kernels).
//!
//! GGUF `dims` are ggml-order (fastest-varying first); Atlas/HF shapes are the
//! reverse, so each tensor's shape is reversed before it enters the store.
//!
//! The PrismML `Q2_0` (id 42) group size is not encoded in the type id. It
//! defaults to group-128 (the shipped Ternary-Bonsai layout); set
//! `ATLAS_GGUF_Q2_GROUP=64` for the fork-master group-64 layout.

mod config;
mod container;
mod dequant_cpu;
mod dequant_gpu;
mod names;

pub use config::config_from_gguf_dir;

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::{WeightDtype, WeightStore, WeightTensor, check_oom_guard, evict_page_cache};
use crate::gpu::{DevicePtr, GpuBackend};

/// Locate the GGUF weight file in `dir`. Returns the lexicographically-first
/// `*.gguf` (also the first shard, `*-00001-of-*`, of a split file).
pub fn find_gguf(dir: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("gguf"))
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

/// The id-42 PrismML group size, from `ATLAS_GGUF_Q2_GROUP` (default 128).
fn q2_group_usize() -> usize {
    match std::env::var("ATLAS_GGUF_Q2_GROUP").ok().as_deref() {
        Some("64") => 64,
        _ => 128,
    }
}

/// Map a group size to the container's `Q2Group` (for on-disk byte sizing).
fn q2_group_variant(g: usize) -> container::Q2Group {
    if g == 64 {
        container::Q2Group::G64
    } else {
        container::Q2Group::G128
    }
}

/// Loads weights from a GGUF file, dequantizing every tensor to BF16 on the GPU.
///
/// Mirrors [`super::SafetensorsLoader`] so the two are interchangeable behind the
/// [`super::WeightLoader`] trait and the serve call-site can pick one on file
/// type.
pub struct GgufLoader {
    /// EP rank (0-based). Only used when `ep_world_size > 1`.
    pub ep_rank: usize,
    /// EP world size. When > 1, remote expert slices are skipped.
    pub ep_world_size: usize,
    /// Total number of MoE experts in the model (for EP partitioning).
    pub num_experts: usize,
    /// Override for the peak-memory multiplier in the pre-flight OOM check.
    pub peak_memory_multiplier: Option<f64>,
}

impl Default for GgufLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufLoader {
    /// Create a loader with no expert parallelism (loads all tensors).
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
        }
    }

    /// Create a loader with EP-aware expert filtering.
    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
        }
    }

    /// True if expert `idx` lives on a remote EP rank and should be skipped.
    fn should_skip_expert(&self, idx: usize) -> bool {
        if self.ep_world_size <= 1 || self.num_experts == 0 {
            return false;
        }
        let per_rank = self.num_experts / self.ep_world_size;
        let local_start = self.ep_rank * per_rank;
        let local_end = if self.ep_rank == self.ep_world_size - 1 {
            self.num_experts
        } else {
            local_start + per_rank
        };
        idx < local_start || idx >= local_end
    }

    /// Split a dequantized, stacked expert buffer into per-expert `WeightTensor`s
    /// that alias offsets into the single BF16 device allocation. `shape[0]` is
    /// the expert count; each expert tensor is `shape[1..]`.
    fn emit_experts(
        &self,
        weights: &mut HashMap<String, WeightTensor>,
        base_ptr: DevicePtr,
        shape: &[usize],
        layer: usize,
        proj: &str,
        skipped: &mut usize,
    ) -> Result<()> {
        let count = *shape
            .first()
            .context("stacked expert tensor has no leading expert dimension")?;
        let per_elems: usize = shape[1..].iter().product();
        let per_bytes = per_elems * WeightDtype::BF16.byte_size();
        let expert_shape: Vec<usize> = shape[1..].to_vec();
        for e in 0..count {
            if self.should_skip_expert(e) {
                *skipped += 1;
                continue;
            }
            let ptr = base_ptr.offset(e * per_bytes);
            let name = names::expert_name(layer, proj, e);
            weights.insert(
                name,
                WeightTensor {
                    ptr,
                    shape: expert_shape.clone(),
                    dtype: WeightDtype::BF16,
                },
            );
        }
        Ok(())
    }
}

/// Dequant one tensor's raw block bytes to a BF16 device buffer. Prefer-GPU /
/// CPU-fallback: use the GPU kernel when one exists for `id` (and `force_cpu`
/// is unset), else the CPU reference dequant (host BF16 → single h2d).
fn dequant_to_device(
    gpu: &dyn GpuBackend,
    id: u32,
    raw: &[u8],
    num_elements: usize,
    q2_group: usize,
    force_cpu: bool,
) -> Result<DevicePtr> {
    if !force_cpu && dequant_gpu::supports(id) {
        let q_ptr = gpu.alloc(raw.len())?;
        gpu.copy_h2d(raw, q_ptr)?;
        let bf16_ptr = dequant_gpu::to_bf16(gpu, id, q_ptr, num_elements, q2_group)
            .with_context(|| format!("GPU dequant failed for ggml type {id}"))?;
        gpu.free(q_ptr)?;
        Ok(bf16_ptr)
    } else if dequant_cpu::supports(id) {
        let host = dequant_cpu::to_bf16_bytes(id, q2_group, raw, num_elements)
            .with_context(|| format!("CPU dequant failed for ggml type {id}"))?;
        debug_assert_eq!(host.len(), num_elements * WeightDtype::BF16.byte_size());
        let bf16_ptr = gpu.alloc(host.len())?;
        gpu.copy_h2d(&host, bf16_ptr)?;
        Ok(bf16_ptr)
    } else {
        bail!("No GPU or CPU dequant available for ggml type {id}");
    }
}

/// Pre-flight: estimate the total BF16 footprint (dequant expands quantized
/// blocks) and bail before allocating if it won't fit under the reserve.
fn preflight_oom(
    gpu: &dyn GpuBackend,
    est_bf16_bytes: usize,
    reserve_bytes: usize,
    multiplier: Option<f64>,
) -> Result<()> {
    // Small transient overhead: the raw quantized scratch buffer coexists with
    // its BF16 output for one tensor at a time (freed immediately after).
    let overhead = multiplier.unwrap_or(1.1);
    let peak = (est_bf16_bytes as f64 * overhead) as usize;
    let free = gpu.free_memory()?;
    let gb = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    tracing::info!(
        "GGUF pre-flight: ~{:.2} GB BF16 after dequant, {:.1}x overhead = {:.2} GB peak, \
         {:.2} GB free, {:.1} GB reserve",
        gb(est_bf16_bytes),
        overhead,
        gb(peak),
        gb(free),
        gb(reserve_bytes),
    );
    if peak + reserve_bytes > free {
        bail!(
            "Pre-flight OOM: GGUF dequant to BF16 needs ~{:.2} GB peak + {:.1} GB reserve, \
             only {:.2} GB free. Use a smaller model or lower --oom-guard-mb.",
            gb(peak),
            gb(reserve_bytes),
            gb(free),
        );
    }
    Ok(())
}

impl super::WeightLoader for GgufLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        let path = find_gguf(model_dir)
            .with_context(|| format!("No .gguf file found in {}", model_dir.display()))?;
        tracing::info!("Loading GGUF weights from {}", path.display());

        let file = std::fs::File::open(&path)
            .with_context(|| format!("Failed to open {}", path.display()))?;
        // SAFETY: same mmap contract as the safetensors loader.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        let gguf = container::GgufFile::parse(&mmap)
            .with_context(|| format!("Failed to parse GGUF container: {}", path.display()))?;

        let arch = gguf
            .get_str("general.architecture")
            .unwrap_or("llama")
            .to_lowercase();
        let force_cpu = std::env::var("ATLAS_GGUF_FORCE_CPU").ok().as_deref() == Some("1");
        let q2_group = q2_group_usize();
        let q2_variant = q2_group_variant(q2_group);

        // Pre-flight: sum BF16 output bytes over tensors we will actually keep.
        let est_bf16: usize = gguf
            .tensors
            .iter()
            .filter(|t| {
                !matches!(
                    names::translate(&t.name, &arch),
                    None | Some(names::GgufName::Drop)
                )
            })
            .map(|t| t.num_elements() * WeightDtype::BF16.byte_size())
            .sum();
        preflight_oom(gpu, est_bf16, oom_reserve_bytes, self.peak_memory_multiplier)?;

        let mut weights: HashMap<String, WeightTensor> = HashMap::new();
        let mut skipped = 0usize;

        for tensor in &gguf.tensors {
            let target = match names::translate(&tensor.name, &arch) {
                Some(names::GgufName::Drop) | None => continue,
                Some(t) => t,
            };

            let num_elements = tensor.num_elements();
            let raw_len = gguf
                .tensor_byte_size(tensor, q2_variant)
                .with_context(|| format!("byte-len for tensor {}", tensor.name))?;
            let start = gguf.tensor_abs_offset(tensor);
            let raw = mmap
                .get(start..start + raw_len)
                .with_context(|| format!("tensor {} out of bounds in GGUF", tensor.name))?;

            let id = tensor.ggml_type.id();
            let bf16_ptr = dequant_to_device(gpu, id, raw, num_elements, q2_group, force_cpu)
                .with_context(|| format!("dequant tensor {}", tensor.name))?;

            // GGUF dims are ggml-order; Atlas/HF shape is the reverse.
            let hf_shape: Vec<usize> = tensor.dims.iter().rev().copied().collect();

            match target {
                names::GgufName::Direct(hf_name) => {
                    weights.insert(
                        hf_name,
                        WeightTensor {
                            ptr: bf16_ptr,
                            shape: hf_shape,
                            dtype: WeightDtype::BF16,
                        },
                    );
                }
                names::GgufName::ExpertStack { layer, proj } => {
                    self.emit_experts(
                        &mut weights,
                        bf16_ptr,
                        &hf_shape,
                        layer,
                        proj,
                        &mut skipped,
                    )?;
                }
                names::GgufName::Drop => unreachable!("Drop filtered above"),
            }
        }

        // Drop the mapping before evicting its pages from the (unified) cache.
        drop(gguf);
        drop(mmap);
        evict_page_cache(&file);

        if skipped > 0 {
            tracing::info!("EP: skipped {} remote expert slices", skipped);
        }
        check_oom_guard(gpu, oom_reserve_bytes, "weight loading (GGUF)")?;
        tracing::info!("Loaded {} weight tensors (GGUF → BF16)", weights.len());
        Ok(WeightStore::from_map(weights))
    }
}

#[cfg(test)]
mod real_file_test;

#[cfg(all(test, feature = "cuda"))]
mod gpu_validate_test;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;
    use crate::weights::WeightLoader;

    fn push_u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_u64(b: &mut Vec<u8>, v: u64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_str(b: &mut Vec<u8>, s: &str) {
        push_u64(b, s.len() as u64);
        b.extend_from_slice(s.as_bytes());
    }

    /// Minimal valid GGUF v3: one F32 1-D tensor + a `general.alignment` KV.
    fn build_single_f32_gguf(name: &str, vals: &[f32]) -> Vec<u8> {
        let mut b = Vec::new();
        push_u32(&mut b, 0x4655_4747); // "GGUF"
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 1); // tensor_count
        push_u64(&mut b, 1); // kv_count
        push_str(&mut b, "general.alignment");
        push_u32(&mut b, 4); // UINT32
        push_u32(&mut b, 32);
        push_str(&mut b, name);
        push_u32(&mut b, 1); // n_dims
        push_u64(&mut b, vals.len() as u64); // dims[0]
        push_u32(&mut b, 0); // ggml_type F32
        push_u64(&mut b, 0); // offset
        let pad = (32 - (b.len() % 32)) % 32;
        b.extend(std::iter::repeat_n(0u8, pad));
        for v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    #[test]
    fn loads_single_tensor_cpu_fallback() {
        // Mock cannot execute kernels, so force the CPU reference dequant path.
        unsafe { std::env::set_var("ATLAS_GGUF_FORCE_CPU", "1") };

        let vals = [1.0f32, -2.0, 3.5, 0.0, 7.0, -0.25];
        let bytes = build_single_f32_gguf("token_embd.weight", &vals);

        let dir = std::env::temp_dir().join(format!("atlas_gguf_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.gguf"), &bytes).unwrap();

        let gpu = MockGpuBackend::new();
        let store = GgufLoader::new()
            .load(&dir, &gpu, 1024 * 1024)
            .expect("GGUF load");

        assert_eq!(store.len(), 1);
        assert!(store.contains("model.embed_tokens.weight"));
        let t = store.get("model.embed_tokens.weight").unwrap();
        assert_eq!(t.shape, vec![6]);
        assert_eq!(t.dtype, WeightDtype::BF16);

        let raw = gpu.read_alloc(t.ptr).expect("bf16 bytes present");
        assert_eq!(raw.len(), 6 * WeightDtype::BF16.byte_size());
        let got: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        assert_eq!(got, vals.to_vec());

        std::fs::remove_dir_all(&dir).ok();
        unsafe { std::env::remove_var("ATLAS_GGUF_FORCE_CPU") };
    }

    #[test]
    fn find_gguf_picks_first() {
        let dir = std::env::temp_dir().join(format!("atlas_gguf_find_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("b.gguf"), b"x").unwrap();
        std::fs::write(dir.join("a.gguf"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        let found = find_gguf(&dir).unwrap();
        assert_eq!(found.file_name().unwrap().to_str().unwrap(), "a.gguf");
        std::fs::remove_dir_all(&dir).ok();
    }
}
