// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loading from safetensors files (SBIO IORouter for filesystem I/O).

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::Path;

/// Advise the OS to evict a file's pages from the page cache.
///
/// On GB10 (unified memory), mmap'd safetensors share the GPU memory pool.
/// After copying tensors to GPU, the mmap pages linger in the page cache,
/// consuming memory that should be available for KV cache and inference buffers.
/// This function tells the kernel those pages are no longer needed.
#[cfg(target_os = "linux")]
pub(crate) fn evict_page_cache(file: &std::fs::File) {
    use std::os::unix::io::AsRawFd;
    // POSIX_FADV_DONTNEED = 4 on Linux (POSIX standard).
    // macOS lacks posix_fadvise — see the non-linux branch below.
    const POSIX_FADV_DONTNEED: libc::c_int = 4;
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn evict_page_cache(_file: &std::fs::File) {
    // No-op: macOS/BSD have no posix_fadvise. Apple Silicon UMA already
    // shares page cache with the GPU pool, so eviction is unnecessary.
}

/// Data type of a weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    BF16,
    FP32,
    FP8E4M3,
    FP8E8M0,
    UInt8,
    Int64,
}

impl WeightDtype {
    pub fn byte_size(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::FP32 => 4,
            Self::FP8E4M3 => 1,
            Self::FP8E8M0 => 1,
            Self::UInt8 => 1,
            Self::Int64 => 8,
        }
    }

    fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self> {
        match dtype {
            safetensors::Dtype::BF16 => Ok(Self::BF16),
            safetensors::Dtype::F32 => Ok(Self::FP32),
            safetensors::Dtype::U8 => Ok(Self::UInt8),
            // I8: raw 1-byte container for 4-bit-packed NVFP4 (DeepSeek-V4 MTP
            // experts). Treat as UInt8 — signedness is irrelevant for packed FP4.
            safetensors::Dtype::I8 => Ok(Self::UInt8),
            safetensors::Dtype::F8_E4M3 => Ok(Self::FP8E4M3),
            safetensors::Dtype::F8_E8M0 => Ok(Self::FP8E8M0),
            safetensors::Dtype::I64 => Ok(Self::Int64),
            other => bail!("Unsupported safetensors dtype: {other:?}"),
        }
    }

    /// Map a raw safetensors header dtype STRING (as it appears in the JSON
    /// header, e.g. `"BF16"`, `"F8_E4M3"`) to a [`WeightDtype`]. This is the
    /// exact closed mapping `fast_weights::header::parse_header` uses, factored
    /// out so the RDMA weight loader (which receives dtype as a wire string in
    /// the peer manifest, not a `safetensors::Dtype`) resolves it identically —
    /// byte-identity depends on the two ends agreeing. Bails on any dtype the
    /// disk loaders don't support.
    pub fn from_safetensors_str(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => Self::FP32,
            "BF16" => Self::BF16,
            "U8" => Self::UInt8,
            // I8 is a 1-byte raw container (packed NVFP4); signedness is
            // irrelevant, treat as raw bytes exactly like the disk path.
            "I8" => Self::UInt8,
            "F8_E4M3" => Self::FP8E4M3,
            "F8_E8M0" => Self::FP8E8M0,
            "I64" => Self::Int64,
            other => bail!("Unsupported safetensors dtype '{other}'"),
        })
    }
}

/// A weight tensor on the GPU.
pub struct WeightTensor {
    pub ptr: DevicePtr,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

impl WeightTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.byte_size()
    }
}

/// All model weights loaded onto the GPU, keyed by HuggingFace name.
pub struct WeightStore {
    weights: HashMap<String, WeightTensor>,
}

impl WeightStore {
    /// Create an empty weight store (for testing).
    pub fn empty() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    /// Wrap a pre-built map. Used by alternate loaders (e.g.
    /// `fast_weights::FastSafetensorsLoader`, and the RDMA weight loader in
    /// `spark-storage`, which lives in a different crate and so needs this
    /// public).
    pub fn from_map(weights: HashMap<String, WeightTensor>) -> Self {
        Self { weights }
    }

    /// Get a weight tensor by name. Fails fast if not found.
    pub fn get(&self, name: &str) -> Result<&WeightTensor> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Weight '{name}' not found in store"))
    }

    /// Check if a weight exists.
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Number of loaded weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// True if no weights are loaded.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Iterator over all weight names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(|s| s.as_str())
    }

    /// Total bytes across all weight tensors on the GPU.
    pub fn total_bytes(&self) -> usize {
        self.weights.values().map(|w| w.byte_size()).sum()
    }

    /// Check if any tensor has FP8 dtype.
    pub fn has_fp8_weights(&self) -> bool {
        self.weights
            .values()
            .any(|w| matches!(w.dtype, WeightDtype::FP8E4M3))
    }
}

/// SBIO IORouter trait for weight loading.
pub trait WeightLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore>;
}

/// Loads weights from safetensors files using mmap.
pub struct SafetensorsLoader {
    /// EP rank (0-based). Only used when ep_world_size > 1.
    pub ep_rank: usize,
    /// EP world size. When > 1, remote expert tensors are skipped.
    pub ep_world_size: usize,
    /// Total number of MoE experts in the model (for EP partitioning).
    pub num_experts: usize,
    /// Expert streaming: skip loading ALL routed-expert tensors (they are served
    /// on demand from the resident store / RDMA peer). Router gate + shared
    /// expert stay resident. This also shrinks the pre-flight byte estimate, so
    /// an over-core checkpoint whose experts exceed GPU memory still loads.
    pub stream_all_experts: bool,
    /// Override for the peak memory multiplier in the pre-flight OOM check.
    /// Set from QuantFormat::peak_memory_multiplier() in the caller.
    /// When None, the pre-flight uses its own heuristic (1.3x NVFP4 / 1.5x FP8).
    pub peak_memory_multiplier: Option<f64>,
}

impl Default for SafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetensorsLoader {
    /// Create a loader with no expert parallelism (loads all tensors).
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            stream_all_experts: false,
            peak_memory_multiplier: None,
        }
    }

    /// Create a loader with EP-aware filtering.
    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            stream_all_experts: false,
            peak_memory_multiplier: None,
        }
    }

    /// Check if a tensor should be skipped at load time.
    /// Under expert streaming, skips ALL routed `*.experts.{E}.*` tensors.
    /// Under EP, skips `*.experts.{E}.*` tensors where E is not in local range.
    /// MTP head experts are never skipped (small, fully replicated).
    fn should_skip_tensor(&self, name: &str) -> bool {
        // MTP head experts are small — always replicate, never skip.
        if name.starts_with("mtp.") {
            return false;
        }
        // Expert streaming: every routed expert is served on demand from the
        // store/peer, so none is loaded here — the router gate + shared expert
        // (no expert index) stay resident.
        if self.stream_all_experts && parse_expert_index(name).is_some() {
            return true;
        }
        if self.ep_world_size <= 1 {
            return false;
        }
        // Parse expert index from patterns like "*.experts.42.gate_proj*"
        if let Some(idx) = parse_expert_index(name) {
            let per_rank = self.num_experts / self.ep_world_size;
            let local_start = self.ep_rank * per_rank;
            let local_end = if self.ep_rank == self.ep_world_size - 1 {
                self.num_experts
            } else {
                local_start + per_rank
            };
            idx < local_start || idx >= local_end
        } else {
            false // Non-expert tensors are always loaded (replicated)
        }
    }
}

/// Parse expert index from tensor name (e.g. "model.layers.3.mlp.experts.42.gate_proj.weight" → 42).
///
/// `pub` so the RDMA weight loader (`spark-storage`) applies the exact same
/// expert-skip predicate as the disk loaders — a divergent copy would change
/// which tensors land resident and break behavioural parity.
pub fn parse_expert_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "experts" && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

pub mod adapter;
mod loader;
pub mod mlx_int8;
pub(crate) use loader::{check_oom_guard, estimate_has_fp8, estimate_load_bytes};

#[cfg(test)]
mod skip_tests {
    use super::*;

    const GATE: &str = "model.language_model.layers.3.mlp.experts.42.gate_proj.weight_packed";
    const SHARED: &str = "model.language_model.layers.3.mlp.shared_expert.gate_proj.weight_packed";
    const ROUTER: &str = "model.language_model.layers.3.mlp.gate.weight";
    const ATTN: &str = "model.language_model.layers.3.self_attn.q_proj.weight";

    #[test]
    fn stream_all_experts_skips_only_routed_experts() {
        let mut l = SafetensorsLoader::new();
        l.stream_all_experts = true;
        // Routed experts stream from the store/peer → skipped at load.
        assert!(l.should_skip_tensor(GATE));
        // Router gate, shared expert, attention stay resident.
        assert!(!l.should_skip_tensor(SHARED));
        assert!(!l.should_skip_tensor(ROUTER));
        assert!(!l.should_skip_tensor(ATTN));
        // MTP experts are never skipped (handled separately, small).
        assert!(!l.should_skip_tensor("mtp.experts.5.gate_proj.weight_packed"));
    }

    #[test]
    fn without_streaming_nothing_is_skipped_single_gpu() {
        let l = SafetensorsLoader::new(); // ep_world_size=1, stream off
        assert!(!l.should_skip_tensor(GATE));
        assert!(!l.should_skip_tensor(SHARED));
    }

    #[test]
    fn from_safetensors_str_matches_disk_mapping() {
        // The RDMA weight peer publishes these raw header strings; the client
        // must resolve them to the exact WeightDtype the disk loaders use, else
        // byte_size/shape diverge and logits break. Locks the closed mapping.
        use WeightDtype::*;
        for (s, want) in [
            ("F32", FP32),
            ("BF16", BF16),
            ("U8", UInt8),
            ("I8", UInt8), // packed NVFP4 raw container
            ("F8_E4M3", FP8E4M3),
            ("F8_E8M0", FP8E8M0),
            ("I64", Int64),
        ] {
            assert_eq!(
                WeightDtype::from_safetensors_str(s).unwrap(),
                want,
                "dtype {s}"
            );
        }
        assert!(WeightDtype::from_safetensors_str("F16").is_err());
        assert!(WeightDtype::from_safetensors_str("bogus").is_err());
    }

    #[test]
    fn ep_skip_still_works_independently() {
        // EP skip: rank 0 of 2 owns experts 0..128.
        let l = SafetensorsLoader::with_ep(0, 2, 256);
        assert!(l.should_skip_tensor("model.layers.0.mlp.experts.200.gate_proj.weight"));
        assert!(!l.should_skip_tensor("model.layers.0.mlp.experts.10.gate_proj.weight"));
    }
}
