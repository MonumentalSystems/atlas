// SPDX-License-Identifier: AGPL-3.0-only

//! `impl WeightLoader for SafetensorsLoader` + sharded/single loader helpers.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;

use super::{SafetensorsLoader, WeightLoader, WeightStore};
use crate::gpu::GpuBackend;

impl WeightLoader for SafetensorsLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        let skip_fn = |name: &str| self.should_skip_tensor(name);

        // Collect all safetensor files (indexed, single, or unindexed shards).
        // Supports both HuggingFace standard (model.safetensors*) and Mistral
        // consolidated format (consolidated.safetensors*).
        let index_path = model_dir.join("model.safetensors.index.json");
        let consolidated_index = model_dir.join("consolidated.safetensors.index.json");
        let shard_files: Vec<std::path::PathBuf>;
        let use_index;
        let actual_index_path;

        if index_path.exists() {
            use_index = true;
            actual_index_path = index_path;
            shard_files = vec![];
        } else if consolidated_index.exists() {
            use_index = true;
            actual_index_path = consolidated_index;
            shard_files = vec![];
        } else {
            use_index = false;
            actual_index_path = index_path; // unused
            let single = model_dir.join("model.safetensors");
            if single.exists() {
                shard_files = vec![single];
            } else {
                // Try both model.safetensors-* and consolidated-* shard patterns
                let mut shards: Vec<_> = std::fs::read_dir(model_dir)?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                            (n.starts_with("model.safetensors-") || n.starts_with("consolidated-"))
                                && n.ends_with(".safetensors")
                        })
                    })
                    .collect();
                shards.sort();
                if shards.is_empty() {
                    bail!(
                        "No safetensor files found in {}. Expected model.safetensors*, \
                         consolidated.safetensors*, or consolidated-*-of-*.safetensors",
                        model_dir.display()
                    );
                }
                shard_files = shards;
            }
        }

        // Pre-flight OOM estimate: scan safetensor headers (no data) to compute
        // total bytes this rank will load, then apply a model-building overhead
        // multiplier and abort early if the model won't fit.
        //
        // Model building creates additional GPU allocations on top of the raw
        // weight store: transposed weight copies for prefill GEMM, predequanted
        // FP8 copies, NVFP4 quantized copies (for FP8 checkpoints), and transient
        // BF16 intermediates during FP8→NVFP4 conversion.
        //
        // Empirical overhead multipliers (peak memory / on-disk weight bytes):
        //   NVFP4 (Sehyo): ~2.0x  (store aliased + transposed/predequant copies)
        //   FP8 native:    ~1.5x  (store stays FP8, only attention prefill gets NVFP4 copies)
        {
            let ep = EpEstimate {
                ep_rank: self.ep_rank,
                ep_world_size: self.ep_world_size,
                num_experts: self.num_experts,
            };
            let estimated = estimate_load_bytes(&shard_files, &skip_fn, ep)?;
            let has_fp8 = estimate_has_fp8(&shard_files, &skip_fn)?;
            let overhead_multiplier: f64 =
                self.peak_memory_multiplier
                    .unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
            let peak_estimated = (estimated as f64 * overhead_multiplier) as usize;
            let free = gpu.free_memory()?;
            let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
            let est_gb = estimated as f64 / (1024.0 * 1024.0 * 1024.0);
            let peak_gb = peak_estimated as f64 / (1024.0 * 1024.0 * 1024.0);
            let reserve_gb = oom_reserve_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!(
                "Pre-flight estimate: {:.2} GB on-disk weights, {:.1}x overhead = {:.2} GB peak, \
                 {:.2} GB free, {:.1} GB reserve (FP8: {})",
                est_gb,
                overhead_multiplier,
                peak_gb,
                free_gb,
                reserve_gb,
                has_fp8,
            );
            if peak_estimated + oom_reserve_bytes > free {
                bail!(
                    "OOM pre-flight: model peak memory ({:.2} GB = {:.2} GB weights × {:.1}x \
                     model-building overhead) + {:.1} GB reserve = {:.2} GB, \
                     but only {:.2} GB GPU memory is available. \
                     This model is too large. Use a smaller quantization (NVFP4 instead of FP8) \
                     or add more GPUs for expert parallelism.",
                    peak_gb,
                    est_gb,
                    overhead_multiplier,
                    reserve_gb,
                    peak_gb + reserve_gb,
                    free_gb,
                );
            }
        }

        let mut weight_map = if use_index {
            load_sharded(
                model_dir,
                &actual_index_path,
                gpu,
                oom_reserve_bytes,
                &skip_fn,
                self.peak_memory_multiplier,
                EpEstimate {
                    ep_rank: self.ep_rank,
                    ep_world_size: self.ep_world_size,
                    num_experts: self.num_experts,
                },
            )?
        } else if shard_files.len() == 1 {
            load_single(&shard_files[0], gpu, oom_reserve_bytes, &skip_fn)?
        } else {
            tracing::info!("Loading {} unindexed safetensor shards", shard_files.len());
            let initial_free = gpu.free_memory()?;
            let mut combined = HashMap::new();
            for (i, shard) in shard_files.iter().enumerate() {
                let map = load_single(shard, gpu, oom_reserve_bytes, &skip_fn)?;
                let free_now = gpu.free_memory().unwrap_or(0);
                let used = initial_free.saturating_sub(free_now);
                tracing::info!(
                    "  Shard {}/{} done — GPU memory: {:.2} GB used, {:.2} GB free",
                    i + 1,
                    shard_files.len(),
                    used as f64 / (1024.0 * 1024.0 * 1024.0),
                    free_now as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                check_oom_guard(
                    gpu,
                    oom_reserve_bytes,
                    &format!("weight loading (shard {}/{})", i + 1, shard_files.len()),
                )?;
                combined.extend(map);
            }
            combined
        };

        // Load extra weight files (e.g. MTP weights grafted from another quantization).
        // Extra weights (MTP) are always fully loaded — they have their own expert lists.
        let no_skip = |_: &str| false;
        let extra = model_dir.join("extra_weights.safetensors");
        if extra.exists() {
            let extra_weights = load_single(&extra, gpu, oom_reserve_bytes, &no_skip)?;
            tracing::info!(
                "Loaded {} extra weight tensors from extra_weights.safetensors",
                extra_weights.len()
            );
            weight_map.extend(extra_weights);
        }

        Ok(WeightStore {
            weights: weight_map,
        })
    }
}

/// Index file format: { "weight_map": { "tensor_name": "shard_filename" } }
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Read only the safetensor header from a file (no mmap, no GPU memory impact).
/// The header is typically a few KB of JSON — safe to read on GB10 unified memory
/// without consuming GPU pages.
pub(crate) fn read_safetensor_header(
    path: &Path,
) -> Result<Vec<(String, Vec<usize>, safetensors::Dtype)>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Pre-flight: failed to open {}", path.display()))?;

    // Safetensors format: 8-byte LE header size, then JSON header, then data.
    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;

    // Sanity check: header shouldn't exceed 64 MB.
    if header_size > 64 * 1024 * 1024 {
        bail!(
            "Safetensor header too large ({} bytes) in {}",
            header_size,
            path.display()
        );
    }

    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)?;

    // Parse the JSON header manually to extract tensor metadata.
    let header: serde_json::Value = serde_json::from_slice(&header_buf)?;
    let obj = header.as_object().context("Invalid safetensor header")?;

    let mut tensors = Vec::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_str = info["dtype"].as_str().unwrap_or("BF16");
        let dtype = match dtype_str {
            "F32" => safetensors::Dtype::F32,
            "F16" => safetensors::Dtype::F16,
            "BF16" => safetensors::Dtype::BF16,
            "I32" => safetensors::Dtype::I32,
            "I16" => safetensors::Dtype::I16,
            "I8" => safetensors::Dtype::I8,
            "U8" => safetensors::Dtype::U8,
            "F8_E4M3" => safetensors::Dtype::F8_E4M3,
            "F8_E5M2" => safetensors::Dtype::F8_E5M2,
            _ => safetensors::Dtype::BF16,
        };
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        tensors.push((name.clone(), shape, dtype));
    }
    Ok(tensors)
}

/// Bytes per element for a safetensor dtype.
fn dtype_elem_bytes(dtype: safetensors::Dtype) -> usize {
    match dtype {
        safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
        safetensors::Dtype::F16
        | safetensors::Dtype::BF16
        | safetensors::Dtype::I16
        | safetensors::Dtype::U16 => 2,
        safetensors::Dtype::I8
        | safetensors::Dtype::U8
        | safetensors::Dtype::F8_E4M3
        | safetensors::Dtype::F8_E5M2 => 1,
        _ => 2,
    }
}

/// Expert-parallelism parameters for the pre-flight byte estimate.
///
/// The name-based `skip_fn` already drops *per-file-sharded* remote experts
/// (`*.experts.<N>.*`), so those never reach the accounting loop. But a
/// checkpoint whose experts are stored as a single *fused* tensor
/// (`*.experts.<proj>` with all experts stacked along dim 0, e.g.
/// Step-3.7-Flash-NVFP4 / Holo-35B-A3B-NVFP4) can't be filtered by name — the
/// remote experts live *inside* the tensor. The loader expert-splits those at
/// load time so only this rank's share resides in memory, so the estimate must
/// divide their bytes by the per-rank expert share too. See issue #194.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EpEstimate {
    pub ep_rank: usize,
    pub ep_world_size: usize,
    pub num_experts: usize,
}

impl EpEstimate {
    /// No expert parallelism — count every tensor at its full size (the
    /// single-node behaviour, byte-identical to pre-#194).
    #[cfg(test)]
    pub(crate) fn none() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
        }
    }

    /// Number of experts THIS rank keeps after expert-split. Mirrors
    /// `SafetensorsLoader::should_skip_tensor`'s local range exactly, including
    /// the remainder that lands on the last rank under uneven division.
    fn local_expert_count(&self) -> usize {
        if self.ep_world_size <= 1 || self.num_experts == 0 {
            return self.num_experts;
        }
        let per_rank = self.num_experts / self.ep_world_size;
        let local_start = self.ep_rank * per_rank;
        let local_end = if self.ep_rank == self.ep_world_size - 1 {
            self.num_experts
        } else {
            local_start + per_rank
        };
        local_end.saturating_sub(local_start)
    }
}

/// True when `name` is a *fused* expert tensor: all experts stacked along dim 0
/// under an `experts.<proj>` segment (e.g. `...mlp.experts.gate_up_proj.weight`)
/// rather than the per-file `experts.<N>.<proj>` layout. The loader expert-
/// splits these at load, so under EP only this rank's slice resides in memory.
///
/// `experts.<N>.*` (numeric index) is NOT fused — it's per-file-sharded and is
/// already handled by the name-based skip filter, so it returns false here.
/// Exact-segment matching means `shared_experts.*` (replicated) is unaffected.
pub(crate) fn is_fused_expert_tensor(name: &str) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "experts" && i + 1 < parts.len() {
            // Fused iff the token after `experts` is not a numeric expert index.
            return parts[i + 1].parse::<usize>().is_err();
        }
    }
    false
}

/// EP-aware byte accounting core (pure, no I/O — unit-testable with in-memory
/// tensor descriptors). Sums per-rank GPU bytes: replicated tensors count in
/// full; fused expert tensors count only this rank's expert share.
fn accumulate_load_bytes(
    tensors: &[(String, Vec<usize>, safetensors::Dtype)],
    skip_fn: &dyn Fn(&str) -> bool,
    ep: EpEstimate,
) -> usize {
    let mut total = 0usize;
    for (name, shape, dtype) in tensors {
        if skip_fn(name) {
            continue;
        }
        let numel: usize = shape.iter().product();
        let mut bytes = numel * dtype_elem_bytes(*dtype);
        if ep.ep_world_size > 1 && ep.num_experts > 0 && is_fused_expert_tensor(name) {
            // Only this rank's slice of the fused tensor is loaded.
            bytes = bytes * ep.local_expert_count() / ep.num_experts;
        }
        total += bytes;
    }
    total
}

/// Scan safetensor file headers (metadata only, no data loaded) to estimate
/// total GPU bytes this rank will load. Reads only the JSON header from each
/// file — does NOT mmap, so it's safe on GB10 unified memory.
///
/// EP-aware (#194): fused expert tensors are counted at this rank's expert
/// share, not the full checkpoint size, so raising `--ep-size` lowers the
/// per-node peak for fused-expert checkpoints (Step-3.7, Holo-35B-A3B).
pub(crate) fn estimate_load_bytes(
    files: &[std::path::PathBuf],
    skip_fn: &dyn Fn(&str) -> bool,
    ep: EpEstimate,
) -> Result<usize> {
    let mut total = 0usize;
    for path in files {
        let tensors = read_safetensor_header(path)?;
        total += accumulate_load_bytes(&tensors, skip_fn, ep);
    }
    Ok(total)
}

/// Check if the model is predominantly FP8 (>50% of weight bytes are FP8).
/// Sehyo NVFP4 models have a few FP8 scale tensors but the bulk is uint8 (NVFP4 packed).
/// True FP8 checkpoints (e.g. Qwen/Qwen3.5-122B-A10B-FP8) have most bytes as FP8.
pub(crate) fn estimate_has_fp8(
    files: &[std::path::PathBuf],
    skip_fn: &dyn Fn(&str) -> bool,
) -> Result<bool> {
    let mut fp8_bytes = 0usize;
    let mut total_bytes = 0usize;
    for path in files {
        for (name, shape, dtype) in read_safetensor_header(path)? {
            if skip_fn(&name) {
                continue;
            }
            let numel: usize = shape.iter().product();
            let bytes = numel * dtype_elem_bytes(dtype);
            total_bytes += bytes;
            if matches!(
                dtype,
                safetensors::Dtype::F8_E4M3 | safetensors::Dtype::F8_E5M2
            ) {
                fp8_bytes += bytes;
            }
        }
    }
    let fp8_frac = if total_bytes > 0 {
        fp8_bytes as f64 / total_bytes as f64
    } else {
        0.0
    };
    tracing::debug!(
        "FP8 fraction: {:.1}% ({} / {} bytes)",
        fp8_frac * 100.0,
        fp8_bytes,
        total_bytes
    );
    Ok(fp8_frac > 0.5)
}

pub(crate) fn check_oom_guard(
    gpu: &dyn GpuBackend,
    reserve_bytes: usize,
    phase: &str,
) -> Result<()> {
    let free = gpu.free_memory()?;
    if free < reserve_bytes {
        let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
        let reserve_gb = reserve_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        bail!(
            "OOM guard: aborting during {phase}. \
             Free GPU memory ({free_gb:.2} GB) is below the {reserve_gb:.1} GB safety reserve. \
             This model is too large for available GPU memory. \
             Reduce --max-seq-len, increase --oom-guard-mb, or use a smaller model."
        );
    }
    Ok(())
}
mod load_fns;
use load_fns::{load_sharded, load_single};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::parse_expert_index;
    use safetensors::Dtype;

    /// Mimic `SafetensorsLoader::should_skip_tensor`: under EP, drop the
    /// per-file-sharded remote experts (`experts.<N>.*`). Fused tensors have no
    /// numeric index so they are never skipped here — they get expert-split by
    /// the byte accounting instead.
    fn skip_fn(ep: EpEstimate) -> impl Fn(&str) -> bool {
        move |name: &str| {
            if ep.ep_world_size <= 1 {
                return false;
            }
            if name.starts_with("mtp.") {
                return false;
            }
            if let Some(idx) = parse_expert_index(name) {
                let per_rank = ep.num_experts / ep.ep_world_size;
                let local_start = ep.ep_rank * per_rank;
                let local_end = if ep.ep_rank == ep.ep_world_size - 1 {
                    ep.num_experts
                } else {
                    local_start + per_rank
                };
                idx < local_start || idx >= local_end
            } else {
                false
            }
        }
    }

    /// Synthetic checkpoint (4 experts) mixing every tensor class:
    ///   - a FUSED expert tensor (all experts stacked, no numeric index)
    ///   - 4 PER-FILE-SHARDED expert tensors (`experts.<N>.*`)
    ///   - replicated attention / router / shared-expert / norm tensors
    /// Byte sizes are round numbers so EP division is exact.
    fn synthetic_tensors() -> Vec<(String, Vec<usize>, Dtype)> {
        vec![
            // (a) fused expert tensor: [num_experts, 500] BF16 = 4000 bytes total,
            //     1000 bytes/expert. `experts.gate_up_proj` has no numeric index.
            (
                "model.layers.0.mlp.experts.gate_up_proj.weight".to_string(),
                vec![4, 500],
                Dtype::BF16,
            ),
            // (b) per-file-sharded experts: 4 × [500] BF16 = 1000 bytes each.
            (
                "model.layers.0.mlp.experts.0.down_proj.weight".to_string(),
                vec![500],
                Dtype::BF16,
            ),
            (
                "model.layers.0.mlp.experts.1.down_proj.weight".to_string(),
                vec![500],
                Dtype::BF16,
            ),
            (
                "model.layers.0.mlp.experts.2.down_proj.weight".to_string(),
                vec![500],
                Dtype::BF16,
            ),
            (
                "model.layers.0.mlp.experts.3.down_proj.weight".to_string(),
                vec![500],
                Dtype::BF16,
            ),
            // (c) replicated tensors (600 bytes, must never be divided).
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                vec![100],
                Dtype::BF16,
            ),
            (
                "model.layers.0.mlp.gate.weight".to_string(),
                vec![50],
                Dtype::BF16,
            ),
            (
                "model.layers.0.mlp.shared_experts.gate_proj.weight".to_string(),
                vec![100],
                Dtype::BF16,
            ),
            (
                "model.layers.0.input_layernorm.weight".to_string(),
                vec![50],
                Dtype::BF16,
            ),
        ]
    }

    // Byte budgets of the synthetic checkpoint (see `synthetic_tensors`).
    const FUSED_FULL: usize = 4 * 500 * 2; // 4000
    const SHARDED_EACH: usize = 500 * 2; // 1000
    const REPLICATED: usize = (100 + 50 + 100 + 50) * 2; // 600

    #[test]
    fn is_fused_expert_tensor_predicate() {
        // Fused: `experts.<proj>` (no numeric index).
        assert!(is_fused_expert_tensor(
            "model.layers.0.mlp.experts.gate_up_proj.weight"
        ));
        assert!(is_fused_expert_tensor("x.experts.down_proj.weight_scale"));
        // Per-file-sharded: `experts.<N>.*` is NOT fused (skip filter handles it).
        assert!(!is_fused_expert_tensor(
            "model.layers.0.mlp.experts.3.down_proj.weight"
        ));
        // Replicated shared expert / attention / norm are never fused.
        assert!(!is_fused_expert_tensor(
            "model.layers.0.mlp.shared_experts.gate_proj.weight"
        ));
        assert!(!is_fused_expert_tensor(
            "model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(!is_fused_expert_tensor(
            "model.layers.0.input_layernorm.weight"
        ));
    }

    #[test]
    fn ep1_is_byte_identical_to_no_ep() {
        let t = synthetic_tensors();
        let ep1 = EpEstimate {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 4,
        };
        let total_ep1 = accumulate_load_bytes(&t, &skip_fn(ep1), ep1);
        let total_none =
            accumulate_load_bytes(&t, &skip_fn(EpEstimate::none()), EpEstimate::none());
        // At ep=1 nothing is skipped or divided: every byte counts once.
        let expected = FUSED_FULL + 4 * SHARDED_EACH + REPLICATED; // 8600
        assert_eq!(total_ep1, expected);
        assert_eq!(total_none, expected);
    }

    #[test]
    fn ep_divides_only_fused_expert_bytes() {
        let t = synthetic_tensors();

        // ep=2, rank 0: fused halved (2 of 4 experts), sharded keeps experts {0,1},
        // replicated unchanged.
        let ep2 = EpEstimate {
            ep_rank: 0,
            ep_world_size: 2,
            num_experts: 4,
        };
        let total_ep2 = accumulate_load_bytes(&t, &skip_fn(ep2), ep2);
        assert_eq!(
            total_ep2,
            FUSED_FULL / 2 + 2 * SHARDED_EACH + REPLICATED // 2000 + 2000 + 600 = 4600
        );

        // ep=4, rank 0: fused quartered (1 of 4), sharded keeps expert {0} only.
        let ep4 = EpEstimate {
            ep_rank: 0,
            ep_world_size: 4,
            num_experts: 4,
        };
        let total_ep4 = accumulate_load_bytes(&t, &skip_fn(ep4), ep4);
        assert_eq!(
            total_ep4,
            FUSED_FULL / 4 + SHARDED_EACH + REPLICATED // 1000 + 1000 + 600 = 2600
        );

        // Raising EP strictly lowers the per-node estimate (the #194 property).
        assert!(total_ep4 < total_ep2);
    }

    #[test]
    fn replicated_only_checkpoint_is_ep_invariant() {
        // No expert tensors at all: EP must not change the estimate.
        let t = vec![
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                vec![100],
                Dtype::BF16,
            ),
            (
                "model.layers.0.input_layernorm.weight".to_string(),
                vec![50],
                Dtype::BF16,
            ),
        ];
        let base = accumulate_load_bytes(&t, &skip_fn(EpEstimate::none()), EpEstimate::none());
        for ws in [2usize, 4, 8] {
            let ep = EpEstimate {
                ep_rank: 0,
                ep_world_size: ws,
                num_experts: 4,
            };
            assert_eq!(accumulate_load_bytes(&t, &skip_fn(ep), ep), base);
        }
    }

    #[test]
    fn uneven_division_last_rank_keeps_remainder() {
        // 4 experts across ep=3: ranks get 1, 1, 2. Fused bytes must follow the
        // same per-rank share (last rank keeps the remainder).
        let fused = vec![(
            "l.mlp.experts.gate_up_proj.weight".to_string(),
            vec![4, 500],
            Dtype::BF16,
        )];
        let share = |rank: usize| {
            let ep = EpEstimate {
                ep_rank: rank,
                ep_world_size: 3,
                num_experts: 4,
            };
            accumulate_load_bytes(&fused, &skip_fn(ep), ep)
        };
        // per_rank = 4/3 = 1 → rank0,1 hold 1 expert; rank2 holds 4-2 = 2.
        assert_eq!(share(0), FUSED_FULL * 1 / 4); // 1000
        assert_eq!(share(1), FUSED_FULL * 1 / 4); // 1000
        assert_eq!(share(2), FUSED_FULL * 2 / 4); // 2000
    }
}
