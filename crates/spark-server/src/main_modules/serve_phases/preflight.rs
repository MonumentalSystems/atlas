// SPDX-License-Identifier: AGPL-3.0-only

//! GPU init + pre-load reserve preflight + post-load OOM check.

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) struct ReservePreflight {
    pub(crate) inference_reserve: usize,
    pub(crate) buffer_arena_bytes: usize,
    pub(crate) gdn_two_phase_bytes: usize,
    pub(crate) ssm_prefill_chunk: usize,
    pub(crate) max_batch_tokens_pre: usize,
}

/// Operator hint: a large resident Marconi SSM pool with the spill tier OFF is
/// holding every live conversation's whole checkpoint chain in HBM. With
/// `ATLAS_SSM_TIER=1` (host-RAM or RDMA spill) the resident pool becomes a hot
/// cache and can shrink to a small size, reclaiming most of that HBM. Returns
/// the hint when it applies (SSM model, tier off, pool ≥ `HINT_SLOTS`); pure so
/// the threshold + arithmetic are unit-tested.
fn ssm_pool_shrink_hint(
    ssm_layers: usize,
    slots: usize,
    per_slot_bytes: usize,
    tier_enabled: bool,
) -> Option<String> {
    const HINT_SLOTS: usize = 64; // at/below this the pool is already small
    const SHRINK_TARGET: usize = 16;
    if ssm_layers == 0 || tier_enabled || slots < HINT_SLOTS {
        return None;
    }
    let mb = |s: usize| s * per_slot_bytes / (1024 * 1024);
    Some(format!(
        "resident Marconi SSM pool = {slots} slots ({} MB) with the spill tier OFF \
         (ATLAS_SSM_TIER unset). Enabling the tier makes the pool a hot cache — shrink to \
         --ssm-cache-slots ~{SHRINK_TARGET} to reclaim ~{} MB HBM (the tier holds the \
         overflow and faults back on warm hits; --ssm-fault-min-tokens guards shallow prefixes).",
        mb(slots),
        mb(slots.saturating_sub(SHRINK_TARGET)),
    ))
}

pub(crate) fn preflight_reserve(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    free_mem: usize,
) -> Result<ReservePreflight> {
    let h_state_bytes = config.ssm_h_state_bytes();
    let conv_state_bytes = config.ssm_conv_state_bytes();
    let ssm_multiplier = if args.speculative || args.self_speculative || args.ngram_speculative {
        1 + (args.num_drafts + 1) + 1
    } else {
        1
    };
    let ssm_pool_bytes = args.max_batch_size
        * config.num_ssm_layers()
        * (h_state_bytes + conv_state_bytes)
        * ssm_multiplier;
    let spec_tokens_pre = if args.speculative || args.self_speculative || args.ngram_speculative {
        args.num_drafts + 2
    } else {
        1
    };
    // B4 (chunked-prefill BF16 KV cliff): the prior `.min(8192)` cap forced
    // every prompt > 8 k to chunk, which compounds K-side BF16 rounding noise
    // at chunk boundaries (per the 4-agent audit 2026-05-27). When the user
    // explicitly passes `--max-prefill-tokens N` (anything other than the
    // default 8192), respect it — no hard cap. Otherwise default to 8192 to
    // bound GDN persistent-buffer reservation for unbounded `max_seq_len`.
    let ssm_prefill_chunk: usize = if config.num_ssm_layers() > 0 {
        if args.max_prefill_tokens != 8192 && args.max_prefill_tokens > 0 {
            args.max_seq_len.min(args.max_prefill_tokens)
        } else {
            args.max_seq_len.min(8192)
        }
    } else {
        0
    };
    let user_set_prefill_pre = args.max_prefill_tokens != 8192;
    let prefill_budget_pre = if user_set_prefill_pre && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else if ssm_prefill_chunk > 0 {
        ssm_prefill_chunk
    } else if args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        args.max_seq_len
    };
    // Issue #15 auto-clamp removed (2026-07-02): snapshot reachability is
    // handled by the tail-checkpoint split in `prefill_chunk_dispatch`, so
    // the budget (and this arena-sizing mirror) stays at full chunk size.
    let max_batch_tokens_pre = prefill_budget_pre
        .max(spec_tokens_pre)
        .max(args.max_batch_size);
    let buffer_arena_bytes = spark_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens_pre,
        args.max_seq_len,
        args.block_size,
    )
    .total_bytes();
    // SSM snapshot pool = Marconi prefix-cache region + Phase-C
    // decode-rollback ring. The decode ring is sized per active
    // sequence (`DECODE_ROLLBACK_RING_SLOTS` slots × `max_batch_size`),
    // and only allocated for SSM models. SSOT: this reservation MUST use
    // the SAME constant the pool actually allocates with
    // (`SsmSnapshotPool::new` in `impl_a1.rs` uses
    // `DECODE_ROLLBACK_RING_SLOTS`). It previously used
    // `ROLLBACK_RESTEER_CAP + 1` (= 3) while the pool allocated
    // `DECODE_ROLLBACK_RING_SLOTS` (= 8), under-reserving the SSM-snapshot
    // GPU budget by `(8 - 3) × max_batch_size × num_ssm_layers ×
    // (h_bytes + conv_bytes)` — the two constants were decoupled when the
    // ring was widened past the rollback cap.
    // SSOT with `SsmSnapshotPool::new`: the decode HBM region is
    // `decode_hbm_lanes_per_seq × max_batch_size` frames, PLUS the shared
    // fault-scratch pool in ROLLING mode (ATLAS_SSM_DECODE_RING_ROLL) — where the
    // per-seq HBM lanes shrink from 8 to `hot_lanes + margin` and the deep tail
    // spills to ATLAS_SSM_DECODE_TIER. Both this reservation and the pool alloc
    // call the same `atlas_kernels` helper so they can never drift.
    let decode_frames = if config.num_ssm_layers() > 0 {
        let rolling = atlas_kernels::decode_ring_rolling_enabled();
        let hot_lanes = atlas_kernels::decode_hot_lanes_runtime();
        let per_seq = atlas_kernels::decode_hbm_lanes_per_seq(rolling, hot_lanes);
        let scratch = if rolling {
            atlas_kernels::DECODE_FAULT_SCRATCH
        } else {
            0
        };
        per_seq * args.max_batch_size + scratch
    } else {
        0
    };
    let ssm_snapshot_bytes = (args.ssm_cache_slots + decode_frames)
        * config.num_ssm_layers()
        * (h_state_bytes + conv_state_bytes);
    let cuda_headroom: usize =
        if args.speculative || args.self_speculative || args.ngram_speculative {
            4 * 1024 * 1024 * 1024
        } else {
            512 * 1024 * 1024
        };
    let gdn_two_phase_bytes: usize = {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        if conv_dim > 0 && config.num_ssm_layers() > 0 {
            let sl = max_batch_tokens_pre;
            sl * conv_dim * 2 + sl * nv * 2 * 4 + sl * value_dim * 2 + sl * value_dim * 2
        } else {
            0
        }
    };
    let inference_reserve: usize =
        ssm_pool_bytes + ssm_snapshot_bytes + gdn_two_phase_bytes + cuda_headroom;
    let total_reserve = inference_reserve + buffer_arena_bytes;
    if total_reserve > free_mem {
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_gb = free_mem as f64 / (1024.0 * 1024.0 * 1024.0);
        let fixed = ssm_pool_bytes + ssm_snapshot_bytes + cuda_headroom;
        let budget_for_seq_term = free_mem.saturating_sub(fixed) / 2;
        let per_tok_bytes = {
            let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
            let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
            let nv = config.linear_num_value_heads;
            let conv_dim = key_dim * 2 + value_dim;
            if conv_dim > 0 && config.num_ssm_layers() > 0 {
                (conv_dim * 2) + (nv * 2 * 4) + (value_dim * 2) + (value_dim * 2)
            } else {
                0
            }
        };
        let suggested = budget_for_seq_term
            .checked_div(per_tok_bytes)
            .map(|q| q.max(2048))
            .unwrap_or(0);
        let hint = if suggested > 0 && suggested < args.max_seq_len {
            format!(
                " Try --max-seq-len {} (or lower --max-batch-size / --num-drafts).",
                suggested
            )
        } else if args.max_batch_size > 1 {
            " Reduce --max-batch-size.".to_string()
        } else {
            " Use a smaller model or a GPU with more memory.".to_string()
        };
        anyhow::bail!(
            "Preflight failed: inference buffers alone need {:.2} GB but only {:.2} GB is free on the GPU \
             (before weights load). SSM pool + GDN chunked prefill scales with --max-seq-len={} × --max-batch-size={}.{}",
            need_gb,
            free_gb,
            args.max_seq_len,
            args.max_batch_size,
            hint,
        );
    }
    tracing::info!(
        "Preflight reserve: inference={} MB, buffer_arena={} MB (pre-load free: {:.1} GB)",
        inference_reserve / (1024 * 1024),
        buffer_arena_bytes / (1024 * 1024),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    if let Some(hint) = ssm_pool_shrink_hint(
        config.num_ssm_layers(),
        args.ssm_cache_slots,
        config.num_ssm_layers() * (h_state_bytes + conv_state_bytes),
        std::env::var_os("ATLAS_SSM_TIER").is_some(),
    ) {
        tracing::info!("SSM pool right-sizing: {hint}");
    }
    // Q09: per-component breakdown so future MTP/spec-decode reserve
    // jumps are diagnosable from the log alone. Each line is dropped at
    // debug to avoid noise on hot startup paths; flip to info if you
    // need to trace a specific deployment's reserve.
    let spec_on = args.speculative || args.self_speculative || args.ngram_speculative;
    tracing::debug!(
        "Preflight reserve breakdown: \
         ssm_pool={} MB ({}× max_batch × {} ssm_layers × (h+conv)), \
         ssm_snapshot={} MB ({} slots), \
         gdn_two_phase={} MB ({} tokens), \
         cuda_headroom={} MB ({}), \
         spec_on={}, num_drafts={}",
        ssm_pool_bytes / (1024 * 1024),
        ssm_multiplier,
        config.num_ssm_layers(),
        ssm_snapshot_bytes / (1024 * 1024),
        args.ssm_cache_slots,
        gdn_two_phase_bytes / (1024 * 1024),
        max_batch_tokens_pre,
        cuda_headroom / (1024 * 1024),
        if spec_on { "spec/MTP on" } else { "no spec" },
        spec_on,
        if spec_on { args.num_drafts as i64 } else { -1 },
    );
    Ok(ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
    })
}

/// Initialize the GPU backend for the active feature.
///
/// Compile-time dispatch:
/// - `cuda` feature → `AtlasCudaBackend` loading PTX modules from `ptx_set`.
/// - `metal` feature → `MetalGpuBackend` loading metallib modules from
///   `atlas_kernels::metallib_modules()`. The `ptx_set` argument is
///   accepted (for ABI symmetry with the cuda variant) but ignored;
///   metal kernels live in a parallel registry.
#[cfg(feature = "cuda")]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(
        spark_runtime::cuda_backend::AtlasCudaBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize CUDA backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    // Baseline for self-relative KV budgeting: free memory now (post context +
    // PTX modules, pre weights) minus free-at-build = this process's own
    // footprint, co-tenants excluded. See gpu::baseline_free_bytes.
    spark_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "GPU {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    _ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    let modules = atlas_kernels::metallib_modules();
    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(
        spark_runtime::metal_backend::MetalGpuBackend::new(args.gpu_ordinal, &modules)
            .context("Failed to initialize Metal backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    spark_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "Metal device {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn post_load_memory_audit(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    weight_bytes: usize,
    free_mem: usize,
    inference_reserve: usize,
    total_reserve: usize,
    gdn_two_phase_bytes: usize,
    max_batch_tokens_pre: usize,
) -> Result<()> {
    let estimated_free = free_mem.saturating_sub(weight_bytes);
    let actual_free = gpu.free_memory().unwrap_or(estimated_free);
    let available_free = if actual_free > 0 {
        actual_free
    } else {
        estimated_free
    };
    if available_free < total_reserve {
        let avail_gb = available_free as f64 / (1024.0 * 1024.0 * 1024.0);
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let hint = if args.max_batch_size > 1 {
            format!(
                " Reduce --max-batch-size (currently {}) or --max-seq-len (currently {}).",
                args.max_batch_size, args.max_seq_len
            )
        } else {
            format!(
                " Reduce --max-seq-len (currently {}) or use a smaller model.",
                args.max_seq_len
            )
        };
        anyhow::bail!(
            "Insufficient GPU memory for inference buffers. \
             After loading {:.2} GB of weights, only {:.2} GB remains \
             but {:.2} GB is needed for SSM state pool ({} slots × {} layers) + scratch buffers.{}",
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            avail_gb,
            need_gb,
            args.max_batch_size,
            config.num_ssm_layers(),
            hint,
        );
    }
    if gdn_two_phase_bytes > 0 {
        tracing::info!(
            "GDN chunked prefill reserve: {} MB (chunk_size={}, max_seq_len={})",
            gdn_two_phase_bytes / (1024 * 1024),
            max_batch_tokens_pre,
            args.max_seq_len,
        );
    }
    tracing::info!(
        "Weights: {:.2} GB, estimated free: {:.1} GB, actual free: {:.1} GB (reserve: {} MB)",
        weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        estimated_free as f64 / (1024.0 * 1024.0 * 1024.0),
        actual_free as f64 / (1024.0 * 1024.0 * 1024.0),
        inference_reserve / (1024 * 1024),
    );
    Ok(())
}

#[cfg(test)]
mod ssm_hint_tests {
    use super::ssm_pool_shrink_hint;
    const PER_SLOT: usize = 64 * 1024 * 1024; // ~64 MB/slot (Holo 35B scale)

    #[test]
    fn fires_for_large_pool_tier_off() {
        let h = ssm_pool_shrink_hint(48, 256, PER_SLOT, false).expect("should hint");
        assert!(h.contains("256 slots (16384 MB)"), "{h}");
        // reclaim ~ (256-16) * 64 MB = 15360 MB
        assert!(h.contains("15360 MB"), "{h}");
    }
    #[test]
    fn silent_when_tier_on() {
        assert!(ssm_pool_shrink_hint(48, 256, PER_SLOT, true).is_none());
    }
    #[test]
    fn silent_when_pool_already_small() {
        assert!(ssm_pool_shrink_hint(48, 16, PER_SLOT, false).is_none());
        assert!(ssm_pool_shrink_hint(48, 63, PER_SLOT, false).is_none());
    }
    #[test]
    fn silent_for_non_ssm_model() {
        assert!(ssm_pool_shrink_hint(0, 256, PER_SLOT, false).is_none());
    }
}
