// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer full/sliding KV allocation and logical-to-physical block mapping.

use anyhow::{Context, Result, bail};

use super::{KvCacheConfig, KvLayerRetention, LayerPool, PagedKvCache};
use crate::gpu::GpuBackend;

impl KvCacheConfig {
    /// Exact K+V bytes allocated for a logical full-attention pool of
    /// `full_blocks` blocks, including bounded sliding-layer rings.
    pub fn resident_bytes_for_blocks(&self, full_blocks: usize) -> Result<usize> {
        (0..self.num_layers).try_fold(0usize, |total, layer_idx| {
            let blocks = self.physical_blocks_for_layer(layer_idx, full_blocks);
            let bytes_per_block = self
                .k_block_bytes_for_layer(layer_idx)
                .checked_add(self.v_block_bytes_for_layer(layer_idx))
                .context("KV K+V block byte size overflow")?;
            let layer_bytes = blocks
                .checked_mul(bytes_per_block)
                .context("KV layer resident byte size overflow")?;
            total
                .checked_add(layer_bytes)
                .context("KV resident byte total overflow")
        })
    }
}

impl PagedKvCache {
    /// Allocate the KV cache pool on the GPU.
    pub fn new(config: KvCacheConfig, num_blocks: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut total_bytes: usize = 0;
        for i in 0..config.num_layers {
            // Per-side block_bytes: for symmetric dtypes both are equal; for
            // asymmetric (e.g. Bf16KTurbo3V) the K pool is allocated bf16-sized
            // and the V pool is allocated turbo3-sized — avoids the 4× V over-
            // allocation that would result from a single MAX-sized stride.
            let k_block_bytes = config.k_block_bytes_for_layer(i);
            let v_block_bytes = config.v_block_bytes_for_layer(i);
            let physical_blocks = config.physical_blocks_for_layer(i, num_blocks);
            let k_pool_bytes = physical_blocks * k_block_bytes;
            let v_pool_bytes = physical_blocks * v_block_bytes;
            let k_pool = gpu.alloc(k_pool_bytes)?;
            let v_pool = gpu.alloc(v_pool_bytes)?;
            total_bytes += k_pool_bytes + v_pool_bytes;
            layers.push(LayerPool {
                k_pool,
                v_pool,
                k_block_stride: k_block_bytes,
                v_block_stride: v_block_bytes,
                dtype: config.dtype_for_layer(i),
                physical_blocks,
                retention: config.retention_for_layer(i),
            });
        }

        let free_blocks: Vec<u32> = (0..num_blocks as u32).rev().collect();
        let block_ref_counts = vec![0u32; num_blocks];

        let has_mixed = !config.layer_dtypes.is_empty()
            && config.layer_dtypes.iter().any(|d| *d != config.dtype);
        if has_mixed {
            let hp_count = config
                .layer_dtypes
                .iter()
                .filter(|d| **d != config.dtype)
                .count();
            tracing::info!(
                "KV cache: {} blocks × {} layers ({} high-precision) = {:.1} GB total (mixed dtype)",
                num_blocks,
                config.num_layers,
                hp_count,
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        } else {
            tracing::info!(
                "KV cache: {} logical blocks × {} layers = {:.1} GB resident",
                num_blocks,
                config.num_layers,
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }

        Ok(Self {
            layers,
            num_blocks,
            free_blocks,
            block_ref_counts,
            config,
        })
    }

    /// Map a shared logical block id to this layer's physical storage.
    /// Full layers preserve the id. Sliding layers use a bounded ring; Metal
    /// append/attention kernels must apply the same mapping before indexing the
    /// pool. The mapping is intentionally explicit so a backend cannot silently
    /// treat a bounded layer as a full paged cache.
    pub fn physical_block_for_layer(&self, layer_idx: usize, block_idx: u32) -> u32 {
        let layer = &self.layers[layer_idx];
        match layer.retention {
            KvLayerRetention::Full => block_idx,
            KvLayerRetention::Sliding { .. } => block_idx % layer.physical_blocks as u32,
        }
    }

    /// Number of physical K/V blocks allocated for a layer.
    pub fn physical_blocks_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].physical_blocks
    }

    /// Compute how many blocks can fit given available GPU memory.
    /// Accounts for mixed dtypes and fixed sliding-window rings.
    pub fn compute_num_blocks(config: &KvCacheConfig, available_bytes: usize) -> Result<usize> {
        let mut full_bytes_per_block = 0usize;
        let mut sliding_fixed_bytes = 0usize;
        for layer_idx in 0..config.num_layers {
            let layer_bytes = config.k_block_bytes_for_layer(layer_idx)
                + config.v_block_bytes_for_layer(layer_idx);
            match config.retention_for_layer(layer_idx) {
                KvLayerRetention::Full => full_bytes_per_block += layer_bytes,
                KvLayerRetention::Sliding { window_tokens } => {
                    let ring_blocks = window_tokens
                        .saturating_add(config.prefill_chunk_tokens)
                        .div_ceil(config.block_size)
                        + 1;
                    sliding_fixed_bytes =
                        sliding_fixed_bytes.saturating_add(ring_blocks.saturating_mul(layer_bytes));
                }
            }
        }
        if full_bytes_per_block == 0 {
            bail!("KV cache block size is zero");
        }
        Ok(available_bytes.saturating_sub(sliding_fixed_bytes) / full_bytes_per_block)
    }
}
