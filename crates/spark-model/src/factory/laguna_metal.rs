// SPDX-License-Identifier: AGPL-3.0-only

//! Single-source profile and KV sizing rules for Laguna on Metal.

use anyhow::{Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, KvLayerRetention};

pub fn validate_profile(
    kv_dtype: KvCacheDtype,
    max_seq_len: usize,
    max_batch_size: usize,
    max_prefill_tokens: usize,
    speculative: bool,
    dflash: bool,
    lora: bool,
) -> Result<()> {
    ensure!(
        kv_dtype == KvCacheDtype::Bf16,
        "Laguna Metal v1 requires BF16 KV cache"
    );
    ensure!(
        max_seq_len <= 65_536,
        "Laguna Metal v1 supports at most 65536 context tokens"
    );
    ensure!(
        max_batch_size == 1,
        "Laguna Metal v1 supports --max-batch-size 1"
    );
    ensure!(
        max_prefill_tokens <= 2_048,
        "Laguna Metal v1 supports at most 2048 prefill tokens per chunk"
    );
    ensure!(
        !speculative && !dflash,
        "Laguna Metal v1 does not support speculative decoding or DFlash"
    );
    ensure!(!lora, "Laguna Metal v1 does not support LoRA adapters");
    Ok(())
}

pub fn layer_retention(config: &ModelConfig) -> Vec<KvLayerRetention> {
    config
        .layer_types
        .iter()
        .filter_map(|layer| match layer {
            LayerType::FullAttention => Some(KvLayerRetention::Full),
            LayerType::SlidingAttention => Some(KvLayerRetention::Sliding {
                window_tokens: config.sliding_window as usize,
            }),
            _ => None,
        })
        .collect()
}

pub fn kv_config(
    config: &ModelConfig,
    block_size: usize,
    prefill_chunk_tokens: usize,
) -> KvCacheConfig {
    KvCacheConfig {
        block_size,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        num_layers: config.num_attention_layers(),
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: config.kv_layer_dims.clone(),
        layer_retention: layer_retention(config),
        prefill_chunk_tokens,
        cache_blocks_per_seq: None,
    }
}

pub fn requested_kv_resident_bytes(
    config: &ModelConfig,
    block_size: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    prefill_chunk_tokens: usize,
) -> Result<usize> {
    let logical_blocks = max_seq_len
        .div_ceil(block_size)
        .checked_mul(max_batch_size)
        .ok_or_else(|| anyhow::anyhow!("Laguna KV logical block count overflow"))?;
    kv_config(config, block_size, prefill_chunk_tokens).resident_bytes_for_blocks(logical_blocks)
}
