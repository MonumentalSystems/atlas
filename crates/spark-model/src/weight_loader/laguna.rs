// SPDX-License-Identifier: AGPL-3.0-only

//! Poolside Laguna-S-2.1 weight loader (target model only; DFlash is separate).

mod load_layers;
mod packed_gguf;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, PackedQ8Weight, dense};

pub struct LagunaWeightLoader;

impl ModelWeightLoader for LagunaWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.get("model.embed_tokens.weight")?.is_packed_q8_0() {
            Ok(DenseWeight {
                weight: DevicePtr::NULL,
            })
        } else {
            dense(store, "model.embed_tokens.weight")
        }
    }

    fn load_packed_q8_embedding(&self, store: &WeightStore) -> Result<Option<PackedQ8Weight>> {
        let name = "model.embed_tokens.weight";
        store
            .get(name)?
            .is_packed_q8_0()
            .then(|| PackedQ8Weight::from_store(store, name))
            .transpose()
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.get("lm_head.weight")?.is_packed_q8_0() {
            Ok(DenseWeight {
                weight: DevicePtr::NULL,
            })
        } else {
            dense(store, "lm_head.weight")
        }
    }

    fn load_packed_q8_lm_head(&self, store: &WeightStore) -> Result<Option<PackedQ8Weight>> {
        let name = "lm_head.weight";
        store
            .get(name)?
            .is_packed_q8_0()
            .then(|| PackedQ8Weight::from_store(store, name))
            .transpose()
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
