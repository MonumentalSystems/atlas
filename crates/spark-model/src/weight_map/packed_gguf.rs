// SPDX-License-Identifier: AGPL-3.0-only

//! Typed views over GGUF weights that remain block-packed in device memory.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::WeightStore;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedGgufQuantType {
    Q4K,
    Q8_0,
}

/// One contiguous `[experts, N, K]` packed allocation.
#[derive(Debug, Clone, Copy)]
pub struct PackedExpertStack {
    pub base: DevicePtr,
    pub expert_stride_bytes: u64,
    pub quant_type: PackedGgufQuantType,
    pub num_experts: u32,
    pub n: u32,
    pub k: u32,
}

impl PackedExpertStack {
    pub fn from_q4_views(views: &[crate::weight_map::PackedQ4Weight]) -> Result<Self> {
        let first = views
            .first()
            .ok_or_else(|| anyhow::anyhow!("packed expert stack is empty"))?;
        let stride = u64::from(first.n)
            .checked_mul(u64::from(first.k / 256))
            .and_then(|blocks| blocks.checked_mul(144))
            .ok_or_else(|| anyhow::anyhow!("packed Q4_K expert stride overflow"))?;
        ensure!(
            first.k.is_multiple_of(256),
            "Q4_K K must be divisible by 256"
        );
        for (expert, view) in views.iter().enumerate() {
            ensure!(
                view.n == first.n && view.k == first.k,
                "expert {expert} shape differs from expert 0"
            );
            ensure!(
                view.weight.0 == first.weight.0 + stride * expert as u64,
                "expert {expert} is not contiguous in its GGUF stack"
            );
        }
        Ok(Self {
            base: first.weight,
            expert_stride_bytes: stride,
            quant_type: PackedGgufQuantType::Q4K,
            num_experts: views.len() as u32,
            n: first.n,
            k: first.k,
        })
    }
}

/// A row-major GGUF Q8_0 matrix kept as native 34-byte blocks.
///
/// Every block represents 32 consecutive values as an inline fp16 scale and
/// 32 signed int8 quants. The backing allocation is owned by [`WeightStore`];
/// this descriptor is a non-owning view used by Metal launch wrappers.
#[derive(Debug, Clone, Copy)]
pub struct PackedQ8Weight {
    pub weight: DevicePtr,
    /// Output rows (`N`) of the logical `[N, K]` matrix.
    pub n: u32,
    /// Input columns (`K`), always a multiple of 32.
    pub k: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct PackedQ8SharedExpert {
    pub gate_proj: PackedQ8Weight,
    pub up_proj: PackedQ8Weight,
    pub down_proj: PackedQ8Weight,
}

impl PackedQ8Weight {
    pub fn is_null(&self) -> bool {
        self.weight == DevicePtr::NULL
    }

    /// Build a checked view over a keep-packed store tensor.
    pub fn from_store(store: &WeightStore, name: &str) -> Result<Self> {
        let tensor = store.get(name)?;
        ensure!(
            tensor.is_packed_q8_0(),
            "{name} is not keep-packed GGUF Q8_0"
        );
        ensure!(
            tensor.shape.len() == 2,
            "{name} must be rank 2, got {:?}",
            tensor.shape
        );
        ensure!(
            tensor.shape[1] % 32 == 0,
            "{name} contraction dimension {} is not a multiple of 32",
            tensor.shape[1]
        );
        Ok(Self {
            weight: tensor.ptr,
            n: tensor.shape[0] as u32,
            k: tensor.shape[1] as u32,
        })
    }
}
