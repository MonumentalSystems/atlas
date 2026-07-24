// SPDX-License-Identifier: AGPL-3.0-only

//! Keep-packed GGUF/Laguna state assembled by the MoE constructor.

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use crate::weight_map::{PackedExpertStack, PackedExpertWeights, QuantWeight};

pub(super) struct PackedGgufInit<'a> {
    gpu: &'a dyn GpuBackend,
    packed_path: bool,
    pub(super) gate_stack: Option<PackedExpertStack>,
    pub(super) up_stack: Option<PackedExpertStack>,
    pub(super) down_stack: Option<PackedExpertStack>,
}

impl<'a> PackedGgufInit<'a> {
    pub(super) fn new(
        experts: Option<&[PackedExpertWeights]>,
        gpu: &'a dyn GpuBackend,
    ) -> Result<Self> {
        let packed_path = experts.is_some();
        let (gate_stack, up_stack, down_stack) = if let Some(experts) = experts {
            let gate = experts.iter().map(|expert| expert.gate).collect::<Vec<_>>();
            let up = experts.iter().map(|expert| expert.up).collect::<Vec<_>>();
            let down = experts
                .iter()
                .map(|expert| match expert.down {
                    QuantWeight::PackedQ4(weight) => Some(weight),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>();
            (
                Some(PackedExpertStack::from_q4_views(&gate)?),
                Some(PackedExpertStack::from_q4_views(&up)?),
                down.as_deref()
                    .map(PackedExpertStack::from_q4_views)
                    .transpose()?,
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            gpu,
            packed_path,
            gate_stack,
            up_stack,
            down_stack,
        })
    }

    /// Packed checkpoints load only the active kernel subset; other formats
    /// preserve the constructor's required-kernel failure behavior.
    pub(super) fn required_kernel(&self, module: &str, function: &str) -> Result<KernelHandle> {
        if self.packed_path {
            Ok(super::super::try_kernel(self.gpu, module, function))
        } else {
            self.gpu.kernel(module, function)
        }
    }

    /// Resolve optional packed-GGUF kernels at the original struct-field
    /// evaluation point, preserving constructor lookup order.
    pub(super) fn kernel(&self, module: &str, function: &str) -> KernelHandle {
        super::super::try_kernel(self.gpu, module, function)
    }
}
