// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time Qwen3.6 NVFP4 repack into vLLM's Marlin MoE layout.

use super::*;
use crate::layers::moe::marlin_scales::{
    combined_scale_factor, process, process_global, transpose_single, transpose_w13,
};

const E: usize = 256;
const H: usize = 2048;
const I: usize = 512;

pub(crate) struct MarlinMoeWeights {
    pub(crate) w13: DevicePtr,
    /// Route-major `[C * topk, 2 * I]` output for the first Marlin GEMM.
    /// It is layer-local so the shared expert can occupy the generic scratch
    /// arena on a second stream without racing the routed Marlin path.
    pub(crate) w13_out: DevicePtr,
    pub(crate) w2: DevicePtr,
    pub(crate) w13_scales: DevicePtr,
    pub(crate) w2_scales: DevicePtr,
    pub(crate) w13_global: DevicePtr,
    pub(crate) w2_global: DevicePtr,
    pub(crate) reduce_tmp: DevicePtr,
    pub(crate) workspace: DevicePtr,
}

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>()
}

fn read_scale(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut host = vec![0; bytes];
    gpu.copy_d2h(ptr, &mut host)?;
    Ok(host)
}

impl MoeLayer {
    pub(crate) fn build_marlin_weights(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        if !ops::marlin_moe::available() {
            return Ok(());
        }
        anyhow::ensure!(
            config.num_experts == E
                && config.hidden_size == H
                && config.moe_intermediate_size == I
                && config.num_experts_per_tok == 8,
            "ATLAS_MOE_MARLIN only supports Qwen3.6-35B geometry E=256,H=2048,I=512,topk=8"
        );
        anyhow::ensure!(
            config.ep_world_size == 1
                && self.weights.experts.len() == E
                && self.weights.experts.iter().all(|x| !x.gate_proj.is_null()),
            "ATLAS_MOE_MARLIN requires all 256 experts resident on one GPU"
        );

        let w13_stride = 2 * I * H / 2;
        let w2_stride = H * I / 2;
        let w13 = gpu.alloc(E * w13_stride)?;
        let w2 = gpu.alloc(E * w2_stride)?;
        let concat = gpu.alloc(w13_stride)?;
        let projection_bytes = I * H / 2;
        for (expert_id, expert) in self.weights.experts.iter().enumerate() {
            anyhow::ensure!(
                expert.gate_proj.weight_scale_2.to_bits()
                    == expert.up_proj.weight_scale_2.to_bits(),
                "expert {expert_id}: gate/up weight_scale_2 mismatch is unsupported by Marlin W13"
            );
            gpu.copy_d2d_async(expert.gate_proj.weight, concat, projection_bytes, stream)?;
            gpu.copy_d2d_async(
                expert.up_proj.weight,
                concat.offset(projection_bytes),
                projection_bytes,
                stream,
            )?;
            ops::marlin_moe::repack(
                concat,
                w13.offset(expert_id * w13_stride),
                H as u32,
                (2 * I) as u32,
                stream,
            )?;
            ops::marlin_moe::repack(
                expert.down_proj.weight,
                w2.offset(expert_id * w2_stride),
                I as u32,
                H as u32,
                stream,
            )?;
        }
        gpu.synchronize(stream)?;
        gpu.free(concat)?;

        let gu_scale_bytes = I * (H / 16);
        let down_scale_bytes = H * (I / 16);
        let mut w13_logical = Vec::with_capacity(E);
        let mut w2_logical = Vec::with_capacity(E);
        for expert in &self.weights.experts {
            let gate = read_scale(gpu, expert.gate_proj.weight_scale, gu_scale_bytes)?;
            let up = read_scale(gpu, expert.up_proj.weight_scale, gu_scale_bytes)?;
            let down = read_scale(gpu, expert.down_proj.weight_scale, down_scale_bytes)?;
            w13_logical.push(transpose_w13(&gate, &up, I, H / 16));
            w2_logical.push(transpose_single(&down, H, I / 16));
        }
        let w13_factor = combined_scale_factor(&w13_logical);
        let w2_factor = combined_scale_factor(&w2_logical);
        let w13_scale_host = w13_logical
            .iter()
            .flat_map(|scale| process(scale, w13_factor))
            .collect::<Vec<_>>();
        let w2_scale_host = w2_logical
            .iter()
            .flat_map(|scale| process(scale, w2_factor))
            .collect::<Vec<_>>();
        let w13_scales = gpu.alloc(w13_scale_host.len())?;
        let w2_scales = gpu.alloc(w2_scale_host.len())?;
        gpu.copy_h2d(&w13_scale_host, w13_scales)?;
        gpu.copy_h2d(&w2_scale_host, w2_scales)?;

        let w13_global_host = f32_bytes(
            self.weights
                .experts
                .iter()
                .map(|x| process_global(x.gate_proj.weight_scale_2, w13_factor)),
        );
        let w2_global_host = f32_bytes(
            self.weights
                .experts
                .iter()
                .map(|x| process_global(x.down_proj.weight_scale_2, w2_factor)),
        );
        let w13_global = gpu.alloc(w13_global_host.len())?;
        let w2_global = gpu.alloc(w2_global_host.len())?;
        gpu.copy_h2d(&w13_global_host, w13_global)?;
        gpu.copy_h2d(&w2_global_host, w2_global)?;

        // vLLM's maximum C_tmp formula is <=3.2 MiB on GB10 for these two
        // exact GEMMs. Keep it layer-local to avoid changing the shared arena.
        // The decode specialization is fixed at C<=8 and topk=8.
        let w13_out = gpu.alloc(8 * 8 * (2 * I) * 2)?;
        let reduce_tmp = gpu.alloc(4 * 1024 * 1024)?;
        let workspace = gpu.alloc(4096)?;
        gpu.memset(workspace, 0, 4096)?;
        self.marlin = Some(MarlinMoeWeights {
            w13,
            w13_out,
            w2,
            w13_scales,
            w2_scales,
            w13_global,
            w2_global,
            reduce_tmp,
            workspace,
        });
        tracing::info!(
            "ATLAS_MOE_MARLIN: repacked 256 experts (W13 factor={w13_factor}, W2 factor={w2_factor})"
        );
        Ok(())
    }
}
