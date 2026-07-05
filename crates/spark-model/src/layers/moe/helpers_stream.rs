// SPDX-License-Identifier: AGPL-3.0-only
//
// Runtime install of streamed expert weights into the prefill pointer tables.
//
// Called once per MoE layer, immediately before its routed grouped GEMM. For
// each LOCAL expert it fetches the resident-layout record into the ring arena
// (blocking, Stage 2) and patches the transposed pointer tables
// (`gate_ptrs_t`/`up_ptrs_t`/`down_ptrs_t`) to the fetched addresses.
//
// Invariants upheld here:
//   * A (tables immortal): patches the CONTENTS of the existing device arrays;
//     the `ExpertPtrTable` allocations built at load are never re-created.
//   * B (batched patch): mutates a host shadow of each `[num_experts]` array,
//     then issues exactly ONE `copy_h2d` per array (9 total: packed/scale/scale2
//     × gate/up/down) — never a per-expert copy.
//   * E (EP scope): only experts in `local_expert_range` are patched; remote
//     experts keep their load-time value (NULL / DevicePtr(0)).

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_storage::expert::Proj;

use super::ExpertPtrTable;
use super::ExpertStreamerShared;
use super::MoeLayer;
use crate::layer::ForwardContext;

impl MoeLayer {
    /// Attach the shared expert streamer + this layer's dense MoE index. Called
    /// by the loader when `--stream-experts` is set.
    pub(crate) fn set_expert_streamer(
        &mut self,
        streamer: Option<Arc<ExpertStreamerShared>>,
        dense_idx: u32,
    ) {
        self.streamer = streamer;
        self.stream_dense_idx = dense_idx;
    }
}

/// Read a device `[n]` u64 array back to host.
fn read_u64(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<u64>> {
    let mut bytes = vec![0u8; n * 8];
    gpu.copy_d2h(ptr, &mut bytes)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect())
}

/// Read a device `[n]` f32 array back to host.
fn read_f32(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    gpu.copy_d2h(ptr, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect())
}

fn write_u64(gpu: &dyn GpuBackend, arr: &[u64], ptr: DevicePtr) -> Result<()> {
    let bytes: Vec<u8> = arr.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, ptr)
}

fn write_f32(gpu: &dyn GpuBackend, arr: &[f32], ptr: DevicePtr) -> Result<()> {
    let bytes: Vec<u8> = arr.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, ptr)
}

/// Host shadow of one `ExpertPtrTable` (the three `[num_experts]` arrays).
struct Shadow {
    packed: Vec<u64>,
    scale: Vec<u64>,
    scale2: Vec<f32>,
}

impl Shadow {
    fn read(gpu: &dyn GpuBackend, t: &ExpertPtrTable, n: usize) -> Result<Self> {
        Ok(Self {
            packed: read_u64(gpu, t.packed_ptrs, n)?,
            scale: read_u64(gpu, t.scale_ptrs, n)?,
            scale2: read_f32(gpu, t.scale2_vals, n)?,
        })
    }
    fn write(&self, gpu: &dyn GpuBackend, t: &ExpertPtrTable) -> Result<()> {
        write_u64(gpu, &self.packed, t.packed_ptrs)?;
        write_u64(gpu, &self.scale, t.scale_ptrs)?;
        write_f32(gpu, &self.scale2, t.scale2_vals)?;
        Ok(())
    }
}

impl MoeLayer {
    /// Patch this layer's transposed pointer tables from the streamed arena,
    /// consuming the residencies the prefetch worker already fetched (blocking
    /// only if the fetch hasn't finished). No-op when streaming is disabled.
    pub(super) fn install_streamed_tables(&self, ctx: &ForwardContext, _stream: u64) -> Result<()> {
        let Some(streamer) = self.streamer.as_ref() else {
            return Ok(());
        };
        // The default NVFP4 prefill path reads the transposed tables; require
        // them (the ATLAS_HOLO_MOE_* CUTLASS/untransposed paths read different
        // buffers and are unsupported while streaming).
        let (Some(gt), Some(ut), Some(dt)) = (
            self.gate_ptrs_t.as_ref(),
            self.up_ptrs_t.as_ref(),
            self.down_ptrs_t.as_ref(),
        ) else {
            bail!(
                "expert streaming requires transposed prefill tables (gate_ptrs_t); \
                 disable ATLAS_HOLO_MOE_* / ensure transpose_for_prefill ran"
            );
        };
        let gpu = ctx.gpu;
        let n = self.weights.experts.len();
        let (lo, hi) = ctx.config.local_expert_range();
        let dense = self.stream_dense_idx;

        // Ensure this layer is being prefetched (idempotent — layer 0 primes
        // itself here; later layers were prefetched during the prior layer's
        // compute), then take the fetched residencies (indexed by expert - lo).
        streamer.prefetch(dense, lo as u32, hi as u32);
        let residencies = streamer
            .take(dense)
            .with_context(|| format!("stream take layer {dense}"))?;
        if residencies.len() != hi - lo {
            bail!(
                "layer {dense}: got {} residencies, expected {} local experts",
                residencies.len(),
                hi - lo
            );
        }

        // Read current (resident) tables into host shadows (invariant A: we
        // mutate contents, not the allocations).
        let mut g = Shadow::read(gpu, gt, n)?;
        let mut u = Shadow::read(gpu, ut, n)?;
        let mut d = Shadow::read(gpu, dt, n)?;

        // Patch only local experts (invariant E) from the prefetched residencies.
        let gi = Proj::Gate as usize;
        let ui = Proj::Up as usize;
        let di = Proj::Down as usize;
        for (i, res) in residencies.iter().enumerate() {
            let e = lo + i;
            g.packed[e] = res.packed_addr[gi];
            g.scale[e] = res.scale_addr[gi];
            g.scale2[e] = res.scale2[gi];
            u.packed[e] = res.packed_addr[ui];
            u.scale[e] = res.scale_addr[ui];
            u.scale2[e] = res.scale2[ui];
            d.packed[e] = res.packed_addr[di];
            d.scale[e] = res.scale_addr[di];
            d.scale2[e] = res.scale2[di];
        }

        // One copy_h2d per array (invariant B): 9 total, independent of expert
        // count. copy_h2d is synchronous, so the host shadows are safe to drop
        // and the patched tables are visible to the GEMM that follows on `stream`.
        g.write(gpu, gt)?;
        u.write(gpu, ut)?;
        d.write(gpu, dt)?;
        Ok(())
    }

    /// After this layer's grouped GEMM: mark its arena slab consumed (invariant
    /// C) and kick off the prefetch of the next MoE layer so its I/O overlaps
    /// this layer's compute. No-op when streaming is disabled.
    pub(super) fn after_streamed_layer(&self, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let Some(streamer) = self.streamer.as_ref() else {
            return Ok(());
        };
        let dense = self.stream_dense_idx;
        // Record on `stream` that the GPU has finished reading this layer's slab.
        streamer.record_consumed(dense, stream)?;
        let next = dense + 1;
        if next < streamer.num_moe_layers() {
            let (lo, hi) = ctx.config.local_expert_range();
            // Deferred-free: don't let the worker overwrite the next slab until
            // its previous occupant's GEMM has completed on the GPU.
            streamer.wait_slab_free(next)?;
            streamer.prefetch(next, lo as u32, hi as u32);
        }
        Ok(())
    }
}
