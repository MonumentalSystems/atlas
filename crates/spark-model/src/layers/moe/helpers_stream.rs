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

use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, bail};

use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_storage::expert::Proj;

use super::ExpertPtrTable;
use super::ExpertStreamerShared;
use super::MoeLayer;
use crate::layer::ForwardContext;

/// Max prefill token count that takes the REACTIVE (expert-granular) path.
/// `M <= threshold` fetches only the layer's active experts; `M > threshold`
/// keeps the DENSE layer-ahead path (bit-identical to pre-reactive behavior).
/// Default 1 (decode-via-prefill only). Set `ATLAS_EXPERT_REACTIVE_MAX_TOKENS`
/// large to force the reactive path on the bit-identical prefill gate.
pub(super) fn reactive_max_tokens() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ATLAS_EXPERT_REACTIVE_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(1)
    })
}

/// Derive the ACTIVE local expert set from an `expert_offsets` host readback.
///
/// `offsets` is the `(num_experts + 1)` element, little-endian **i32**
/// exclusive-prefix-sum produced by `moe_sort_by_expert`; expert `e` routed
/// `offsets[e+1] - offsets[e]` tokens. The returned ids are exactly
/// `{ e in [lo, hi) : count(e) > 0 }` — the precise fetch set the routed
/// grouped GEMM will dispatch on (missing one → garbage; extra → wasted I/O).
/// EP scope (invariant E): only local experts `[lo, hi)` are considered; a
/// count>0 expert outside the local range is intentionally not fetched.
pub(super) fn active_experts_from_offsets(offsets: &[u8], lo: usize, hi: usize) -> Vec<u32> {
    let read = |i: usize| -> i32 {
        i32::from_le_bytes([
            offsets[i * 4],
            offsets[i * 4 + 1],
            offsets[i * 4 + 2],
            offsets[i * 4 + 3],
        ])
    };
    let mut out = Vec::new();
    for e in lo..hi {
        if read(e + 1) - read(e) > 0 {
            out.push(e as u32);
        }
    }
    out
}

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
    /// Decide whether this prefill takes the REACTIVE expert-granular path, and
    /// if so return its active local expert set (else `None` → dense path).
    ///
    /// Reactive iff streaming is on, `num_tokens <= reactive_max_tokens()`, and
    /// graphs are off (invariant F: the D2H readback below is illegal mid-graph;
    /// streaming already forces eager decode). The readback is a `(ne+1)*4`-byte
    /// stream-synchronizing copy of `expert_offsets` — the same pattern the
    /// `ATLAS_MOE_PREFILL_EXACT_TILES` path uses. Runs BETWEEN `moe_sort_by_expert`
    /// and `install_streamed_tables`, where the active set is known on device.
    pub(super) fn compute_reactive_active(
        &self,
        ctx: &ForwardContext,
        expert_offsets: DevicePtr,
        num_experts: usize,
        num_tokens: usize,
        stream: u64,
    ) -> Result<Option<Vec<u32>>> {
        if self.streamer.is_none() || ctx.graph_capture || num_tokens as u32 > reactive_max_tokens()
        {
            return Ok(None);
        }
        let (lo, hi) = ctx.config.local_expert_range();
        let mut offsets = vec![0u8; (num_experts + 1) * 4];
        ctx.gpu
            .copy_d2h_on_stream(expert_offsets, &mut offsets, stream)?;
        Ok(Some(active_experts_from_offsets(&offsets, lo, hi)))
    }

    /// Patch this layer's transposed pointer tables from the streamed arena,
    /// consuming the residencies the prefetch worker already fetched (blocking
    /// only if the fetch hasn't finished). No-op when streaming is disabled.
    ///
    /// `active`:
    ///   * `None`      → DENSE path: fetch/patch the whole local range `[lo, hi)`
    ///                   (worker prefetched it during the prior layer's compute).
    ///   * `Some(ids)` → REACTIVE path: just-in-time fetch/patch ONLY `ids` (the
    ///                   count>0 experts). Inactive table entries keep their prior
    ///                   value; the grouped GEMM never dereferences them.
    pub(super) fn install_streamed_tables(
        &self,
        ctx: &ForwardContext,
        _stream: u64,
        active: Option<&[u32]>,
    ) -> Result<()> {
        let Some(streamer) = self.streamer.as_ref() else {
            return Ok(());
        };
        // The default NVFP4 prefill path reads the transposed tables (fused K64
        // gate/up + moe_w4a16_grouped_gemm_ptrtable_n128 for down, which reads
        // down_ptrs_t when it is Some). Patch all three *_ptrs_t; the store holds
        // gate/up/down all transposed. (ATLAS_HOLO_MOE_* paths are unsupported.)
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

        // Build the (global expert id -> residency) patch list. The DENSE path
        // fetches the full local range (prefetched during the prior layer's
        // compute); the REACTIVE path just-in-time fetches only the active ids.
        let patch: Vec<(usize, spark_storage::expert_tier::ExpertResidency)> = match active {
            None => {
                // DENSE: idempotent prime (layer 0 primes itself here), then take
                // the full-range residencies (indexed by expert - lo).
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
                residencies
                    .into_iter()
                    .enumerate()
                    .map(|(i, res)| (lo + i, res))
                    .collect()
            }
            Some(ids) => {
                // REACTIVE: wait for the prior GEMM on this slab to finish
                // (invariant C — the worker is about to overwrite active slots),
                // then fetch ONLY the active experts and zip positionally with
                // `ids` (worker preserves the passed order).
                streamer.wait_slab_free(dense)?;
                streamer.prefetch_sparse(dense, ids, lo as u32, hi as u32);
                let residencies = streamer
                    .take_active(dense)
                    .with_context(|| format!("stream take_active layer {dense}"))?;
                if residencies.len() != ids.len() {
                    bail!(
                        "layer {dense}: got {} reactive residencies, expected {} active experts",
                        residencies.len(),
                        ids.len()
                    );
                }
                ids.iter().map(|&e| e as usize).zip(residencies).collect()
            }
        };

        // Read current (resident) tables into host shadows (invariant A: we
        // mutate contents, not the allocations). Reading preserves every entry
        // we don't patch (inactive experts, remote experts) at its prior value.
        let mut g = Shadow::read(gpu, gt, n)?;
        let mut u = Shadow::read(gpu, ut, n)?;
        let mut d = Shadow::read(gpu, dt, n)?;

        // Patch only local experts (invariant E) from the fetched residencies.
        let gi = Proj::Gate as usize;
        let ui = Proj::Up as usize;
        let di = Proj::Down as usize;
        for (e, res) in &patch {
            let e = *e;
            let res = *res;
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
    /// C) and — on the DENSE path — kick off the layer-ahead prefetch of the
    /// next MoE layer so its I/O overlaps this layer's compute. No-op when
    /// streaming is disabled.
    ///
    /// `reactive`: on the reactive path there is NO layer-ahead prefetch — the
    /// next layer's active set is unknown until its own router runs (that
    /// latency-hiding is the deferred Gate-0(a) predictor). We still record the
    /// slab-consumed event so the reactive fetch of THIS layer next token (which
    /// re-overwrites the same slab) can `wait_slab_free` on it.
    pub(super) fn after_streamed_layer(
        &self,
        ctx: &ForwardContext,
        stream: u64,
        reactive: bool,
    ) -> Result<()> {
        let Some(streamer) = self.streamer.as_ref() else {
            return Ok(());
        };
        let dense = self.stream_dense_idx;
        // Record on `stream` that the GPU has finished reading this layer's slab.
        streamer.record_consumed(dense, stream)?;
        if reactive {
            return Ok(());
        }
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

#[cfg(test)]
mod tests {
    use super::active_experts_from_offsets;

    /// Encode an exclusive-prefix-sum of per-expert counts into the (ne+1) i32
    /// little-endian byte layout `moe_sort_by_expert` produces.
    fn offsets_bytes(counts: &[i32]) -> Vec<u8> {
        let mut acc = 0i32;
        let mut prefix = Vec::with_capacity(counts.len() + 1);
        prefix.push(0i32);
        for &c in counts {
            acc += c;
            prefix.push(acc);
        }
        prefix.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn active_set_is_exactly_count_gt_zero() {
        // 6 experts, only 1, 3, 4 routed tokens (counts 2, 1, 5).
        let counts = [0, 2, 0, 1, 5, 0];
        let bytes = offsets_bytes(&counts);
        let active = active_experts_from_offsets(&bytes, 0, 6);
        assert_eq!(active, vec![1, 3, 4]);
    }

    #[test]
    fn active_set_respects_local_range_ep_scope() {
        // Same counts, but this EP rank only owns experts [2, 5). A count>0
        // expert outside the local range (expert 1) must NOT be fetched.
        let counts = [0, 2, 0, 1, 5, 0];
        let bytes = offsets_bytes(&counts);
        let active = active_experts_from_offsets(&bytes, 2, 5);
        assert_eq!(active, vec![3, 4]);
    }

    #[test]
    fn decode_m1_yields_at_most_top_k() {
        // M=1, top_k=8: exactly 8 distinct experts each with count 1.
        let mut counts = vec![0i32; 256];
        for e in [3, 17, 42, 99, 128, 200, 201, 255] {
            counts[e] = 1;
        }
        let bytes = offsets_bytes(&counts);
        let active = active_experts_from_offsets(&bytes, 0, 256);
        assert_eq!(active, vec![3, 17, 42, 99, 128, 200, 201, 255]);
        assert_eq!(active.len(), 8);
    }

    #[test]
    fn empty_active_set_when_no_local_expert_routed() {
        let counts = [0, 5, 0, 0];
        let bytes = offsets_bytes(&counts);
        // local range [2,4) sees no routed expert.
        assert!(active_experts_from_offsets(&bytes, 2, 4).is_empty());
    }
}
