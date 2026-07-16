// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 MoE LoRA on [`MoeLayer`]: installed router + routed-expert deltas,
//! apply scratch, the prefill fold hooks, and the decode-path refusal.
//!
//! CORRECTNESS-FIRST (phase 1): the base grouped GEMM stays byte-identical; a
//! BF16 delta is folded onto its output buffers via `crate::lora::expert_apply`
//! (which wraps `apply_lora_delta` — no new CUDA kernel). Single active adapter,
//! host-synced `expert_offsets` (graph-breaking, legal in eager prefill). The
//! decode/verify forward paths REFUSE (never silently drop the delta) — wiring
//! the unsorted per-token decode fold is the phase-1 followup.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::MoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops::lora_delta::{LoraKernels, LoraPair};
use crate::lora::{ExpertLoraLayer, ExpertProj, apply_expert_lora_sorted, apply_router_lora};

/// Per-token cap for the LoRA apply scratch (`ATLAS_LORA_EXPERT_MAX_TOKENS`,
/// default 4096). Folds over more rows than this are chunked; scratch is sized
/// from it, so a huge prefill chunk stays bounded. Read once.
fn max_tokens() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ATLAS_LORA_EXPERT_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&t: &u32| t > 0)
            .unwrap_or(4096)
    })
}

/// One MoE layer's installed router + routed-expert LoRA + apply scratch.
pub(crate) struct MoeLoraWeights {
    /// Router (`mlp.gate`) delta on the routing logits (`None` if unadapted).
    router: Option<LoraPair>,
    /// This layer's sparse per-expert pairs.
    experts: ExpertLoraLayer,
    kernels: LoraKernels,
    /// Row cap the folds chunk to (== scratch capacity in rows).
    cap: u32,
    /// `[cap, max_rank]` BF16 shrink scratch.
    xa: DevicePtr,
    /// `[cap, max_n_out]` BF16 expand scratch.
    delta: DevicePtr,
}

impl MoeLayer {
    /// Install this layer's router + routed-expert LoRA. Allocates apply scratch
    /// sized from the pairs' max `n_out` / `max_rank` and the token cap. A layer
    /// with neither a router nor any expert pair installs nothing.
    pub(crate) fn set_lora_weights(
        &mut self,
        router: Option<LoraPair>,
        experts: ExpertLoraLayer,
        kernels: LoraKernels,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if router.is_none() && experts.is_empty() {
            self.lora = None;
            return Ok(());
        }
        let all = router.iter().chain(experts.pairs.values());
        let max_n_out = all.clone().map(|p| p.n_out).max().unwrap_or(0) as usize;
        let max_rank = all.map(|p| p.max_rank).max().unwrap_or(0) as usize;
        let cap = max_tokens() as usize;
        let xa = gpu.alloc(cap * max_rank.max(1) * 2)?;
        let delta = gpu.alloc(cap * max_n_out.max(1) * 2)?;
        gpu.memset(xa, 0, cap * max_rank.max(1) * 2)?;
        gpu.memset(delta, 0, cap * max_n_out.max(1) * 2)?;
        tracing::info!(
            "MoE LoRA installed: router={}, {} expert pair(s), cap={cap} rows, \
             scratch={:.2} MiB",
            router.is_some(),
            experts.pairs.len(),
            (cap * (max_rank.max(1) + max_n_out.max(1)) * 2) as f64 / (1024.0 * 1024.0),
        );
        self.lora = Some(MoeLoraWeights {
            router,
            experts,
            kernels,
            cap: cap as u32,
            xa,
            delta,
        });
        Ok(())
    }

    /// Fold the router LoRA delta onto `gate_logits` (`[n, num_experts]`) in
    /// place, BEFORE top-k. No-op when the layer has no router delta. Called
    /// from `forward_prefill` right after the base gate GEMM.
    pub(crate) fn apply_router_lora_prefill(
        &self,
        router_in: DevicePtr,
        gate_logits: DevicePtr,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else { return Ok(()) };
        let Some(ref rp) = l.router else { return Ok(()) };
        apply_router_lora(
            ctx.gpu, &l.kernels, rp, router_in, gate_logits, n, l.cap, l.xa, l.delta, stream,
        )
    }

    /// Fold the routed-expert down_proj LoRA deltas onto the sorted
    /// `expert_down_out` (`[total_expanded, hidden]`), BEFORE the unpermute +
    /// weighted reduce (so the router weight multiplies base+delta). `x` = the
    /// post-SiLU sorted activations (`expert_gate_out`). D2H-copies
    /// `expert_offsets` to drive the host per-expert loop (graph-breaking —
    /// eager prefill only). No-op when the layer adapts no expert down_proj.
    ///
    /// gate/up-proj folds inject inside `run_routed_grouped_gemm` (before
    /// `silu_mul`); wiring them is a phase-1 followup — down_proj is the
    /// primary, per the design.
    pub(crate) fn apply_expert_lora_prefill_down(
        &self,
        expert_gate_out: DevicePtr,
        expert_down_out: DevicePtr,
        expert_offsets: DevicePtr,
        num_experts: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else { return Ok(()) };
        if l.experts.is_empty() {
            return Ok(());
        }
        // Host-sync the expert row boundaries ([num_experts + 1] u32).
        let ne1 = num_experts as usize + 1;
        let mut bytes = vec![0u8; ne1 * 4];
        ctx.gpu.copy_d2h(expert_offsets, &mut bytes)?;
        let off_host: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        apply_expert_lora_sorted(
            ctx.gpu,
            &l.kernels,
            &l.experts,
            ExpertProj::Down,
            &off_host,
            expert_gate_out,
            expert_down_out,
            l.cap,
            l.xa,
            l.delta,
            stream,
        )
    }

    /// Phase-1 guard: the decode/verify MoE forward paths do not yet fold the
    /// expert/router delta (unsorted per-token top-k dispatch). Rather than
    /// silently serve wrong output, REFUSE when an expert adapter is installed.
    /// A no-op when no MoE LoRA is present (base decode byte-identical).
    pub(crate) fn reject_decode_lora(&self, path: &str) -> Result<()> {
        if self.lora.is_some() {
            anyhow::bail!(
                "MoE LoRA (Feature-1) is prefill-only in phase 1; the {path} decode/verify \
                 path does not yet fold the expert/router delta. Use the adapter for \
                 prefill-logit scoring, or wait for the phase-1 decode-fold followup."
            );
        }
        Ok(())
    }
}
