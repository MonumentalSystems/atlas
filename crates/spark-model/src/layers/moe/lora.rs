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
//!
//! HARDENING (this pass): the fold is now (a) zero-overhead when off (the
//! `self.lora == None` early-return, unchanged) AND per-request — a base or
//! non-active request SKIPs so base tokens pay nothing (`moe_route_gate` reading
//! `ctx.moe_lora_route`, the request-granularity mirror of the attention BGMV
//! `seq_slot < 0` skip); (b) graph-safe by REFUSAL — the `expert_offsets` D2H is
//! guarded by `ctx.graph_capture` and refuses loudly rather than silently
//! corrupting a captured graph; a packed/mixed batch (`MoeLoraRoute::Refuse`)
//! also bails rather than fold one adapter onto rows it does not own. The
//! device-side grouped fold that reads `expert_offsets` on device (removing the
//! D2H entirely, enabling capture + mixed batches) is the follow-up:
//! `docs/design/lora-solid.md` Incr 1-3.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::MoeLayer;
use crate::layer::{ForwardContext, MoeLoraRoute};
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
        // Phase-1 folds only expert down_proj (apply_expert_lora_prefill_down).
        // A gate/up-proj expert pair would be stored but never folded — refuse
        // loudly rather than silently ignore it (gate/up fold is a followup).
        if let Some(((e, proj), _)) = experts
            .pairs
            .iter()
            .find(|((_, p), _)| *p != ExpertProj::Down)
        {
            anyhow::bail!(
                "MoE LoRA (Feature-1) folds only expert down_proj in phase 1; adapter \
                 targets expert {e} {proj:?}-proj. Refusing rather than silently dropping \
                 it — restrict target_modules to down_proj, or wait for the gate/up fold."
            );
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

    /// Per-request fold gate (Feature-1). Returns `true` to fold, `false` to
    /// skip with zero overhead, or a loud error to refuse. Consults
    /// `ctx.moe_lora_route` (resolved from the request's `adapter_slot`):
    /// `Skip` (base / non-active) folds nothing so base tokens pay nothing;
    /// `Refuse` (packed/mixed batch or a non-installed adapter) bails rather
    /// than fold one adapter onto rows it does not own — the device-side
    /// per-row fold that would skip base rows individually is the follow-up
    /// (`docs/design/lora-solid.md` Incr 1/3). Called only after the caller has
    /// confirmed `self.lora.is_some()`, so it never fires on a base run.
    fn moe_route_gate(&self, ctx: &ForwardContext, path: &str) -> Result<bool> {
        match ctx.moe_lora_route {
            MoeLoraRoute::Fold => Ok(true),
            MoeLoraRoute::Skip => Ok(false),
            MoeLoraRoute::Refuse => anyhow::bail!(
                "MoE LoRA (Feature-1) cannot honor per-row adapter identity in this {path} pass \
                 (packed/mixed batch, or a non-active adapter under single-active phase-1); \
                 refusing rather than folding one adapter onto rows it does not own. The \
                 device-side per-row grouped fold is the follow-up (docs/design/lora-solid.md \
                 Incr 1/3)."
            ),
        }
    }

    /// Fold the router LoRA delta onto `gate_logits` (`[n, num_experts]`) in
    /// place, BEFORE top-k. No-op when the layer has no router delta or the
    /// request opts out (base / non-active). Called from `forward_prefill` right
    /// after the base gate GEMM. Graph-safe (pure `apply_lora_delta` kernels, no
    /// host D2H) so it needs no capture guard.
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
        if !self.moe_route_gate(ctx, "router")? {
            return Ok(());
        }
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
        // Per-request skip (base / non-active pays nothing; mixed batch refuses).
        if !self.moe_route_gate(ctx, "expert-down")? {
            return Ok(());
        }
        // GRAPH-SAFETY: the D2H of `expert_offsets` below is a blocking host copy
        // (status 900 = STREAM_CAPTURE_UNSUPPORTED inside a capture region) AND it
        // drives a data-dependent host launch loop. Under CUDA-graph capture this
        // silently corrupts the captured graph. Refuse LOUDLY instead. The fix is
        // to consume `expert_offsets` device-side in a single grouped kernel (the
        // base grouped GEMM already reads it as a kernel arg — never D2Hs it);
        // that device-side grouped fold is the documented follow-up
        // (docs/design/lora-solid.md Incr 1-2). Until it lands, expert LoRA stays
        // eager-prefill only — never captured, never on the decode/verify path
        // (reject_decode_lora), so this guard only fires if a future capture path
        // reaches the host-loop fold before the device kernel replaces it.
        if ctx.graph_capture {
            anyhow::bail!(
                "MoE expert LoRA down-fold host-copies expert_offsets (graph-breaking D2H + \
                 host-driven launch count); refusing under CUDA-graph capture rather than \
                 corrupting the captured graph. The device-side grouped fold that reads \
                 expert_offsets on device is the follow-up (docs/design/lora-solid.md Incr 1-2)."
            );
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
