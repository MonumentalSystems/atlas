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
use crate::layers::ops;
use crate::layers::ops::lora_delta::{LoraKernels, LoraPair};
use crate::layers::ops::moe_lora_grouped::{MoeExpertRoute, pack_expert_tables};
use crate::lora::{ExpertLoraLayer, ExpertProj, apply_router_lora};

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
    kernels: LoraKernels,
    /// Row cap the folds chunk to (== scratch capacity in rows).
    cap: u32,
    /// `[cap, max_rank]` BF16 shrink scratch. For the device-side expert
    /// down-fold this is indexed by ABSOLUTE sorted row, so `cap` must cover the
    /// live `total_expanded = num_tokens*top_k` (guarded at the call site).
    xa: DevicePtr,
    /// `[cap, max_n_out]` BF16 expand scratch (router fold only — the device
    /// expert down-fold fuses the fold into the expand kernel and needs no
    /// separate delta buffer).
    delta: DevicePtr,
    /// Feature-1 device-side expert down_proj fold route (per-expert A/B/scale
    /// tables). `None` for a router-only adapter (no `Down` pairs installed).
    expert_route: Option<MoeExpertRoute>,
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
        // Build the device-side per-expert down-fold route: dense [n_experts]
        // u64 A/B pointer tables + f32 scale table, indexed by expert id (0 =
        // unadapted). Load-time-fixed addresses -> stable capture args. `None`
        // for a router-only adapter (no Down pairs).
        let expert_route = Self::build_expert_route(&experts, gpu)?;
        tracing::info!(
            "MoE LoRA installed: router={}, {} expert pair(s), cap={cap} rows, \
             scratch={:.2} MiB",
            router.is_some(),
            experts.pairs.len(),
            (cap * (max_rank.max(1) + max_n_out.max(1)) * 2) as f64 / (1024.0 * 1024.0),
        );
        self.lora = Some(MoeLoraWeights {
            router,
            kernels,
            cap: cap as u32,
            xa,
            delta,
            expert_route,
        });
        Ok(())
    }

    /// Pack the layer's `Down` expert pairs into the device-side per-expert
    /// route tables (`a`/`b` u64, `scale` f32; dense `[n_experts]`, `0` where an
    /// expert is unadapted). `n_experts` is the table length (max adapted id +
    /// 1). Returns `None` for a router-only adapter. `k_in`/`n_out`/`max_rank`
    /// come from the `Down` pairs (uniform per layer — the pool pads all pairs
    /// to the same rank; down maps `moe_inter -> hidden`).
    fn build_expert_route(
        experts: &ExpertLoraLayer,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MoeExpertRoute>> {
        let entries: Vec<(u16, u64, u64, f32)> = experts
            .pairs
            .iter()
            .filter(|((_, p), _)| *p == ExpertProj::Down)
            .map(|((e, _), pair)| (*e, pair.a.weight.0, pair.b.weight.0, pair.scale))
            .collect();
        let Some(tables) = pack_expert_tables(&entries) else {
            return Ok(None);
        };
        // Dims from any Down pair (all identical for this projection on a layer).
        let sample = experts
            .pairs
            .iter()
            .find(|((_, p), _)| *p == ExpertProj::Down)
            .map(|(_, pr)| pr)
            .expect("pack_expert_tables returned Some => a Down pair exists");
        let up = |vals: &[u8]| -> Result<DevicePtr> {
            let d = gpu.alloc(vals.len())?;
            gpu.copy_h2d(vals, d)?;
            Ok(d)
        };
        let a_bytes: Vec<u8> = tables.a.iter().flat_map(|p| p.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = tables.b.iter().flat_map(|p| p.to_le_bytes()).collect();
        let s_bytes: Vec<u8> = tables.scale.iter().flat_map(|s| s.to_le_bytes()).collect();
        Ok(Some(MoeExpertRoute {
            a_table: up(&a_bytes)?,
            b_table: up(&b_bytes)?,
            scale_table: up(&s_bytes)?,
            n_experts: tables.n_experts,
            k_in: sample.k_in,
            n_out: sample.n_out,
            max_rank: sample.max_rank,
        }))
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
    /// post-SiLU sorted activations (`expert_gate_out`).
    ///
    /// DEVICE-SIDE (Feature-1 Incr-1/2): a single two-launch grouped kernel
    /// (`moe_lora_grouped_down`) reads `expert_offsets` on device — NO `copy_d2h`,
    /// NO host launch loop — so it is CUDA-graph-capture LEGAL (unlike the former
    /// host-synced loop). `expert_offsets`/`sorted_token_ids` are device arrays
    /// carved from `gate_logits`; `te = total_expanded = num_tokens*top_k`. One
    /// kernel serves the nvfp4, bf16, and fp8 grouped paths (identical sorted
    /// BF16 `expert_down_out`). No-op when the layer adapts no expert down_proj
    /// (router-only adapter -> `expert_route == None`).
    ///
    /// gate/up-proj folds inject inside `run_routed_grouped_gemm` (before
    /// `silu_mul`); wiring them is a followup — down_proj is the primary.
    pub(crate) fn apply_expert_lora_prefill_down(
        &self,
        expert_gate_out: DevicePtr,
        expert_down_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        te: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else { return Ok(()) };
        let Some(ref route) = l.expert_route else {
            return Ok(()); // router-only adapter: no expert down fold
        };
        // Per-request skip (base / non-active pays nothing; mixed batch REFUSES
        // loudly until the device per-row map lands — Incr-2). This keeps the
        // request-granularity opt-out that lets a base run stay byte-identical.
        if !self.moe_route_gate(ctx, "expert-down")? {
            return Ok(());
        }
        // `xa` is indexed by ABSOLUTE sorted row, so the scratch must cover all
        // `te` rows (the host loop's cap-chunking is gone). Bail loudly rather
        // than overrun — raise ATLAS_LORA_EXPERT_MAX_TOKENS for long prefills.
        anyhow::ensure!(
            te <= l.cap,
            "MoE expert LoRA down-fold: total_expanded ({te}) exceeds LoRA scratch cap ({}); \
             raise ATLAS_LORA_EXPERT_MAX_TOKENS to >= num_tokens*top_k for this prefill chunk.",
            l.cap
        );
        // Incr-1: single active adapter -> moe_row_adapter NULL (the device
        // per-row base skip via ForwardContext.moe_row_adapter is Incr-2; the
        // request-level opt-out above already handles a pure base request).
        ops::moe_lora_grouped_down(
            ctx.gpu,
            &l.kernels,
            route,
            expert_gate_out,
            expert_down_out,
            expert_offsets,
            sorted_token_ids,
            DevicePtr::NULL,
            l.xa,
            te,
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
