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
    pub(super) kernels: LoraKernels,
    /// Row cap the folds chunk to (== scratch capacity in rows).
    pub(super) cap: u32,
    /// `[cap, max_rank]` BF16 shrink scratch. For the device-side expert
    /// folds this is indexed by ABSOLUTE sorted row (prefill) / flat slot
    /// (decode), so `cap` must cover the live `total_expanded = num_tokens*top_k`
    /// (guarded at the call site). Reused SERIALLY across gate/up/down folds on
    /// the same stream (each shrink fully precedes its expand).
    pub(super) xa: DevicePtr,
    /// `[cap, max_n_out]` BF16 expand scratch (router fold + the decode down-fold
    /// post-swiglu `x` recompute — the device grouped folds fuse into the expand
    /// kernel and need no separate delta buffer).
    delta: DevicePtr,
    /// Feature-1 device-side expert down_proj fold route (per-expert A/B/scale
    /// tables; `k_in=moe_inter`, `n_out=hidden`). `None` for an adapter with no
    /// `Down` pairs.
    expert_route: Option<MoeExpertRoute>,
    /// Expert gate_proj fold route (`k_in=hidden`, `n_out=moe_inter` — the
    /// TRANSPOSE of down). `None` when no `Gate` pairs are installed.
    pub(super) gate_route: Option<MoeExpertRoute>,
    /// Expert up_proj fold route (same dims as gate). `None` when no `Up` pairs.
    pub(super) up_route: Option<MoeExpertRoute>,
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
        // All three expert projections (gate/up/down) now fold on device via their
        // own per-expert route table (built below); the router delta folds on the
        // routing logits. No proj is silently dropped, so no install-time bail.
        let all = router.iter().chain(experts.pairs.values());
        let max_n_out = all.clone().map(|p| p.n_out).max().unwrap_or(0) as usize;
        let max_k_in = all.clone().map(|p| p.k_in).max().unwrap_or(0) as usize;
        let max_rank = all.map(|p| p.max_rank).max().unwrap_or(0) as usize;
        let cap = max_tokens() as usize;
        // `delta` is the router expand scratch (`[cap, max_n_out]`) AND, for the
        // decode expert down-fold, the packed `[n_slots, k_in]` post-swiglu `x`
        // recompute scratch — size it for the wider of the two so a model whose
        // `moe_intermediate_size` (k_in) exceeds `hidden` (n_out) still fits.
        let delta_cols = max_n_out.max(max_k_in).max(1);
        let xa = gpu.alloc(cap * max_rank.max(1) * 2)?;
        let delta = gpu.alloc(cap * delta_cols * 2)?;
        gpu.memset(xa, 0, cap * max_rank.max(1) * 2)?;
        gpu.memset(delta, 0, cap * delta_cols * 2)?;
        // Build the device-side per-expert down-fold route: dense [n_experts]
        // u64 A/B pointer tables + f32 scale table, indexed by expert id (0 =
        // unadapted). Load-time-fixed addresses -> stable capture args. `None`
        // for a router-only adapter (no Down pairs).
        let expert_route = Self::build_expert_route(&experts, ExpertProj::Down, gpu)?;
        let gate_route = Self::build_expert_route(&experts, ExpertProj::Gate, gpu)?;
        let up_route = Self::build_expert_route(&experts, ExpertProj::Up, gpu)?;
        tracing::info!(
            "MoE LoRA installed: router={}, {} expert pair(s) (gate={} up={} down={}), \
             cap={cap} rows, scratch={:.2} MiB",
            router.is_some(),
            experts.pairs.len(),
            gate_route.as_ref().map_or(0, |r| r.n_experts),
            up_route.as_ref().map_or(0, |r| r.n_experts),
            expert_route.as_ref().map_or(0, |r| r.n_experts),
            (cap * (max_rank.max(1) + delta_cols) * 2) as f64 / (1024.0 * 1024.0),
        );
        self.lora = Some(MoeLoraWeights {
            router,
            kernels,
            cap: cap as u32,
            xa,
            delta,
            expert_route,
            gate_route,
            up_route,
        });
        Ok(())
    }

    /// Pack the layer's `proj` expert pairs into the device-side per-expert
    /// route tables (`a`/`b` u64, `scale` f32; dense `[n_experts]`, `0` where an
    /// expert is unadapted). `n_experts` is the table length (max adapted id + 1)
    /// for THAT proj (each fold launches its own `grid.z`, so per-proj lengths are
    /// independent). Returns `None` when no pair targets `proj`. `k_in`/`n_out`/
    /// `max_rank` come from any `proj` pair (uniform per layer — the pool pads all
    /// pairs to the same rank): down maps `moe_inter -> hidden`, gate/up the
    /// transpose `hidden -> moe_inter`.
    pub(super) fn build_expert_route(
        experts: &ExpertLoraLayer,
        proj: ExpertProj,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MoeExpertRoute>> {
        let entries: Vec<(u16, u64, u64, f32)> = experts
            .pairs
            .iter()
            .filter(|((_, p), _)| *p == proj)
            .map(|((e, _), pair)| (*e, pair.a.weight.0, pair.b.weight.0, pair.scale))
            .collect();
        let Some(tables) = pack_expert_tables(&entries) else {
            return Ok(None);
        };
        // Dims from any `proj` pair (all identical for this projection on a layer).
        let sample = experts
            .pairs
            .iter()
            .find(|((_, p), _)| *p == proj)
            .map(|(_, pr)| pr)
            .expect("pack_expert_tables returned Some => a matching pair exists");
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
    pub(super) fn moe_route_gate(&self, ctx: &ForwardContext, path: &str) -> Result<bool> {
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
    /// after the base gate GEMM AND from the single-seq decode `forward` right
    /// after the gate GEMV (n=1) — this hook is n-generic. Graph-safe (pure
    /// `apply_lora_delta` kernels, no host D2H) so it needs no capture guard.
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
    /// The sibling gate/up-proj folds (`apply_expert_lora_prefill_gateup`, in
    /// `moe/lora_gateup.rs`) inject inside `run_routed_grouped_gemm` before
    /// `silu_mul`; this down fold runs after the down GEMM.
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
            0, // x_gather=0: down x is already sorted (x-row == sorted row)
            stream,
        )
    }

    /// SOLID Incr-4: fold the routed-expert down_proj LoRA deltas onto the
    /// slot-major decode `expert_down_out` (`[n_slots, hidden]`), IN PLACE and
    /// BEFORE `moe_weighted_sum_blend` (so the router weight multiplies
    /// base+delta — same ordering as the prefill fold). `n_slots =
    /// num_tokens*top_k` flat `(token, slot)` rows; `indices_dev` is the same
    /// `[n_slots]` u32 expert-id array the fused expert GEMV routed on.
    ///
    /// `x = silu(gate)*up` is recomputed into the fixed `l.delta` scratch via the
    /// EXISTING `moe_silu_mul` kernel (`self.moe_act_mul` — the SAME handle +
    /// ±10 swiglu clamp + BF16 round the prefill fold's `x` uses), so the folded
    /// delta is BF16-ULP identical to `apply_expert_lora_prefill_down`. The fold
    /// itself is `moe_lora_gather_bgmv` — the unsorted, expert-gather analogue of
    /// the prefill `moe_lora_grouped_down`, sharing its byte-identical reduction
    /// body. Zero-overhead when off (`self.lora == None` early-return); a
    /// router-only adapter (`expert_route == None`) also no-ops here.
    ///
    /// `row_adapter` is the device `[num_tokens]` i32 per-row base map (`< 0` =
    /// base skip) or `DevicePtr::NULL` to fold every row on the single-active
    /// homogeneous path — the caller (a genuine single-token decode) passes NULL
    /// and relies on `moe_route_gate` for the request-granularity opt-out (`Skip`
    /// folds nothing; `Refuse` bails). CUDA-graph-capture legal: all args are
    /// pointer/value-stable, and the launch shape is exact per captured graph.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_expert_lora_decode_down(
        &self,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        indices_dev: DevicePtr,
        n_slots: u32,
        top_k: u32,
        row_adapter: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else { return Ok(()) };
        let Some(ref route) = l.expert_route else {
            return Ok(()); // router-only adapter: no expert down fold
        };
        // SOLID Incr-4: when a per-row `row_adapter` map is supplied (batched
        // decode), the launch is UNCONDITIONAL and route-agnostic — base rows
        // no-op device-side via `row_adapter[row/top_k] < 0`, so the fold is
        // captured into the shared `padded_n` graph and replays correctly across
        // arbitrary base/adapter compositions. Consult the request-granularity
        // `moe_route_gate` ONLY on the single-seq NULL-map path (its `Skip`/
        // `Refuse` host branch would otherwise bake one route into the graph).
        if row_adapter == DevicePtr::NULL && !self.moe_route_gate(ctx, "expert-down-decode")? {
            return Ok(());
        }
        anyhow::ensure!(
            n_slots <= l.cap,
            "MoE expert LoRA decode down-fold: n_slots ({n_slots}) exceeds LoRA scratch cap \
             ({}); raise ATLAS_LORA_EXPERT_MAX_TOKENS to >= num_tokens*top_k.",
            l.cap
        );
        // x = silu(gate)*up -> BF16 into l.delta (prefill's EXACT boundary: same
        // kernel, same clamp, same round). Packed [n_slots, k_in] contiguous.
        ops::moe_silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            expert_gate_out,
            expert_up_out,
            l.delta,
            n_slots * route.k_in,
            stream,
        )?;
        ops::moe_lora_gather_bgmv(
            ctx.gpu,
            &l.kernels,
            route,
            l.delta,
            expert_down_out,
            indices_dev,
            row_adapter,
            l.xa,
            n_slots,
            top_k,
            0, // x_gather=0: decode down x is the packed per-slot post-swiglu activation
            stream,
        )
    }

    /// Phase-1 guard: the decode/verify MoE forward paths do not yet fold the
    /// expert/router delta (unsorted per-token top-k dispatch). Rather than
    /// silently serve wrong output, REFUSE when an expert adapter is installed.
    /// A no-op when no MoE LoRA is present (base decode byte-identical).
    pub(crate) fn reject_decode_lora(&self, ctx: &ForwardContext, path: &str) -> Result<()> {
        // Route-aware: bail ONLY when this decode actually routes to the adapter
        // (Fold/Refuse). A pure-base decode batch (Skip) has no delta to fold, so
        // base requests decode normally even while an adapter is resident — the
        // decode route is stamped per batch at the Model entry
        // (`stamp_decode_moe_*`). The per-row decode fold is SOLID Incr-4.
        if self.lora.is_some() && !matches!(ctx.moe_lora_route, MoeLoraRoute::Skip) {
            anyhow::bail!(
                "MoE LoRA (Feature-1) is prefill-only in phase 1; the {path} decode/verify \
                 path does not yet fold the expert/router delta for an adapter-routed request. \
                 Use the adapter for prefill-logit scoring, or wait for the decode-fold followup \
                 (docs/design/lora-solid.md Incr-4)."
            );
        }
        Ok(())
    }
}
