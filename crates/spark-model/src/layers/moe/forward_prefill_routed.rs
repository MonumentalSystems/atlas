// SPDX-License-Identifier: AGPL-3.0-only

//! Routed grouped-GEMM phase of `MoeLayer::forward_prefill`.
//!
//! Hoisted from `forward_prefill.rs` to keep that file under the 500 LoC
//! cap. The single entry point [`MoeLayer::run_routed_grouped_gemm`]
//! mirrors the original block 1:1 — same control flow, same kernel
//! launches, same buffer wiring. Covers steps 4-6 of the prefill
//! pipeline: grid sizing, grouped gate+up GEMM, SiLU, grouped down GEMM.

use super::*;

/// Whether the single-launch CUTLASS grouped NVFP4 gate_up path is enabled
/// (`ATLAS_HOLO_MOE_GROUPED_CUTLASS=1`). Off by default; falls back to the
/// hand-rolled fused FP4/FP8 grouped kernels when unset.
fn grouped_cutlass_gate_up_enabled() -> bool {
    std::env::var("ATLAS_HOLO_MOE_GROUPED_CUTLASS")
        .ok()
        .as_deref()
        == Some("1")
}

/// Whether the CUTLASS grouped path builds its per-group problem sizes ON
/// DEVICE (`ATLAS_MOE_CUTLASS_DEVICE_OFFSETS=1`). Same GEMM kernel, but no
/// host D2H of `expert_offsets` and a fixed `sm_count` launch grid
/// (`host_problem_shapes = nullptr`) — CUDA-graph-capture-legal, so multi-seq
/// decode keeps its graphs while the grouped CUTLASS MoE fires (see the
/// `grouped_is_graph_safe` gate in `decode_a2.rs`). Off by default: the
/// shipped host-driven (eager-under-decode) path is unchanged.
pub(crate) fn grouped_cutlass_device_offsets_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_MOE_CUTLASS_DEVICE_OFFSETS")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Lower bound (inclusive) on `total_expanded` for the small-M FP4 decode GEMV
/// to fire. The slot-major W4A16 GEMV only BEATS the CUTLASS/K64 tensor-core
/// path once concurrency is high enough to amortize its per-slot bf16-A
/// staging: measured on Laguna-S-2.1 (GB10), smallm is +7.9 tok/s at
/// total_expanded=80 (padded n=8) but −5.8 at 40 (padded n=4). Decode graphs
/// pad n to {2,4,8} so total_expanded is discrete (20/40/80 at top_k=10); a
/// default of 64 cleanly selects the n=8 tier only and leaves n<=4 on CUTLASS.
/// Tunable via `ATLAS_MOE_FP4_DECODE_SMALLM_MIN`; 0 = fire at any M (old
/// behavior).
pub(crate) fn fp4_decode_smallm_min() -> u32 {
    use std::sync::OnceLock;
    static MIN: OnceLock<u32> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("ATLAS_MOE_FP4_DECODE_SMALLM_MIN")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(64)
    })
}

impl MoeLayer {
    /// Routed-expert grouped-GEMM path: upper-bound grid sizing → grouped
    /// gate+up GEMM → SiLU+mul → grouped down GEMM.
    ///
    /// Writes the routed expert outputs into `ctx.buffers.expert_down_out()`.
    /// `t0` carries the running profile timer so per-step timing output
    /// matches the original inline pipeline exactly.
    #[allow(clippy::too_many_arguments)]
    // sfb Options are guarded by `.is_some()` in the enclosing `if`; `.expect`
    // after that is intentional (can't if-let-bind inside the `&&` guard chain).
    #[allow(clippy::unnecessary_unwrap)]
    pub(super) fn run_routed_grouped_gemm(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        num_tokens: usize,
        ne: usize,
        t0: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        macro_rules! prof_step {
            ($label:expr) => {
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    *t0 = Some(std::time::Instant::now());
                }
            };
        }

        let avg_per_expert = (num_tokens * top_k as usize).div_ceil(ne);
        // Default to the absolute worst case (one expert receives every routed
        // token) to prevent silent truncation. An opt-in load-factor cap lets
        // Holo experiments trade that safety margin for fewer empty expert
        // tiles after validating the router histogram.
        let worst_case_m_tiles = (num_tokens * top_k as usize).div_ceil(64).max(1) as u32;
        let exact_tiles = std::env::var("ATLAS_MOE_PREFILL_EXACT_TILES")
            .ok()
            .as_deref()
            == Some("1")
            && !ctx.graph_capture;
        let max_m_tiles = if exact_tiles {
            let mut offsets = vec![0u8; (ne + 1) * 4];
            ctx.gpu
                .copy_d2h_on_stream(expert_offsets, &mut offsets, stream)?;
            let mut prev = 0u32;
            let mut max_rows = 0u32;
            for raw in offsets.chunks_exact(4).skip(1) {
                let cur = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                max_rows = max_rows.max(cur.saturating_sub(prev));
                prev = cur;
            }
            max_rows.div_ceil(64).max(1).min(worst_case_m_tiles)
        } else {
            std::env::var("ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&factor| factor > 0)
                .map(|factor| {
                    let capped_rows = avg_per_expert.saturating_mul(factor);
                    worst_case_m_tiles.min(capped_rows.div_ceil(64).max(1) as u32)
                })
                .unwrap_or(worst_case_m_tiles)
        };
        super::dump::dump_expert_load(
            ctx.gpu,
            stream,
            expert_offsets,
            ne,
            num_tokens,
            avg_per_expert,
            max_m_tiles,
        );
        prof_step!("grid_setup");

        let total_expanded = n * top_k;

        // 5. Grouped gate+up GEMM — cp.async pipelined FP8-MMA K64 (transposed).
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        // EP remote experts return without writing, so the destination must be
        // zeroed before dispatch. In non-EP, `moe_sort_by_expert` produces a
        // dense token_to_perm over exactly [0, total_expanded), and grouped
        // kernels write every row that can be referenced by unpermute_reduce.
        // Skipping the memset removes ~138 MB/layer of scratch clears on Holo.
        let force_zero = std::env::var("ATLAS_MOE_PREFILL_ZERO").ok().as_deref() == Some("1");
        if ctx.comm.is_some() || force_zero {
            let gate_bytes = total_expanded as usize * inter as usize * 2;
            let up_bytes = gate_bytes;
            let down_bytes = total_expanded as usize * h as usize * 2;
            ctx.gpu
                .memset_async(expert_gate_out, 0, gate_bytes, stream)?;
            ctx.gpu.memset_async(expert_up_out, 0, up_bytes, stream)?;
            ctx.gpu
                .memset_async(ctx.buffers.expert_down_out(), 0, down_bytes, stream)?;
        }
        // Host expert_offsets from the CUTLASS gate_up, reused by down to skip
        // a second D2H + host-blocking synchronize.
        let mut cutlass_eoff: Option<Vec<i32>> = None;
        // ── SMALL-M FP4 decode GEMV arm (ATLAS_MOE_FP4_DECODE_SMALLM=1) ──
        // At decode-M (1-2 rows/expert) every M_TILE=64 grouped kernel — CUTLASS
        // and FP4-K64 alike — pays a full 64-row tile per 1-2 real rows. The
        // slot-major GEMV pair reads the SAME shared `_t` tables + scale2 and
        // writes the SAME sorted C_gate/C_up/down layout, so silu/LoRA-fold/
        // unpermute are unchanged; it intercepts BEFORE the CUTLASS branch so the
        // canonical serve config (GROUPED_CUTLASS=1) also gets the small-M win.
        // Graph-capture-legal (device offsets binary-searched in-kernel, grid =
        // host-known n*top_k). Prefill (total_expanded > threshold) is untouched.
        let smallm_m = self.fp4_decode_smallm_max;
        let smallm_nvfp4 = self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4;
        let smallm_gate_up = smallm_m > 0
            && total_expanded <= smallm_m
            && total_expanded as u32 >= fp4_decode_smallm_min()
            && smallm_nvfp4
            && self.moe_fused_gate_up_fp4_smallm.0 != 0
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some();
        if max_m_tiles > 0 {
            if smallm_gate_up {
                let gp = self.gate_ptrs_t.as_ref().expect("smallm_gate_up checked");
                let up = self.up_ptrs_t.as_ref().expect("smallm_gate_up checked");
                ops::moe_w4a16_fp4_smallm_gate_up(
                    ctx.gpu,
                    self.moe_fused_gate_up_fp4_smallm,
                    expert_input,
                    gp.packed_ptrs,
                    gp.scale_ptrs,
                    gp.scale2_vals,
                    up.packed_ptrs,
                    up.scale_ptrs,
                    up.scale2_vals,
                    expert_gate_out,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    total_expanded,
                    stream,
                )?;
            } else
            // CUTLASS grouped NVFP4 gate_up reads the ORIGINAL [N,K/2] tables
            // (CUTLASS B is ColumnMajor = K-contiguous), NOT the Atlas
            // transposed ones, so it must be reachable when gate_ptrs_t is
            // absent — that is exactly the originals-only layout a
            // checkpoint-native model runs in.
            if grouped_cutlass_gate_up_enabled()
                && self.gate_sfb_cutlass.is_some()
                && self.up_sfb_cutlass.is_some()
            {
                // ── SINGLE-LAUNCH CUTLASS grouped NVFP4 gate_up
                // (ATLAS_HOLO_MOE_GROUPED_CUTLASS=1) ── one
                // GemmUniversalMode::kGrouped launch over all active experts in
                // place of the per-expert collective loop. Weights: the decode
                // `gate_ptrs`/`up_ptrs` packed `[N,K/2]` (CUTLASS ColumnMajor B) +
                // the load-built swizzled SFB tables (`gate_sfb_cutlass`) + the real
                // per-expert scale2 (epilogue alpha). The token gather is FUSED into
                // the kernel's per-group A-pack (lever 2): pass token-major
                // expert_input + sorted_token_ids directly, no separate permute pass.
                // Writes C_gate/C_up in the sorted layout so silu+down+unpermute are
                // unchanged.
                if grouped_cutlass_device_offsets_enabled() {
                    // DEVICE-OFFSET variant (ATLAS_MOE_CUTLASS_DEVICE_OFFSETS=1):
                    // same grouped GEMM, but per-group problem sizes/pointers are
                    // built on-device from `expert_offsets` and the launch grid is
                    // the fixed sm_count persistent grid — no D2H + no host sync,
                    // so this arm is CUDA-graph-capture-legal (decode graphs stay
                    // captured; see decode_a2.rs grouped_is_graph_safe).
                    ops::moe_grouped_gate_up_cutlass_dev(
                        expert_input,
                        sorted_token_ids,
                        self.gate_ptrs.packed_ptrs,
                        self.gate_sfb_cutlass.expect("gate sfb checked above"),
                        self.gate_ptrs.scale2_vals,
                        self.up_ptrs.packed_ptrs,
                        self.up_sfb_cutlass.expect("up sfb checked above"),
                        self.up_ptrs.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        num_experts as usize,
                        total_expanded,
                        inter,
                        h,
                        stream,
                    )?;
                } else {
                    cutlass_eoff = Some(ops::moe_grouped_gate_up_cutlass(
                        ctx.gpu,
                        expert_input,
                        sorted_token_ids,
                        self.gate_ptrs.packed_ptrs,
                        self.gate_sfb_cutlass.expect("gate sfb checked above"),
                        self.gate_ptrs.scale2_vals,
                        self.up_ptrs.packed_ptrs,
                        self.up_sfb_cutlass.expect("up sfb checked above"),
                        self.up_ptrs.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        num_experts as usize,
                        inter,
                        h,
                        stream,
                    )?);
                }
            } else if let (Some(gp), Some(up)) = (&self.gate_ptrs_t, &self.up_ptrs_t) {
                if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                    // ── ARM-2 Phase-K: native-MXFP4 (E8M0) fused gate_up ──
                    // Leading branch so E8M0 routed experts NEVER reach the
                    // NVFP4-only cutlass/fp4/m128 sub-paths below (structurally
                    // off for the V4 serve, but the branch makes it provable).
                    assert!(
                        self.moe_fused_gate_up_t_k64_e8m0.0 != 0,
                        "ARM-2: routed experts Mxfp4E8m0 but fused_gate_up_t_k64_e8m0 unresolved"
                    );
                    ops::moe_w4a16_fused_gate_up_k64_n128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64_e8m0,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles,
                        stream,
                    )?;
                } else if self.gateup_fp4 && self.moe_fused_gate_up_t_k64_fp4.0 != 0 {
                    // ── FUSED FP4 gate_up (ATLAS_HOLO_MOE_GATEUP_FP4) ──
                    // Block-scaled FP4 over the SHARED FAST_MOE=full [K/2,N] tables
                    // (gate_ptrs_t/up_ptrs_t — the SAME bytes the FP8 fused path
                    // reads, selected here only by kernel handle, so NO extra MoE
                    // memory). The kernel loads them coalesced K-major and re-gathers
                    // N-major on-chip (FP4_TRANSPOSE). gp/up carry the REAL per-expert
                    // scale2 (applied at writeback) — not the legacy hardcoded 1.0.
                    // Single launch, grid z = num_experts; writes C_gate/C_up in the
                    // same sorted layout as FP8 so silu+down+unpermute are unchanged.
                    ops::moe_w4a16_fused_gate_up_k64_n128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64_fp4,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles,
                        stream,
                    )?;
                } else {
                    // Block D #3 dispatch: M=128 path needs the env var on AND
                    // the kernel actually loaded (try_kernel returns 0 on
                    // models that don't ship it). max_m_tiles_m128 = ceil(...
                    // /128) instead of /64; reuse the same upper bound by
                    // halving (each m128 tile covers 2 m64 tiles).
                    let use_m128 =
                        self.nvfp4_gate_up_m128 && self.moe_fused_gate_up_t_k64_m128.0 != 0;
                    if use_m128 {
                        let max_m_tiles_m128 = max_m_tiles.div_ceil(2).max(1);
                        ops::moe_w4a16_fused_gate_up_k64_m128(
                            ctx.gpu,
                            self.moe_fused_gate_up_t_k64_m128,
                            expert_input,
                            gp.packed_ptrs,
                            gp.scale_ptrs,
                            gp.scale2_vals,
                            up.packed_ptrs,
                            up.scale_ptrs,
                            up.scale2_vals,
                            expert_gate_out,
                            expert_up_out,
                            expert_offsets,
                            sorted_token_ids,
                            num_experts,
                            inter,
                            h,
                            max_m_tiles_m128,
                            stream,
                        )?;
                    } else {
                        ops::moe_w4a16_fused_gate_up_k64_n128(
                            ctx.gpu,
                            self.moe_fused_gate_up_t_k64,
                            expert_input,
                            gp.packed_ptrs,
                            gp.scale_ptrs,
                            gp.scale2_vals,
                            up.packed_ptrs,
                            up.scale_ptrs,
                            up.scale2_vals,
                            expert_gate_out,
                            expert_up_out,
                            expert_offsets,
                            sorted_token_ids,
                            num_experts,
                            inter,
                            h,
                            max_m_tiles,
                            stream,
                        )?;
                    }
                }
            } else {
                // ARM-2 Phase-K straggler net: V4 native builds gate_ptrs_t, so
                // E8M0 never reaches this non-transposed fallback. If it does,
                // panic (a real finding) rather than run NVFP4-on-E8M0 garbage.
                self.experts_scale_kind.expect(
                    crate::weight_map::WeightQuantFormat::Nvfp4,
                    "prefill non-transposed gate_up fallback (no E8M0 variant wired)",
                );
                let (gp, up) = (&self.gate_ptrs, &self.up_ptrs);
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_input,
                    gp.packed_ptrs,
                    gp.scale_ptrs,
                    gp.scale2_vals,
                    expert_gate_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_input,
                    up.packed_ptrs,
                    up.scale_ptrs,
                    up.scale2_vals,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_gate_up");

        // 6. Activation+mul for routed experts + grouped down GEMM (K64 pipelined).
        let expert_down_out = ctx.buffers.expert_down_out();
        if max_m_tiles > 0 {
            // Feature-1: fold the routed-expert gate/up_proj LoRA deltas onto the
            // sorted `expert_gate_out`/`expert_up_out` BEFORE `silu_mul` consumes
            // them in place (x = token-major `expert_input`, gathered per sorted
            // row). No-op unless gate/up deltas are installed. Covers all five
            // nvfp4 gate_up sub-branches (all wrote sorted BF16 gate/up).
            self.apply_expert_lora_prefill_gateup(
                expert_gate_out,
                expert_up_out,
                expert_input,
                expert_offsets,
                sorted_token_ids,
                total_expanded,
                ctx,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            // ── FP4 down (ATLAS_HOLO_MOE_DOWN_FP4) ── single block-scaled FP4
            // MMA per k64 tile (mxf4nvf4.scale_vec::4X.m16n8k64), reading the
            // post-SiLU intermediate (expert_gate_out) and the per-expert FP4
            // down tables. Same sorted layout + null sorted_token_ids as the
            // FP8/w4a16 down kernels, so unpermute downstream is unchanged.
            // Compounds with the FP4 gate_up path to run the whole FFN at FP4.
            // CUTLASS grouped down reads the ORIGINAL [N,K/2] table, so like
            // gate_up it must be reachable without down_ptrs_t.
            // Small-M FP4 GEMV down (see the gate_up arm above): intercepts both
            // the CUTLASS and K64 down kernels at decode-M. Same sorted layout +
            // null sorted_token_ids (identity row gather), unpermute unchanged.
            let smallm_down = smallm_m > 0
                && total_expanded <= smallm_m
                && total_expanded as u32 >= fp4_decode_smallm_min()
                && smallm_nvfp4
                && self.moe_down_fp4_smallm.0 != 0
                && self.down_ptrs_t.is_some();
            if smallm_down {
                let dp = self.down_ptrs_t.as_ref().expect("smallm_down checked");
                ops::moe_w4a16_fp4_smallm_down(
                    ctx.gpu,
                    self.moe_down_fp4_smallm,
                    expert_gate_out,
                    dp.packed_ptrs,
                    dp.scale_ptrs,
                    dp.scale2_vals,
                    expert_down_out,
                    expert_offsets,
                    DevicePtr(0),
                    num_experts,
                    h,
                    inter,
                    total_expanded,
                    stream,
                )?;
            } else if grouped_cutlass_gate_up_enabled()
                && self.down_sfb_cutlass.is_some()
                && std::env::var("ATLAS_HOLO_MOE_GROUPED_DOWN").ok().as_deref() == Some("1")
            {
                // ── CUTLASS grouped NVFP4 down (ATLAS_HOLO_MOE_GROUPED_CUTLASS
                //    + ATLAS_HOLO_MOE_GROUPED_DOWN) ──
                // A = post-SiLU expert_gate_out, already expert-contiguous (the grouped
                // gate_up wrote it sorted), so NO gather. Weights = decode down_ptrs
                // packed [N=hidden,K/2] + load-built swizzled SFB + real scale2. Writes
                // expert_down_out in the sorted layout (unpermute downstream unchanged).
                if grouped_cutlass_device_offsets_enabled() {
                    // DEVICE-OFFSET variant — see the gate_up branch above. No
                    // host `expert_offsets` exist in this mode (nothing was
                    // copied down), so the down projection re-reads the device
                    // prefix sum in its own prep kernel.
                    ops::moe_grouped_down_cutlass_dev(
                        expert_gate_out,
                        self.down_ptrs.packed_ptrs,
                        self.down_sfb_cutlass.expect("down sfb checked above"),
                        self.down_ptrs.scale2_vals,
                        expert_down_out,
                        expert_offsets,
                        num_experts as usize,
                        total_expanded,
                        h,
                        inter,
                        stream,
                    )?;
                } else {
                    ops::moe_grouped_down_cutlass(
                        ctx.gpu,
                        cutlass_eoff.as_deref(),
                        expert_gate_out,
                        self.down_ptrs.packed_ptrs,
                        self.down_sfb_cutlass.expect("down sfb checked above"),
                        self.down_ptrs.scale2_vals,
                        expert_down_out,
                        expert_offsets,
                        num_experts as usize,
                        h,
                        inter,
                        stream,
                    )?;
                }
            } else if let Some(dp) = &self.down_ptrs_t {
                if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                    // ── ARM-2 Phase-K: native-MXFP4 (E8M0) grouped down ──
                    // Leading branch (bypasses NVFP4-only cutlass/fp4/fp8_down).
                    assert!(
                        self.moe_grouped_gemm_t_k64_e8m0.0 != 0,
                        "ARM-2: routed experts Mxfp4E8m0 but grouped_gemm_t_k64_e8m0 unresolved"
                    );
                    ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                        ctx.gpu,
                        self.moe_grouped_gemm_t_k64_e8m0,
                        expert_gate_out,
                        dp.packed_ptrs,
                        dp.scale_ptrs,
                        dp.scale2_vals,
                        expert_down_out,
                        expert_offsets,
                        DevicePtr(0),
                        num_experts,
                        h,
                        inter,
                        max_m_tiles,
                        stream,
                    )?;
                } else if self.down_fp4 && self.moe_down_t_k64_fp4.0 != 0 {
                    // ── FP4 down (ATLAS_HOLO_MOE_DOWN_FP4) over the SHARED down_ptrs_t
                    // [K/2,N] table (real per-expert scale2; coalesced K-major load +
                    // on-chip DN4_TRANSPOSE). Same sorted layout + null
                    // sorted_token_ids as the FP8/w4a16 down kernels, so unpermute is
                    // unchanged. No extra MoE memory (shared table).
                    ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                        ctx.gpu,
                        self.moe_down_t_k64_fp4,
                        expert_gate_out,
                        dp.packed_ptrs,
                        dp.scale_ptrs,
                        dp.scale2_vals,
                        expert_down_out,
                        expert_offsets,
                        DevicePtr(0),
                        num_experts,
                        h,
                        inter,
                        max_m_tiles,
                        stream,
                    )?;
                } else {
                    let fp8_down = std::env::var("ATLAS_MOE_PREFILL_FP8_DOWN").ok().as_deref()
                        == Some("1")
                        && self.moe_fp8_grouped_gemm_t.0 != 0
                        && self.bf16_to_fp8_k.0 != 0;
                    if fp8_down {
                        ops::bf16_to_fp8(
                            ctx.gpu,
                            self.bf16_to_fp8_k,
                            expert_gate_out,
                            expert_up_out,
                            total_expanded * inter,
                            stream,
                        )?;
                        ops::moe_fp8_grouped_gemm_ptrtable_n128(
                            ctx.gpu,
                            self.moe_fp8_grouped_gemm_t,
                            expert_up_out,
                            dp.packed_ptrs,
                            dp.scale_ptrs,
                            dp.scale2_vals,
                            expert_down_out,
                            expert_offsets,
                            DevicePtr(0),
                            num_experts,
                            h,
                            inter,
                            max_m_tiles,
                            stream,
                        )?;
                    } else {
                        ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                            ctx.gpu,
                            self.moe_grouped_gemm_t_k64,
                            expert_gate_out,
                            dp.packed_ptrs,
                            dp.scale_ptrs,
                            dp.scale2_vals,
                            expert_down_out,
                            expert_offsets,
                            DevicePtr(0),
                            num_experts,
                            h,
                            inter,
                            max_m_tiles,
                            stream,
                        )?;
                    }
                }
            } else {
                // ARM-2 Phase-K straggler net (see gate_up fallback above).
                self.experts_scale_kind.expect(
                    crate::weight_map::WeightQuantFormat::Nvfp4,
                    "prefill non-transposed down fallback (no E8M0 variant wired)",
                );
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_gate_out,
                    self.down_ptrs.packed_ptrs,
                    self.down_ptrs.scale_ptrs,
                    self.down_ptrs.scale2_vals,
                    expert_down_out,
                    expert_offsets,
                    DevicePtr(0),
                    num_experts,
                    h,
                    inter,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_silu_down");

        Ok(())
    }
}
