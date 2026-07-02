# Handoff — native per-row FP8 for Ornith-1.0-35B-FP8 (WIP, NOT production-correct)

Branch: `feat/native-fp8-attn-modelopt-mixed` (off `test/220-figdn`).
Commits (this work): `0d82b36`, `f3da561`, `d290136`, `9cfb669` (+ `1197192` = nvidia-Qwen27B handoff).
Date: 2026-07-02. See also memory `holo-perrow-fp8-ornith.md`, `holo-modelopt-fp8-attn-requant.md`.

## Goal
`deepreinforce-ai/Ornith-1.0-35B-FP8` ships **per-channel (per-row) FP8**: `weight_scale [N,1]` (one fp32
per output row, constant over k) for attention q/k/v/o AND MoE experts. Atlas FP8 kernels assumed BLOCK
scale `[N/128,K/128]` → misread the `[N]` buffer → garbage. Goal: keep it native FP8 (no requant).

## Ground truth (proven)
Full BF16 dequant (`ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1` + `ATLAS_FP8_DEQUANT_MOE_TO_BF16=1`) = PERFECT:
one-shot numeric (0,1,1,2,3,5,8,13,21,34; 17×24=408) AND agentic ~85%. So model/config/tokenizer/GDN are
fine; the garbage is purely the per-row FP8 compute paths. **BF16-dequant is the only correct Ornith path today.**

## What's in the commits
Adds `unsigned int per_row` (LAST kernel param) + threads a bool through ops/call sites. Per-row math:
`out[n] = scale[n] * Σ_k A·W` (scale constant over k) → row-parallel kernels load `scale[n]` constant;
tiled kernels fold UNSCALED + apply `scale[col]` per output column in the epilogue.
- **Loader** `weight_map/loaders_fp8.rs`: detect `[N,1]` → tag `Fp8PerRow` + widened `[N]` fp32 (f3da561).
- **Attention kernels** (f3da561): `w8a16_gemv`, `w8a16_gemm_t`, `w8a16_gemm_t_pipelined`, `w8a16_gemm_t_m128`.
- **gemv LUT fix** (d290136): shared E4M3 LUT fill moved above `if(n>=N)return` (pre-existing partial-block bug).
- **MoE kernels** (9cfb669): `moe_fp8_grouped_gemm`, `moe_shared_expert_fused_fp8{,_batch2,_batch3,_t,_batch2_t,
  _batch3_t}`, `w8a16_gemm`/`_pipelined`; `MoeLayer::fp8_experts_per_row` set in `set_fp8_experts`.
- `quantized.rs` `Fp8WeightTransposed.scale_format`; `transpose_for_gemm` aliases the `[N]` per-row scale
  untransposed. `qwen35/load_layers.rs` + `qwen35_dense.rs` route per-row FP8 attn to the native arm.

## STATUS: NOT production-correct (two independent problems)
1. **MoE per-row FP8 — wrong under load.** Kernels are UNIT-TEST BIT-EXACT in isolation (standalone nvcc vs
   host ref, on GB10), and one-shot numeric is perfect. BUT the agentic A/B is definitive: **MoE-FP8 ~13%
   (1/7, 1/8; 5 `<tool_call>` max-token loops) vs MoE-BF16 ~85% (3/3, 3/4; 1-3 loops).** A residual error
   small enough to pass short prompts but large enough to derail multi-turn into degeneration/loops.
2. **Attention per-row FP8 — never actually runs in prefill.** Decode `w8a16_gemv` per-row IS correct at
   runtime (verified: per_row=true, valid `[N]` scale, scale[0]=9e-5). But **attention PREFILL never routes
   to any FP8 GEMM** — instrumented all 4 prefill ops (m128/gemm_t/pipelined/w8a16_gemm), NONE fire even on
   a 600-tok prompt. `paged_qkv.rs:132-316` (10-way chain) falls through to `dense_gemm` using arm-626's NULL
   `attn.q_proj` dummy → garbage first hidden state → degeneration. Ornith resolves to the
   `holo-3.1-35b-a3b` NVFP4 kernel target ("kernel=nvfp4 model=fp8"); the fp8w_t branch (line ~198) isn't
   reached despite "FP8 weights transposed for fast prefill" logging. This is a **prefill dispatch / kernel-
   handle wiring** bug, NOT the kernels.

Both are the same class: **bit-exact in isolation, wrong in the full pipeline.** The kernels pass every unit
test; the bugs are in dispatch/accumulation-under-real-shapes. Chasing them needs a **golden-output
integration test** (token-exact @temp0 per model+flags), not more unit tests.

## Next steps
1. Attention: find why `paged_qkv.rs` prefill skips the `fp8w_t` branch for the holo-3.1-35b-a3b target
   (kernel-handle 0? a different prefill fn — cache_skip_qkv / a fused/flashinfer path? fp8w_t not visible?).
   Route Ornith FP8 attn prefill to the (correct) per-row kernels.
2. MoE: golden-output diff MoE-FP8 vs MoE-BF16 per layer to localize the residual error (the grouped/shared
   FP8 kernels or the scale_format plumbing through `build_fp8_ptr_table`).
3. Until fixed: ship Ornith on BF16-dequant (`ATLAS_HOLO_FP4_PROJ_DECODE=0 ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1`
   + MoE also BF16 for correctness) — ~85% agentic, correct.

## Serving notes (independent, both real)
- `ATLAS_MAX_INTER_TOOL_PROSE=8192` (default 384, `helpers.rs:444`) — the 384 default fights a thinking
  agentic model (inter-tool prose watchdog rollback loops). Holo-35B control: 6/6 (100%), 0 rollbacks with 8192.
- Serve container BAKES many `ATLAS_*` flags (incl. `ATLAS_HOLO_FP4_PROJ_DECODE=1`, `ATLAS_CUTLASS_NVFP4_GEMM=1`)
  — silently changes the kernel mix vs a bare `spark serve`.

## Reproduce
Build local `ATLAS_TARGET_MODEL=*` → scp gx10:~/spark-dbg; serve in `atlas-holo:cuda13.2-fp4test` + FI-GDN.
Agentic: gx10 `~/agentic_test_notemp.py` (has the discover-verify fix) / `~/agentic_test_dual.py --stream`.
