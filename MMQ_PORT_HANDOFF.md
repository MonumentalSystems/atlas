# Handoff: Beat llama.cpp prefill on GB10 via a faithful int8 MMQ-tile port

**Status (2026-06-27):** Goal PROVEN ACHIEVABLE, de-risked, wall broken, NEAR PARITY on down.
`int8_gemm_faith` (all levers combined) hit 33.78. Then `int8_gemm_faith2` (= faith + big-K-tile-128
loaded once + ROLLING weight pre-stage, sb-outer/j-inner) now hits **gate/up 44.75, down 48.93 TFLOP/s
at cosine 0.999999** — gate/up 75% of llama (60), down 82%. The structural skeleton is validated; the
remaining gap is the last ~15-25% of tuning (occupancy 1→2 CTA, ldmatrix-B, NVFP4-native MMA stretch).
HEAD-TO-HEAD TARGET: llama cfff1fc agentic-2.5h on dgx1 = **8369.87s wall, TTFT median 1393ms**
(/workspace/endpoints-agentic/results/agentic_coding_perf_2.5h_gb10_cfff1fc). Atlas FP8 run = 14360s
(worse + incoherent). Gap is entirely prefill TTFT (~4× at median). FP8-KV + prefix-caching ALLOWED as
levers. Snapshot branch: perf/int8-prefill-faith2 (PR, not merged). **Solely on dgx1.**

---

## 0. SUCCESS CRITERION (what "done" means — gate every claim, never n=3 smoke)
1. **Kernel milestone:** an int8 W4A8 gate/up GEMM hitting **≥50 TFLOP/s** on `[M=4096,N=17408,K=5120]`
   (llama measures 60-65 there) at **cosine ≥0.999** vs the host reference in `examples/int8_gemm_test.rs`.
2. **Quality gate:** wired into the model, generation stays coherent (NO whitespace/length runaway like fp8),
   ST/BFCL accuracy neutral, agentic-2.5h **IoU ≥ 0.63** (llama = 0.6326). Use N≥10 runs, not n=3.
3. **Wall:** agentic-2.5h wall **< llama 8370s (2h19m)** on dgx1. (Even ~2× prefill only *halves* the gap;
   full beat may also need stream-K + the down-proj win + decode parity — measure, don't assume.)

## 1. THE BREAKTHROUGH (why the old "impossible/hardware-capped" conclusion was WRONG)
- **ldmatrix is NOT broken on GB10.** Proof: `/workspace/ldmatrix_probe.cu` (nvcc -arch=sm_121a) —
  m16n8k16 MMA with A via `ldmatrix.sync.aligned.m8n8.x4.b16` == manual == CPU, **cosine 1.000000**.
  Atlas's 4 in-tree "ldmatrix broken" comments are a MIS-PORT of the `.trans` variant only (it needs the
  output-reg permute `{xi[0],xi[2],xi[1],xi[3]}`, llama `mma.cuh:811`). Over-generalized → scalar smem
  loads → the "90% L1/TEX wall" I measured. Self-inflicted, not silicon.
- **llama does 2× on the IDENTICAL shape.** Measured via `test-backend-ops -o MUL_MAT perf` on this GB10:
  gate/up `[17408,n,5120]` Q4_K **60-65 TFLOP/s**, down 47-58. Data: `/workspace/atlas-prefill32k/scratch_llama_perf2.txt`.
  Atlas pinned ~30 (bf16) / ~24 (int8). So gate/up IS where Atlas loses; it is NOT a hardware cap.
- **My methodological error:** tested each lever (K_STEP, 8-warp, split-K, ldmatrix, ILP) IN ISOLATION on
  the unchanged straight-line base → each looked dead. They are synergistic; the win is the full skeleton.

## 2. MEASURED BASELINES (dgx1, examples/int8_gemm_test.rs, TFLOP/s)
gate/up M=4096: bf16-v2 **30** | int8 M128 24 | M64 12(spill) | K64 18 | split-K sk2-16 12-20 | 8-warp 23
  | 8w+3stage-pipe 24 | 8w+ldmatrix 23 | 8w+2-phase-ILP 23  → all ~24, NONE beat bf16 (each lever alone).
  **int8_gemm_faith (ALL levers combined): gate/up 33.78, down 33.80, cosine 0.999999 — FIRST int8 > bf16.**
  Levers in faith: big K-tile (FK_TILE=64) loaded once via cp.async; bank-fixed smem stride-20 int32
  (16B-aligned, r*5 mod 8 distinct); register pre-stage of ALL weight ldmatrix frags+scales before the
  j-MMA loop; activations via cheap scalar load.
  **int8_gemm_faith2 (faith + 2 structural changes): gate/up 44.75, down 48.93, cosine 0.999999.**
  Change 1: K-tile 64→128 loaded ONCE (40 outer steps for K=5120 vs 80 → halves cp.async+2-sync traffic).
  Change 2: ROLLING weight pre-stage — sb-loop OUTER, j-loop INNER, so only WA[2][4]=8 regs live per
  sub-block (vs 32 for full pre-stage of 4 sub-blocks atop acc[2][8][4]); decouples tile size from regs.
  Swept F2_TILE: 64→34, **128→44.75/48.93 (best)**, 256→37.6/39.9 (smem cuts occupancy). 128 = sweet spot.
  NEXT levers to close to 60: (a) occupancy 1→2 CTA/SM (launch_bounds(256,2); regs+smem permitting at
  F2_TILE=128 sW+sA=36.8KB→2 CTA=73.6KB<100KB ✓ — try it); (b) ldmatrix-B (collapse 16 scalar B-loads);
  (c) double-buffer smem to drop the trailing sync; (d) STRETCH: native NVFP4 block-scale MMA (Colfax SM12x).
down M=4096: int8 split-K **sk8 35** (beats bf16 30 — the one int8 win; few base CTAs + big K).
ncu (int8_gemm_8w_ldm gate/up): **stall = SHORT_SCOREBOARD (smem-read dep) 37%, 11.5 warp-cyc/instr**,
  occupancy 33%, L1/TEX 30%, DRAM 38% — nothing saturated; it's smem-read *latency* with no ILP to hide it.

## 3. THE REMAINING WORK — faithful port of llama's MMQ tile skeleton (NOT more variants)
llama's `mmq` (q8_0/q4_K int8 path) differs STRUCTURALLY from every Atlas variant:
- **(a) Load a BIG K-tile (`MMQ_ITER_K=256`) into smem ONCE, iterate `k01` WITHIN it.** Inner loop has
  ZERO global loads + ZERO `__syncthreads`. Mine reloads from global + syncs every 32-K (~160×). THIS is
  the structural fix for the smem-scoreboard/no-eligible wall.
- **(b) Register-blocked `tile_C` + `ntx` minitiles** → several MMAs issue before dependent scale-multiplies
  (ILP that hides smem latency).
- **(c) `load_ldmatrix` for A AND B** (B via `load_generic`, mmq.cuh:1433). ldmatrix.x4.b16 ALREADY VERIFIED
  to map onto int8 m16n8k32 A-frag: `xs=(int*)&smem[wrow][0]+(lane%16)*8+(lane/16)*4`, non-trans order
  matches MMA directly (cosine 0.999999 in int8_gemm_8w_ldm).
- **(d) q8_1 ds (d/scale) layout** for the per-32-block scales, folded once: `sum += C.x[l]*dA*dB` (this part
  Atlas already matches — `mma.cuh:1206-1212`; NOT the bottleneck).
**Template files (llama, /workspace/llama-cfff1fc/ggml/src/ggml-cuda/):**
  - `mmq.cuh:1159-1215` vec_dot_q8_0_q8_1_mma (the inner k01 loop + tile_C accumulate)
  - `mmq.cuh:~3485-3518` the kb0 outer loop (load big tile, iterate)
  - `mmq.cuh:3528,3641-3719` stream-K + fixup reduction (for the down-proj / SM-fill, AFTER int8 path works)
  - `mma.cuh:751-758` load_ldmatrix x4 non-trans; `:806-813` trans + the permute
  - `mmq.cuh` get_mmq_x_max/get_mmq_y/MMQ_NWARPS for tile sizing on this CC

### Suggested build order (each step gated on cosine + bench + ncu)
1. New kernel `int8_gemm_mmq`: 8-warp, load 128-256 K-tile once, iterate k01 within (no per-32 global/sync),
   register-blocked tile_C, ldmatrix A + manual B, epilogue per-block scale. Microbench → target >40, then >50.
2. Tune MMQ_X/Y/ntx/launch_bounds via ncu (kill the short_scoreboard; watch reg spill — s[16][4]+acc[16][4]
   is heavy, may need ntx chunking or (256,1)).
3. Add stream-K for down + a host shape-dispatch (split-K already wins down at 35).
4. **Highest ceiling (do AFTER int8 works):** native NVFP4 block-scale MMA
   `mma.sync...mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.f32.e2m1.e2m1.f32.ue4m3` — zero software dequant;
   Atlas weights are already E2M1+per-16-E4M3 (1:1 format). llama's NVFP4 path = +43-68% on THIS model
   (PR#22196). FP4 operand load via `ldmatrix.b8x16.b4x16_p64`. Ref: Colfax SM12x NVFP4 tutorial.
5. Integrate behind env `ATLAS_INT8_PREFILL` in `dense_ffn.rs` (pattern already there: see fp8 M64 wiring,
   `w4a16_gemm_t_k` handle + `ATLAS_FP8_M64_PREFILL` flag + highest-priority macro arm) + requant kernels
   (NVFP4→int8-per32 weights at load; bf16→int8-per32 activations per-prefill — task #15). Quality-gate, then agentic.

## 4. INFRA (all on dgx1, all WORKING)
```
cd /workspace/atlas-prefill32k
export PATH=/usr/local/cuda-13.0/bin:$PATH
# build the microbench/test (kernels in kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu, module "w4a16"):
CARGO_TARGET_DIR=/workspace/scratch-bench ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-27b \
  ATLAS_TARGET_QUANT=nvfp4 cargo build --release -p spark-model --example int8_gemm_test   # ~15s
LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 /workspace/scratch-bench/release/examples/int8_gemm_test
# ncu (needs sudo -E): sudo -E /usr/local/cuda-13.0/bin/ncu --target-processes all \
#   --kernel-name "regex:int8_gemm_..." --launch-count 1 --section WarpStateStats --section SpeedOfLight <bin>
# standalone ldmatrix probe: nvcc -arch=sm_121a -o ldmatrix_probe /workspace/ldmatrix_probe.cu && ./ldmatrix_probe
# server build (native, for end-to-end): same env, cargo build --release -p spark-server --bin spark
#   serve flags + ATLAS_BF16_TC_PREFILL etc: see /workspace/time_prefill.sh, /workspace/fp8m64_gate.sh
# llama per-shape ref: test-backend-ops in /workspace/llama-cfff1fc-pin/bin (see scratch_llama_perf2.txt)
```
In-tree int8 kernels (module w4a16): int8_gemm_t_m128, _m64, _m128_k64, int8_gemm_splitk + int8_splitk_reduce,
int8_gemm_8w, int8_gemm_8w3, int8_gemm_8w_ldm, int8_gemm_8w_ilp. The CORRECT base to start from = int8_gemm_8w_ldm
(8-warp + ldmatrix, cosine 0.999999). Test harness: examples/int8_gemm_test.rs (host-ref cosine + speed sweep).

## 5. GOTCHAS / context
- Memory note (the canonical record, has the corrected conclusion + all data):
  /workspace/.claude/projects/-workspace/memory/project_prefill_bubble_bound_not_mma_2026_06_26.md
- The bf16/fp8 "ldmatrix broken" comments in inferspark_prefill.cu / dense_gemm_tc.cu / gated_delta_rule_fla.cu
  are WRONG for x4-non-trans (only .trans needs the permute). Safe to use ldmatrix.x4 going forward.
- fp8 M64 path (ATLAS_FP8_M64_PREFILL, dense_ffn.rs) is fast (1.2x e2e) but BREAKS coherence (3-bit mantissa
  whitespace runaway) — do NOT ship it; it's the cautionary tale that motivates int8 (8-bit).
- The atlas-prefill32k working tree is DIRTY with all these WIP kernels (uncommitted). Branch + commit before
  big changes if you want a clean base. Other session wins already on origin:
  perf/strix-rocmfp4-full1004-87.85, feat/agentic-2.5h-bf16tc-prefill, perf/agentic-2.5h-prefill.
- Strix (separate box, gfx1151) is NOT this goal — keep all work on dgx1.

## 6. ONE-LINE RESTART
"Read this doc + the memory note. Build `int8_gemm_mmq` as a faithful port of llama mmq.cuh:1159-1215
(big-K-tile-once + iterate-within + register-blocked tile_C + ldmatrix A/B), starting from int8_gemm_8w_ldm.
Gate on examples/int8_gemm_test.rs cosine≥0.999 + bench ≥50 TFLOP/s on gate/up, ncu the short_scoreboard.
Then integrate + quality-gate + agentic. Solely on dgx1."
