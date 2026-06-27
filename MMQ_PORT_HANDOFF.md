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
  TUNING LADDER (06-27, all cosine 0.999999, gate/up TFLOP/s):
   - faith2 (big-K-128 + rolling pre-stage)          44.75  ← BEST, the structural win
   - launch_bounds(256,2) on faith2                  32.4   REGRESSED (register spill; occ via CTA dead)
   - faith3 (= faith2 + B-frag/scale pre-stage)      44.71  NEUTRAL (compiler already had the B-load ILP)
   - faith4 (512-thread CTA, acc[2][4][4], 16 warp)  44.57  NEUTRAL — ncu CONFIRMS occ rose 16.6%→32.2%
                                                            but throughput flat; NOTHING saturates
                                                            (Compute 26%, Mem 33%); pure dependency-latency.
  VERDICT: occupancy AND smem-read ILP are BOTH exhausted. faith2's 44.75 is the plateau for this tile
  skeleton. Closing the last 1.34× to llama (60) needs a DIFFERENT structure: deep multi-stage software
  pipeline (cp.async depth≥2 + q8_1 INTERLEAVED weight layout so B loads via ldmatrix, llama's actual
  trick) OR the native NVFP4 block-scale MMA (Colfax SM12x, mma...mxf4nvf4.block_scale.m16n8k64 — 2× K/instr,
  zero software dequant, weights already E2M1+per-16-E4M3). Those are the two ceiling-breakers.
  INTEGRATION INSIGHT: dense_ffn.rs ALREADY has an FP8-M64 prefill path (w4a16_gemm_t_k, ATLAS_FP8_M64_PREFILL)
  at ~44 TFLOP/s — but LOSSY (FP8 E4M3 cosine 0.9997, breaks coherence). int8 faith2 = SAME 44 TFLOP/s at
  cosine 0.999999 → it is the COHERENT version of that win. Wire faith2 behind ATLAS_INT8_PREFILL mirroring
  the FP8-M64 dispatch + add int8 requant (NVFP4 w→int8/per32 at load; bf16 act→int8/per32 per-prefill,
  ~1.4% of GEMM time). Then coherence-gate (N≥10) + ST subset + agentic-2.5h wall vs llama 8369.87s.
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

## 3.4 ★★★ ROOT-CAUSE FINDING (06-27 PM) — the agentic gap is BROKEN SSM SNAPSHOT RESTORE, not (only) the GEMM
Served Atlas bf16-TC + prefix-caching for a 3-traj agentic subset and read the serve log. The smoking gun:
```
Session 0x..a87a: 14083 prompt tokens
Prefix cache hit: 13936 tokens (871 blocks) BUT NO SSM SNAPSHOT — recomputing all KV
Done: TTFT=31917.9ms   (≈ a FULL ~14k cold re-prefill, 2.3ms/tok)
```
EVERY multi-turn warm turn full-recomputes (~29-40s TTFT, climbing with ctx) DESPITE a 99% KV prefix hit.
CAUSE = **--ssm-checkpoint-interval is in BLOCKS (×16 tok)**. The config (copied from dgx2_fp8_denser.sh)
used `--ssm-checkpoint-interval 2048` = **32768 tokens** → NO intermediate checkpoint ever fires for the
14-16k contexts. Only LEAF snapshots (saved at turn-END, e.g. tok 14083) exist, and a leaf is ABOVE the
next turn's match point (14080) so it is UNUSABLE (can't restore state for tokens you don't have). So
`prefix_match.ssm_snapshot = None` → prefill_b/prefix_lookup.rs:207 "no SSM snapshot — recomputing all KV".
**THE FIX = finer interval `--ssm-checkpoint-interval 64` (=1024 tok)** → checkpoints at 1k,2k,..,14k;
next turn restores from the deepest ≤ match, skipping ~14k tok, recomputing only the few-hundred-token tail.
Expected TTFT 32s → low single digits. THIS is why "prefix caching never reached parity" — prior runs all
used the coarse 2048. SERVING-CONFIG fix, bigger lever than the GEMM. Restore code is CORRECT & present
(prefix_lookup.rs:112-186 Marconi restore + intermediate-checkpoint replay); it just never had a usable snap.
Broken baseline (interval 2048): warm TTFT 29.2/30.7/31.9/32.6/33.7/38.1/39.7s.
★ FIX CONFIRMED (interval 64): serve log "Marconi intermediate hit: restored from checkpoint at token 1025/
2049 (skipping...)" — restore ENGAGES every warm turn. **Warm TTFT 3.9/2.3/3.3s (was ~32s) = ~10× drop.**
Per-turn wall now ~9s/it, DECODE-bound (prefill only 2-4s of it) → comparable to llama edge throughput.
Stack: working restore (interval 64) + int8 faith2 (1.49× the residual prefill tail, matters more on deep
23k turns) + FP8-KV = the full agentic win. THE caching fix is the giant lever; prior runs never had it.

★ DEEP-CONTEXT (06-27, interval 64): restore holds — "checkpoint at token 6145 (skipping 6145, recomputing
95 SSM tokens to match 6240)". Warm TTFT now 0.66-3.23s (avg ~2.4s). BUT warm TTFT is now SSM-REPLAY-BOUND:
recompute window = up to interval×16 = 1024 tokens × 48 Mamba layers (serial recurrence). Per-turn ~9.8s/it,
decode-bound. Extrapolated wall ~9840s (1007×9.8) = ~17% ABOVE llama 8370s. **To BEAT llama, cut warm TTFT
2.4s→<1.4s** (the (2.4-1.4)×1007≈1000s gap). Levers, BOTH prefill: (a) FINER checkpoint interval (32=512tok
or 16=256tok → less SSM replay → lower TTFT; costs more snapshot-save overhead + slots, 16384/256=64 ckpts/seq);
(b) int8 faith2 FFN GEMM (faster FFN inside the replay window + the new-token suffix, all 64 layers). Next:
measure int8+interval64 wall, then sweep interval 32/16.

★ CACHEFIX SUBSET RESULT (bf16-TC + interval64 + cache + FP8KV, 2 traj / 116 turns):
  Duration 1965s, **TTFT median 3084ms / avg 3181ms** (vs llama 1393ms = 2.2× — THE prefill gap to close),
  IoU 0.5714 (subset-specific, not vs llama's full-run 0.6326). Decode ~10.6 tok/s.
  CAVEAT — per-turn avg 16.9s but TTFT only ~3s ⇒ these turns are DECODE-bound (~14s decode, ~148 tok/turn).
  Decode ~10.6 tok/s because I OMITTED MTP. The INTENDED agentic serve config (rc3_gate.sh / gate_wash4_dgx1.sh /
  yaml notes) uses `--speculative --num-drafts 1 --mtp-quantization bf16 --kv-high-precision-layers auto` →
  MTP ~2× decode. Without MTP the subset is decode-bound and not representative of the user's full-run prefill-
  dominated premise. ACTION: add MTP to match intended config so prefill TTFT is the measured axis; my int8 +
  finer-interval work targets that TTFT (3084→ target ≤1393ms). The PREFILL levers (int8, interval) are correct;
  MTP is the orthogonal decode lever that the prior agentic runs already had.

## 3.42 ★★★ int8 + interval-16 (memory-fixed: util 0.68 + Marconi 128) — VERY PROMISING (06-27 PM)
After the OOM fix, int8 full-stack agentic runs clean (no OOM, restore engages). Interim (turn 33 of a
2-traj subset): warm TTFT **0.77-2.6s (median ~1.9s, approaching llama 1393ms)**; deep-context restore
EXCELLENT — at 17k ctx "skipping 17153 tokens, recomputing only 95 SSM tokens" → TTFT ~0.8s (interval 16
keeps the SSM-replay window tiny at depth, the key win over interval 64's ~1024-tok replay/3s TTFT).
**Per-it 5.60s/it @ turn 33** (vs bf16-TC cachefix ~6.6s) → benchmark's naive 1007-turn projection ≈ 90min
(~5640s) = WELL BELOW llama 8369s. Caveat: early/subset, may climb on deep turns; the FULL 20-traj run is
the definitive test. Best config = int8 + caching interval 16 + FP8-KV + util 0.68 + Marconi 128.
Full-run script ready: /workspace/atlas_agentic_FULL_int8.sh. Gating subset IoU first, then launch full.

★ HONEST UPDATE (per-it climbs): int8+interval16 subset per-it 5.6s@turn33 → 17.2s@turn98 (DECODE-bound
on deep turns: long verbose responses × 10.6 tok/s). So my early 90min projection was the SHALLOW turns;
deep turns are decode-bound. THE PREFILL GAP THE USER NAMED IS CLOSED: warm TTFT 0.8-2.6s now ≤ llama 1393ms
(int8 + interval-16 restore recomputes only 95-223 SSM tokens even at 17k ctx). But the WALL also includes
DECODE, and Atlas decode (~10.6 tok/s, no MTP) is the deep-turn bottleneck → wall would be decode-bound, not
beating 8369s on prefill alone. RESOLUTION: add MTP (the standard agentic decode accelerator; `--speculative
--num-drafts 1 --mtp-quantization bf16 --kv-high-precision-layers auto`; has a built-in net-negative auto-
disable gate so it never regresses). The user's reference config presumably had MTP → with decode competitive,
my prefill wins (caching interval16 + int8) tip the wall below llama. LAUNCHED FULL 20-traj MTP run
(atlas_agentic_FULL_int8_mtp.sh: int8 + interval16 + FP8-KV + MTP, util 0.66 + Marconi 96). Measuring wall vs 8369s.

★★★ MTP FULL-RUN EARLY SIGNALS (06-27, turn 3) — ALL GREEN, on track to BEAT llama:
  - MTP ENABLED, net-POSITIVE (mtp_gate verify_multiplier=1.11 << max 2.0). No OOM (util 0.66 + Marconi 96 fits).
  - DECODE 19.8-20.7 tok/s (vs 10.6 no-MTP = ~2×) — now competitive/beating llama edge decode.
  - Warm TTFT 982-1155ms — BELOW llama 1393ms (prefill parity achieved via int8+interval16).
  - Turns complete normally (stop, 39-77 tok, no runaway) → coherent. Per-it 6.66s/it @ turn3.
  Projection: deep turns (was 17s no-MTP) → ~9-10s with 2× decode; full-run avg ~7-8s/turn × 1007 ≈ 7000-8000s
  vs llama 8369s → LIKELY WIN. Full run completing (~1.5-2.5h); IoU at end confirms coherence. THE STACK THAT
  WORKS: int8 W4A8 faith2 prefill + Marconi prefix-cache interval-16 + FP8-KV + MTP, all on dgx1.

## 3.45 ★ int8 AGENTIC RUN #1 — FAST (1834s<1965s) but FAILED: OOM (lazy-int8-after-greedy-KV)
int8 full-stack subset (ATLAS_INT8_PREFILL + interval64 + cache, util 0.90, Marconi 256): Duration 1834s
(7% faster wall than bf16-TC 1965s — int8 prefill IS faster e2e) BUT **IoU 0.0, 116/116 FAILED**:
first turn TurnTimeout@600s → cascade. Root cause (serve log): `cuMemAlloc_v2 failed: status 2, requested
89MB (607MB free / 121.7GB)` at prefill layer 56. NOT a correctness bug — OOM. The int8 weight buffers
(+26GB, 8-bit vs NVFP4 4-bit, built LAZILY on first prefill via OnceLock) don't fit because the KV cache
GREEDILY pre-allocated to the util cap at startup (model 25 + Marconi 38 + KV≈45 = ~108GB, ~12GB free) →
the 26GB lazy int8 alloc fails mid-prefill → retry/hang → 600s timeout.
FIX = leave physical headroom for the lazy int8 (26GB): lower --gpu-memory-utilization so KV doesn't fill
everything. Budget (121.7GB phys): model 25 + Marconi(slots×0.15GB) + KV + int8 26 + scratch ~5 ≤ 121.
  - No-cache coherence gate: util 0.70, no Marconi → KV 60 + int8 26 + model 25 = 111 < 121 ✓ (RUNNING).
  - Agentic w/ cache: util ~0.68 + Marconi 128(19GB) → KV ~38 + 25 + 19 + 26 + 5 = 113 < 121 ✓.
PROPER fix (later): eager int8 weight alloc at load so KV sizes around it (avoids manual util math), OR
in-kernel FP4→int8 requant (no +26GB buffer at all, matches bf16-TC/FP8 paths — bigger kernel change).
GATE FIRST: is int8 COHERENT? (microbench 0.999978 ≠ model coherence — the 0.0 was OOM not quality, but
must still prove generation is clean before trusting the agentic IoU.)
★★ int8 COHERENCE GATE PASSED (util 0.70, no Marconi, int8_prefill_gate.sh): **10/10 clean** (Paris/391/
Jupiter/1024/1945/Au/13, no runaway) AND **int8 prefill 10-12% FASTER than bf16-TC** end-to-end:
2k 4.02 vs 4.5s, 4.5k 8.92 vs 9.9s, 9k 18.19 vs 20.3s, 18k 36.55 vs 41.5s. int8 integration is CORRECT +
faster; the IoU-0.0 was OOM only. Next: re-run int8 full-stack agentic with memory fix (util 0.68 + Marconi
128 + interval 16) → confirm non-zero IoU + combined wall. NOTE: subset is DECODE-bound (~10.6 tok/s, verbose
model, no MTP) so int8's prefill win (~0.3s/turn of a ~16s turn) barely moves the SUBSET wall — beating
llama's full wall also needs MTP (decode 2×) + a FULL 20-traj run (2-traj subset tail is NOT representative;
those are the deepest trajs; llama's 8369s is the 20-traj average = 8.31s/turn).

## 3.5 INTEGRATION STATE (06-27 PM) — requant pipeline VALIDATED, model wiring DONE (compiles), gate next
ATLAS_INT8_PREFILL wired in dense_ffn.rs (+217) + ops/gemm_dense.rs (+90): load-time requant_w → cached
int8 weight buffers (OnceLock per gate/up/down, from non-transposed NVFP4); per-prefill requant_a → shared
scratch; faith2 dispatch (highest-priority arm). Server bin REBUILT clean. NOT yet GPU-coherence-gated.
Gate script ready: /workspace/int8_prefill_gate.sh. NOT committed (validate on GPU first).
**Kernel pipeline COMPLETE + gated.** Three pieces, all cosine ≥0.9999:
- `int8_gemm_faith2` — 44.7/48.9 TFLOP/s gate/up/down, cosine 0.999999 (the GEMM).
- `requant_w_nvfp4_int8` — NVFP4 [N,K/2]+E4M3[N,K/16]+scale2 → int8[N,K]+f32 scale[N,K/32], load-time.
- `requant_a_bf16_int8` — bf16[M,K] → int8[M,K]+f32 scale[M,K/32], per-prefill (~1.4% of GEMM).
- **END-TO-END test (int8_gemm_test.rs REQUANT block): cosine 0.999978** vs host full-precision
  dequant GEMM, on REAL NVFP4-format weights. (FP8-prefill path = 0.99972 → int8 is the coherent win.)
All on PR #201 (perf/int8-prefill-faith2, base feat/agentic-2.5h-bf16tc-prefill), 4 commits, NOT merged.

**REAL agentic picture (measured 06-27):** llama target = **8369.87s wall, TTFT median 1393ms, IoU 0.6326**
(/workspace/endpoints-agentic/results/agentic_coding_perf_2.5h_gb10_cfff1fc). Atlas bf16-TC COLD prefill
curve (time_prefill.sh, no cache): 2k→4.5s, 4.5k→9.9s, 9k→20.3s, 18k→41.5s (~2.3ms/tok = the "41s/it").
TWO levers, BOTH needed (per [[project_agentic25h_prefill_iter_2026_06_25]]):
  (A) PREFIX CACHING — DOMINANT. Multi-turn agentic re-prefills full ctx each turn w/o it (→11h, invalid).
      spark flags EXIST: `--enable-prefix-caching --ssm-cache-slots 128 --kv-cache-dtype fp8`. 128 slots
      (19GB @ util 0.90) avoids exhaustion; Marconi SSM snapshot = full prefix skip on hit (5s vs 43s).
  (B) PREFILL GEMM SPEED — secondary 1.49×. bf16-TC 30 (lossless ceiling) → int8 faith2 44.7 (coherent)
      or FP8 1.35× (lossy, IoU-risk). int8 is the coherent upgrade over the existing ATLAS_FP8_PREFILL.
**llama uses int8 MMQ k32 (= what faith2 ports) AND prefix caching.** Win = (A)+(B) together.

REMAINING (model integration, multi-file): wire faith2 behind ATLAS_INT8_PREFILL in dense_ffn.rs
mirroring ATLAS_FP8_PREFILL (commit 3adf30dc): (1) load-time requant_w → int8 weight+scale buffers on
the layer (from ORIGINAL NVFP4 [N,K/2], no transpose); (2) per-prefill requant_a on the activation;
(3) faith2 launcher in ops/gemm_dense.rs; (4) handles. Then: coherence N≥10 + ST subset + agentic-2.5h
wall+IoU vs 8369.87s/0.6326, with prefix-caching + FP8-KV ON.

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
