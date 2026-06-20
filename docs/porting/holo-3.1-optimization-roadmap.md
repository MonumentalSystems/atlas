# Holo 3.1 Atlas Optimization Roadmap

Date: 2026-06-19. Source: multi-agent investigation (workflow wf_681a86d7-f37) +
captured baseline (`scripts/bench_holo_atlas.py`). Measuring stick: vLLM
`results.csv` (tg128 c1=75.4, c2=118.2, c5=151.2, c10=196.0; pp2048 c1=4540).

## Diagnosis

- **C=4 aggregate ≈ C=1 (decode doesn't batch).** Three serializations:
  1. GDN/SSM mixer per-sequence `for i in 0..n` loop — `crates/spark-model/src/layers/qwen3_ssm/trait_decode_multi_seq.rs:59`. Kernel is batch-capable (`gated_delta_rule.cu:637`, indexes per-batch by `blockIdx.y`) but called N× with batch=1.
  2. MoE N≥4 per-token fallback — `trait_decode_multi_seq.rs:140-157`.
  3. CUDA graphs disabled for N≥2 — `decode_a2.rs:191` (SSM state ptrs baked into kernel args).
- **C=1 = 45 vs 75:** C=1 already graph-captures (`decode_a.rs:194`); gap is in-graph compute, dominated by BF16-expanded SSM GEMVs. FP8-SSM is the lever.
- **Prefill ~10× gap (460 vs 4540 tok/s):** GDN FLA chunked recurrence (64-tok chunks, `trait_prefill_recur.rs:108`) + MoE prefill. Needs its own dig.
- **Memory ~53 GB pre-KV:** BF16 SSM expansion + never-freed conversion sources + rollback ring allocated on non-spec path.

## Ranked roadmap

| # | Item | Axis | Effort | Risk | Kind |
|---|------|------|--------|------|------|
| 1 | Batch GDN/SSM decode mixer across N seqs (kill per-seq loop) | speed | weeks | high | mixed |
| 2 | Indirect SSM state ptrs via device table → re-enable CUDA graphs N≥2 (pairs w/ #1) | speed | weeks | high | mixed |
| 3 | Batch MoE for N≥4 (re-bench grouped GEMM on Holo shape) | speed | days | med | mixed |
| 4 | Batch LM head: per-row GEMV loop → one GEMM (`decode_a2.rs:344-384`) | speed | days | med | pure-rust |
| 5 | `WeightStore::take/free` + free embed F32 source (~2 GB) | memory | hours | low | pure-rust |
| 6 | Free post-concat intermediate BF16 SSM buffers (~1.5 GB) | memory | hours | low | pure-rust |
| 7 | Gate decode rollback ring on spec flags not num_ssm_layers (~3-6 GB) | memory | hours | med | pure-rust |
| 8 | Fix native FP8-SSM prefill crash (layer 36) → FP8-SSM default (C=1 + ~3.5 GB) | mixed | days | med | mixed |
| 9 | Batch short-prompt prefill admission (`phase_start_prefills`) | speed | days | med | pure-rust |
| 10 | Post-build memory audit w/ per-component breakdown (diagnostic, do first) | memory | hours | low | pure-rust |
| 11 | KV-cache dtype bf16→fp8 default (quality-gated) | memory | hours | med | pure-rust |
| 12 | Bench retarget at Atlas — DONE (`scripts/bench_holo_atlas.py`) | speed | hours | low | pure-rust |

## Dependencies
- #2 depends on #1's device-side SSM state pointer table → treat #1+#2 as one refactor. Use the MTP `decode_batched_inner` GEMM structure (`trait_decode_batched.rs:130-417`) and batched-prefill `h_state_ptrs[]` template as in-tree models.
- #8 prerequisite for FP8-SSM default (delivers ~3.5 GB + C=1 SSM-GEMV speedup).
- #5 API enables embed/attn/SSM source frees.
- #10 lands first among memory items so #5/#6/#7/#8/#11 are measurable.

## Execution order (this session → later)
1. #10 audit (measure), #7, #6, #5 memory quick-wins (~6-9 GB, safe). Build once, verify no perf regression + memory drop.
2. #8 FP8-SSM (C=1 + memory).
3. #1+#2+#3 batching refactor (C>1). #4, #9, #11 in parallel.

## Progress (2026-06-19 session)

Landed + verified (all correctness-checked via n=1 vs n=4 needle test):
- **CUDA graphs for multi-seq decode** (`ATLAS_DECODE_GRAPHS_MULTISEQ=1`, `decode_a2.rs`). The disable was over-conservative: scheduler keeps active seqs in contiguous SSM slots [0..n) (verified empirically), so per-position state addresses are stable across replays. Capture/replay machinery already existed; just gated. Result: **c8 48→78 tok/s (+62%), c4 48→60 (+27%)**.
- **Batched SSM projections** (`try_decode_multi_seq_ssm_batched`): QKVZ + out_proj as single M=N GEMMs (weights read once). Correct.
- **Memory: ~10 GB freed** — rank-6 dead post-concat SSM buffers (~1 GB, pre-KV 53→52) + KV right-sized to ~440k tokens / 8.4 GB (was 907k/17 GB) via 32K/C8/util-0.52 config.
- Bench harness `scripts/bench_holo_atlas.py`, serve `scripts/holo_serve.sh`.

Tried + REVERTED (measured loss, evidence kept):
- Grouped-GEMM MoE for N>=4 (`ATLAS_MOE_GROUPED_DECODE`, default off): c4 60→33 even under graphs. Holo's 256-expert/512-intermediate shape doesn't suit grouped GEMM at small token counts. Per-token MoE wins.

Current vs vLLM (tg128, graphs on): c1 48/75 (64%), c2 42/118, c4 60/145 (42%), c8 78/175 (45%).

REFINED diagnosis: decode is **per-seq compute/bandwidth bound**, NOT launch bound. Step time ≈ ~8ms fixed + ~12ms/seq (vLLM ~7ms/seq). Graphs removed launch overhead; remaining gap is per-seq kernel efficiency.

Next levers (ranked):
1. **FP8 SSM decode weights** (vs BF16): halves the ~2 GB/step SSM weight read (fixed per-step cost) AND ~3.5 GB memory. Blocked by native-FP8-SSM prefill crash at layer 36 (rank 8) — but a decode-only FP8 path (BF16/NVFP4 prefill, FP8 decode GEMV) sidesteps the crash.
2. **MoE decode kernel occupancy**: per-token MoE (8 experts × 40 layers) dominates per-seq cost; needs a fused small-batch MoE decode kernel (forward_k2/k3 style extended), not grouped GEMM.
3. **Prefill** still ~10× gap (460 vs 4540) — separate GDN/MoE prefill efficiency problem.
4. C=1 (48 vs 75): same per-seq efficiency (FP8 SSM + MoE).

## FP8 SSM decode + PR #177 (2026-06-19, later)

**FP8 SSM decode overlay** (`ATLAS_HOLO_FP8_SSM_DECODE=1`, `linear_attn_arms.rs` + batched path): install on-disk block-scaled FP8 QKVZ/out_proj for the decode GEMV/GEMM (`w8a16_gemv`/`w8a16_gemm`) while keeping BF16 for prefill (no crash). Result: **c1 48→61 tok/s (64%→81% of vLLM)** — halving SSM weight bandwidth is the single-stream lever. c>1 unchanged (SSM weight is shared per-step there; per-seq MoE/attention dominates). Correct (n=1/prefill/n=4-needle all pass). Cost: +~750 MB (BF16 kept for prefill).

Combined config now (graphs + FP8 SSM decode): c1 61 (81%), c2 43, c4 60 (41%), c8 78 (44%).

**PR #177 (Avarok-Cybersecurity/atlas) — applicable, recommend porting.** Same model family (Qwen3.6-35B-A3B FP8). Dominant fix: block-scaled FP8 prefill made DEFAULT (was falling through to single-scale `fp8_gemm_n128`, collapsing per-128-block scales → agentic degeneration, B1 drift 1400→100-200). Local repo HAS the `fp8_gemm_t_blockscaled` kernel + `ATLAS_FP8_W8A8` gate but NOT #177's `fp8_blockscaled_prefill_enabled()` default-on across all 6 prefill sites. Applicability to Holo:
- SSM + attention projections are FP8 block-scaled → block-scaled prefill improves **accuracy** (agentic/qwen3_coder serving) and is the likely key to making **full native FP8 SSM (decode+prefill)** work → drop BF16 → **~3.5 GB memory win** on top of the decode-speed win.
- Bugs #2-4 (thinking-budget guillotine at 256 on adaptive thinking, tool-arg coercion in stringified nested arrays, EOS guard on non-tool turns) are model-agnostic agentic-correctness fixes.
- NVFP4 MoE path: #177 found no analogous bug (per-16 micro-scales already correct) — so no MoE change for Holo.

## C=1 profile breakdown (2026-06-19, `--profile`)

`PROFILE tok=45: total=16.4ms attn=4.4ms(10L) ssm=10.9ms(30L) head=1.1ms`

By function: **MoE ≈ 7 ms (43%)**, SSM mixer ≈ 5 ms (qkvz FP8 GEMV 3.4 ms — bandwidth-optimal at ~112µs/layer), attention proper ≈ 3.4 ms, lm_head 1.1 ms. To hit 75 tok/s (13.3 ms) need to cut ~3 ms — MoE is the only target big enough.

**MoE is ~3.5× off bandwidth — NOT an occupancy problem** (corrected: an earlier
"~18 blocks" note was a misread). `moe_expert_gate_up_shared`'s grid
`[div_ceil(n,8), top_k+1, 2]` uses `n = inter = 512` (OUTPUT dim, one token at
decode) → grid `[64, 9, 2]` = **1152 blocks, well-occupied** (`moe_expert.rs:230`;
`forward.rs:351` passes `inter, h`). MoE moves ~540 MB/step (8 routed + shared,
gate/up/down NVFP4) but takes ~7 ms vs ~2 ms bandwidth-optimal. Gap = **w4a16 GEMV
micro-efficiency at M=1** (B_packed coalescing, warp-reduction GEMV in
`moe_shared_expert_fused.cu`), not tiling — deep kernel micro-opt, uncertain payoff.

Revised next-lever ranking:
1. **Port PR #177** (block-scaled FP8 prefill default + agentic fixes): accuracy +
   path to full native FP8 SSM → drop BF16 → ~3.5 GB leaner. Moderate, validated.
2. **Attention FP8** for q/k/v/o decode projections if currently BF16 (attn proper
   3.4 ms/10L) — analogous to the SSM FP8 decode win.
3. MoE w4a16 GEMV micro-optimization (hard, deep CUDA).

## C>=2 profile + precise next levers (2026-06-19, ATLAS_MS_PROFILE)

Multi-seq decode step breakdown (n=8, eager+synced; relative split is the signal):
`total~87ms = ssm 56ms (30L, 64%) + attn 24ms (10L, 27%) + head 8ms (9%)`

The C>=2 gap is NOT projections (batched), LM head (batched, L2-masked), or graphs
(on). It's the parts that scale with N inside the layers:
1. **Per-token MoE (xN)** — biggest single cost. 8 separate `ffn.forward()` per step
   across 40 layers. Grouped GEMM (`forward_prefill`) is a measured LOSS for Holo's
   256-expert/512-intermediate shape even under graphs. Needs a purpose-built batched
   small-M MoE decode kernel (the deep, high-value nut).
2. **Per-seq SSM conv/gdn recurrence (xN)** — N serial small kernels (poor GPU
   occupancy). Fully batchable to one `batch=N` launch each:
   - `gdn_decode`: already batch-capable; pass `h_state = states[0].h_state` (pool
     base; slots are contiguous [0..n)) + contiguous q/k/v from a `[N, conv_dim]` conv
     output.
   - `gated_rms_norm`: use the existing `gated_rms_norm_prefill` (grid heads×N).
   - `conv1d_update_l2norm`: ONLY missing piece — add an `input_stride` param to the
     kernel (f32 + bf16) so it can read the qkv slice of the `[N, qkvz_size]`
     deinterleaved buffer (stride=qkvz_size). Then batch=N, conv_state=pool base.
   - BA+gates: keep per-seq (cheap, 64 outputs) writing into a contiguous `[N, 2*nv]`
     gates buffer, or batch later.

Both are kernel-adjacent efforts (revert-safe; verify vs needle/coherence). The SSM
conv/gdn batching is the more tractable; the batched MoE decode kernel is the larger
win but harder. C=1 (62/75) is per the user "fine"; focus is C>=2.

## C>=2 kernel-level scope (final, 2026-06-19)

Exhaustive feasibility done. The two C>=2 levers both require core-kernel changes
(revert-safe but high-care; verify vs needle/coherence + SSM checksum):

1. **Batch SSM recurrence (gdn + gated-norm)** — partial SSM-mixer win.
   - `gated_delta_rule_decode_f32` (kernels/gb10/holo-3.1-35b-a3b/nvfp4/gated_delta_rule.cu
     + common) has NO stride params (unlike `gated_delta_rule_prefill` which has
     qk_stride/v_stride/gb_stride). To batch decode across N reading the interleaved
     `[N, conv_dim]` conv output, ADD qk_stride/v_stride/gb_stride to the decode kernel
     (mirror the prefill kernel) + op wrapper + 3 callers (ssm_forward, this path, MTP
     conv_gdn pass d_inner/v_dim/2*nv to preserve behavior).
   - Then in `try_decode_multi_seq_ssm_batched`: per-seq ba_gates→gates[N,2*nv] +
     per-seq conv→conv_out[N,conv_dim] (no conv kernel change needed; batch=1 writes
     to per-i offset), then ONE batched gdn (h_state=states[0].h_state pool base,
     strides=conv_dim) + ONE `gated_rms_norm_prefill` (exists) + batched out_proj (done).
   - Buffers (provably fit, M>=N): conv_out=ssm_conv_out_f32, gdn_out=ssm_qkvz,
     normed_out reuses ssm_conv_out_f32 (free after gdn), ssm_out=moe_output.
   - RISK: stride bug in the recurrence kernel = silent SSM-state corruption. High care.

2. **Batched small-M MoE decode kernel** — the BIGGER C>=2 cost (per-token MoE xN,
   ~MoE is the largest linear-scaling term). `forward_prefill` grouped GEMM is a
   measured loss (sort/permute overhead at tiny per-expert M). Needs a purpose-built
   kernel that processes N tokens' top-k expert GEMVs in one launch without the sort
   (e.g. token-major expert dispatch). The largest win, the hardest.

Both are dedicated kernel efforts, not quick edits. C=1 (62/75) is "fine" per user;
C>=2 parity hinges on #2 primarily, #1 secondarily.

## Empirical MoE-kernel result (2026-06-19, smem-A experiment)

Attempted: cache the per-token activation A[K] in shared memory in
`moe_expert_gate_up_shared` (every block re-reads the same A in the K-loop).
Result: **no-op** (c1 61.4, c4 57.5, c8 79.4 — all within noise). Reverted.

Conclusion (empirical, not just analysis): A is already L2-resident; the MoE
expert GEMV is **weight-read bound** (each output's unique NVFP4 weight row +
unpack), in an M=1 warp-per-output structure. It is already fully GPU-occupied
at one token (~1152 blocks), so batching N tokens gives no occupancy win and the
weight bytes are inherent (64 expert activations at N=8, same as vLLM). The ~2.3×
gap vs bandwidth-optimal is the warp-per-output GEMV structure + per-output
warp-reduction, NOT activation traffic or occupancy.

So vLLM's ~2× C>=2 decode edge must come from a more efficient MoE expert kernel
(tensor-core grouped-GEMM tiling that achieves higher effective weight-read
throughput than warp-per-output GEMV). That is the single remaining C>=2 lever and
it is a dedicated, research-grade CUDA kernel effort — not a tweak. Also note:
steady-state decode (≈87ms/8-tok step ≈ 92 tok/s) is well above the bench's 78
tok/s c8, so a few % of the c8 gap is prefill-admission ramp (pure-Rust lead #2),
but the dominant gap is the MoE-bound steady decode step.
