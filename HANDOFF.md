# DFlash + Harness Handoff (2026-06-29)

Status snapshot for the DFlash speculative-decode work on Holo-3.1 and the
official-harness integration. Everything below is **pushed to the
MonumentalSystems fork**. Nothing is merged to Avarok `main` yet.

---

## TL;DR

- **Config-driven RoPE for the DFlash drafter is DONE and BUILDS** but is
  **UNTESTED** (test was interrupted before the run). It needs one validation
  run with the Qwen3.5 drafter.
- The **real DFlash-on-Holo blocker is verify-step cost** (verify = 7.26×
  decode), which the throughput-aware MTP gate correctly auto-disables. This is
  kernel work, explicitly **out of scope** per the user. RoPE fixes are
  necessary-but-not-sufficient.
- The **official-harness DFlash plumbing** (`--dflash-check` + Holo round) is
  done and pushed.

---

## Branches (all on `origin` = github.com:MonumentalSystems/atlas)

| Branch | Contains | Tested? |
|---|---|---|
| `test/holo-dflash-209fixes` | #209 RoPE fix cherry-picked onto #210 Holo stack + **config-driven RoPE** | RoPE-fix: **YES** (e2e, drafter loads, log-confirmed). Config-driven RoPE: **NO** (interrupted) |
| `feat/dflash-harness-wip` | Fork of PR #209 + harness DFlash plumbing (`--dflash-check`, Holo round) | Plumbing exercised via the test runs |
| `feat/holo-official-harness` (PR #212, draft) | vision test + matrix harness | YES (6-model matrix) |
| `feat/holo-fp8-on-holo-gb10` (PR #210) | native FP8 | YES |
| `upstream/holo-gb10` (#203), `upstream/vision-vit-perf` (#202), `feat/gpu-nvfp4-dequant` (#211) | the landing stack | YES |

### Commits on `test/holo-dflash-209fixes` (off `feat/holo-fp8-on-holo-gb10`)
```
b45914e fix(dflash): config-driven RoPE selection (yarn vs standard)   <-- NEW, UNTESTED
d6e3721 fix(dflash): drop orphaned profiling refs from cherry-picked RoPE fix
65bc7b0 fix(dflash): YaRN→standard RoPE, position IDs, noise0, argmax  <-- from #209 (743276f)
```

---

## What was done this session

### 1. Mined PR #209 (Sujimoshi fork, `fix/dflash-correctness`)
#209 fixes 3 DFlash correctness bugs on **Qwen3.6-27B-NVFP4** + 2 perf/loader
deps. Accounting against our needs:

| #209 piece | Decision |
|---|---|
| `743276f` RoPE→standard + position-IDs + noise0 + argmax-skip | **Integrated** (cherry-picked) |
| `4da06ee` embed-stride `fp32=4→2` | Already in our main (no-op) |
| `5de5a1b` BF16/FP32 conv | **Not needed** — Qwen3.6-27B-target-specific. Our holo/qwen3.5/qwen3.6-35b targets bind `gdn_k`→`gated_delta_rule_decode` (BF16) + BF16 conv → dtype-matched, no mismatch. (Verified in `qwen3_ssm/init.rs:47,70`.) Only triggers in the K∉{2,3,4,17} sequential fallback anyway, which γ=4 (K=4, fused WY4) never hits. |
| `f306a86` native-U8 NVFP4 SSM loader | **Deferred** — our dequant→BF16 path already loads our models; #209's is a memory optimization. Evaluate vs our approach + #211. |
| `9699af6` norm-cadence-skip (+1.1%) | **Deferred** — adds `do_norm` to `gdn_decode` kernel sig; collides with the wmma-vblock GDN rewrite. Evaluate against that. |

No hidden 4th fix in #209's 31-file diff — all map to the 7 commits.

### 2. Config-driven RoPE (commit `b45914e`) — the real gap #209 left
**Problem:** #209 *hardcodes* standard RoPE. Correct for the Qwen3.5 / gemma-4
drafters (`rope_scaling=null`), **wrong** for the Qwen3.6 drafter
(`rope_type=yarn, factor=64`). Hardcoding either corrupts the other's
attention rotations → ~0% acceptance.

**Verified drafter configs (on gx10 HF cache):**
- `z-lab/Qwen3.5-35B-A3B-DFlash`: `rope_scaling=null`, θ=1e7 → **standard** (Holo's production drafter)
- `z-lab/Qwen3.6-35B-A3B-DFlash`: `rope_scaling=yarn factor=64`, θ=1e7 → **yarn**
- `z-lab/gemma-4-26B-A4B-it-DFlash`: `rope_scaling=null`, θ=**1e6** → standard (note θ!)

**Fix:**
- `crates/spark-model/src/weight_loader/dflash_loader.rs`: `DflashConfig` now
  parses `rope_theta` (default 1e7) + `rope_scaling: Option<DflashRopeScaling>`
  (reads `rope_type` or legacy `type` alias, `factor/beta_fast/beta_slow/
  original_max_position_embeddings`). `is_yarn()` helper.
- `crates/spark-model/src/layers/dflash_head/from_weights.rs`: builds a
  YaRN-interpolated inv_freq table when `rope_scaling` is yarn, else a standard
  table, using the config's own θ. Faithful YaRN math restored from the
  pre-#209 path (NTK-by-parts, find_correction_dim/ramp).
- **Compiles clean** (`cargo check -p spark-model`, holo-3.1-35b-a3b/nvfp4).
- Binary built at `/home/ms/atlas/target/release/spark` (mtime 2026-06-28 21:00).

### 3. Official-harness DFlash plumbing (on `feat/dflash-harness-wip`, pushed)
- `tests/single_gpu_suite.py`: `run_dflash_test` scrapes `/metrics`
  `atlas_spec_decode_verify_total{k,outcome}`, computes accept-rate, reports
  accept% + decode tok/s. Opt-in `--dflash-check`. Floor via
  `ATLAS_DFLASH_MIN_ACCEPT` (default 5%).
- `tests/run_all_models.py`: `TestSpec.dflash` flag; Round 12 = Holo-3.1-35B +
  `z-lab/Qwen3.5-35B-A3B-DFlash`, serve flags via `extra_args`.

---

## Test results so far (gx10, e2e)

Served Holo-3.1-35B-A3B-NVFP4 (`/tank/holo-bf16kv-test`) + Qwen3.5 drafter,
`--dflash --dflash-gamma 4`, binary with #209 RoPE fix (NOT yet the
config-driven build):

- ✅ Drafter resolves + installs; `DFlash ENABLED (γ=4)`.
- ✅ RoPE fix live: `DFlash RoPE inv_freq: 64 pairs, theta=10000000 (standard, no interpolation)`.
- ❌ **MTP gate DISABLES spec-decode:**
  `verify_multiplier=7.26, max_effective=4.0 (decode=10.70ms verify=77.70ms) => DISABLED (net-negative at any acceptance)`.
  → `Verify events: 0`. Harness reported `DFlash 0/1 (accept 0.0%)` = correct
  gate-disabled signal.
- Verify cost is **structural**: `ATLAS_DFLASH_CTX_WINDOW=96` did NOT reduce it
  (stayed 78ms / 7.25×) — it's the target K-verify + drafter forward, NOT the
  drafter ctx-attention. ctx_window is not the lever.

**Interpretation:** DFlash cannot win on Holo at γ=4 regardless of acceptance
(best case 4 tokens / 7.26 decode-cost = 0.55×). The blocker is verify cost,
not acceptance or RoPE. Default `ATLAS_DFLASH_DRAFT_CAP=1` also means only K=2
verify runs (K=γ path still corrupts SSM state — pre-existing, out of scope).

---

## NEXT STEP (the one unfinished thing)

**Validate config-driven RoPE** — one run, ~5 min when a GPU box is free:

1. Binary is already built: `scp /home/ms/atlas/target/release/spark gx10-9959:/tmp/spark-dflash`
2. Test harness is ready: `/tmp/claude-1000/-home-ms-atlas/2d4f11c0-d64a-4e1d-a696-94dec811f7ab/scratchpad/dflash_test.py`
   (serves Holo NVFP4 + Qwen3.5 drafter `--dflash --dflash-gamma 4`, runs the
   suite `--dflash-check`). Also on gx10 at `/tmp/dflash_test.py` (+ suite at
   `/tmp/dflash_suite.py`).
3. Run: `ssh gx10-9959 'cd /tmp && python3 -u dflash_test.py'`
4. **Pass criterion:** logs show `DFlash RoPE inv_freq: ... (standard, no
   interpolation)` for the Qwen3.5 drafter (`rope_scaling=null` → standard
   branch selected by config, NOT hardcoded). Same correct behavior as before,
   now config-driven. Output stays coherent. (Gate will still disable spec —
   that's expected and unrelated to RoPE.)
5. To validate the **yarn** branch too: serve Qwen3.6-35B-A3B + the Qwen3.6
   drafter; expect `DFlash RoPE inv_freq: ... YaRN (factor=64, ...)`.

Then `test/holo-dflash-209fixes` is ready to graduate into a real PR (pair with
the harness plumbing from `feat/dflash-harness-wip`).

## Explicitly OUT OF SCOPE (user: "not our issue")
- Verify-step cost reduction (the actual DFlash perf blocker) — kernel work.
- K=γ verify SSM-state corruption (the `ATLAS_DFLASH_DRAFT_CAP=1` reason).

---

## PR #199 (`fix/moe-shared-expert-bf16`) — effectively already incorporated

#199 ("mixed-precision + per-channel FP8 expert loading; AgentWorld-35B,
Ornith-1.0-35B-FP8") is OPEN + **CONFLICTING** with main. Reviewed its 3 logical
changes against our stack (`feat/holo-fp8-on-holo-gb10`) — **all three have
functional equivalents already**, grown independently via #200 (merged) + our
Holo-FP8 / qwen35-loader work. That independent re-solving is *why* #199
conflicts. The only difference is structure (a named `scalar_scale_f32` helper +
explicit channel-strategy comments vs. our inline logic).

| #199 part | Equivalent in our tree |
|---|---|
| (1) BF16 shared/routed experts → `quantized_any()` fallback | `weight_map/fp8_lut.rs:265/274/283` (qwen35 MoE gate/up/down) + gemma4 loader |
| (2) Accept BF16-stored per-tensor FP8 `weight_scale` (`(bits)<<16`) | `weight_map/model_a.rs:214,250` (both dequant sites — exactly #199's targets) + `quant_helpers.rs:114`, `ssm_qwen35_more.rs:275,334` |
| (3a) `float-quantized` quant gate | `serve.rs:753` + `nvfp4_detect.rs:49` (our Holo-FP8 work) |
| (3b) Per-channel `weight_scale [N]` handling | `quantize_fns.rs:56/78/119` |

**Conclusion:** nothing in #199 is missing from our tree; **#199 can likely be
closed as superseded.** Do NOT cherry-pick #199 code.

**One unverified case (code-reading can't close it):** whether
**`deepreinforce-ai/Ornith-1.0-35B-FP8`** specifically loads — its exact
compressed-tensors `strategy="channel"` + BF16 linear-attn-on-ignore-list combo
is #199's narrowest case. Building blocks are all present, but confirm with a
one-command load test (checkpoint is downloaded on gx10):

```
# build any current target binary, then:
ssh gx10-9959 'docker run --rm --gpus all -v /home/ms/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /home/ms/atlas/target/release/spark:/usr/local/bin/spark:ro atlas-holo:cuda13.2-fp4test \
  serve deepreinforce-ai/Ornith-1.0-35B-FP8 --model-name ornith35bfp8 --port 8890 ... '
# PASS = loads without weight_scale/weight_global_scale errors + coherent generation.
```
If it loads → close #199. If it fails on the channel-strategy path → cherry-pick
ONLY #199's `quant_helpers.rs`/`ssm_qwen35.rs` per-channel routing (not the rest,
which we already have).

## Cleanup note
gx10 may still have `/tmp/spark-dflash`, `/tmp/dflash_test.py`,
`/tmp/dflash_suite.py`, and possibly a stopped `holo_dflash` container. The
`holo_dflash` container was removed at last cleanup; re-check with
`ssh gx10-9959 'docker ps -a | grep holo'`.

See memory `holo-dflash-state.md` for the full diagnostic history.
