# Laguna S 2.1 handoff

Date: 2026-07-21

## Repositories and pull requests

- Atlas worktree: `/home/ms/.codex/worktrees/laguna/atlas`
- Atlas branch: `feat/laguna-s-2.1`
- Atlas fork remote: `origin = MonumentalSystems/atlas`
- Atlas upstream remote: `avarok = Avarok-Cybersecurity/atlas`
- Atlas draft PR: https://github.com/Avarok-Cybersecurity/atlas/pull/350
- Recipe worktree: `/home/ms/.codex/worktrees/laguna/atlas-recipes`
- Recipe branch: `feat/laguna-s-2.1-recipe`
- Recipe draft PR: https://github.com/Avarok-Cybersecurity/atlas-recipes/pull/12

Keep all further Atlas work in draft PR #350. The recipe necessarily has its
own PR because it is a separate repository.

## User constraints

- Model: `poolside/Laguna-S-2.1-NVFP4`; do not use DFlash yet.
- Use FP8 KV, NFS shard prefetch, prefix caching, 200K logical context with KV
  overcommit, 16K prefill, 4 GB swap, SLAI with a 100 ms TBT deadline, and GPU
  memory utilization no higher than 0.80 unless required.
- Laguna has no SSM layers; keep SSM snapshot storage disabled in the recipe.
- Test baseline ISL prefill/decode plus concurrency C=1,2,4. Profile with
  Nsight Systems or an equivalent real profiler; `MS_PROFILE` is not timing
  evidence.
- Preserve model-card sampling and thinking behavior. Audit both Atlas and
  client prompt/tool reinjection.
- Exact MoE tiles matter. Investigate CUTLASS and other MoE prefill routes.
- Bank logical progress as commits and push it to the existing draft PR.
- The 500-line limit applies when cleaning up; do not expand scope solely to
  split historical files.

## Banked Atlas commits

- `c559370d` `spark-model: add Laguna-S-2.1 inference support`
- `c1926fd1` `spark-model: preserve Laguna BF16 shared experts`
- `b3e26427` `spark-server: align Poolside agentic behavior`
- `fac4d4f8` `spark-model: accelerate Laguna BF16 prefill`
- `442c84b2` `spark-model: fix native dense batched decode`
- `a49a64fc` `kernels: correct Laguna parameter metadata`
- `2088fd74` `spark-server: repair punctuation-drifted agent cwd`
- `96c1cdf3` `spark-server: honor Laguna agent defaults`

Recipe commits are `d79e215`, `1928289`, and `273a8cb`.

## Current source behavior

- Laguna defaults to thinking enabled, including tool turns. Requests can
  disable it with `enable_thinking=false`.
- Omitted temperature/top-p/top-k use the Laguna MODEL presets
  `0.7 / 0.95 / 20`; explicit request values still win. Other models retain
  their existing `generation_config.json` behavior.
- Atlas still extracts the client cwd for validation, but Laguna suppresses
  the duplicate Atlas `<environment>` block. The Poolside parser contributes
  no second system/tool prompt.
- Bash `workdir` and cwd occurrences inside Bash command strings are repaired
  only when the parent is exact and the basename differs solely by
  punctuation/case.
- The recipe retains a measured operational `--max-thinking-budget 1024`.
  A diagnostic with the model-owned 8192 cap was worse and was reverted.

## Validation completed

- `cargo fmt --all`
- `git diff --check`
- `ATLAS_SKIP_BUILD=1 RUSTC_WRAPPER= cargo check -p spark-server --tests`
- Poolside parser prompt-reinjection regression test
- Laguna duplicate-cwd-injection regression test
- Three path sanitizer tests, including Bash command cwd repair
- CUDA 13.2 release builds: 152/152 Laguna NVFP4 kernels
- Startup audit proved NFS prefetch across 14 shards, prefix caching, FP8 KV,
  0.80 memory utilization, 200K KV overcommit, 16K prefill, 4 GB swap, and
  SLAI at 100 ms.
- Recipe: `sparkrun recipe validate ...` passes and `sparkrun run ...
  --dry-run` renders the intended command. The dry run prints harmless
  read-only Hugging Face `.no_exist` cache warnings.

The full all-features Clippy gate is blocked on this Linux host by the known
Apple `objc2` dependency issue. Host-linked tests also require NCCL; targeted
tests run successfully in the CUDA container.

## Agentic harness evidence

Use this isolated client configuration so OpenCode does not inherit the stale
global Qwen model label:

```bash
OPENCODE_CONFIG=/tmp/opencode-laguna-card.json \
OPENCODE_DISABLE_PROJECT_CONFIG=1 \
OC_TIMEOUT=600 \
/home/ms/atlas/bench/fp8_dgx2_drift/harness/run_tier.sh \
  laguna-s21-card 1 --container hardcore_aryabhata --skip-warmup
```

Live Atlas logs then correctly show
`model=poolside/Laguna-S-2.1-NVFP4`, not `qwen3.6-35b-a3b`.

The validated 1024-budget run had two reasoning-only empty starts, then a
successful retry: 2 files, valid Cargo manifest, webserver scorer pass, no path
drift, 6 tool calls, and 442 seconds wall time. It scored
`followed_directions=false` because Laguna wrote the project and tests but said
it would run build/curl instead of issuing the final Bash commands.

A diagnostic with confidence early-stop disabled, inter-tool prose raised to
3072, and the model-owned thinking cap produced reasoning-only outputs of 1331
and 4539 tokens with no tool call. The third retry was terminated to avoid more
GPU time. Those experimental source changes were reverted. Do not reintroduce
them without a more targeted hypothesis.

## Hugging Face chat template

The checkpoint's `tokenizer_config.json` contains only:

```jinja
{% include 'chat_template.jinja' %}
```

Atlas currently loads one MiniJinja template and has no include loader. The
standalone checkpoint template also uses `{% generation %}` markers MiniJinja
does not implement and Python-style `tojson(ensure_ascii=False)`. Therefore the
runtime currently prefers `jinja-templates/laguna.jinja`, which expands the
include, removes non-rendering generation markers, and uses `tojson_hf` where
needed. A container must run with `/workspace/atlas` as its workdir or the
relative override is not found.

The cleaner follow-up is to resolve the checkpoint's exact standalone include,
strip only the non-rendering generation markers, adapt its JSON filter, prove
rendered byte/token parity, and then delete the Laguna override. Do not change
tool formatting without a parity test.

## Kernel audit and MoE findings

The startup warning reports 53 unresolved optional lookups. Building
`ATLAS_TARGET_MODEL=*` will not merge all model artifacts into Laguna; runtime
still selects the Laguna target. Most misses are irrelevant families (MLA,
SSM, Hyper-Connection, Q4K).

The base `moe_w4a16` module is embedded and marked `used`. Required Laguna
grouped handles (`ptrtable`, `_t`, `_t_k64`, fused gate/up) resolve; otherwise
model construction would fail. Missing MoE entries are primarily DeepSeek E8M0
and optional FP4 variants. They are still a useful performance lead:

- The grouped CUTLASS wrapper is gated by
  `ATLAS_HOLO_MOE_GROUPED_CUTLASS`, but loader preparation is hard-coded to
  `model_type == "holo3_1_moe"`, so setting Holo flags does not activate it for
  Laguna.
- Enabling the current Holo full transpose blindly could exceed memory across
  48 layers. Make this capability/layout driven and measure a bounded layer
  subset or build CUTLASS SFB data without retaining unnecessary transpose
  copies.
- The optional FP4 gate/up and down kernels are not compiled by the Laguna
  target today. Port and validate them before exposing their flags in the
  recipe.

## Performance evidence and next work

User-provided vLLM targets:

| context | TTFT | decode | prefill |
|---|---:|---:|---:|
| 1K | 467 ms | 19.3 tok/s | 2170 tok/s |
| 8K | 2249 ms | 19.0 tok/s | 3451 tok/s |
| 31K | 8342 ms | 18.2 tok/s | 3703 tok/s |

Concurrency target aggregate/per-stream: C1 `19/19.1`, C2 `35/17.5`, C4
`51/14.5`, C8 `84/11.4` tok/s.

Current observations: C1 prefill is roughly 790-803 tok/s. Decode observed
around C1 15.3, C2 15.7 aggregate, C4 19.1 aggregate. Exact tiles improved
cold 8K TTFT by 3.7%. Unified MoE improved prefill about 1.38x but added about
47 seconds startup and regressed decode roughly 15%, so it was excluded.

Prioritized leads:

1. Physically skip old KV blocks in sliding-window attention. Laguna masks but
   still computes from block zero; 36/48 layers use a 512-token window.
2. For concurrent prefill, validate `ATLAS_PREFILL_CODISPATCH=1` and
   `ATLAS_Q12_BATCHED_FIRST_CHUNK=1`. Permit Q12 only when all prefix-cache
   peeks are cold, share the aggregate 16K arena across streams, and keep one
   SSOT predicate.
3. For decode C=2/4, wire the existing BF16 batch-2 dense GEMV for Laguna QKV,
   shared expert, and head before adding batch-4.
4. Profile with Nsight Systems. `MS_PROFILE` distorts timing.
5. Make grouped CUTLASS/FP4 MoE support capability-driven for Laguna and test
   memory, correctness, prefill, and decode before adding recipe flags.
6. Revisit the native HF template and reasoning-only first-turn failures with
   rendered-token parity evidence.

The current Poolside NVFP4 model card says non-speculative GB10 engines are
typically 600-800 prefill tok/s and 13-14 decode tok/s, which matches Atlas more
closely than the user-provided target. Preserve both references and identify
the exact vLLM flags/build behind the higher numbers before declaring parity.

## Runtime/container notes

- Active test container name: `hardcore_aryabhata`
- CUDA image: `atlas-gb10:gdnf32-build` (CUDA 13.2)
- Old stopped backup: `hardcore_aryabhata-budget1024-backup`
- Model cache is mounted read-only from `/home/ms/.cache/huggingface`.
- Atlas worktree is mounted at `/workspace/atlas`; set container workdir to
  `/workspace/atlas` so template overrides resolve.
- The release target mount is `/tmp/atlas-cuda132-target.2wazQQ` on the host.

Before new measurements, verify the running binary and committed source agree,
then check the boot audit. The known-good serve command is the rendered recipe
command, with env `ATLAS_KV_OVERCOMMIT=1`,
`ATLAS_MOE_PREFILL_EXACT_TILES=1`, and the image-provided
`ATLAS_CUTLASS_NVFP4_GEMM=1`.
