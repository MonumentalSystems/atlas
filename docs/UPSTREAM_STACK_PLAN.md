# Upstreaming `wip-laguna-lora` → Avarok `main`: the stack plan

Status: plan (2026-07-26). Target upstream: `avarok/main` (Avarok-Cybersecurity/atlas).
Working branch: `wip-laguna-lora` on `origin` (MonumentalSystems/atlas).

## Current state

- `wip-laguna-lora` = `avarok/main` (22adb23e, #366) + **~127 commits**, zero upstream
  drift, but a **merge-laden history** (4 merge commits fold port branches in mid-stream:
  `4d830406` avarok sync, `657c474f` #335 lora, `34e0ff2a` #334 gguf-bonsai, `d93dfbd4`
  origin sync).
- **Three foundation PRs already OPEN on Avarok, and the branch sits on top of them:**
  - **#334** — generic GGUF loader + Ternary-Bonsai-27B (Q2 keep-packed).
  - **#335** — LoRA MoE per-expert/router + embed/lm_head/vocab overlay.
  - **#332** — lm_head batched-GEMV (35B + 27B).
  The Laguna work *depends* on #334 (Laguna-GGUF) and #335 (Laguna-LoRA).
- Iteration noise to squash on extraction (e.g. 5 thinking-watchdog commits →
  "disable the whole family"; fp8-kv calibration → freeze-on-first-observe; 4
  `instr(prefill)` timers).
- **Hold / exclude:** `9018fcec wip(dflash)` (degenerate + net slowdown), the
  uncommitted native-BF16 loader (opt-in, not yet correct), internal handoff docs
  (`b70d76c5`, `61d28e5e`).

## Strategy: split by AUDIENCE, three tiers

### Tier 1 — land the open foundations (#332 / #334 / #335)
Already up; everything depends on them. Just rebase current + get reviewed/merged.
(Rebasing a PR branch is a force-push — left to the PR owner.) Removes ~25 commits
from the delta and de-risks the rest.

### Tier 2 — extract model-agnostic fixes as standalone PRs onto `avarok/main`
High-value / low-risk / not Laguna-bound, so they land independent of the big feature.
Draft PRs except the crash fix.

| PR branch | commits | note |
|---|---|---|
| `fix/kv-prefix-exhaustion-crash` ★ | `d27ec6fd` | real CUDA-700 under prefix caching, all models; ships tests. **Ready PR.** |
| `fix/chat-thinking-correctness` | `19a2704e f3bb1167 353e690c 3ba80752 3feb225c 3a1c1c78` | `</think>` leak, reasoning parser, EOS-in-thinking, tool-markup, enable_thinking bool |
| `feat/jinja-converter` | converter halves of `fa2fc21d d50e9f6d` | `{% generation %}` strip + `{% include %}` resolution (any model) |
| `fix/fp8-kv-calibration` | `ea73fbb9 40a63114` | freeze-on-first-observe + scale detection |
| `fix/gen-budget-refund` | `dcaf577b` | suppressed-EOS budget refund |
| `feat/ignore-eos-parity` / `fix/tool-gen-caps` | `f69fd414 c1ad0437 4deef7d8` | vLLM-parity |
| `fix/cuda13-capture` / `ci/*` | `616ef7a6 1482f7c3 67c9eaf3 ce2c3469 19ca7aa7` | CUDA-13 link + CI/clippy/rustdoc |

**Caveat verified per-branch:** several of these touch files the Laguna stack also
restructured (e.g. `prefill_b/prefix_lookup.rs`, `SequenceState`), so each must be
test-cherry-picked onto `avarok/main`; any that don't apply cleanly are NOT truly
independent and move onto the Tier-3 stack instead.

### Tier 3 — Laguna feature as a dependency-ordered stack (HOLD until Tier 1/2 merge)
Built as fresh topic branches off `avarok/main` by cherry-pick (not by untangling the
merge history):

```
avarok/main
└─ feat/laguna-s-core            (~9)  model support, BF16 shared experts, agentic behavior, kernel metadata
   └─ perf/laguna-prefill-moe    (~28) co-dispatch, ragged attn, unified MoE, CUTLASS grouped GEMM,
      │                                sliding-window, RoPE table, flashinfer ragged, cuBLASLt projections
      └─ perf/laguna-decode-concurrency (~8) grouped read-once MoE, batched decode GEMV, graph-safe MoE, small-M FP4
         ├─ feat/laguna-gguf     (~20) needs #334 — keep-packed Q4_K/Q6_K, device-grouped MoE, NFS prefetch
         ├─ feat/laguna-lora     (~3)  needs #335 — family allow-list + expert gate/up fold
         └─ feat/laguna-xs       (~3)  needs jinja-converter — XS variant + thinking/bundled-template gating
            └─ tune/laguna-sampling (~4) presets, prose budget, context-length output
```

`feat/laguna-gguf` and `feat/laguna-lora` fork off decode and review in parallel once
#334/#335 merge. If the 28-commit perf cluster draws review resistance, split it
(CUTLASS-grouped-MoE / attention+RoPE / cuBLASLt-projections) — mirrors the earlier
"CUTLASS split into #227" precedent.

## Mechanics
- Rebuild clean topic branches off `avarok/main` by **cherry-pick**, not `rebase -i` of
  the merge-laden 127-commit history.
- Convention: `fix/<topic>-on-main` (standalone) / `feat/laguna-*` (stack) /
  `port/*-avarok` (#334/#335). Head branches pushed to `origin`; PRs opened
  cross-fork against `avarok/main`.
- Per branch: build + `cargo fmt` + clippy-1.93 + ≤500-LoC-per-file (Avarok CI enforces).
- Never force-push shared branches / never push to `main`.

## Sequencing
1. **Now:** rebase #332/#334/#335 (owner); open `fix/kv-prefix-exhaustion-crash`; open
   the rest of Tier 2 as **drafts**.
2. **After #334/#335 merge:** the Tier-3 Laguna stack, bottom-up.

## Risks
- The perf cluster is the heaviest review (28 kernel commits) — pre-split if needed.
- Cherry-pick + squash loses bisectable iteration history (accepted for reviewability).
- Tier-3 GGUF/LoRA leaves must rebase onto the FINAL merged form of #334/#335.
