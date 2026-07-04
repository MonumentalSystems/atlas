# Atlas LoRA Adapter Support — Final MVP Framework Proposal

*Synthesis of the design panel (2026-07-04, branch `wip/decode-concurrency`). Skeleton: the [simplest] proposal, which two of three judges scored best and whose anchors all verified in-repo. Grafted: the [risk] proposal's verified hazard findings and validation gates, and the [perf] proposal's frozen data-layout/kernel-contract discipline. Contradictions between the panel's proposals are resolved explicitly in §"Resolved contradictions". Path corrections vs the original brief carry forward: the server crate is `spark-server`, `ForwardContext` is at `crates/spark-model/src/layer.rs:206`, MoE kernels live in `kernels/gb10/common/`, and `session_hash` in `inference_types.rs` is at :63/:140.*

## Executive summary

Ship **one adapter, loaded at startup, applied to every request, computed as a runtime BF16 delta** (`y += scale·(x@Aᵀ)@Bᵀ`). This v0 needs **zero new CUDA kernels, zero scheduler changes, and zero graph-machinery changes**, because a startup-static adapter's A/B pointers and persistent-arena scratch are exactly as immutable as base weights — so the delta GEMMs are simply *captured inside* the existing CUDA decode graphs. That converts the codebase's hardest constraint (no `cuGraphExecUpdate`, pointer-baking replay) into a no-op and yields a served fine-tuned Qwen3 in under a week.

Runtime delta (never merge) is mandatory, not stylistic: base weights are NVFP4/FP8 behind pointer-keyed `OnceLock` prep caches (`crates/spark-model/src/layers/ops/dispatch_proj.rs:79,184,297`) that an in-place merge would poison — the [risk] panel's verified finding, adopted here as the documented justification.

While v0 stays single-adapter, we **freeze the three contracts that are expensive to change later** ([perf]'s core contribution): (1) A/B stored rank-padded to `max_lora_rank` in one fixed-address `[max_loras, …]` pool; (2) a persistent per-slot adapter-index device buffer with a `-1` sentinel, populated even when it holds one value; (3) the future BGMV kernel's argument shape (pointer tables + slot buffer, never adapter identity as a kernel arg). Multi-adapter then becomes data population plus one kernel port from `moe_expert_gemv.cu`, not a re-architecture.

Everything unsupported is **hard-rejected at load with a named reason — never silently skipped**, bidirectionally: unmatched `target_modules` fail, and unconsumed adapter tensors fail equally.

## Resolved contradictions

| # | Conflict | Decision | Why |
|---|---|---|---|
| 1 | **v0 graph strategy:** capture deltas inside existing graphs ([simplest]/[perf]) vs eager-only via global `suppress_graphs` ([risk]) | **Capture in graphs.** Plus a permanent `ATLAS_LORA_EAGER=1` escape hatch and a graph-vs-eager parity test in CI (grafted from [risk]). | With a startup-static adapter, pointer stability is identical to base weights — judges confirmed the claim is technically sound. [risk]'s per-batch toggling of the global `suppress_graphs` AtomicBool is an *unverified* capability of that flag and pays an unnecessary ~2× global decode tax. |
| 2 | **Indirection timing:** kernels up front ([perf]) vs deferred ([simplest]) | **Freeze layout + contract in v0; write kernels in M2.** | All three judges converged: the layout is what's expensive to change; the kernel is a bounded port whose spec ([perf]'s 1:1 `moe_expert_gemv` mapping) is adopted verbatim as the M2 spec. |
| 3 | **FP16 adapters:** extend `WeightDtype` whitelist, "convert on `copy_h2d`" ([perf]) vs host-side convert ([simplest]/[risk]) | **Host-side F16→BF16 conversion at load.** | `copy_h2d` is a raw memcpy — [perf]'s mechanism doesn't exist (judge-verified). Adapter tensors are tiny; host conversion keeps F16 out of every kernel path and the global whitelist untouched. |
| 4 | **SGMV prefill kernel:** write up front ([perf]) vs never ([simplest]) | **Don't write it.** Document `moe_bf16_grouped_gemm.cu:86` as the ready template. | Verified: Atlas prefills one sequence's chunk per pass, so prefill never mixes adapters; plain `dense_gemm` suffices indefinitely. |
| 5 | **Per-request `model`-field routing in M0** ([perf]) vs deferred to M2 ([simplest]) | **Deferred to M2, with indirection.** | Routing before indirection bakes a host-side `lora.is_some()` branch into captured per-slot graphs — replayed wrongly for requests with different adapter state (judge-identified correctness gap). |
| 6 | **Cache-key contamination fix:** fold adapter id into prefix/Marconi keys ([simplest] M2) vs mix `adapter_slot` into `session_hash` at request build ([risk]) | **Mix into `session_hash` at request build.** | One line, no cache-internal surgery; equivalent isolation. (v0 needs neither — keys are uniform by construction with one always-on adapter.) |

## Scope table

| Area | v0 (M0–M1) | v1 (M2) | Deferred (M3+) |
|---|---|---|---|
| Adapters | 1, startup-loaded, always on | Per-request on/off + multi-adapter slots | Dynamic `/v1/load_lora_adapter`, CPU LRU stage |
| Targets | `q/k/v/o_proj`; dense FFN `gate/up/down` (cuttable to attention-only) | — | `embed_tokens`/`lm_head`, MoE experts, MLA, GDN/SSM, `attn_gated` q_proj |
| Model families | Dense attention (Qwen3-style); per-`LayerType` **allow-list** — unknown layer types reject | — | GDN (needs exact-replay harness, cf. `gdn_exact_replay`), MLA, MoE |
| Base dtype | NVFP4/FP8 **and** BF16 (runtime delta covers both) | — | BF16 merge fast-path (probably never — forks correctness by dtype) |
| Adapter dtype | BF16/F32 native; F16 host-converted | — | quantized adapters, DoRA (rejected) |
| TP | TP=1 only; startup error otherwise; shard math documented | — | TP>1 enablement + 2-rank parity |
| Graphs | On, deltas captured; `ATLAS_LORA_EAGER=1` hatch | On, via device indirection | — |
| Kernels | Zero new | `lora_bgmv` (one port) | SGMV prefill (likely never needed) |
| Speculative | MTP/self-spec/ngram inherit deltas free (shared forwards); DFlash drafter unadapted → acceptance-rate only, documented; optional `--no-spec-with-lora` | — | Adapted external drafter |

## Architecture

**New types (name → owning location):**
- `PeftAdapterConfig { r, lora_alpha, target_modules, use_rslora }` + `fn scaling() -> f32` (`alpha/r`, or `alpha/√r` under rslora) — `crates/atlas-core/src/config/parsers/lora.rs`, mirroring `parse_quantization_config`. **Naming discipline:** all new code uses `adapter_*`/`peft_*`; never `lora_rank`, which collides with MLA's `kv_lora_rank`/`q_lora_rank` (`config.rs:182-207`).
- `LoraPair { a, b: DenseWeight, rank, scale }`, `LoraLayerWeights`, `LoraWeights` — `crates/spark-model/src/lora/mod.rs`. A/B land as plain BF16 `DenseWeight` via existing `dense()` helpers, but allocated **rank-padded to `max_lora_rank` inside a single fixed-address `[max_loras, …]` pool** (`SsmStatePool` template, `model/types.rs:87`) with per-module device **u64 pointer tables** (`build_ptr_table` pattern, `nemotron_moe.rs:414`) built at load, even though v0 fills one slot. NULL entry = base-only. This is the frozen v1 contract.
- `apply_lora_delta(gpu, buffers, x, y_out, pair, m, out_offset)` — `crates/spark-model/src/layers/ops/lora_delta.rs`: GEMV (m=1) or GEMM (m>1) shrink, expand into scratch, fold via `bf16_scaled_add` at `out_offset` (offsets handle fused gate|up segments).

**Load path** — verbatim clone of the DFlash second-checkpoint pattern: trait hook `load_lora_adapter(...) -> Result<Option<LoraWeights>>` beside `load_dflash_weights` (`crates/spark-model/src/weight_loader/mod.rs:240`), with a working generic default (strip `base_model.model.` → loader's `layer_prefix(i)` skeleton, honoring `weight_prefix` variance: `backbone`, `model.language_model`). Adapter dir via `model_resolver::resolve_model_dir` (`crates/spark-server/src/model_resolver.rs:15`); tensors via the existing `SafetensorsLoader` mmap+`copy_h2d`. **Bidirectional audit:** unmatched targets *and* unconsumed adapter tensors are both fatal. Adapter bytes (`max_loras × padded(A+B)`) enter the OOM preflight (`weight_loader/mod.rs:76-84`) **budgeted against KV-cache sizing**, not just total memory — GB10 unified-memory OOM is a system freeze.

**Reaching kernels:** `pub lora: Option<&'a LoraWeights>` on `ForwardContext` (`crates/spark-model/src/layer.rs:206`; precedent: `token_ids` field). Scratch `lora_xa [max_batch_tokens × max_rank]` and `lora_delta [max_batch_tokens × max_proj_out]` as persistent `BufferArena` allocations (`fp8_act` precedent, `crates/spark-runtime/src/buffers.rs:87-90,149`) — fixed addresses, hence graph-safe. No beta=1 GEMM epilogue exists, hence the scratch+`bf16_scaled_add` structure.

**Insertion points (all verified):**

| Path | Function | Location |
|---|---|---|
| Prefill Q/K/V | tail of `prefill_one_proj` | `crates/spark-model/src/layers/qwen3_attention/prefill/paged_qkv.rs:94` |
| Prefill O | `prefill_attention_paged_oproj` | `.../prefill/paged_oproj.rs:17` |
| Decode Q/K/V | GEMV block in `attention_forward` | `.../decode/attention_forward.rs:61-217` |
| Decode O | `attention_forward_oproj` | `.../decode/attention_forward_oproj.rs` |
| Dense FFN | fused gate+up / down sites | `crates/spark-model/src/layers/dense_ffn.rs` |

## Serving semantics & API

**v0: the server *is* the fine-tuned model.** One adapter, applied to all requests: no `adapter_id` plumbing, no mixed batches, no cache contamination (keys uniform by construction), no swap-in/out state. CLI: `--lora-adapter <name>=<path-or-hf-id>` (single occurrence) + `--max-lora-rank`, mirroring `--dflash-model` (`crates/spark-server/src/cli/serve_args.rs:161ff`). `request.model` accepts base name or adapter name; `/v1/models` advertises the **adapter name** as the served model. Known wart (accepted for v0, documented): requests naming the base model still get adapted output — logged with a per-request warning.

**M2 per-request routing (designed now, built then):** `adapter_id` threads exactly like `session_hash` (`crates/spark-server/src/api/inference_types.rs:63,140` → `PrefillInProgress`/`ActiveSeq`/`SwappedSeq` in `scheduler/types.rs` → `SequenceState`, `crates/spark-model/src/traits.rs:110`) — no Model-trait signature change; draft and verify forwards read the same field, so speculative modes inherit it. Cache isolation via one-line `session_hash = mix(session_hash, adapter_slot)` at request build. Because `step_decode_only` sorts by pool slot (`crates/spark-server/src/scheduler/decode_step.rs:31-33`) — an invariant LoRA must never re-sort — dispatch is a device-side per-slot adapter-index buffer (`-1` = base).

## CUDA-graph & TP strategy

**v0: capture the delta, don't fight the graphs.** Graphs bake device pointers at capture; that only matters if pointers change. Startup-loaded A/B, the pointer tables, and arena scratch never do — identical status to base weights, asserted at capture. No `suppress_graphs`, no graph-key growth, no indirection needed yet. `ATLAS_LORA_EAGER=1` remains a permanent debugging hatch; CI runs a graph-vs-eager parity test from M1 onward.

**M2:** delta kernels switch to indirection — kernels *always launched*, reading the per-slot index buffer refreshed pre-replay (the proven `token_ids` upload pattern, `decode_a.rs:102-141`; no host sync inside capture) and no-op'ing on `-1`/NULL. Graph topology constant; key space unchanged; adapter switch = memcpy + pointer-table rewrite, never recapture.

**TP (M3, math frozen now):** column-parallel (q/k/v/gate/up): shard **B** on the output axis, replicate A. Row-parallel (o/down): shard **A** on the input axis, replicate B, fold delta **before** the existing `all_reduce_async` — zero extra collectives. LoRA ops unconditional per rank (identical NCCL op order, or deadlock). Reuse `shard_dense_bf16` (`tp_shard.rs:57`).

## Kernel plan

**v0 — zero new kernels (all verified present):** `dense_gemv_bf16` (`layers/ops/gemm_quant.rs:110`) for decode shrink/expand; `dense_gemm_tc`/`dense_gemm` (`layers/ops/gemm_dense.rs:26,90`) or cuBLASLt `bf16_gemm_act_weight_t` for prefill; merge epilogue `bf16_scaled_add` (`kernels/gb10/common/residual_add.cu:60`, literally `output += scale·src`). Precedent that on-GPU low-rank composition needs nothing new: the Mistral MLA loader.

**M2 — one kernel, spec adopted verbatim from [perf]:** `lora_bgmv`, a port of `moe_expert_gemv.cu:57` with a 1:1 argument mapping — `expert_indices` → `per_token_slot`, `packed_ptrs` → A-table then B-table, NULL → base-only row no-op; two passes (shrink K=hidden→r, expand K=r→hidden), expand folding with per-slot scale. **SGMV prefill: not built.** Atlas prefills one sequence's chunk per pass, so prefill never mixes adapters; `moe_bf16_grouped_gemm.cu:86` (`expert_offsets`/`sorted_token_ids`/B-ptr table) stands as the 1:1 template if batched multi-sequence prefill ever lands. Porting refs: punica, vLLM Triton bgmv.

## Milestones

**M0 — Load, validate, account (~2 days, ~500-600 LOC).** CLI flag, PEFT config parser, generic key remap, host F16→BF16, rank-padded slot pool + pointer tables (1 slot filled), per-`LayerType`-allow-list/`attn_gated`/target validation with hard rejection, bidirectional tensor audit, VRAM preflight.
*Exit:* real PEFT adapter loads with per-module ranks/scales logged; **host-side offline parity test multiplies loaded A/B/scale against PEFT-exported reference deltas to ≤1e-2 rel-err** (catches remap/F16/rslora bugs before any kernel wiring); every §Scope rejection case errors with a named reason; key-remap unit suite covers `backbone`/nested prefixes.

**M1 — Runtime delta on all requests, graphs intact (~4-6 days, ~700 LOC; the reference harness + fused-FFN offsets historically bleed — budget the top of the range).** `ForwardContext.lora`, arena scratch, `apply_lora_delta` at the 5 insertion points; FFN segment offsets last (cuttable).
*Exit:* greedy logits match HF `transformers`+PEFT within BF16 tolerance on a golden prompt set, including with MTP/self-spec enabled; CUDA-graph decode still captures/replays; graph-vs-eager parity in CI; adapter decode overhead at r=16 documented (< ~5% expected); **base-only throughput regression < 1%** (adapter absent).

**M2 — Per-request routing + multi-adapter groundwork (~1.5 weeks, ~1,000 LOC + 1 kernel).** `adapter_id` threading, `session_hash` mixing, per-slot index buffer + pre-replay refresh, `lora_bgmv`, `model`-field routing, `/v1/models` enumeration, swap-out/in round trip.
*Exit:* mixed base+adapter batch under one graph replay is per-row correct, with **NULL-slot rows bit-identical to base**; adapter-after-swap parity; no recapture on adapter toggle; graph-vs-eager parity holds.

**M3+ (unscheduled):** TP enablement (TP-2 ≡ TP-1 parity, NCCL soak), multi-slot population >1, load/unload endpoints + CPU LRU, GDN/gated-q targets behind an exact-replay harness.

## Risk register

| Risk | Mitigation |
|---|---|
| Wrong scaling (`alpha/r` vs rslora) — silent quality loss | Scale read per adapter, never defaulted; M0 offline parity + M1 golden-logit gates |
| Graph capture breaks on delta ops | Only pointer stability matters; A/B/tables/scratch are load-time-fixed, asserted at capture; `ATLAS_LORA_EAGER=1` hatch + CI parity |
| Merge poisons quantized bases | Banned by design: `OnceLock` prep caches keyed by weight pointer (`dispatch_proj.rs:79,184,297`) assume immutable weights — runtime delta only |
| Fused-layout mis-slicing (gate\|up, `attn_gated` Q+gate) | Offset-write unit tests per layout; `attn_gated` models hard-rejected in v0 |
| Silent module drop by lenient mapper | Bidirectional fatal audit (unmatched targets ⇔ unconsumed tensors) |
| GDN recurrence corruption | GDN targets rejected until an exact-replay parity harness exists — `gdn_exact_replay` (`layer.rs:231`) shows how unforgiving this path is |
| GB10 unified-memory OOM = freeze | Adapter bytes in preflight, budgeted against KV cache, before any alloc |
| Cache cross-contamination (M2) | `adapter_slot` mixed into `session_hash` at request build; swap round-trip test |
| Batch re-sort breaking SSM/graph contiguity (M2) | LoRA never re-sorts; per-row device index aligned to the existing slot-sorted batch |
| Drafter/adapter mismatch (DFlash) | Correctness unaffected (verify uses adapted weights); acceptance-rate impact documented; `--no-spec-with-lora` opt-out |
| NCCL divergence under TP (M3) | Delta folded before `all_reduce_async`; LoRA ops unconditional per rank |

## Open questions for the maintainer

1. **v0 base-name semantics:** is warn-and-serve-adapted acceptable for requests naming the base model, or should they 400 until M2 routing lands?
2. **`max_lora_rank` / `max_loras` defaults** (proposed 64 / 8): what KV-cache headroom is acceptable to reserve on GB10 unified memory?
3. **FFN targets in M1:** ship attention-only first if fused gate|up offsets slip, or hold M1 for full 7-projection coverage?
4. **Is `--no-spec-with-lora` wanted**, or is a documented acceptance-rate note on DFlash sufficient?
5. **BF16-merge fast path:** permanently rejected (correctness forks by base dtype), or kept as an opt-in benchmark-only mode?
6. **M2 `lora_bgmv` review ownership:** who owns sign-off on the `moe_expert_gemv` port, given the MoE kernels' existing invariants?
