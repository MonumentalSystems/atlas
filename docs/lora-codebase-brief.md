# Atlas LoRA Adapter Support — Codebase Map & Design Brief

## 1. Load-path story (PEFT safetensors → device memory)

Atlas already has a clean, reusable pipeline for a *second* weight checkpoint. The DFlash drafter is the exact template: a fully separate `WeightStore` loaded from its own dir via the `load_dflash_weights` optional trait hook (`crates/spark-model/src/weight_loader/mod.rs:240-248`, `weight_loader/dflash_loader.rs`), returning `Ok(None)` by default so older loaders inherit a no-op. **An adapter loader should follow this verbatim as `load_lora_adapters`.**

- Adapter safetensors load unchanged through `SafetensorsLoader::new().load(adapter_dir, gpu, reserve)` (`crates/spark-runtime/src/weights/loader.rs:12-171`), which mmaps + `copy_h2d` each tensor (`load_fns.rs:106-124`) into `WeightStore = HashMap<String, WeightTensor{DevicePtr, shape, dtype}>` (`weights.rs:90-92`). PEFT `lora_A`/`lora_B` are plain BF16/FP32 tensors.
- **Blocker — FP16:** `WeightDtype::from_safetensors` (`weights.rs:56-69`) accepts BF16/F32/FP8/U8/I8/I64 but **rejects FP16**, the PEFT default. Either extend the whitelist or require/convert BF16 adapter checkpoints at load.
- **Name remapping:** store lookup is exact-match, hard-fail on miss (`weights.rs:109-113`). PEFT keys `base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight` must strip to the `{layer_prefix(i)}.{module}` skeleton Atlas loaders already build (`config/methods.rs:154`, `dflash_loader.rs:158-217`). Watch `weight_prefix` variance: `backbone` (Nemotron-H), auto-detected `model.language_model` for nested models (`serve.rs:221`).
- **adapter_config.json:** parse only `r`, `lora_alpha`, `target_modules`, `use_rslora` — reuse the sidecar-config pattern from `parse_quantization_config` (`serve.rs:65`, `config.rs:486-497`), adding a parser under `atlas-core/src/config/parsers/`.
- **A/B storage:** land as `DenseWeight` (BF16) via existing `dense()` helpers (`weight_map/model_a.rs:121,167`). No new struct kinds needed. Mistral's MLA loader already runs on-GPU `dense_gemm_bf16` low-rank composition at load (`mistral_loader.rs:38-57`) — proof the shrink/expand matmuls need no new kernels.

## 2. Compute-path story (where delta GEMMs hook in)

Deltas must be a **separate BF16 side-path** (`y += (alpha/r)·(x@Aᵀ)@Bᵀ`) — base weights are NVFP4/FP8 consumed by custom w4a16/w8a16 kernels, and pointer-keyed `OnceLock` prep caches (`layers/ops/dispatch_proj.rs:79,184,298`) assume immutable base weights, so in-place merge is infeasible for quantized bases.

**Narrow choke points already exist per projection:**
- **Prefill:** `prefill_one_proj` handles Q/K/V for all quant formats (`qwen3_attention/prefill/paged_qkv.rs:94`); `prefill_attention_paged_oproj` for o (`paged_oproj.rs:17`); `prefill_qkvz_proj` for SSM (`qwen3_ssm/trait_prefill_proj.rs:19`). ~4 tail-insertion sites cover most prefill.
- **Decode (M=1):** o_proj is one function (`attention_forward_oproj.rs`); Q/K/V GEMV block (`attention_forward.rs:61-217`); SSM qkvz/out_proj (`ssm_forward.rs:42-165, 392-435`); dense FFN (`dense_ffn.rs`).

**Kernels — what exists vs. what's needed:**
- *Exist, reusable now:* `dense_gemv_bf16` (M=1), `dense_gemm_tc`/`dense_gemm` (`layers/ops/gemm_dense.rs:26,90`), `cublaslt::bf16_gemm_act_weight_t` (prefill M>1), and critically `bf16_scaled_add` (`kernels/gb10/common/residual_add.cu:60`, `output += scale*src`) — the exact LoRA merge epilogue. A **single-adapter** path needs **zero new kernels**.
- *Must be written for multi-adapter:* a gathered/indexed LoRA kernel. Atlas already ships both required shapes as MoE precedents — **BGMV-analog** `moe_expert_gemv.cu:57` (device index array → device u64 pointer-table lookup, NULL=no-op) for decode, and **SGMV-analog** `moe_bf16_grouped_gemm.cu:86` (sorted tokens + `expert_offsets` + device B-ptr table) for prefill. `build_ptr_table` (`nemotron_moe.rs:414`) shows how to publish per-adapter A/B device-pointer tables. These map 1:1 onto per-token-adapter-index LoRA dispatch.
- *Gap:* no general grouped/batched bf16 library path; cuBLASLt is single-GEMM-per-call (costly heuristic for tiny r×h). Multi-adapter batched GEMM = hand-written kernel modeled on the MoE ones.

**Plumbing:** thread adapter state via `ForwardContext` (`layer.rs:206`) — it already carries the optional per-token `token_ids` device buffer (`layer.rs:237`), the exact precedent for an `Option<LoraPassState>` (adapter-table ptrs + per-token adapter indices). Add `lora_xa [max_batch_tokens, r_max]` and `lora_delta` scratch to `BufferArena`/`BufferSizes` following the `fp8_act` precedent (`buffers.rs:87-90`). No beta=1 GEMM epilogue exists, so the delta needs its own output scratch + `bf16_scaled_add`.

## 3. Serving story (request → adapter id → batch)

Single-model server: one `AppState.model_name`, one `request_tx`, one scheduler thread, one `Box<dyn Model>` (`app_state.rs:19-21`).
- **Free routing slot:** `ChatCompletionRequest.model` (`openai/chat_request.rs:11`) is parsed, logged, never routed on — an adapter name routes with zero client change. `/v1/models` (`completions.rs:429`) trivially extends to enumerate adapters.
- **Thread the id like `session_hash`:** add `adapter_id` to `InferenceRequest::{Blocking,Streaming}` (`inference_types.rs:54`) → `PrefillInProgress`→`ActiveSeq`→`SwappedSeq` (3 places, `scheduler/types.rs`) → `SequenceState.adapter_slot: Option<u32>` (`spark-model/src/traits.rs:107-110`). This reaches every forward (prefill/decode_batch/mixed/verify) **without any Model-trait signature change** — exact precedent: `disable_mtp`, `grammar_spec`, `session_hash`.
- **Batch dispatch must be per-row, not per-adapter-subgroup:** `step_decode_only` sorts the batch by SSM/KV pool slot (`decode_step.rs:31-33`) — a CUDA-graph/SSM-contiguity invariant that LoRA **cannot re-sort**. So adapter dispatch must be a device-side per-slot/per-token adapter-index gather (SGMV/BGMV-style), matching the slot-sorted contiguous batch (batch position == pool slot).
- **Resolver reuse:** `model_resolver::resolve_model_dir` (`model_resolver.rs:15`) resolves adapter HF ids/paths verbatim. CLI: repeatable `--lora-adapter name=path` + `--max-lora-rank`, mirroring the `--dflash`/`--draft-model` pair (`cli/serve_args.rs:160`).

## 4. Hard constraints (ranked)

1. **CUDA-graph decode replay.** Decode graphs bake weight/scratch/SSM-state DevicePtrs as kernel args at capture; n=1 graphs keyed by `slot_idx`, batched by padded-n bucket (`model/types.rs:72-86`, `decode_a.rs:231`, `decode_a2.rs:176`). **No `cuGraphExecUpdate` infra exists anywhere.** Adapter identity must flow through device-memory indirection refreshed pre-replay (the proven `token_ids`/metadata pattern, `decode_a.rs:102-141`) — never as kernel args, or graphs become adapter-specific and the key space explodes. Host sync inside capture fails (status 900). `suppress_graphs` (`types.rs:110`) is a global escape hatch but disables graphs for *all* sequences (~2x decode throughput loss).
2. **Quantized base weights.** No merge into NVFP4/FP8; runtime BF16 delta mandatory. Prep caches (`dispatch_proj.rs:79`) would be poisoned by in-place merge.
3. **TP sharding.** Column-parallel projections (q/k/v/gate/up/qkvz): shard LoRA **B** on output axis, replicate A. Row-parallel (o_proj/down/out_proj): shard LoRA **A** on input axis, replicate B, and fold delta in **before** the existing post-projection `all_reduce_async` (`decode_inner.rs:111`) — zero extra collectives. Reuse `shard_dense_bf16` + `TpAttentionDims::proj_shape` (`tp_shard.rs:57-117,242`). Multi-rank must issue identical LoRA ops in identical order or NCCL deadlocks.
4. **Gated/segmented projections.** `attn_gated` (Qwen3-Next/3.5/3.6, `config.rs:288`) makes q_proj output 2×q_dim interleaved Q+gate — a PEFT q_proj delta maps only to the Q half. GDN in_proj is segmented `[Q|K|V|Z]` (`tp_shard/gdn.rs:22-42`). Dense FFN uses fused dual gate+up GEMV. LoRA B must slice per-segment, matching base layout.
5. **SSM/GDN layers.** LinearAttention layers have no q/k/v_proj; targets are `in_proj_qkvz`/`in_proj_ba`/`out_proj`. Per-`LayerType` target validation required (`config.layer_type(i)`).
6. **MLA (DeepSeek).** Absorbed-weight attention — no materialized k_proj output. Targets would be `wq_a`/`wq_b`/`wkv_a`/`wo`. **Naming collision:** `kv_lora_rank`/`q_lora_rank`/`o_lora_rank` (`config.rs:182-207`) are MLA, not adapter LoRA — new code must use `adapter_*`/`peft_*`.
7. **MoE experts.** Fused pointer-table expert kernels (`moe/forward.rs:284-361`) have no per-expert GEMM call sites — per-expert LoRA is a large lift. Realistic scope: attention + dense-FFN + SSM projections (+ MoE router at most).
8. **Memory accounting.** No Drop/dealloc on weight structs; adapter VRAM must be added to OOM preflight (`weight_loader/mod.rs:76-84`) and budgeted against KV cache. On GB10 unified memory, GPU OOM = system freeze (watchdog).
9. **Prefix/SSM cache keying.** `session_hash`-keyed KV/Marconi reuse (`traits.rs:96-118`) cross-contaminates across adapters unless adapter id is folded into cache keys. Adapter must survive swap-out/swap-in round trip and apply consistently in draft AND verify (MTP/self-spec/ngram/dflash).

## 5. MVP shortcuts the prior art validates

vLLM/SGLang/TGI/S-LoRA/Punica all converge on: frozen base + preallocated fixed GPU slots padded to `max_lora_rank` + CPU LRU stage + per-token adapter-index metadata in persistent buffers (keeps graphs replayable). Validated simplifications:
- **Phase 0 — merge-into-weights for BF16 bases only:** `W' = W + (alpha/r)·B@A` at load, zero runtime cost, graph-safe. Atlas already runtime-quantizes BF16 fine-tunes via `quantize_to_nvfp4` (`weight_loader/mod.rs:111`) and Mistral does on-GPU low-rank composition — so merge *before* quantization is near-free. Invalid for quantized bases / fast switching.
- **Phase 1 — single active adapter, runtime delta:** a plain GEMM pair + `bf16_scaled_add` per target layer, no custom kernels, cuBLAS suffices at rank 8–64. Run LoRA-active sequences eager (`suppress_graphs`) first.
- **Decode-first:** ship BGMV-shaped decode delta; use naive segmented GEMM (or per-request merge) for prefill — vLLM shipped BGMV-only initially.
- **Design indirection from day one even if unused:** a persistent per-token→adapter-slot buffer with `-1` sentinel makes the jump to multi-adapter BGMV a kernel swap, not an architecture change. Rank-pad to fixed `max_lora_rank` so one `[slots, max_rank, hidden]` buffer serves heterogeneous adapters via memcpy.
- **CPU cache = plain HashMap of pinned tensors** with manual `/v1/load_lora_adapter`-shaped endpoints; LRU + async prefetch are later.
- **Kernel porting refs:** punica-ai/punica, vLLM Triton bgmv/sgmv, shreyansh26/multilora-llm-inference — no need to derive from scratch; Atlas's MoE kernels are already structurally identical.

## 6. Open questions a design must decide

1. **Scaling exactness:** `alpha/r` vs `alpha/sqrt(r)` (`use_rslora`) — silent quality degradation if wrong. Must read from adapter_config.json per adapter.
2. **FP16 adapters:** extend `WeightDtype` whitelist or force BF16 conversion at load?
3. **Merge (phase 0) vs runtime-delta (phase 1) boundary:** ship BF16-merge first, or go straight to runtime delta so quantized bases work day one?
4. **Slot pool sizing:** `max_loras`, `max_lora_rank`, `max_cpu_loras` chosen against KV-cache budget on unified GB10 memory; where does OOM preflight account for it?
5. **CUDA-graph strategy:** eager fallback for LoRA-active sequences (simple, ~2x cost) vs. per-`(slot, adapter_set)` graph keys vs. device-indirection with one shared graph. The `SsmStatePool` fixed-address design (`types.rs:87-93`) is the template for a replay-safe adapter slot pool.
6. **Target-module coverage & validation:** reject adapters whose `target_modules` include unsupported modules (embed_tokens/lm_head/MoE experts) rather than silently skipping (wrong output). Per-`LayerType` + `attn_gated` validation.
7. **Fused-projection A/B stacking:** how to slice PEFT's separate q/k/v (and gate/up) into Atlas's fused/interleaved layouts, and write expand-kernel outputs at correct sub-offsets.
8. **Cache-key contamination:** fold adapter id into `session_hash`/prefix/Marconi keys, and into `SwappedSeq`.
9. **Speculative consistency:** ensure deltas apply in draft AND verify forwards (MTP/self-spec/ngram/dflash).
10. **Prefill kernel path:** BGMV-only first, or invest in SGMV/segmented prefill up front given long shared segments?

**Recommended entry points to touch first:** `weight_loader/mod.rs` (`load_lora_adapters` hook), `layer.rs:206` (`ForwardContext` LoRA field), the ~4 prefill choke functions + decode GEMV blocks listed in §2, `buffers.rs` (scratch), `inference_types.rs`+`SequenceState` (id plumbing), `cli/serve_args.rs` + `completions.rs:428` (serving surface).
