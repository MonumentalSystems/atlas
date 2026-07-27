# Atlas /metrics — vLLM compatibility gap analysis + plan (RESEARCH ONLY, nothing implemented)

Reference: vLLM main @ `0934b267` (2026-07-26), latest tag v0.26.0. **V0 metrics are gone**
(`vllm/engine/metrics.py` 404s from v0.12.0); everything below is V1.

## What Atlas exports today (10 registry + 7 hand-rolled)

Implementation: `prometheus` crate v0.14 (`spark-server/Cargo.toml:77`), lazy_static globals
(`metrics.rs:11-69`), route `GET /metrics` → `api::metrics_handler` (`serve_router.rs:109`),
handler = `prometheus::gather()` + TextEncoder then **hand-appended text**
(`api/misc_handlers.rs:59-128`).

- Registry: `atlas_requests_total`, `atlas_requests_active`,
  `atlas_time_to_first_token_seconds` (hist), `atlas_generation_tokens_total`,
  `atlas_prompt_tokens_total`, `atlas_loop_detector_verdicts_total{verdict,channel,spinning}`,
  `atlas_spec_decode_verify_total{k,outcome}`, `atlas_tool_calls_total`
- Hand-rolled text: `atlas_prefix_cache_{hits,misses,hit_tokens}_total`,
  `atlas_prefix_cache_hit_rate`, `atlas_token_entropy_last`,
  `atlas_low_entropy_tokens_total`, `atlas_low_entropy_ratio`

Histograms ARE supported. **No metric has any label** (no `model_name`). No `process_*`
(`default-features=false`).

## Bugs found while inventorying (independent of vLLM parity)

1. **`/v1/completions` is entirely uninstrumented** — `completions.rs:355` submits straight to
   `request_tx`, never touching request/token counters or TTFT. (`/v1/messages` is fine.)
2. **Our TTFT is not vLLM's TTFT** — it measures `decode_start − request_start` where
   `request_start` is stamped at *prefill start* (`prefill_a_step.rs:120`), so **queue wait is
   excluded**. vLLM measures from frontend arrival.
3. Spec-decode counters cover only K=2 + DFlash; `verify_k3/k4/mtp_step` emit nothing.
4. Hand-rolled text appended after `encode()` → a future registry metric of the same name emits
   a duplicate `# TYPE` block silently.

## KEY vLLM V1 corrections (things that no longer exist)

| Assumed | Reality in V1 |
|---|---|
| `gpu_cache_usage_perc` | **renamed** → `vllm:kv_cache_usage_perc` |
| `cpu_cache_usage_perc` | **removed** |
| `num_requests_swapped` | **removed** (V1 dropped swapping entirely) |
| `gpu_prefix_cache_hit_rate` | **removed** → queries/hits counter pair, **denominated in TOKENS** |
| `time_per_output_token_seconds` | **removed** → `vllm:inter_token_latency_seconds` + `vllm:request_time_per_output_token_seconds` |

Universal labels in V1: `["model_name","engine"]`. Counters must be spelled with a literal
`_total` in Rust (the Python client appends it automatically; the Rust crate does not).
`vllm:` names are legal in the Rust crate (`:` passes `is_valid_metric_name`).

## ★ The headline caveat — KV usage will read ~100% on a warm server

Atlas's prefix cache holds a refcount on every cached block, so `num_free_blocks()` sits near
zero on any warm server — Atlas's own scheduler comment measured exactly this
(`scheduler/mod.rs:390-397`). A naive `vllm:kv_cache_usage_perc = 1 − free/total` pins at ~1.0
and **every inherited vLLM dashboard alert fires continuously**.

Recommendation: emit `vllm:kv_cache_usage_perc` as the **effective** ratio
`1 − (free + reclaimable)/total` (semantically closest to vLLM's "how close am I to being
unable to admit work"), and expose the raw ratio only as `atlas_kv_cache_usage_ratio`.

⚠️ Blocking prerequisite: `RadixTree::stats()` **fakes** the block count — it returns
`(entries, entries)` (`radix_tree.rs:288`). A real reclaimable-block accessor is needed
(sum `RadixNode.block_idx` + `partial_suffix` over non-root nodes, `inner.rs:11-38`), and the
exact evictable-vs-pinned split should be confirmed against `inner.rs::evict` first.

## Architectural constraint

`metrics_handler` is a bare `async fn()` (`misc_handlers.rs:59`); `AppState` holds **no model
handle**; the model is `Box<dyn Model>` owned by the scheduler thread which requires
`bind_gpu_to_thread`. **Every scheduler/KV metric must be PUSHED from the scheduler loop into
global gauges** (like the prefix-cache atomics already are), not pulled at scrape time. Natural
push point: top of the main loop, `scheduler/mod.rs:311`.

## Naming: dual-emit (recommended)

Emit both `atlas_*` and `vllm:*`, behind `--metrics-vllm-compat` (default on). Rationale:
cardinality cost is ~30-60 extra series (all unlabeled or single-series), renaming everything to
`vllm:` is misleading and strands existing dashboards, and keeping only `atlas_*` fails the
drop-in goal. Ship `atlas_build_info{engine="atlas"}` so the real engine is discoverable. One
`emit()` helper per metric writing both handles so they cannot drift. Only alias metrics whose
semantics we can honour.

## Phases

- **Phase 0 (blocking, small):** add `Model::num_kv_blocks()`; add a real
  `cached_block_count()`; add a scheduler→global gauge push module; `OnceLock<String>` model name.
- **Phase 1 (asked for, highest value):** `vllm:kv_cache_usage_perc` (effective),
  `atlas_kv_cache_{blocks_total,blocks_free,blocks_reclaimable,usage_ratio,usage_effective_ratio}`.
- **Phase 2:** `num_requests_running` (`active.len()`), `num_requests_waiting` (pending +
  `prefilling.len()`), preemptions.
  ⚠️ **Divergence:** vLLM preemption is recompute-and-retry (request survives); Atlas's
  decode-time preempt **kills the request** (`decode_step.rs:118` `send_error`). Count only the
  swap-out path as `vllm:num_preemptions_total`; give the decode kill its own
  `atlas_requests_killed_kv_exhaustion_total` — it deserves a page, not a graph line.
- **Phase 3 (medium):** per-request histograms from `lifecycle.rs::finish_sequence` (most values
  already computed). Then add `arrival: Instant` to `InferenceRequest` (~8 files) to unlock
  queue time, true e2e latency, and a corrected TTFT. Copy vLLM bucket lists verbatim; keep the
  existing `atlas_time_to_first_token_seconds` buckets so current dashboards survive.
  Note `finished_reason="tool_calls"` is Atlas-only — pass through, don't remap to `stop`.
- **Phase 4:** swap/offload under `atlas_swap_*` / `atlas_hss_*` ONLY — do not synthesise
  `vllm:num_requests_swapped`; it no longer exists upstream and our two mechanisms (classic
  whole-sequence `--swap-space-gb` vs `--high-speed-swap` sliding block window) don't collapse
  into one gauge.
- **Phase 5:** fix `/v1/completions`; move hand-rolled text into the registry; token-denominated
  prefix-cache queries/hits; `iteration_tokens_total`; cache_config_info / lora_requests_info;
  rework spec-decode into vLLM's draft/accepted shape.

Not recommended (no Atlas analogue, conditionally registered upstream so absence is harmless):
`external_prefix_cache_*`, `mm_cache_*`, `nixl_*`, `kv_offload_*`, `engine_sleep_state`,
`corrupted_requests_total`, `prompt_tokens_by_source_total`.

## Unverified / confirm before implementing

- The exact evictable-vs-pinned split behind `reclaim_prefix_blocks` (drives the Phase 1 formula).
- Whether `AppState.model_name` matches the `model` string clients send.
- Whether multi-node EP/TP worker ranks also serve `/metrics` (would need an `engine`-like label
  to avoid series collision).
