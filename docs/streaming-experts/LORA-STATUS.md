# Atlas LoRA — status report (2026-07-06)

Where the LoRA effort stands on `feat/streaming-experts-mvp` (PR #9): what's built,
what's validated live on GB10, the concurrency benchmark, the known cut lines, and
the ranked roadmap.

## TL;DR

- **Serving a fine-tuned adapter, multi-adapter, runtime swap, and a fused
  per-request routing kernel are all built and compile clean.** Two demo adapters
  were trained on GB10 (STARFALL-4728/"Sparky", MOONVEIL-3390/"Vega") and pushed to
  HF (`MonumentalSystems/Holo-3.1-0.8B-lora-demo`, `-demo-2`).
- **Per-request routing works in *decode* under `--scheduling-policy slai`** — two
  concurrent requests naming different adapters each route to their own adapter; the
  prefill-consistent request is byte-clean.
- **Routing is free**: at C=2/4/8 concurrent, per-request LoRA routing carries the
  same req/s, prefill, and decode tok/s as the base model (within noise).
- **The remaining work is correctness *sequencing*, not the kernel**: request-scoped
  routing (esp. prefill + single-seq), adapter-correct KV, then the bgmv as the
  batch>1 optimization. See the roadmap.

## What's built (commits)

| piece | commit | state |
|---|---|---|
| M0+M1 single adapter (runtime BF16 delta @ attn k/v/o) | upstream `research/lora-mvp` (merged `fdb4e30`) | ✅ shipped, offline parity 0.0 rel-err |
| Multi-adapter pack into the frozen `[max_loras]` pool + global rotation (`/v1/lora/active`) over RDMA | `489f372` | ✅ build+clippy clean, 36 unit tests |
| Pool-size-1 dynamic swap (`/v1/lora/load`, disk swap-into-slot) | `4d6dedf` | ✅ demoed live (swap starfall↔vega in one slot) |
| Fused two-kernel bgmv per-request routing (`lora_bgmv.cu`) | `d3ea611` (WIP) + `e40a70f` (status) | ⚠️ decode routing works under SLAI; **disabled in prod** pending request-scoped routing |

## Validated live (holo-3.1-0.8b, GB10)

- **Pool=1 dynamic swap**: `starfall` → "…STARFALL-4710"; `POST /v1/lora/load vega` →
  same prompt "…MOONVEIL-3390"; swap back → "…STARFALL-4710". Runtime event logged
  (`LoRA disk swap: 'vega' packed into slot 0`).
- **Concurrent routing (SLAI)**: `adapter=starfall` → `STARFALL-7725` (byte-clean,
  identical to single-seq); `adapter=vega` → routed to Vega's persona. Under the
  default `fifo` the requests serialized and fell to the active adapter — SLAI is
  required for them to co-decode.
- **bf16 head vs nvfp4 head**: adapter #1's exact codeword digits are sensitive to
  lm-head precision (NVFP4 → 4710, bf16 → rambles); adapter #2 (overfit harder,
  loss 0.0008) is robust to both. Exact-digit fidelity would need the **base** in
  BF16, not just the head.

## Concurrency benchmark (routed vs base, SLAI, 64-tok gens)

| C | mode | req/s | prefill TTFT mean | agg decode |
|---|---|---|---|---|
| 2 | routed | 1.74 | 719 ms | 88 tok/s |
| 2 | base | 1.61 | 704 ms | 95 tok/s |
| 4 | routed | 1.99 | 1178 ms | 116 tok/s |
| 4 | base | 1.77 | 1427 ms | 104 tok/s |
| 8 | routed | 2.44 | 1894 ms | 134 tok/s |
| 8 | base | 2.05 | 2420 ms | 121 tok/s |

Per-request routing adds **no measurable overhead** vs base (routed even marginally
ahead — within shared-GPU noise). req/s scales with C; prefill grows from queuing.
The ceiling is scheduling/batching + request-scoped correctness, not the LoRA math.

## Known cut lines (why the WIP bgmv stays disabled)

1. **Single-seq (n==1) requests don't route** — a lone request naming an adapter
   returns the *global active* adapter.
2. **Prefill always uses the active adapter** — a routed request's prompt KV is
   active-flavored (persona survives in decode, exact recall degrades).
3. **Routes gated on active-adapter coverage** — heterogeneous adapters (different
   target modules/layers) mis-route on the modules the active adapter lacks.
4. **`pack_store_into_slot` drops `_slot_ptrs`** — a reused slot can keep a stale
   pointer-table/scale/mask entry.
5. **seq_slot metadata** at `meta_base+128` is safe only for padded_n ≤ 32.

## Roadmap (ranked; reviewer + on-HW findings converge)

Batching is the ceiling and the global-active-adapter is the wart. BGMV is the
*last* step, gated on batch>1 — not the near-term work.

1. **#23 Request-scoped AdapterSelector** — kill the global active adapter; route
   prefill *and* single-seq by the request's adapter. Highest value; fixes cut
   lines 1+2. *(next)*
2. **#24 Adapter-correct KV** — cache keyed by stable `adapter_id`, base reuses base
   blocks; fixes the warm-prefix-skips-delta cut line.
3. **#25 Slot generation + ref_count** — safety substrate for swap+graphs.
4. **#27 Demand-driven RDMA promotion** — `ensure_adapter_hot(id)` on miss + load
   coalescing (adapters are tiny; 1000+ is I/O-trivial on the CX7 tier).
5. **#28 Generation-keyed graphs** — retire `lora_rotatable` forced-eager.
6. **#26 Fix `pack_store_into_slot` stale table** — cheap hardening.
7. **#29 / bgmv** — wire the fused kernel into the reliable batch>1 path (last).
