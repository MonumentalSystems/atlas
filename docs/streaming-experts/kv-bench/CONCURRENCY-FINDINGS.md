# KV overflow tier — concurrency / agentic-load A/B (2026-07-06)

> **Scale addendum (35B-A3B) is at the bottom** — the big-model granule where RDMA's
> bandwidth finally beats NVMe. The small-model (0.8b) study below is the worst-case floor.


First A/B of the KV backends under **concurrent** load: normal HBM KV vs
`--high-speed-swap` overflow to {local NVMe, RDMA peer}. Ran on **dgx-00** (GB10)
↔ **gx10** kv-peer over a single ConnectX-7 rail (the proven live-serving bounce
path; zero-copy/dual-rail are not yet in the serving path — see `RDMA-KV-TIER.md` §6).

**Vehicle:** `holo-3.1-0.8b`. Deliberately small — model compute is trivial, so the
KV-tier restore latency is **worst-case exposed** (no compute to hide it). A big
model shows a *smaller* relative penalty. KV overflow is forced with a ~2500-token
prompt (a tagged secret `TANGERINE-7742` at the top → the model must attend across
the whole, RDMA-resident context) and a small HBM cap. Greedy (temp 0), 48-tok gens.
Harness: `bench_client.py` + `kv_sweep.sh` in this dir.

## Headline

- **Normal HBM KV scales cleanly** to C=8 (274 tok/s aggregate, correct recall).
- **The KV-overflow tier (RDMA *and* NVMe) is single-sequence-only today.** Correct
  at **C=1**; recall **degrades at C=2**; **hard-fails at C≥4**.
- The failure is **upstream of the backend** — NVMe and RDMA fail *identically* — so
  it is **not** an RDMA problem and **not** a budget knob (raising `--high-speed-swap-gb`
  8→48 did not change it).

## Root cause (not RDMA)

The disk-block-id allocator is capped at **`max_blocks_per_layer` — one sequence's
worth** of blocks (`crates/spark-storage/src/high_speed_swap.rs:239`, comment:
"Capacity == max_blocks_per_layer"; error raised at
`crates/spark-model/src/model/block_mgmt.rs:298`). Concurrent sequences share that
single-sequence id namespace:
- **C=2**: ids collide → the restored KV is mis-mapped → **wrong recall**
  (`"vault_access_code"` / `"vault-123456"` instead of `TANGERINE-7742`).
- **C≥4**: namespace exhausts → `disk-block-id pool exhausted` → NULL logits →
  **empty output** for every stream.

Enabling concurrency is a real feature (size the id pool **and** the backend arena
for `max_batch_size` sequences, then validate concurrent restore correctness) — not
a benchmark-time tweak.

## Measured — C=1 (the regime that works)

| KV backend | HBM cap | TTFT (prefill) | decode tok/s | recall |
|---|---|---|---|---|
| **normal HBM** | full | **297 ms** | **184.5** | ✅ |
| RDMA peer, partial (cap=64) | 1024 tok | 587 ms | 38.7 | ✅ |
| RDMA peer, full (cap=8) | 128 tok | 894 ms | 40.2 | ✅ |
| local NVMe, full (cap=8) | 128 tok | 977 ms | 39.2 | ✅ |

**Insights:**
- Overflowing KV to *any* tier costs **~4.7× decode** vs HBM at C=1 (184→~40 tok/s)
  **for this tiny model** — the worst case; a big model hides most of it behind compute.
- **RDMA ≈ NVMe here.** At 0.8b the KV granule is ~16 KiB and the tier is
  *latency-bound*, so RDMA's 24 GB/s bandwidth advantage doesn't show. RDMA's win
  needs **large KV granules** (big model) — cf. the 35B-A3B result in `RDMA-KV-TIER.md`
  (57% of local-KV, bandwidth-bound). This benchmark is the small-model floor.
- Partial overflow (cap=64) beats full (cap=8) on **prefill** (587 vs 894 ms — more
  KV stays HBM) at equal decode.

## Measured — concurrency (normal HBM only; overflow tiers fail)

| C | normal HBM: TTFT | decode/req | agg decode | req/s | overflow tiers |
|---|---|---|---|---|---|
| 1 | 297 ms | 184.5 | 184.5 | 2.0 | ✅ correct (table above) |
| 2 | 604 ms | 85.6 | 171 | 1.9 | ⚠️ wrong recall (id collision) |
| 4 | 1143 ms | 56.1 | 224 | 2.2 | ❌ empty (pool exhausted) |
| 8 | 2242 ms | 34.2 | 274 | 2.4 | ❌ empty (pool exhausted) |

## Follow-up to unlock agentic concurrency on the RDMA KV tier

1. Scope the disk-block-id namespace **per sequence** (or size it
   `max_blocks_per_layer × max_batch_size`) + grow the peer/NVMe arena to match.
2. Validate concurrent **restore correctness** (the C=2 mis-recall must be gone).
3. Re-run this sweep on a **big model** (35B-A3B) where RDMA's bandwidth beats NVMe,
   and land the zero-copy + dual-rail serving path (`RDMA-KV-TIER.md` §6/§7) so the
   21 GB/s restore reaches live inference.

---

# Scale addendum — 35B-A3B (2026-07-06)

Same A/B on **Holo-3.1-35B-A3B-NVFP4** (a Qwen3.5-style hybrid SSM+MoE **thinking**
model; 40 layers, 256 experts, ~3B active). This is the model size the tier is *for*.
Single ConnectX-7 rail, bounce path (no zero-copy/dual-rail yet). Timing parsed from
the server log (`Done: N tokens X tok/s TTFT=Yms`) since client `delta.content`
counting misses the `<think>` reasoning tokens. 2500-tok context (forces overflow),
greedy, ~99-tok gens. All rows recalled `TANGERINE-7742` correctly.

| config | HBM cap | TTFT | **decode tok/s** | % of HBM | recall |
|---|---|---|---|---|---|
| **normal HBM** | full | 5615 ms | **42.5** | 100% | ✅ |
| RDMA peer, full (cap=8) | 128 tok | 6322 ms | **19.5** | **46%** | ✅ |
| RDMA peer, partial (cap=64) | 1024 tok | 5392 ms | 16.9 | 40% | ✅ |
| local NVMe, full (cap=8) | 128 tok | 6173 ms | 16.1 | 38% | ✅ |

(norm concurrency: C=2 → 12.1 tok/s·req, C=4 → 6.6 tok/s·req — aggregate decode
plateaus ~26 tok/s; this hybrid 35B is memory-bound under batching. Overflow tiers
remain C=1-only per the single-seq block-id finding above — not re-measured here.)

## The two scale results

1. **Full KV offload costs ~2.2× decode at 35B (42.5 → 19.5 tok/s, 46% of HBM)** —
   vs **4.7×** (21%) on the 0.8b. The big model's per-token compute hides most of the
   KV-restore latency, exactly as predicted. **The tier is far more viable at real
   model scale.** (Comparable to the doc's earlier 5-turn A3B "57% of local-KV"; this
   is a harder single-turn full-offload at cap=8.)

2. **RDMA beats NVMe at the big granule: 19.5 vs 16.1 tok/s (1.21×)** at equal cap=8.
   On the 0.8b they were *tied* (~40 tok/s) because the ~16 KiB granule is latency-
   bound; at 35B the granule is large enough that RDMA's bandwidth wins. And this is
   only **single-rail bounce** — zero-copy + dual-rail (`RDMA-KV-TIER.md` §6/§7, the
   standalone path already hits 21 GB/s restore) should widen RDMA's lead further.

**Bottom line for scaling:** at production model size the RDMA KV tier delivers ~46%
of local-KV decode at C=1 while spilling ~all KV to a remote blade, and beats the SSD
tier by ~1.2× (more once zero-copy lands). The gating limitation is **concurrency**
(the single-seq disk-block-id namespace), not throughput — that's the fix to land next.

## Zero-copy + dual-rail (the bandwidth-optimized serving path)

Re-tested the 35B full-offload (cap=8) with `ATLAS_KV_ZERO_COPY=1 ATLAS_KV_DUAL_RAIL=1`
(2-rail peer). Log confirms it engaged: `2 rail(s)`, `scratch pool UMA=true (zero-copy
restore enabled)`, `registered UMA landing region 256 MiB — zero-copy restore live`.
(Note: `RDMA-KV-TIER.md` §6.3's "zero-copy not in the serving path yet" is **stale** —
`high_speed_swap.rs:131` wires a UMA scratch pool with a safe fallback; GB10 is UMA so
it activates.)

| path | decode tok/s | TTFT | recall |
|---|---|---|---|
| single-rail bounce (rdmaB) | 19.5 | 6322 ms | ✅ |
| **zero-copy + dual-rail** | **19.7** | **5097 ms** | ✅ |

**Decode is unchanged (19.7 vs 19.5); TTFT improves (~5.1 vs 6.3 s).** Why: **live
decode restore is latency-bound, not bandwidth-bound** — each step restores only the
few KV blocks that step attends to (small transfers), so zero-copy/dual-rail (which
optimize *bandwidth*, and gave the 24/21 GB/s in the bulk standalone test) don't move
decode. They help the *bulk* offload during **prefill** — hence the better TTFT. This
also explains RDMA's modest 1.21× over NVMe: the per-step restore is latency-bound for
both. To speed the tier's **decode**, the lever is restore *latency* (fewer round-trips
/ deeper prefetch overlap), not more bandwidth.

**Caveat — shared-box contention:** midway through, a concurrent `vllm serve` of the
*same* 35B model appeared on dgx-00 (46 GB, ~92% GPU util). The sweep + the zero-copy
sample above ran 23:04–23:22 UTC while vLLM was still loading (compute free); a later
3-rep zero-copy run (23:23+) hit vLLM going compute-active and degraded to ~9 tok/s
(TTFT ~8.9 s) — **discarded as contaminated**. The clean numbers are mutually
consistent (two independent cap=8 RDMA samples at 19.5/19.7). A fully-clean multi-rep
re-run needs a quiet GPU; I did not kill the other tester's server.
