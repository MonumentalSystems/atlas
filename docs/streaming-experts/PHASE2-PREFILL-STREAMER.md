# Phase 2 — Prefill expert streamer (design, staged for review)

> **Status: design + verified integration anchors. NOT landed.** Phase 2 edits
> the freshly-tuned prefill/decode hot path (PR #229) and must be validated
> bit-identically against the Posix oracle on real hardware before it merges —
> that is an operator-in-the-loop step, not an autonomous one. Phase 0 (Gate 0
> measurements) and Phase 1 (record format + offline builder) are landed and
> testable; this file is the blueprint for the next increment.

## What ships in Phase 2

A prefill-path expert residency cache: an over-core NVFP4 MoE checkpoint loads
and runs on a single GB10 by streaming cold experts from the Phase 1 store.
Experts live on NVMe in resident layout; **layer L+1's expert set is prefetched
into a pinned arena while layer L computes**; dispatch is redirected by patching
the contents of the device-resident `ExpertPtrTable`. Correctness is proven
bit-for-bit against `PosixBackend`.

## Prefetch is the whole ballgame — and doubly so over NFS

The bandwidth ratio measured in Gate 0(b) — streaming ceiling ~7 GB/s vs the
273 GB/s LPDDR compute bus, ~40× — means a *reactive* fetch (wait until layer L's
router picks experts, then fetch) stalls the GEMM every layer. Streaming only
wins because prefill reuses each fetched byte across a whole 8k-token chunk, and
because **the fetch is issued far enough ahead of the compute that consumes it.**
Prefetch is not an optimization here; it is the mechanism.

This gets sharper on our fleet, where model weights live on **`/tank` NFS**
(gx10 ext4 export over ConnectX-7; see the fleet's tank-nfs notes). Two
consequences drive the design:

1. **Stage the runtime store on local NVMe when you can.** Gate 0(b)'s ~7 GB/s
   O_DIRECT number is `nvme0n1p2` *local*. Served over NFS, sustained bandwidth
   drops and — the real killer — per-fetch *latency* (network RTT + server
   queueing) rises by 1–2 orders of magnitude vs a local O_DIRECT completion.
   The builder can run against an NFS checkpoint (it is offline and latency-
   tolerant), but the `.xpr` store the streamer reads at runtime should live on
   the local NVMe of the streaming box. `ExpertIndex` + per-layer files make
   staging a plain `rsync`.

2. **When the store *is* on NFS, prefetch depth must grow past double-buffering.**
   A 2-arena double buffer hides one layer's fetch under one layer's compute
   *only if* `t_fetch(layer) ≤ t_compute(layer)`. Over NFS, `t_fetch` rises and
   acquires a fixed RTT tail, so one layer of lookahead is not enough. Provision
   a **ring of `D+1` pinned arenas**, where

   ```
   D = ceil( (rtt + layer_bytes / sustained_bw) / t_compute_layer ) + 1   // +1 = latency slack
   ```

   and issue prefetches for layers `L+1 .. L+D` (bounded by the ring), each on
   the side `prefill_stream`, joined by per-arena events — never a stream sync.
   On local NVMe `D` collapses to 1 (the classic double buffer); on NFS it is
   whatever covers the RTT. The ring depth is a config knob, defaulted from a
   one-shot probe of the store's mount (local vs NFS) at engage time.

   Corollary: prefetch must also keep **enough requests in flight** to saturate
   bandwidth — the io_uring backend already runs SQPOLL at QD≥4 (Gate 0(b)
   showed QD=1→3.9 GB/s vs QD≥4→7 GB/s), and over NFS the in-flight count wants
   to be higher still to fill the bandwidth-delay product. Feed the backend a
   whole layer's expert `ReadRequest`s at once, not one at a time.

## Verified integration anchors (against `main` @ PR #229)

The recon that produced this plan checked every anchor against the working tree.
The engine to reuse is already present; Phase 2 is mostly wiring.

| Hook | Location | Note |
|---|---|---|
| Expert pointer table | `spark-model/.../moe/mod.rs:18-30` | `ExpertPtrTable { packed_ptrs, scale_ptrs, scale2_vals }` — 3 device arrays |
| Table build | `moe/mod.rs:399-435`, called `moe/init.rs:39-41` | patch = `copy_h2d` of 8 B at `packed_ptrs + e*8` (+ scale ptr, + scale2 f32) |
| Prefill grouped GEMM | `moe/forward_prefill_routed.rs:34` (`run_routed_grouped_gemm`) | reads `gate_ptrs_t/up_ptrs_t/down_ptrs_t` — patch these for prefill |
| Model prefill loop | `model/trait_impl/prefill_a.rs:369` | has `self.layers[i+1]` — where L+1 prefetch is issued |
| Side stream + events | `moe/mod.rs:246-249` | `prefill_stream`, `event_a`, `event_b` already exist |
| Router known pre-GEMM | `moe/forward_prefill.rs:216-288` | top-k + `expert_offsets` on device before the grouped GEMM launches |
| Decode graph gate | `model/trait_impl/decode_a.rs:173-179` | add `&& !expert_streaming_engaged` next to existing `!hss_engaged` |
| EP scoping | `atlas-core/.../config/methods.rs:105-117` | `local_expert_range`; remote experts are NULL `packed_ptrs` entries |
| Reused I/O engine | `spark-storage/src/backend/{io_uring,posix}.rs` | `StorageBackend::read(&[ReadRequest], stream)` — backend-agnostic, KV-only by convention |
| Slot pool / eviction | `spark-storage/src/{scratch_pool,eviction}.rs` | `ScratchPool` assign/evict + `EvictionPolicy` rank — swap geometry to one expert record |

## The one genuinely new primitive: the UMA zero-copy arena

Gate 0(b) proved (on dgx-00) that a `cudaHostAlloc` pinned buffer is GPU-
addressable at the *same* VA and reads at 113 GB/s with no HtoD copy. But the
current `spark-storage` engine does **not** exploit this — `backend::read`
always issues `copy_h_to_d_async` into a separate `DeviceBuffer` (`io_uring.rs`
CQE loop; `cuda_min.rs` `PinnedBuffer`). Phase 2 adds a pinned *arena* the
streamer reads NVMe into directly and points `packed_ptrs`/`scale_ptrs` straight
at — deleting the bounce. Concretely: a new backend read path (or arena mode)
that lands O_DIRECT bytes in a pinned slot and returns the slot's device address,
skipping the memcpy. This is the "delete the hop" correction from the plan,
now hardware-confirmed.

## Correctness invariants (each needs a test, not a comment)

* **A — Tables are immortal.** Never re-run `build_ptr_table` after init (its
  device arrays are baked CUDA-graph arguments). Allocate once, patch in place.
* **B — Patch contents, batched.** Rewrite `u64` at `packed_ptrs + e*8` via a
  host shadow table + **one contiguous `copy_h2d` per dirty table per layer** —
  not 3,840 tiny copies/token.
* **C — Deferred free behind events.** Free an evicted slot only past its last
  consumer's completion event (`CudaEvent`), or a graph replay reads recycled
  bytes.
* **D — Disk format = resident format.** Builder writes post-transpose /
  pre-repacked bytes (Phase 1 does this); the header is versioned. ✔ landed.
* **E — Scope to `local_expert_range` under EP.** Never clobber NULL entries for
  remote experts — that corrupts EP indistinguishably from a stale pointer.
* **F — Decode graphs off when engaged.** `&& !expert_streaming_engaged` in the
  `use_graphs` conjunction at `decode_a.rs:173-179`.

## Acceptance gate (Phase 2 exit)

* Bit-identical logits vs `PosixBackend` oracle under a forced-eviction (capped)
  arena, on `qwen3.5-35b-a3b` shape (validation vehicle: cap the arena to ~2
  layers to emulate a 10× over-core model without the 200 GB checkpoint).
* Cold prefill fetch fully hidden under compute on A3B shape (no per-layer stall)
  on the **local-NVMe** store; and, with the store on NFS, hidden at the probed
  ring depth `D`.
* Warm path (streaming enabled but model fits) regresses PR #229 cold-prefill
  by < 10%.

Decode streaming stays out of scope until Gate 0(a) (router hit-rate) is run.
