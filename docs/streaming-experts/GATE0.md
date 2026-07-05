# Streaming Experts — Gate 0 measurements (GB10)

> **Verdict: Gate 0(b) CLEARED.** The "phantom ceiling" thesis holds on real
> GB10 hardware. On this box a pinned host allocation is GPU-addressable at the
> *same* virtual address (true zero-copy, no HtoD bounce), and NVMe sustains
> ~7 GB/s O_DIRECT reads at the expert granule — above the 5 GB/s kill line.
> The prefill/batch expert streamer is viable. Decode streaming remains gated
> on Gate 0(a) (router hit-rate), which is **not** yet run.

The plan (`Streaming Experts on GB10`, written from the PR #229 vantage, merged
as `main`'s HEAD) sequences two cheap measurements before any streamer code,
because either can kill the feature:

* **Gate 0(a)** — decode router hit-rate simulation. *Not run here* (needs logged
  router traces). Prefill does not depend on it; decode does.
* **Gate 0(b)** — UMA zero-copy + NVMe burst-granule bandwidth. **Run, below.**

## Hardware under test

| | |
|---|---|
| Host | `dgx-00` — NVIDIA **GB10** (Grace-Blackwell, unified LPDDR5X) |
| CUDA | 13.0 (V13.0.88) |
| NVMe | `nvme0n1` (ESL01TBTLCZ, 931 GB), root fs `/dev/nvme0n1p2` |
| Date | 2026-07-05 |

## Gate 0(b) part 1 — is the "second hop" real on GB10?

The stock streaming model (`ADR-0008`) assumes a discrete GPU: `NVMe -> host RAM
-> cuMemcpyHtoD -> VRAM`, PCIe-bound at ~3 GB/s. The plan's central claim is that
GB10 has no second hop — CPU and GPU share one LPDDR pool, so an `O_DIRECT` read
into a pinned arena is *already* GPU-addressable.

**Confirmed.** `docs/streaming-experts/gate0/uma_probe.cu` reports:

```
unified_addressing=1 can_map_host=1 concurrent_managed=1 pageable_mem_access=1
host_ptr=0x32ee00000 dev_ptr=0x32ee00000 get_devptr_rc=no error same_addr=1
zero-copy GPU read of pinned host buf: 0.593 ms, 113.2 GB/s, checksum OK
```

* `cudaHostAlloc` returns a pointer the GPU addresses at the **identical VA**
  (`same_addr=1`) — no `cudaHostGetDevicePointer` translation needed.
* A GPU kernel reading that pinned host buffer **directly** (no copy) sustains
  **113 GB/s** with a correct checksum. The HtoD bounce can be deleted: the
  expert pointer table can point straight at the NVMe-fill arena.

> ⚠️ Note for implementers: the *current* `spark-storage` engine does **not** do
> this yet. `PinnedBuffer` (`cuda_min.rs`) is a classic bounce buffer and every
> backend read issues an explicit `copy_h_to_d_async` into a separate
> `DeviceBuffer`. Deleting that copy — the UMA zero-copy arena — is Phase 2 work,
> not something that ships today. Gate 0(b) proves it *will* work; it does not
> mean it is wired.

Reproduce: `nvcc -arch=sm_121 -o uma_probe gate0/uma_probe.cu && ./uma_probe`

## Gate 0(b) part 2 — NVMe burst-granule bandwidth

Kill line: **< 5 GB/s ⇒ even prefill isn't hideable**, fall back to EP sharding.

Measured with `fio`, `--direct=1` (cache-bypassed), `bs=1712k` (≈ the a3b
per-expert record of 1.6875 MiB), across the io_uring backend's queue depths:

| queue depth | bandwidth |
|---|---|
| QD=1 | ~3704 MiB/s (~3.9 GB/s) |
| QD=4 | ~7024 MiB/s (~7.4 GB/s) |
| QD=8 | ~7006 MiB/s (~7.3 GB/s) |
| QD=16 | ~6988 MiB/s (~7.3 GB/s) |

At QD≥4 the disk sustains **~7 GB/s**, comfortably above the kill line. QD=1
(the `PosixBackend` oracle's regime) already clears ~3.9 GB/s. Combined with
part 1, the effective UMA streaming ceiling is **~7 GB/s, NVMe-bound** — exactly
the plan's "~5–7 GB/s" prediction, and ~40× below the 273 GB/s LPDDR compute bus.

That bandwidth ratio is the whole regime split: streaming wins where each fetched
byte is reused across many rows (prefill amortizes one expert fetch over an
8k-token chunk) and loses where it isn't (decode fetches one expert per token).

Reproduce: `bash gate0/nvme_granule_bench.sh <dir-on-nvme>`

## What this greenlights

* **Prefill / batch-offline streamer — build it.** Both physical premises hold.
* **Decode streaming — still deferred** behind Gate 0(a) (unmeasured router skew).
* **Warm (fits in core)** — streaming stays strictly opt-in / off by default.

Extrapolation to the real targets (per-expert bytes = `3·inter·hidden·9/16`,
verified in `spark_storage::expert` unit tests): a3b = 1.6875 MiB/expert,
397B = 6.75 MiB/expert (≈ 202 GB across 512 experts × 60 layers). At ~7 GB/s a
full 397B layer's expert set (≈ 3.4 GB) streams in ~0.5 s — hidden under an 8k
prefill chunk's grouped GEMMs, which is the regime the MVP targets.
