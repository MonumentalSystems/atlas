# Streaming Experts

Unlock over-core NVFP4 MoE checkpoints (e.g. `qwen3.5-397b-a17b`, 512 experts,
~200 GB — today gated behind a 4-node EP=4 cluster) on a **single GB10** by
streaming cold experts from NVMe on the prefill path. Experts live on NVMe in
resident layout; layer L+1 is prefetched while layer L computes; dispatch is
redirected by patching the device-resident `ExpertPtrTable`.

## Status

| Phase | What | State |
|---|---|---|
| Gate 0(b) | UMA zero-copy + NVMe bandwidth on real GB10 | ✅ **measured, cleared** — [`GATE0.md`](GATE0.md) |
| Gate 0(a) | Decode router hit-rate simulation | ⬜ not run (gates decode only) |
| Phase 1 | Expert-record format + offline builder | ✅ **landed + tested** — `spark-storage::expert{,_pack}`, `atlas-expert-pack` |
| Phase 2 | Prefill streamer (arena, prefetch, ptr-patch) | 📐 **designed, staged for review** — [`PHASE2-PREFILL-STREAMER.md`](PHASE2-PREFILL-STREAMER.md) |
| Post-MVP | Decode streaming; RDMA weight-tier | 🔬 research — [`RESEARCH-RDMA-TIER.md`](RESEARCH-RDMA-TIER.md) |

## Key results so far

* **The phantom ceiling is real.** On dgx-00 a pinned host allocation is GPU-
  addressable at the *same* virtual address (zero-copy, 113 GB/s GPU read); the
  3 GB/s two-hop ceiling in `ADR-0008` is a discrete-GPU assumption that does not
  apply to GB10's unified LPDDR. NVMe O_DIRECT sustains ~7 GB/s at the expert
  granule (QD≥4), above the 5 GB/s kill line. → prefill streaming is viable.
* **Prefetch is the mechanism, not an optimization** — and it is *especially*
  critical when the store is served over `/tank` NFS (deeper prefetch ring to
  cover RTT). See Phase 2 doc.
* **The offline builder works on a real checkpoint** (AEON-7 A3B-NVFP4): produces
  4 KiB-aligned resident records + manifest, verified byte-identical round trip.

## Try it

```bash
# Inspect a checkpoint's expert grid without writing anything:
atlas-expert-pack --checkpoint <hf-snapshot-dir> --dry-run

# Build a 1-layer store with round-trip verification:
atlas-expert-pack --checkpoint <dir> --out ./expert-store --max-layers 1 --verify 16

# Reproduce the Gate 0(b) measurements on any GB10:
nvcc -arch=sm_121 -o uma_probe gate0/uma_probe.cu && ./uma_probe
bash gate0/nvme_granule_bench.sh <dir-on-local-nvme>
```

## Provenance

Synthesized from a 12-agent fable workflow (recon → review → design →
adversarial-verify) over Atlas, then recon-verified against `main` @ the PR #229
merge (`6d79e14`). The maximal "stream everything incl. interactive decode"
design was correctly rated *not viable* on GB10 by both adversarial reviewers;
this plan is the intersection of their salvage lists — prefill-only, decode
deferred behind a measurement gate.
