# Streaming Experts

Unlock over-core NVFP4 MoE checkpoints (e.g. `qwen3.5-397b-a17b`, 512 experts,
~200 GB — today gated behind a 4-node EP=4 cluster) on a **single GB10** by
streaming cold experts from NVMe on the prefill path. Experts live on NVMe in
resident layout; layer L+1 is prefetched while layer L computes; dispatch is
redirected by patching the device-resident `ExpertPtrTable`.

## Status

| Stage | What | State |
|---|---|---|
| Gate 0(b) | UMA zero-copy + NVMe bandwidth on real GB10 | ✅ **measured, cleared** — [`GATE0.md`](GATE0.md) |
| Gate 0(a) | Decode router hit-rate simulation | ⬜ not run (gates decode only) |
| Stage 0 | Config/CLI contract + invariant-F decode gate | ✅ **landed + tested** (no-op by default) |
| Stage 1 | ExpertTier + UMA zero-copy arena + Posix oracle | ✅ **landed, GPU-parity validated** |
| Phase 1 | Expert-record format + offline builder | ✅ **landed + tested** — `atlas-expert-pack` |
| Stage 2 | Prefill streamer (arena, ptr-patch) | ✅ **BIT-IDENTICAL on real A3B** (resident == uma == posix) |
| Stage 3 | Async prefetch overlap + deferred-free | ✅ **BIT-IDENTICAL** (capped arena, uma + rdma) |
| Stage 4 | RDMA weight-tier (TCP-over-CX7) + peer | ✅ **BIT-IDENTICAL** (peer-as-tier, end-to-end) |
| Stage 4 · Phase B | One-sided RC RDMA_READ (verbs) | 🔬 gated on a cabled peer + CX7 28.45→28.47 FW — [`RESEARCH-RDMA-TIER.md`](RESEARCH-RDMA-TIER.md) |
| Post-MVP | Decode streaming | 🔬 gated on Gate 0(a) |

**All five residency configs are bit-identical on the real AEON-7 A3B-NVFP4 model
(GB10):** `resident == uma == posix == rdma`, blocking and async, same final-norm
hash. Reproduce with `scripts/streaming-experts/verify_logits.sh`.

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
# Single-GPU build (skip NCCL) for the A3B target:
ATLAS_TARGET_MODEL=qwen3.6-35b-a3b \
  cargo build --release -p spark-server --no-default-features --features cuda

# Build the resident expert store from a checkpoint (stage on LOCAL NVMe):
cargo run --release -p atlas-expert-pack -- \
  --checkpoint <hf-snapshot-dir> --out /path/on/nvme/expert-store --verify 8

# Serve with streaming experts (uma zero-copy | posix oracle | rdma peer):
spark serve --model-from-path <ckpt> --stream-experts /path/on/nvme/expert-store \
  --expert-arena-layers 2 --expert-backend uma   # ~20x over-core emulation

# RDMA peer tier (start the peer, point the server at it):
atlas-expert-peer --store /path/on/nvme/expert-store --listen 0.0.0.0:9909
ATLAS_EXPERT_PEER=<peer-host>:9909 spark serve ... --expert-backend rdma

# Prove bit-identity (resident == uma == posix) on any A3B checkpoint:
BIN=./target/release/spark bash scripts/streaming-experts/verify_logits.sh <ckpt> <store>

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
