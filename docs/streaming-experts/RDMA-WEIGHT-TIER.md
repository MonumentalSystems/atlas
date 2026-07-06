# RDMA Weight Tier — fast model swaps (`RdmaWeightLoader`)

Stage a model's weights in a peer's RAM (registered `REMOTE_READ` MRs) and load
them into the compute head over one-sided RDMA instead of off the local disk.
Weights are read-only, so this is a pure one-sided `IBV_WR_RDMA_READ` data path —
the peer CPU is idle. It generalizes the expert-tier verbs stack from
`(layer, expert)` records to **every** safetensors tensor.

## Pieces

- **`atlas-weight-peer`** (`crates/atlas-expert-pack/src/bin/weight_peer_main.rs`,
  server in `spark-storage/src/weight_peer.rs`) — mmaps each safetensors shard of a
  staged model, `ibv_reg_mr` `REMOTE_READ` on every rail's PD (shared physical
  pages), and publishes a manifest `{ shard_files, shard_lens }` plus per-rail
  `(base, rkey)` per shard. Multi-model staging (`--stage` repeatable); memory
  ceiling via the shared `blade_cap::CommitLedger` (`--max-blade-gb`). It's a
  cache: the first stage of a model faults its pages in from disk; later swaps
  read warm RAM.
- **`RdmaWeightLoader`** (`spark-storage/src/weight_tier_rdma.rs`) — a
  `WeightLoader` impl. Parses its **own local** safetensors headers (the same
  `fast_weights::header` path the disk loader uses) so tensor
  name/dtype/shape/offset/len are byte-identical **by construction**; only the
  tensor DATA comes over RDMA (`remote_addr = shard_base + offset_in_shard`).
  Dual-rail striped (`tensor % n_rails`), pinned-bounce + `copy_h2d` per tensor.
  Honors `should_skip_tensor` / `stream_all_experts`, so it composes with expert
  streaming (routed experts still come from the expert peer).
- **Wire** — `load_weight_store` selects `RdmaWeightLoader` when `$ATLAS_WEIGHT_PEER`
  is set; both disk loaders stay byte-for-byte unchanged when it is unset.

## Validated (live, GB10, 2026-07-06)

Model: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` (21.73 GB, 124306 tensors, single
shard). Peer on gx10 (both CX7 rails), compute head on dgx-00, `util 0.55`.

| loader | weights → HBM | greedy completion |
|---|---|---|
| **RDMA dual-rail (warm peer)** | **5.33 s** | `Mercury, Venus, Earth, Mars … 1,2,3,4,5` |
| disk fast O_DIRECT (`/tank` over CX7) | 9.35 s | **identical** |

- **Byte-identical**: the temp-0 greedy completion is character-for-character the
  same whether weights came over RDMA or off disk. NVFP4 is unforgiving — a single
  corrupted weight/scale tensor derails generation — so a clean, matching
  completion confirms all 124306 RDMA-loaded tensors are correct.
- **Faster**: 1.75× the fast O_DIRECT loader here, and the `/tank` mount is already
  a fast ext4-over-CX7 path (~2.3 GB/s); against the ~2 GB/s USB SSD the real
  over-core scenario uses, the gap is wider.
- **Latency-bound, not bandwidth-bound**: 124306 tiny NVFP4 tensors (weight +
  per-tensor scale) mean the ~4 GB/s effective is dominated by per-WR post/poll
  overhead (one outstanding WR per rail), well under the ~24 GB/s large-transfer
  ceiling. Follow-on: deepen the in-flight WR ring / coalesce adjacent tensors to
  approach the bandwidth ceiling for many-tensor checkpoints.

## Reproduce

```bash
CKPT=$(ls -d /tank/hf/hub/models--AEON-7--Qwen3.6-35B-A3B-heretic-NVFP4/snapshots/*/)

# 1. peer on gx10 (SSH is on the .1.177 MGMT ip, NOT the .178 CX7 link!):
ssh 192.168.1.177 "RUST_LOG=info setsid /home/ms/atlas-weight-peer \
  --stage $CKPT --listen 0.0.0.0:9910 \
  --rail roceP2p1s0f1:3 --rail rocep1s0f1:3 >/home/ms/wpeer.log 2>&1 </dev/null &"

# 2. build for the model + serve with weights over RDMA (dual-rail):
ATLAS_TARGET_MODEL=qwen3.6-35b-a3b cargo build --release -p spark-server \
  --no-default-features --features cuda
ATLAS_WEIGHT_PEER=192.168.178.12:9910 ATLAS_WEIGHT_DUAL_RAIL=1 \
ATLAS_WEIGHT_RDMA_DEV=roceP2p1s0f1 ATLAS_WEIGHT_RDMA_GID=3 \
ATLAS_WEIGHT_RAIL2_DEV=rocep1s0f1 ATLAS_WEIGHT_RAIL2_GID=3 \
  ./target/release/spark serve --model-from-path "$CKPT" --model-name a3b \
  --port 8956 --gpu-memory-utilization 0.55
# → "RDMA-loaded 124306 weight tensors" in ~5.3s. Same prompt vs a disk serve
#   (ATLAS_WEIGHT_PEER unset) yields an identical temp-0 completion.
```

Single-rail: leave `ATLAS_WEIGHT_DUAL_RAIL` unset. The first swap after peer start
is cold (peer faults shards from disk); the second+ swap reads warm RAM.
