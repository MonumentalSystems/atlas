# KV-tier benchmark harness

A/B of the KV backends (normal HBM vs `--high-speed-swap` overflow to {local NVMe,
RDMA peer}) under load, on GB10 (dgx-00) ↔ gx10 `atlas-kv-peer` over ConnectX-7.

## Scripts
- `bench_client.py` — fires C concurrent **streaming** chat requests, reports
  per-request TTFT + decode tok/s + aggregate req/s + correctness (recall of the
  tagged secret). Good for non-thinking models (client counts `delta.content`).
- `kv_sweep.sh` — small-model sweep (holo-3.1-0.8b), 4 backends × C=1/2/4/8.
- `kv_sweep_35b.sh` — big-model sweep (holo-3.1-35B-A3B). **Timing is parsed from
  the SERVER LOG** line `Done: N tokens (stop) X tok/s, TTFT=Yms`, NOT client SSE —
  the 35B is a THINKING model, so client `delta.content` counting misses the
  reasoning tokens and under-reports decode. Correctness from the response body.

## Peer (run on the other GB10)
    atlas-kv-peer --listen 0.0.0.0:9920 --rail roceP2p1s0f1:3
Client env: `ATLAS_KV_PEER=<peer-cx7-ip>:9920 ATLAS_EXPERT_RDMA_DEV=roceP2p1s0f1
ATLAS_EXPERT_RDMA_GID=3`. See `../RDMA-KV-TIER.md` §4/§5.

## Findings
`CONCURRENCY-FINDINGS.md` (0.8b) — overflow tier is single-sequence-only today
(disk-block-id namespace sized for one seq). 35B scale numbers appended there.
