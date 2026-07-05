# Research (post-MVP) — RDMA weight channel as an expert tier

> **Status: research / TODO. Not scheduled.** Captures an idea worth pursuing
> once the Phase 2 prefill streamer lands. Nothing here blocks the MVP.

## The idea

Instead of **expert parallelism (EP)** — which *splits* the experts across nodes
and does an all-to-all of activations every MoE layer, coupling compute with a
collective — treat RDMA as **just another storage tier**: a faster pipe to a peer
that holds the weights. A second Atlas instance (a "weight server") keeps the
expert store resident in its LPDDR and answers expert-record fetches over an
RDMA channel (ConnectX-7). The streaming box does **all** the compute on one GPU;
the peer is a dumb, stateless weight cache.

```
   EP (today):   node A ⇄ node B ⇄ node C ⇄ node D   — activations all-to-all
                 compute is distributed; forward pass is a collective

   RDMA tier:    [ streaming box: full forward ]  --RDMA READ-->  [ weight peer ]
                 compute stays on ONE box; peer only serves bytes
```

Why it's attractive:

* **~20 GB/s** peer-LPDDR RDMA (artifact bandwidth tier) vs ~7 GB/s local NVMe —
  ~3× the fetch bandwidth, and **no local disk capacity** needed on the
  streaming box (the 200 GB store lives once, on the peer).
* **Simpler failure/consistency model than EP.** The forward pass is not
  distributed; losing the peer degrades to "cache miss / fall back to NVMe or
  EP," not a broken collective. One peer can serve several streaming clients.
* **Slots into the same abstraction as everything else.** It is a tier in the
  residency hierarchy `device < pinned < NVMe < RDMA-peer`, behind one trait.

## Sketch

An `ExpertTier` trait the streamer fetches through, so NVMe and RDMA are
interchangeable and stackable:

```rust
trait ExpertTier {
    /// Fetch record `key` into `dst` (a pinned/GPU-addressable arena slot),
    /// stream-ordered; returns an event the consumer joins on. Same contract
    /// the Phase 2 NVMe path already uses.
    fn fetch(&self, key: ExpertKey, dst: DevPtr, stream: Stream) -> Result<CudaEvent>;
}
```

* `NvmeTier` — Phase 2 (io_uring / O_DIRECT into the pinned arena).
* `RdmaPeerTier` — research: a **one-sided RDMA READ** from the peer's registered
  expert arena straight into the local pinned arena. On GB10 that landing buffer
  is GPU-addressable at the same VA (Gate 0(b)), so — if GPUDirect RDMA over CX7
  cooperates — the bytes are consumable with no extra copy, same zero-copy win as
  the NVMe path.
* The **weight peer** is just an Atlas instance that `mmap`s / pins the Phase 1
  `.xpr` store, registers it as an RDMA memory region, and runs a tiny responder.
  It reuses the exact same `ExpertIndex` geometry — the offset math for
  `(layer, expert)` is identical whether the bytes come from a local fd or a
  remote MR.

**Prefetch still rules.** RDMA latency sits between local NVMe and NFS, so the
Phase 2 prefetch ring (`D+1` arenas, layer-ahead) is unchanged in shape — only
the probed depth `D` differs. If anything, the deep-prefetch discipline built for
the NFS case is exactly what makes an RDMA tier viable.

## Open questions to settle before committing

* **GPUDirect RDMA on GB10 + CX7.** Does one-sided RDMA READ land coherently in
  pinned LPDDR that the GPU then reads zero-copy, or is a host-staged bounce
  required? (Note the fleet's CX7 firmware/wedge history — validate on a healthy
  link.) cuFile/GDS is already rejected on GB10 (`ADR-0008`); RDMA is a separate
  path.
* **MR registration cost + pinning** of a 200 GB arena on the peer; huge-page
  backing; one MR vs per-layer MRs.
* **One-sided vs two-sided.** One-sided READ keeps the peer CPU-idle (best), but
  needs the client to know remote offsets — trivial here since geometry is shared
  via `ExpertIndex`.
* **Multi-client fairness / QoS** if one peer serves several streaming boxes.
* **Coherence with eviction** — the peer is read-only; the client's arena/tier
  bookkeeping is unchanged, but measure whether RDMA completion events compose
  cleanly with the deferred-free invariant (C).

## Relationship to existing plan

This is the concrete form of the reuse ledger's deferred row (*"ExpertTier trait
w/ RDMA stub"*) and the Foresight design's *"RDMA-as-faster-tier"* framing — the
one salvaged idea both adversarial reviewers kept for post-MVP multi-node scale.
It is explicitly **not** the maximal "stream everything" design they killed;
it is a bandwidth tier under the same prefill-hides-fetch core.

## Measured on the cabled fabric (2026-07-05, dgx-00 ↔ gx10, CX7 RoCEv2)

Peer cabled: dgx-00 (192.168.178.11) ↔ gx10-9959 (192.168.178.12), `roceP2p1s0f1`
ACTIVE 200 Gb/s, RoCEv2 GID index 3. Bandwidth (dgx-00 reads from gx10):

| transport | bandwidth | peer CPU |
|---|---|---|
| local NVMe O_DIRECT (uma tier) | ~7 GB/s | — |
| TCP-over-CX7, single stream (current RdmaTier) | 41 Gb/s (~5.2 GB/s) | busy |
| TCP-over-CX7, 8 streams | 111 Gb/s (~13.9 GB/s) | busy |
| **one-sided RDMA READ (`ib_read_bw`, 1.7 MB records, 4 QP)** | **112 Gb/s (~14 GB/s)** | **idle** |

Verdict: verbs one-sided RDMA READ delivers ~14 GB/s (2× local NVMe) at the
expert-record size with the peer CPU completely idle — the intended Phase B win.
(Both plateau ~112 Gb/s, below the 200 Gb/s line rate — a per-NIC/PCIe ceiling on
GB10, not a protocol limit.) Reproduce: `gate0/verbs_rdma_read_bw.sh <peer-ip>`.
Phase B integration (RdmaTier verbs data path) is the remaining code step; the
TCP RdmaTier already provides a bit-identical peer path today.
