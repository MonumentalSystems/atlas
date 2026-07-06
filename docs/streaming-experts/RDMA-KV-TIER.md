# RDMA KV Overflow Tier — HANDOFF

A remote-RAM KV cache tier: when the KV cache overflows the GPU's HBM budget,
cold blocks stream to a peer's RAM over one-sided RDMA (RoCE/CX7) instead of the
local SSD — **~11–24 GB/s vs the ~2 GB/s USB SSD**. Extends the streaming-experts
verbs stack from read-only expert weights to a **read-write** KV tier.

- **Branch:** `feat/streaming-experts-mvp` · **PR:** MonumentalSystems/atlas#9
- **Companion:** `HANDOFF.md` (expert streaming), `RESEARCH-RDMA-TIER.md` (the idea).
- **All measured live** on dgx-00 (192.168.178.11) ↔ gx10 (192.168.178.12), CX7.

---

## 1. What landed (commits, newest first)

| commit | what |
|---|---|
| `94d5420` | **wire into high-speed-swap** — KV overflow routes to the RDMA peer |
| `618425e` | **zero-copy restore** — RDMA straight into UMA dst, restore dual-rails to 21.65 GB/s |
| `7409d88` | **dual-rail** — stripe both CX7 adapters, offload 13→24.45 GB/s |
| `e566a68` | **pipeline** — bounce ring, 1.5/3.5→9.6/13 GB/s |
| `aa537df` | bandwidth probe (revealed the small-group latency bound) |
| `e080926` | **RdmaKvBackend prototype** — StorageBackend over one-sided RDMA |

Sibling work this session (expert side): `3489273` reactive expert-granular
prefetch (fetch only routed experts, 32× less decode I/O — bit-identical, A3B sha
`ac48cd2d`); `4467096` smart loading (skip routed experts at load so over-core
checkpoints don't OOM mid-load); `fa66a42` compile-without-rdma-core fix.

---

## 2. Performance — the full journey (16 KiB KV groups, live CX7, GB10)

| stage | OFFLOAD (write) | RESTORE (read) |
|---|---|---|
| serial single-bounce | 3.5 GB/s | 1.5 GB/s |
| + pipelined (depth-16 ring) | 13.0 | 9.6 |
| + dual-rail (both adapters) | **24.45** | 9.8 ← *copy-bound* |
| + zero-copy restore | 24.2 | **21.65** |
| *vs USB SSD baseline* | *~12×* | *~11×* |

**Why each step mattered:**
- **Pipeline:** small KV groups (16 KiB, ~100× smaller than the 1.7 MB expert
  records) are latency-bound serial; a ring of `depth` registered bounces keeps
  `depth` ops in flight so latency overlaps.
- **Dual-rail:** the two CX7 ports are *independent* PCIe paths (measured: both
  `ib_read_bw` rails concurrently = 196 Gb/s vs 112 single, ~1.75×). RDMA does
  **not** auto-aggregate — you stripe QPs across both devices yourself.
- **Zero-copy restore:** the bounce→HBM `copy_h2d` was the restore bottleneck
  (~9.7 GB/s copy engine, doesn't dual-rail). On GB10 a pinned buffer is
  GPU-addressable at the **same VA** (host==dev), so RDMA lands **directly** into
  a UMA destination — no bounce, no copy, no `stream_sync` (completion == bytes
  GPU-visible). The expert-arena trick applied to KV.

**Concurrency A/B (2026-07-06, holo-0.8b):** first sweep of the tier under
*concurrent* load — see `kv-bench/CONCURRENCY-FINDINGS.md`. Finding: the overflow
tier (RDMA **and** NVMe alike) is **single-sequence-only** today — correct at C=1,
wrong recall at C=2, hard-fail at C≥4 — because the disk-block-id namespace is sized
for one sequence (`high_speed_swap.rs:239`). Not an RDMA fault; the fix is to scope
the id pool per-seq + grow the arena. At 0.8b the granule is latency-bound so
RDMA≈NVMe; RDMA's bandwidth win needs the big-model granule (the A3B number below).

**Agentic end-to-end (5-turn, A3B, identical task):**
- Local KV (all HBM): **65.5 tok/s** aggregate.
- RDMA remote KV (`cap=8` blocks = 128 tok HBM, ~entire cache remote): **37.6 tok/s**
  = **57% of local** — a ~1.7× slowdown for a fully-remote KV cache (single-rail,
  bounce path; the predictor restoring only relevant blocks is why it's 1.7× not 10×).

---

## 3. Architecture

```
  GPU KV (HBM budget)  →  local pinned scratch  →  PEER RAM (RDMA)  →  [ USB SSD ]
                                                   ↑ this tier          ↑ cold archive
```

**`RdmaKvBackend`** (`spark-storage/src/rdma_kv_backend.rs`) — a `StorageBackend`
(the *same* trait `IoUringBackend`/`PosixBackend` implement, so it drops into the
existing swap machinery). `write_from_host(group, src)` → RDMA WRITE to the peer
at `base + group_id*group_stride`; `read(reqs, stream)` → RDMA READ back. Holds
N **rails** (`Rail{verbs, bounces, inflight}`), one QP per CX7 adapter; ops stripe
round-robin. Restore is pipelined (interleaved reap across rails) with an optional
zero-copy path. Group addressing is the flat `GroupLayout::group_id` space
(simpler than experts — no per-layer files).

**`kv_peer`** (`spark-storage/src/kv_peer.rs`) + `atlas-kv-peer` bin — a dumb
RW **memory blade**: client sends `total_bytes`+`n_rails`; peer `mmap`s an
anonymous arena and `ibv_reg_mr`s it **once per rail** (shared physical pages →
NOT N× RAM), publishes `{qpn,psn,gid,base,rkey}` per rail, connects, idles. Peer
CPU never touches a byte (one-sided). Each group belongs to one client → no
coherence protocol.

**Verbs shim** (`spark-storage/src/rdma_shim.c` + `rdma_verbs.rs`) — shared with
the expert tier. This session added `rs_post_write` (`IBV_WR_RDMA_WRITE`) and a
`rs_reg_mr` access-flags bitmask (bit0=REMOTE_READ, bit1=REMOTE_WRITE|LOCAL_WRITE,
0=LOCAL_WRITE); QP INIT now grants REMOTE_WRITE (MR flags still enforce — the
expert store stays read-only).

**Wiring** (`spark-storage/src/high_speed_swap.rs`) — `HighSpeedSwap.backend` is
now `Box<dyn StorageBackend>`; constructed as `RdmaKvBackend` when `$ATLAS_KV_PEER`
is set, else `IoUringBackend` (default unchanged, gated on `atlas_rdma_verbs`).
The offload/restore call sites (`high_speed_swap/impl_more.rs`) already used the
trait, so nothing else changed.

---

## 4. Config knobs

**Client (the serving process):**
- `$ATLAS_KV_PEER=host:port` — enable the RDMA KV tier (else local NVMe).
- `$ATLAS_KV_DUAL_RAIL=1` — stripe both adapters (rail 0 = `$ATLAS_EXPERT_RDMA_DEV`
  /`_GID`, rail 1 = `$ATLAS_KV_RAIL2_DEV`/`_GID`; defaults `roceP2p1s0f1`:3 /
  `rocep1s0f1`:3).
- `$ATLAS_KV_ZERO_COPY=1` — RDMA directly into UMA dst (needs UMA KV scratch — see
  §6; without UMA scratch, `reg_dst` errors, so leave off until that lands).
- `$ATLAS_KV_PIPELINE_DEPTH=16` — bounces per rail.
- CLI: `--high-speed-swap --high-speed-swap-dir <d> --high-speed-swap-gb <g>
  --high-speed-swap-cache-blocks-per-seq <N>` (N blocks × block_size = HBM-resident
  tokens; the rest overflow to the tier).

**Peer:** `atlas-kv-peer --listen 0.0.0.0:9910 --rail roceP2p1s0f1:3 --rail rocep1s0f1:3`
(repeat `--rail` per adapter).

---

## 5. Reproduce

```bash
# 1. peer on gx10 (SSH is on the .1.177 MGMT ip, NOT the .178 CX7 link!):
ssh 192.168.1.177 '/home/ms/atlas-kv-peer --listen 0.0.0.0:9916 \
  --rail roceP2p1s0f1:3 --rail rocep1s0f1:3'

# 2. bit-identical round-trip + bandwidth (GPU + live peer):
ATLAS_KV_PEER=192.168.178.12:9916 ATLAS_KV_DUAL_RAIL=1 ATLAS_KV_ZERO_COPY=1 \
ATLAS_EXPERT_RDMA_DEV=roceP2p1s0f1 ATLAS_EXPERT_RDMA_GID=3 \
ATLAS_KV_RAIL2_DEV=rocep1s0f1 ATLAS_KV_RAIL2_GID=3 \
  cargo test -p spark-storage --release --lib rdma_kv_bandwidth -- --ignored --nocapture

# 3. live over-budget recall (KV overflows to the remote, model recalls it):
CKPT=$(ls -d /tank/hf/hub/models--AEON-7--Qwen3.6-35B-A3B-heretic-NVFP4/snapshots/*/)
ATLAS_KV_PEER=192.168.178.12:9916 ATLAS_EXPERT_RDMA_DEV=roceP2p1s0f1 ATLAS_EXPERT_RDMA_GID=3 \
  ./target/release/spark serve --model-from-path "$CKPT" --model-name a3b \
  --port 8955 --gpu-memory-utilization 0.55 \
  --high-speed-swap --high-speed-swap-dir /tmp/atlas-hss-rdma --high-speed-swap-gb 8 \
  --high-speed-swap-cache-blocks-per-seq 640
# then POST a >10K-token prompt with a fact at the start; it recalls it (KV came off the peer).
```

Validated: round-trip **bit-identical** (bounce + zero-copy); live A3B recall of a
secret at the start of a 17,137-token prompt (HBM cap 10,240) → correct
("TANGERINE-7742"); peer arena RSS 0→2.51 GB (pages faulted by the RDMA WRITEs).

---

## 6. Gotchas / open items (READ before extending)

1. **`cache-blocks-per-seq=1` hits pre-existing high-speed-swap bug #31** (sliding
   window evicts a block before all attention layers offload it). Minimum viable
   is `~8` (128 tok HBM). Literal "0 tokens HBM" is not reachable via this path —
   it's an hss eviction-race bug, **not** the RDMA tier.
2. **The peer has NO memory cap** — `kv_peer` only rejects `total > 4 TiB` (sanity).
   No per-client cap, no aggregate budget, no eviction under pressure. A client can
   OOM the peer. **TODO: `--max-blade-gb` (reject oversized at handshake) + a
   running aggregate across connections.** Applies to the expert peer too.
3. **Zero-copy in the serving path — LANDED (this note was stale).**
   `ATLAS_KV_ZERO_COPY=1` requires the read destination to be pinned UMA (host==dev
   VA) so `ibv_reg_mr` succeeds and the GPU reads it. The swap ScratchPool now
   allocates a UMA pool for this (`high_speed_swap.rs:131`, `new_preferring_uma`,
   with a safe fall-back to device memory on a non-UMA host). Verified live on the
   35B (2026-07-06): log shows `scratch pool UMA=true (zero-copy restore enabled)` +
   `registered UMA landing region … zero-copy restore live`. **BUT** it does NOT
   speed decode — the per-step restore is latency-bound, so zero-copy/dual-rail
   (bandwidth wins) only help prefill TTFT, not decode tok/s. See
   `kv-bench/CONCURRENCY-FINDINGS.md`.
4. **Coherence assumption:** after an RDMA completion the NIC's DMA to LPDDR is
   assumed GPU-visible (same as the expert arena). Holds on GB10; revalidate on
   other hardware.
5. **gx10 SSH is `192.168.1.177` (management), not the CX7 `.178` link.** Its sshd
   throttles under rapid reconnects — space out connections.
6. **Never blanket-`pkill` on the shared boxes** — scope to your own PIDs/ports
   (see the repo memory; a broad pkill killed another tester's server this session).

---

## 7. Recommended next steps

1. **Peer memory cap + aggregate budget** (§6.2) — small, do before unattended use.
2. **UMA KV scratch pool** → zero-copy restore end-to-end in the live serving path
   (currently the standalone bandwidth path is zero-copy, serving is bounce). This
   is the last piece to get the 21 GB/s restore into real inference.
3. **Dual-rail the expert tier** — same pattern; the expert store is bandwidth-bound
   at 1.7 MB records, so ~1.75× (14→24 GB/s) for warm prefill/decode, immediate.
   (2 QPs on 2 devices, peer registers the store per-PD = shared pages.)
4. **3-tier cascade** (local RAM → peer RAM → SSD) with eviction policy choosing
   the tier — currently RDMA fully *replaces* the SSD backend when enabled.
5. **LoRA adapter rotation** (separate branch, stacked on base LoRA support):
   adapters are read-only weights → the one-sided READ expert tier applies verbatim;
   the peer holds an adapter pool, the client fetches/activates on rotation.

---

## 8. File map (this tier)

- `crates/spark-storage/src/rdma_kv_backend.rs` — `RdmaKvBackend` (client, cuda +
  verbs); `Rail`, pipeline, dual-rail, zero-copy; round-trip + bandwidth tests.
- `crates/spark-storage/src/kv_peer.rs` — RW blade server + wire protocol.
- `crates/atlas-expert-pack/src/bin/kv_peer_main.rs` — `atlas-kv-peer` bin.
- `crates/spark-storage/src/rdma_shim.c` / `rdma_verbs.rs` — verbs FFI (shared;
  `rs_post_write` + RW MR flags added here).
- `crates/spark-storage/src/high_speed_swap.rs` — backend selection (the wiring).
- `crates/spark-storage/src/high_speed_swap/impl_more.rs` — offload/restore calls.
- `crates/spark-storage/src/group.rs` — `GroupLayout` (the flat group-id space).
