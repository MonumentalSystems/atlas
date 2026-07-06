# Streaming Experts — HANDOFF

Run an over-core NVFP4 MoE checkpoint (e.g. `qwen3.5-397b-a17b`, 512 experts,
~200 GB, today gated behind a 4-node EP=4 cluster) on a **single GB10** by
streaming cold experts from NVMe/RDMA on the prefill path.

- **Branch:** `feat/streaming-experts-mvp` · **PR:** MonumentalSystems/atlas#9
- **Everything below is validated BIT-IDENTICAL** on the real AEON-7 A3B-NVFP4
  model on dgx-00 (GB10): same final-norm hash across every residency config.
- **Base:** branched from `main` @ `6d79e14` (the merged "PR #229" the plan cites).

---

## 1. Status at a glance

| Stage | What | State |
|---|---|---|
| Gate 0(b) | UMA zero-copy + NVMe bandwidth on GB10 | ✅ measured (`GATE0.md`) |
| Stage 0 | config/CLI + invariant-F decode gate | ✅ landed, 5 tests, no-op by default |
| Stage 1 | `ExpertTier` + UMA arena + Posix oracle | ✅ landed, GPU byte-parity |
| Phase 1 | expert-record format + offline builder | ✅ `atlas-expert-pack`, byte-identical round trip |
| Stage 2 | prefill streamer (arena, batched ptr-patch) | ✅ **bit-identical** on A3B |
| Stage 3 | async prefetch overlap + deferred-free | ✅ **bit-identical** (capped arena) |
| Stage 4 | RDMA-peer tier over **TCP** + `atlas-expert-peer` | ✅ **bit-identical**, end-to-end |
| **WS1** | **resident-skip → real over-core** | ✅ **bit-identical + memory freed + over-core proof** |
| **WS3** | **persistent residency (agentic warm prefill)** | ✅ **bit-identical + ~5.6× warm speedup** |
| — | **over-core generation** (decode-via-prefill) | ✅ **works, output == resident** (26.5 tok/s), see §5 |
| WS2 | verbs one-sided RDMA READ **bandwidth** | ✅ measured ~14 GB/s (`ib_read_bw`) |
| **WS2** | **verbs data-path INTEGRATION** | ✅ **bit-identical + ~12 GB/s in-app, 6.6× TCP prefill** — see §6 |
| **WS4** | **smart loading** (skip routed experts at load) | ✅ over-core checkpoints don't OOM mid-load; 122B validated |
| **WS5** | **reactive expert-granular prefetch** (fetch only routed) | ✅ **bit-identical** (sha `ac48cd2d`); 7× A3B over-core decode |
| **WS6** | **RDMA KV overflow tier** (KV → remote RAM blade) | ✅ **24 GB/s offload / 21.65 restore, live over-budget recall** — see `RDMA-KV-TIER.md` |

Over-core now does full **generation** on one box (not just prefill) — see §5.
The verbs one-sided RDMA READ tier is landed and bit-identical. Since then this
session also added **smart loading** (WS4, over-core checkpoints skip loading
routed experts so they don't OOM — validated on the 122B), **reactive
expert-granular prefetch** (WS5, decode fetches only the ~8 routed experts/layer
not all 256 — 32× less store I/O, 7× faster A3B over-core decode, bit-identical),
and a whole **RDMA KV overflow tier** (WS6): the KV cache spills to a peer's RAM
over one-sided RDMA at 24/21 GB/s (~11× the SSD), wired into `--high-speed-swap`,
proven with a live over-budget recall + an agentic benchmark (57% of local-KV).
**WS6 has its own handoff: `RDMA-KV-TIER.md`.**

**Vehicles beyond A3B:** 122B-A10B (Sehyo, `qwen3.5-122b-a10b`) over-core validated
(store built, streams over verbs). MiniMax-M2.7-NVFP4 and DeepSeek-V4-Flash are
supported for *resident* serving but their loaders don't yet attach the streamer
/ resident-skip (qwen-family only) + the store builder uses `.mlp.experts.N.` not
their `block_sparse_moe`/`w1w2w3` naming — so streaming them needs a loader+builder
port (moderate; the MoE forward is the shared `MoeLayer`). qwen3.5-397b-a17b kernels
exist but no checkpoint.

---

## 2. Measured numbers (all on dgx-00 GB10, AEON-7 A3B-NVFP4)

**Bandwidth**
| path | GB/s | notes |
|---|---|---|
| pinned host = GPU-addressable | 113 GB/s | zero-copy, same VA (Gate 0b) |
| local NVMe O_DIRECT (uma tier) | ~7 | QD≥4, 1.7 MB granule |
| TCP-over-CX7 single-stream (rdma tier, TCP) | ~5.2 | peer CPU busy |
| TCP-over-CX7 8-stream | ~13.9 | peer CPU busy |
| verbs one-sided RDMA READ, 1.7 MB records | ~14 | peer CPU idle (`ib_read_bw`) |
| **verbs data path, in-app (rdma-verbs tier)** | **~12** | **peer CPU idle** (16.9 GiB/1.45 s cold prefill) |

**Streaming cold-prefill throughput (1824-token prompt, `--expert-arena-layers 2`,
each layer re-streamed)** — the app-level view of the tier bandwidth:
| tier | prefill tok/s | note |
|---|---|---|
| rdma-verbs | **1262** | one-sided RDMA READ, peer idle — near compute-bound |
| uma (local NVMe) | 312 | O_DIRECT ~7 GB/s |
| rdma TCP | 190 | single-stream, peer pread+socket-write bound |

**Memory (A3B, util 0.85, free after load)**
- resident: 62 GB free · **streaming (WS1): 80 GB free** → ~18 GB expert weights freed.
- Over-core proof: at `--gpu-memory-utilization 0.45` resident **OOMs**, streaming **loads + serves**.
- Streaming load is ~2× faster (17 s vs 32 s — no resident expert materialization).

**Prefill throughput (1441-token prompt)**
- resident **2641 tok/s** · streaming cold **256 tok/s** · **streaming warm (WS3) ~1440 tok/s**.
- Warm prefill: cold 6.9 s → **~1.0 s (~5.6×)** once the working set is arena-resident.

---

## 3. Architecture

`ExpertTier` (`spark-storage/src/expert_tier.rs`) sits ABOVE the record layer
(not a `StorageBackend` impl — that's KV-tile shaped and syncs on return):

- **`PosixTier`** — deterministic bounce oracle (`pread` → `copy_h2d`). The
  bit-identical acceptance reference.
- **`UmaArenaTier`** — the zero-copy path: O_DIRECT NVMe fill straight into a
  pinned LPDDR arena (`ExpertArena`, `host_va==dev_va`); the ptr table points at
  the pinned VA. No HtoD bounce.
- **`RdmaTier`** (`expert_tier_rdma.rs`) — peer weight tier; today lands records
  via TCP `read_exact` into the same pinned arena. **Verbs is a transport swap.**

**Store format** (`expert.rs` / `expert_pack.rs`): one 4 KiB-aligned record per
`(dense_moe_layer, expert)`, versioned header carrying per-proj `weight_scale_2`
+ `input_scale`. Built offline by `atlas-expert-pack` (checkpoint → resident
store, no GPU). **All three projections are stored TRANSPOSED [K/2,N].**

**Streamer** (`spark-model/src/layers/moe/streamer.rs`): `Arc<ExpertStreamerShared>`
shared across MoE layers, with:
- a background **fetch worker thread** (owns the tier; prefetches layer L+1 while
  the GPU computes L),
- a **per-slab `CudaEvent` ring** for deferred-free slab reuse (invariant C),
- a **persistent residency cache** (WS3) so warm prefills skip re-fetch.

**Install** (`helpers_stream.rs::install_streamed_tables`): reads the transposed
`*_ptrs_t` tables into host shadows, patches local experts from the arena
residencies, writes back **one `copy_h2d` per array (9 total)** — invariant B.
Called from `forward_prefill.rs` before `run_routed_grouped_gemm`;
`after_streamed_layer` records the slab-consumed event + prefetches the next layer.

**Invariants A–F** all enforced + tested. `A` immortal tables (patch-in-place),
`B` batched 9-copy, `C` event-gated deferred-free, `D` disk==resident format,
`E` EP `local_expert_range` scoping, `F` decode-graphs-off-when-engaged
(`decode_a.rs::decode_graphs_allowed`).

---

## 4. Gotchas / hard-won learnings (READ before touching this)

1. **The router-predequant MUST run under streaming.** `predequant_for_prefill`
   FP8-dequantizes the *router* gate (`gate_nvfp4 → gate_fp8`) + the shared
   expert (both stay resident). Skipping it silently drops the router to NVFP4 →
   slightly different routing logits → NOT bit-identical. (This cost a debug cycle
   in WS1 — the perturbation was tiny, top experts matched, only ~0.25 logit
   drift.) `load_layers.rs` keeps predequant on when streaming.
2. **Projection layout is asymmetric.** gate/up are read by the fused K64 GEMM
   from the **transposed** `gate_ptrs_t/up_ptrs_t`; down is read by
   `moe_w4a16_grouped_gemm_ptrtable_n128` from the **transposed `down_ptrs_t`**
   (NOT the untransposed `down_ptrs` — that path is only taken when `down_ptrs_t`
   is `None`). The store keeps all three transposed; install patches all three
   `*_ptrs_t`. (A wrong turn in WS1 tried down-untransposed — reverted.)
3. **Resident-skip** (`load_layers.rs`): `skip_routed_load = skip_nvfp4_experts ||
   expert_streaming` routes routed experts through the existing
   `ExpertWeight::null()` no-alloc path (zero GPU bytes). The `*_ptrs_t` tables
   are still built (empty) by `transpose_for_prefill` over the null experts, then
   patched per prefill. Decode's untransposed `gate_ptrs/up_ptrs/down_ptrs`
   become NULL → decode is prefill-scoped only (§5).
4. **Build flags (single-GPU):** `--no-default-features --features cuda` (skips
   NCCL — the `spark` bin fails to link `-lnccl` otherwise). Kernels are
   per-model: `ATLAS_TARGET_MODEL=qwen3.6-35b-a3b` for AEON-7 (`qwen3_6_moe`),
   `qwen3.5-122b-a10b` for Sehyo 122B, etc.
5. **KV auto-size is greedy.** For the bit-identical gate use
   `--gpu-memory-utilization 0.55` (fits model + small KV + overlap headroom).
   0.30 is too low even for the streaming model; 0.85 is fine in isolation.
6. **O_DIRECT rejects tmpfs.** The UMA tier's O_DIRECT reads need a block-backed
   fs; tests use `CARGO_TARGET_TMPDIR` (under `target/`, ext4), not `/tmp`.
7. **`rdma-sys` won't build on aarch64** (bindgen `stddef.h` path) — so verbs
   uses a hand-rolled C-shim (`rdma_shim.c`, built with the `cc` crate). SHIPPED
   (§6). The shim's access flags are exact: server MR = REMOTE_READ only (never
   LOCAL_WRITE on the read-only mmap), client arena MR = LOCAL_WRITE only.
8. **The dump used for the gate is `atlas_final_norm.bin`** (via `ATLAS_NEMO_DUMP`),
   the lm-head input — it isolates the MoE path and is written during PREFILL
   (before decode), so decode garbage/failure doesn't affect the gate.
9. **`CudaEvent` is `Send+Sync`** (`cuda_module.rs`) so the streamer holds the
   event ring behind `Arc`; only the compute thread records/syncs them.

---

## 5. Over-core generation (decode-via-prefill) — WORKS ✅

The on-disk record is the **prefill-transposed** layout, so the scalar decode
kernels (which read the untransposed `gate_ptrs/up_ptrs/down_ptrs`, NULL under
resident-skip) can't be used. Instead, when streaming, decode routes every single
token through `forward_prefill(M=1)` — the grouped-GEMM prefill path that reads
the streamed transposed `*_ptrs_t` tables + `install_streamed_tables` from the
arena (`forward.rs`, extending the existing DFlash "Frankenstein" route to fire
whenever `self.streamer.is_some()`).

VALIDATED on A3B (`--expert-arena-layers 40`): full generation over-core, **output
IDENTICAL to resident**, 40 coherent tokens at **26.5 tok/s** (vs resident 62.4 —
the grouped GEMM + per-token table patch is ~2.4× the scalar decode path). Experts
are never held resident. Decode graphs are gated off (invariant F); the WS3
residency cache prevents per-token re-fetch. So over-core does full **generation**
on one box, not just prefill/batch — the classic trade (runs vs. needs 4 nodes).

Follow-ups (perf, not correctness): skip the per-token table patch on a warm hit;
a decode-shaped (untransposed) arena variant to use the faster scalar kernels.
Decode router hit-rate work stays behind Gate 0(a).

---

## 6. WS2 LANDED: verbs one-sided RDMA READ data path ✅

The last uncoded piece is done. `--expert-backend rdma-verbs` pulls each expert
record straight out of the peer's registered store MR into the pinned arena via
`IBV_WR_RDMA_READ`, peer CPU idle. **Bit-identical to resident** (and to the TCP
peer tier) on A3B, and ~6.6× the TCP prefill throughput (§2).

**Cabled setup (validated):** dgx-00 (192.168.178.11) ↔ gx10-9959
(192.168.178.12), device `roceP2p1s0f1`, RoCEv2 **GID index 3**, 200 Gb/s ACTIVE.

**What shipped:**
1. **C-shim** `spark-storage/src/rdma_shim.c` (compiled by `build.rs` via the `cc`
   crate, links `libibverbs`; emits `cfg(atlas_rdma_verbs)` so it's Linux+rdma-core
   only). Wraps the inline ibverbs calls (`ibv_post_send`/`ibv_poll_cq` are static
   inlines — must be called from C): `rs_create(dev, gid_idx)` (PD/CQ/RC-QP→INIT),
   `rs_reg_mr(addr, len, remote_read)`→{lkey,rkey}, `rs_qpn`/`rs_gid`,
   `rs_connect(remote_qpn, remote_psn, local_psn, remote_gid)` (RTR→RTS),
   `rs_post_read(...)`, `rs_poll`. Rust wrapper: `src/rdma_verbs.rs` (`Verbs`).
2. **Peer** `expert_peer.rs`: client sends a **transport mode byte** after the
   manifest (`MODE_TCP`|`MODE_VERBS`); one `atlas-expert-peer` serves both. Verbs
   branch `mmap`s each `experts_{l:05}.xpr` + `ibv_reg_mr` **REMOTE_READ only**
   (LOCAL_WRITE on the PROT_READ mapping fails — the one real gotcha), then
   publishes `VerbsServerParams{qpn,psn,gid, per-layer (base,rkey)}`, reads the
   client's QP, connects, and goes idle until the client hangs up.
   `remote_addr(layer,expert) = layer_base[layer] + expert*record_stride`.
   New peer flags: `--rdma-dev` / `--rdma-gid` (default `roceP2p1s0f1` / 3).
3. **Client** `expert_tier_rdma.rs`: `RdmaTier` now holds a `Transport::{Tcp,Verbs}`.
   Verbs registers the arena `LOCAL_WRITE`, exchanges QP params over the TCP
   control channel, connects the RC QP (RoCEv2, dev/gid from
   `$ATLAS_EXPERT_RDMA_DEV`/`$ATLAS_EXPERT_RDMA_GID`), and `fetch` posts a READ
   into `arena.slot_host_ptr(...)` + blocking `poll` on the prefetch worker
   thread. `residency_from` + the record header identity check still guard every
   fetch, so the tier cannot change a GEMM byte.
4. **Gate:** `scripts/streaming-experts/verify_verbs.sh` — resident vs rdma-verbs
   vs rdma-TCP final-norm compare. **PASS** (all three bit-identical).

**Reproduce (link up):**
```bash
# peer on gx10 (store already at /home/ms/expert-store-a3b):
ssh 192.168.178.12 '/home/ms/atlas-expert-peer --store /home/ms/expert-store-a3b \
  --listen 0.0.0.0:9909 --rdma-dev roceP2p1s0f1 --rdma-gid 3'
# gate on dgx-00:
ATLAS_EXPERT_PEER=192.168.178.12:9909 \
  bash scripts/streaming-experts/verify_verbs.sh "$CKPT" /home/ms/expert-store-a3b
# (set PEER_HOST=192.168.178.12 to let the script start/stop the peer over ssh)
```

**Gotchas hit (now fixed):** REMOTE_READ MR must NOT request LOCAL_WRITE on the
read-only mmap; `memlock` was already unlimited on GB10 (not the culprit). QP
bring-up worked first try once the access flags were right.

---

## 7. Build / run / reproduce

```bash
cd /home/ms/atlas/.claude/worktrees/streaming-experts-mvp   # the worktree

# Single-GPU release build for the A3B kernel target:
ATLAS_TARGET_MODEL=qwen3.6-35b-a3b \
  cargo build --release -p spark-server -p atlas-expert-pack --no-default-features --features cuda
# (spark-model builds default; spark-storage tests: --no-default-features --features metal for CPU)

# Build the resident expert store (no GPU; ~40s over NFS):
CKPT=$(ls -d /tank/hf/hub/models--AEON-7--Qwen3.6-35B-A3B-heretic-NVFP4/snapshots/*/)
./target/release/atlas-expert-pack --checkpoint "$CKPT" --out /home/ms/expert-store-a3b --verify 8

# Bit-identical gate (resident == uma == posix):
BIN=./target/release/spark bash scripts/streaming-experts/verify_logits.sh "$CKPT" /home/ms/expert-store-a3b

# Serve with streaming (uma zero-copy | posix oracle | rdma TCP peer):
./target/release/spark serve --model-from-path "$CKPT" --model-name a3b \
  --gpu-memory-utilization 0.55 --stream-experts /home/ms/expert-store-a3b \
  --expert-arena-layers 2 --expert-backend uma        # arena-layers = resident window

# RDMA peer tier (one peer binary serves BOTH transports; client picks per-conn):
./target/release/atlas-expert-peer --store /home/ms/expert-store-a3b --listen 0.0.0.0:9909 \
  --rdma-dev roceP2p1s0f1 --rdma-gid 3
ATLAS_EXPERT_PEER=<peer-ip>:9909 ./target/release/spark serve ... --expert-backend rdma        # two-sided TCP
ATLAS_EXPERT_PEER=<peer-ip>:9909 ./target/release/spark serve ... --expert-backend rdma-verbs  # one-sided RDMA READ

# Verbs bit-identical gate (resident == rdma-verbs == rdma-TCP):
ATLAS_EXPERT_PEER=<peer-ip>:9909 bash scripts/streaming-experts/verify_verbs.sh "$CKPT" /home/ms/expert-store-a3b

# Bandwidth reproducers:
nvcc -arch=sm_121 -o uma_probe docs/streaming-experts/gate0/uma_probe.cu && ./uma_probe
bash docs/streaming-experts/gate0/nvme_granule_bench.sh <dir-on-local-nvme>
bash docs/streaming-experts/gate0/verbs_rdma_read_bw.sh 192.168.178.12
```

Unit/GPU tests: `cargo test -p atlas-expert-pack`; `cargo test -p spark-storage
--test expert_stream_parity -- --ignored` (GPU + O_DIRECT fs); decode-gate +
config tests in spark-model/atlas-core.

---

## 8. File map

- `crates/spark-storage/src/` — `expert.rs` (geometry/header), `expert_pack.rs`
  (format + writer/reader + `ExpertIndex`), `expert_arena.rs` (pinned UMA arena),
  `expert_tier.rs` (trait + Posix/Uma + `open_tier`), `expert_tier_rdma.rs`
  (`RdmaTier` — `Transport::{Tcp,Verbs}`), `expert_peer.rs` (peer protocol/server
  + verbs handshake), `rdma_shim.c` + `rdma_verbs.rs` (one-sided RDMA READ FFI),
  `cuda_min.rs`/`cuda_module.rs` (FFI + `CudaEvent`).
- `crates/atlas-expert-pack/` — offline builder (`build.rs`, `transpose.rs`,
  `safetensors_min.rs`, `checkpoint.rs`) + `atlas-expert-peer` bin.
- `crates/spark-model/src/layers/moe/` — `streamer.rs` (worker + residency cache
  + slab events), `helpers_stream.rs` (install + after-layer + setter),
  `forward_prefill.rs` (call sites), `mod.rs`/`init.rs` (MoeLayer fields).
- `crates/spark-model/src/weight_loader/qwen35/load_layers.rs` — WS1 resident-skip.
- `crates/atlas-core/src/config.rs`(+`factory.rs`), `crates/spark-server/src/cli/serve_args.rs`
  (+ `main_modules/serve_phases/topology.rs`) — config/CLI plumbing.
- `crates/spark-model/src/model/trait_impl/decode_a.rs` — invariant-F gate.
- `docs/streaming-experts/` — GATE0, PHASE2, RESEARCH-RDMA-TIER, README, this file,
  `gate0/*` reproducers. `scripts/streaming-experts/verify_logits.sh` — the gate.

---

## 9. Local test assets (dgx-00)

- Expert store: `/home/ms/expert-store-a3b` (v1, 40 layers, ~17 GB).
- Checkpoints (`/tank/hf/hub/`): AEON-7 A3B-NVFP4 (validated vehicle), AgentWorld
  A3B, **Sehyo 122B-A10B-NVFP4** (the real over-core vehicle — build its store +
  `ATLAS_TARGET_MODEL=qwen3.5-122b-a10b` to demo a model that can't fit resident).

## 10. Recommended next steps

(WS2 verbs integration — §6 — is now DONE and bit-identical.)

1. **122B over-core demo** — build the Sehyo store, run at a capped arena that
   can't hold resident experts; confirm it serves (now over verbs too).
2. **Skip the ptr-patch on warm hit** — WS3 warm prefill still does the 9-copy
   patch per layer (~1.0 s vs resident 0.55 s); skipping when addresses are
   unchanged closes most of that gap.
3. **Decode-over-core** (§5) via decode-via-prefill, if generation (not just
   prefill/batch) over-core is wanted. Gated on Gate 0(a).
4. **Verbs polish (perf, not correctness):** the `rs_poll` busy-poll spins the
   prefetch thread; a batched multi-post (issue a layer's experts as one WR chain
   then poll once) and/or a completion-channel wait would cut CPU + latency.
   A persistent peer connection across prefills (currently one per serve) is
   already the case; multi-client fairness on one peer is untested.
