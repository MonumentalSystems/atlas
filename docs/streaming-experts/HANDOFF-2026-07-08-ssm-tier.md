# SSM-snapshot tier + WS-A/B — Session Handoff (2026-07-08)

Branch `feat/streaming-experts-mvp` (PR #9). All work committed + pushed. Pick-up
point for a fresh session to blast out the remaining greenlit tasks. See memory:
`ssm-snapshot-nvme-tier-followup.md`, `streaming-experts.md`,
`ssm-decode-rollback-rightsizing.md`, `atlas-kv-peer-service.md`.

## What shipped this session (context)

1. **3 tier-fault-in bugs fixed** → warm 15K-context agentic turns 6,750 → ~1,480 ms:
   - `47a1f4f` session-tag: `free()` didn't clear `session_tags`, so a
     spilled-then-reacquired slot kept the victim's tag → `session_matches`
     rejected → the tier restore NEVER completed. Fix: `free` clears the tag +
     fault-in re-tags (`ssm_snapshot.rs`, `prefix_lookup.rs`).
   - `2a17dd7` skip-point: the Marconi skip read resident `ssm_snapshot_tokens`
     (=0 on a fault-in); use `eff_snapshot_tokens` → the restore now elides the
     forward pass instead of re-running SSM over the full prefix.
   - `c0e64c6` + `cfba063` the checkpoint ladder (tuned, see below).

2. **WS-A: disk-backed SHARED paging tier (infinite depth)** — atlas-cache-peer is
   now a paging server: RDMA arena = page-cache over an O_DIRECT NVMe swap, peer
   owns residency (`key→{Reserved|Resident|OnDisk}`+LRU), never drops.
   `snapshot_swap.rs` (core, 19 unit tests), `cache_peer.rs` (control channel +
   registry), `rdma_snapshot.rs` (client `connect_paging`/`paging_put/get`),
   `ssm_tier.rs` (`PagingSnapshotStore`). Cross-connection sharing + per-model
   namespace fold + 50 GiB disk cap. **Live-proven** on full Holo-3.1-35B.

3. **Item 8: per-(kind,shape) arena REGISTRY** — `LazyLock<Mutex<HashMap<(kind,blob),
   SharedPaging>>>`; versioned handshake (v1→SSM byte-exact, v2→`[kind]` byte,
   legacy KV→dumb path, kind≥2→reject); disk-cap hard-ceiling carve; v1
   wire-golden test. *RO tiers (experts/weights/lora) are OUT of scope — they need
   a manifest+VerbsServerParams reply, not CacheServerParams.*

4. **Ladder tuned → tail-clustered** (`cfba063`): rungs at `last-i*bs` (i=1..N),
   just below the tail where warm matches land. **A block = 16 tokens**
   (`kv_cache.block_size()`). SWEEP (serialized ssm_deep, holo-p10):
   `N=0`→6 miss/3253ms cold; **`N=2`→0 miss/4282ms** (SWEET SPOT); N=3→4638ms;
   N=4→5333ms; old even-spread N=8→~7400ms. Matches land 2 blocks (32 tok) below
   the tail → **use `ATLAS_SSM_PREFILL_CHECKPOINTS=2`** (dial per template tightness
   via the miss-depth histogram). Default N=0 = byte-identical.

## Correctly NOT built (don't re-chase)

- **RDMA speedup (inc 6/7 pipeline + zero-copy UMA)** — abandoned. The 0.1 GB/s
  was a SMOKE-HARNESS artifact; the MODEL fault is 2.5 GB/s (`RDMA read=26,480us`
  for 66 MB) = **1.6% of the ~1600 ms warm turn**. UMA premise was false (KV
  zero-copy uses the same `cuMemAllocHost` pinned host memory). The striped
  pipeline (`1bda3d9`, flag `ATLAS_SSM_STAGING`, default-off) is a harmless
  foundation; leave it. **The RDMA fault is NOT a bottleneck.**

## OPEN GREENLIT TASKS (blast these out)

### DONE (2026-07-08, commit 635cbcf + GPU-validated on holo-p11)
- **#5 — cost-aware fault-vs-recompute gate.** `ATLAS_SSM_FAULT_MIN_TOKENS`
  (default 256, 0=disabled): below it a matched prefix is shallow enough that
  recompute beats the fixed ~28 ms blob fault+replay. Lives in the shared
  `trait_impl/ssm_fault_in.rs` helper so it applies to all three prefill paths.
- **#6 — wire `prefill_a`/`prefill_c` to the tier.** Both previously ignored
  `ssm_snapshot_tier_key` → recompute; now fault a spilled anchor back to a
  resident slot and fold via `eff_snapshot`/`eff_snapshot_tokens`, exactly like
  `prefill_b` (extracted into `try_fault_in_ssm_snapshot`). Also hardened
  `prefill_a`'s exact-match skip (only skip whole prompt when `snap_tok >=
  matched`) to match the prefill_b/c intermediate-checkpoint fix.
  - **GPU evidence (Holo-3.1-35B-A3B-NVFP4, host-RAM tier, ladder N=2, ssm_deep
    6×3):** `--ssm-cache-slots 4` → 103 spills, 12 tier fault-ins RESTORED via
    the helper, 12 intermediate hits, 18/18 coherent, 0 errors (proves 4
    slots/0.25 GB + tier ≈ 256 slots/16 GB). Gate flip
    `ATLAS_SSM_FAULT_MIN_TOKENS=20000` → 12 SKIPs, 0 fault-ins, 12 recompute
    fallbacks, 18/18 coherent. Workload harness: `scripts/streaming-experts/ssm_deep.py`.

### SSM-snapshot / prefill path
- **#3 — warm-TTFT (the real latency lever, ~1600 ms turn).** MEASURED: a warm
  turn = `484ms` (embed+lookup over 15K) + `532ms` (tail-cut pass 1 + checkpoint)
  + `538ms` (pass 2 + finalize). The **2 tail-cut passes = 69%**, each ~530 ms
  for ~30 tokens. NOT expert-streaming (experts aren't streamed — no
  `--stream-experts`; `ATLAS_HOLO_LOW_MEMORY_MOE=1` uses load-time prefill copies).
  Prime suspect: **attention — each suffix token attends over the full ~15K KV**,
  ×10 attn layers, ×2 passes. NEXT: add gated per-op timing
  (`ATLAS_PREFILL_PROFILE`) around embed / prefix-lookup / per-layer attn / MoE /
  finalize; rebuild; serialized warm run; read the breakdown. Then target: the
  attention-over-long-context kernel, or a one-pass+mid-chunk-checkpoint
  restructure (blocked today: layers process a whole chunk at once → can't
  snapshot SSM state mid-chunk). Part of the 484ms is `acquire_or_spill_slot`
  spilling a 66 MB victim + faulting in on the warm path.
- **#5 — cost-aware fault-vs-recompute gate.** Don't fault-in (28 ms + replay)
  when a shallow prefix is cheaper to recompute. Hook: `prefix_lookup.rs`
  fault-in decision (`ssm_snapshot_tier_key` present) — add a depth guard.
- **#6 — wire `prefill_a`/`prefill_c` to the tier.** Only `prefill_b` faults in;
  `prefill_a`/`prefill_c` ignore the tier key → recompute. Mirror
  `prefill_b/prefix_lookup.rs:123-163`.

### KV / over-core cluster (separate subsystem from the SSM tier)
- **#9 — RDMA-tier C=8 over-core KV measurement.** Inc 1+2+3 batched framework is
  landed + bitwise-validated; run the real over-core-thesis test (RDMA KV tier via
  `ATLAS_KV_PEER=gx10:9916`, C=8 overflow decode) vs the NVMe-read-bound result.
  Realistic flags (`--scheduling-policy slai`, `--ssm-slots 256`, 32K), NOT the
  pathological 128-tok window.
- **#10 — zero-copy RDMA KV as default.** Already wired (`ATLAS_KV_ZERO_COPY=1`,
  logs "zero-copy restore live"); flip default + measure. (REAL for KV — lands
  into UMA scratch, unlike the SSM path.)
- **#11 — prefetch-overlap + CudaEvent coexistence.** Needed before Phase-3
  prefetch combines with batched decode (currently serial-falls-back under
  `ATLAS_KV_PREFETCH`). Add a main-stream event waited on `prefetch_stream` +
  `if !reqs.is_empty()` empty-read guard.
- **#12 — decode-rollback ring right-sizing.** 4 GB ring (16× Marconi) =
  `DECODE_ROLLBACK_RING_SLOTS(8) × max_batch_size × 63.75MB`, scales with batch.
  Lazy per-active-seq alloc or fewer slots. See `ssm-decode-rollback-rightsizing.md`.
  NOT tier work (hot ephemeral, stays HBM).
- **#13 — graphs-on production tok/s.** Every number so far is a profiling floor.
  Gotcha: spark serve wedges on SIGTERM → `kill -9` measurement servers.

### Resident Marconi pool shrink — VALIDATED on the real model (2026-07-08)
- **Proven live on Holo-3.1-35B:** `--ssm-cache-slots 16` + `ATLAS_SSM_TIER=1`
  (host-RAM spill, no peer needed) boots and serves — `SSM snapshot pool: Marconi
  16 slots (1020 MB)` vs baseline `256 slots (16320 MB)` = **~15.3 GB HBM
  reclaimed** (per-slot 63.75 MB × 30 SSM layers), spill tier `ENABLED
  (66846720 bytes/snapshot)`, warm repeated-prefix recall intact. `MemBlobStore`
  cap 0 = unbounded, so `ATLAS_SSM_TIER=1` alone gives an infinite-depth host-RAM
  tier — the RDMA peer (`ATLAS_SSM_RDMA_TIER=…:9920`) is an optional upgrade, not
  required for the shrink. Preflight now emits an INFO "SSM pool right-sizing"
  hint when a large pool runs with the tier off (`ssm_pool_shrink_hint`,
  unit-tested); `--ssm-cache-slots` docstring updated with the tier caveat.
  NOTE: the decode-rollback ring (4080 MB here) is separate (#12) and does NOT
  shrink with this.

### Resident Marconi pool shrink (UNBLOCKED — eviction pin is live)
- With the tier + the GET→RDMA-read eviction pin deployed, the resident Marconi
  pool is a hot cache in front of the infinite-depth spill tier — it no longer
  needs to hold every live conversation's whole checkpoint chain (PR #278's
  reason for `--ssm-cache-slots 256` / 16 GB). **Guidance: run a small resident
  pool + tier**, e.g. `--ssm-cache-slots 8–32` (0.5–2 GB) with
  `ATLAS_SSM_TIER=1` (+ `ATLAS_SSM_SWAP=1 ATLAS_SSM_RDMA_TIER=…:9920` for the
  shared RDMA tier), reclaiming ~14–15 GB HBM. GPU-proven: a 4-slot pool + tier
  reproduced the 256-slot hit behavior coherently (18/18, 12 fault-ins). The #5
  cost gate (`ATLAS_SSM_FAULT_MIN_TOKENS`) keeps a shrunk pool from faulting
  shallow prefixes cheaper to recompute. **Multi-model is now safe** (the
  eviction pin closed the concurrent-ALLOC race). Size the pool to
  ~checkpoints-restored-per-warm-turn + concurrency headroom, not chain×sessions.

### WS-A loose ends (before multi-tenant / to finish item 8)
- **GET→RDMA-read eviction pin.** `run_paging_loop_shared` releases the residency
  Mutex before the client's one-sided RDMA-READ; a concurrent ALLOC (another
  client, same geometry) could evict+reuse the slot mid-read → torn restore. PUT
  is safe (Reserved excluded from LRU); GET is not. Add `OP_RELEASE` / a
  read-pinned `Loc` that `evict_coldest_to_disk` skips. **Single-model safe today;
  gate concurrent multi-model until this lands.**
- **Retire the legacy dumb one-sided path** once KV migrates to paging (finishes
  #8). Stays only because `RdmaKvBackend` (KV overflow) shares `cache_peer`.
  **Scoped:** `KV-PAGING-MIGRATION.md` (2026-07-08) — confirmed the legacy path is
  LIVE (KV's only transport, the path just benchmarked), so this is a real
  client rewrite of `RdmaKvBackend` onto the paging `OP_*` protocol + reconciling
  KV group-stride ↔ paging-slot addressing, NOT a dead-code delete. Full file:line
  checklist + risk in that doc. Peer is now unattended-safe on the legacy path via
  `--max-blade-gb` regardless, so migration is capability (disk spill/shared arena),
  not safety.
- **Per-kind `--swap-cap-gb-<kind>` overrides + explicit memlock ceiling** for the
  multi-arena registry (`RdmaConfig.max_blade_bytes` default is unlimited).
- **Deploy the registry binary to gx10:9920** — systemd peer is still the
  pre-registry binary (`/home/ms/atlas-cache-peer-paging`). Redeploy when a
  registry-consuming client lands. Backward-compatible so no rush.

## Environment / recipes (fresh session)

- **Infra:** gx10 (mgmt ssh `ms@192.168.1.177`, data path `192.168.178.12`).
  - `:9916` — production unified peer, systemd `atlas-cache-peer`. DON'T disturb.
  - `:9920` — WS-A PAGING peer, systemd `atlas-cache-peer-paging`, `--swap-dir
    /tank/atlas-ssm-swap --swap-cap-gb 50`, dual-rail, enabled (survives reboot).
- **Container:** `atlas-gb10:holo-p10` = latest w/ ALL fixes. Rebuild:
  `docker build -f docker/gb10/Dockerfile.builder --build-arg
  ATLAS_TARGET_MODEL=holo-3.1-35b-a3b --build-arg ATLAS_TARGET_QUANT=nvfp4 -t
  atlas-gb10:holo-pN .` (~15 min).
- **Container serve needs:** `--network host --ipc=host --device /dev/infiniband
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 -v
  /home/ms/.cache/huggingface:/root/.cache/huggingface`.
- **SSM tier flags:** `ATLAS_SSM_TIER=1` (host-RAM default); `+ATLAS_SSM_SWAP=1
  ATLAS_SSM_RDMA_TIER=192.168.178.12:9920` (paging peer);
  `ATLAS_SSM_PREFILL_CHECKPOINTS=2` (tuned); `ATLAS_SSM_TAIL_PROTECT=1`;
  diagnostics `ATLAS_SSM_SNAP_STATS=1` / `ATLAS_SSM_TIER_TIMING=1`.
- **Workload:** `ssm_deep.py` (6 sessions ×3 turns deep-prefix; `reps 800`≈15K,
  `1600`≈32K). Paging smoke: `cargo run -p spark-storage --features cuda --example
  snapshot_paging_smoke` (ATLAS_SNAP_PEER=host:port).
- **GOTCHA:** host smoke shows 0.1 GB/s RDMA (artifact); trust the model's
  `ATLAS_SSM_TIER_TIMING` logs (2.5 GB/s).

## The meta-lesson (bank it)
This session's wins came from **measuring before implementing** — caught THREE
wrong premises (RDMA UMA relayout, expert-residency, "skip the 2nd pass") before
they cost a wasted cross-crate change, by reading one comment / checking one
config / dissecting the phase timeline. When a fix hinges on an unverified cause,
profile first.
