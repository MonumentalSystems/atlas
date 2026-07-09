# Codex handoff — #3 warm-TTFT (the real latency lever)

**Self-contained brief. You have no prior context — everything you need is here.**
Read top to bottom before touching code.

---

## 0. The task in one sentence

A **warm** conversational turn on Holo-3.1-35B (a prompt that hits an existing
deep prefix cache, so almost nothing should recompute) still spends most TTFT in
two "tail-cut" prefill passes over a tiny suffix. Older notes measured this at
**~1600 ms**; the current verified resident-cache baseline is lower but the same
shape remains: **~350-380 ms warm TTFT**, with **~240 ms in attention** across
the two tiny suffix passes. **Keep cutting that.**

**This is a measure-first task.** The profiling scaffolding is already landed.
Your **first deliverable is a confirmed breakdown**, not a code change. Do not
start optimizing until the profile tells you which op dominates.

**Codex status, 2026-07-09:** baseline tagged as
`warm-ttft-baseline-2026-07-08` at `4f9f4e`. The original 6-session repro below
can self-evict with `--target-kv-tokens 100000`; use the one-session command in
§4 for resident-cache profiling. A low-risk BR64 dispatch experiment is now
available via `ATLAS_PREFILL_PAGED_BR64_MIN_Q=1`; on the one-session Holo/NVFP4
run it cut warm TTFT from **376/354 ms** to **328/301 ms** by reducing attention
from **~120 ms/pass** to **~94 ms/pass**. This is not yet the final fix.

---

## 1. Repo / branch / where you are

- Repo: `github.com:MonumentalSystems/atlas` (origin). Also has `avarok` remote
  (`Avarok-Cybersecurity/atlas`) — ignore it for this task.
- Work off branch **`feat/streaming-experts-mvp`**. Branch from it; open a draft
  PR against it (not `main`).
- **Work in a git worktree** so you don't collide with other sessions.
- Hardware: this must run on the **dgx-00 GB10 (sm_121)** box where the repo
  lives. It is a shared machine — see the hard rules (§6).

---

## 2. Current understanding (measured, not guessed)

A warm turn originally decomposed (measured on a ~15K-token deep prefix) as:

| Phase | Time | Notes |
|---|---|---|
| embed + prefix-lookup over 15K | ~484 ms | part is `acquire_or_spill_slot` spilling a 66 MB victim + faulting in on the warm path |
| tail-cut **pass 1** + checkpoint | ~532 ms | ~30 suffix tokens |
| tail-cut **pass 2** + finalize | ~538 ms | ~30 suffix tokens |

The current one-session, reps-1600 profile is:

| Config | Warm TTFT | Tail pass attention | Tail pass SSM | Notes |
|---|---:|---:|---:|---|
| default BR32 for `q_len < 256` | 376 ms / 354 ms | ~120 ms/pass | ~48-51 ms/pass | resident cache, 30K prompt |
| `ATLAS_PREFILL_PAGED_BR64_MIN_Q=1` | 328 ms / 301 ms | ~94 ms/pass | ~48-51 ms/pass | same workload |

The two tail-cut passes remain the turn's main cost. The cost scales with the
*prefix* length because each suffix token attends over the full paged KV.

**Prime suspect: attention.** Each suffix token attends over the **full ~15K KV**,
× the attention layers, × 2 passes. If the per-layer profile shows the attention
layers dominating, that confirms it. The candidate fixes in §5 follow from that.

**Rule out up front — do NOT chase these:**
- **Not expert-streaming.** Experts are *not* streamed in this config (there is no
  `--stream-experts`; `ATLAS_HOLO_LOW_MEMORY_MOE=1` uses load-time prefill copies).
  Attention and SSM layers carry the *same* MoE FFN, so a per-mixer time split
  isolates the mixer, not the FFN.
- **Not RDMA snapshot fault.** An RDMA snapshot fault is ~2.5 GB/s ≈ 26 ms ≈ 1.6%
  of the turn. It is not the bottleneck. The tail-prefill is.

---

## 3. The profiling that is ALREADY landed (use it first)

There are **two separate signals**. You will want **both on at once**.

### (a) Per-phase breakdown — env `ATLAS_PREFILL_PROFILE=1`
- Code: `crates/spark-model/src/model/trait_impl/prefill_b.rs` (~line 200+, the
  `phase!` macro; emit ~line 428).
- Low perturbation (~8 phase-boundary syncs/chunk).
- Grep the serve log for: **`PREFILL_PROFILE chunk[start=.. len=.. proc=.. last=..]: X.Xms phases | embed=.. lookup=.. forward=.. finalize=..`**
- This tells you which *phase* (embed / lookup+restore / forward / finalize+save)
  owns the wall.

### (b) Per-layer + attn-vs-SSM mixer split — CLI `--profile`
- Code: `crates/spark-model/src/model/trait_impl/prefill_b/forward_layers.rs`
  (~line 260+). Gated by `self.profile` (the `--profile` CLI flag) **AND**
  `is_last_chunk && proc_count > 1 && (chunk_start + chunk_len) > 16384`.
- Higher perturbation (~40 per-layer syncs) — expected, it's a diagnostic.
- Grep the serve log for: **`mixer split: attn X.Xms (N layers, ..ms/layer) vs ssm Y.Yms (M layers, ..ms/layer)`** and the `top5: L..=..ms` per-layer hotspots.
- **Only fires on the last chunk of a >16K-token warm turn with proc_count>1** —
  so your workload must build a prefix > 16384 tokens (see §4, use `reps 1600`).

> To see the mixer split you need **`--profile` on the CLI** *and* a warm turn
> whose prefix exceeds 16K tokens. `ATLAS_PREFILL_PROFILE=1` alone gives only the
> phase breakdown (a), not the mixer split (b). Turn on both.

---

## 4. How to build, serve, and reproduce

### Build (docker — do NOT use host `cargo` for the serve binary)
Builds for glibc/hardware parity take ~15 min:
```
docker build -f docker/gb10/Dockerfile.builder \
  --build-arg ATLAS_TARGET_MODEL=holo-3.1-35b-a3b \
  --build-arg ATLAS_TARGET_QUANT=nvfp4 \
  -t atlas-gb10:warmttft .
```
(You may still use host `cargo test`/`cargo build` for fast unit-test iteration on
non-CUDA logic — but the served GPU binary comes from docker.)

### Serve (Holo-3.1-35B-NVFP4, warm-TTFT profiling ON)
Model weights are already on-box at `/home/ms/.cache/huggingface` (and `/tank/hf`).
```
docker run -d --name warmttft --gpus all --network host --ipc=host \
  --security-opt seccomp=docker/gb10/seccomp-io_uring.json \
  --ulimit memlock=-1 --cap-add=SYS_NICE \
  -v /home/ms/.cache/huggingface:/root/.cache/huggingface \
  -e ATLAS_HOLO_NATIVE_FP8_ATTN=1 -e ATLAS_HOLO_NATIVE_FP8_SSM=1 \
  -e ATLAS_CUTLASS_NVFP4_GEMM=1 -e ATLAS_CUTLASS_NVFP4_SSM_OUT=1 \
  -e ATLAS_GDN_FLASHINFER=1 -e ATLAS_SSM_TIER=1 -e ATLAS_SSM_TAIL_PROTECT=1 \
  -e ATLAS_KV_OVERCOMMIT=1 -e ATLAS_FAST_LOAD_PREFETCH_SHARDS=1 \
  -e ATLAS_PREFILL_PROFILE=1 \
  -e ATLAS_TARGET_MODEL=holo-3.1-35b-a3b -e RUST_LOG=info \
  --entrypoint bash atlas-gb10:warmttft -c \
  "spark serve Hcompany/Holo-3.1-35B-A3B-NVFP4 --bind 0.0.0.0 --port 8888 \
   --profile --enable-prefix-caching --ssm-cache-slots 256 \
   --scheduling-policy slai --tbt-deadline-ms 100 \
   --max-seq-len 100000 --max-batch-size 8 --max-num-seqs 8 \
   --max-prefill-tokens 16384 --kv-cache-dtype fp8 \
   --fp8-kv-calibration-tokens 256 --target-kv-tokens 100000"
```
Wait for `curl -sf http://127.0.0.1:8888/v1/models` to return the model id.
For the BR64 suffix-attention experiment, add
`-e ATLAS_PREFILL_PAGED_BR64_MIN_Q=1` to the `docker run` env list.

### Drive a deep-prefix WARM turn (>16K tokens so the mixer split fires)
```
python3 scripts/streaming-experts/ssm_deep.py \
  http://127.0.0.1:8888/v1/chat/completions 1600 \
  --sessions 1 --turns 3 --max-tokens 24 --target-kv-tokens 100000
```
`ssm_deep.py <url> <reps>` defaults to 6 sessions × 3 turns of deep-prefix
context. For warm-TTFT profiling with `--target-kv-tokens 100000`, force
`--sessions 1`: six reps-1600 sessions are roughly 180K live prompt tokens and
can evict each other before warm turns, producing a cold trace with zero prefix
hits. `reps 800` ≈ 15K tokens, `reps 1600` ≈ 30K. Turn 2/3 are the **warm** turns
when `cached_tokens` is nonzero. Run it **serialized** (one request in flight) so
the timings are clean.

### Read the breakdown
```
docker logs warmttft 2>&1 | grep -E "PREFILL_PROFILE|mixer split|top5"
```
Confirm: which phase dominates (a), and whether attn ≫ ssm per-layer (b).
Also confirm the driver printed nonzero `cached=` on post-t0 turns and
`atlas_prefix_cache_hits_total` increased. If not, you profiled a cold run.

---

## 5. Where to attack (once the profile confirms the cause)

Two candidate fixes; the profile decides which:

1. **If attention-over-long-context dominates** (confirmed): the ~30-token suffix
   attends the full paged KV every layer, twice. Target the suffix-attention
   kernel — it should attend the long prefix KV far more cheaply (the suffix is
   tiny; the cost is all in the K/V length). This is the highest-leverage fix.
   First cheap lever: test/lower `ATLAS_PREFILL_PAGED_BR64_MIN_Q` for NVFP4
   suffixes. It improves Holo/NVFP4 but needs a broader sweep before becoming a
   production default.

2. **One-pass + mid-chunk checkpoint restructure.** The turn runs **two** tail-cut
   passes. Collapsing to one would ~halve the tail cost. **Blocked today** because
   layers process a whole chunk at once, so SSM state can't be snapshotted
   *mid-chunk* — the two-pass structure exists to get a checkpoint at the cut
   point. Making mid-chunk SSM checkpointing possible unblocks the single pass.

3. **Secondary:** part of the 484 ms embed/lookup phase is `acquire_or_spill_slot`
   spilling a 66 MB victim + faulting in on the warm path. If phase (a) shows
   lookup dominating instead of forward, chase that instead.

Code entry points for all of the above:
- Prefill orchestration + phases: `crates/spark-model/src/model/trait_impl/prefill_b.rs`
- Per-layer forward + mixer timing: `.../prefill_b/forward_layers.rs`
- Prefix lookup / fault-in / slot spill: `.../prefill_b/prefix_lookup.rs`
- Attention layer: `crates/spark-model/src/layers/qwen3_attention/` (prefill path
  under `trait_impl/prefill_inner.rs`, `prefill/`)

---

## 6. Hard rules (violating these has burned people — do not)

- **NEVER `RUST_LOG=warn`.** Always `RUST_LOG=info`. The operator watches server
  logs live; `warn` hides everything and makes them furious. Every serve command
  here already sets `info` — keep it.
- **NEVER use `--speculative`.** It has never been used on these models; Holo-35B
  reports "No MTP weights found." Not part of this task.
- **Do not run `cargo fmt`** (repo is not fmt-enforced; a crate-wide fmt churns
  dozens of files). Format by hand; keep diffs to the lines you intend to change.
- **Do not `pkill` broadly on this box.** It is shared. Scope kills to your own
  container (`docker rm -f warmttft`) / your own PIDs. A broad `pkill` once killed
  another user's server.
- **Do not touch the RDMA peer on `gx10:9916`** (systemd `atlas-cache-peer`) — it
  is a production daemon. This task does not need it (`ATLAS_SSM_TIER=1` uses
  host-RAM, no peer).
- **Build the served binary in docker, not host `cargo`** (glibc/hardware parity).
- **`spark serve` wedges on SIGTERM** — stop it with `docker rm -f <name>` (or
  `kill -9` a bare process), not a graceful signal.
- **Add unit tests** alongside any fix and run `cargo test` before committing.
  The cut-point/boundary math in `prefill_b.rs` is deliberately split out to be
  unit-testable without a GPU — extend those tests.
- **No fudged numbers.** Report measured values or "n/a". Never invent a figure.

---

## 7. Success criteria

1. A posted **confirmed breakdown** of the warm turn from `PREFILL_PROFILE` +
   `mixer split` logs (this alone is valuable — it validates or refutes the
   attention hypothesis).
2. A change that measurably cuts warm-turn TTFT, with **before/after** numbers
   from the same resident-cache `ssm_deep.py reps 1600 --sessions 1` serialized
   workload, and unit tests.
3. No regression in correctness — a warm turn must produce the same tokens as a
   cold turn for the same prompt (coherence check).

---

## 8. Related reading (optional, in-repo)

- `docs/streaming-experts/HANDOFF-2026-07-08-ssm-tier.md` — the broader SSM-tier
  backlog; the `#3 — warm-TTFT` bullet is the parent of this doc. `#5` and `#6`
  (cost-aware fault-vs-recompute gate; wiring `prefill_a`/`prefill_c` to the tier)
  are adjacent but separate.
- `docs/streaming-experts/DECODE-RING-ROLLING-TIER.md`, `KV-PAGING-MIGRATION.md` —
  context on the tiering machinery, not required for #3.
