# Batched-Decode CUDA Graph Capture — Design Study

**Audience:** the engineer implementing batched-decode CUDA graphs for Atlas (NVFP4 / Holo first).
**Status:** design study. The capture machinery already exists and is opt-in; this document specifies how to make it correct, ship it incrementally, and extend it to the GDN/SSM long pole.
**Date of record:** 2026-06.

---

## 0. TL;DR for the implementer

Atlas already has a batched-decode graph capture path keyed by `padded_n ∈ {2,4,8}`
(`crates/spark-model/src/model/trait_impl/decode_a2.rs:218-256`,
`crates/spark-model/src/model/types.rs:82`). It is OFF by default behind
`ATLAS_DECODE_GRAPHS_MULTISEQ=1`. The **address-stability** half of the problem
is solved (fixed SSM pool, fixed metadata scratch, contiguous-slot invariant).
The work is NOT building capture — it is (a) closing three real correctness gaps
that block flipping it on, (b) validating capture+replay bit-parity on the
*existing* per-seq SSM loop before touching kernels, and (c) only then promoting
the experimental single-launch batched-recurrent GDN path into the captured
graph. **Do not gate concept validation on an experimental kernel's numerics.**

---

## 1. PROBLEM

### 1.1 Batched decode runs eager today

The single-token decode path (`n==1`, `decode_dispatch` → `decode()`) captures
and replays CUDA graphs **by default** in production. Its gate
(`decode_a.rs:156-162`) only disables graphs for EP/NCCL, profile, FP8-KV
calibration, high-speed-swap, and the step-0 dump:

```
let use_graphs = self.comm.is_none()
    && !self.profile
    && !self.suppress_graphs.load(Ordering::Relaxed)
    && !hss_engaged
    && !dump_step0;
```

The multi-sequence path (`n>=2`, `decode_batch_compute_main`, `decode_a2.rs`)
runs **eager by default**. Its gate (`decode_a2.rs:216-222`) is opt-in:

```
let ms_profile = std::env::var("ATLAS_MS_PROFILE").ok().as_deref() == Some("1");
let use_graphs = !ms_profile
    && std::env::var("ATLAS_DECODE_GRAPHS_MULTISEQ").ok().as_deref() == Some("1");
```

With the env var unset, `graph_capture` in `ForwardContext` is `false`
(`decode_a2.rs:231`), no `begin_capture`/`end_capture` is issued, and every
kernel in every layer is launched individually each decode step. The fused mixed
prefill+decode path (`mixed_forward`) hardwires `graph_capture: false`
(`decode_b.rs:362-382`) — it never captures at all.

### 1.2 The concurrency-scaling cost

The code itself quantifies the overhead: `decode_a2.rs:213-215` states that
graphs for `n>=2` would eliminate **~1500 kernel launches/step** and calls this
"the dominant lever for n>=2 decode." Per attention layer the eager path pays
roughly `1 (norm) + 2-3 QKV + ~5n copies + n rope + n cache-write + n Q-copies +
optional WHT(3) + 1 paged-decode + (n or 1) o-proj + FFN`. Per SSM layer it pays
a per-seq recurrent loop (`rms_norm_residual + ssm_forward[multiple] +
residual_add_rms_norm`) × n plus MoE. Multiplied across ~36 hybrid layers in
qwen3.5-122b, that is the ~1500 figure.

The observable symptom is the **flat NVFP4 ~50 tok/s curve from C=1 to C=8**:
concurrency does not amortize because each decode step re-pays full per-launch
overhead. The SSM comment is explicit (`trait_decode_multi_seq.rs:179-181`):
"The real fix for this launch overhead is CUDA graphs for n>=2, not MoE batching
(graphs capture these per-token launches for free)." Per the memory notes
(`holo-vllm-parity-poc`, `holo-decode-batching`), decode does not amortize and
SSM dominates the per-step cost; CUDA graphs off = pure launch overhead is the
lever.

### 1.3 Proof of concept: the working single-seq slot-keyed graph

The `n==1` path is the existence proof that Atlas's decode is graph-capturable.
Its cache is `decode_graph: Mutex<HashMap<usize, GraphHandle>>` keyed by
`seq.slot_idx` (`types.rs:80`). Replay path: `decode_a.rs:194-202` — on a slot
hit with a non-null handle it calls `launch_graph` and returns immediately.
Capture path: `decode_a.rs:204-354` — on a miss it `begin_capture`s, runs the
full layer loop + final norm + LM head, `end_capture`s, inserts by slot, and
launches once.

The reason this works for **any** sequence length / block count is documented
verbatim at `decode_a.rs:188-193`: `max_blocks_per_seq` is baked as a scalar but
only used as a `block_table` stride multiplier (`seq_idx * stride`), and with
`batch=1` `seq_idx=0` so the stride is multiplied by zero and never matters. All
per-step-varying data (positions, slot, seq_len, block_table) is uploaded into a
**fixed** device scratch region (`meta_base = scratch + 32768`, `decode_a.rs:97-130`)
each step and read fresh on replay. This invariant — fixed device addresses +
per-step device uploads — is exactly what `n>=2` must preserve.

---

## 2. WHAT'S STATIC VS DYNAMIC

### 2.1 Per-step varying quantities (the inputs)

For each active sequence, `decode_batch_compute_main` re-derives every step
(`decode_a2.rs:137-198,251-254`; `impl_b1.rs:57-96`; `decode_step.rs:20`):

1. **token to embed** — `tokens[i] = a.last_token` (`decode_step.rs:20`).
2. **per-seq position** = `seq.seq_len` (`impl_b1.rs:58`), incremented each step
   (`decode_a2.rs:253`).
3. **per-seq KV slot** = `physical_block_for(pos/bs)*bs + pos%bs`
   (`impl_b1.rs:61-67`) — moves to a new physical block at block boundaries.
4. **per-seq seq_len(+1)** as i32 (`impl_b1.rs:69`).
5. **per-seq block_table contents** = `seq.block_table` (`impl_b1.rs:84-86`) —
   grows as `ensure_blocks_through_decode` allocates (`decode_a2.rs:185-195`).
6. **per-seq block count** = `seq.block_table.len()` — grows over the sequence's life.
7. **per-seq SSM pool slot** = `seq.slot_idx` → which `h_state`/`conv_state`
   addresses are used.
8. **batch composition** — `N` changes as sequences retire
   (`retire_finished_sequences`) and new ones are admitted; continuous batching
   may fuse prefill chunks via `mixed_forward_batch`.

### 2.2 Already device-resident (a graph reads them on replay — NOT recapture-forcing)

`upload_batch_metadata_fixed` (`impl_b1.rs:34-124`) writes positions, KV slots,
seq_lens(+1), and the flattened block_table into the **same fixed device
address** every step: `meta_base = scratch + 32768`, sub-arrays at +0/+256/+512/+768
(`impl_b1.rs:98-124`). These are `copy_h2d_async`'d **before** replay so captured
kernels dereference `meta_base` and see the current step's values. The block_table
is uploaded with a **constant stride** `self.max_blocks_per_seq` (`impl_b1.rs:43,84-85`)
so the per-row offset baked at capture stays valid as block_tables grow.

Embed targets (`hidden[i*h]`) and all scratch/activation buffers
(`gate_logits`, `expert_gate_out/up/down`, `qkv/attn_output`, `ssm_*`,
`splitk_workspace`, `logits`, `fp8_act`) are **persistent fixed-address**
allocations — `BufferArena::new` allocates each once sized for
`max_batch_tokens` and reuses them (`buffers.rs:15-25,84-160,162-258`).

SSM `h_state`/`conv_state` live in a fixed-base contiguous pool addressed as
`pool_base + slot*stride` (`ssm_pool.rs:29-44,205-226`). Because active
sequences are kept in contiguous slots `[0..N)`, position `i`'s state is always
at the same address, so even the recurrent-state pointers baked into kernel args
stay correct across replays.

**Net:** items 1–7 above (positions, KV slots, seq_lens, block_table
contents/counts, SSM-state pointers) are all handled by the upload-to-fixed-address
pattern and do **NOT** force recapture.

### 2.3 The crux: batch SIZE is the one quantity that drives launch geometry

`KernelLaunch` bakes `grid:[u32;3]` / `block:[u32;3]` as host values at launch
time and forwards them straight into `cuLaunchKernel`
(`kernel_args.rs:51-93,154-183`; `gpu.rs:124-132`; `gpu_impl.rs:172-193`). There
is **no** device-memory grid sizing anywhere. Ops compute grid from a host-side
count: `rms_norm` `grid=[num_tokens,1,1]` (`norm.rs:21-39`), `silu_mul`
`grid=[div_ceil(num_elements,256),1,1]` (`activations.rs:30-31`), etc.

The per-layer batched kernels take the sequence count as a grid/loop dimension:
`decode_multi_seq` is called with `padded_n` as `num_seqs`
(`decode_a2.rs:353-363`); SSM batched kernels take `n` as a `u32` launch param
(`ssm_batched.rs:102,143,184,248,287,319`); the LM-head GEMM and final RMS-norm
use `padded_n` as the row count (`decode_a2.rs:388-445`);
`AttnMetadataDev.num_seqs = padded_n` (`impl_b1.rs:123`). Everything else (head
count, hidden size, block stride, vocab) is a model constant.

**Conclusion:** batch SIZE changes the launch shape and therefore forces a
distinct captured graph. Everything else is already device-read and free. This
is the entire design premise.

### 2.4 The existing answer: bucketing by `padded_n`

Batch size is bucketed: `padded_n = first of [2,4,8] that is >= n`
(`decode_a2.rs:168`). `n==1` is special-cased to the single-seq `decode()` graph
(`decode_a2.rs:40-44`). The graph cache `batch_decode_graphs` is keyed by
`padded_n` (`types.rs:82`; lookup `decode_a2.rs:242-243`; insert
`decode_a2.rs:462-468`). Padding positions `[n..padded_n)` get a dummy KV block,
position 0, seq_len 1, and the dedicated dummy SSM slot
(`ssm_pool.dummy_slot()`, `ssm_pool.rs:205-207`) so pad kernels can't corrupt
live state (`impl_b1.rs:89-96`; `decode_a2.rs:177-180,284-304`).

### 2.5 Scheduler invariants that make this sound

Two scheduler-maintained invariants are load-bearing:

- **Contiguous slots.** `retire_finished_sequences` compacts survivors into SSM
  pool slots `[0..N)` (`mod_helpers.rs:169-216`, two-phase claim-then-migrate).
  `free_slots = (0..max_slots).rev().collect()` (`ssm_pool.rs:127`) is a LIFO
  popping the **lowest** index first, so combined with `compact_sequence` the
  invariant is structurally enforced, not merely empirical (the `decode_a2.rs:204-209`
  comment under-states this as "verified empirically").
- **Constant block_table stride** `self.max_blocks_per_seq` baked at capture
  (`impl_b1.rs:43`).

Under EP v2 slots are pre-allocated and NOT compacted; that path uses slot-keyed
caches and is out of scope here (graphs disabled under EP regardless — see §6).

---

## 3. THE DESIGN

### 3.1 Recommended approach (from the architect's judgment)

**Batch-bucket graph pool + persistent device input buffers + padded/masked
lanes, keyed by bucket.** This is the existing `batch_decode_graphs` mechanism,
hardened and validated. Concretely:

- A fixed set of buckets `{2,4,8}` (already `decode_a2.rs:168`). Pad `n` up to
  the nearest bucket; the unused lanes `[n..padded_n)` are masked by routing them
  to the dummy SSM slot + dummy KV block.
- Persistent device input buffers: the `meta_base` metadata region
  (`impl_b1.rs:98-124`) and all `BufferArena` activations
  (`buffers.rs:84-160`) are written each step **before** replay; the captured
  graph reads them by baked pointer.
- One `GraphHandle` per bucket in `batch_decode_graphs` (`types.rs:82`); replay
  on bucket hit (`decode_a2.rs:242-248`), capture on miss
  (`decode_a2.rs:257-468`).

We explicitly **reject** the MAX-grid + device-resident `active_count` early-exit
alternative for v1: a grep for `num_active`/`active_count`/device-resident count
yields **zero** hits — no kernel reads an active count from device memory today
(`map:buffers-launch (d)`). Adding that read touches every kernel and is a much
larger, separate program. Bucketing amortizes with a small discrete graph set
and is already in the tree.

### 3.2 Decision-freeze discipline (grafted from the canonical design)

Capture must not contain a capture-illegal *decision*. Resolve every `env::var`
read and every `kernel_handle.0 != 0` / `*_ptrs.is_some()` capability branch on
the decode hot path into bools/handles **before** `begin_capture`. Confirmed
in-window decisions that must be hoisted before Phase 3:

- `ATLAS_SSM_BATCHED_RECURRENT`, `ATLAS_GDN_FUSED_NORM`, and
  `gdn_f32_strided_k.0 != 0` read inside the batched-recurrent hot path
  (`ssm_batched_recurrent.rs:60,77-79,132-133`).
- `forward_k2` branches on `self.bf16_gate_weight_ptrs.is_some()`
  (`moe/forward_k2.rs`). For Holo NVFP4 this resolves to the non-bf16 path and is
  stable per model — but it is a decision-in-window and must be frozen before
  capture, not evaluated inside it.

In slice 1 the per-seq SSM inner loop is already free of these, so the freeze is
cheap insurance there; bake the **pattern** now because Phase 3 is impossible
without it.

### 3.3 Cache-key typing (grafted, deferred to Phase 2)

Adopt an explicit `GraphKey { bucket: usize, kind: GraphKind }` typing of
`batch_decode_graphs` (`types.rs:82`) at the first refactor touch so pure decode,
MTP, and verify (`kind: MixedVerify{K}`) don't fight over a bare `usize` later.
Defer the actual retype until Phase 2 (do not do it in slice 1).

### 3.4 Concrete capture/replay mechanism and file:lines

| Concern | File:lines | Action |
|---|---|---|
| Replay on bucket hit | `decode_a2.rs:242-256` | unchanged; `launch_graph` if `graph.0 != 0`, then bump seq state, return |
| Capture on miss | `decode_a2.rs:257-468` | `begin_capture` (`decode_a2.rs:306-308`) → layer loop → final norm + LM head → `end_capture` (`decode_a2.rs:462-471`) → insert by `padded_n` |
| Restrict slice-1 capture to bucket 2 | `decode_a2.rs:462-468` | only insert/launch graph when `padded_n == 2`; leave `{4,8}` eager |
| Metadata upload (fixed addr) | `impl_b1.rs:34-124` | unchanged; runs each step before replay |
| Backend capture API | `gpu_impl.rs:294-344` | unchanged; `begin_capture` uses `CU_STREAM_CAPTURE_MODE_RELAXED=2` |
| Capture-validity flag | `decode_a2.rs:224-233`; `layer.rs:217` | `ForwardContext.graph_capture = use_graphs`; layers consult it to skip syncs |

---

## 4. THE GDN/SSM LONG POLE

### 4.1 Why it dominates

SSM/GDN layers are ~half the model and per the memory notes
(`holo-vllm-parity-poc`: "SSM 65%") are the dominant per-step decode cost. The
hybrid decode is graph-hostile for two reasons (`map:gdn-ssm-batched`):

1. A **per-seq Rust loop** in `decode_multi_seq` reads each sequence's state
   pointer from its `SsmLayerState` and emits `N` kernel launches whose count and
   pointers vary with batch size (`trait_decode_multi_seq.rs:79-122`). The
   recurrent inner (BA/gates, conv1d, GDN recurrence, gated-norm) stays a per-seq
   loop of byte-identical single-token kernels.
2. **num_tokens-dependent K∈{2,3,4,17} dispatch** with env-gated branches in the
   MTP/verify path (`trait_decode_batched_conv_gdn.rs:50-57,91-426`): each K arm
   issues a different set/count of kernels, so a graph captured for one K cannot
   replay for another.

### 4.2 What `ssm_pool.rs` already did (the address half)

`ssm_pool.rs` delivered the address-stability prerequisite, its explicit design
goal (`ssm_pool.rs:29-33`):

- **Fixed contiguous per-layer pools** with deterministic per-slot offsets
  (`h_state(i,slot)=pool[i].offset(slot*h_bytes)`, `conv_state` likewise),
  one alloc per SSM layer at init, memset 0 (`ssm_pool.rs:88-127,205-226`). Same
  GPU addresses reused every step → graph-safe.
- **Zero-on-claim + RAII `SlotGuard`** free-list guaranteeing exactly-once slot
  release on every exit path (normal/abort/error/panic/migrate)
  (`ssm_pool.rs:78-127`; field doc `types.rs:83-89`) — prevents two sequences
  sharing a slot, which would corrupt state during replay.
- **Reserved dummy pad slot** at index `max_slots` (`ssm_pool.rs:205-207`) so
  pad lanes never alias a live slot.
- **MTP intermediate/checkpoint pools** pre-allocated at fixed addresses for
  K=2/3/4 verify stability.
- **`compact_sequence`/`copy_slot`/`claim_specific`** to keep active slots
  contiguous — the invariant the strided batched kernels require.

### 4.3 What's still missing (the launch-structure half)

The pool fixed the addresses; the **launch graph is not yet static**
(`map:gdn-ssm-batched (c cont.)`):

- The default SSM mixer is still a per-seq loop emitting `N` variable launches.
  The genuinely single-launch batched path
  (`dense_gemm_ba_gates_prefill` + batched `conv1d_update_l2norm` with `batch=n`
  via `blockIdx.y` + `gdn_decode_f32_strided[_norm]`) **exists** but is gated
  behind `ATLAS_SSM_BATCHED_RECURRENT=1` AND a runtime contiguous-slot check AND
  `gdn_f32_strided_k` present (`ssm_batched_recurrent.rs:60-89,132-208`) —
  experimental/opt-in, not the captured path.
- The strided kernel **hard-codes contiguity**: `H = h_state +
  ((b*num_v_heads+vh)*k_dim*v_dim)`, `b=blockIdx.y` (`gated_delta_rule.cu:676-720`);
  batch-aware conv indexes `conv_state + (b*dim+ch)*d_conv`
  (`causal_conv1d.cu:96-170`). Valid only for contiguous slots.
- MTP verify has no single-launch batched-over-tokens GDN for general K — one
  captured graph per K bucket would be needed.
- Env/capability decisions resolve at runtime inside the hot path — must be
  hoisted (§3.2).

### 4.4 Decode strategy: branch-free fixed-K, per-seq loop captured first

The key insight from the judgment: for a **fixed `padded_n`**, the per-seq SSM
loop has a **fixed launch count** (`N` iterations of byte-identical kernels). It
is therefore already capturable as-is — capturing it eliminates the per-launch
overhead (Win A) **without** promoting the experimental batched-recurrent kernel
(Win B). So slice 1 captures the trusted per-seq loop; the single-launch batched
path is a later occupancy upgrade gated on bit-parity proof. The contiguity
requirement becomes a **capture-time precondition** (refuse to capture and fall
back to eager if unmet), not a per-step branch.

---

## 5. PHASED PLAN (NVFP4-FIRST)

### Phase 0 — Gate: prove eager `n==2` NVFP4 is already bit-parity (BLOCKS EVERYTHING)

No behavior change. With `ATLAS_DECODE_GRAPHS_MULTISEQ=1`, `ATLAS_MS_PROFILE`
unset, capture disabled, run `scripts/longctx_needle.py` + `ATLAS_CONC_HSD` and
confirm eager `n==2` logits match the `n==1` reference. The known `pos>=1`
divergence diagnostic must be shown absent (or a pre-existing eager bug) **before**
capture is allowed to inherit it. If this fails, no capture design is shippable
until it's fixed.

### Milestone 1 (SHIP) — capture+replay `padded_n==2` NVFP4 Holo decode, per-seq SSM loop, env-gated

The smallest shippable unit. One new variable: *does capture/replay preserve
numerics?*

- Restrict capture/insert to `padded_n==2` only (`decode_a2.rs:462-468`); leave
  `{4,8}` eager.
- Keep the **per-seq SSM mixer loop** (`ATLAS_SSM_BATCHED_RECURRENT` OFF). **No
  kernel work.** Attention + FFN + the existing per-seq SSM loop are all captured;
  SSM stays eager-shaped *inside* the captured graph (its launches are recorded,
  not skipped). Assessment: SSM does **not** need to stay outside the graph —
  the per-seq loop has a fixed launch count at fixed `padded_n`, so it captures
  cleanly. Leaving it eager (outside capture) is unnecessary and would forfeit
  most of Win A since SSM is ~65% of launches.
- Add the three mandatory gates (§6.3).
- **Acceptance:** captured `n==2` logits **bit-identical** to eager `n==2` over a
  needle run, plus a low-concurrency `soak-holo-atlas-64k.py` run showing the
  launch-overhead win with zero correctness regressions. **No default flip.**

### Phase 2 — widen buckets to `{4,8}`

Capture/insert for `padded_n ∈ {4,8}`. Retype `batch_decode_graphs` to
`GraphKey { bucket, kind }` (§3.3) at this touch. Re-run bit-parity at each
bucket. Note: the NVFP4-batched QKV fast path only exists at exactly `n==2`/`n==3`
(`qkv.rs:39-89`); `padded_n==4/8` fall to the per-token sequential GEMV loop —
that's fine, it's still capturable, just lower occupancy (a Phase-3-adjacent
kernel opportunity, not a blocker).

### Phase 3 — SSM occupancy upgrade: batched-recurrent as the captured path

Promote `gdn_decode_f32_strided[_norm]` + batched conv into the captured graph.
Preconditions resolved in the decision-freeze **before** `begin_capture`
(refuse-capture-and-fall-back-to-eager if unmet):

- contiguity check (slots `[0..n)`),
- `gdn_f32_strided_k.0 != 0` and fused-norm kernel present,
- env decisions hoisted.

Gate this phase on **bit-parity of the strided kernel vs `ssm_forward`** proven
*independently* of capture (the maps flag it "still experimental"). Per
`holo-gdn-wmma-vblock`, the wmma+DV-block variant is bit-parity + 1.4×@batch2
isolated — that is the candidate, but prove parity standalone first.

### Phase 4 — generalize to FP8 / BF16, then default-on

Extend capture to FP8 (`w8a16_gemv`) and BF16 (`dense_gemv`/`dense_gemm`) decode
shapes. Freeze the `bf16_gate_weight_ptrs.is_some()` decision per model (§3.2).
Only after soak across dtypes flip `ATLAS_DECODE_GRAPHS_MULTISEQ` default to on.
Per `holo-fp8-kv-recommend`, FP8 KV is the recommended default — sequence the FP8
capture validation alongside that.

**Deferred indefinitely (reject for v1):** MAX-grid + device `active_count`
early-exit; MTP/verify batched-K capture; mixed prefill+decode capture
(`mixed_forward_batch` — varies on both `N` decode rows and `M` prefill streams,
the harder bucketing problem, `map:dynamic-state`).

---

## 6. SCOPE + RISK

### 6.1 Files / subsystems touched

- `crates/spark-model/src/model/trait_impl/decode_a2.rs` — capture/replay,
  `use_graphs` gate fix (~218), bucket restriction (~462-468), slot assert.
- `crates/spark-model/src/model/trait_impl/sequence.rs:194` — drain on retire
  (exists); mirror onto admit path.
- `crates/spark-model/src/model/trait_impl/meta.rs:~90` — add admit-side drain.
- `crates/spark-model/src/layers/qwen3_ssm/trait_decode_multi_seq/ssm_batched_recurrent.rs:60-89,132-133`
  — in-window env/capability reads to hoist (Phase 3).
- `crates/spark-model/src/model/ssm_pool.rs:127,205` — slot LIFO + dummy_slot
  (foundation, read-only).
- `crates/spark-model/src/model/types.rs:82` — cache, eventual `GraphKey` retype
  (Phase 2).
- Backend `gpu_impl.rs:294-344` — unchanged (capture API is complete).

### 6.2 Overhaul magnitude (honest)

**Slice 1: S** (~30–80 LoC; reuses all capture machinery, no kernel work).
**Full program through Phase 4: L–XL.** Phase 3 alone (batched-recurrent
promotion, decision-freeze, contiguity preconditions, kernel bit-parity) is the
bulk and is genuinely major. The canonical-layer refactor (GraphKey, n=4/8
kernels) is ~500–700 LoC if pursued. Be honest with stakeholders: shipping the
*concept* is small; shipping the *full SSM occupancy win + default-on across
dtypes* is a large multi-phase effort. The maps' "address half done, launch half
is the dominant remaining cost" is the accurate framing.

### 6.3 Correctness risks and de-risking

**Graph-capture illegal ops** (cause `cuStreamSynchronize` → status 900
CAPTURE_UNSUPPORTED): per-step sync, host allocations, D2H copies, host-dependent
control flow. `ForwardContext.graph_capture` (`decode_a2.rs:231`; `layer.rs:217`)
is the hook layers use to skip these; `ssm_forward.rs:22-25` already guards debug
syncs with `!ctx.graph_capture`. Any new in-window decision (§3.2) is a hazard.

**Three slice-1-blocking gaps the existing path does NOT handle** (verified):

1. **`use_graphs` under-gated.** `decode_a2.rs:218` derives `use_graphs` from
   **only** `!ms_profile && env=="1"`. It does NOT check `comm.is_none()` or
   `suppress_graphs`, unlike `decode_a.rs:156`. "Safe" today only because the env
   is off. **Fix:** gate on `self.comm.is_none() && !self.suppress_graphs.load(...)`
   — mirror `decode_a.rs:156`. Without this, EP and FP8-KV-calibration steps
   attempt illegal capture (NCCL all-reduce / forced-eager calibration) inside
   `begin_capture` the moment the path is exercised broadly.
2. **No drain on admit.** `batch_decode_graphs` is drained in `free_sequence`
   (retire, `sequence.rs:194-198`) but **not** when a new sequence is admitted. A
   newly admitted sequence claims the lowest free slot and joins the active set,
   changing the composition a captured graph baked, with no teardown. **Fix:**
   add the wholesale drain to the admit/`alloc_sequence` path
   (`meta.rs:~90`).
3. **Empirical → enforced slot invariant.** Upgrade the `decode_a2.rs:204-209`
   comment into a `debug_assert!(seq.slot_idx == i)` for all active `i` before
   `begin_capture`.

**De-risking — the oracle.** The eager `n==2` path (Phase 0) and the trusted
`n==1` single-seq graph are the oracles. The standing acceptance rule for every
phase: **bit-compare graphed vs eager logits** over a `longctx_needle.py` run
before any default change. Capture only records the launches the eager path
already issues, so any divergence localizes to a capture-illegal op (the
graph-capture hazards above), not to numerics — making failures diagnosable.

---

## 7. OPEN QUESTIONS FOR THE IMPLEMENTER

1. **Phase 0 outcome:** does eager `n==2` NVFP4 already match `n==1` bit-for-bit,
   or is there a pre-existing `pos>=1` divergence? If the latter, is it a paged-decode
   batching bug independent of capture, and must it be fixed first?
2. **Admit-side drain cost:** draining `batch_decode_graphs` on every admit means
   re-capturing whenever a sequence joins. Under high churn does the recapture cost
   (instantiate via `cuGraphInstantiateWithFlags`) dominate the replay savings? If
   so, do we need a generation counter / lazy-invalidate instead of eager drain?
3. **Bucket coverage:** are `{2,4,8}` the right buckets for Holo's concurrency
   profile, or do we need `{16}`? What is the cap at `max_batch_tokens`?
4. **SSM-inside-graph vs eager-outside:** confirm (measure) that capturing the
   per-seq SSM loop is bit-identical and that leaving it eager would forfeit the
   majority of Win A (SSM ~65% of launches). The plan assumes capture-inside.
5. **EP path:** EP v2 doesn't compact slots (non-contiguous `slot_idx`). Is
   batched-decode-graphs simply off under EP (the §6.3.1 gate), or is a
   slot-keyed batched cache wanted later?
6. **Phase 3 kernel parity:** does `gdn_decode_f32_strided_norm` (or the
   wmma+DV-block `tc_vblock` variant) pass standalone bit-parity vs `ssm_forward`
   at `n∈{2,4,8}`? This gates the entire occupancy upgrade.
7. **GraphKey timing:** retype at Phase 2, or earlier if MTP/verify capture is
   pulled forward?
8. **Mixed prefill+decode:** is fused-mixed-step capture ever in scope, given it
   needs 2-D bucketing on `(N decode, M prefill streams)`?

---

## Appendix — files of record

| Path | Role |
|---|---|
| `crates/spark-model/src/model/trait_impl/decode_a.rs:156-162,188-212,338-354` | n==1 proof-of-concept (gate, capture, slot-keyed replay) |
| `crates/spark-model/src/model/trait_impl/decode_a2.rs:168,216-256,257-468` | batched capture/replay; the two gating fixes + slot assert |
| `crates/spark-model/src/model/impl_b1.rs:34-124` | `upload_batch_metadata_fixed` (fixed-addr metadata) |
| `crates/spark-model/src/model/types.rs:80,82` | `decode_graph` / `batch_decode_graphs` caches |
| `crates/spark-model/src/model/ssm_pool.rs:29-44,88-127,205-226` | fixed SSM pool, dummy slot, LIFO free-list |
| `crates/spark-model/src/model/trait_impl/sequence.rs:194-198` | retire-side drain (mirror onto admit) |
| `crates/spark-model/src/layers/qwen3_ssm/trait_decode_multi_seq.rs:79-122,179-181` | per-seq SSM loop + the "graphs are the fix" comment |
| `crates/spark-model/src/layers/qwen3_ssm/trait_decode_multi_seq/ssm_batched_recurrent.rs:60-89,132-208` | experimental single-launch batched path (Phase 3) |
| `crates/spark-runtime/src/cuda_backend/gpu_impl.rs:294-344` | begin/end/launch/destroy graph API |
| `crates/spark-runtime/src/buffers.rs:84-160` | persistent fixed-address activation buffers |
| `crates/spark-server/src/scheduler/mod_helpers.rs:169-216` | contiguous-slot compaction invariant |

---

## EMPIRICAL FINDINGS (2026-06-29, NVFP4 Qwythos-9B, GB10, varlen+correctness bench)

Systematic A/B of the batching levers (C=1/2/4/8, agg tok/s, speedup vs C=1, correctness probes):

| Config | C=2 | C=4 | C=8 | correct@C8 |
|---|---|---|---|---|
| eager (no levers) | 1.38x | 1.29x | 1.39x | — |
| graphs only (`DECODE_GRAPHS_MULTISEQ`) | 1.38x | 1.28x | 1.40x | 4/4 ✓ |
| graphs + FFN-batched (dense n≥4, committed) | 1.40x | 1.30x | 1.47x | 4/4 ✓ |
| graphs + `SSM_BATCHED_RECURRENT` | 1.39x | 1.30x | 1.36x | **0/3 @C4** ✗ |

Conclusions:
1. **The slot-sort fix (committed) was a prerequisite** — without it the active list is reverse-slot-order, contiguity fails, and the batched paths silently fall back to per-seq.
2. **FFN-batched decode (committed) is correct but a small win** (1.40→1.47x). The FFN is ~half the weights but NOT the decode-time bottleneck.
3. **The SSM/GDN per-seq loop is the dominant ceiling (~65% of decode cost).** It re-reads the ~1.6B SSM projection weights (in_proj_qkvz/ba, out_proj across 24 layers) n times per step.
4. **The existing `ssm_batched_recurrent` is a dead end**: it CORRUPTS at n≥4 (0/3 correct) AND gives ZERO scaling even when correct at n=2 (1.39x = per-seq). It is not the path.
5. CUDA graphs alone only remove launch overhead; they cannot break a per-seq *bandwidth* ceiling.

**The real lever for vLLM-like scaling:** batch the *projection GEMMs* (read weights once per step) the way the committed FFN fix does — extend to (a) attention q/k/v/o on the 8 full-attn layers, and (b) the SSM in_proj_qkvz/ba + out_proj on the 24 linear-attn layers — while leaving the cheap per-seq recurrent scan alone. The SSM-projection batching is the dominant remaining bandwidth win and the largest restructure (ssm_forward splits into batched-in_proj → per-seq scan → batched-out_proj). The broken batched-recurrent scan fusion is a separate, lower-value concern.

Honest magnitude: each projection-batching increment is M (~50-150 LoC, mirrors the FFN pattern); the SSM restructure is the bulk. Full vLLM-parity scaling on this GDN-heavy model is a multi-step kernel/dispatch effort, not a single flag.

## PREFILL vs DECODE split (2026-06-29, graphs+FFN-batched, varlen+large-prefill probes)

| C | prefill tok/s | pf speedup | decode tok/s | dec speedup | TTFT ms | correct |
|---|---|---|---|---|---|---|
| 1 | 63  | 1.00x | 63.7 | 1.00x | 350  | 1/1 |
| 2 | 120 | 1.91x | 72.7 | 1.14x | 400  | 2/2 |
| 4 | 137 | 2.18x | 47.1 | 0.74x | 674  | 3/3 |
| 8 | 607 | 9.67x | 52.7 | 0.83x | 1376 | 4/4 |

**PREFILL ALREADY SCALES (~9.7x @C8)** — the batched prefill GEMMs + ATLAS_PREFILL_VARLEN path work; this is much of vLLM's concurrency win and we have it. **DECODE does NOT scale (0.74–0.83x, regressing)** — the per-seq projection-GEMV bandwidth ceiling: each added sequence re-reads ~1.6B SSM + 0.56B attn weights. Graphs+FFN-batching can't break it (SSM 65% + attn still per-seq GEMV). The goal reduces to: **batch the DECODE-path projection GEMMs (SSM in_proj_qkvz/ba + out_proj, attn qkv/o) into M=n GEMMs**, the same pattern the committed dense-FFN fix uses. Decode metric is noisy under varlen (wall ≈ longest request) but direction is unambiguous and matches the architecture.

## DECODE-SCALING A/B — uniform-length probe (2026-06-29, conv-fix applied)

Uniform 200-tok generations + uniform prefill so all C sequences decode concurrently for the WHOLE window (the varlen bench masks this: short requests finish early → concurrency collapses to ~1).

| C | A: FFN-batched | B: FFN + SSM-batched (conv-fix) |
|---|---|---|
| 1 | 44.4 (1.00x) | 44.2 (1.00x) |
| 2 | 54.3 (1.22x) | 55.7 (1.26x) |
| 4 | 52.8 (1.19x) | 53.7 (1.22x) |
| 8 | 63.3 (1.42x) | 65.3 (1.48x) |

**Conclusion (systematic, evidence-based):**
1. The **conv input-stride fix restores correctness** of `ATLAS_SSM_BATCHED_RECURRENT` (was 0/3 at C=4 → now 3/3 at C=4, 4/4 at C=8, 0 errors). This is a real, committed bug fix — the flag was previously unusable.
2. But batching the dominant SSM projections lifts decode scaling only 1.42x → 1.48x at C=8. **Batched-GDN is confirmed a "dead lever" for decode scaling** (matches the prior holo-vllm-parity-poc decomposition).
3. **Why:** transformer decode scales with batching because it's weight-bandwidth-bound (FFN/attn projections amortize across the batch as M=n GEMMs). A GDN/linear-attention model's decode is dominated by the **recurrent scan** — per-seq O(num_v_heads·k_dim·v_dim) state work that is genuinely n× at batch n and does NOT amortize. Per-seq throughput collapses 44→8 tok/s (1→8) in BOTH configs, confirming the work scales ~linearly with n regardless of projection batching.
4. **vLLM-like concurrency for this model therefore lives in PREFILL (already ~9.7x), not decode.** Decode is architecturally capped near ~1.5x with all batch levers correct.

**Remaining decode lever** (per holo-vllm-parity-poc): SSM *vertical fusion* — fuse the per-token SSM op chain (ba/gates→conv→gdn→norm→out_proj) to cut launch + intermediate-memory overhead. That lowers per-token latency (helps C=1 AND aggregate), but it is latency work, not batch-scaling. It is the honest next step if decode tok/s is the target.

## PREFILL CORRECTION — large-context probe (2026-06-29, Holo-35B NVFP4, in cuda13.2 container)

The earlier "prefill scales 9.7–13×" was a METHODOLOGY ARTIFACT (heterogeneous tiny prompts 16–320 tok; latency-bound; slow tiny-prompt C=1 baseline). Re-measured at REAL contexts:

C=1 by workload: text-7k (9916 tok) 1836 tok/s / 5.4s TTFT; text-11k (16125 tok) 1839 tok/s / 8.77s; image (268 tok) 438 tok/s / 611ms; image+text (10174 tok) 1807 tok/s / 5.63s.

Prefill concurrency @ 11k: C1 1840 → C2 1852 → C4 1852 → C8 1845 tok/s = **FLAT (1.00–1.01×)**; mean TTFT grows LINEARLY 8.8s→17.4s→34.8s→69.9s.

**At realistic contexts prefill is COMPUTE-saturated by a single stream — concurrency gives zero throughput gain and linear TTFT growth (expected; vLLM is the same for large prefills).** The prefill lever is SINGLE-STREAM throughput (~1840 tok/s in-container), NOT concurrency: i.e. the cuda13.2 container ~6× penalty ([[holo-container-perf-gap]]) + prefill GEMM/GDN efficiency ([[holo-cuda132-container-plan]]). Image prefill is token-cheap (vision encoder) and adds negligible cost atop text.

## PREFILL RE-CORRECTION (2026-06-29) — the 1840 was a HARNESS bug, real is ~3700 tok/s

The "1840 tok/s flat" was a MEASUREMENT ARTIFACT in my probe: it sent each workload probe back-to-back with NO gap, so a large prefill QUEUED behind the previous request's still-running decode → client TTFT ~doubled. Binary/flags/container were NOT the cause (GPU 0% idle between reqs, no contention). Authoritative server-side TTFT (lifecycle.rs "Done: ... TTFT=") with SPACED single requests (image's own cutlass/flashinfer binary, full prod flags, prefix-cache on):
- 3k (4389 tok): server TTFT 1161ms → ~3780 tok/s
- 7k (9909 tok): 2581ms → ~3840 tok/s
- 11k (16120 tok): 4465ms → ~3610 tok/s

Clean large-context prefill CONCURRENCY (drained between levels) @ 11k: C1 3489 → C2 3545 → C4 3559 → C8 3566 tok/s = still FLAT (1.02×), mean TTFT linear (4.6s→36s). So: single-stream prefill ~3700 tok/s; large-context concurrency saturates one stream (flat agg, linear TTFT) — expected, matches vLLM for large prefills. LESSON: always space/drain between latency probes, and prefer server-side TTFT (lifecycle log) over client wall-time.

## ATTENTION-PROJECTION TILING + FINAL DECODE VERDICT (2026-06-29, Holo-35B)

Per-phase decode @C8 (ATLAS_MS_PROFILE, eager): total~64ms = **ssm 72.5% (30L) / attn 15.5% (10L) / head 12%**.

Implemented BOTH remaining attention-projection levers (commits on branch): n>=4 qkv + o_proj now TILE the proven batch3/batch2 GEMV kernels (n=8 -> 3+3+2 = 3 weight reads vs 8), byte-identical layout to ms_qkv_batch3 → correct (3/4 probes, 0 errors, no corruption). Decode @C8:
- baseline (conv-fix SSM-batched only): 1.93x
- + qkv-tiling: 1.96x
- + qkv+oproj-tiling: 1.92x
All within run-to-run noise (±0.04x). **The attention tiling is correct but does NOT meaningfully move decode** — under CUDA graphs the per-seq launch overhead was already gone, and projection weight-reads (8->3) are tiny vs the SSM wall.

**FINAL VERDICT:** decode @C8 ceiling ~1.96x, firmly SSM-bound (72.5%, the per-seq recurrent scan is genuinely n× — same property as the FLA/FlashInfer GDN kernel vLLM uses, per holo-vllm-parity-poc: "B=8 = 5.74× B=1, state-bandwidth bound"). Every tractable lever is now implemented + validated: conv-fix SSM-batched (the real win, 1.76->1.93 on MoE), qkv-tiling, o_proj-tiling. Crossing 2x cleanly requires a fundamentally different GDN decode kernel (chunked/wmma — [[holo-gdn-wmma-vblock]], [[holo-gdn-chunk-rfc-ref]]), a multi-week kernel-research effort, NOT a tuning lever. 1.96x is vLLM-parity-class for a linear-attention hybrid (vLLM does not exceed ~2x on GDN decode either).
