# Plan: High-Speed-Swap subsumes classic whole-sequence preempt (`--swap-space-gb`)

Author: research/planning pass (no source changed).
Target: Atlas inference engine, single GB10 (sm_121), Rust + CUDA.
Goal: let `--high-speed-swap` (HSS, mechanism B) absorb the whole-sequence
evict/restore role of classic `--swap-space-gb` (mechanism A) so classic swap
can eventually be deprecated and Atlas keeps ONE swap subsystem. The
scheduler's admission **policy** (which victim, when) stays; only the
**mechanism** (where evicted blocks go, how they come back) migrates.

---

## 1. CLASSIC swap mechanism (mechanism A) — end-to-end, with evidence

### 1.1 Where it lives
- CLI flag: `crates/spark-server/src/cli/serve_args.rs:367-371`
  (`--swap-space-gb`, default **3**, files under `/tmp/atlas-swap/`).
- File lifecycle manager: `crates/spark-runtime/src/kv_spill.rs`
  (`KvSpillManager`): `create_file`/`open_file`/`remove_file`/`record_usage`/
  `has_space`, sequential `swap_{id}.bin` files, byte budget, stale-file
  cleanup on `new()` and `Drop`.
- Serialization (model-owned): `crates/spark-model/src/model/trait_impl/sequence/state_io.rs`
  (`save_sequence_state_dispatch` / `restore_sequence_state_dispatch`), exposed
  on the `Model` trait at `crates/spark-model/src/traits/model.rs:862` (`save_sequence_state`)
  and `:873` (`restore_sequence_state`).
- Scheduler orchestration:
  `crates/spark-server/src/scheduler/lifecycle.rs:143` (`swap_out_sequence`),
  `:246` (`resume_swapped_seq`); trigger + resume loops in
  `crates/spark-server/src/scheduler/mod.rs`.
- The swapped-out CPU-side record: `SwappedSeq` with `swap_id: u64` at
  `crates/spark-server/src/scheduler/types.rs:386-471`.

### 1.2 Trigger conditions
- **Swap-out** (`mod.rs:385-417`): before starting each newly drained request,
  while `model.num_free_blocks() < blocks_needed` and `active` is non-empty,
  the scheduler picks a **victim** = the active seq with the largest
  `block_table.len()` **excluding grammar-active seqs** (`grammar_state.is_none()`
  filter — grammar state is not serializable, `lifecycle.rs:348-349`). If all
  active seqs are grammar-active it logs "No swappable sequences" and breaks.
  `blocks_needed = prompt_len / block_size + 1` (`mod.rs:388-389`).
- **Swap-in** (`mod.rs:664-689`): after `retire_finished_sequences`, while
  `swapped` non-empty and `active.len() < max_batch_size`, it resumes the first
  swapped seq whose `num_blocks <= model.num_free_blocks()`.
- A **separate**, unrelated preempt path exists for SSM-pool exhaustion:
  `phase_start_prefills.rs:330-347` (`handle_prefill_start_error`) — on a
  `"pool exhausted"` error it **drops** the newest active seq and returns a
  503-equivalent error (`send_error`, no disk save). This is the "hard"
  preempt; classic swap is the "soft" (restorable) preempt. **Both are
  in-scope conceptually but only the swap path saves/restores state.**

### 1.3 On-disk format and data path (swap-out)
`save_sequence_state_dispatch` (`state_io.rs:30-74`):
1. Under the `kv_cache` lock, for each `block_idx` in `seq.block_table`, for
   each layer, `kv.read_block(layer, block, gpu)` → host `(k_data, v_data)`
   (device→host copy).
2. Lock released; write all `(k,v)` byte buffers to the writer in
   block-major, layer-minor order.
3. For every `LinearAttention` (SSM) layer, `copy_d2h` the `h_state`
   (`ssm_pool.h_bytes`) and `conv_state` (`conv_bytes`) and append them.
So the file = `[blocks × layers × (K,V)]` then `[ssm_layers × (h,conv)]`.
Granularity: **whole sequence**, full history, D2H→`/tmp` file.

### 1.4 Block-table free / re-alloc (the mechanism to migrate)
`swap_out_sequence` (`lifecycle.rs:143-243`):
- `active.swap_remove(victim_idx)` → owns `a`.
- **Slot compaction**: if the seq swapped into `victim_idx` now sits at a
  `slot_idx != victim_idx`, `model.compact_sequence(&mut active[victim_idx], victim_idx)`
  migrates its SSM pool slot down, then `model.detach_slot_for_reuse(&mut a.seq)`
  neutralizes the victim's RAII slot guard so an early `?`-return can't
  double-release the migrated slot (`traits/model.rs:465,476`).
- `spill.create_file()` → `save_sequence_state` → `record_usage`.
- `model.free_sequence(&mut a.seq)` (`trait_impl/sequence.rs`): frees KV blocks
  (`kv_cache.free_blocks(&block_table)`, `:153`), releases the SSM pool slot,
  dec_disk_refs any HSS disk ids (`:164-179`), clears SSM buffer refs.
- `ep_broadcast_cmd_for_seq(slot, 0xFFFFFFF1)` tells EP workers to
  free+realloc their mirror.
- Returns `SwappedSeq` capturing `tokens`, `seq_len`, `num_blocks`, all
  sampler/thinking/tool CPU state, and `swap_id`.

`resume_swapped_seq` (`lifecycle.rs:246-361`):
- `model.alloc_sequence()` → **fresh** `SequenceState` (new slot, possibly a
  **different** `slot_idx` than before swap-out).
- `spill.open_file(swap_id)` → `restore_sequence_state(seq, num_blocks, reader)`
  (`state_io.rs:76-139`): reads all `(k,v)` into host bufs, then under the lock
  `kv.alloc_block()` **num_blocks** fresh physical blocks, `kv.write_block`
  each (H2D), builds a **new** `block_table`; then reads+`copy_h2d` the SSM
  `h/conv` state into the new slot's SSM buffers.
- `spill.remove_file(swap_id)`; re-acquire adapter slot (`:269`); rebuild an
  `ActiveSeq`. **Grammar state, cancel_flag, decode-rollback SSM ring are NOT
  restored** (documented lossy fields, `lifecycle.rs:286,330-335,348`).

### 1.5 Scheduler state transitions
Three vectors in `run()` (`mod.rs:290-292`): `active`, `prefilling`,
`swapped: Vec<SwappedSeq>`. Swap-out moves `active → swapped`; swap-in moves
`swapped → active`. LoRA rotations are gated on `swapped.is_empty()`
(`mod.rs:324`) because a spilled seq has released its adapter ref (#25); a
rotation while a seq is spilled could corrupt the slot its KV was computed
under. Shutdown drains `active` (finish) and deletes remaining swap files
(`mod.rs:708-712`).

### 1.6 CUDA-graph / slot implications
- `slot_idx` is **reused/reassigned**: swap-out frees the slot; swap-in
  `alloc_sequence` may hand back a different slot. Single-seq decode CUDA
  graphs are keyed by slot and destroyed on `free_sequence`
  (memory: `holo-decode-graph-recapture-323`), so a resumed seq re-captures.
  This is already tolerated by the classic path.
- Granularity is whole-sequence; no intra-sequence streaming.

---

## 2. HIGH-SPEED-SWAP (mechanism B) — architecture and maturity

### 2.1 What it is
An **intra-sequence, block-level KV streaming** subsystem. Instead of freeing a
whole sequence, it keeps a **rolling HBM window** of the most recent
`cache_blocks_per_seq` blocks and streams the older blocks to per-layer NVMe
files, reading them back into a transient HBM **scratch pool** only for the
duration of an attention kernel. Throughput ~3.4 GB/s (io_uring, QD=8).

### 2.2 Data model (`crates/spark-storage/src/high_speed_swap.rs` + `.../high_speed_swap/impl_more.rs`)
- `HighSpeedSwap` orchestrator (`high_speed_swap.rs:31-51`): owns `Predictor`,
  `ScratchPool`, `TierBackend` (io_uring/POSIX), `TiledAttention`,
  `EvictionPolicy`, reusable device scratch, and `DiskState`.
- `DiskState` (`:53-68`): **global** disk-block-id allocator — `next_id`,
  `free_list`, `refcount: Vec<u32>`. A `disk_block_id` indexes the **same
  logical slot in every layer's file** (layer-agnostic). API:
  `alloc_disk_block_id() -> Option<u32>` (`:154`, returns `None` at capacity =
  `max_blocks_per_layer`), `inc_disk_ref` (`:169`), `dec_disk_ref -> new_rc`
  (`:177`, pushes to free list at 0), `disk_refcount` (`:189`),
  `disk_free_count` (`:193`).
- Per-sequence state on `SequenceState` (`crates/spark-model/src/traits.rs:146-172`):
  `disk_block_ids: Vec<u32>` (full historical block list, grows monotonically),
  `disk_last_offloaded_per_layer: Vec<u32>` (per-layer offload cursor).
  Invariant: `disk_block_ids.len() == hss_window_start() + block_table.len()`
  (`traits.rs:153-155`, `hss_window_start()` at `:237-242`,
  `physical_block_for()` at `:250-256`).

### 2.3 Existing primitives (the composable pieces)
- **Evict HBM→NVMe (per block, per layer)**: `offload_block_on_stream(stream,
  layer, block, k_block_dev, k_host, v_host)` (`impl_more.rs:42`) and the
  quant variant `offload_block_no_predict_on_stream` (`:70`). Writes each
  kv-head stripe to `backend.write_from_host(GroupKey…)`, invalidates the
  resident-cache copy.
- **Restore NVMe→HBM (for attention only)**: `attend_layer_on_stream(stream,
  layer, seq_block_ids, q_dev, output_dev)` (`impl_more.rs:166`) and the
  causal `_with_q_pos` variant. It tiles `seq_block_ids`, reads **missing**
  blocks from disk into **scratch-pool slots** via `backend.read`, runs tiled
  attention, and finalizes. **Crucial: it restores into transient scratch
  slots for the kernel, NOT back into addressable paged-KV blocks.**
- **Allocator/refcount**: `alloc_disk_block_id`/`inc_disk_ref`/`dec_disk_ref`.
- Diagnostics: `diagnostic_summary()` (`:202`).

### 2.4 Install / lifecycle
- Thread-local install: `install_local(stream, cfg, model_dims)`
  (`high_speed_swap.rs:244`), `local_installed()` (`:253`), `with_local(f)`
  (`:258`). Installed once on the scheduler thread by
  `crates/spark-server/src/scheduler/mod_helpers.rs:18` (`install_high_speed_swap`),
  called at `mod.rs:309` right after CUDA bind. Requires the model to expose
  `high_speed_swap_dims()` (`traits/model.rs:315`; dispatch in
  `model/trait_impl/meta.rs:35`).
- Engagement gate (`decode/high_speed_swap.rs:20-44`,
  `high_speed_swap_engaged`): requires **both**
  `kv_cache.config().cache_blocks_per_seq.is_some()` **and** `local_installed()`
  **and** a supported KV dtype. `cache_blocks_per_seq` is a **separate** knob
  (`--high-speed-swap-cache-blocks-per-seq`, default 64) that shrinks the KV
  allocation (`serve_phases/kv_cache.rs:45-75`) — it is NOT in
  `HighSpeedSwapConfig` (`config.rs:12-32`).
- Decode wiring: `decode/attention_forward.rs:668-697` routes attention
  through the orchestrator when engaged (`offload_new_blocks` then
  `attend_layer_on_stream`); prefill wiring in
  `qwen3_attention/trait_impl/prefill_inner.rs:211-228,611-624`.
- Prefix-cache integration (`model/block_mgmt.rs`): `apply_evicted_blocks`
  (`:52`, dec_disk_ref on cache eviction), `cache_acquires_disk_refs` (`:86`),
  `reuse_prefix_match_disk_ids` (`:113`); `free_sequence` dec_disk_refs the
  seq's ids (`trait_impl/sequence.rs:164-179`). Sliding-window alloc:
  `ensure_blocks_through_decode`/`_prefill` (`block_mgmt.rs:196+`),
  `check_safe_to_evict`/`advance_layer_cursors_after_slide` (`:145,170`).

### 2.5 Maturity — implemented vs phased/stubbed
- **Implemented + validated**: disk-id allocator + refcount (Phase 6.1.a/e,
  tests in `high_speed_swap/disk_id_tests.rs`), per-block offload, tiled
  streaming attention, sliding-window HBM shrink, per-layer offload cursors,
  prefix-cache disk-ref accounting, chunked-prefill boundary correctness
  (issue #31 fixed), quant dtype dequant paths.
- **Phased / partial**: `TiledAttention` and scratch pool are **single-
  sequence** (`high_speed_swap.rs:116` `max_seqs: 1`, `:99` scratch is one
  seq); `attend_layer_on_stream` scores/tiles one seq per call. The full
  per-layer replacement path (`ATLAS_HIGH_SPEED_SWAP_REPLACE`) is gated behind
  an env var explicitly labelled "UNTESTED on real models" (`mod_helpers.rs:45-51`).
  Markers "Phase 6.1.x / 6.2.c / 6.3" throughout.
- **Not implemented**: SSM (`h_state`/`conv_state`) is **never offloaded** to
  disk — HSS keeps it HBM-resident in the SSM pool. There is **no**
  "offload/restore whole sequence" API and **no** "restore disk block back into
  an addressable paged-KV block" primitive.

---

## 3. GAP ANALYSIS — what HSS lacks to serve whole-sequence preempt

The core conceptual mismatch: **classic swap frees ALL of a sequence's
resources (KV blocks + SSM slot) and reconstitutes them later; HSS keeps a
rolling HBM window + SSM slot permanently resident and only *streams* history
for attention.** To make HSS subsume preempt, a preempted sequence must
release the same resources classic frees. Concrete gaps:

1. **No "evict sequence X entirely" API.** HSS has per-block `offload_block`
   but no method that (a) ensures every block of a seq is on disk across every
   layer, then (b) reports the seq can drop all HBM blocks. `offload_block` is
   driven from inside the attention layers, not callable stand-alone by the
   scheduler for an arbitrary victim outside a forward pass.

2. **No "restore sequence X into addressable HBM" API.** `attend_layer_on_stream`
   restores into scratch slots for a single kernel, not into paged-KV blocks
   with a real `block_table`. Whole-sequence resume needs disk→HBM into a fresh
   `block_table` (what classic `restore_sequence_state` does).

3. **SSM state is not covered.** The pressure that triggers preempt is often
   **SSM pool exhaustion** (`phase_start_prefills.rs:335` `"pool exhausted"`,
   and the KV-block exhaustion in `mod.rs:390`). HSS never frees the SSM slot,
   so a purely-HSS "preempt" would not relieve SSM-pool pressure and would not
   reproduce classic's `free_sequence` (which releases the slot). **Any
   subsumption MUST still save+free the SSM state** — either reuse classic's
   D2H-of-SSM-to-disk, or add SSM offload to HSS.

4. **Scheduler integration point missing.** The preempt decision
   (`mod.rs:385-417` victim selection; `phase_start_prefills.rs:330`) currently
   calls `swap_out_sequence`/`resume_swapped_seq`. There is no hook to route
   that through HSS. HSS is only invoked implicitly from inside layer forward
   passes.

5. **Prefix-cache / shared-block refcounts.** A preempted seq may share cached
   blocks with live seqs. Classic swap **copies** the KV bytes out and frees
   the physical blocks (safe, but duplicates shared data on disk). HSS's
   disk-ids are **refcounted and shared** (`block_mgmt.rs:52-134`), so a
   sequence-level evict that reuses disk-ids must **not** free a disk block
   still referenced by another seq or the prefix cache — the existing
   `dec_disk_ref`/`inc_disk_ref` accounting already gives us this **if** the
   evict path goes through it (classic's copy path bypasses it entirely). This
   is actually an argument *for* HSS: dedup of shared prefixes on disk.

6. **CUDA-graph slot on restore.** Classic already tolerates restoring to a
   different `slot_idx` (graphs re-captured, memory `-323`). HSS resident
   scratch is slot-agnostic, but if resume re-materializes into paged-KV it
   inherits classic's re-capture behavior — acceptable. Must ensure the
   resumed seq's `disk_block_ids`/`disk_last_offloaded_per_layer` and
   `block_table` invariant (`traits.rs:153`) is reconstructed consistently.

7. **Single-sequence orchestrator.** `attend_layer_on_stream` and the scratch
   pool assume one seq. Whole-sequence evict/restore that goes disk→HBM (not
   streaming) sidesteps this, but a future "resume in streaming mode" would hit
   the single-seq limit.

8. **Capacity coupling.** `alloc_disk_block_id` caps at `max_blocks_per_layer`
   (`high_speed_swap.rs:160`, returns `None`); a whole extra sequence's worth of
   blocks must fit the disk-id space and the `--high-speed-swap-bytes` budget.
   Exhaustion must degrade gracefully (fall back to classic or refuse), **not**
   panic — note `inc_disk_ref` **panics** on a freed id (`:172`). This is the
   crash-class to avoid (see §4d).

---

## 4. THE PLAN — phased, minimal-risk

**Design decision (recommended): "copy-out to HSS disk, free all HBM + SSM,
restore disk→HBM."** i.e. reuse classic's *policy and resource lifecycle*
(free_sequence on evict, alloc_sequence + rebuild block_table on resume) but
replace the **byte transport** from `KvSpillManager`/`/tmp` files to HSS's
per-layer NVMe backend + disk-id allocator. This is the smallest correct step:
it does NOT require HSS to gain multi-seq streaming attention or SSM offload
kernels, and it lets shared prefixes dedup via refcounts. A later phase can
optionally upgrade "resume" to lazy streaming.

### Phase 0 — Instrumentation + baseline (0.5 day)
- Add `tracing::debug!` counters for classic swap-out/in frequency, victim
  size, and disk bytes (keep permanent per memory `add-permanent-tracing-logs`).
- Capture a baseline: run the standard soak (memory `holo-soak-standard`) with
  `--swap-space-gb 3` under enough concurrency to force swap; record swap
  events, correctness (needle test), and that no CUDA-700 occurs.

### Phase 1 — New HSS "sequence blob" API surface (2-3 days)
On `HighSpeedSwap` (spark-storage), add sequence-granular primitives that
compose the existing per-block ones. Proposed signatures:
```
// Evict: write one block's K/V for one layer to disk under an owned disk_id,
// returning the id. (Thin wrapper making offload callable by id, not by the
// implicit seq window.)
pub fn stage_block_to_disk(&mut self, stream, layer, disk_id, k_host, v_host) -> Result<()>
// Restore: read one block's K/V for one layer from disk into a caller HBM ptr.
pub fn load_block_from_disk(&mut self, stream, layer, disk_id, k_dst_dev, v_dst_dev) -> Result<()>
// Bulk alloc/free of a contiguous run of disk ids for a whole sequence.
pub fn alloc_seq_disk_ids(&mut self, n: usize) -> Option<Vec<u32>>
pub fn free_seq_disk_ids(&mut self, ids: &[u32])   // dec_disk_ref each
```
- `load_block_from_disk` is the genuinely new capability (disk→arbitrary HBM
  ptr); implement it via `backend.read` into the destination pointer instead of
  a scratch-pool slot (reuse the `GroupKey`/`ReadRequest` plumbing from
  `attend_layer_on_stream`, `impl_more.rs:243-266`).
- Guard `alloc_seq_disk_ids` to return `None` (never panic) on capacity
  exhaustion; callers fall back to classic.
- Unit tests mirroring `disk_id_tests.rs`: alloc N, stage, load-back, byte-
  parity round trip on host buffers; free returns ids to the pool.

### Phase 2 — Model trait: HSS-backed save/restore (2-3 days)
Add two default-`bail!` trait methods on `Model` (siblings of
`save_sequence_state`/`restore_sequence_state`, `traits/model.rs:862/873`):
```
fn swap_out_sequence_hss(&self, seq: &SequenceState) -> Result<SeqSwapHandle>
fn swap_in_sequence_hss(&self, seq: &mut SequenceState, handle: &SeqSwapHandle) -> Result<()>
```
`SeqSwapHandle` = `{ disk_ids: Vec<u32>, num_blocks, ssm_blob: Vec<u8> }`.
Implement for `TransformerModel` (dispatch file next to `state_io.rs`):
- **swap_out**: for each `block_table` entry, for each layer, `read_block`
  (D2H, reuse `state_io.rs:41-46` logic) then `stage_block_to_disk` under an
  allocated disk-id (or reuse the seq's existing `disk_block_ids` when HSS was
  already engaged — then it's a no-op copy, big win). SSM `h/conv` still goes
  D2H into `ssm_blob` (Gap #3 — SSM stays host-side; small, this matches
  classic exactly). Returns the handle. **Does not free** (scheduler frees via
  existing `free_sequence`).
- **swap_in**: `alloc_block` num_blocks fresh HBM blocks, `load_block_from_disk`
  each (disk→HBM), rebuild `block_table`; `copy_h2d` the SSM blob into the new
  slot; `free_seq_disk_ids(handle.disk_ids)` (respecting shared refcounts).
- Reuse the exact SSM-layer iteration from `state_io.rs:56-69/122-136`.

### Phase 3 — Scheduler routing behind a flag (2 days)
- Add `--swap-backend {classic|hss|auto}` (default `classic`) OR reuse a
  boolean, e.g. `--preempt-via-high-speed-swap`. Keep classic as the fallback;
  **do not delete it.**
- In `swap_out_sequence` (`lifecycle.rs:143`): when the flag is `hss` and HSS
  is installed, call `model.swap_out_sequence_hss` instead of
  `spill.create_file()+save_sequence_state`; store the returned
  `SeqSwapHandle` on `SwappedSeq` (add a `hss_handle: Option<SeqSwapHandle>`
  field next to `swap_id`, `types.rs:471`). Keep the whole slot-compaction /
  `detach_slot_for_reuse` / `free_sequence` / EP-broadcast lifecycle **exactly
  as today** — only the transport changes.
- In `resume_swapped_seq` (`lifecycle.rs:246`): branch on
  `hss_handle.is_some()` → `swap_in_sequence_hss`, else classic.
- `auto` mode: prefer HSS when installed and `alloc_seq_disk_ids` succeeds,
  else classic. Any HSS error → log + fall back to classic for that seq (never
  crash the scheduler thread).
- Victim selection policy (`mod.rs:391-400`, grammar filter) is **unchanged**.

### Phase 4 — Prefix-cache / refcount correctness (1-2 days, folded into 2-3)
- When the victim was **already HSS-engaged** (`cache_blocks_per_seq` set), its
  `disk_block_ids` already exist and are refcounted; swap-out should **reuse**
  them (just retain the refs, drop HBM window) rather than allocate a second
  copy. This is the dedup win and avoids double disk usage.
- When the victim was **not** HSS-engaged (classic KV cache, HSS installed only
  for preempt transport), swap-out allocates fresh disk-ids with refcount 1 and
  frees them on resume — no sharing, byte-for-byte like classic.
- Never `dec_disk_ref` a block still referenced by the prefix cache: rely on
  the existing `apply_evicted_blocks`/`cache_acquires_disk_refs` accounting
  (`block_mgmt.rs:52-134`); resume's `free_seq_disk_ids` only decs the seq's
  own refs.
- Assert-audit: `inc_disk_ref` panics on freed id (`high_speed_swap.rs:172`) —
  add a `checked_inc_disk_ref -> Result` used on all preempt paths so a logic
  bug degrades to a logged error, not a thread panic.

### Phase 5 — Validation strategy (2-3 days)
- **Bit-correctness (differential)**: with `--high-speed-swap-cache-blocks-per-seq`
  set to `max_seq_len/block_size` (no eviction, per `serve_args.rs:415-418`
  note) as a control, run a fixed prompt through: (a) no swap, (b) classic
  swap forced, (c) HSS-preempt forced, and diff output tokens — they must be
  identical (greedy, temp=0). Force swap by shrinking KV blocks / raising
  concurrency so `num_free_blocks() < blocks_needed`.
- **Round-trip unit test** (Phase 1/2): host-buffer parity of stage→load per
  layer, and a full seq save→free→alloc→restore producing identical KV bytes
  (compare via `read_block`).
- **Needle-in-haystack** at 48K (memory `holo-prefill-attention-gap`) with
  forced preempt mid-decode to prove long-context history survives evict/
  restore.
- **Crash class to avoid** (the exhaustion→CUDA-700): drive disk-id and
  SSM-pool exhaustion simultaneously and assert the system returns a
  503/preempt (`handle_prefill_start_error`) or falls back to classic — never
  a CUDA-700 / panic. Verify `alloc_seq_disk_ids` returns `None` and the
  scheduler handles it. Run the standard soak (memory `holo-soak-standard`) for
  300s at 6-client concurrency under `--swap-backend hss` and confirm zero
  crashes + parity vs classic baseline.
- Keep `RUST_LOG=info` (memory `deploy-rust-log-info`); use `max_tokens>=250`
  in all bench requests (memory `bench-max-tokens`).

### Phase 6 — Convergence / deprecation of classic (later, gated on data)
- After HSS-preempt is proven bit-correct + crash-free across all served model
  families (dense + hybrid-SSM), flip `--swap-backend` default to `auto`.
- One release later, default to `hss` when HSS is installed; keep `classic`
  selectable.
- Only after a full release cycle with no HSS-preempt regressions: mark
  `--swap-space-gb` / `KvSpillManager` / `/tmp/atlas-swap` deprecated (warn on
  use), then remove `save_sequence_state`/`restore_sequence_state` +
  `KvSpillManager` once no path selects classic. **Do not remove up front.**
- End state: one swap subsystem (HSS) serving both intra-sequence streaming
  and whole-sequence preempt; the scheduler policy code is untouched.

### Optional Phase 7 — Lazy streaming resume (stretch)
Instead of disk→HBM copy on resume, resume a seq in **HSS streaming mode**
(window=0, attend fully from disk) so restore is O(1) HBM. Blocked on the
single-seq orchestrator limitation (Gap #7) and SSM residency (Gap #3); track
as future work, not required for classic deprecation.

---

## 5. Risks, open questions, effort

| # | Risk / open question | Severity | Mitigation |
|---|---|---|---|
| R1 | **SSM slot not freed by HSS** → preempt doesn't relieve SSM-pool pressure | HIGH | Keep classic's `free_sequence` (frees slot) on evict; only transport changes. SSM blob stays host-side like classic. |
| R2 | Disk-id / `--high-speed-swap-bytes` exhaustion → `inc_disk_ref` **panic** (`high_speed_swap.rs:172`) crashes scheduler thread (the CUDA-700-class failure) | HIGH | `alloc_seq_disk_ids` returns `None`; `checked_inc_disk_ref`; auto-fallback to classic; validation Phase 5 forces exhaustion. |
| R3 | Shared prefix disk blocks double-freed or freed while still referenced | MED | Route all evict/resume through existing refcount accounting; only dec seq-owned refs. |
| R4 | Single-seq orchestrator can't stream two preempted seqs concurrently | LOW (copy-based resume sidesteps) | Copy disk→HBM on resume (Phase 2) avoids streaming; defer streaming resume to Phase 7. |
| R5 | Grammar / rollback-ring / cancel_flag lossy fields | LOW | Identical to classic today (`lifecycle.rs:286,330,348`); preserve exact behavior; keep the grammar-active victim exclusion (`mod.rs:394`). |
| R6 | HSS only wired for Qwen3-style attention layers; other model families | MED | `swap_out_sequence_hss` default-`bail!`; scheduler auto-falls back to classic for unsupported models. |
| R7 | CUDA-graph re-capture on resume to a new slot | LOW | Already tolerated by classic (memory `-323`). |

**Open questions to resolve before Phase 2:**
- Should SSM state also be offloaded via HSS's NVMe backend (uniformity) or
  stay in the host `ssm_blob` (simplicity)? Recommend host blob first.
- Does `--high-speed-swap` remain a *separate* enablement from
  `--swap-space-gb`, or does `--swap-backend hss` auto-install the orchestrator
  even without the KV-shrink (`cache_blocks_per_seq`)? The orchestrator install
  (`mod_helpers.rs:18`) and the KV-shrink (`kv_cache.rs:45`) are already
  decoupled, so preempt-only HSS (install orchestrator, full-size KV window) is
  feasible and is the cleanest migration target.

**Rough effort:** Phase 0: 0.5d · Phase 1: 2-3d · Phase 2: 2-3d · Phase 3: 2d ·
Phase 4: folded · Phase 5: 2-3d · Phase 6: staged over releases · Phase 7:
stretch. **Core migration (Phases 1-5) ≈ 8-12 engineering days.**

---

## 6. Notes / things I could NOT determine
- No plan file for HSS exists on disk: `.claude/plans/` is empty and no
  `valiant-bunny` / `i-want-to-ensure-valiant-bunny.md` file is present in the
  worktree (the `serve_args.rs:378` comment references it, but the file is
  gone). `HANDOFF.md` (untracked, git status) is **empty**.
- The specific commit for the "recent exhaustion→CUDA-700 crash" was not found
  in the local git log; the failure *mechanism* is nonetheless clear from code
  (`alloc_disk_block_id` capacity return + `inc_disk_ref` panic +
  `"pool exhausted"` preempt path) and is addressed by R2.
- Whether every production model family that uses classic swap also satisfies
  `high_speed_swap_dims()` (`meta.rs:35`) was not exhaustively enumerated —
  Phase 3 `auto` fallback makes this non-blocking, but it should be audited
  before flipping the default (Phase 6).
