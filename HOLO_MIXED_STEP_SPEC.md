The source confirms the spec's claims with two important corrections I'll fold in:

1. **The mixed path already bridges streams** at `run_standard.rs:108-114`: on `is_last` it does `normalize_ssm_states(prefill_stream)` then `record_event(prefill_event, prefill_stream)` + `stream_wait_event(default_stream, prefill_event)`. But `mixed_forward` runs on `default_stream` (decode_b.rs:100/353). So the order is backwards for the *write*: the GDN write is on default_stream, the normalize is on prefill_stream — and the event bridge is recorded on prefill_stream AFTER the normalize, which does NOT make prefill_stream wait for default_stream's writes. The ssm-correctness verdict is correct: per-chunk normalize must run on default_stream.

2. `can_mix` is already the gate; the two-phase branch (`use_twophase`) at `:194` fires on tick 1 and starves decode — confirmed.

Here is the deliverable.

---

# Decode keep-alive during prefill bursts — A status, B verdict, corrected spec

## 1. A status — DEFERRED (no clean GPU window)

The chunk-cap A/B (sweep `ATLAS_HOLO_MAX_PREFILL` ∈ {4080, 2048, 1024, 512} and measure decode TBT under a concurrent burst) is **deferred**: ComfyUI prod is up on the shared GB10, so the other tenants contaminate SM occupancy and TBT — any TBT number measured now is noise, not signal.

**Working assumption (proceed on this):** chunk-size alone is a **weak lever**. Shrinking the flat cap shortens each prefill-only step but does nothing about the structural problem — the scheduler still does prefill-**OR**-decode, so during a burst decode only advances when a prefill step happens *not* to be scheduled (~5 tokens/burst). **B (fuse decode into every prefill step so decode advances every iteration) is the real fix.** The SSM 2-launch-per-layer reality (GDN decode-recurrence and GDN prefill-scan are different kernels, cannot merge into one launch) means B buys **weight-load amortization on the prefill region + decode flowing every step**, not a fused recurrence — so the ceiling is bounded (see §4).

**Exact A measurements to run later (Atlas-only window, ComfyUI down) — these tune B's constants and confirm payoff:**

1. **Fused-step floor sweep (the gating number for B's feedback loop).** Drive a single `mixed_forward` at `slice=32` for N∈{1,2,4,8} decodes; record per-step wall time, isolating the `gpu.synchronize(stream)` at `decode_b.rs:330`, the padded decode kernels (`padded_n` = 2/4/8), and the dummy `alloc_state`×40 (`decode_b.rs:379-398`). **This `T_floor(N)` is the constant the budget controller needs and is the go/no-go gate**: if `T_floor(8) ≥ 0.8·tbt_deadline`, even a 32-token slice can't fit and the always-mix design must fall back to suppress under max batch.
2. **Marginal prefill cost.** `T_fused(N, slice)` for slice∈{32,128,512,2048,4080} → fit `cost_per_tok = (T_fused − T_floor)/slice`. Feeds the budget formula.
3. **Standalone-decode baseline.** `T_dec(N)` for N∈{1,2,4,8} via the standalone path (`mod.rs:307`) — to confirm fused at slice=32 isn't slower than just running decode alone at low concurrency (the c1/c2 erosion risk).
4. **Burst TBT, A/B on B.** Soak (`soak-holo-atlas-64k.py`, 6-client mixed) with B off vs on; report **p50/p99 TBT** and **decode tokens advanced per scheduler iteration during a prefill burst** (instrument the burst window). Target: ≥1 decode token/iteration (vs ~5/burst today), p99 TBT ≤ `tbt_deadline`.
5. **Chunk-cap interaction.** With B on, the deferred cap sweep — confirm the cap is now second-order (B's slice budget, not the flat cap, governs fused-step cost).
6. **MIN_PREFILL_SLICE tune.** Sweep `ATLAS_SLAI_MIN_PREFILL_SLICE`∈{4,16,32,64}; pick the largest that keeps p99 TBT under deadline (prefill-progress vs decode-latency tradeoff).

---

## 2. B verdict — **GO WITH FIXES (blocking)**

The design is sound in shape (reuse `decode_b.rs:35`, no new GDN batch kernel) but ships **three on-by-default defects** the adversarial lenses caught. None is fatal; all fold into the plan. **Do not ship the design as written** — the SLAI budget and the per-chunk normalize are both wrong as specified.

| # | Blocker | Lens | Mitigation (folded into spec) |
|---|---------|------|-------------------------------|
| **B1** | **SSM state corruption.** Per-chunk `normalize_ssm_states` on `prefill_stream` races the GDN recurrence writes on `default_stream` (mixed_forward forces default_stream, ignores the passed stream). The normalized (corrupt) h_state becomes the *input* to the next chunk → compounds across every chunk of a long prompt. The existing `is_last` normalize hides this because its output is terminal. **Confirmed in source:** `run_standard.rs:109` normalizes on `prefill_stream`, but `decode_b.rs:100/353` runs GDN on `default_stream`, and the event bridge at `:113-114` is recorded *after* the normalize so it doesn't order the write→read. | ssm-correctness (blocker, high) | **Move the per-chunk normalize INSIDE `mixed_forward_dispatch`** (after the layer loop, `decode_b.rs:458`, gated on `!prefill_is_last`), launched on the same `stream` local (= default_stream) as the GDN writes. Do **NOT** add the design's "duplicate at run_standard.rs:99 on prefill_stream" — that is the bug. In-order, same-stream, no event needed. |
| **B2** | **SLAI budget is open-loop and self-defeating.** `worst = now − last_token_time` is reset to ~0 by the fused step's own `process_decode_logits` every iteration → in steady burst `slack ≈ margin`, `f ≈ 1`, budget snaps back to full 4080 nearly every step. The slice-shrink that the whole TBT bound rests on never engages (only fires once right after a stall, then flaps 4080↔32). **No TBT bound delivered; p99 jitter instead.** | liveness + perf-reality (concern, high) | **Drive the budget off measured step cost, not the live SLAI clock.** `slice_budget = WY4_align( clamp( (TBT_target − T_dec(N)) / cost_per_tok, MIN, full_chunk ) )`, with `T_dec`/`cost_per_tok` as EWMAs (seed from a conservative constant). Keep the `worst≥margin` slack term only as a **secondary clamp → MIN**, plus a **genuine suppress when `worst ≥ 1.0·tbt_deadline`** (decode already past hard deadline — never inject prefill, run decode-only). Add ≤2× per-step grow hysteresis to stop flapping. |
| **B3** | **Two-phase branch starves decode on tick 1.** `use_twophase` (`phase_continue_prefills.rs:194`) fires when `chunk_offset==0 && len > max_prefill_tokens` — runs the **entire** prompt with **no decode fused, ignoring slice_budget** → multi-second monolithic forward, TBT spike on the first tick of every long prefill. **Confirmed at source.** | ssm-correctness secondary | **Gate two-phase on `active.is_empty()`:** `let use_twophase = active.is_empty() && p.chunk_offset == 0 && p.prompt_tokens.len() > max_prefill_tokens;`. When decodes are active, force the chunked mixed path so tick 1 also fuses and respects the budget. |
| **B4** | **Buffer-overflow silent de-fuse.** `padded_n + n_prefill > max_batch_tokens` (`decode_b.rs:67`, confirmed) silently drops to sequential decode_batch + prefill_chunk (weights loaded twice — worse than today). With 16 decodes + a slice this can trip. | perf-reality + invariants | **Clamp:** `slice_budget = slice_budget.min(max_batch_tokens − padded_n)` with a `debug_assert`. Keeps fusion intact under max batch. |
| **B5** | **Per-step host overhead now on every decode tick.** Dummy `alloc_state`×40 (`decode_b.rs:379-398`) + unconditional `synchronize` (`decode_b.rs:330`) erode c1/c2 decode tok/s. Design lists these as "optional" but ships always-mix. | perf-reality | **Promote both to REQUIRED before always-on:** pre-allocate the dummy attention-layer states once (like the dummy SSM slot); replace the unconditional `synchronize(stream)` with an event-wait. Until landed, gate always-mix behind a flag. |

Everything else in the design (single-stream fused path only, `can_batch_mixed`→false fall-through, strict admission, INVARIANT 1/2/3/7, spec/vision/EP exclusions) is correct and stays.

---

## 3. Implementation spec (corrected, smallest-shippable-first)

Each step has a validation gate. **Ship in order; do not proceed past a failing gate.**

### Step 0 — Make the existing mixed path correct under multi-chunk (prereq, no behavior change)
The current `is_last`-only normalize already ships; B exposes the multi-chunk race. Fix it *first*, independent of any scheduler change.

- **`crates/spark-model/src/model/trait_impl/decode_b.rs`** — after the fused layer loop (`:458`), before final-norm/LM-head:
  ```rust
  if !prefill_is_last {
      // same `stream` local (= default_stream) the GDN recurrence wrote on — in-order, no race
      self.normalize_ssm_states_on(prefill_seq, stream)?;   // or inline the clamp-norm launch on `stream`
  }
  ```
  Do **not** touch the `is_last` normalize at `run_standard.rs:108-114` (terminal, already correct). Remove the design's "duplicate at line 99" instruction entirely (**B1**).

**Gate 0:** force the chunked mixed path (`ATLAS_HOLO_MAX_PREFILL=512`, one concurrent decode) on a **>4096-token** prompt; diff the prefilling request's output **token-for-token** vs today's suppress-then-prefill path. **Run 5–10×** (race is nondeterministic — one clean run is insufficient).

### Step 1 — Two-phase gate (tiny, removes the tick-1 starvation)
- **`phase_continue_prefills.rs:194`**: `let use_twophase = active.is_empty() && p.chunk_offset == 0 && p.prompt_tokens.len() > max_prefill_tokens;` (**B3**).

**Gate 1:** long prefill + active decode → confirm tick 1 takes the chunked path (log shows `Mixed forward: prefill <chunk>/<len>` on tick 1, not a single monolithic two-phase forward). Decode advances on tick 1.

### Step 2 — SLAI prefill-slice budget (the core change), feedback-driven
- **`crates/spark-server/src/scheduling_policy.rs`**: add trait method
  `fn prefill_slice_budget(&self, active_timings: &[ActiveSeqTiming], full_chunk: usize) -> usize;`
  - `FifoPolicy` → `full_chunk`.
  - `SlaiPolicy`: store `min_prefill_slice` (env `ATLAS_SLAI_MIN_PREFILL_SLICE`, default 32, clamp ≥4, WY4-align) read once in `new`. Budget = **cost-driven, not slack-driven** (**B2**):
    ```
    if active_timings.is_empty() { return full_chunk; }
    worst = max(now − last_token_time)
    if worst >= tbt_deadline { return 0; }          // hard suppress: decode past deadline, run decode-only
    raw   = (TBT_target − T_dec_ewma) / cost_per_tok_ewma   // measured-cost controller
    budget = clamp(raw, min_prefill_slice, full_chunk)
    budget = (budget/4)*4; budget.max(4)            // WY4-align
    ```
    `T_dec_ewma` / `cost_per_tok_ewma` live on `SlaiPolicy` (interior-mutable / updated by the scheduler after each step via a new `record_step(n_decode, slice, dur)`), seeded conservatively so step 1 is bounded. Apply ≤2× per-step grow hysteresis. **Note:** this returns **0** only in the hard-deadline case — the scheduler treats 0 as "suppress, decode-only this tick."
  - Keep `should_prefill` **unchanged** (admission-gate contract).
  - Unit tests: full_chunk when empty/fresh; MIN under moderate pressure; **0 only when `worst ≥ tbt_deadline`**; monotonic decrease as `cost_per_tok` rises; never exceeds full_chunk; always WY4-aligned.

- **`phase_continue_prefills.rs:93-129`**: replace the binary `do_chunks` early-return.
  - Move `single_active_with_spec` (currently `:114-115`) **above** line 93.
  - `let fusable_mixed = !active.is_empty() && !model.is_ep() && !single_active_with_spec;`
  - `let slas_ok = active.is_empty() || policy.should_prefill(&timings);`
  - `if !slas_ok && !fusable_mixed { return did_mixed_step; }` (genuine suppress: EP / single-active-spec / no decode).
  - `let slice_budget = policy.prefill_slice_budget(&timings, max_prefill_tokens);`
  - **`if slice_budget == 0 { return did_mixed_step; }`** — hard-deadline suppress; decode runs standalone at `mod.rs:307` (**B2** late-decode protection).
  - **Clamp (B4):** `let slice_budget = slice_budget.min(max_batch_tokens.saturating_sub(padded_n_for(active.len())));` with `debug_assert`.
  - `let can_batch_mixed = false;` (or gate behind off-by-default `ATLAS_BATCHED_MIXED=1` escape hatch). `can_batch_prefill_only` (active-empty cold-start) **unchanged**.
  - Thread `slice_budget` into the `prefilling.first_mut()` → `run_standard_chunk_loop` call.

- **`run_standard.rs:18-37`**: add `slice_budget: usize` param. At `:60`: `let cap = effective_max.min(slice_budget); let mut chunk_len = remaining.min(cap);` (MLA gate at `:55-59` untouched; `is_last`/WY4 at `:61-65` already handle small slices). **Do not** add any normalize here — Step 0 handles it inside `mixed_forward`.

**Gate 2:** (a) Gate-0 token-for-token re-run under SLAI pressure (`MIN=32`), ×5–10. (b) Soak: p99 fused-step time ≤ `tbt_deadline`; decode advances ≥1 token/iteration during a burst; budget does **not** flap 4080↔32 (log the EWMA-driven slice). (c) Prefill tok/s under multi-prefill concurrency does not drop materially vs current path.

### Step 3 — Per-step overhead removal (required before declaring always-on) (**B5**)
- **`decode_b.rs:379-398`**: hoist dummy attention-layer states to a pre-allocated set (mirror `dummy_ssm_slot`), reused per call.
- **`decode_b.rs:330`**: replace unconditional `synchronize(stream)` with `record_event` + `stream_wait_event` so metadata H2D ordering is enforced without a full device sync.

**Gate 3:** c1/c2 decode tok/s with B on ≥ standalone-decode baseline (A-measurement #3). If fused@slice=32 for N=1 is slower than standalone decode, keep the floor-aware suppress (`fusable_mixed &&= est_floor < tbt_deadline`) so low-conc falls back to today's cheap path.

### Untouched (per design)
- `mod_helpers.rs:133` admission `should_prefill` — strict, unchanged (prefilling set must not grow under pressure).
- `select_prefills` shortest-first; co-dispatch idle-only window — unchanged (co-dispatch-into-decode DEFERRED).
- INVARIANT 6 (prefix cache, `kv_write_start=0` in fused path, `decode_b.rs:454`): **decide explicitly.** Resting config has `ATLAS_HOLO_PREFIX_CACHING=true`, so fused attention prefill rewrites KV a prefix match meant to skip → attention-layer output regression (SSM layers unaffected). **Recommendation for first cut: disable prefix cache when a fused step occurs** (simplest, correct); thread `kv_write_start` through `mixed_forward_dispatch` as a fast-follow. Document either way.

---

## 4. Expected payoff + ceiling

**What B delivers:** during a concurrent prefill burst, decode advances **every scheduler iteration** instead of ~5 tokens/burst. With the cost-driven budget (B2), each fused step is held under `tbt_deadline`, so per-token decode latency stays bounded while prefill creeps forward at full **weight-load amortization** (prefill region reads MoE/proj weights once, shared with the decode region's reuse out of L2). **p99 TBT under burst is the headline win** — the metric the soak (A-measurement #4) must confirm.

**Realistic magnitude (bounded by the SSM 2-launch reality):** mixing amortizes weights on the **prefill region only**. The decode region pays its own `decode_multi_seq`×40-layer launches + the per-call sync **in full every tick** — that's the fixed `T_floor(N)` from A-measurement #1. So:
- **Decode keep-alive: near-total** — decode goes from starved (~5/burst) to ≥1/iteration. This is the real, large UX win (no multi-second TBT stalls during bursts).
- **Aggregate throughput: roughly neutral to slightly negative.** Dribbling small prefill slices per fused step means a long prompt takes more ticks, each paying the decode floor. Prefill TTFT under a burst may rise slightly vs today's suppress-then-big-chunk; that's the **accepted trade** (ground truth: batched prefill is overhead-bound; the win is decode liveness, not prefill speed).

**What B does NOT fix (needs the deferred shared-budget rework C):**
- **Multi-prefill throughput.** Streams serialize one-per-tick (FIFO head only). N concurrent long prefills still don't overlap — needs the true N-prefill-fused `mixed_forward_batch` + the Phase-2b **GDN/attention batch-axis kernels** (DEFERRED). B explicitly does not build the N-prefill batch builder.
- **Co-dispatch-into-decode.** New requests still can't join an in-flight decode batch mid-stream; admission stays idle-only.
- **The prefill single-stream wall** (the ~16–dropped, now ~cuBLAS proj / GDN-FLA scan / MoE per-token efficiency) — orthogonal; B is a scheduling change, not a kernel change.
- **Low-concurrency decode tok/s** can regress unless Step 3 (overhead removal) lands — B is net-positive **only** with the dummy-alloc hoist + sync→event done.

**Bottom line:** B is the correct, lowest-risk fix for decode starvation and ships with the five folded mitigations. It makes decode smooth under bursts; it does **not** make concurrent prefill scale — that remains C (batch-axis GDN/attention kernels), still deferred.