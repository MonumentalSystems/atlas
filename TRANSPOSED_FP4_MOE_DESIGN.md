# Transposed-FP4 MoE kernel — design + runbook (from ultracode wf_d19ffb1b-3b1)

## WINNING DESIGN

# FINAL DESIGN SPEC — Transposed-`[K/2,N]` FP4 Fused MoE gate_up + down kernels (share FAST_MOE=full tables, zero extra MoE memory)

Authoritative implementation spec. All three judge panels converged on **Design 3** as the base, with specific grafts from Designs 1 and 2. This spec resolves every architectural fork and names every file/line to touch.

---

## 0. Ground truth (verified, load-bearing — do not re-litigate)

| Fact | Source | Consequence |
|---|---|---|
| `pack_bf16_weight_to_nvfp4_t` emits **`[N,K/2]`** (N-major, K-contiguous), despite its `_t` name | `crates/spark-runtime/cuda/cutlass_nvfp4_gemm.cu:214-217` (`packed_t[col*(k/2)+i/2]`, comment "NOT Atlas transposed `[K/2,N]`") | The existing `_t_k64_fp4` kernel reads `[N,K/2]` and is fed by its OWN re-packed tables (`build_fp4_gate_up`) → a **second resident copy** of all experts. This is the memory cost we remove. |
| `transpose_for_gemm` emits **`[K/2,N]`** (N-contiguous) packed nibbles + **`[K/16,N]`** scales, via a **pure byte transpose** | `quantized.rs:119` (`t_buf[j*n+i]=buf[i*half_k+j]`), scales at `:133` | This is FAST_MOE=full's `gate_ptrs_t`/`up_ptrs_t`/`down_ptrs_t` layout. The byte transpose **preserves the `(2j,2j+1)` e2m1 nibble pair intact** in each byte → no nibble swap needed. |
| The m16n8k64 mxf4 B-fragment requires **8 contiguous-K e2m1 per N-row = one aligned u32** | `moe_w4a16_grouped_gemm.cu:1615` (`FP4_FRAG`), MMA at `:1623` | A `[K/2,N]` smem tile (K-strided per N-row) cannot feed this directly → an **on-chip transpose is unavoidable**; the only fork is WHERE to pay it. |
| The FP8 `_t` kernel's B-load reads `B_expert[(gke>>1)*N+gns]` (coalesced, N-contiguous) and stages K-major, then runs `FGU64_DEQUANT` into a single-buffered `[N][K]` e4m3 tile | `moe_w4a16_grouped_gemm.cu:802, 827-865` | We mirror this load verbatim and **replace the dequant pass** with a cheaper nibble-only transpose. |
| FP8 `_t` scale load reads `S_expert[sg*N+gns]` into group-major `[group][N]` smem; `sfb_raw` assembles `smem_Bs[0..3][sfn]` — **identical orientation to the FP4 kernel's scale path** | FP8 `:806`; FP4 `:1538, 1617-1621` | The scale path composes **as-is, zero change, no transpose**. |
| FP8 `_t` folds per-expert `scale2` into dequant (`sv=f*scale2`); the FP4 path applies `scale2` at writeback | FP8 `:821`; FP4 `:1679` | `gate_ptrs_t` carries the **real** `weight_scale_2` (NOT the FP4 builder's hardcoded 1.0). Since `scale2` is a per-expert scalar it distributes out of the dot product → apply at writeback. **Must verify the constant** (§7). |

**Stale doc to fix as part of this work:** `crates/spark-runtime/src/cutlass.rs:339-341` says `pack_bf16_weight_to_nvfp4_t` outputs `[K/2,N]`. It outputs `[N,K/2]`. Correct the comment.

---

## 1. Architecture decisions (the resolved forks)

1. **Replace the existing kernel bodies in place** (keep symbols `moe_w4a16_fused_gate_up_t_k64_fp4` and `moe_w4a16_down_t_k64_fp4`). Do **not** add a coexisting `_tlayout`/`_kt` kernel (rejected from Design 1) — a parallel kernel + parallel tables risks leaving the redundant `[N,K/2]` path reachable, defeating the memory goal. One kernel, one table layout.
2. **Horn B — coalesced K-major cp.async + a one-per-K-step smem nibble-transpose pass** (Design 3), NOT an in-loop strided fragment gather (Design 2's distinctive choice). The coalesced global load is byte-identical to the proven FP8 `_t` path; the transpose is amortized **once per K-step over 16 N-subtiles × 2 projections**, leaving the hot MMA loop's single-u32 `FP4_FRAG` untouched. Design 2's per-MMA 4-byte strided gather adds ALU to the innermost loop (16×/K-step) and risks regressing to FP8 parity.
3. **Share FAST_MOE=full's existing `gate_ptrs_t`/`up_ptrs_t`/`down_ptrs_t` tables; build ZERO new tables; delete the `build_fp4_*` re-pack under FAST_MOE=full.** The memory win is **not allocating** the duplicate `[N,K/2]` copy — not freeing after. No dangling-decode risk.
4. **scale2 applied at writeback** (`acc*scale2`), reading the real per-expert `scale2` from the shared table. **No ldmatrix** anywhere (broken on GB10) — all fragments via explicit `*(u32*)` smem reads, as both reference kernels already do.
5. **One CTA per (expert, m-tile, 2N-strip)** — keep the existing non-persistent grid. Persistent-CTA rejected (expert imbalance handled by `expert_offsets` early-return; complicates the gather for no gain on a load-bound op).
6. **smem-staged scales** (not register-resident) — the FP8 `_t` scale orientation already matches `sfb_raw`; register-residency would pressure the 64-fp32-accumulator budget for no benefit.

---

## 2. Kernel: `moe_w4a16_fused_gate_up_t_k64_fp4` (rewritten body)

**File:** `kernels/gb10/holo-3.1-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu`, replace the body at lines ~1409-1684. Keep the `extern "C"` signature and symbol unchanged (so `init.rs:109` and the Rust wrapper are untouched).

### 2.1 Constants / grid / block (unchanged from both reference kernels)
`M_TILE=64`, `N_TILE_LG=128`, `K_STEP_T64=64`, `GROUP_SIZE=16`, `PAD_T64=8`, `BP_PAD=16`. Grid `(ceil(2*N/128), max_m_tiles, num_experts)`, block 128 (4 warps). `blockIdx.z=expert`, `blockIdx.x` tiles `2*N` (`is_up = global_n >= N`). Per-expert range from `expert_offsets`; A-rows gathered via `sorted_token_ids` into `smem_tok`. Holo gate_up: K=hidden=2048 (→32 k64 steps), N=moe_intermediate=512.

### 2.2 smem tiles
```c
// staging — K-major / N-contiguous, double-buffered, coalesced cp.async target (mirrors FP8 smem_Bp_fgu64)
__shared__ unsigned char smem_BpT[2][K_STEP_T64/2][N_TILE_LG + BP_PAD];          // [2][32][144]
__shared__ unsigned char smem_Bs [2][K_STEP_T64/GROUP_SIZE][N_TILE_LG + BP_PAD]; // [2][4][144] ue4m3 group scales (UNCHANGED from FP8/FP4)
// MMA-ready — N-major / K-contiguous, SINGLE-buffered (regenerated each K-step from smem_BpT[cur])
__shared__ unsigned char smem_Bp [N_TILE_LG][K_STEP_T64/2 + 16];                 // [128][48]
// A path — reused VERBATIM from existing FP4 kernel
__shared__ __nv_bfloat16 smem_A  [2][M_TILE][K_STEP_T64 + PAD_T64];              // [2][64][72]
__shared__ unsigned char smem_Ap [M_TILE][K_STEP_T64/2 + 4];                     // [64][36]
__shared__ unsigned char smem_As [M_TILE][K_STEP_T64/GROUP_SIZE];               // [64][4]
__shared__ int           smem_tok[M_TILE];
```
Total ≈ 37KB/CTA (< FP8's ≈39KB; `smem_Bp` 48B/N-row replaces FP8's `smem_B_fp8` 80B/N-row) → ≥2 CTAs/SM occupancy preserved.

### 2.3 B-load + scale-load — copy `FGU64_ISSUE_LOADS` (FP8 `:779-812`) VERBATIM
```c
// B nibbles: [K/2,N] N-contiguous → 16B coalesced cp.async per thread, K-major smem dest.
unsigned int kp = threadIdx.x >> 3;            // 16 K-pair rows/round
unsigned int ns = (threadIdx.x & 7) << 4;      // 16 N-cols (bytes)/thread
unsigned int gns = cta_n + ns;                 // cta_n = N within the 2*N strip
for (int rnd = 0; rnd < 2; rnd++) {
    unsigned int kp_cur = rnd*16 + kp;                            // 0..31
    unsigned int gke    = kb + (kp_cur << 1);
    moe_cp_async_pred_16(&smem_BpT[buf][kp_cur][ns],
        &B_expert[(unsigned long long)(gke>>1)*N + gns],          // [K/2,N] N-contiguous
        (gke+1 < K) && (gns+15 < N));
    if (kp_cur < K_STEP_T64/GROUP_SIZE) {                         // 4 scale groups
        unsigned int sg = kb/GROUP_SIZE + kp_cur;
        moe_cp_async_pred_16(&smem_Bs[buf][kp_cur][ns],
            &S_expert[(unsigned long long)sg*N + gns], (gns+15 < N));
    }
}
```
A-load and the `smem_tok` gather are copied verbatim from the existing FP4 kernel (`:1503-1515, 1476-1482`).

### 2.4 The new piece — `FP4_TRANSPOSE(buf)` (replaces FGU64_DEQUANT slot)
Runs after `cp.async wait + __syncthreads`, before the MMA, in the exact pipeline slot the FP8 kernel runs `FGU64_DEQUANT`. Each of 128 threads owns one N-row and gathers its 32 packed K-bytes from the K-major staging tile into a contiguous K-run, **vectorizing the writes** to 4-byte stores:
```c
#define FP4_TRANSPOSE(buf) do {                                                  \
    unsigned int my_n = threadIdx.x;                                            \
    if (my_n < N_TILE_LG) {                                                     \
        _Pragma("unroll")                                                       \
        for (int q = 0; q < (K_STEP_T64/2)/4; q++) {   /* 8 u32 = 32 bytes */   \
            unsigned int w = (unsigned)smem_BpT[(buf)][q*4+0][my_n]             \
                | ((unsigned)smem_BpT[(buf)][q*4+1][my_n] << 8)                 \
                | ((unsigned)smem_BpT[(buf)][q*4+2][my_n] << 16)               \
                | ((unsigned)smem_BpT[(buf)][q*4+3][my_n] << 24);              \
            *(unsigned int*)&smem_Bp[my_n][q*4] = w;                            \
        }                                                                       \
    }                                                                           \
} while(0)
```
- **Reads** `smem_BpT[buf][kp][my_n]`: 128 lanes read the same `kp` row at consecutive `my_n` → one 128-byte row, conflict-free (the `BP_PAD=16`/144-stride is the FP8-proven layout). This IS the access pattern `FGU64_DEQUANT` already runs (`:827-865`).
- **Writes** `*(u32*)&smem_Bp[my_n][q*4]`: lane `my_n` writes row `my_n`, stride 48 — the `+16` pad spreads the 128 lanes across banks.
- **No nibble swap** — pure byte copy; `transpose_for_gemm`'s byte transpose already preserved the low=even-k/high=odd-k pairing (verified §0).

### 2.5 Pipeline (mirror FP8 single-buffered-target discipline EXACTLY)
Prologue: `ISSUE_LOADS(buf0)` → commit → `wait_all` → `__syncthreads` → `FP4_QUANT_A(0)` → `FP4_TRANSPOSE(0)` → `__syncthreads`. Steady loop over `k_base` step 64: `ISSUE_LOADS(nxt)` → commit → `FP4_COMPUTE_MMA(cur)` → `wait_all` → `__syncthreads` → `FP4_QUANT_A(nxt)` → `FP4_TRANSPOSE(nxt)` → `__syncthreads` → swap. Epilogue: `FP4_COMPUTE_MMA(cur)`. The **post-transpose `__syncthreads` before the MMA is mandatory** (single-buffered `smem_Bp` is read cross-warp via `nc=nt*8+group_id`) — this mirrors the FP8 post-dequant sync, so the proven ordering carries over; `smem_Bp` for `nxt` is regenerated only after `COMPUTE_MMA(cur)` has consumed it.

### 2.6 MMA + scale + A-quant + fusion + gather — REUSE VERBATIM from existing FP4 kernel
- `FP4_QUANT_A` (`:1549-1591`), `FP4_FRAG` (`:1593-1597`), `FP4_COMPUTE_MMA` (`:1601-1635`): unchanged. After `FP4_TRANSPOSE`, `smem_Bp` is byte-identical to what the working `[N,K/2]` FP4 kernel produced → `b0=FP4_FRAG(smem_Bp,nc,tid*8)`, `b1=FP4_FRAG(smem_Bp,nc,32+tid*8)` are single aligned u32 reads.
- Scale assembly `sfb_raw` from `smem_Bs[cur][0..3][sfn]` (`:1617-1621`): unchanged (orientation already matches).
- gate/up fusion (`is_up`, pointer/scale/C selection `:1444-1454`), in-kernel gather (`:1476-1482`), writeback (`:1671-1683`): unchanged.
- **scale2 at writeback** (`acc*scale2`, `:1679`): unchanged in form, but now `scale2` is the **real per-expert value** from the shared table (see §7 verification).

---

## 3. Down kernel: `moe_w4a16_down_t_k64_fp4` (SAME port — it is NOT already done)

The existing down kernel reads `[N,K/2]` (`:1791`, `B[gns*half_K+(gke>>1)]`) fed by `build_fp4_down`'s `[N,K/2]` re-pack — same mismatch as gate_up. `down_ptrs_t` (from `transpose_for_gemm(down_proj, h, inter)`) is `[K/2,N]` with N=hidden. Apply the **identical** §2.2-2.5 treatment: coalesced K-major load + `FP4_TRANSPOSE` + verbatim MMA/scale. Differences only: N=hidden=2048, K=inter=512 (→8 k64 steps), `sorted_token_ids=null`, single output (no gate/up split, no `is_up`). Keep the symbol unchanged.

---

## 4. Rust dispatch changes — `crates/spark-model/src/layers/moe/forward_prefill_routed.rs`

**gate_up** (Branch 1, `:115-146`): repoint the FP4 fused branch from `fp4.gate_t`/`fp4.up_t` to **`self.gate_ptrs_t`/`self.up_ptrs_t`**. Gate condition becomes: `self.gate_ptrs_t.is_some() && self.up_ptrs_t.is_some()` (FAST_MOE=full active) AND `holo_moe_gateup_fp4()` (env `ATLAS_HOLO_MOE_GATEUP_FP4`) AND `self.moe_fused_gate_up_t_k64_fp4.0 != 0`. Keep it **above** the FP8 `_t` branch. Call the same `ops::moe_w4a16_fused_gate_up_k64_n128` wrapper with the FP4 handle + `gate_ptrs_t`/`up_ptrs_t` tables. The FP8 `_t` branch (`:147`) is unchanged and serves as fallback when FP4 env is off — both read the same `gate_ptrs_t` bytes, selected by handle.

**down** (Branch 1, `:254-274`): repoint from `fp4d.down_t` to **`self.down_ptrs_t`**; gate on `down_ptrs_t.is_some()` AND `holo_moe_down_fp4()` AND `self.moe_down_t_k64_fp4.0 != 0`. Same `_n128` wrapper.

The legacy `[N,K/2]` FP4 branches and `build_fp4_*` tables are retained **only** for the FAST_MOE=off path (out of scope here); under FAST_MOE=full they are not built (§5).

---

## 5. Rust builder + memory — `crates/spark-model/src/weight_loader/qwen35/load_layers.rs`

**Skip `build_fp4_gate_up`/`build_fp4_down` entirely under FAST_MOE=full.** At the call sites (`:256`, `:266`), gate them on `!skip_moe_prefill_copies` / FAST_MOE-mode != full, i.e. only run the re-pack on the off path that actually needs `[N,K/2]` tables. Under FAST_MOE=full the FP4 kernel consumes `gate_ptrs_t`/`up_ptrs_t`/`down_ptrs_t` (already built by `transpose_for_prefill` at `:293`).

**No new table builder, no new packer.** `pack_bf16_weight_to_nvfp4_t` (emits `[N,K/2]` — wrong) and `transpose_nvfp4_packed_kton` (goes `[K/2,N]→[N,K/2]` — wrong direction) are NOT used.

**Free plan:** nothing new to free. The memory win is **not allocating** the duplicate `[N,K/2]` FP4 `_owned` buffers. Source NVFP4 experts (`expert.*_proj`, aliased by decode `gate_ptrs`/`up_ptrs`/`down_ptrs` at `init.rs:24-26`) stay resident — they still feed decode (`forward.rs:355-426`) and token-major (`forward_token_major.rs`). No dangling-pointer machinery; reject the "free originals after build" plan (it risked dangling the decode aliases). Whatever source-expert freeing FAST_MOE=full already performs (its existing Phase-B/D frees in `transpose_for_prefill`) is unchanged and out of scope.

**Doc fix:** `crates/spark-runtime/src/cutlass.rs:339-341` — change `[K/2,N]` → `[N,K/2]` for `pack_bf16_weight_to_nvfp4_t`.

---

## 6. Validation — `crates/spark-model/examples/moe_fp4_shapetest.rs`

Add cases (the kernels keep their symbols, so the existing Phase-2 fused block `:465-517` and down block `:648-809` already launch them; extend the weight prep + gates):

1. **Pack weights with `transpose_for_gemm`'s `[K/2,N]` output** (the production table), NOT `pack_bf16_weight_to_nvfp4_t`. (If the harness lacks a direct `transpose_for_gemm` call, host-byte-transpose a `pack_bf16_weight_to_nvfp4_t` `[N,K/2]` result to `[K/2,N]` to match.)
2. **Tightest gate (from Design 2):** assert the new kernel's C is **cos ≥ 0.999 / bit-identical vs the FP8 `_t` kernel reading the IDENTICAL `gate_ptrs_t` table** — this single assertion catches both nibble/k-order bugs in `FP4_TRANSPOSE` and any scale2 discrepancy.
3. cos ≥ 0.98 vs bf16 fp32-accum oracle; cos ≥ 0.999 vs CUTLASS collective FP4.
4. **Add a `num_experts>1` + non-trivial `sorted_token_ids` permutation case** (current harness only tests 1 expert/identity — exercises the in-kernel gather/scatter).
5. Speed: `fp4/fp8` time ratio > 1 at M=64; M∈{32,64,128,2048}.

Acceptance: `RESULT: PASS`, FP4 faster than FP8 at the shared layout.

---

## 7. Top correctness risks + mitigations

1. **Nibble pairing / k-order in `FP4_TRANSPOSE`** (highest — feeding FP8-packed bytes to an e2m1 MMA). Mitigation: `transpose_for_gemm` is a pure byte move preserving `(2j,2j+1)` pairing (verified `quantized.rs:119`), so the byte copy needs no swap; the §6.2 bit-identical-vs-FP8-`_t`-on-same-table gate catches any off-by-one with a 1-line `swap_nibbles` fix if needed.
2. **scale2 = real-value vs hardcoded-1.0.** `gate_ptrs_t` carries the real per-expert `weight_scale_2`; the legacy FP4 builder used 1.0. **Before wiring, read `transpose_for_gemm` once** to confirm whether it pre-divides nibbles by scale2 or leaves it as a post-scale, and **match exactly what the FP8 `_t` kernel does** (FP8 folds `sv=f*scale2` into dequant `:821`). For the FP4 path, apply `scale2` at writeback (`acc*scale2`); equivalence holds because `scale2` is a per-expert scalar that distributes out of the dot product. The §6.2 vs-FP8 assertion verifies the constant numerically.
3. **Single-buffered `smem_Bp` race + bank conflicts.** Mitigation: identical fence structure to FP8's single-buffered `smem_B_fp8` — post-transpose `__syncthreads` before MMA, regenerate only after `COMPUTE_MMA(cur)`. `BP_PAD=16`/stride-48 padding keeps the 128-lane writes off shared banks; the K-major reads are the FP8-proven conflict-free pattern. Validate with `sudo ncu --set full` l1tex conflict + occupancy counters on one launch before the e2e A/B.

---

## 8. Build / ship sequence

1. Edit the two kernel bodies + dispatch + builder gating + doc fix on dgx-00; **rsync to gx10-9959**.
2. Build via the CLAUDE.md GB10 block (`ATLAS_TARGET_MODEL=holo-3.1-35b-a3b`, `CUTLASS_HOME`, `FLASHINFER_HOME`, NCCL `RUSTFLAGS`, `--no-default-features --features cuda`); confirm log `compiled N kernels for target 0 (gb10, holo-3.1-35b-a3b, nvfp4)`.
3. `moe_fp4_shapetest` (§6) — cosine + bit-identical-vs-FP8 gates pass, FP4 faster than FP8.
4. e2e: `ATLAS_HOLO_FAST_MOE_MODE=full ATLAS_HOLO_MOE_GATEUP_FP4=1 ATLAS_HOLO_MOE_DOWN_FP4=1` — token-for-token greedy logits **identical** vs FP8-full; prefill tok/s (target ~3140→~4400); **MoE residency unchanged vs FP8-full** (the shared-table proof); Saturn image-correctness smoke test.

### Files to touch (absolute)
- `/home/ms/atlas/kernels/gb10/holo-3.1-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu` — rewrite B-load + add `FP4_TRANSPOSE` (replace `FGU64_DEQUANT` slot) in `moe_w4a16_fused_gate_up_t_k64_fp4` (~1409) and `moe_w4a16_down_t_k64_fp4` (~1704); reuse `FP4_QUANT_A`/`FP4_FRAG`/`FP4_COMPUTE_MMA`/scale path verbatim.
- `/home/ms/atlas/crates/spark-model/src/layers/moe/forward_prefill_routed.rs` — repoint FP4 gate_up (`:115`) to `gate_ptrs_t`/`up_ptrs_t`, FP4 down (`:254`) to `down_ptrs_t`.
- `/home/ms/atlas/crates/spark-model/src/weight_loader/qwen35/load_layers.rs` — skip `build_fp4_gate_up`/`build_fp4_down` under FAST_MOE=full (`:256`, `:266`).
- `/home/ms/atlas/crates/spark-runtime/src/cutlass.rs` — fix `:339-341` docstring (`[K/2,N]`→`[N,K/2]`).
- `/home/ms/atlas/crates/spark-model/examples/moe_fp4_shapetest.rs` — add shared-table (`transpose_for_gemm`-packed) + bit-identical-vs-FP8 + multi-expert/gather cases.
- `/home/ms/atlas/crates/spark-model/src/layers/moe/init.rs` — handles unchanged (symbols kept).

## RUNBOOK

I have everything I need. The key facts are confirmed:

1. **The implemented kernel reads [N,K/2]** (`B_expert[gns*half_K + (gke>>1)]`), NOT the shared [K/2,N] tables. No FP4_TRANSPOSE pass.
2. **Dispatch reads `fp4.gate_t` / `fp4d.down_t`** — fresh [N,K/2] tables built by `build_fp4_gate_up`/`build_fp4_down`, which re-pack every expert (~30GB duplicate under FAST_MOE=full).
3. **`cutlass.rs:339-341` doc still says `[K/2,N]`** but the CUDA emits [N,K/2] — stale.
4. **`build_fp4_*` gated only on env flags**, no FAST_MOE-mode guard → silent OOM risk under full.
5. **Example only tests num_experts=1**, identity permutation — gather/scatter untested.
6. The holo dir file is a **symlink** to qwen3.6 — editing either updates both; only the qwen3.6 real file needs rsync. **The work is already committed** (clean tree).

The reviewers "load-and-smem" and "mma-and-scales" are **correct**; "rust-dispatch-memory" (marked likely-wrong) is actually **also correct** on the substantive points. The task's IMPLEMENTATION block describing `gate_ptrs_t`-sharing / `nvfp4_gate_up_fp4` bools is fiction — the real code uses `fp4_gate_up: Option<MoeFp4GateUp>` reading [N,K/2] re-pack tables.

Below is the runbook and consolidated fix-list.

---

# RUNBOOK — Land transposed-FP4 MoE on GB10 (`gx10-9959`)

## 0. STATE-OF-TRUTH (verified in this worktree, base 5318ceb, branch feat/container-perf-fp8kv)

- **The FP4 work is already committed** (clean `git status`; no pending diff). The kernels live in the real file `kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu`; `kernels/gb10/holo-3.1-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu` is a **symlink** to it. The Holo target maps to the qwen3.6 dir. **Rsync the qwen3.6 real file only.**
- **The shipped kernels read `[N,K/2]` (K-contiguous, N-major)** — `B_expert[gns*half_K + (gke>>1)]` at `moe_w4a16_grouped_gemm.cu:1525`. There is **no FP4_TRANSPOSE pass**. Dispatch feeds them from `fp4.gate_t` / `fp4d.down_t` (`forward_prefill_routed.rs:131-136, 263-265`), which `build_fp4_gate_up`/`build_fp4_down` (`helpers_a.rs:281,310 / 437`) build as a **fresh resident `[N,K/2]` FP4 copy of every expert** via `pack_bf16_weight_to_nvfp4_t`.
- **CONSEQUENCE: the memory win was NOT achieved.** Under `FAST_MOE=full` both the `[K/2,N]` `gate_ptrs_t` set (built by `transpose_for_prefill`) **and** the `[N,K/2]` FP4 copy are resident → ~30GB duplicate, likely OOM on the shared 121GB box. This is the **gating blocker** — do NOT run the e2e A/B with `ATLAS_HOLO_MOE_GATEUP_FP4=1 ATLAS_HOLO_FAST_MOE_MODE=full` until FIX-1 (below) is resolved, or it will OOM / blow up footprint.

So this runbook has two arms:
- **Arm A — SAFE VALIDATION (do now):** numerics + isolated FP4-vs-FP8 kernel timing via `moe_fp4_shapetest`. Proves the kernels are correct and FP4 is faster *at the same layout*. Memory-safe.
- **Arm B — E2E PREFILL A/B (only after FIX-1 lands):** the `3140 → ~4400` claim. As-shipped this either OOMs or carries the duplicate; the headline win does not exist yet.

---

## 1. RSYNC dgx-00 → gx10-9959 (separate filesystems)

Run from `dgx-00` (this shell). Push only the changed source (the symlink target + Rust + example). Do NOT rsync the holo symlink as a file.

```bash
cd /home/ms/atlas

# Kernel (the REAL file; holo path is a symlink to it on both hosts)
rsync -avz kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu \
  gx10-9959:~/atlas/kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu

# Rust: dispatch, handles, init, builders, ops, example, Cargo.toml, cutlass doc
rsync -avz \
  crates/spark-model/src/layers/moe/forward_prefill_routed.rs \
  crates/spark-model/src/layers/moe/mod.rs \
  crates/spark-model/src/layers/moe/init.rs \
  crates/spark-model/src/layers/moe/helpers_a.rs \
  crates/spark-model/src/layers/moe/ops.rs \
  gx10-9959:~/atlas/crates/spark-model/src/layers/moe/

rsync -avz crates/spark-model/src/weight_loader/qwen35/load_layers.rs \
  gx10-9959:~/atlas/crates/spark-model/src/weight_loader/qwen35/load_layers.rs

rsync -avz crates/spark-model/examples/moe_fp4_shapetest.rs \
  gx10-9959:~/atlas/crates/spark-model/examples/moe_fp4_shapetest.rs

rsync -avz crates/spark-model/Cargo.toml \
  gx10-9959:~/atlas/crates/spark-model/Cargo.toml

rsync -avz crates/spark-runtime/src/cutlass.rs \
  gx10-9959:~/atlas/crates/spark-runtime/src/cutlass.rs
```

(If you prefer a whole-tree sync, exclude build/heavy dirs: `rsync -avz --exclude target --exclude .git --exclude '*.safetensors' /home/ms/atlas/ gx10-9959:~/atlas/`.)

---

## 2. BUILD on gx10-9959 (the EXACT CLAUDE.md block — all flags mandatory)

```bash
ssh gx10-9959 'cd ~/atlas && source ~/.cargo/env
  export PATH=/usr/local/cuda/bin:$PATH
  export CUTLASS_HOME=$HOME/cutlass
  export FLASHINFER_HOME=$HOME/flashinfer
  export RUSTFLAGS="-L/home/ms/nccl/build/lib -L/usr/local/cuda/lib64"
  export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4
  cargo build --release -p spark-server --bin spark --no-default-features --features cuda'
```

Sanity: the build log MUST contain `compiled N kernels for target 0 (gb10, holo-3.1-35b-a3b, nvfp4)`. The two new symbols `moe_w4a16_fused_gate_up_t_k64_fp4` and `moe_w4a16_down_t_k64_fp4` auto-register in module `moe_w4a16` (no KERNEL.toml edit — confirmed the .toml has no per-symbol entry for the existing `_t` kernels either). If the build complains about the mxf4nvf4 MMA, confirm `-arch=sm_121f` (GB10 = sm_121).

---

## 3. ARM A — VALIDATE: `moe_fp4_shapetest` (memory-safe, do now)

```bash
ssh gx10-9959 'cd ~/atlas && source ~/.cargo/env
  export PATH=/usr/local/cuda/bin:$PATH
  export CUTLASS_HOME=$HOME/cutlass FLASHINFER_HOME=$HOME/flashinfer
  export RUSTFLAGS="-L/home/ms/nccl/build/lib -L/usr/local/cuda/lib64"
  export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4
  cargo run --release --example moe_fp4_shapetest \
    --no-default-features --features "cuda gpu-examples"'
```

**PASS THRESHOLDS** (asserts already in `examples/moe_fp4_shapetest.rs`):
- `fused_cos_vs_collective >= 0.999` and `grp_cos_vs_collective >= 0.999` — FP4 fused/grouped vs the FP8 `_t` kernel on the **identical** table. This is the tight gate that catches FP4 nibble-order / scale2 bugs. **Must hold for M ∈ {32, 64, 128}, both gate_up and down shapes.**
- `fused_cos_vs_oracle >= 0.98` and `grp_cos_vs_oracle >= 0.98` — vs bf16 fp32-accum oracle (`COSINE_GATE = 0.98`).
- **Timing:** `fp4_us < fp8_us` at each M (the ~1.4× isolation win; expect ~55µs FP4 vs ~78µs FP8 at M=64).

> COVERAGE GAP (see FIX-4): the example only runs `num_experts=1`, `expert_offsets=[0,m]`, identity `sorted_token_ids`. The per-expert pointer-table indexing, multi-segment `expert_offsets` early-return, and in-kernel gather are **never exercised**. A gather/scatter bug passes this test and silently corrupts 256-expert routed output. **Add a `num_experts>=2`, non-identity-permutation case (cos≥0.999 vs FP8 `_t`) before trusting production output.**

---

## 4. ARM B — E2E Holo prefill A/B  (BLOCKED on FIX-1; see §6)

> DO NOT run with `ATLAS_HOLO_MOE_GATEUP_FP4=1` + `FAST_MOE_MODE=full` until FIX-1 lands: under full, `build_fp4_*` allocates a second `[N,K/2]` copy of all 256 experts (~30GB) on top of the `[K/2,N]` `gate_ptrs_t` set → OOM / double footprint, and there is **no guard** preventing it.

### B0. Baseline — FP8 fused, FAST_MOE=full (the number to beat: ~3140 tok/s)
```bash
ssh gx10-9959 'cd ~/atlas
  ATLAS_HOLO_FAST_MOE_MODE=full ATLAS_HOLO_FAST_MOE_LAYERS=0-39 \
  ATLAS_HOLO_MOE_GATEUP_FP4=0 ATLAS_HOLO_MOE_DOWN_FP4=0 \
  bash scripts/holo_serve.sh /tmp/holo_fp8full.log'
# wait for "serving" / :8890 bound, then:
ssh gx10-9959 'python3 /tmp/single_bench.py http://127.0.0.1:8890 fp8_full'
# record prefill tok/s (single 1403-tok req, max_tokens>=250 per Bench memory)
ssh gx10-9959 'pkill -9 -f "release/spark serve --model"'
```

### B1. Treatment — transposed-FP4 (only valid AFTER FIX-1 repoints dispatch to `gate_ptrs_t`)
```bash
ssh gx10-9959 'cd ~/atlas
  ATLAS_HOLO_FAST_MOE_MODE=full ATLAS_HOLO_FAST_MOE_LAYERS=0-39 \
  ATLAS_HOLO_MOE_GATEUP_FP4=1 ATLAS_HOLO_MOE_DOWN_FP4=1 \
  bash scripts/holo_serve.sh /tmp/holo_fp4full.log'
ssh gx10-9959 'python3 /tmp/single_bench.py http://127.0.0.1:8890 fp4_full'
ssh gx10-9959 'pkill -9 -f "release/spark serve --model"'
```

**ACCEPTANCE (Arm B):**
- Prefill tok/s **UP**: target ~3140 → ~4400 (MoE ≈ 28% of prefill × ~1.4× MoE → ~+11% blended; ~4400 is the optimistic ceiling). Any value **≤ 3140 is a FAIL** (regression — do not enable).
- **Memory:** preflight log "Atlas-own" footprint must NOT grow vs B0 (the entire point is zero extra MoE memory). If it grows ~30GB, FIX-1 is not done.
- **Correctness in-server:** the canonical image probe still returns the right answer (e.g. Saturn → "a planet with rings"); greedy decode token-stream unchanged vs B0 on a fixed prompt.

If, after FIX-1, you choose instead to ship the **[N,K/2] re-pack as a standalone FP4 path** (Arm B alt): it is ONLY admissible under `FAST_MOE_MODE=off` (no duplicate there), and the A/B must show FP4-off **> 3140** (FP8-full) to be worth enabling — today FP4-off ≈ 2552 < 3140, i.e. a net regression. So the [N,K/2] path as-is should stay **default-off**.

---

## 5. Monitoring helpers
```bash
# server up?
ssh gx10-9959 'pgrep -f "release/spark serve --model" || echo DOWN'
# footprint / KV budget from preflight
ssh gx10-9959 'grep -E "Atlas-own|KV|budget|free|util" /tmp/holo_fp4full.log | head'
# build kernel-count sanity
ssh gx10-9959 'grep "compiled .* kernels for target 0" /tmp/build.log'
```

---

# CONSOLIDATED FIX-LIST (blockers first)

The three reviewers AGREE on the substance (the "likely-wrong" tag on rust-dispatch-memory is mis-set — its blocker/major points are confirmed true in the live tree). Findings below are deduplicated.

### BLOCKER 1 — Memory win not realized: kernel reads `[N,K/2]`, dispatch builds a duplicate ~30GB FP4 table
The shipped kernels (`moe_w4a16_grouped_gemm.cu:1525`, B-load `B_expert[gns*half_K + (gke>>1)]`) consume `[N,K/2]`; dispatch (`forward_prefill_routed.rs:131-136, 263-265`) feeds them from `fp4.gate_t`/`fp4d.down_t`, freshly re-packed per-expert by `build_fp4_gate_up`/`build_fp4_down` (`helpers_a.rs:281,310,437`). Under `FAST_MOE=full` this is **additive** to the `[K/2,N]` `gate_ptrs_t` set → ~30GB duplicate, the exact thing the task set out to remove. **The `3140→~4400` win does not exist as shipped.**
**Fix (do this to meet the task goal):** change the B-load in BOTH `_fp4` kernels to the K-major coalesced read of the shared `[K/2,N]` tables (`B_expert[(gke>>1)*N + gns]` into a `smem_BpT` staging buffer), add an on-chip **FP4_TRANSPOSE** pass (mirror the FP8 `_t` kernel's dequant slot) that re-gathers into the N-major `smem_Bp` each m16n8k64 B-fragment expects, repoint dispatch to `self.gate_ptrs_t/up_ptrs_t/down_ptrs_t`, read the **real per-expert `scale2`** from those tables (NOT 1.0 — see Blocker 2), and **stop calling `build_fp4_*` under full**. Re-validate bit-identical vs FP8 `_t` on the shared table (the `cos>=0.999` gate).

### BLOCKER 2 — `scale2` hardcoded to `1.0`; correct ONLY for the [N,K/2] re-pack, WRONG on the shared tables
`build_fp4_*` push `1.0` for every expert (`helpers_a.rs:332,335,341,345; ~458,464`) and the kernels apply `acc*scale2` at writeback. This is correct **today** because `pack_bf16_weight_to_nvfp4_t` folds the full dynamic range into the per-group ue4m3 scale and emits no global weight_scale_2. But when Blocker 1 is fixed and dispatch reads `gate_ptrs_t`, the **real per-expert weight_scale_2 carried by `transpose_for_gemm`** must be used or results are wrong. **Tie this fix to Blocker 1** (it's the same code path); guard with the `cos>=0.999` vs-FP8 gate.

### MAJOR 3 — `build_fp4_*` gated only on env flags, NOT on FAST_MOE mode → silent ~30GB OOM
`load_layers.rs:282,291` call `build_fp4_gate_up`/`build_fp4_down` whenever `ATLAS_HOLO_MOE_GATEUP_FP4`/`_DOWN_FP4` is set, with **no** dependency on `holo_fast_moe_mode`, despite the comments (`:280,289`) claiming "intended for FAST_MOE=off". So the documented prod combo `FP4=1 + FAST_MOE=full` double-allocates with no warning.
**Fix (do BEFORE any e2e run, even if Blocker 1 is deferred):** gate the `build_fp4_*` calls on `skip_moe_prefill_copies` (i.e. only build when FAST_MOE=off and the FP4 tables will actually be the resident set), OR at minimum `tracing::warn!` (better: hard error) when an FP4 flag is set with `holo_fast_moe_mode.is_some()`. This converts a silent OOM into a guard.

### MAJOR 4 — Validation example never exercises multi-expert gather/scatter
`moe_fp4_shapetest.rs` (`:417,442,~507`) only runs `num_experts=1`, `expert_offsets=[0,m]`, identity `sorted_token_ids`. The per-expert ptr-table indexing, multi-segment early-return, and in-kernel gather (`smem_tok_fp4[local_row]=sorted_token_ids[...]`) are untested — a gather bug passes and corrupts 256-expert routed output.
**Fix:** add a `num_experts>=2` case with a real boundary (e.g. `expert_offsets=[0,m/2,m]`) and a non-identity permutation, asserting FP4 fused gate_up + FP4 down row-for-row vs the FP8 `_t` kernels on the identical table (`cos>=0.999`).

### MINOR 5 — Stale doc on `pack_bf16_weight_to_nvfp4_t` (the layout this whole task hinges on)
`cutlass.rs:339-341` says packed `[K/2,N]`; the CUDA (`cutlass_nvfp4_gemm.cu:206/216`) emits packed **`[N,K/2]`** (N-major, K-contiguous) + scales `[K/16,N]`. This is the exact `[N,K/2]`-vs-`[K/2,N]` confusion that misled the implementation.
**Fix:** correct the doc to `[N,K/2]` + scales `[K/16,N]`.

### MINOR 6 — Kernel/spec/impl-block documentation drift
The task's IMPLEMENTATION block describes fields that **don't exist** (`nvfp4_gate_up_fp4`/`nvfp4_down_fp4` bools; kernel "reading `gate_ptrs_t`"; dispatch gated on `self.nvfp4_gate_up_fp4`). The real code uses `fp4_gate_up: Option<MoeFp4GateUp>`, gates on `.is_some()` + `moe_fused_gate_up_t_k64_fp4.0 != 0`, env consumed at build time. Also the kernel header comment correctly says `[N,K/2]`, directly contradicting the design spec — a future reader will be misled.
**Fix:** add a one-line note on each `_fp4` kernel that it is the **[N,K/2] (non-shared-table) variant**; reconcile the IMPLEMENTATION/spec text with the shipped code (or with the post-Blocker-1 code).

### MINOR 7 — No `K % 64 == 0` guard
The B-load predicate `(gke + 31 < K)` gates a full 32-K-element cp.async chunk all-or-nothing; a chunk straddling a non-64-aligned K boundary is dropped wholesale, leaving stale smem (no per-element mask). Holo is safe (gate_up K=2048, down K=512, both %64==0), but any future model reusing these symbols with non-64 K silently miscomputes.
**Fix:** host-side/compile-time `assert!(K % 64 == 0)` at the launch site (or document the constraint at the `extern "C"` / ops-wrapper site). No Holo-correctness change today.

---

## Recommended order to actually ship the win
1. **Fix Major-3 first** (one-line guard) — makes every subsequent run memory-safe.
2. Run **Arm A** (`moe_fp4_shapetest`) → confirm the [N,K/2] kernels are numerically correct + FP4<FP8 in isolation. This validates the MMA/quant/scale plumbing independent of the layout decision.
3. **Fix Blocker-1 + Blocker-2 together** (the transposed B-load + FP4_TRANSPOSE + repoint to `gate_ptrs_t` + real scale2 + drop `build_fp4_*` under full). Add **Major-4**'s multi-expert case and re-run Arm A (now also bit-identical on the shared tables).
4. Run **Arm B** e2e A/B → confirm prefill UP (~3140→~4400) AND Atlas-own footprint flat.
5. Clean up Minors 5/6/7.

Until step 3, FP4 must remain **default-off**; enabling it as shipped is a net regression (duplicate memory under full, or FP4-off 2552 < FP8-full 3140).

Key file paths: `/home/ms/atlas/kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu` (real file; holo path is a symlink), `/home/ms/atlas/crates/spark-model/src/layers/moe/forward_prefill_routed.rs`, `/home/ms/atlas/crates/spark-model/src/layers/moe/helpers_a.rs`, `/home/ms/atlas/crates/spark-model/src/weight_loader/qwen35/load_layers.rs`, `/home/ms/atlas/crates/spark-runtime/src/cutlass.rs`, `/home/ms/atlas/crates/spark-model/examples/moe_fp4_shapetest.rs`, `/home/ms/atlas/scripts/holo_serve.sh`.