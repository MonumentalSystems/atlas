# Kernel backend alternatives: FlashInfer and CUTLASS on GB10

**Audience:** Azeez, and anyone weighing a kernel-library swap on DGX Spark / GB10.
**Date:** 2026-07-27. **Hardware:** GB10 (sm_121), gx10-9959.
**Model under test:** `Hcompany/Holo-3.1-35B-A3B-NVFP4`.

## TL;DR

Neither dependency is the reason we're behind vLLM on concurrent prefill, and on
GB10 the usual "just use the other library" escapes mostly do not exist.

- **FlashInfer is carrying ~3% of prefill cost.** Replacing it cannot close the
  gap. It is also the one library with *explicit* sm_121 / DGX Spark support, so
  it is the last thing you should want to remove for performance reasons.
- **CUTLASS has no real alternative on sm_121 for FP4 grouped GEMM.** The
  obvious candidate (NVIDIA's trtllm-gen fused MoE, shipped via FlashInfer)
  falls back to a CUTLASS SM120 module on our arch. You would be swapping one
  CUTLASS configuration for another.
- **The bottleneck is the MoE grouped GEMM, and the evidence says it is
  occupancy/launch-bound, not math-bound.** A different GEMM library does not
  fix a tile-shape problem. Profile first.

There is exactly one dependency-removal worth doing on its own merits, and it is
motivated by *packaging cost*, not speed: finish wiring the `tc_vblock` wmma
`chunk_delta_h` kernel we already own and drop the FI-GDN toolchain.

## 1. What we actually depend on

### FlashInfer — two call sites, two env gates

| Gate | Use | Files |
|---|---|---|
| `ATLAS_GDN_FLASHINFER` | Chunked Gated DeltaNet scan | `layers/ops/gdn_flashinfer.rs`, `layers/qwen3_ssm/trait_prefill_gdn.rs` |
| `ATLAS_FLASHINFER_PREFILL` | Ragged prefill attention | `layers/qwen3_attention/prefill/paged.rs` |

That is the entire surface. It is a small, well-isolated dependency.

**Operational trap:** unsetting `ATLAS_GDN_FLASHINFER` does not error — it
silently falls back to the scalar FLA scan, which is 11–13× slower. Any
migration must fail loudly rather than degrade quietly.

### CUTLASS — the MoE grouped GEMM and friends

Required at build time (`CUTLASS_HOME`); without it the binary builds but
refuses to serve (`CUTLASS support was not built; set CUTLASS_HOME`). The
performance-critical consumer is the grouped NVFP4 MoE GEMM, reached via
`ATLAS_HOLO_LOW_MEMORY_MOE=1` / `ATLAS_HOLO_MOE_GROUPED_CUTLASS=1`. Without
those flags we silently fall back to a much slower w4a16 path.

## 2. The measurement that governs the decision

Measured this session on gx10, `bench/agentic/prefill_matrix.py`:

Aggregate prefill throughput is pinned at **~6.5K tok/s regardless of
concurrency**. At ISL 16384, C=1/2/4/8:

| Config | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|
| baseline (`--max-prefill-tokens 16384`) | 5,908 | 5,882 | 4,819 | 4,813 |
| arena 65536, prefix caching on | 5,908 | 5,882 | 4,819 | 4,813 |
| arena 65536, prefix caching **off** | 5,884 | 6,417 | 6,296 | 5,854 |

During a C=8 / ISL=16K prefill the GPU sits at **96% utilization, ~51 W,
2275 MHz**. The SMs are occupied essentially full-time — there is no idle
capacity for concurrency to fill, which is why unblocking batching removes the
cliffs but leaves the plateau untouched.

High utilization at low power is the signature of kernels that are resident but
not doing dense math: latency- or occupancy-bound, not compute-bound. Prior nsys
work attributes the cost accordingly — **MoE grouped GEMM 46–80% of prefill,
attention ~1.2%, GDN scan ~2%.**

Read those two facts together and the conclusion is forced: the two things
FlashInfer does for us total ~3% of prefill, and the thing CUTLASS does for us is
the whole problem — but it is a *tuning* problem, not a *library* problem.

## 3. FlashInfer alternatives

FlashInfer's attention path has explicit support for our arch, which is unusual
and worth preserving:

- `csrc/fmha_v2_run.cu:365` — "Map SM12x variants (e.g. SM121 on DGX Spark) to
  base SM120 for kernel dispatch."
- `csrc/trtllm_fmha_v2_binding.cu:290` — "Target architecture: SM120/SM121."

### For the GDN scan

| Option | Status | Verdict |
|---|---|---|
| **`tc_vblock` wmma `chunk_delta_h` (ours)** | Bit-parity + 1.4× at batch 2, isolated. e2e wiring is the only remaining step. | **Do this.** Faster than what it replaces *and* removes the dependency. |
| Triton FLA | vLLM's actual sm_121 path. Validated AOT-cubin integration plan; AOT launch proven on GB10. | Viable fallback, ~2 weeks. |
| Scalar FLA (existing) | 11–13× slower. | This is the floor, not an option. |

### For prefill attention

| Option | Verdict |
|---|---|
| cuDNN fused attention | The only serious non-CUTLASS vendor path (CUDA 13.x, Blackwell). Untested by us. |
| Triton attention | Portable, what vLLM uses. |
| FlashAttention 2 / 3 | FA2 sm_120/121 support is patchy; FA3 is Hopper-oriented. |
| Our hand-rolled per-Q-head kernel | This is what FlashInfer replaced for +12% @17K. Reverting is a measured regression. |

**Verdict:** keep FlashInfer for attention. The only migration worth funding is
the GDN one, and its justification is packaging, not throughput — FI-GDN
requires `libatlasgdn.so`, the cute runtime, the cuda-compat shim baked into the
image, and AOT export. That is real recurring build and deploy cost, and the
wmma kernel deletes it without giving up performance.

## 4. CUTLASS alternatives

The natural candidate is NVIDIA's trtllm-gen fused MoE, which FlashInfer ships
(`csrc/trtllm_fused_moe_runner.cu`). It does not help us:

```python
# flashinfer/fused_moe/core.py:277
if backend in ("120", "121"):
    module = gen_cutlass_fused_moe_sm120_module(use_fast_build).build_and_load()
```

On sm_121 the trtllm-gen path **falls back to CUTLASS**. Separately,
`flashinfer/fused_moe/api.py:333` notes the CuteDSL kernel "throws at launch on
SM120/SM121/SM130". NVIDIA's own stack offers no non-CUTLASS tensor-core FP4
grouped GEMM on this architecture.

| Option | Verdict |
|---|---|
| trtllm-gen fused MoE (via FlashInfer) | **Not an alternative on sm_121** — resolves to CUTLASS SM120. |
| CuteDSL | Throws at launch on SM120/121/130. |
| Triton | Real, and it is what vLLM runs on sm_121. The argument is *not* better kernels — it is that autotuning over tile shapes and launch configs becomes nearly free, which is precisely the lever the evidence points at. ~2 weeks. |
| cuBLASLt | Already the default for BF16 dense projections (+44% @C=1). Separately established as a **dead end for FP4 grouped**. Nothing left. |
| MARLIN | Parked, and it is the wrong lever: Atlas already has native-FP4 tensor-core MoE, whereas MARLIN is W4A16 software-dequant — a regression. |
| Hand-written CUDA | We already have ~30 MoE kernels. Hand-rolled mma.sync projections reached only ~30% of cuBLAS. This is the floor. |

**Verdict:** you cannot leave CUTLASS on this hardware for FP4 grouped MoE. The
question is whether to *retune* it or to move to Triton for the tuning
ergonomics.

## 5. Recommended order of work

1. **nsys capture of a C=8 / ISL=16K prefill.** Name the kernel burning the 96%.
   Hours of work, and it decides everything below. Do not skip it: the MARLIN
   investigation already concluded MoE is occupancy/launch-bound, and if that
   holds, no library swap helps.
2. **Small-M tile-shape sweep of the existing CUTLASS grouped kernel.** The
   cheapest possible test of the tile-shape hypothesis, using code we already
   ship.
3. **Finish `tc_vblock` wmma `chunk_delta_h` wiring.** Independent of the above.
   Removes the heaviest external dependency and is faster than FI-GDN.
4. **Triton integration — only if (1) and (2) say the config space is the
   problem and CUTLASS cannot reach it.** ~2 weeks. Fund it on the strength of
   the profile, not on the intuition that a different library is faster.

## 6. Two defects found while investigating

Worth filing regardless of which path is chosen:

- `ATLAS_MAX_BATCH_TOKENS` **silently ignores** a value below the derived
  default (warns, then proceeds with the default). It should be an error — a
  mis-set value looks exactly like a correctly-set one.
- `prefill_b/prefix_lookup.rs:364` rejects every chunk-0 batch when
  `num_ssm_layers() != 0`. Holo has 30 GDN layers of 40, so **Q12 batched prefill
  is dead code for every hybrid model whenever prefix caching is on** — which is
  our production config.

## Appendix: what is verified vs carried over

**Measured this session (2026-07-27, gx10, Holo-3.1-35B-A3B-NVFP4):** the prefill
matrix above; 96% util / 51 W / 2275 MHz under concurrent prefill; the two Q12
admission gates and their effect; all FlashInfer source citations
(`core.py:277`, `api.py:333`, `fmha_v2_run.cu:365`,
`trtllm_fmha_v2_binding.cu:290`); Atlas's FlashInfer surface area.

**Carried from earlier measured work, not re-confirmed here:** the nsys cost
decomposition (MoE 46–80%, attention 1.2%, GDN 2%); FI-GDN +10–20% prefill;
FlashInfer ragged prefill +12% @17K; `tc_vblock` bit-parity and 1.4× at batch 2;
Triton FLA AOT launch proven on GB10; hand-rolled mma.sync at ~30% of cuBLAS;
cuBLASLt BF16 projection win and the FP4 grouped dead end; MARLIN parked as
occupancy-bound; vLLM using Triton/FLA on sm_121.

**Not verified by us at all:** that vLLM's concurrent prefill at 16K is
substantially higher than ours — this is Richard's report from a separate
comparison and is the premise motivating the question. cuDNN fused attention on
GB10 is untested.
