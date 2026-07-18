# Qwen3.6-35B-A3B NVFP4 / GB10 decode pursuit — 2026-07-17

## Purpose and scope

This is the experiment ledger for the July 17 decode-performance loop.  It
records measurements and decisions for the **NVIDIA Qwen3.6-35B-A3B NVFP4**
checkpoint on one GB10.  It supersedes any interpretation of earlier
`unsloth/...` runs for this investigation: that was not the model under test.

The objective is concurrent decode at C=8.  The external reference is the
reported vLLM result of **270.1 aggregate generated tokens/s**.  Atlas's
canonical control is currently **172–175.5 aggregate tokens/s** under the
same model family and C=8 serving shape.  The remaining gap is therefore
roughly 35%, not a small tuning discrepancy.

This document deliberately separates production-safe changes, diagnostic
experiments, and rejected designs.  A number is not a baseline unless it was
reproduced end to end with the canonical model and serving configuration.

## Reproduction boundary

| Item | Value |
|---|---|
| Hardware | single NVIDIA GB10 |
| Model | NVIDIA Qwen3.6-35B-A3B NVFP4 checkpoint |
| Decode target | C=8, aggregate generated tok/s |
| Prompt/generation workload | approximately 2,498-token input and 256 generated tokens per request unless noted |
| Atlas runtime | CUDA 13.2 `atlas-gb10:b12x-ready` Docker image |
| MoE route shape | 256 experts, top-8 routing; at C=8 at most 64 routed rows before padding |
| Comparator | vLLM Marlin NVFP4 MoE backend; reported C=8 reference 270.1 tok/s |

The CUDA 13.2 container is material: it provides the compatible NVCC,
CUTLASS, and link environment used to build Atlas targets and the raw-pointer
Marlin bridge.  Host-only builds are suitable for Rust formatting/checks but
not for making GPU performance claims.

## Benchmark results

### Valid comparison points

| System / configuration | C=8 decode aggregate tok/s | Status |
|---|---:|---|
| vLLM reference, no speculative decode (reported) | 270.1 | Target reference |
| Atlas canonical NVIDIA-NVFP4 Marlin control | 172–175.5 | Reproduced control range |
| Atlas historical canonical best base | approximately 180.2 | Historical; do not use as the current control without rerun |
| Atlas token-major result from the initial measurement | 126.7–137 | Earlier flag/configuration context; not comparable to the later Marlin control |
| Fixed-grid scalar persistent-worker prototype | 76.0 | Reproduced; rejected for throughput |

The persistent-worker measurement used the 2,498/256 workload and completed:

| C | Prefill aggregate tok/s | Mean TTFT | Decode aggregate tok/s | Decode/request |
|---:|---:|---:|---:|---:|
| 8 | 5,599 | 4.35 s | 76.0 | 9.5 |

### Numbers explicitly not used as evidence

The previously observed 207–208 tok/s values were not reproducible as a
canonical end-to-end Atlas run.  They are not treated as a performance
baseline, nor as proof that only numerical cleanup remains.  There is no
validated numerical mode that preserves the 207–208 rate.

Likewise, a vLLM log inspected during this loop showed MTP enabled.  It is
useful confirmation that vLLM selected the Marlin backend, but it must not be
used to equate its throughput directly with the no-spec 270.1 reference.

## What shipped safely

### Token-major batched MoE decode (`14af3008`)

The prior default issued a per-token MoE loop with aliased scratch buffers:
at C=8 this meant 24 serialized launches per layer and left the weighted
blend with only eight CTAs.  The replacement batches all tokens' experts into
roughly three launches per layer.

Measured at the time:

| C | Old path | Token-major path | Change |
|---:|---:|---:|---:|
| 4 | 83 | 103 tok/s | +24% |
| 8 | 128 | 137 tok/s | +7% |

Greedy output was byte-identical to the old path in the checked base and MTP
cases.  `ATLAS_MOE_LEGACY_PERTOKEN_DECODE=1` remains the kill switch.  This
is the production default and remains a real launch-overhead improvement even
though it does not close the vLLM gap.

### Marlin AOT bridge and diagnostics

Atlas's Qwen3.6 NVFP4 C=4–8 path uses a raw-pointer CUDA 13.2 bridge around
the vLLM/Marlin kernel family.  It repacks weights once at load, uses a
graph-capturable route-alignment kernel, runs Marlin W13 and W2, and performs
activation/blending in Atlas.  All 40 layers were verified to dispatch this
Marlin path at C=8.

Two explicitly opt-in diagnostic controls were added:

| Commit | Control | Result |
|---|---|---|
| `2067060b` | `ATLAS_MARLIN_VLLM_AUTO_CONFIG=1`: consider vLLM's three small-batch tile geometries and occupancy limit | about 174.2 tok/s; no win |
| `4422a048` | `ATLAS_MARLIN_VLLM_FULL_SHARED=1`: reserve vLLM-style full opt-in shared memory | about 175.1 tok/s; no win |

Those tests rule out these two simplified launch-policy changes as the
explanation for a 35% gap.  They do **not** establish exact equivalence to
vLLM's complete dispatcher; that still needs a targeted, per-GEMM trace and
comparison before further changes are promoted.

## Measurements that changed the diagnosis

### Rejected bottleneck theories

| Theory | Measurement / outcome | Disposition |
|---|---|---|
| GDN scan dominates decode | About 5% in the measured slice; the spilling candidate was dormant while production used the clean kernel | Rejected |
| Attention is the primary gap | GQA packing lost in two experiments | Rejected for this target |
| `h_state` should be lower precision | vLLM keeps it FP32 as well | Rejected |
| Router MMA is the large win | Experiment did not win and was removed in `4199eb42` | Rejected |
| Generic compact K=64 MoE is the right decode GEMM | Measured around 149 tok/s | Rejected |
| Fused batched SSM C=8 closes the gap | 149.9 vs 172.1 tok/s (about -13%) | Rejected |
| Full Marlin shared-memory reservation | No material gain | Rejected |

The earlier component subtraction pointed at the MoE FFN as roughly 57% of a
decode step, primarily because of launch serialization rather than poor
expert-GEMV occupancy.  That led directly to token-major decode.  It remains
the best kernel leverage point, but it does not mean every MoE-shaped kernel
is faster.

### Whole-step profile context

One long-context C=8 profile measured approximately 40,195 microseconds per
step:

| Region | Time | Interpretation |
|---|---:|---|
| 30 SSM/GDN layers | 20,567 us | Major remaining whole-step cost |
| 10 attention layers | 11,630 us | Material, but GQA experiments did not help |
| Head / remaining work | 7,998 us | Non-zero floor |

Within a representative SSM mixer, QKVZ was about 135 us, convolution about
30 us, GDN+norm about 123–135 us, and output projection about 50 us.  These
numbers mean a perfect MoE win must still be evaluated against a substantial
SSM/attention floor; they do not invalidate MoE work, whose calls are embedded
in the wider layer schedule.

## Persistent-worklist experiment

### Why it was built

At C=8, 64 routes across 256 experts are sparse: experts generally do not
collide.  Token-major batching removes launch serialization but cannot create
weight reuse that the routing distribution does not provide.  A long-term
resident scheduler that consumes GPU worklists remains architecturally
interesting because it can eliminate host scheduling/relaunch overhead and
can execute only routed work.

The following commits establish the minimum ABI and experimental consumer:

| Commit | Contents |
|---|---|
| `2d27528b` | Unit-tested decode worklist contract: expert-major groups, 1–8 real rows, compact descriptors |
| `6ecc05be` | CUDA producer that creates unpadded C=8 expert worklists |
| `11b1300c` | Opt-in fixed-grid gate/up and down workers, gated by `ATLAS_MOE_PERSISTENT_DECODE=1` |

The worklist unit tests cover sparse C=8 routing (64 routes / 57 groups in a
representative case) and descriptor packing.  The new kernels compile in the
CUDA 13.2 target build.  Rust formatting, `spark-model` check, and the
worklist tests passed.  A C=1 response sanity check returned the expected
literal response.

### Result and decision

The C=8 measurement was 76.0 aggregate tok/s.  This is a decisive loss versus
the 172–175.5 Marlin control.  The prototype uses scalar/GEMV-like arithmetic;
its better scheduling cannot compensate for abandoning the tensor-core NVFP4
Marlin math.  It must not be enabled by default or cited as a performance
advance.  It remains opt-in only as a correctness/worklist reference for a
future tensor-core consumer.

The C=8 prototype has not yet passed a byte-for-byte output comparison against
the Marlin path.  Do not make a numerical-parity claim for it based on the
C=1 sanity response.

## Current technical conclusion

There are two separate tracks:

1. **Near-term throughput:** retain tensor-core NVFP4 Marlin-class execution
   and make the Atlas bridge demonstrably match the relevant vLLM launch and
   dataflow choices.  The next check is an exact per-W13/per-W2 trace of
   selected tile, CTAs-per-SM, dynamic shared-memory reservation, and
   effective padded route count.  The existing global auto-config probe is
   insufficient evidence of full dispatcher equivalence.
2. **Long-term architecture:** use the tested unpadded worklist as the input
   to a tensor-core grouped-GEMM/persistent consumer.  A scalar persistent
   kernel is not viable.  A real weight-deduplicating grouped GEMM only pays
   when routed experts collide more frequently (larger batch); at C=8 it is
   multi-week kernel work, not an overnight claim.

The important constraint is that C=8 has little inherent weight reuse: 64
top-k routes across 256 experts are normally close to unique.  The immediate
win must therefore come from matching or improving efficient tensor-core
execution and launch/dataflow overhead, not from assuming a large
expert-deduplication dividend exists at this batch size.

## Commit timeline

| Commit | Outcome |
|---|---|
| `14af3008` | Shipped token-major default and legacy kill switch |
| `c08219d6` | Staged C=8 concurrent NVFP4 decode investigation |
| `4199eb42` | Removed non-winning Router-MMA experiment |
| `40ea9757` | Added grouped Marlin stage profiling |
| `2067060b` | Added non-winning vLLM auto-config diagnostic |
| `75e468df` | Added decode checkpoint performance control |
| `39ed0f4c`, `fb1ce3a3`, `7cbb1e46`, `fc80323f`, `fa4ebe79` | Built/tuned compact NVFP4 worklist experiments; no winning result |
| `4422a048` | Added non-winning full-shared Marlin diagnostic |
| `2d27528b`, `6ecc05be`, `11b1300c` | Added persistent-worklist specification, producer, and rejected scalar consumer |

## Verification record

The changes in this loop were checked with the appropriate available layers:

```bash
cargo fmt --all -- --check
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo check -p spark-model
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
  cargo test -p spark-model layers::moe::decode_worklist --lib
```

CUDA targets, including the persistent-worker sources, were compiled inside
the CUDA 13.2 Docker image.  A broader Clippy run was blocked by a pre-existing
unrelated explicit-auto-deref lint in
`crates/spark-model/src/model/trait_impl/verify_e.rs`; that result is not
attributed to this work.

## Next benchmark gate

Do not merge another default-on kernel merely because it compiles.  The next
candidate must pass all of the following before it is promoted:

1. CUDA 13.2 container build of the actual target.
2. Numerical comparison against the current Marlin control for C=4 and C=8.
3. Canonical C=8, approximately 2,498/256 end-to-end measurement with the
   NVIDIA checkpoint, no speculative-decode conflation.
4. At least a reproducible improvement over the 172–175.5 control range.
5. Whole-step profile showing where the claimed improvement lands.
