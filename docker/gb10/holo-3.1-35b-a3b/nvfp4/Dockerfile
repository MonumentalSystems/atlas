# Atlas Spark — Holo-3.1-35B-A3B-NVFP4 on GB10 (CUDA 13.2)
#
# Hybrid model: 30 GDN/SSM + 10 full-attention + 256-expert MoE.
# Unlike the other gb10 per-model images, Holo REQUIRES the CUTLASS NVFP4 GEMM
# and FlashInfer varlen prefill paths, so this build pins CUTLASS + FlashInfer
# and uses the full `--no-default-features --features cuda` flags (a bare
# `cargo build -p spark-server` produces a binary that loads but panics on the
# first prefill: "CUTLASS support was not built").
#
# Built/validated on CUDA 13.2 (host driver may be 13.0 — minor-version compat
# within CUDA 13; vLLM runs a 13.2 container on the same GB10). If PTX-JIT fails
# to load on an older host driver, add the `cuda-compat-13-x` package to the
# runtime stage.
#
# Build from repo root:
#   docker build -f docker/gb10/holo-3.1-35b-a3b/nvfp4/Dockerfile -t atlas-holo:cuda13.2-perf .
#
# Run (mount the WHOLE /tank, not just the model dir: the weights are symlinks
# into /tank/hf/hub/... so mounting only the model subdir leaves them dangling →
# "No safetensor files found"). Validated config 2026-06-25 — 32K chunk, ~300K-KV
# static with overcommit (util 0.54), co-dispatch ON, prefix-caching OFF (REQUIRED
# while co-dispatch is baked; see the ENV note below):
#   docker run -d --name atlas-holo --gpus all --ipc=host -p 8890:8890 \
#     -v /tank:/tank:ro \
#     -e ATLAS_HOLO_FAST_MOE_MODE=full -e ATLAS_HOLO_FAST_MOE_LAYERS=0-39 \
#     -e ATLAS_KV_OVERCOMMIT=1 \
#     atlas-holo:cuda13.2-perf \
#     serve --model-from-path /tank/holo-bf16kv-test --model-name holo3.1-atlas-poc \
#           --port 8890 --bind 0.0.0.0 --kv-cache-dtype bf16 \
#           --gpu-memory-utilization 0.54 --max-seq-len 200000 --max-num-seqs 8 \
#           --max-batch-size 8 --max-prefill-tokens 32768 --scheduling-policy slai \
#           --tool-call-parser qwen3_coder --enable-prefix-caching false \
#           --default-chat-template-kwargs '{"enable_thinking":true}' --vision-max-pixels 262144

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev libnccl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
# Defer to the toolchain rustup installed (>=1.88 required by a transitive dep);
# overrides rust-toolchain.toml's older pin.
ENV RUSTUP_TOOLCHAIN=stable

# CUTLASS (NVFP4 GEMM) + FlashInfer (varlen ragged prefill). Default = the
# known-good commits the bare-metal build validated under CUDA 13.2. Override via
#   --build-arg CUTLASS_REF=v4.4.2 --build-arg FLASHINFER_REF=v0.6.13
# to test the newer releases (sm_121 NVFP4 grouped GEMM + Blackwell GDN prefill —
# the likely source of vLLM's 3x prefill).
ARG CUTLASS_REF=cf064d2e6bad2886238ac565b3b49007764f4939
ARG FLASHINFER_REF=a671c02ee2fbcdde7cc991f5a01c7cf5eb4a8972
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass /opt/cutlass && \
    git -C /opt/cutlass checkout ${CUTLASS_REF}
RUN git clone --filter=blob:none https://github.com/flashinfer-ai/flashinfer.git /opt/flashinfer && \
    git -C /opt/flashinfer checkout ${FLASHINFER_REF} && \
    git -C /opt/flashinfer submodule update --init --recursive --depth 1
ENV CUTLASS_HOME=/opt/cutlass
ENV FLASHINFER_HOME=/opt/flashinfer

# cudarc v0.19.2's build script hardcodes a CUDA-version allowlist and panics on
# "13.2" — pin the API version explicitly (the override CI uses; 13.0/13.2 cudarc
# bindings are compatible and it dlopens libcuda at runtime anyway).
ENV CUDARC_CUDA_VERSION=13000
# nccl from libnccl-dev (aarch64 = GB10/Grace) + cuda libs, for the -lnccl link.
ENV RUSTFLAGS="-L/usr/lib/aarch64-linux-gnu -L/usr/local/cuda/lib64"

# Kernel target — overridable so the same image builds sibling models (e.g. the
# dense Qwen3.5/Ornith VL family). Default = the 35B-A3B MoE. Override with:
#   --build-arg ATLAS_TARGET_MODEL=ornith-1.0-9b
# (the build context already COPYs all of kernels/ + crates/, so any target dir
# under kernels/gb10/<name>/ is buildable without further Dockerfile changes).
ARG ATLAS_TARGET_MODEL=holo-3.1-35b-a3b
ARG ATLAS_TARGET_QUANT=nvfp4
ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=${ATLAS_TARGET_MODEL}
ENV ATLAS_TARGET_QUANT=${ATLAS_TARGET_QUANT}

WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/

# Holo target build — the exact flags (omitting any produces a binary that
# loads but fails at runtime; see CLAUDE.md "BUILD"). BuildKit cache-mounts on
# the cargo registry + target dir make REBUILDS incremental (~2 min vs ~20):
# the dep graph + unchanged crates are cached across builds. The binary is
# copied OUT of the (non-persisted) target cache-mount to a real layer path so
# the runtime stage can COPY it. Requires BuildKit (default in modern docker).
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release -p spark-server --bin spark --no-default-features --features cuda && \
    cp /build/target/release/spark /usr/local/bin/spark

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04

LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

# NCCL ≥ 2.28 (symmetric-memory APIs) + RDMA userspace. EP=2 only; Holo POC is
# single-GPU but the floor check guards future multi-GPU.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    echo "Linked NCCL: $NCCL_VER" && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      { echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable" >&2; exit 1; }

# Binary copied out of the build cache-mount to /usr/local/bin/spark in the
# builder stage (the target/ cache-mount isn't persisted to a layer).
COPY --from=builder /usr/local/bin/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE /LICENSE
COPY README.md /README.md

ENV RUST_LOG=info

# ── Perf defaults (the holo_serve.sh-validated stack) ────────────────
# WITHOUT these the container runs Holo prefill ~2x slower than bare-metal:
# they are NOT code-defaults (holo_serve.sh sets them with `:-` fallbacks, so
# the in-binary default is OFF). Measured 2026-06-24 on a clean GPU: baking
# these took big-single (1978 tok) 575 -> 1120 tok/s (+95%) and varlen-x4 agg
# 550 -> 1225. All overridable via `-e`. cuBLASLt needs the CUDA lib path.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
# FP4 dense GEMM (CUTLASS NVFP4) is the PRIMARY projection path: on CUTLASS 4.4.2
# it beats cuBLAS by ~13% on large-M prefill (1978-tok big-single 1336->1516 tok/s,
# clean dgx-00 2026-06-24) — the version bump flipped the bare-metal verdict (old
# CUTLASS FP4 lost to cuBLAS). cuBLAS stays ON as the fallback for shapes FP4 skips.
# Large chunks are now SAFE: the old "status -2 above ~4-8K M" was a 64MB-workspace
# overflow guard in the NVFP4 GEMM wrapper, not a kernel limit — fixed by M-tiling
# into <=4096-row sub-GEMMs (cutlass_nvfp4_gemm.cu, commit 3a6bda8). The server
# also clamps --max-prefill-tokens to a block-aligned <65535 at startup (the prefill
# kernels map one grid-Y block per token; CUDA caps grid.y at 65535), so any chunk
# size — incl. the 32768 default — is safe with FP4 SSM-out ON.
ENV ATLAS_CUTLASS_NVFP4_GEMM=1 \
    ATLAS_CUBLAS_GEMM=1 \
    ATLAS_CUTLASS_NVFP4_SSM_OUT=1 \
    ATLAS_GDN_BATCHED_FLA=1 \
    ATLAS_GDN_FUSED_NORM=1 \
    ATLAS_MOE_PREFILL_FP8_DOWN=1 \
    ATLAS_MOE_PREFILL_EXACT_TILES=1 \
    ATLAS_HYBRID_MOE_LAYOUT=1 \
    ATLAS_UNIFIED_MOE_LAYOUT=0 \
    ATLAS_HOLO_LOW_MEMORY_MOE=1 \
    ATLAS_HOLO_NATIVE_FP8_ATTN=1 \
    ATLAS_ATTN_PREFILL_Q_T=1 \
    ATLAS_ATTN_PREFILL_T_PIPE=1 \
    ATLAS_DECODE_GRAPHS_MULTISEQ=1 \
    ATLAS_HOLO_FP8_SSM_DECODE=1 \
    ATLAS_HOLO_FP4_PROJ_DECODE=1 \
    ATLAS_PREFILL_CODISPATCH=1 \
    ATLAS_FLASHINFER_PREFILL=1 \
    ATLAS_Q12_BATCHED_FIRST_CHUNK=1 \
    ATLAS_HOLO_MOE_GROUPED_CUTLASS=1 \
    ATLAS_HOLO_MOE_GROUPED_DOWN=1
# CUTLASS 4.4.2 sm120 NVFP4 single-launch GROUPED MoE gate_up + down
# (ATLAS_HOLO_MOE_GROUPED_CUTLASS / _GROUPED_DOWN). Default-ON: correct + the FP4
# MoE win — single-stream prefill +11-12% vs FP8-full (8K/11K 4240/4222 vs
# 3824/3774), +9-12% concurrent at large prompts, parity at small (2026-06-26).
# Engages ONLY under ATLAS_HOLO_FAST_MOE_MODE=full (it shares gate_ptrs_t + builds
# per-expert swizzled SFB at load); a safe no-op otherwise (falls back to FP8/FP4
# fused). Needs CUTLASS 4.4.2 (this image) + the 512MB shared CUTLASS workspace
# (code default; override ATLAS_CUTLASS_WORKSPACE_MB). Beats the hand-rolled
# transposed-FP4 MoE (which only reached ~parity). The down path also keys on
# ATLAS_MOE_PREFILL_FP8_DOWN below as its fallback when grouped-down is off.
# Varlen co-dispatch (ATLAS_PREFILL_CODISPATCH / _FLASHINFER_PREFILL /
# _Q12_BATCHED_FIRST_CHUNK) is now BAKED ON: 2x agg throughput + 0 timeouts vs OFF
# under real concurrent mixed load (soak 2026-06-24), validated at parity inside
# this container 2026-06-25.
# ⚠️  It SEGFAULTS when prefix-caching is on (partial-prefill geometry bug). The
# default --enable-prefix-caching is FALSE, which is safe; do NOT pass
# --enable-prefix-caching true with this image until that bug is fixed.
# FAST_MOE_MODE is left to the run command (memory<->speed tradeoff for co-tenancy).

EXPOSE 8890

ENTRYPOINT ["spark"]
