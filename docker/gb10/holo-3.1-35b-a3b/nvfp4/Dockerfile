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
#   docker build -f docker/gb10/holo-3.1-35b-a3b/nvfp4/Dockerfile -t atlas-holo:cuda13.2 .
#
# Run (mount the WHOLE /tank, not just the model dir: the weights are symlinks
# into /tank/hf/hub/... so mounting only the model subdir leaves them dangling →
# "No safetensor files found"):
#   docker run -d --name atlas-holo --gpus all --ipc=host -p 8890:8890 \
#     -v /tank:/tank:ro \
#     -e ATLAS_HOLO_FAST_MOE_MODE=off -e ATLAS_KV_OVERCOMMIT=1 \
#     atlas-holo:cuda13.2 \
#     serve --model-from-path /tank/holo-bf16kv-test --model-name holo3.1-atlas-poc \
#           --port 8890 --bind 0.0.0.0 --kv-cache-dtype bf16 \
#           --gpu-memory-utilization 0.40 --max-seq-len 200000 --max-num-seqs 16 \
#           --max-batch-size 12 --scheduling-policy slai --tool-call-parser qwen3_coder \
#           --enable-prefix-caching true --ssm-cache-slots 32 --ssm-checkpoint-interval 256 \
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

# CUTLASS (NVFP4 GEMM) + FlashInfer (varlen ragged prefill), pinned to the
# known-good commits the bare-metal build validated under CUDA 13.2.
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass /opt/cutlass && \
    git -C /opt/cutlass checkout cf064d2e6bad2886238ac565b3b49007764f4939
RUN git clone --filter=blob:none https://github.com/flashinfer-ai/flashinfer.git /opt/flashinfer && \
    git -C /opt/flashinfer checkout a671c02ee2fbcdde7cc991f5a01c7cf5eb4a8972 && \
    git -C /opt/flashinfer submodule update --init --recursive --depth 1
ENV CUTLASS_HOME=/opt/cutlass
ENV FLASHINFER_HOME=/opt/flashinfer

# cudarc v0.19.2's build script hardcodes a CUDA-version allowlist and panics on
# "13.2" — pin the API version explicitly (the override CI uses; 13.0/13.2 cudarc
# bindings are compatible and it dlopens libcuda at runtime anyway).
ENV CUDARC_CUDA_VERSION=13000
# nccl from libnccl-dev (aarch64 = GB10/Grace) + cuda libs, for the -lnccl link.
ENV RUSTFLAGS="-L/usr/lib/aarch64-linux-gnu -L/usr/local/cuda/lib64"

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=holo-3.1-35b-a3b
ENV ATLAS_TARGET_QUANT=nvfp4

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
EXPOSE 8890

ENTRYPOINT ["spark"]
