#!/usr/bin/env bash
# Launch the Holo 3.1 Atlas POC server (GB10 hybrid MoE, 32K/C8).
# Detached via setsid per porting-doc note (nohup gets reaped before CUDA init).
set -u
LOG="${1:-/tmp/holo-atlas.log}"
BIN=/home/ms/atlas/target/release/spark
GPU_UTIL="${ATLAS_HOLO_GPU_UTIL:-0.70}"
MAX_SEQ_LEN="${ATLAS_HOLO_MAX_SEQ_LEN:-32768}"
MAX_SEQS="${ATLAS_HOLO_MAX_SEQS:-8}"
MAX_BATCH="${ATLAS_HOLO_MAX_BATCH:-8}"
MAX_PREFILL="${ATLAS_HOLO_MAX_PREFILL:-16384}"
LOW_MEMORY_MOE="${ATLAS_HOLO_LOW_MEMORY_MOE:-1}"
FAST_MOE_MODE="${ATLAS_HOLO_FAST_MOE_MODE:-full}"
FAST_MOE_LAYERS="${ATLAS_HOLO_FAST_MOE_LAYERS:-0-39}"
HYBRID_MOE_LAYOUT="${ATLAS_HYBRID_MOE_LAYOUT:-1}"
UNIFIED_MOE_LAYOUT="${ATLAS_UNIFIED_MOE_LAYOUT:-0}"
NATIVE_FP8_ATTN="${ATLAS_HOLO_NATIVE_FP8_ATTN:-1}"
ATTN_Q_T="${ATLAS_ATTN_PREFILL_Q_T:-1}"
ATTN_T_PIPE="${ATLAS_ATTN_PREFILL_T_PIPE:-1}"
EXACT_MOE_TILES="${ATLAS_MOE_PREFILL_EXACT_TILES:-1}"
GDN_FUSED_NORM="${ATLAS_GDN_FUSED_NORM:-1}"
LM_HEAD_DTYPE="${ATLAS_HOLO_LM_HEAD_DTYPE:-bf16}"
# Native CUTLASS NVFP4 dense prefill projections (qkvz/Q/K/V/O + SSM out) — the
# default prefill path (+~24% prefill, op-level cos 0.99 vs bf16, server-validated
# coherent). REQUIRES the binary to be built with CUTLASS_HOME set; otherwise the
# nvfp4 GEMM bails at runtime. Override with ATLAS_CUTLASS_NVFP4_GEMM=0 to fall back.
CUTLASS_NVFP4_GEMM="${ATLAS_CUTLASS_NVFP4_GEMM:-1}"
CUTLASS_NVFP4_SSM_OUT="${ATLAS_CUTLASS_NVFP4_SSM_OUT:-1}"
# cudart/cublasLt (+nccl) for the runtime dynamic links; keep any caller value.
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/home/ms/nccl/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Self-clean: kill any prior server (setsid -f detaches, so a stale instance
# survives restarts and keeps :8890 bound — the new launch then orphans).
for pid in $(pgrep -f "release/spark serve --model" 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
sleep 2
setsid -f env RUST_BACKTRACE=1 RUST_LOG=info \
  ATLAS_DECODE_GRAPHS_MULTISEQ=1 ATLAS_HOLO_FP8_SSM_DECODE=1 \
  ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
  ATLAS_FAST_LOAD_PREFETCH_SHARDS=1 ATLAS_HOLO_LOW_MEMORY_MOE="$LOW_MEMORY_MOE" \
  ATLAS_HOLO_FAST_MOE_MODE="$FAST_MOE_MODE" ATLAS_HOLO_FAST_MOE_LAYERS="$FAST_MOE_LAYERS" \
  ATLAS_HYBRID_MOE_LAYOUT="$HYBRID_MOE_LAYOUT" ATLAS_UNIFIED_MOE_LAYOUT="$UNIFIED_MOE_LAYOUT" \
  ATLAS_MOE_PREFILL_EXACT_TILES="$EXACT_MOE_TILES" ATLAS_GDN_FUSED_NORM="$GDN_FUSED_NORM" \
  ATLAS_HOLO_NATIVE_FP8_ATTN="$NATIVE_FP8_ATTN" ATLAS_ATTN_PREFILL_Q_T="$ATTN_Q_T" ATLAS_ATTN_PREFILL_T_PIPE="$ATTN_T_PIPE" \
  ATLAS_CUTLASS_NVFP4_GEMM="$CUTLASS_NVFP4_GEMM" ATLAS_CUTLASS_NVFP4_SSM_OUT="$CUTLASS_NVFP4_SSM_OUT" \
  "$BIN" serve \
    --model-from-path /tank/holo-bf16kv-test --model-name holo3.1-atlas-poc \
    --port 8890 --bind 127.0.0.1 --max-seq-len "$MAX_SEQ_LEN" --max-num-seqs "$MAX_SEQS" --max-batch-size "$MAX_BATCH" \
    --max-prefill-tokens "$MAX_PREFILL" --kv-cache-dtype bf16 --lm-head-dtype "$LM_HEAD_DTYPE" \
    --gpu-memory-utilization "$GPU_UTIL" --oom-guard-mb 256 --ssm-cache-slots 0 --ssm-checkpoint-interval 0 \
    --enable-prefix-caching false --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --fast-load-prefetch-shards --vision-max-pixels 262144 \
    > "$LOG" 2>&1 </dev/null
echo "launched (log=$LOG)"
