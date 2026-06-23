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
# Prefill chunk (tokens/scheduler step). Default 2048 — the soak-validated +
# profile-derived sweet spot on the 64K target. Why 2048 specifically: the MoE
# routes top_k=8 over 256 experts, so a C-token chunk gives C/32 tokens/expert;
# the fused gate_up kernel tiles M in 64-blocks, so C=2048 → exactly 64 tok/expert
# = one FULL tile (100% MoE GEMM efficiency, 96% SM). C=1024 → 32/expert → HALF-
# empty tiles → ~50% MoE eff (~85% SM) AND 2× the kernel launches (1 CPU dispatch
# core is the bottleneck during prefill — no CUDA graphs there). Soak 2048 vs 1024:
# 24 vs 17 completions, big_ctx 53s vs 72s, needles 6/6 vs 4/4, tools 9/9 vs 3/4.
# 1024 only wins raw agg tok/s on a 128K-streaming worker (more decode interleaving)
# — not representative of agent traffic. HARD CEILING: the SSM out_proj CUTLASS
# NVFP4 GEMM (M=chunk) FAILS (status -2) above ~4-8K M, so chunk=16384 crashes any
# >~4K prompt with HTTP 500. Capped at 4096 below until that kernel tiles M.
MAX_PREFILL="${ATLAS_HOLO_MAX_PREFILL:-2048}"
# Bind address: 127.0.0.1 (loopback, default) or 0.0.0.0 to expose on the LAN
# (so it can be driven without an ssh tunnel). LAN exposure on an untrusted
# network should be paired with --require-auth/--auth-tokens-file.
BIND="${ATLAS_HOLO_BIND:-127.0.0.1}"
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
# Native CUTLASS NVFP4 dense prefill projections (qkvz/Q/K/V/O + SSM out) — the
# default prefill path (+~24% prefill, op-level cos 0.99 vs bf16, server-validated
# coherent). REQUIRES the binary to be built with CUTLASS_HOME set; otherwise the
# nvfp4 GEMM bails at runtime. Override with ATLAS_CUTLASS_NVFP4_GEMM=0 to fall back.
CUTLASS_NVFP4_GEMM="${ATLAS_CUTLASS_NVFP4_GEMM:-1}"
CUTLASS_NVFP4_SSM_OUT="${ATLAS_CUTLASS_NVFP4_SSM_OUT:-1}"
# Prefix caching (radix-tree KV reuse + Marconi SSM-snapshot restore for the
# hybrid GDN layers). Off by default. To enable, also give SSM-snapshot slots +
# a checkpoint interval so the recurrent state is restorable at prefix boundaries.
PREFIX_CACHING="${ATLAS_HOLO_PREFIX_CACHING:-false}"
SSM_CACHE_SLOTS="${ATLAS_HOLO_SSM_CACHE_SLOTS:-0}"
SSM_CKPT_INTERVAL="${ATLAS_HOLO_SSM_CHECKPOINT_INTERVAL:-0}"
# Scheduler: fifo (default) or slai (SLO-aware: decode-first near TBT deadline +
# shortest-prompt-first prefill ordering). slai prevents a giant prefill from
# starving concurrent decodes / blocking small prompts (the soak failure mode).
SCHED_POLICY="${ATLAS_HOLO_SCHED_POLICY:-slai}"
# MAX_PREFILL safety + prefix-cache alignment:
#  (1) CUTLASS cap — the SSM out_proj NVFP4 GEMM (M = chunk size) FAILS (status -2)
#      above ~4-8K M, so a chunk > 4096 crashes any prompt that prefills in one big
#      chunk (soak @16384: every >4K prompt returned HTTP 500). Hard-cap at 4096.
#  (2) Checkpoint alignment — Marconi intermediate checkpoints save only when a
#      chunk boundary lands on SSM_CKPT_INTERVAL*16 tokens, i.e. chunk must divide
#      (SSM_CKPT_INTERVAL*16). The 1024 default divides 4096 (interval 256) ✓, so a
#      checkpoint fires every 4 chunks; warm prefix hits skip in 4096-token steps.
if [ "$MAX_PREFILL" -gt 4096 ]; then
    echo "WARN: MAX_PREFILL=$MAX_PREFILL exceeds the SSM NVFP4 GEMM M-limit (~4-8K); capping to 4096 to avoid status-2 crash" >&2
    MAX_PREFILL=4096
fi
if [ "$PREFIX_CACHING" = "true" ] && [ "${SSM_CKPT_INTERVAL:-0}" -gt 0 ] \
   && [ $(( (SSM_CKPT_INTERVAL * 16) % MAX_PREFILL )) -ne 0 ]; then
    echo "WARN: MAX_PREFILL=$MAX_PREFILL does not divide checkpoint interval $((SSM_CKPT_INTERVAL*16))t; prefix-cache checkpoints may not fire" >&2
fi
# cudart/cublasLt (+nccl) for the runtime dynamic links; keep any caller value.
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/home/ms/nccl/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Self-clean: kill any prior server (setsid -f detaches, so a stale instance
# survives restarts and keeps :8890 bound — the new launch then orphans).
for pid in $(pgrep -f "release/spark serve --model" 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
sleep 2
setsid -f env RUST_BACKTRACE=1 RUST_LOG=info \
  ATLAS_DECODE_GRAPHS_MULTISEQ=1 ATLAS_HOLO_FP8_SSM_DECODE=1 ATLAS_KV_OVERCOMMIT="${ATLAS_KV_OVERCOMMIT:-0}" \
  ATLAS_KV_EXTERNAL_RESERVE_GB="${ATLAS_KV_EXTERNAL_RESERVE_GB:-0}" \
  ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
  ATLAS_FAST_LOAD_PREFETCH_SHARDS=1 ATLAS_HOLO_LOW_MEMORY_MOE="$LOW_MEMORY_MOE" \
  ATLAS_HOLO_FAST_MOE_MODE="$FAST_MOE_MODE" ATLAS_HOLO_FAST_MOE_LAYERS="$FAST_MOE_LAYERS" \
  ATLAS_HYBRID_MOE_LAYOUT="$HYBRID_MOE_LAYOUT" ATLAS_UNIFIED_MOE_LAYOUT="$UNIFIED_MOE_LAYOUT" \
  ATLAS_MOE_PREFILL_EXACT_TILES="$EXACT_MOE_TILES" ATLAS_GDN_FUSED_NORM="$GDN_FUSED_NORM" \
  ATLAS_HOLO_NATIVE_FP8_ATTN="$NATIVE_FP8_ATTN" ATLAS_ATTN_PREFILL_Q_T="$ATTN_Q_T" ATLAS_ATTN_PREFILL_T_PIPE="$ATTN_T_PIPE" \
  ATLAS_CUTLASS_NVFP4_GEMM="$CUTLASS_NVFP4_GEMM" ATLAS_CUTLASS_NVFP4_SSM_OUT="$CUTLASS_NVFP4_SSM_OUT" \
  "$BIN" serve \
    --model-from-path /tank/holo-bf16kv-test --model-name holo3.1-atlas-poc \
    --port 8890 --bind "$BIND" --max-seq-len "$MAX_SEQ_LEN" --max-num-seqs "$MAX_SEQS" --max-batch-size "$MAX_BATCH" \
    --max-prefill-tokens "$MAX_PREFILL" --kv-cache-dtype bf16 \
    --gpu-memory-utilization "$GPU_UTIL" --oom-guard-mb 256 --ssm-cache-slots "$SSM_CACHE_SLOTS" --ssm-checkpoint-interval "$SSM_CKPT_INTERVAL" \
    --enable-prefix-caching "$PREFIX_CACHING" --scheduling-policy "$SCHED_POLICY" --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --fast-load-prefetch-shards --vision-max-pixels 262144 \
    > "$LOG" 2>&1 </dev/null
echo "launched (log=$LOG)"
