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
# FP8 down-GEMM for MoE prefill: pre-quantizes the post-SiLU intermediate to FP8
# and routes the down projection through the faster fp8 grouped GEMM. ~14% faster
# down-GEMM (~1% total prefill), numerically VERIFIED clean — needle 3/3 @10/30/55K,
# trick-reasoning + tool-call all correct. Safe because the down activation feeds
# the residual (FP8-tolerant), unlike the gate_up INPUT activation which diverged
# under FP8. Default on; set 0 to disable.
FP8_DOWN="${ATLAS_MOE_PREFILL_FP8_DOWN:-1}"
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
# Prefill DENSE PROJECTIONS (SSM qkvz + attn Q/K/V/O). Two paths, A/B'd on a quiet
# GB10 (2026-06-24, prefill_ab/prefill_conc, lean config):
#   - cuBLASLt BF16 (ATLAS_CUBLAS_GEMM=1): NOW DEFAULT. c1 best 3165 tok/s, agg
#     2558 c4 / 2617 c8. W16A16 = MORE accurate than FP4. Costs only +0.6GB pre-KV
#     (bf16 weight-dequant cache). Wins every conc: c1 +44% (noisy) / c4 +4.5% /
#     c8 +13%. Requires a cuBLAS-linked binary (build after commit 0b32505).
#   - cutlass NVFP4 (ATLAS_CUTLASS_NVFP4_GEMM=1): prior default, slower at every
#     conc (c1 2202, c4 2448, c8 2311). REQUIRES CUTLASS_HOME at build. Kept as
#     fallback. To revert: ATLAS_CUBLAS_GEMM=0 ATLAS_CUTLASS_NVFP4_GEMM=1.
# SSM out_proj stays NVFP4 (separate gate below — not covered by the cuBLAS wiring).
CUBLAS_GEMM="${ATLAS_CUBLAS_GEMM:-1}"
# Concurrency — NOW DEFAULT ON. Co-dispatch batches concurrent prefills into one
# forward (+ FlashInfer ragged attention + chunk-0 batched fallback); the varlen
# batched-GDN scan (ATLAS_GDN_BATCHED_FLA) batches the ragged GDN. Soak A/B
# (2026-06-24, 6-client mixed, 768 thinking): co-dispatch ON = 24.3 tok/s agg / 0
# errors vs OFF = 11.5 / 4 timeouts — 2× and no head-of-line blocking under real
# concurrent load. (Synthetic prefill-only c4 TIES serial — GB10 has no L2 weight-
# reuse — which hid this; the real-world soak is what shows it.) Off→ set each =0.
CODISPATCH="${ATLAS_PREFILL_CODISPATCH:-1}"
FLASHINFER_PREFILL="${ATLAS_FLASHINFER_PREFILL:-1}"
Q12_BATCHED_FIRST_CHUNK="${ATLAS_Q12_BATCHED_FIRST_CHUNK:-1}"
GDN_BATCHED_FLA="${ATLAS_GDN_BATCHED_FLA:-1}"
CUTLASS_NVFP4_GEMM="${ATLAS_CUTLASS_NVFP4_GEMM:-0}"
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
# Always-on fused mixed step (decode keep-alive during prefill bursts). Default
# OFF — when off the scheduler is byte-identical to the resting production path
# (binary should_prefill, no slice budget). Set to 1/true to enable the
# slice-budget always-mixed path (see HOLO_MIXED_STEP_SPEC.md, Step 2).
ALWAYS_MIXED="${ATLAS_HOLO_ALWAYS_MIXED:-false}"
# MAX_PREFILL note (the old hard-cap at 4096 is LIFTED):
#  The CUTLASS NVFP4 GEMM "status -2 above ~4-8K M" was a workspace-overflow guard in
#  the wrapper (packed-act reservation ∝ M overflowed the 64MB shared buffer), NOT a
#  kernel limit. `cutlass_nvfp4_gemm.cu` now M-tiles into ≤4096-row sub-GEMMs, so any
#  chunk size is safe with FP4 (validated: 16384 chunk, full FP4, needle 3/3, no crash).
#  Larger chunks just grow the prefill arena (≈ MAX_PREFILL-proportional) — a memory,
#  not a correctness, cost. Checkpoint alignment (below) still applies: the chunk must
#  divide SSM_CKPT_INTERVAL*16 for Marconi intermediate prefix-cache checkpoints to fire.
if [ "$PREFIX_CACHING" = "true" ] && [ "${SSM_CKPT_INTERVAL:-0}" -gt 0 ] \
   && [ $(( (SSM_CKPT_INTERVAL * 16) % MAX_PREFILL )) -ne 0 ]; then
    echo "WARN: MAX_PREFILL=$MAX_PREFILL does not divide checkpoint interval $((SSM_CKPT_INTERVAL*16))t; prefix-cache checkpoints may not fire" >&2
fi
# fp8 KV needs calibrated k/v scales or it clips BF16 to E4M3 and destroys dynamic
# range (hallucinations at long context). Default to online calibration + a few
# bf16 high-precision layers when KV dtype is fp8. Override via the env vars.
KV_EXTRA=""
if [ "${ATLAS_HOLO_KV_DTYPE:-bf16}" = "fp8" ]; then
  KV_EXTRA="--fp8-kv-calibration-tokens ${ATLAS_HOLO_FP8_KV_CALIB:-256} --kv-high-precision-layers ${ATLAS_HOLO_KV_HIGH_PREC_LAYERS:-3}"
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
  ATLAS_MOE_PREFILL_FP8_DOWN="$FP8_DOWN" \
  ATLAS_PREFILL_CODISPATCH="$CODISPATCH" ATLAS_FLASHINFER_PREFILL="$FLASHINFER_PREFILL" \
  ATLAS_Q12_BATCHED_FIRST_CHUNK="$Q12_BATCHED_FIRST_CHUNK" ATLAS_GDN_BATCHED_FLA="$GDN_BATCHED_FLA" \
  ATLAS_HOLO_NATIVE_FP8_ATTN="$NATIVE_FP8_ATTN" ATLAS_ATTN_PREFILL_Q_T="$ATTN_Q_T" ATLAS_ATTN_PREFILL_T_PIPE="$ATTN_T_PIPE" \
  ATLAS_HOLO_ALWAYS_MIXED="$ALWAYS_MIXED" \
  ATLAS_CUBLAS_GEMM="$CUBLAS_GEMM" ATLAS_CUTLASS_NVFP4_GEMM="$CUTLASS_NVFP4_GEMM" ATLAS_CUTLASS_NVFP4_SSM_OUT="$CUTLASS_NVFP4_SSM_OUT" \
  "$BIN" serve \
    --model-from-path /tank/holo-bf16kv-test --model-name holo3.1-atlas-poc \
    --port 8890 --bind "$BIND" --max-seq-len "$MAX_SEQ_LEN" --max-num-seqs "$MAX_SEQS" --max-batch-size "$MAX_BATCH" \
    --max-prefill-tokens "$MAX_PREFILL" --kv-cache-dtype "${ATLAS_HOLO_KV_DTYPE:-bf16}" $KV_EXTRA \
    --gpu-memory-utilization "$GPU_UTIL" --oom-guard-mb 256 --ssm-cache-slots "$SSM_CACHE_SLOTS" --ssm-checkpoint-interval "$SSM_CKPT_INTERVAL" \
    --enable-prefix-caching "$PREFIX_CACHING" --scheduling-policy "$SCHED_POLICY" --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --fast-load-prefetch-shards --vision-max-pixels 262144 \
    > "$LOG" 2>&1 </dev/null
echo "launched (log=$LOG)"
