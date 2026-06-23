#!/usr/bin/env bash
# Launch Holo 3.1 Atlas POC with DFlash drafting enabled (acceptance test).
# Conservative base config + --dflash. γ=4 for the current Atlas K=2
# acceptance path; raise only after K=γ SSM verify is stable. Keep the
# high-prefill-memory MoE copies opt-in here because the drafter adds memory pressure.
set -u
LOG="${1:-/tmp/holo-atlas-dflash.log}"
BIN=/home/ms/atlas/target/release/spark
GPU_UTIL="${ATLAS_HOLO_GPU_UTIL:-0.55}"
MAX_SEQ_LEN="${ATLAS_HOLO_MAX_SEQ_LEN:-32768}"
MAX_SEQS="${ATLAS_HOLO_MAX_SEQS:-8}"
MAX_BATCH="${ATLAS_HOLO_MAX_BATCH:-8}"
MAX_PREFILL="${ATLAS_HOLO_MAX_PREFILL:-8192}"
DFLASH_GAMMA="${ATLAS_HOLO_DFLASH_GAMMA:-4}"
LOW_MEMORY_MOE="${ATLAS_HOLO_LOW_MEMORY_MOE:-1}"
NATIVE_FP8_ATTN="${ATLAS_HOLO_NATIVE_FP8_ATTN:-1}"
ATTN_Q_T="${ATLAS_ATTN_PREFILL_Q_T:-1}"
ATTN_T_PIPE="${ATLAS_ATTN_PREFILL_T_PIPE:-1}"
EXACT_MOE_TILES="${ATLAS_MOE_PREFILL_EXACT_TILES:-1}"
# Qwen3.5 drafter matches Holo 3.1's Qwen3_5Moe base (recipe + user's vLLM
# config both specify z-lab/Qwen3.5-35B-A3B-DFlash, n=4). The 3.6 drafter has
# identical config but weights trained on a different target → weak acceptance.
DRAFTER=/tank/hf/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash/snapshots/a6ab3a277f856d91c43f28711611e7929073d56d
for pid in $(pgrep -f "release/spark serve --model" 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
sleep 2
setsid -f env RUST_BACKTRACE=1 RUST_LOG=info \
  ATLAS_DECODE_GRAPHS_MULTISEQ=1 ATLAS_HOLO_FP8_SSM_DECODE=1 \
  ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
  ATLAS_FAST_LOAD_PREFETCH_SHARDS=1 ATLAS_HOLO_LOW_MEMORY_MOE="$LOW_MEMORY_MOE" \
  ATLAS_MOE_PREFILL_EXACT_TILES="$EXACT_MOE_TILES" \
  ATLAS_HOLO_NATIVE_FP8_ATTN="$NATIVE_FP8_ATTN" ATLAS_ATTN_PREFILL_Q_T="$ATTN_Q_T" ATLAS_ATTN_PREFILL_T_PIPE="$ATTN_T_PIPE" \
  ${ATLAS_DFLASH_NO_MSCALE:+ATLAS_DFLASH_NO_MSCALE=$ATLAS_DFLASH_NO_MSCALE} ${ATLAS_DFLASH_DEBUG_CTX_OFF:+ATLAS_DFLASH_DEBUG_CTX_OFF=$ATLAS_DFLASH_DEBUG_CTX_OFF} \
  "$BIN" serve \
    --model-from-path /tank/holo-bf16kv-test --model-name holo3.1-atlas-poc \
    --port 8890 --bind 127.0.0.1 --max-seq-len "$MAX_SEQ_LEN" --max-num-seqs "$MAX_SEQS" --max-batch-size "$MAX_BATCH" \
    --max-prefill-tokens "$MAX_PREFILL" --kv-cache-dtype bf16 --lm-head-dtype bf16 \
    --gpu-memory-utilization "$GPU_UTIL" --oom-guard-mb 256 --ssm-cache-slots 0 --ssm-checkpoint-interval 0 \
    --enable-prefix-caching false --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --fast-load-prefetch-shards --vision-max-pixels 262144 \
    --dflash --draft-model "$DRAFTER" --dflash-gamma "$DFLASH_GAMMA" \
    > "$LOG" 2>&1 </dev/null
echo "launched dflash (log=$LOG)"
