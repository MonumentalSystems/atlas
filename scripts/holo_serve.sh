#!/usr/bin/env bash
# Launch the Holo 3.1 Atlas POC server (GB10 low-memory MoE, 64K/C12).
# Detached via setsid per porting-doc note (nohup gets reaped before CUDA init).
set -u
LOG="${1:-/tmp/holo-atlas.log}"
BIN=/home/ms/atlas/target/release/spark
# Self-clean: kill any prior server (setsid -f detaches, so a stale instance
# survives restarts and keeps :8890 bound — the new launch then orphans).
for pid in $(pgrep -f "target/release/spark serve" 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
sleep 2
setsid -f env RUST_BACKTRACE=1 RUST_LOG=info \
  ATLAS_DECODE_GRAPHS_MULTISEQ=1 ATLAS_HOLO_FP8_SSM_DECODE=1 \
  ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
  ATLAS_FAST_LOAD_PREFETCH_SHARDS=1 ATLAS_HOLO_LOW_MEMORY_MOE=1 \
  "$BIN" serve \
    --model-from-path /tank/holo-bf16kv-test --model-name holo3.1-atlas-poc \
    --port 8890 --bind 127.0.0.1 --max-seq-len 32768 --max-num-seqs 8 --max-batch-size 8 \
    --max-prefill-tokens 8192 --kv-cache-dtype bf16 --lm-head-dtype bf16 \
    --gpu-memory-utilization 0.55 --oom-guard-mb 256 --ssm-cache-slots 0 --ssm-checkpoint-interval 0 \
    --enable-prefix-caching false --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --fast-load-prefetch-shards --vision-max-pixels 262144 \
    > "$LOG" 2>&1 </dev/null
echo "launched (log=$LOG)"
