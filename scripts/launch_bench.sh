#!/bin/bash
# Launch spark for the burst-TBT A/B. $1 = "off" | "on" (ATLAS_HOLO_ALWAYS_MIXED).
# Config held constant: MAX_PREFILL=2048, prefix OFF, util 0.40, FAST_MOE off,
# max_seqs 16, max_batch 12.
set -u
MODE="${1:-off}"
cd ~/atlas
for pid in $(pgrep -f "release/spark serve --model" 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
sleep 2
export ATLAS_HOLO_GPU_UTIL=0.40
export ATLAS_HOLO_MAX_SEQ_LEN=200000
export ATLAS_HOLO_MAX_SEQS=16
export ATLAS_HOLO_MAX_BATCH=12
export ATLAS_HOLO_FAST_MOE_MODE=off
export ATLAS_KV_OVERCOMMIT=1
export ATLAS_HOLO_PREFIX_CACHING=false
export ATLAS_HOLO_MAX_PREFILL=2048
unset ATLAS_KV_EXTERNAL_RESERVE_GB
if [ "$MODE" = "on" ]; then export ATLAS_HOLO_ALWAYS_MIXED=1; else unset ATLAS_HOLO_ALWAYS_MIXED; fi
date +%s > /tmp/holo_start_epoch
setsid bash scripts/holo_serve.sh /tmp/holo.log </dev/null >/tmp/holo_launch.log 2>&1 &
echo "launched mode=$MODE"
