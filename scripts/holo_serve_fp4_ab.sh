#!/usr/bin/env bash
# Launch the Holo server with the baseline production config, optionally enabling
# ATLAS_HOLO_MOE_GATEUP_FP4. KV sized for the 60K A/B workload (70K ctx / 4 seqs)
# so a single 60K prefill fits the pool on this shared box. Applied IDENTICALLY to
# both FP4-on and FP4-off; production 200K config is restored at the end.
# Usage: launch_ab.sh <fp4: 0|1> <logfile>
set -u
FP4="${1:-0}"
LOG="${2:-/tmp/holo_ab.log}"
export ATLAS_HOLO_BIND=0.0.0.0
export ATLAS_HOLO_GPU_UTIL=0.66
export ATLAS_HOLO_MAX_SEQ_LEN=70000
export ATLAS_HOLO_MAX_SEQS=4
export ATLAS_HOLO_MAX_BATCH=4
export ATLAS_HOLO_MAX_PREFILL=2048
export ATLAS_HOLO_FAST_MOE_MODE=full
export ATLAS_HOLO_FAST_MOE_LAYERS=0-39
export ATLAS_HOLO_PREFIX_CACHING=true
export ATLAS_HOLO_SSM_CACHE_SLOTS=32
export ATLAS_HOLO_SSM_CHECKPOINT_INTERVAL=256
export ATLAS_HOLO_SCHED_POLICY=slai
export ATLAS_KV_OVERCOMMIT=1
export ATLAS_KV_EXTERNAL_RESERVE_GB=26
export ATLAS_HOLO_MOE_GATEUP_FP4="$FP4"
cd ~/atlas
bash scripts/holo_serve.sh "$LOG"
