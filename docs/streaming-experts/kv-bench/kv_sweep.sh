#!/usr/bin/env bash
# KV-tier A/B: normal-HBM vs high-speed-swap{NVMe, RDMA-partial, RDMA-full} across
# concurrency, on holo-3.1-0.8b (small model = worst-case exposure of KV overhead).
set -uo pipefail
cd /home/ms/atlas/.claude/worktrees/streaming-experts-mvp
BIN=./target/release/spark
MODEL=/home/ms/lora-demo/hf-cache/hub/models--Hcompany--Holo-3.1-0.8B/snapshots/72da4c53b351eb60e10a7022279019633c479292
CLIENT=/home/ms/.claude/jobs/42b99a42/tmp/bench_client.py
PORT=8961
CTX=2500; MAXTOK=48
CONCS="1 2 4 8"
RES=/home/ms/.claude/jobs/42b99a42/tmp/kv_results.tsv
export HF_HOME=/home/ms/lora-demo/hf-cache RUST_LOG=warn   # warn: keep the sweep log readable; serve still prints INFO summary lines? no—use info but grep
export RUST_LOG=info
echo -e "config\tconc\tttft_ms\tdecode_toks_mean\tagg_decode\treq_per_s\twall_s\tsample" > "$RES"

run_config() {
  local name="$1"; shift
  local envline="$1"; shift   # extra env (KV peer) or ""
  local log=/home/ms/.claude/jobs/42b99a42/tmp/serve_${name}.log
  rm -rf /tmp/atlas-hss-${name}; mkdir -p /tmp/atlas-hss-${name}
  echo ">>> serving config=$name flags: $*" >&2
  env $envline HF_HOME=/home/ms/lora-demo/hf-cache RUST_LOG=info \
    $BIN serve "$MODEL" --port $PORT --max-seq-len 8192 \
    --gpu-memory-utilization 0.55 --max-batch-size 8 --scheduling-policy slai \
    "$@" >"$log" 2>&1 &
  local srv=$!
  local up=0
  for i in $(seq 1 100); do
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { up=1; break; }
    kill -0 $srv 2>/dev/null || { echo "  $name SERVER DIED"; tail -20 "$log" >&2; break; }
    sleep 1
  done
  if [ $up -eq 1 ]; then
    grep -iE "overflow tier =|orchestrator installed|RdmaKvBackend connected|does not expose" "$log" | head -3 >&2
    # warmup (compile graphs / connect)
    timeout 90 python3 "$CLIENT" "http://127.0.0.1:$PORT/v1/chat/completions" 1 8 $CTX >/dev/null 2>&1
    for c in $CONCS; do
      local out; out=$(timeout 150 python3 "$CLIENT" "http://127.0.0.1:$PORT/v1/chat/completions" $c $MAXTOK $CTX 2>/dev/null)
      echo "  [$name C=$c] $out" >&2
      # parse json -> tsv row
      echo "$out" | python3 -c "import sys,json;
try:
 d=json.load(sys.stdin)
 print('$name\t%s\t%s\t%s\t%s\t%s\t%s\t%s'%(d['conc'],d['ttft_mean_ms'],d['decode_toks_mean'],d['agg_decode_toks'],d['req_per_s'],d['wall_s'],(d['sample'] or '').replace(chr(9),' ').replace(chr(10),' ')[:30]))
except Exception as e: print('$name\terr\t\t\t\t\t\t%s'%e)" >> "$RES"
    done
  else
    echo -e "$name\tSERVE_FAILED\t\t\t\t\t\t" >> "$RES"
  fi
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  pkill -f "release/spark serv[e].*--port $PORT" 2>/dev/null
  sleep 3
}

RDMA_ENV="ATLAS_KV_PEER=192.168.178.12:9920 ATLAS_EXPERT_RDMA_DEV=roceP2p1s0f1 ATLAS_EXPERT_RDMA_GID=3"

# 1. normal HBM (baseline, no swap)
run_config norm ""
# 2. NVMe overflow, cap=8 (SSD tier, full offload) — contextualizes RDMA vs SSD
run_config nvmeB "" --high-speed-swap --high-speed-swap-dir /tmp/atlas-hss-nvmeB --high-speed-swap-gb 8 --high-speed-swap-cache-blocks-per-seq 8
# 3. RDMA overflow, cap=64 (partial / realistic budget, tail spills)
run_config rdmaA "$RDMA_ENV" --high-speed-swap --high-speed-swap-dir /tmp/atlas-hss-rdmaA --high-speed-swap-gb 8 --high-speed-swap-cache-blocks-per-seq 64
# 4. RDMA overflow, cap=8 (forced-full offload, ~entire cache remote)
run_config rdmaB "$RDMA_ENV" --high-speed-swap --high-speed-swap-dir /tmp/atlas-hss-rdmaB --high-speed-swap-gb 8 --high-speed-swap-cache-blocks-per-seq 8

echo "=== RESULTS ==="; cat "$RES"
