#!/usr/bin/env bash
# 35B-A3B KV-tier A/B. Authoritative timing from the SERVER LOG line
# "Done: N tokens (stop) X tok/s, TTFT=Yms" (the 35B is a THINKING model; client
# content-delta counting misses reasoning tokens). Correctness from the response
# body (must contain TANGERINE-7742). Long context forces KV overflow.
set -uo pipefail
cd /home/ms/atlas/.claude/worktrees/streaming-experts-mvp
BIN=./target/release/spark
MODEL=$(ls -d /tank/hf/hub/models--Hcompany--Holo-3.1-35B-A3B-NVFP4/snapshots/*/ | head -1)
PORT=8977; CTX=2500; MAXTOK=64
RES=/home/ms/.claude/jobs/42b99a42/tmp/kv_results_35b.tsv
TMP=/home/ms/.claude/jobs/42b99a42/tmp
export HF_HOME=/tank/hf
SECRET="The vault access code is TANGERINE-7742."
UNIT="Atlas is a pure-Rust CUDA inference engine streaming experts and KV over RDMA on GB10 across ConnectX-7. "
CTXBODY=$(yes "$UNIT" | head -n $((CTX/28)) | tr -d '\n')
PROMPT="$SECRET $CTXBODY Question: what is the vault access code?"
PJSON=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT")
echo -e "config\tconc\tttft_ms_mean\tdecode_toks_mean\ttokens_mean\tcorrect\twall_s" > "$RES"

fire() {  # $1=conc -> echoes "<correct>/<conc> <wall_s>"
  local c="$1"; rm -f $TMP/body_* 2>/dev/null
  local t0=$(date +%s.%N)
  for i in $(seq 1 $c); do
    curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":$PJSON}],\"temperature\":0,\"max_tokens\":$MAXTOK}" \
      >"$TMP/body_$i" 2>/dev/null &
  done
  wait
  local t1=$(date +%s.%N)
  local correct=0
  for i in $(seq 1 $c); do grep -q "TANGERINE-7742" "$TMP/body_$i" 2>/dev/null && correct=$((correct+1)); done
  echo "$correct/$c $(echo "$t1 - $t0" | bc)"
}

run_config() {
  local name="$1" envline="$2" concs="$3"; shift 3
  local log=$TMP/serve35_${name}.log
  rm -rf /tmp/atlas-hss-${name}; mkdir -p /tmp/atlas-hss-${name}
  echo ">>> 35B config=$name concs=[$concs] flags: $*" >&2
  env $envline HF_HOME=/tank/hf RUST_LOG=info \
    $BIN serve "$MODEL" --port $PORT --max-seq-len 4096 \
    --gpu-memory-utilization 0.85 --max-batch-size 8 --scheduling-policy slai \
    "$@" >"$log" 2>&1 &
  local srv=$!; local up=0
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { up=1; break; }
    kill -0 $srv 2>/dev/null || { echo "  $name DIED"; tail -20 "$log" >&2; break; }
    sleep 2
  done
  if [ $up -eq 1 ]; then
    grep -iE "overflow tier =|RdmaKvBackend connected|clamping max_prefill" "$log" | head -2 >&2
    fire 1 >/dev/null 2>&1  # warmup
    for c in $concs; do
      local mark=$(wc -l < "$log")
      local res; res=$(fire "$c")
      local correct=${res%% *}; local wall=${res##* }
      # parse the last c "Done:" lines added since mark
      local stats; stats=$(tail -n +$((mark+1)) "$log" | grep -oE "Done: [0-9]+ tokens \(stop\) [0-9.]+ tok/s, TTFT=[0-9.]+ms" | tail -n "$c" | \
        sed -E 's/Done: ([0-9]+) tokens \(stop\) ([0-9.]+) tok\/s, TTFT=([0-9.]+)ms/\1 \2 \3/' | \
        awk '{tok+=$1; ts+=$2; ttft+=$3; n++} END{if(n>0) printf "%.0f %.1f %.1f", ttft/n, ts/n, tok/n; else printf "NA NA NA"}')
      local ttft_m=${stats% * *}; ttft_m=${stats%% *}
      local ts_m=$(echo "$stats" | awk '{print $2}'); local tok_m=$(echo "$stats" | awk '{print $3}')
      echo "  [$name C=$c] TTFT=${ttft_m}ms decode=${ts_m}tok/s tokens=${tok_m} correct=$correct wall=${wall}s" >&2
      echo -e "$name\t$c\t$ttft_m\t$ts_m\t$tok_m\t$correct\t$wall" >> "$RES"
    done
  else echo -e "$name\tSERVE_FAILED" >> "$RES"; fi
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  pkill -f "release/spark serv[e].*--port $PORT" 2>/dev/null; sleep 4
}

RDMA_ENV="ATLAS_KV_PEER=192.168.178.12:9920 ATLAS_EXPERT_RDMA_DEV=roceP2p1s0f1 ATLAS_EXPERT_RDMA_GID=3"
run_config norm  ""          "1 2 4"
run_config nvmeB ""          "1"     --high-speed-swap --high-speed-swap-dir /tmp/atlas-hss-nvmeB --high-speed-swap-gb 16 --high-speed-swap-cache-blocks-per-seq 8
run_config rdmaA "$RDMA_ENV" "1"     --high-speed-swap --high-speed-swap-dir /tmp/atlas-hss-rdmaA --high-speed-swap-gb 16 --high-speed-swap-cache-blocks-per-seq 64
run_config rdmaB "$RDMA_ENV" "1"     --high-speed-swap --high-speed-swap-dir /tmp/atlas-hss-rdmaB --high-speed-swap-gb 16 --high-speed-swap-cache-blocks-per-seq 8
echo "=== 35B RESULTS ==="; column -t -s $'\t' "$RES"
