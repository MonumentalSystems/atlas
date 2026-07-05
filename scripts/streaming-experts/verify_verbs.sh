#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# WS2 Phase B acceptance: the one-sided RDMA READ (verbs) peer tier is
# BIT-IDENTICAL to resident, and (as a bonus cross-check) to the two-sided TCP
# peer tier. Same methodology as verify_logits.sh: compare the final-norm dump
# (atlas_final_norm.bin via ATLAS_NEMO_DUMP — the lm-head input, isolating the
# MoE path) across residency configs served from the SAME checkpoint + prompt.
#
#   C0 resident    — experts held resident (golden)
#   C1 rdma-verbs  — experts pulled from the peer via IBV_WR_RDMA_READ
#   C2 rdma (TCP)  — experts streamed from the peer over the socket (optional)
#
# A small arena cap (2 layers) forces eviction so every layer is re-fetched over
# the fabric — the real streaming exercise.
#
# Requires an `atlas-expert-peer` reachable at $ATLAS_EXPERT_PEER serving the
# SAME store. If PEER_HOST is set, the script starts/stops the peer there over
# ssh; otherwise it assumes one is already running.
#
# Usage: ATLAS_EXPERT_PEER=192.168.178.12:9909 \
#        [PEER_HOST=192.168.178.12] [PEER_STORE=/home/ms/expert-store-a3b] \
#        verify_verbs.sh <checkpoint-dir> <local-store-dir>
set -euo pipefail

CKPT="${1:?usage: verify_verbs.sh <checkpoint-dir> <local-store-dir>}"
STORE="${2:?usage: verify_verbs.sh <checkpoint-dir> <local-store-dir>}"
BIN="${BIN:-./target/release/spark}"
OUT="${OUT:-/tmp/es-verbs-gate}"
PORT="${PORT:-8919}"
ARENA_LAYERS="${ARENA_LAYERS:-2}"
PEER="${ATLAS_EXPERT_PEER:?set ATLAS_EXPERT_PEER=host:port (the atlas-expert-peer)}"
RUN_TCP="${RUN_TCP:-1}"           # also cross-check the two-sided TCP tier
RDMA_DEV="${ATLAS_EXPERT_RDMA_DEV:-roceP2p1s0f1}"
RDMA_GID="${ATLAS_EXPERT_RDMA_GID:-3}"
PROMPT="${PROMPT:-Streaming experts verbs bit-identical probe 7d1e: name three prime numbers then stop}"

rm -rf "$OUT"; mkdir -p "$OUT"
DUMP="atlas_final_norm.bin"
PORT_SEQ="$PORT"

PEER_PID=""
start_peer() {
  [ -n "${PEER_HOST:-}" ] || { echo "(assuming an external peer at $PEER)"; return; }
  local store="${PEER_STORE:-$STORE}"
  echo ">>> starting atlas-expert-peer on $PEER_HOST ($store, dev $RDMA_DEV gid $RDMA_GID)"
  ssh -o StrictHostKeyChecking=no "$PEER_HOST" \
    "pkill -9 -f atlas-expert-peer 2>/dev/null; sleep 1; \
     RUST_LOG=info nohup /home/ms/atlas-expert-peer --store '$store' \
       --listen 0.0.0.0:${PEER##*:} --rdma-dev $RDMA_DEV --rdma-gid $RDMA_GID \
       > /home/ms/expert-peer.log 2>&1 &" &
  PEER_PID=$!
  sleep 3
}
stop_peer() {
  [ -n "${PEER_HOST:-}" ] || return
  ssh -o StrictHostKeyChecking=no "$PEER_HOST" 'pkill -9 -f atlas-expert-peer 2>/dev/null' || true
}
trap stop_peer EXIT

run() {
  local tag="$1"; shift
  local dir="$OUT/$tag"; mkdir -p "$dir"
  local port="$PORT_SEQ"; PORT_SEQ=$((PORT_SEQ + 1))
  echo ">>> [$tag] serve (port $port) $*"
  ATLAS_NEMO_DUMP="$dir" ATLAS_EXPERT_PEER="$PEER" \
  ATLAS_EXPERT_RDMA_DEV="$RDMA_DEV" ATLAS_EXPERT_RDMA_GID="$RDMA_GID" \
    "$BIN" serve --model-from-path "$CKPT" --model-name a3b \
    --port "$port" --lm-head-dtype bf16 --gpu-memory-utilization 0.55 "$@" \
    > "$dir/server.log" 2>&1 &
  local pid=$!
  local ready=0
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[$tag] server exited early:"; tail -40 "$dir/server.log"; exit 1; fi
    sleep 2
  done
  [ "$ready" = 1 ] || { echo "[$tag] server never became ready"; tail -40 "$dir/server.log"; kill "$pid" 2>/dev/null || true; exit 1; }
  curl -s "http://127.0.0.1:$port/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"a3b\",\"prompt\":\"${PROMPT}\",\"temperature\":0,\"max_tokens\":1}" \
    > "$dir/completion.json" 2>&1 || true
  sleep 1
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
  sleep 12
  [ -f "$dir/$DUMP" ] || { echo "[$tag] no $DUMP produced"; tail -40 "$dir/server.log"; exit 1; }
  echo "[$tag] $DUMP = $(stat -c%s "$dir/$DUMP") bytes, sha=$(sha256sum "$dir/$DUMP" | cut -c1-16)"
}

start_peer

run resident
run verbs --stream-experts "$STORE" --expert-arena-layers "$ARENA_LAYERS" --expert-backend rdma-verbs
if [ "$RUN_TCP" = 1 ]; then
  run tcp --stream-experts "$STORE" --expert-arena-layers "$ARENA_LAYERS" --expert-backend rdma
fi

echo "=== WS2 Phase B gate: resident vs rdma-verbs (final-norm) ==="
ok=1
cmp -s "$OUT/resident/$DUMP" "$OUT/verbs/$DUMP" && echo "  resident == rdma-verbs : PASS" || { echo "  resident == rdma-verbs : FAIL"; ok=0; }
if [ "$RUN_TCP" = 1 ]; then
  cmp -s "$OUT/resident/$DUMP" "$OUT/tcp/$DUMP" && echo "  resident == rdma-tcp   : PASS" || { echo "  resident == rdma-tcp   : FAIL"; ok=0; }
  cmp -s "$OUT/verbs/$DUMP" "$OUT/tcp/$DUMP"    && echo "  rdma-verbs == rdma-tcp : PASS" || { echo "  rdma-verbs == rdma-tcp : FAIL"; ok=0; }
fi
if [ "$ok" = 1 ]; then echo "WS2 PHASE B GATE: PASS (one-sided RDMA READ is bit-identical)"; else echo "WS2 PHASE B GATE: FAIL"; exit 1; fi
