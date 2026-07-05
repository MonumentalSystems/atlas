#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Stage 2 acceptance: the streaming prefill path is BIT-IDENTICAL to resident.
#
# Runs the same A3B checkpoint + fixed prompt under three residency configs and
# compares the final-norm dump (atlas_final_norm.bin, via ATLAS_NEMO_DUMP — the
# lm-head input, which isolates the MoE path from lm-head nondeterminism):
#   C0 resident  — experts held resident (golden)
#   C1 uma       — experts streamed from NVMe into the zero-copy pinned arena
#   C2 posix     — experts streamed via the bounce oracle
# A 2-layer arena cap forces eviction, emulating a ~20x over-core model.
#
# Also runs C0 TWICE first as a determinism precondition: if the grouped GEMM is
# not run-to-run reproducible, an exact cmp is meaningless (investigate before
# trusting the cross-config result).
#
# Usage: verify_logits.sh <checkpoint-dir> <expert-store-dir>
set -euo pipefail

CKPT="${1:?usage: verify_logits.sh <checkpoint-dir> <expert-store-dir>}"
STORE="${2:?usage: verify_logits.sh <checkpoint-dir> <expert-store-dir>}"
BIN="${BIN:-./target/release/spark-server}"
OUT="${OUT:-/tmp/es-gate}"
PORT="${PORT:-8899}"
ARENA_LAYERS="${ARENA_LAYERS:-2}"
# Novel prompt (avoid any prefix-cache hit); identical across all configs.
PROMPT="${PROMPT:-Streaming experts bit-identical probe 4f2a9c: name three prime numbers then stop}"

rm -rf "$OUT"; mkdir -p "$OUT"
DUMP="atlas_final_norm.bin"
PORT_SEQ="$PORT"

run() {
  local tag="$1"; shift
  local dir="$OUT/$tag"; mkdir -p "$dir"
  local port="$PORT_SEQ"; PORT_SEQ=$((PORT_SEQ + 1))  # fresh port per run
  echo ">>> [$tag] serve (port $port) $*"
  # Cap KV cache low (only a short prompt is served) so overlapping teardown of
  # the previous server can't tip startup into OOM. Same for every config, and
  # KV size doesn't affect the MoE math, so the comparison stays valid.
  ATLAS_NEMO_DUMP="$dir" "$BIN" serve --model-from-path "$CKPT" --model-name a3b \
    --port "$port" --lm-head-dtype bf16 --gpu-memory-utilization 0.55 "$@" \
    > "$dir/server.log" 2>&1 &
  local pid=$!
  local ready=0
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[$tag] server exited early:"; tail -30 "$dir/server.log"; exit 1; fi
    sleep 2
  done
  [ "$ready" = 1 ] || { echo "[$tag] server never became ready"; tail -30 "$dir/server.log"; kill "$pid" 2>/dev/null || true; exit 1; }
  curl -s "http://127.0.0.1:$port/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"a3b\",\"prompt\":\"${PROMPT}\",\"temperature\":0,\"max_tokens\":1}" \
    > "$dir/completion.json" 2>&1 || true
  sleep 1  # let the dump flush
  kill "$pid" 2>/dev/null || true
  # wait for the process to actually exit (frees the port cleanly)
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
  sleep 12  # let the killed process drain its GPU memory before the next run
  [ -f "$dir/$DUMP" ] || { echo "[$tag] no $DUMP produced"; tail -30 "$dir/server.log"; exit 1; }
  echo "[$tag] $DUMP = $(stat -c%s "$dir/$DUMP") bytes, sha=$(sha256sum "$dir/$DUMP" | cut -c1-16)"
}

# Determinism precondition.
run resident0
run resident1
echo "=== determinism precheck ==="
if cmp -s "$OUT/resident0/$DUMP" "$OUT/resident1/$DUMP"; then
  echo "OK: resident is run-to-run bit-identical"
else
  echo "WARNING: resident is NOT run-to-run deterministic — exact cmp below may be unreliable."
  echo "         (grouped-GEMM tile scheduling / atomic reductions?) Investigate before trusting."
fi

# Streaming configs.
run uma   --stream-experts "$STORE" --expert-arena-layers "$ARENA_LAYERS" --expert-backend uma
run posix --stream-experts "$STORE" --expert-arena-layers "$ARENA_LAYERS" --expert-backend posix

echo "=== Stage 2 gate: resident0 vs uma vs posix (final-norm) ==="
ok=1
cmp -s "$OUT/resident0/$DUMP" "$OUT/uma/$DUMP"   && echo "  resident == uma   : PASS" || { echo "  resident == uma   : FAIL"; ok=0; }
cmp -s "$OUT/resident0/$DUMP" "$OUT/posix/$DUMP" && echo "  resident == posix : PASS" || { echo "  resident == posix : FAIL"; ok=0; }
cmp -s "$OUT/uma/$DUMP" "$OUT/posix/$DUMP"       && echo "  uma == posix      : PASS" || { echo "  uma == posix      : FAIL"; ok=0; }
if [ "$ok" = 1 ]; then echo "STAGE 2 GATE: PASS (bit-identical prefill across resident/uma/posix)"; else echo "STAGE 2 GATE: FAIL"; exit 1; fi
