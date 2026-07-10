#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# ============================================================================
# Two-machine REAL RDMA integration harness — the paging stack minus the model.
#
#   THIS HOST (client, needs a GPU + rdma-core)  ──RoCEv2──▶  PEER HOST
#                                                             (atlas-cache-peer)
#
# Exercises: connect_paging handshake, the TCP control channel (alloc/commit/get),
# the one-sided RDMA data plane, and the peer's NVMe swap + rehydrate. It PUTs far
# more blobs than the peer's RAM arena holds, so the peer MUST spill the coldest to
# disk; it then GETs them all back and asserts byte-identity — a fault-from-disk on
# every evicted key. Runs in seconds. No inference stack, no model weights.
#
# PORTABILITY: nothing here is specific to one fleet. Hosts, ports, rails and NVMe
# paths come from flags/env or a config file. Per AGENTS.md's PCND invariant there
# are NO implicit host defaults: if the peer is not configured, we fail fast with an
# actionable message rather than dialling someone else's machine.
#
# QUICK START
#   cp scripts/rdma-harness.env.example ~/.config/atlas/rdma-harness.env   # then edit
#   scripts/rdma-integration-harness.sh
# or entirely from flags:
#   scripts/rdma-integration-harness.sh --peer-ssh box2 --peer-data 10.0.0.2 --swap-dir /nvme/atlas
#
# PROFILES
#   tiny   (default)  256 KiB arena, ~2 MiB moved      — seconds, CI-friendly
#   stress            32 MiB arena, ~1 GiB through swap — still well under 10 GB RAM
#
# SAFETY
#   * Refuses to bind a port that is already listening on the peer.
#   * Hard RAM/disk ceilings on the peer (--max-blade-gb / --swap-cap-gb).
#   * Teardown kills ONLY the PID we started and removes ONLY the swap dir we made.
#     Never a broad pkill: a broad pkill once killed a colleague's server.
# ============================================================================
set -uo pipefail

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
say() { printf '  %s\n' "$*"; }
hdr() { printf '\n=== %s ===\n' "$*"; }

# ── config: file < env < flags ───────────────────────────────────────────────
CONF="${ATLAS_RDMA_HARNESS_CONF:-${XDG_CONFIG_HOME:-$HOME/.config}/atlas/rdma-harness.env}"
# shellcheck disable=SC1090
[ -f "$CONF" ] && . "$CONF"

PEER_SSH="${PEER_SSH:-}"        # ssh target for the peer's MANAGEMENT interface
PEER_DATA="${PEER_DATA:-}"      # host/IP the client dials over the RDMA data link
PEER_PORT="${PEER_PORT:-9930}"  # dedicated test port; must be free on the peer
PEER_SWAP_DIR="${PEER_SWAP_DIR:-}"   # NVMe dir on the peer for the swap file
PEER_RAILS="${PEER_RAILS:-}"    # e.g. "roceP2p1s0f1:3 rocep1s0f1:3"; auto-detected if empty
CLIENT_RAIL="${CLIENT_RAIL:-}"  # e.g. "rocep1s0f1:3"; auto-detected if empty
MAX_BLADE_GB="${MAX_BLADE_GB:-4}"
SWAP_CAP_GB="${SWAP_CAP_GB:-2}"
PROFILE="${PROFILE:-tiny}"
REPO="${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
KEEP_PEER="${KEEP_PEER:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --peer-ssh)   PEER_SSH="$2"; shift 2 ;;
    --peer-data)  PEER_DATA="$2"; shift 2 ;;
    --port)       PEER_PORT="$2"; shift 2 ;;
    --swap-dir)   PEER_SWAP_DIR="$2"; shift 2 ;;
    --peer-rails) PEER_RAILS="$2"; shift 2 ;;
    --client-rail) CLIENT_RAIL="$2"; shift 2 ;;
    --profile)    PROFILE="$2"; shift 2 ;;
    --max-blade-gb) MAX_BLADE_GB="$2"; shift 2 ;;
    --swap-cap-gb)  SWAP_CAP_GB="$2"; shift 2 ;;
    --keep-peer)  KEEP_PEER=1; shift ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    *) die "unknown flag: $1 (see --help)" ;;
  esac
done

# PCND: explicit config or fail fast. No fleet-specific fallbacks.
[ -n "$PEER_SSH" ]      || die "PEER_SSH is unset. Pass --peer-ssh <host> or set it in $CONF
       (this is the peer's MANAGEMENT ssh target, not its RDMA data-path address)."
[ -n "$PEER_DATA" ]     || die "PEER_DATA is unset. Pass --peer-data <host-or-ip> or set it in $CONF
       (the address the client dials over the RDMA link — often DIFFERENT from PEER_SSH)."
[ -n "$PEER_SWAP_DIR" ] || die "PEER_SWAP_DIR is unset. Pass --swap-dir </nvme/path> or set it in $CONF
       (an NVMe-backed directory on the peer; the swap file lands here)."

case "$PROFILE" in
  tiny)   BLOB=65536;   SLOTS=4;  KEYS=32  ;;
  stress) BLOB=4194304; SLOTS=8;  KEYS=256 ;;
  *) die "unknown --profile '$PROFILE' (tiny|stress)" ;;
esac
ARENA=$(( SLOTS * BLOB ))
RUN_ID="atlas-smoke-$$-$(date +%s)"
SWAP_DIR="${PEER_SWAP_DIR%/}/${RUN_ID}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"

PEER_PID=""
cleanup() {
  [ "$KEEP_PEER" = "1" ] && { hdr "teardown skipped (--keep-peer)"; say "peer pid=$PEER_PID swap=$SWAP_DIR"; return; }
  hdr "teardown (scoped to THIS run only)"
  if [ -n "$PEER_PID" ]; then
    say "kill our peer pid $PEER_PID (never a broad pkill)"
    $SSH "$PEER_SSH" "kill $PEER_PID 2>/dev/null; sleep 1; kill -9 $PEER_PID 2>/dev/null; rm -f /tmp/atlas-cache-peer-${RUN_ID}" || true
  fi
  $SSH "$PEER_SSH" "rm -rf '$SWAP_DIR'" 2>/dev/null && say "removed $SWAP_DIR"
}
trap cleanup EXIT

# ── 0. preflight ─────────────────────────────────────────────────────────────
hdr "preflight"
$SSH "$PEER_SSH" 'echo reached $(hostname)' | sed 's/^/  peer: /' || die "cannot ssh to PEER_SSH=$PEER_SSH"
$SSH "$PEER_SSH" "ss -ltn 2>/dev/null | grep -q ':${PEER_PORT}[[:space:]]'" \
  && die "port :$PEER_PORT is already listening on $PEER_SSH — pick another --port"
say "peer port :$PEER_PORT free"

# Auto-detect rails from the RDMA devices present, unless told otherwise.
# Only ACTIVE ports: the alphabetically-first IB device is frequently DOWN (e.g.
# rocep1s0f0 while roceP2p1s0f1 carries the link), so a naive `ls | head -1`
# picks a dead rail and the handshake hangs or fails obscurely.
active_rails() {  # $1 = "" for local, else an ssh target
  local runner=(bash -c); [ -n "${1:-}" ] && runner=($SSH "$1")
  "${runner[@]}" '
    for d in /sys/class/infiniband/*; do
      [ -e "$d/ports/1/state" ] || continue
      case "$(cat "$d/ports/1/state")" in *ACTIVE*) basename "$d";; esac
    done' 2>/dev/null
}
if [ -z "$PEER_RAILS" ]; then
  PEER_RAILS=$(active_rails "$PEER_SSH" | awk '{printf "%s:3 ", $1}')
  [ -n "$PEER_RAILS" ] || die "no ACTIVE RDMA port on $PEER_SSH (check /sys/class/infiniband/*/ports/1/state); pass --peer-rails"
  say "auto-detected peer rails (ACTIVE only): $PEER_RAILS"
fi
if [ -z "$CLIENT_RAIL" ]; then
  CLIENT_RAIL="$(active_rails "" | head -1):3"
  [ "$CLIENT_RAIL" != ":3" ] || die "no ACTIVE local RDMA port; pass --client-rail <dev>:<gid_idx>"
  say "auto-detected client rail (ACTIVE only): $CLIENT_RAIL"
fi
say "caps: ${MAX_BLADE_GB} GB peer RAM / ${SWAP_CAP_GB} GB peer disk"
say "profile=$PROFILE: blob=${BLOB} B, slots=$SLOTS (arena=$((ARENA/1024)) KiB), keys=$KEYS"

# ── 1. build peer + client (client needs --features cuda for the pinned bounce) ──
hdr "build"
cd "$REPO" || die "REPO=$REPO is not a directory"
cargo build --release -p atlas-expert-pack --bin atlas-cache-peer 2>&1 | tail -1 | sed 's/^/  peer:   /'
cargo build --release -p spark-storage --features cuda --example snapshot_paging_smoke 2>&1 | tail -1 | sed 's/^/  client: /'
PEER_BIN="$REPO/target/release/atlas-cache-peer"
CLIENT_BIN="$REPO/target/release/examples/snapshot_paging_smoke"
[ -x "$PEER_BIN" ]   || die "peer binary not built: $PEER_BIN"
[ -x "$CLIENT_BIN" ] || die "client example not built: $CLIENT_BIN (needs --features cuda)"

# Same-arch assumption: ship the binary rather than build on the peer.
LOCAL_ARCH=$(uname -m); PEER_ARCH=$($SSH "$PEER_SSH" 'uname -m')
[ "$LOCAL_ARCH" = "$PEER_ARCH" ] || die "arch mismatch: local=$LOCAL_ARCH peer=$PEER_ARCH — build the peer on the peer"

# ── 2. start the capped test peer ────────────────────────────────────────────
hdr "start capped test peer on ${PEER_SSH}:${PEER_PORT}"
scp -q "$PEER_BIN" "$PEER_SSH:/tmp/atlas-cache-peer-${RUN_ID}" || die "scp peer binary failed"
$SSH "$PEER_SSH" "mkdir -p '$SWAP_DIR'" || die "cannot create $SWAP_DIR"
RAIL_ARGS=""; for r in $PEER_RAILS; do RAIL_ARGS="$RAIL_ARGS --rail $r"; done
PEER_PID=$($SSH "$PEER_SSH" "
  chmod +x /tmp/atlas-cache-peer-${RUN_ID}
  nohup /tmp/atlas-cache-peer-${RUN_ID} \
    --listen 0.0.0.0:${PEER_PORT} \
    --swap-dir '${SWAP_DIR}' \
    --swap-cap-gb ${SWAP_CAP_GB} \
    --max-blade-gb ${MAX_BLADE_GB} \
    ${RAIL_ARGS} \
    > /tmp/atlas-peer-${RUN_ID}.log 2>&1 &
  echo \$!")
[ -n "$PEER_PID" ] || die "peer did not start"
say "peer pid=$PEER_PID  swap=$SWAP_DIR  log=/tmp/atlas-peer-${RUN_ID}.log"

for _ in $(seq 1 30); do
  $SSH "$PEER_SSH" "ss -ltn 2>/dev/null | grep -q ':${PEER_PORT}[[:space:]]'" && break
  sleep 1
done
$SSH "$PEER_SSH" "ss -ltn 2>/dev/null | grep ':${PEER_PORT}[[:space:]]'" | sed 's/^/  listening: /' \
  || { $SSH "$PEER_SSH" "tail -20 /tmp/atlas-peer-${RUN_ID}.log" | sed 's/^/  peer: /'; die "peer never listened"; }

# ── 3. drive the real RDMA round-trip ────────────────────────────────────────
hdr "client: paging PUT/GET over real RDMA (forces peer NVMe spill + rehydrate)"
set +e
ATLAS_SNAP_PEER="${PEER_DATA}:${PEER_PORT}" \
ATLAS_EXPERT_RDMA_DEV="${CLIENT_RAIL%%:*}" ATLAS_EXPERT_RDMA_GID="${CLIENT_RAIL##*:}" \
SMOKE_BLOB="$BLOB" SMOKE_SLOTS="$SLOTS" SMOKE_KEYS="$KEYS" SMOKE_MODE=putget \
  "$CLIENT_BIN" 2>&1 | sed 's/^/  /'
RC=${PIPESTATUS[0]}
set -e

# ── 4. evidence the spill actually happened (a green PUT/GET alone proves little) ──
hdr "peer-side evidence"
$SSH "$PEER_SSH" "grep -icE 'spill|evict|swap' /tmp/atlas-peer-${RUN_ID}.log" | sed 's/^/  spill-or-evict log lines: /'
$SSH "$PEER_SSH" "du -sh '$SWAP_DIR' 2>/dev/null | cut -f1" | sed 's/^/  swap dir size: /'

hdr "RESULT"
if [ "$RC" -eq 0 ]; then
  echo "  PASS — two-machine RDMA paging round-trip, byte-identical after a peer-side NVMe spill"
else
  echo "  FAIL — client exited $RC"
  $SSH "$PEER_SSH" "tail -30 /tmp/atlas-peer-${RUN_ID}.log" | sed 's/^/    peer: /'
fi
exit "$RC"
