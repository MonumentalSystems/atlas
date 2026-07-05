#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Measure one-sided RDMA READ bandwidth over CX7 (RoCEv2) between two hosts,
# using perftest's ib_read_bw. This is the ceiling for the verbs expert tier.
#   Usage: verbs_rdma_read_bw.sh <peer-cx7-ip> [dev] [gid_index] [msg_bytes]
# GID index = the RoCEv2 IPv4 GID (`show_gids | grep v2` — index 3 on GB10/CX7).
set -euo pipefail
PEER="${1:?peer CX7 ip, e.g. 192.168.178.12}"
DEV="${2:-roceP2p1s0f1}"; GID="${3:-3}"; SZ="${4:-1773568}"  # default = a3b record_stride
ssh -o StrictHostKeyChecking=no "$PEER" "pkill -9 -f ib_read_bw 2>/dev/null; sleep 1" || true
ssh -o StrictHostKeyChecking=no "$PEER" "ib_read_bw -d $DEV -x $GID -F --report_gbits -s $SZ -q 4" >/tmp/ibr_srv.log 2>&1 &
SSHPID=$!; sleep 4
ib_read_bw -d "$DEV" -x "$GID" -F --report_gbits -s "$SZ" -q 4 "$PEER" | grep -E "#bytes|$SZ"
kill $SSHPID 2>/dev/null || true
