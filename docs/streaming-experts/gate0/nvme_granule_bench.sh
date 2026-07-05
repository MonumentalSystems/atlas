#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Gate 0(b) NVMe reproducer: sustained O_DIRECT sequential read bandwidth at the
# expert-record granule the streamer actually fetches (~1.7 MiB for a3b), across
# the queue depths the io_uring backend uses. Cache-bypassed (--direct=1), so the
# numbers are cold-read, not page-cache.
#
# Usage: nvme_granule_bench.sh [dir] [granule_kib]
#   dir         directory on the NVMe under test (default: cwd)
#   granule_kib per-read size in KiB (default 1712 ~= a3b per-expert 1.6875 MiB)
set -euo pipefail
DIR="${1:-.}"
BS="${2:-1712}k"
FILE="$DIR/.gate0_nvme_test.bin"
SIZE="8G"

command -v fio >/dev/null || { echo "fio not installed" >&2; exit 1; }

cleanup() { rm -f "$FILE"; }
trap cleanup EXIT

echo "prep: writing $SIZE test file (O_DIRECT)…"
fio --name=prep --rw=write --bs=1m --size="$SIZE" --direct=1 --ioengine=io_uring \
    --filename="$FILE" --output-format=terse >/dev/null

echo "sustained O_DIRECT sequential READ @ $BS granule (cache-bypassed):"
for qd in 1 4 8 16; do
  mibps=$(fio --name=r --rw=read --bs="$BS" --size="$SIZE" --direct=1 --ioengine=io_uring \
      --iodepth="$qd" --filename="$FILE" --invalidate=1 --minimal 2>/dev/null \
      | awk -F';' '{printf "%.0f", $7/1024}')
  gbps=$(awk "BEGIN{printf \"%.2f\", $mibps/953.674}")   # MiB/s -> GB/s (decimal)
  printf "  iodepth=%-2s  %6s MiB/s  (~%s GB/s)\n" "$qd" "$mibps" "$gbps"
done
