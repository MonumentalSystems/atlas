#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Wire sccache into HOST cargo builds:  source scripts/dev-sccache.sh
#
# Deliberately a sourceable script and NOT `[build] rustc-wrapper` in
# .cargo/config.toml: that key is unconditional, so every machine and CI runner
# without an sccache binary would hard-fail `cargo build`. Opt-in beats a
# repo-wide landmine.
#
# MEASURED on dgx-00 (spark-storage, release profile, cache cleaned between runs):
#     cold  18.37 s
#     warm   1.68 s      (~11x; 40% hit rate — the misses are build scripts)
#
# What sccache will NOT speed up, so nobody over-expects it:
#   * The ~150 nvcc kernel compiles. `atlas-kernels/build.rs` invokes nvcc
#     DIRECTLY, so RUSTC_WRAPPER never sees them. Switching ATLAS_TARGET_MODEL
#     still costs a full kernel rebuild (~27-38 s). A persistent kernel cache is
#     separate, unrelated work.
#   * Build scripts and proc-macros: sccache reports these `crate-type`
#     non-cacheable. Expected, not a misconfiguration.
#   * DEV-profile builds, unless CARGO_INCREMENTAL=0 — an incremental rustc call
#     is non-cacheable by construction. Release implies incremental=0 already.
#     We export it here so `cargo test` / `cargo check` benefit too. The cost is
#     losing incremental recompiles WITHIN one unchanged working tree; the win is
#     hits ACROSS worktrees and branch switches, which is the common case here
#     (this repo is normally worked in .claude/worktrees/*).

command -v sccache >/dev/null 2>&1 || {
    echo "sccache not on PATH — install it (cargo install sccache) or don't source this." >&2
    return 1 2>/dev/null || exit 1
}

export RUSTC_WRAPPER=sccache
export CARGO_INCREMENTAL=0
export SCCACHE_DIR="${SCCACHE_DIR:-$HOME/.cache/sccache}"
export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-20G}"

sccache --start-server >/dev/null 2>&1 || true

echo "sccache wired: RUSTC_WRAPPER=sccache CARGO_INCREMENTAL=0 SCCACHE_DIR=$SCCACHE_DIR"
echo "  stats:  sccache --show-stats     (watch 'Cache hits rate')"
echo "  reset:  sccache --zero-stats"
echo "  note:   nvcc kernel builds are NOT cached — ATLAS_TARGET_MODEL switches still rebuild."
