// SPDX-License-Identifier: AGPL-3.0-only
//
// atlas-expert-peer: serve a resident expert store to streaming clients over the
// RoCE fabric (Stage 4, Phase A: TCP). A dumb weight cache — holds the store and
// answers (layer, expert) record requests. Distinct from EP sharding: the client
// does all the compute; this peer only moves bytes.
//
//   atlas-expert-peer --store <dir> [--listen 0.0.0.0:9909]
//
// The client connects with `--expert-backend rdma` and $ATLAS_EXPERT_PEER set to
// this peer's host:port.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let mut store: Option<PathBuf> = None;
    let mut listen = String::from("0.0.0.0:9909");
    let mut rdma = spark_storage::expert_peer::RdmaConfig::default();
    // Repeatable `--rail <dev>:<gid>`; if any are given they REPLACE the
    // single-rail default. `--rdma-dev`/`--rdma-gid` still mutate rail 0 in place
    // so the pre-dual-rail single-adapter invocation is unchanged.
    let mut custom_rails: Vec<(String, u32)> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--store" | "-s" => store = Some(it.next().context("--store needs a dir")?.into()),
            "--listen" | "-l" => listen = it.next().context("--listen needs host:port")?,
            "--rdma-dev" => {
                rdma.rails[0].0 = it.next().context("--rdma-dev needs a device name")?
            }
            "--rdma-gid" => {
                rdma.rails[0].1 = it
                    .next()
                    .context("--rdma-gid needs an index")?
                    .parse()
                    .context("--rdma-gid must be an integer")?
            }
            // Repeatable rail selector for dual-rail serving. First --rail
            // replaces the single-rail default; give it twice for both CX7
            // adapters (a client then enables striping with ATLAS_EXPERT_DUAL_RAIL=1).
            "--rail" => {
                let spec = it.next().context("--rail needs <dev>:<gid_idx>")?;
                let (dev, gid) = spec
                    .rsplit_once(':')
                    .context("--rail format is <dev>:<gid_idx>")?;
                custom_rails.push((dev.to_string(), gid.parse().context("gid must be int")?));
            }
            // Ceiling on total registered store RAM (GiB) across concurrent verbs
            // connections. 0 or absent = unlimited (unchanged default).
            "--max-blade-gb" => {
                let gb: f64 = it
                    .next()
                    .context("--max-blade-gb needs a number")?
                    .parse()
                    .context("--max-blade-gb must be a number")?;
                if !gb.is_finite() || gb < 0.0 {
                    bail!("--max-blade-gb must be a non-negative number (got {gb})");
                }
                rdma.max_blade_bytes = (gb * 1024.0 * 1024.0 * 1024.0) as u64;
            }
            "-h" | "--help" => {
                eprintln!(
                    "atlas-expert-peer — serve a resident expert store over RoCE\n\n\
                     USAGE: atlas-expert-peer --store <dir> [--listen host:port]\n\
                     \x20                      [--rdma-dev <ibdev>] [--rdma-gid <idx>]\n\
                     \x20                      [--rail <dev>:<gid> ...] [--max-blade-gb <g>]\n\
                     Serves BOTH TCP (--expert-backend rdma) and one-sided verbs\n\
                     (--expert-backend rdma-verbs) clients; the client picks per-connection.\n\
                     --rail <dev>:<gid>: repeatable; give twice to serve dual-rail clients\n\
                     \x20                 (ATLAS_EXPERT_DUAL_RAIL=1). Replaces the 1-rail default.\n\
                     --max-blade-gb <g>: cap total registered store RAM (0/absent = unlimited)\n\
                     defaults: listen 0.0.0.0:9909, single rail roceP2p1s0f1:3"
                );
                return Ok(());
            }
            other => bail!("unknown arg: {other} (try --help)"),
        }
    }
    if !custom_rails.is_empty() {
        rdma.rails = custom_rails;
    }
    let store = store.context("--store <dir> is required")?;
    spark_storage::expert_peer::serve(&store, &listen, rdma)
}
