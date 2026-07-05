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
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--store" | "-s" => store = Some(it.next().context("--store needs a dir")?.into()),
            "--listen" | "-l" => listen = it.next().context("--listen needs host:port")?,
            "--rdma-dev" => rdma.dev = it.next().context("--rdma-dev needs a device name")?,
            "--rdma-gid" => {
                rdma.gid_idx = it
                    .next()
                    .context("--rdma-gid needs an index")?
                    .parse()
                    .context("--rdma-gid must be an integer")?
            }
            "-h" | "--help" => {
                eprintln!(
                    "atlas-expert-peer — serve a resident expert store over RoCE\n\n\
                     USAGE: atlas-expert-peer --store <dir> [--listen host:port]\n\
                     \x20                      [--rdma-dev <ibdev>] [--rdma-gid <idx>]\n\
                     Serves BOTH TCP (--expert-backend rdma) and one-sided verbs\n\
                     (--expert-backend rdma-verbs) clients; the client picks per-connection.\n\
                     defaults: listen 0.0.0.0:9909, rdma-dev roceP2p1s0f1, rdma-gid 3"
                );
                return Ok(());
            }
            other => bail!("unknown arg: {other} (try --help)"),
        }
    }
    let store = store.context("--store <dir> is required")?;
    spark_storage::expert_peer::serve(&store, &listen, rdma)
}
