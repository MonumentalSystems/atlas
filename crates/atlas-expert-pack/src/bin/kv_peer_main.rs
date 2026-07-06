// SPDX-License-Identifier: AGPL-3.0-only
//
// atlas-kv-peer: a RW remote-RAM overflow blade for the high-speed-swap KV
// cache. A dumb memory tier — a streaming client offloads cold K/V groups into
// it via one-sided RDMA WRITE and restores them via RDMA READ; the peer CPU
// never touches a byte. Faster-than-SSD KV overflow (peer RAM ~12 GB/s over CX7
// vs ~2 GB/s USB SSD).
//
//   atlas-kv-peer [--listen 0.0.0.0:9910] [--rdma-dev <ibdev>] [--rdma-gid <idx>]
//
// The client selects it with $ATLAS_KV_PEER=host:port. Arena size is set by the
// client's GroupLayout at connect time (one RW MR per connection).

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let mut listen = String::from("0.0.0.0:9910");
    let mut rdma = spark_storage::kv_peer::RdmaConfig::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
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
                    "atlas-kv-peer — RW RDMA overflow blade for the KV cache\n\n\
                     USAGE: atlas-kv-peer [--listen host:port] [--rdma-dev <ibdev>] [--rdma-gid <idx>]\n\
                     defaults: listen 0.0.0.0:9910, rdma-dev roceP2p1s0f1, rdma-gid 3\n\
                     client selects via $ATLAS_KV_PEER=host:port"
                );
                return Ok(());
            }
            other => bail!("unknown arg: {other} (try --help)"),
        }
    }
    spark_storage::kv_peer::serve(&listen, rdma)
}
