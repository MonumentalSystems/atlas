// SPDX-License-Identifier: AGPL-3.0-only
//
// atlas-cache-peer: a RW remote-RAM overflow blade for the high-speed-swap KV
// cache. A dumb memory tier — a streaming client offloads cold K/V groups into
// it via one-sided RDMA WRITE and restores them via RDMA READ; the peer CPU
// never touches a byte. Faster-than-SSD KV overflow (peer RAM ~12 GB/s over CX7
// vs ~2 GB/s USB SSD).
//
//   atlas-cache-peer [--listen 0.0.0.0:9910] [--rdma-dev <ibdev>] [--rdma-gid <idx>]
//
// The client selects it with $ATLAS_KV_PEER=host:port. Arena size is set by the
// client's GroupLayout at connect time (one RW MR per connection).

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let mut listen = String::from("0.0.0.0:9910");
    let mut rdma = spark_storage::cache_peer::RdmaConfig::default();
    let mut custom_rails: Vec<(String, u32)> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--listen" | "-l" => listen = it.next().context("--listen needs host:port")?,
            // Repeatable: --rail <dev>:<gid_idx>. First --rail replaces the
            // 2-rail default; give it twice for both CX7 adapters.
            "--rail" => {
                let spec = it.next().context("--rail needs <dev>:<gid_idx>")?;
                let (dev, gid) = spec
                    .rsplit_once(':')
                    .context("--rail format is <dev>:<gid_idx>")?;
                custom_rails.push((dev.to_string(), gid.parse().context("gid must be int")?));
            }
            // Ceiling on total committed blade RAM (GiB) across all concurrent
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
            // Directory for NVMe swap files backing paging-mode clients (WS-A):
            // a paging connection's RDMA arena becomes a page-cache over an
            // O_DIRECT swap file here, giving the SSM-snapshot tier infinite
            // depth. Absent = paging clients are refused (RAM-only, legacy).
            "--swap-dir" => {
                let d = it.next().context("--swap-dir needs a path")?;
                rdma.swap_dir = Some(std::path::PathBuf::from(d));
            }
            // Disk cap for the paging swap file (GiB); coldest on-disk snapshot
            // dropped when full. 0 = unbounded. Default 50 GiB.
            "--swap-cap-gb" => {
                let g: f64 = it
                    .next()
                    .context("--swap-cap-gb needs a number")?
                    .parse()
                    .context("--swap-cap-gb must be a number")?;
                if !g.is_finite() || g < 0.0 {
                    bail!("--swap-cap-gb must be non-negative (got {g})");
                }
                rdma.swap_cap_bytes = (g * 1024.0 * 1024.0 * 1024.0) as u64;
            }
            "-h" | "--help" => {
                eprintln!(
                    "atlas-cache-peer — RW RDMA overflow blade for the KV cache\n\n\
                     USAGE: atlas-cache-peer [--listen host:port] [--rail <dev>:<gid> ...]\n\
                     \x20                  [--max-blade-gb <g>] [--swap-dir <dir>]\n\
                     defaults: listen 0.0.0.0:9910, rails roceP2p1s0f1:3 rocep1s0f1:3\n\
                     --max-blade-gb <g>: cap total blade RAM (0/absent = unlimited)\n\
                     --swap-dir <dir>: NVMe dir backing PAGING clients (WS-A) →\n\
                     \x20                 RAM arena becomes a page-cache over disk\n\
                     --swap-cap-gb <g>: disk cap for the paging swap (default 50; 0=unbounded)\n\
                     client selects via $ATLAS_KV_PEER=host:port ($ATLAS_KV_DUAL_RAIL=1 for both)"
                );
                return Ok(());
            }
            other => bail!("unknown arg: {other} (try --help)"),
        }
    }
    if !custom_rails.is_empty() {
        rdma.rails = custom_rails;
    }
    spark_storage::cache_peer::serve(&listen, rdma)
}
