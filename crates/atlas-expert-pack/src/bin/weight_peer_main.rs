// SPDX-License-Identifier: AGPL-3.0-only
//
// atlas-weight-peer: serve staged model weights to swap clients over the RoCE
// fabric for FAST MODEL SWAPS. Holds one or more models' safetensors shards
// mmap'd + `ibv_reg_mr`'d REMOTE_READ in RAM; a client (`$ATLAS_WEIGHT_PEER`)
// requests a model by id/path and one-sided RDMA-READs its weight tensors
// (~24 GB/s dual-rail) instead of the ~2 GB/s USB SSD.
//
//   atlas-weight-peer --stage <model_dir> [--stage <dir2> ...] [--listen 0.0.0.0:9910]
//                     [--rail <dev>:<gid> ...] [--max-blade-gb <g>] [--allow-any-path]
//
// It's a cache: the first stage of a model faults its pages in from disk (slow);
// every later swap reads them from the peer's warm RAM (fast). Pre-warm the
// rotation set with `--stage`.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let mut listen = String::from("0.0.0.0:9910");
    let mut cfg = spark_storage::weight_peer::WeightPeerConfig::default();
    // Repeatable `--rail <dev>:<gid>`; if any are given they REPLACE the
    // single-rail default (mirrors atlas-expert-peer).
    let mut custom_rails: Vec<(String, u32)> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--stage" | "-s" => {
                let d: PathBuf = it.next().context("--stage needs a model dir")?.into();
                cfg.staged_dirs.push(d);
            }
            "--listen" | "-l" => listen = it.next().context("--listen needs host:port")?,
            "--rdma-dev" => cfg.rails[0].0 = it.next().context("--rdma-dev needs a device name")?,
            "--rdma-gid" => {
                cfg.rails[0].1 = it
                    .next()
                    .context("--rdma-gid needs an index")?
                    .parse()
                    .context("--rdma-gid must be an integer")?
            }
            "--rail" => {
                let spec = it.next().context("--rail needs <dev>:<gid_idx>")?;
                let (dev, gid) = spec
                    .rsplit_once(':')
                    .context("--rail format is <dev>:<gid_idx>")?;
                custom_rails.push((dev.to_string(), gid.parse().context("gid must be int")?));
            }
            "--max-blade-gb" => {
                let gb: f64 = it
                    .next()
                    .context("--max-blade-gb needs a number")?
                    .parse()
                    .context("--max-blade-gb must be a number")?;
                if !gb.is_finite() || gb < 0.0 {
                    bail!("--max-blade-gb must be a non-negative number (got {gb})");
                }
                cfg.max_blade_bytes = (gb * 1024.0 * 1024.0 * 1024.0) as u64;
            }
            // Allow a client to request ANY filesystem path (trusted LAN). Off by
            // default: clients may only request a pre-staged dir (or its basename).
            "--allow-any-path" => cfg.allow_any_path = true,
            "-h" | "--help" => {
                eprintln!(
                    "atlas-weight-peer — serve staged model weights over RoCE for fast swaps\n\n\
                     USAGE: atlas-weight-peer --stage <model_dir> [--stage <dir> ...]\n\
                     \x20                      [--listen host:port] [--rdma-dev <ibdev>] [--rdma-gid <idx>]\n\
                     \x20                      [--rail <dev>:<gid> ...] [--max-blade-gb <g>] [--allow-any-path]\n\
                     Client: set $ATLAS_WEIGHT_PEER=host:port; the server serves one-sided\n\
                     verbs (RDMA READ) clients. --rail: repeatable; give twice for dual-rail.\n\
                     --max-blade-gb <g>: cap total staged RAM (0/absent = unlimited)\n\
                     defaults: listen 0.0.0.0:9910, single rail roceP2p1s0f1:3"
                );
                return Ok(());
            }
            other => bail!("unknown arg: {other} (try --help)"),
        }
    }
    if !custom_rails.is_empty() {
        cfg.rails = custom_rails;
    }
    if cfg.staged_dirs.is_empty() && !cfg.allow_any_path {
        bail!("nothing to serve: pass --stage <model_dir> (or --allow-any-path)");
    }
    spark_storage::weight_peer::serve(&listen, cfg)
}
