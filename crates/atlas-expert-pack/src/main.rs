// SPDX-License-Identifier: AGPL-3.0-only
//
// CLI for the offline expert-file builder.
//
//   atlas-expert-pack --checkpoint <dir> --out <dir> [--max-layers N]
//                     [--verify N] [--dry-run]
//
// `--dry-run` discovers geometry and prints what would be built without writing.

use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::time::Instant;

use atlas_expert_pack::build::{BuildOpts, discover_geometry, run_build};
use atlas_expert_pack::checkpoint::Checkpoint;

struct Args {
    checkpoint: PathBuf,
    out: PathBuf,
    max_layers: u32,
    verify: u32,
    dry_run: bool,
}

fn parse_args() -> Result<Args> {
    let mut checkpoint: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut max_layers = 0u32;
    let mut verify = 8u32;
    let mut dry_run = false;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--checkpoint" | "-c" => {
                checkpoint = Some(it.next().context("--checkpoint needs a value")?.into())
            }
            "--out" | "-o" => out = Some(it.next().context("--out needs a value")?.into()),
            "--max-layers" => {
                max_layers = it.next().context("--max-layers needs a value")?.parse()?
            }
            "--verify" => verify = it.next().context("--verify needs a value")?.parse()?,
            "--dry-run" => dry_run = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other} (try --help)"),
        }
    }
    Ok(Args {
        checkpoint: checkpoint.context("--checkpoint <dir> is required")?,
        out: out.unwrap_or_else(|| PathBuf::from("./expert-store")),
        max_layers,
        verify,
        dry_run,
    })
}

fn print_help() {
    eprintln!(
        "atlas-expert-pack — build a resident-layout expert store from an NVFP4 MoE checkpoint\n\n\
         USAGE:\n  atlas-expert-pack --checkpoint <dir> --out <dir> [options]\n\n\
         OPTIONS:\n\
           -c, --checkpoint <dir>   checkpoint dir (model.safetensors[.index.json])\n\
           -o, --out <dir>          output store dir (default ./expert-store)\n\
               --max-layers <N>     convert only the first N MoE layers (0 = all)\n\
               --verify <N>         read back + byte-verify N sampled records (default 8)\n\
               --dry-run            discover geometry and print a plan; write nothing\n"
    );
}

fn human(bytes: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.2} {}", U[i])
}

fn main() -> Result<()> {
    let args = parse_args()?;

    if args.dry_run {
        let ckpt = Checkpoint::open(&args.checkpoint)?;
        let geo = discover_geometry(&ckpt)?;
        println!("checkpoint : {}", args.checkpoint.display());
        println!("prefix     : {}", geo.base_prefix);
        println!(
            "moe layers : {} (abs {}..={})",
            geo.moe_layers.len(),
            geo.moe_layers.first().copied().unwrap_or(0),
            geo.moe_layers.last().copied().unwrap_or(0),
        );
        println!("experts    : {} per layer", geo.num_experts);
        println!(
            "dims       : inter={} hidden={} group_size={} input_scale={}",
            geo.inter, geo.hidden, geo.group_size, geo.has_input_scale
        );
        let per_expert = 3 * geo.inter * geo.hidden * 9 / 16;
        let total = per_expert * geo.num_experts as u64 * geo.moe_layers.len() as u64;
        println!(
            "payload    : {}/expert, ~{} full model (pre-alignment)",
            human(per_expert),
            human(total)
        );
        return Ok(());
    }

    println!(
        "building expert store: {} -> {}",
        args.checkpoint.display(),
        args.out.display()
    );
    let t0 = Instant::now();
    let report = run_build(
        &args.checkpoint,
        &args.out,
        BuildOpts {
            max_layers: args.max_layers,
            verify_samples: args.verify,
        },
    )?;
    let dt = t0.elapsed().as_secs_f64();

    let g = &report.geometry;
    println!("done in {dt:.1}s");
    println!(
        "  dims          : inter={} hidden={} group_size={} experts={}",
        g.inter, g.hidden, g.group_size, g.num_experts
    );
    println!(
        "  layers built  : {} of {}",
        report.layers_built,
        g.moe_layers.len()
    );
    println!("  experts built : {}", report.experts_built);
    println!(
        "  payload/expert: {}   record stride: {}",
        human(report.bytes_per_expert_payload),
        human(report.record_stride)
    );
    println!("  bytes written : {}", human(report.bytes_written));
    if report.experts_built > 0 && dt > 0.0 {
        let read_bytes = report.bytes_per_expert_payload * report.experts_built;
        println!(
            "  throughput    : {}/s (source read+transpose+write)",
            human((read_bytes as f64 / dt) as u64)
        );
    }
    println!(
        "  verified      : {} sampled records byte-identical after round trip",
        report.verified
    );
    Ok(())
}
