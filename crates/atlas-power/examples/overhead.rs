// SPDX-License-Identifier: AGPL-3.0-only

//! `overhead` — measure the per-request hot-path cost and the sampler cost.
//!
//! ```text
//! cargo run --release -p atlas-power --example overhead
//! ```
//!
//! The hot path (what every gated request pays) is `begin_span` + `finish`.
//! The sampler cost is one `source.sample()` + integrate, paid at `hz` per
//! second on ONE dedicated thread — its budget is a fraction of one core.

use std::sync::Arc;
use std::time::Instant;

use atlas_power::{NullSource, Policy, PowerMonitor, PowerSpan, SpbmSource};

fn main() {
    // Use the real GB10 source if present, else the Null source (the hot-path
    // cost is independent of which — it's atomic loads + arithmetic).
    let monitor: PowerMonitor = match SpbmSource::discover() {
        Some(s) => {
            println!("source: gb10_spbm (real)");
            PowerMonitor::spawn(Arc::new(s), 100)
        }
        None => {
            println!("source: null (no GB10) — hot-path timing still representative");
            PowerMonitor::spawn(Arc::new(NullSource::new()), 100)
        }
    };

    // Warm up.
    for _ in 0..1000 {
        let sp: PowerSpan = monitor.begin_span();
        let _ = sp.finish(Some(0.5), Policy::WorkWeighted, Some("bench".into()));
    }

    let n = 200_000u64;
    let t0 = Instant::now();
    for _ in 0..n {
        let sp = monitor.begin_span();
        let _ = sp.finish(Some(0.5), Policy::WorkWeighted, Some("bench".into()));
    }
    let per = t0.elapsed().as_secs_f64() / n as f64;
    let ns = per * 1e9;
    println!("hot path (begin_span + finish): {ns:.0} ns/request over {n} iters");

    // Sampler budget: fraction of one core = (sample cost) × hz.
    // Estimate sample cost by timing snapshot reads (the dominant hot part of
    // the sampler when the source read is cached); the sysfs read on a real
    // GB10 dominates but is still ~single-digit µs.
    let sn = 1_000_000u64;
    let t1 = Instant::now();
    let mut acc = 0.0;
    for _ in 0..sn {
        acc += monitor.p_idle_watts();
    }
    let snap_ns = t1.elapsed().as_secs_f64() / sn as f64 * 1e9;
    std::hint::black_box(acc);
    println!("atomic accessor read: {snap_ns:.1} ns");
    let core_frac = snap_ns * 1e-9 * 100.0 * 100.0; // ×100 Hz, ×100 for percent
    println!(
        "sampler atomic-read budget at 100 Hz: ~{core_frac:.4}% of one core \
         (real cost adds one sysfs read per tick; target <0.5%)"
    );

    let (attributed, spent, ratio) = monitor.invariant();
    println!("invariant: attributed={attributed:.1} J spent={spent:.1} J ratio={ratio:.3}");
}
