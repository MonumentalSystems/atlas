// SPDX-License-Identifier: AGPL-3.0-only

//! `probe` — dump the GB10 power rail map and a live attributed window.
//!
//! Run on a GB10 (DGX Spark / gx10) to validate the [`atlas_power`] reader
//! against the raw `spbm` hwmon values:
//!
//! ```text
//! cargo run -p atlas-power --example probe
//! ```
//!
//! On any non-GB10 host it prints that the source is unavailable and exits 0
//! (never fabricating a number).

use std::sync::Arc;
use std::time::Duration;

use atlas_power::{Policy, PowerMonitor, PowerSource, SpbmSource};

fn main() {
    let Some(source) = SpbmSource::discover() else {
        println!("no spbm hwmon device found — not a GB10; power source unavailable.");
        return;
    };
    let descriptor = source.descriptor().clone();
    println!("== spbm source ==");
    println!("  backend           : {}", descriptor.backend);
    println!("  rails             : {:?}", descriptor.rails);
    println!("  energy_counter_hz : {}", descriptor.energy_counter_hz);
    println!("  quality           : {}", descriptor.quality.as_str());

    // One raw read.
    match source.sample() {
        Ok(s) => {
            println!("\n== raw sample ==");
            println!("  node_watts : {:?}", s.node_watts);
            println!("  pkg_uj     : {:?}", s.energy.pkg_uj);
            println!("  cpu_uj     : {:?}", s.energy.cpu_uj);
            println!("  gpu_uj     : {:?}", s.energy.gpu_uj);
        }
        Err(e) => println!("raw sample error: {e}"),
    }

    // Spawn the sampler and let idle calibrate (no request in flight).
    let monitor = PowerMonitor::spawn(Arc::new(source), 100);
    println!("\ncalibrating idle (1s, no load)…");
    std::thread::sleep(Duration::from_secs(1));
    println!("  p_idle ≈ {:.2} W", monitor.p_idle_watts());

    // Attributed window: measure ~2 s of whatever this box is doing.
    println!("\nmeasuring a 2s window…");
    let span = monitor.begin_span();
    std::thread::sleep(Duration::from_secs(2));
    let Some(report) = span.finish(None, Policy::Exclusive, hostname()) else {
        println!("source not measured — no report.");
        return;
    };

    println!("\n== attributed window (JSON) ==");
    match serde_json::to_string_pretty(&report) {
        Ok(j) => println!("{j}"),
        Err(e) => println!("serialize error: {e}"),
    }

    // Sanity cross-check: counter-derived avg node power over the window.
    let avg_w = report.joules_gross / (report.duration_ms / 1000.0).max(1e-9);
    println!(
        "\ncounter-derived avg node power over window: {avg_w:.2} W \
         (compare against `sys_total` in `sensors`/hwmon)"
    );
}

/// Best-effort node id from `/etc/hostname` (no extra deps).
fn hostname() -> Option<String> {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
