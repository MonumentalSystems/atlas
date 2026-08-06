// SPDX-License-Identifier: AGPL-3.0-only

//! GB10 `spbm` rail classifier.
//!
//! Lifted from the fleet node agent
//! (`monumental-dashboard/crates/agent/src/collect/power.rs`). The GB10
//! driver exposes a rail HIERARCHY, so loose substring matching
//! double-counts: the combined `cpu_gpu` compute rail contains the same GPU
//! draw as the `gpu` leaf, and `sys_total` already contains every leaf. The
//! symptom that motivated this (a real 2026-05-30 bug): a node reading
//! ~165 W total showed CPU 132 W + GPU 84 W = 216 W. The rules here are the
//! institutional fix; keep them in sync with the agent if new rails appear.

/// Which power bucket a hwmon channel feeds.
///
/// `Other` rails are aggregate parents / components already contained in the
/// authoritative total; `Ignore` rails are power *limits* (caps), not draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerRail {
    /// The authoritative whole-system total (`sys_total`).
    Total,
    /// A CPU leaf cluster (`cpu_p`, `cpu_e`).
    Cpu,
    /// The GPU leaf (`gpu`).
    Gpu,
    /// An aggregate parent or component already inside `Total`.
    Other,
    /// A power-limit controller rail (`pl*`, `syspl*`) — not draw.
    Ignore,
}

/// Classify a hwmon power-channel label. Pure (unit-tested against the real
/// GB10 `spbm` rail set).
///
/// Rules:
/// - `sys_total` / `total` → the authoritative whole-system total.
/// - `pl1`/`pl2`/`syspl*`/`*_limit` → power LIMITS (caps), not draw → ignore.
/// - aggregate parents that already contain leaves (`cpu_gpu`, `soc_pkg`,
///   `dc_input`, any `*_pkg`) → `Other` (never the cpu/gpu breakdown).
/// - CPU LEAF rails (`cpu_p`, `cpu_e`, generic `core`/`module`) → `Cpu`
///   (excluding anything with `gpu` in it so a combined label can't leak in).
/// - `gpu` leaf → `Gpu`.
pub fn classify_power_label(label: &str) -> PowerRail {
    let l = label.trim().to_lowercase();
    if l == "total" || l.contains("sys_total") {
        PowerRail::Total
    } else if matches!(l.as_str(), "pl1" | "pl2") || l.starts_with("syspl") || l.ends_with("_limit")
    {
        PowerRail::Ignore
    } else if matches!(l.as_str(), "cpu_gpu" | "soc_pkg" | "dc_input") || l.ends_with("_pkg") {
        // Aggregate/parent rails — already contain the leaf rails below.
        PowerRail::Other
    } else if (l.contains("cpu") || l.contains("core") || l.contains("module")) && !l.contains("gpu")
    {
        PowerRail::Cpu
    } else if l.contains("gpu") {
        PowerRail::Gpu
    } else {
        PowerRail::Other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full GB10 `spbm` power rail set (from `spark_hwmon/spbm.c`
    /// `pwr_chans[]`). Classifying it must never double-count: the sum of
    /// CPU-leaf + GPU-leaf rails must not exceed the authoritative total.
    #[test]
    fn gb10_spbm_rails_classify_without_double_counting() {
        // (label, watts) — representative live values from hwmon6.
        let rails = [
            ("sys_total", 50.7),
            ("soc_pkg", 32.7),
            ("cpu_gpu", 20.8),
            ("cpu_p", 5.6),
            ("cpu_e", 0.3),
            ("vcore", 4.4),
            ("dc_input", 52.7),
            ("gpu", 13.9),
            ("prereg", 8.0),
            ("dla", 0.3),
            ("pl1", 28.5),
            ("pl2", 32.5),
            ("syspl1", 44.7),
            ("syspl2", 46.0),
        ];
        let mut total: f64 = 0.0;
        let mut cpu: f64 = 0.0;
        let mut gpu: f64 = 0.0;
        for (label, w) in rails {
            match classify_power_label(label) {
                PowerRail::Total => total = w,
                PowerRail::Cpu => cpu += w,
                PowerRail::Gpu => gpu += w,
                PowerRail::Other | PowerRail::Ignore => {}
            }
        }
        assert!(total > 0.0, "sys_total must be recognized");
        // cpu = the CPU-domain rails caught by the leaf rules: cpu_p (5.6) +
        // cpu_e (0.3) + vcore (4.4, swept in by the generic `core` rule —
        // faithful to the upstream agent classifier). gpu = the gpu leaf.
        assert!((cpu - 10.3).abs() < 1e-6, "cpu leaves, got {cpu}");
        assert!((gpu - 13.9).abs() < 1e-6, "gpu should be the leaf only, got {gpu}");
        // The regression guard that actually matters: the breakdown never
        // overshoots the authoritative total (the 216 W > 165 W bug).
        assert!(cpu + gpu <= total, "cpu {cpu} + gpu {gpu} overshoots total {total}");
    }

    #[test]
    fn limits_are_ignored() {
        for l in ["pl1", "pl2", "syspl1", "syspl2", "gpu_limit"] {
            assert_eq!(classify_power_label(l), PowerRail::Ignore, "{l}");
        }
    }

    #[test]
    fn aggregates_are_other_not_breakdown() {
        for l in ["cpu_gpu", "soc_pkg", "dc_input", "foo_pkg"] {
            assert_eq!(classify_power_label(l), PowerRail::Other, "{l}");
        }
    }

    #[test]
    fn leaves_map_to_their_bucket() {
        assert_eq!(classify_power_label("cpu_p"), PowerRail::Cpu);
        assert_eq!(classify_power_label("cpu_e"), PowerRail::Cpu);
        assert_eq!(classify_power_label("gpu"), PowerRail::Gpu);
        assert_eq!(classify_power_label(" GPU "), PowerRail::Gpu);
    }

    #[test]
    fn total_recognized_case_insensitively() {
        assert_eq!(classify_power_label("SYS_TOTAL"), PowerRail::Total);
        assert_eq!(classify_power_label("total"), PowerRail::Total);
    }
}
