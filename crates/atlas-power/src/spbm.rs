// SPDX-License-Identifier: AGPL-3.0-only

//! GB10 `spbm` HWMON backend + a `Null` fallback.
//!
//! [`SpbmSource`] reads the out-of-tree `spbm` driver
//! (`~/spark_hwmon`, hwmon name `spbm`, ACPI `NVDA8800`):
//! - `power*_input` (microwatts) — classified via [`crate::classify`]; the
//!   `sys_total` rail is the authoritative node power.
//! - `energy*_input` (microjoules, cumulative, ~100 Hz) — the hardware
//!   energy accumulators for the `pkg` / `cpu` / `gpu` domains.
//!
//! [`NullSource`] is used on any host without spbm; it reports
//! [`SampleQuality::Unavailable`] and never fabricates a number.

use std::path::{Path, PathBuf};

use crate::classify::{PowerRail, classify_power_label};
use crate::source::{
    EnergyCounters, PowerSample, PowerSource, SampleQuality, SourceDescriptor, SourceError,
};

/// Best-effort effective update rate of the spbm energy accumulators.
/// Measured live on hwmon6: the counters step every ~10.3 ms.
const ENERGY_COUNTER_HZ: u32 = 100;

/// A discovered `power*_input` channel and its classified rail.
#[derive(Debug, Clone)]
struct PowerChan {
    rail: PowerRail,
    path: PathBuf,
}

/// A discovered `energy*_input` accumulator and its domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnergyDomain {
    Pkg,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone)]
struct EnergyChan {
    domain: EnergyDomain,
    path: PathBuf,
}

/// Classify an `energy*_label`. `pkg` → whole SoC package; `cpu_p`/`cpu_e`
/// → the CPU leaves (summed); `gpu` → the GPU domain. Anything else is
/// ignored (no other energy rails exist on the GB10 today).
fn classify_energy_label(label: &str) -> Option<EnergyDomain> {
    let l = label.trim().to_lowercase();
    if l == "pkg" || l.ends_with("_pkg") {
        Some(EnergyDomain::Pkg)
    } else if l.contains("gpu") {
        Some(EnergyDomain::Gpu)
    } else if l.contains("cpu") || l.contains("core") {
        Some(EnergyDomain::Cpu)
    } else {
        None
    }
}

/// The GB10 spbm HWMON source.
pub struct SpbmSource {
    power: Vec<PowerChan>,
    energy: Vec<EnergyChan>,
    descriptor: SourceDescriptor,
}

impl SpbmSource {
    /// Scan `/sys/class/hwmon` for the `spbm` device. Returns `None` if no
    /// spbm hwmon node is present (i.e. not a GB10).
    pub fn discover() -> Option<Self> {
        Self::discover_in(Path::new("/sys/class/hwmon"))
    }

    /// Scan a hwmon-class root (parameterized for tests against a synthetic
    /// sysfs tree). Returns `None` if no `spbm`-named device with power rails
    /// is found.
    pub fn discover_in(hwmon_root: &Path) -> Option<Self> {
        let entries = std::fs::read_dir(hwmon_root).ok()?;
        for e in entries.flatten() {
            let dir = e.path();
            let name = std::fs::read_to_string(dir.join("name"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            if name != "spbm" {
                continue;
            }
            if let Some(src) = Self::from_device_dir(&dir) {
                return Some(src);
            }
        }
        None
    }

    /// Build a source from a single hwmon device directory (the `spbm`
    /// node, or a synthetic equivalent in tests).
    pub fn from_device_dir(dir: &Path) -> Option<Self> {
        let (power, energy) = discover_channels(dir);
        if power.is_empty() && energy.is_empty() {
            return None;
        }
        let mut rails: Vec<String> = power
            .iter()
            .filter(|c| c.rail == PowerRail::Total)
            .map(|_| "sys_total".to_string())
            .collect();
        for c in &energy {
            let name = match c.domain {
                EnergyDomain::Pkg => "pkg",
                EnergyDomain::Cpu => "cpu",
                EnergyDomain::Gpu => "gpu",
            };
            // Two CPU leaf counters (cpu_p + cpu_e) share one `cpu` domain —
            // list each rail name once.
            if !rails.iter().any(|r| r == name) {
                rails.push(name.to_string());
            }
        }
        let descriptor = SourceDescriptor {
            backend: "gb10_spbm",
            rails,
            energy_counter_hz: if energy.is_empty() { 0 } else { ENERGY_COUNTER_HZ },
            quality: SampleQuality::Measured,
        };
        Some(Self {
            power,
            energy,
            descriptor,
        })
    }
}

/// Read a `*_input` sysfs file into a `u64` (microwatts / microjoules).
fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
}

/// Enumerate the `power*`/`energy*` `_input` channels in a hwmon device dir,
/// pairing each with its `_label` and classifying it.
fn discover_channels(dir: &Path) -> (Vec<PowerChan>, Vec<EnergyChan>) {
    let mut power = Vec::new();
    let mut energy = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (power, energy);
    };
    for s in entries.flatten() {
        let sp = s.path();
        let Some(sn) = sp.file_name().and_then(|x| x.to_str()) else {
            continue;
        };
        if !sn.ends_with("_input") {
            continue;
        }
        if let Some(n) = sn.strip_prefix("power").and_then(|r| r.strip_suffix("_input")) {
            let label = read_label(dir, "power", n);
            power.push(PowerChan {
                rail: classify_power_label(&label),
                path: sp,
            });
        } else if let Some(n) = sn.strip_prefix("energy").and_then(|r| r.strip_suffix("_input")) {
            let label = read_label(dir, "energy", n);
            if let Some(domain) = classify_energy_label(&label) {
                energy.push(EnergyChan { domain, path: sp });
            }
        }
    }
    (power, energy)
}

fn read_label(dir: &Path, kind: &str, n: &str) -> String {
    std::fs::read_to_string(dir.join(format!("{kind}{n}_label")))
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

impl PowerSource for SpbmSource {
    fn sample(&self) -> Result<PowerSample, SourceError> {
        let mut node_watts = None;
        for ch in &self.power {
            if ch.rail != PowerRail::Total {
                continue;
            }
            if let Some(uw) = read_u64(&ch.path) {
                node_watts = Some(uw as f64 / 1_000_000.0);
            }
        }
        let mut energy = EnergyCounters::default();
        for ch in &self.energy {
            let Some(uj) = read_u64(&ch.path) else {
                continue;
            };
            match ch.domain {
                EnergyDomain::Pkg => energy.pkg_uj = Some(uj),
                EnergyDomain::Gpu => energy.gpu_uj = Some(uj),
                // cpu_p + cpu_e accumulate into one CPU-domain counter.
                EnergyDomain::Cpu => energy.cpu_uj = Some(energy.cpu_uj.unwrap_or(0) + uj),
            }
        }
        if node_watts.is_none() && energy == EnergyCounters::default() {
            return Err(SourceError::NoData);
        }
        Ok(PowerSample {
            node_watts,
            energy,
            quality: SampleQuality::Measured,
        })
    }

    fn descriptor(&self) -> &SourceDescriptor {
        &self.descriptor
    }
}

/// A source for hosts without spbm: always [`SampleQuality::Unavailable`].
pub struct NullSource {
    descriptor: SourceDescriptor,
}

impl Default for NullSource {
    fn default() -> Self {
        Self::new()
    }
}

impl NullSource {
    pub fn new() -> Self {
        Self {
            descriptor: SourceDescriptor {
                backend: "null",
                rails: Vec::new(),
                energy_counter_hz: 0,
                quality: SampleQuality::Unavailable,
            },
        }
    }
}

impl PowerSource for NullSource {
    fn sample(&self) -> Result<PowerSample, SourceError> {
        Ok(PowerSample::unavailable())
    }

    fn descriptor(&self) -> &SourceDescriptor {
        &self.descriptor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(dir: &Path, name: &str, contents: &str) {
        fs::write(dir.join(name), contents).unwrap();
    }

    /// Build a synthetic `spbm` hwmon device directory with the real rail
    /// layout, then read it through the source.
    fn synth_spbm(root: &Path) -> PathBuf {
        let dev = root.join("hwmon6");
        fs::create_dir_all(&dev).unwrap();
        write(&dev, "name", "spbm\n");
        // power rails (microwatts)
        let power = [
            ("1", "sys_total", "50687000"),
            ("2", "soc_pkg", "32672000"),
            ("4", "cpu_p", "5647000"),
            ("8", "gpu", "13897000"),
            ("11", "pl1", "28548000"),
        ];
        for (n, label, val) in power {
            write(&dev, &format!("power{n}_label"), &format!("{label}\n"));
            write(&dev, &format!("power{n}_input"), &format!("{val}\n"));
        }
        // energy accumulators (microjoules)
        let energy = [
            ("1", "pkg", "2135422449000"),
            ("2", "cpu_e", "95361764000"),
            ("3", "cpu_p", "1496686770000"),
            ("4", "gpu", "870868565000"),
        ];
        for (n, label, val) in energy {
            write(&dev, &format!("energy{n}_label"), &format!("{label}\n"));
            write(&dev, &format!("energy{n}_input"), &format!("{val}\n"));
        }
        dev
    }

    #[test]
    fn discovers_and_reads_synthetic_spbm() {
        let tmp = std::env::temp_dir().join(format!("atlas-power-spbm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        synth_spbm(&tmp);

        let src = SpbmSource::discover_in(&tmp).expect("should find spbm device");
        assert_eq!(src.descriptor().backend, "gb10_spbm");
        assert_eq!(src.descriptor().energy_counter_hz, ENERGY_COUNTER_HZ);
        assert_eq!(src.quality(), SampleQuality::Measured);

        let s = src.sample().unwrap();
        assert_eq!(s.quality, SampleQuality::Measured);
        // sys_total 50.687 W.
        assert!((s.node_watts.unwrap() - 50.687).abs() < 1e-3);
        // pkg / gpu counters pass through; cpu = cpu_e + cpu_p.
        assert_eq!(s.energy.pkg_uj, Some(2135422449000));
        assert_eq!(s.energy.gpu_uj, Some(870868565000));
        assert_eq!(s.energy.cpu_uj, Some(95361764000 + 1496686770000));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn no_spbm_device_is_none() {
        let tmp = std::env::temp_dir().join(format!("atlas-power-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        // an unrelated hwmon device (e.g. nvme) — not spbm.
        let dev = tmp.join("hwmon1");
        fs::create_dir_all(&dev).unwrap();
        fs::write(dev.join("name"), "nvme\n").unwrap();
        assert!(SpbmSource::discover_in(&tmp).is_none());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn null_source_never_fabricates() {
        let n = NullSource::new();
        assert_eq!(n.quality(), SampleQuality::Unavailable);
        let s = n.sample().unwrap();
        assert_eq!(s.quality, SampleQuality::Unavailable);
        assert!(s.node_watts.is_none());
    }

    #[test]
    fn energy_label_classification() {
        assert_eq!(classify_energy_label("pkg"), Some(EnergyDomain::Pkg));
        assert_eq!(classify_energy_label("gpu"), Some(EnergyDomain::Gpu));
        assert_eq!(classify_energy_label("cpu_p"), Some(EnergyDomain::Cpu));
        assert_eq!(classify_energy_label("cpu_e"), Some(EnergyDomain::Cpu));
        assert_eq!(classify_energy_label("weird"), None);
    }
}
