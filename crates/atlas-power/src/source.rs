// SPDX-License-Identifier: AGPL-3.0-only

//! The measurement-source abstraction.
//!
//! A [`PowerSource`] is a hardware backend that can be sampled for the
//! current node power and the cumulative hardware energy counters. The
//! trait boundary is what makes the attribution engine portable to other
//! hardware later — the GB10 `spbm` backend lives in [`crate::spbm`]; a
//! [`crate::spbm::NullSource`] returns [`SampleQuality::Unavailable`] on
//! non-GB10 hosts and never fabricates a number.

use std::fmt;

/// Whether a sample carries a real measurement.
///
/// `Measured` — read from a hardware sensor.
/// `Modeled` — derived/estimated (reserved for future model-based backends).
/// `Unavailable` — no sensor on this host; the caller MUST omit power
/// metadata entirely rather than report a zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleQuality {
    Measured,
    Modeled,
    Unavailable,
}

impl SampleQuality {
    /// Lower-case wire token (`"measured"` / `"modeled"` / `"unavailable"`).
    pub fn as_str(self) -> &'static str {
        match self {
            SampleQuality::Measured => "measured",
            SampleQuality::Modeled => "modeled",
            SampleQuality::Unavailable => "unavailable",
        }
    }
}

/// Cumulative hardware energy counters, in **microjoules** (the native
/// `spbm` `energy*_input` unit). Monotonic; each is `None` when the backing
/// rail is absent. `pkg` is the SoC package domain, `gpu` the GPU domain,
/// `cpu` the sum of the CPU leaf clusters — these are the domains the GB10
/// exposes a real accumulator for. The node-wide `sys_total` rail has no
/// counter and is integrated in software (see [`crate::monitor`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EnergyCounters {
    pub pkg_uj: Option<u64>,
    pub cpu_uj: Option<u64>,
    pub gpu_uj: Option<u64>,
}

/// One instantaneous read of a [`PowerSource`].
///
/// `node_watts` is the authoritative whole-system rail (`sys_total`),
/// instantaneous. `energy` carries the cumulative hardware energy counters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerSample {
    /// Instantaneous node-wide power in watts (`sys_total`), if the rail is
    /// present.
    pub node_watts: Option<f64>,
    /// Cumulative hardware energy accumulators (microjoules).
    pub energy: EnergyCounters,
    /// Whether this sample is a real measurement.
    pub quality: SampleQuality,
}

impl PowerSample {
    /// A sample from a host with no power sensor.
    pub fn unavailable() -> Self {
        Self {
            node_watts: None,
            energy: EnergyCounters::default(),
            quality: SampleQuality::Unavailable,
        }
    }
}

/// Static description of a source: which rails it reads and at what
/// resolution. Serialized into the response `source` sub-object.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceDescriptor {
    /// Backend identifier, e.g. `"gb10_spbm"` or `"null"`.
    pub backend: &'static str,
    /// Human-readable rail labels this source reads (`sys_total`, `pkg`, …).
    pub rails: Vec<String>,
    /// Hardware energy-counter update rate in Hz (best-effort; 0 if the
    /// backend has no energy counter).
    pub energy_counter_hz: u32,
    /// Whether the backend can currently produce a `Measured` sample.
    pub quality: SampleQuality,
}

/// Error reading a source.
#[derive(Debug)]
pub enum SourceError {
    /// The sysfs/hwmon surface could not be read.
    Io(std::io::Error),
    /// The backend is present but produced no usable rail this read.
    NoData,
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceError::Io(e) => write!(f, "power source io error: {e}"),
            SourceError::NoData => write!(f, "power source produced no data"),
        }
    }
}

impl std::error::Error for SourceError {}

impl From<std::io::Error> for SourceError {
    fn from(e: std::io::Error) -> Self {
        SourceError::Io(e)
    }
}

/// A hardware power/energy measurement backend.
///
/// Implementations must be cheap to [`sample`](PowerSource::sample) — the
/// monitor calls it at ~100 Hz — and must be `Send + Sync` so the sampler
/// thread can own one behind an `Arc`.
pub trait PowerSource: Send + Sync {
    /// Read the current node power and cumulative energy counters.
    fn sample(&self) -> Result<PowerSample, SourceError>;

    /// Static description of what this source measures.
    fn descriptor(&self) -> &SourceDescriptor;

    /// Current best assessment of whether this source yields real
    /// measurements. Defaults to the descriptor's recorded quality.
    fn quality(&self) -> SampleQuality {
        self.descriptor().quality
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_wire_tokens() {
        assert_eq!(SampleQuality::Measured.as_str(), "measured");
        assert_eq!(SampleQuality::Modeled.as_str(), "modeled");
        assert_eq!(SampleQuality::Unavailable.as_str(), "unavailable");
    }

    #[test]
    fn unavailable_sample_has_no_numbers() {
        let s = PowerSample::unavailable();
        assert_eq!(s.quality, SampleQuality::Unavailable);
        assert!(s.node_watts.is_none());
        assert_eq!(s.energy, EnergyCounters::default());
    }

    #[test]
    fn source_error_from_io() {
        let e: SourceError = std::io::Error::other("boom").into();
        assert!(matches!(e, SourceError::Io(_)));
        assert!(e.to_string().contains("boom"));
    }
}
