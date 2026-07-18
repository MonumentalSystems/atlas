// SPDX-License-Identifier: AGPL-3.0-only

//! Atlas native power/energy attribution.
//!
//! Reports, as opt-in per-request metadata, how much energy a request
//! consumed on a GB10 (NVIDIA Grace-Blackwell / DGX Spark). This is the
//! same physical quantity the fleet's Saturn/Hyades metering pipeline
//! attributes externally, done natively inside the single Atlas process
//! with far better temporal resolution and no cross-system clock
//! reconciliation.
//!
//! ## Honesty contract
//!
//! Apportioned energy is an *estimate*. Every [`PowerReport`] carries
//! `estimated: true` unconditionally, the [`Policy`] that produced it, and
//! a [`Confidence`]. When no measurement is available the report is omitted
//! entirely rather than zero-filled — [`SampleQuality::Unavailable`] never
//! fabricates a number. This mirrors the fleet's existing
//! "show measured or n/a" rule.
//!
//! ## Units (reconciliation with Hyades)
//!
//! Energy crosses the fleet boundary as **plain SI joules = watt-seconds**
//! (`f64`); `kWh = joules / 3.6e6`. This crate uses joules everywhere on its
//! public surface. Internally the GB10 `spbm` hardware energy counters are
//! microjoules; conversion happens at the source boundary only.
//!
//! ## Measurement source
//!
//! On a GB10 the out-of-tree `spbm` HWMON driver exposes cumulative
//! hardware **energy accumulators** (`energy*_input`, microjoules, ~100 Hz,
//! monotonic) for the `pkg`/`cpu`/`gpu` domains — the hardware integrates
//! power to energy for us, so a request's compute-domain energy is an exact
//! counter delta rather than a Riemann sum over sampled watts. The
//! node-wide `sys_total` rail has *no* energy counter, so its cumulative
//! joules are integrated in software by the [`monitor`] sampler thread.
//!
//! See [`classify`] for the rail taxonomy (lifted from the fleet agent's
//! anti-double-count classifier) and [`attribute`] for the marginal /
//! gross / baseline split and apportionment policies.

pub mod attribute;
pub mod classify;
pub mod monitor;
pub mod source;
pub mod span;
pub mod spbm;

pub use attribute::{Confidence, ConcurrencyStats, PowerReport, Policy, SourceInfo, attribute};
pub use classify::{PowerRail, classify_power_label};
pub use monitor::PowerMonitor;
pub use source::{
    EnergyCounters, PowerSample, PowerSource, SampleQuality, SourceDescriptor, SourceError,
};
pub use span::PowerSpan;
pub use spbm::{NullSource, SpbmSource};
