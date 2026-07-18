// SPDX-License-Identifier: AGPL-3.0-only

//! [`PowerSpan`] — the RAII per-request measurement window.
//!
//! `enter` captures the monitor's cumulative counters and registers the
//! request as in-flight; [`PowerSpan::finish`] reads the counters again and
//! produces an honest [`PowerReport`]. Dropping the span always balances the
//! in-flight count, so a request that errors out mid-flight still
//! deregisters.
//!
//! ```ignore
//! let span = monitor.begin_span();
//! // ... handler work ...
//! let report = span.finish(work_share, Policy::WorkWeighted, node_id);
//! ```

use std::sync::Arc;
use std::time::Instant;

use crate::attribute::{
    AttributionInput, ConcurrencyStats, MeasuredEnergy, Policy, PowerReport, SourceInfo, attribute,
};
use crate::monitor::{MonitorSnapshot, PowerMonitor, Shared};
use crate::source::SampleQuality;

/// A per-request power-measurement window.
pub struct PowerSpan {
    shared: Arc<Shared>,
    start: Instant,
    start_snap: MonitorSnapshot,
    /// In-flight count at entry (includes this request).
    start_inflight: u64,
    finished: bool,
}

impl PowerMonitor {
    /// Begin a measurement window and register the request as in-flight.
    pub fn begin_span(&self) -> PowerSpan {
        let shared = self.shared();
        let start_inflight = shared.inc_inflight();
        let start_snap = shared.snapshot();
        PowerSpan {
            start: Instant::now(),
            start_snap,
            start_inflight,
            shared,
            finished: false,
        }
    }
}

impl PowerSpan {
    /// Close the window and compute the report.
    ///
    /// `work_share` is this request's fraction of total work over the window
    /// (e.g. its tokens / all tokens served concurrently), used only under
    /// [`Policy::WorkWeighted`]; pass `None` to fall back to an equal slice.
    /// Returns `None` when the source is not `Measured` — the caller must
    /// then omit power metadata entirely (never zero-fill).
    pub fn finish(
        &self,
        work_share: Option<f64>,
        requested_policy: Policy,
        node_id: Option<String>,
    ) -> Option<PowerReport> {
        let duration_s = self.start.elapsed().as_secs_f64();
        self.finish_inner(duration_s, work_share, requested_policy, node_id)
    }

    /// Core of [`finish`] with the window duration supplied explicitly so
    /// tests can keep it consistent with a simulated integration clock.
    fn finish_inner(
        &self,
        duration_s: f64,
        work_share: Option<f64>,
        requested_policy: Policy,
        node_id: Option<String>,
    ) -> Option<PowerReport> {
        if self.shared.quality() != SampleQuality::Measured {
            return None;
        }
        let end = self.shared.snapshot();

        let gross_joules = (end.node_joules - self.start_snap.node_joules).max(0.0);
        let measured = MeasuredEnergy {
            pkg_joules: delta_uj_to_j(self.start_snap.energy.pkg_uj, end.energy.pkg_uj),
            cpu_joules: delta_uj_to_j(self.start_snap.energy.cpu_uj, end.energy.cpu_uj),
            gpu_joules: delta_uj_to_j(self.start_snap.energy.gpu_uj, end.energy.gpu_uj),
        };

        let concurrency = self.concurrency(&end, duration_s);
        let samples_in_window = end.samples.saturating_sub(self.start_snap.samples);

        let desc = self.shared.descriptor();
        let source = SourceInfo {
            backend: desc.backend.to_string(),
            rails: desc.rails.clone(),
            quality: self.shared.quality().as_str().to_string(),
            energy_counter_hz: desc.energy_counter_hz,
            samples_in_window,
        };

        Some(attribute(AttributionInput {
            duration_s,
            gross_joules,
            p_idle_watts: self.shared.p_idle_watts(),
            concurrency,
            requested_policy,
            work_share,
            measured,
            source,
            node_id,
        }))
    }

    /// Recover concurrency stats over the window. Mean is time-weighted from
    /// the monitor's in-flight integral; min/max are approximated from the
    /// window endpoints (flat v1 — nested/instantaneous peaks are not tracked).
    fn concurrency(&self, end: &MonitorSnapshot, duration_s: f64) -> ConcurrencyStats {
        let mean = if duration_s > 0.0 {
            let d_integral =
                end.inflight_integral_us.saturating_sub(self.start_snap.inflight_integral_us);
            (d_integral as f64 / 1_000_000.0 / duration_s).max(1.0)
        } else {
            self.start_inflight as f64
        };
        let a = self.start_inflight;
        let b = end.inflight.max(1);
        // `round`, not `ceil`: a time-weighted mean of 1.003 (solo request,
        // integ_/duration jitter) must not inflate max to 2.
        ConcurrencyStats {
            min: a.min(b) as u32,
            max: a.max(b).max(mean.round() as u64).max(1) as u32,
            mean,
        }
    }
}

impl Drop for PowerSpan {
    fn drop(&mut self) {
        if !self.finished {
            self.finished = true;
            self.shared.dec_inflight();
        }
    }
}

fn delta_uj_to_j(start: Option<u64>, end: Option<u64>) -> Option<f64> {
    match (start, end) {
        (Some(s), Some(e)) => Some(e.saturating_sub(s) as f64 / 1_000_000.0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{
        EnergyCounters, PowerSample, PowerSource, SourceDescriptor, SourceError,
    };
    use std::sync::Mutex;

    struct StepSource {
        seq: Mutex<std::collections::VecDeque<PowerSample>>,
        last: PowerSample,
        descriptor: SourceDescriptor,
        quality: SampleQuality,
    }

    impl StepSource {
        fn new(seq: Vec<PowerSample>, quality: SampleQuality) -> Arc<Self> {
            let last = *seq.last().unwrap();
            Arc::new(Self {
                seq: Mutex::new(seq.into()),
                last,
                descriptor: SourceDescriptor {
                    backend: "step",
                    rails: vec!["sys_total".to_string(), "pkg".to_string()],
                    energy_counter_hz: 100,
                    quality,
                },
                quality,
            })
        }
    }

    impl PowerSource for StepSource {
        fn sample(&self) -> Result<PowerSample, SourceError> {
            Ok(self.seq.lock().unwrap().pop_front().unwrap_or(self.last))
        }
        fn descriptor(&self) -> &SourceDescriptor {
            &self.descriptor
        }
        fn quality(&self) -> SampleQuality {
            self.quality
        }
    }

    fn sample(w: f64, pkg_uj: u64) -> PowerSample {
        PowerSample {
            node_watts: Some(w),
            energy: EnergyCounters {
                pkg_uj: Some(pkg_uj),
                cpu_uj: None,
                gpu_uj: None,
            },
            quality: SampleQuality::Measured,
        }
    }

    #[test]
    fn unavailable_source_yields_no_report() {
        let src = StepSource::new(vec![PowerSample::unavailable()], SampleQuality::Unavailable);
        let m = PowerMonitor::manual(src);
        let span = m.begin_span();
        assert!(span.finish(None, Policy::WorkWeighted, None).is_none());
    }

    #[test]
    fn solo_span_reports_measured_energy_and_marginal() {
        // Calibrate idle at 30 W with no request in flight (pkg counter flat
        // at 1 MJ during idle), then one hot 130 W tick with pkg +50 J.
        let src = StepSource::new(
            vec![
                sample(30.0, 1_000_000),
                sample(30.0, 1_000_000),
                sample(130.0, 51_000_000),
            ],
            SampleQuality::Measured,
        );
        let m = PowerMonitor::manual(src);
        // Two idle ticks → idle ≈ 30 W.
        let p = m.drive(0.0, None);
        m.drive(1.0, p);
        assert!((m.p_idle_watts() - 30.0).abs() < 2.0);

        // Now a request: window integrates 130 W for 1 s, pkg counter +50 J.
        let span = m.begin_span();
        m.drive(1.0, Some(130.0));
        // Inject duration = 1.0 s to match the simulated integration tick.
        let r = span
            .finish_inner(1.0, None, Policy::WorkWeighted, Some("gx10".to_string()))
            .unwrap();

        // Alone the whole window → Exclusive/High.
        assert_eq!(r.attribution, "exclusive");
        assert_eq!(r.confidence, "high");
        // gross ≈ 130 J over the 1 s tick; baseline ≈ 30 J; marginal ≈ 100 J.
        assert!((r.joules_gross - 130.0).abs() < 5.0, "gross {}", r.joules_gross);
        assert!((r.joules_marginal - 100.0).abs() < 6.0, "marginal {}", r.joules_marginal);
        // Exact hardware pkg energy delta = 50 J.
        assert!((r.measured.pkg_joules.unwrap() - 50.0).abs() < 1e-6);
        assert!(r.estimated);
        assert_eq!(r.node_id.as_deref(), Some("gx10"));
    }

    #[test]
    fn concurrent_window_apportions_by_work_share() {
        let src = StepSource::new(vec![sample(200.0, 0)], SampleQuality::Measured);
        let m = PowerMonitor::manual(src);
        // Prime idle low so most of the window's energy is marginal.
        let p = m.drive(0.0, None);
        m.drive(1.0, p);

        // Three requests overlap the window.
        let sh = m.shared();
        sh.inc_inflight();
        sh.inc_inflight(); // two OTHER requests already running
        let span = m.begin_span(); // this one → inflight now 3
        m.drive(1.0, Some(200.0));
        let r = span
            .finish_inner(1.0, Some(0.5), Policy::WorkWeighted, None)
            .unwrap();
        sh.dec_inflight();
        sh.dec_inflight();

        assert_eq!(r.attribution, "work_weighted");
        assert_eq!(r.confidence, "medium");
        assert_eq!(r.concurrent_requests.max, 3);
        // Not alone → not exclusive.
        assert!(r.joules_marginal < r.joules_gross);
    }

    #[test]
    fn drop_balances_inflight() {
        let src = StepSource::new(vec![sample(50.0, 0)], SampleQuality::Measured);
        let m = PowerMonitor::manual(src);
        {
            let _span = m.begin_span();
            assert_eq!(m.shared().snapshot().inflight, 1);
        }
        assert_eq!(m.shared().snapshot().inflight, 0);
    }
}
