// SPDX-License-Identifier: AGPL-3.0-only

//! Attribution: turning a measured energy window into an honest,
//! apportioned per-request [`PowerReport`].
//!
//! ## The marginal / gross / baseline split
//!
//! Over a request's wall-clock window the node spends `J_gross` joules
//! (node-wide `sys_total`, integrated). Some of that it would have burned
//! idle: `J_baseline = P_idle × duration`. The rest is request-caused:
//! `J_marginal_total = J_gross − J_baseline`. Marginal answers *"what did
//! this request cost"*; gross answers *"what did the node spend"* —
//! conflating them is the standard failure mode, so the report carries both.
//!
//! ## Apportionment across concurrency
//!
//! When N requests overlap, the node-wide marginal must be split. The
//! [`Policy`] decides how, and the [`Confidence`] it earns reflects how much
//! authority the number deserves:
//! - [`Policy::Exclusive`] — the window was owned by one request the whole
//!   time (`max concurrency == 1`). Ground truth → `High`.
//! - [`Policy::WorkWeighted`] — split by a work unit (tokens / GPU-seconds).
//!   The intended default → `Medium`.
//! - [`Policy::EqualSlice`] — split evenly by concurrent count. Cheap and
//!   defensible but wrong under heterogeneous load → `Low`.
//!
//! Exact hardware-counter energies (`pkg`/`cpu`/`gpu`, which need no
//! sampling) ride along in [`MeasuredEnergy`] as evidence and feed the
//! continuous invariant check.

use serde::Serialize;

/// How the node-wide marginal energy was apportioned to this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Only valid when the request owned the node for the whole window.
    Exclusive,
    /// Marginal divided evenly by the concurrent request count.
    EqualSlice,
    /// Apportioned by a work unit (tokens / GPU-seconds / FLOPs proxy).
    WorkWeighted,
}

impl Policy {
    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Exclusive => "exclusive",
            Policy::EqualSlice => "equal_slice",
            Policy::WorkWeighted => "work_weighted",
        }
    }
}

/// How much authority the apportioned number deserves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    High,
    Medium,
    Low,
}

impl Confidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Confidence::High => "high",
            Confidence::Medium => "medium",
            Confidence::Low => "low",
        }
    }
}

/// In-flight request count observed across the window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ConcurrencyStats {
    pub min: u32,
    pub max: u32,
    pub mean: f64,
}

/// Exact energy from the hardware accumulators over the window (joules).
/// Each is `None` when its rail is absent. Unlike the apportioned marginal,
/// these are direct counter deltas — no sampling error — so they are the
/// reference the invariant check measures against.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Default)]
pub struct MeasuredEnergy {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pkg_joules: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_joules: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_joules: Option<f64>,
}

impl MeasuredEnergy {
    fn is_empty(&self) -> bool {
        self.pkg_joules.is_none() && self.cpu_joules.is_none() && self.gpu_joules.is_none()
    }
}

/// Description of the measurement source, serialized into the report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourceInfo {
    pub backend: String,
    pub rails: Vec<String>,
    pub quality: String,
    pub energy_counter_hz: u32,
    pub samples_in_window: u64,
}

/// The opt-in per-request power metadata attached to a response.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PowerReport {
    /// Request-caused energy apportioned to this request (joules).
    pub joules_marginal: f64,
    /// Total node energy spent during the window (joules, node-wide).
    pub joules_gross: f64,
    /// Energy the node would have burned idle over the window (joules).
    pub joules_baseline: f64,
    /// Average marginal power over the window (watts).
    pub avg_watts_marginal: f64,
    pub duration_ms: f64,
    /// Apportionment policy that produced `joules_marginal`.
    pub attribution: String,
    pub confidence: String,
    pub concurrent_requests: ConcurrencyStats,
    #[serde(skip_serializing_if = "MeasuredEnergy::is_empty")]
    pub measured: MeasuredEnergy,
    pub source: SourceInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    /// Always `true`. Apportioned sensor energy is an estimate regardless of
    /// sensor quality.
    pub estimated: bool,
}

/// Inputs to [`attribute`].
#[derive(Debug, Clone)]
pub struct AttributionInput {
    /// Window duration in seconds.
    pub duration_s: f64,
    /// Node-wide energy spent during the window (joules).
    pub gross_joules: f64,
    /// Estimated node idle power (watts).
    pub p_idle_watts: f64,
    /// Concurrency observed across the window (includes this request).
    pub concurrency: ConcurrencyStats,
    /// Requested apportionment policy. Auto-upgraded to [`Policy::Exclusive`]
    /// when the window was actually owned alone (`concurrency.max <= 1`).
    pub requested_policy: Policy,
    /// This request's work share in `0.0..=1.0` for [`Policy::WorkWeighted`]
    /// (e.g. its tokens / total tokens over the window). `None` falls back to
    /// an equal slice.
    pub work_share: Option<f64>,
    /// Exact hardware-counter energies over the window (joules).
    pub measured: MeasuredEnergy,
    /// Measurement source description.
    pub source: SourceInfo,
    /// Serving node id, if known.
    pub node_id: Option<String>,
}

/// Compute the honest, apportioned per-request power report. Pure.
pub fn attribute(input: AttributionInput) -> PowerReport {
    let dur = input.duration_s.max(0.0);
    let gross = input.gross_joules.max(0.0);
    let baseline = (input.p_idle_watts.max(0.0) * dur).min(gross);
    let marginal_total = (gross - baseline).max(0.0);

    // Ground truth beats a requested guess: if the request owned the node
    // for the whole window, it is Exclusive regardless of what was asked.
    let policy = if input.concurrency.max <= 1 {
        Policy::Exclusive
    } else {
        input.requested_policy
    };

    let (share, policy) = match policy {
        Policy::Exclusive => (1.0, Policy::Exclusive),
        Policy::WorkWeighted => match input.work_share {
            Some(w) => (w.clamp(0.0, 1.0), Policy::WorkWeighted),
            // No work signal — degrade to an equal slice rather than lie.
            None => (equal_share(input.concurrency.mean), Policy::EqualSlice),
        },
        Policy::EqualSlice => (equal_share(input.concurrency.mean), Policy::EqualSlice),
    };

    let confidence = match policy {
        Policy::Exclusive => Confidence::High,
        Policy::WorkWeighted => Confidence::Medium,
        Policy::EqualSlice => Confidence::Low,
    };

    let joules_marginal = marginal_total * share;
    let avg_watts_marginal = if dur > 0.0 { joules_marginal / dur } else { 0.0 };

    PowerReport {
        joules_marginal,
        joules_gross: gross,
        joules_baseline: baseline,
        avg_watts_marginal,
        duration_ms: dur * 1000.0,
        attribution: policy.as_str().to_string(),
        confidence: confidence.as_str().to_string(),
        concurrent_requests: input.concurrency,
        measured: input.measured,
        source: input.source,
        node_id: input.node_id,
        estimated: true,
    }
}

/// Equal-slice share = 1 / mean concurrency (clamped so a sub-1 mean, which
/// can happen from sampling, never inflates the share above 1).
fn equal_share(mean_concurrency: f64) -> f64 {
    (1.0 / mean_concurrency.max(1.0)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src() -> SourceInfo {
        SourceInfo {
            backend: "gb10_spbm".to_string(),
            rails: vec!["sys_total".to_string(), "pkg".to_string()],
            quality: "measured".to_string(),
            energy_counter_hz: 100,
            samples_in_window: 500,
        }
    }

    fn base_input() -> AttributionInput {
        AttributionInput {
            duration_s: 10.0,
            gross_joules: 1000.0,
            p_idle_watts: 30.0, // 300 J baseline over 10s
            concurrency: ConcurrencyStats { min: 1, max: 1, mean: 1.0 },
            requested_policy: Policy::WorkWeighted,
            work_share: Some(0.5),
            measured: MeasuredEnergy::default(),
            source: src(),
            node_id: Some("gx10-9959".to_string()),
        }
    }

    #[test]
    fn solo_window_is_exclusive_high_confidence() {
        let r = attribute(base_input());
        // Alone the whole window → Exclusive regardless of requested policy.
        assert_eq!(r.attribution, "exclusive");
        assert_eq!(r.confidence, "high");
        assert_eq!(r.joules_gross, 1000.0);
        assert_eq!(r.joules_baseline, 300.0);
        // Exclusive → full marginal, no apportionment.
        assert_eq!(r.joules_marginal, 700.0);
        assert_eq!(r.avg_watts_marginal, 70.0);
        assert!(r.estimated);
    }

    #[test]
    fn work_weighted_splits_concurrent_marginal() {
        let mut i = base_input();
        i.concurrency = ConcurrencyStats { min: 2, max: 4, mean: 3.0 };
        i.work_share = Some(0.25);
        let r = attribute(i);
        assert_eq!(r.attribution, "work_weighted");
        assert_eq!(r.confidence, "medium");
        // marginal_total = 1000 - 300 = 700; this request's share 0.25.
        assert_eq!(r.joules_marginal, 175.0);
    }

    #[test]
    fn equal_slice_uses_mean_concurrency() {
        let mut i = base_input();
        i.concurrency = ConcurrencyStats { min: 2, max: 6, mean: 4.0 };
        i.requested_policy = Policy::EqualSlice;
        let r = attribute(i);
        assert_eq!(r.attribution, "equal_slice");
        assert_eq!(r.confidence, "low");
        // 700 marginal / mean 4 = 175.
        assert_eq!(r.joules_marginal, 175.0);
    }

    #[test]
    fn work_weighted_without_signal_degrades_to_equal_slice() {
        let mut i = base_input();
        i.concurrency = ConcurrencyStats { min: 2, max: 4, mean: 2.0 };
        i.work_share = None;
        let r = attribute(i);
        // No work signal → honest downgrade, not a fabricated full share.
        assert_eq!(r.attribution, "equal_slice");
        assert_eq!(r.confidence, "low");
        assert_eq!(r.joules_marginal, 350.0); // 700 / 2
    }

    #[test]
    fn baseline_never_exceeds_gross() {
        let mut i = base_input();
        i.p_idle_watts = 1000.0; // absurd idle → baseline would be 10000 J
        let r = attribute(i);
        assert_eq!(r.joules_baseline, 1000.0); // clamped to gross
        assert_eq!(r.joules_marginal, 0.0); // nothing marginal left
    }

    #[test]
    fn measured_energy_serializes_when_present_and_omits_when_empty() {
        let mut i = base_input();
        i.measured = MeasuredEnergy {
            pkg_joules: Some(420.0),
            cpu_joules: None,
            gpu_joules: Some(310.0),
        };
        let r = attribute(i);
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["measured"]["pkg_joules"], 420.0);
        assert_eq!(v["measured"]["gpu_joules"], 310.0);
        assert!(v["measured"].get("cpu_joules").is_none());
        assert_eq!(v["estimated"], true);
        assert_eq!(v["source"]["backend"], "gb10_spbm");

        // Empty measured → the whole object is omitted.
        let r2 = attribute(base_input());
        let v2 = serde_json::to_value(&r2).unwrap();
        assert!(v2.get("measured").is_none());
    }
}
