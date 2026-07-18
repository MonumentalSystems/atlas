// SPDX-License-Identifier: AGPL-3.0-only

//! Integration glue between the server and the `atlas-power` crate.
//!
//! Gated behind the `power_attribution` Cargo feature — non-feature builds
//! do not compile this module and carry literally zero cost. When the
//! feature is on, energy metadata is still triple-gated at runtime:
//!   1. config — `ATLAS_POWER_ATTRIBUTION` must be truthy at startup;
//!   2. node capability — a GB10 `spbm` source that reports `Measured`
//!      (otherwise [`PowerHandle::init`] returns `None` and no field is ever
//!      emitted — never zero-filled);
//!   3. per-request — the caller must send `X-Atlas-Power: 1`.
//!
//! Only when all three hold does a request do any work beyond the sampler's
//! background thread.

use std::sync::Arc;

use atlas_power::{Policy, PowerMonitor, PowerSource, PowerSpan, SampleQuality, SpbmSource};
use lazy_static::lazy_static;
use prometheus::{Counter, register_counter};

use crate::api::chat::ChatOutcome;

lazy_static! {
    /// Sum of attributed marginal joules across measured requests. Mirrors
    /// the fleet's `atlas.power.joules.attributed` semantics for Hyades
    /// reconciliation (plain SI joules).
    static ref POWER_JOULES_ATTRIBUTED: Counter = register_counter!(
        "atlas_power_joules_attributed_total",
        "Sum of attributed marginal joules over measured request windows"
    )
    .unwrap();
    /// Sum of gross node-wide joules over measured request windows.
    static ref POWER_JOULES_GROSS: Counter = register_counter!(
        "atlas_power_joules_gross_total",
        "Sum of gross node-wide joules over measured request windows"
    )
    .unwrap();
}

/// Server-side handle to the power monitor + per-node attribution config.
pub struct PowerHandle {
    monitor: Arc<PowerMonitor>,
    node_id: Option<String>,
    default_policy: Policy,
}

impl PowerHandle {
    /// Build the handle if power attribution is enabled and this node can
    /// actually measure power. Returns `None` (feature inert) otherwise —
    /// callers then omit power metadata entirely rather than fabricate it.
    pub fn init() -> Option<Arc<Self>> {
        if !env_truthy("ATLAS_POWER_ATTRIBUTION") {
            return None;
        }
        let Some(source) = SpbmSource::discover() else {
            tracing::warn!(
                "power_attribution requested but no spbm hwmon device found \
                 (not a GB10) — power metadata disabled"
            );
            return None;
        };
        if source.quality() != SampleQuality::Measured {
            tracing::warn!("power_attribution: spbm source not measured — disabled");
            return None;
        }
        let rails = source.descriptor().rails.clone();
        let hz = source.descriptor().energy_counter_hz.max(100);
        let monitor = Arc::new(PowerMonitor::spawn(Arc::new(source), hz));
        let node_id = hostname();
        let default_policy = parse_policy(std::env::var("ATLAS_POWER_POLICY").ok().as_deref());
        tracing::info!(
            "power_attribution ENABLED: backend=gb10_spbm rails={rails:?} node={node_id:?} \
             default_policy={} sampler={hz}Hz",
            default_policy.as_str()
        );
        Some(Arc::new(Self {
            monitor,
            node_id,
            default_policy,
        }))
    }

    /// Begin a per-request measurement window.
    pub fn begin(&self) -> PowerSpan {
        self.monitor.begin_span()
    }

    /// Close the window, update the Prometheus counters, and return the
    /// report as a JSON value ready to drop into the response. `None` when
    /// the source is not measured (defensive; `init` already guarantees it).
    fn finish_report(&self, span: &PowerSpan) -> Option<serde_json::Value> {
        let report = span.finish(None, self.default_policy, self.node_id.clone())?;
        POWER_JOULES_ATTRIBUTED.inc_by(report.joules_marginal);
        POWER_JOULES_GROSS.inc_by(report.joules_gross);
        serde_json::to_value(&report).ok()
    }
}

/// Whether a request opted into power metadata via `X-Atlas-Power: 1`
/// (also accepts `true`, case-insensitive).
pub fn header_requests_power(headers: &axum::http::HeaderMap) -> bool {
    headers
        .get("x-atlas-power")
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            let s = s.trim();
            s == "1" || s.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

/// Attach an opt-in power report to a completed blocking response.
///
/// A no-op for streaming / HTTP outcomes (the span is simply dropped, which
/// balances the in-flight count). Streaming power attribution is a
/// documented follow-up.
pub fn attach_power(
    mut outcome: ChatOutcome,
    span: Option<(Arc<PowerHandle>, PowerSpan)>,
) -> ChatOutcome {
    if let Some((handle, span)) = span
        && let ChatOutcome::Blocking(ir) = &mut outcome
        && let Some(v) = handle.finish_report(&span)
    {
        ir.power = Some(v);
    }
    outcome
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

fn parse_policy(s: Option<&str>) -> Policy {
    match s.map(|v| v.trim().to_lowercase()).as_deref() {
        Some("exclusive") => Policy::Exclusive,
        Some("work_weighted") | Some("work-weighted") => Policy::WorkWeighted,
        // Default: EqualSlice is honest under concurrency (Low confidence)
        // and auto-upgrades to Exclusive when a request owns the node alone.
        _ => Policy::EqualSlice,
    }
}

fn hostname() -> Option<String> {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_gate_accepts_1_and_true_only() {
        let mut h = axum::http::HeaderMap::new();
        assert!(!header_requests_power(&h));
        h.insert("x-atlas-power", "1".parse().unwrap());
        assert!(header_requests_power(&h));
        h.insert("x-atlas-power", "TRUE".parse().unwrap());
        assert!(header_requests_power(&h));
        h.insert("x-atlas-power", "0".parse().unwrap());
        assert!(!header_requests_power(&h));
        h.insert("x-atlas-power", "yes".parse().unwrap());
        assert!(!header_requests_power(&h)); // only 1/true gate the field
    }

    #[test]
    fn policy_parsing_defaults_to_equal_slice() {
        assert_eq!(parse_policy(Some("exclusive")).as_str(), "exclusive");
        assert_eq!(parse_policy(Some("work_weighted")).as_str(), "work_weighted");
        assert_eq!(parse_policy(Some("work-weighted")).as_str(), "work_weighted");
        assert_eq!(parse_policy(Some("EQUAL_SLICE")).as_str(), "equal_slice");
        assert_eq!(parse_policy(None).as_str(), "equal_slice");
        assert_eq!(parse_policy(Some("garbage")).as_str(), "equal_slice");
    }
}
