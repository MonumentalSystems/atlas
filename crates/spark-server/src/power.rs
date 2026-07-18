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

use std::sync::atomic::Ordering;

use atlas_power::{Policy, PowerMonitor, PowerSource, PowerSpan, SampleQuality, SpbmSource};
use futures::StreamExt;
use lazy_static::lazy_static;
use prometheus::{Counter, Gauge, register_counter, register_gauge};

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
    /// Continuous conservation self-check: Σ(attributed marginal) ÷ measured
    /// marginal energy spent while busy. Should hover near 1.0; sustained
    /// drift means the attribution engine has silently broken.
    static ref POWER_INVARIANT_RATIO: Gauge = register_gauge!(
        "atlas_power_invariant_ratio",
        "Ratio of attributed marginal joules to measured marginal joules spent"
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

    /// Begin a per-request measurement window. Returns the span and the
    /// generated-token counter value at entry (the denominator baseline for
    /// the work-weighted share).
    pub fn begin(&self) -> (PowerSpan, u64) {
        (self.monitor.begin_span(), tokens_now())
    }

    /// Close the window, update the Prometheus counters, and return the
    /// report as a JSON value ready to drop into the response. `None` when
    /// the source is not measured (defensive; `init` already guarantees it).
    ///
    /// `work_share` is this request's fraction of tokens generated across all
    /// requests during its window — `Some` selects [`Policy::WorkWeighted`]
    /// (auto-upgraded to Exclusive when the request owned the node alone);
    /// `None` falls back to the configured default policy.
    fn finish_report(&self, span: &PowerSpan, work_share: Option<f64>) -> Option<serde_json::Value> {
        let policy = if work_share.is_some() {
            Policy::WorkWeighted
        } else {
            self.default_policy
        };
        let report = span.finish(work_share, policy, self.node_id.clone())?;
        POWER_JOULES_ATTRIBUTED.inc_by(report.joules_marginal);
        POWER_JOULES_GROSS.inc_by(report.joules_gross);
        POWER_INVARIANT_RATIO.set(self.monitor.invariant().2);
        serde_json::to_value(&report).ok()
    }
}

/// Finish a legacy `/v1/completions` (blocking) span and stamp the report
/// onto the response. Work share comes from the response's completion_tokens.
pub fn attach_completion_power(
    resp: &mut crate::openai::CompletionResponse,
    handle: &PowerHandle,
    span: PowerSpan,
    start_tokens: u64,
) {
    let share = work_share(start_tokens, resp.usage.completion_tokens);
    resp.power = handle.finish_report(&span, share);
}

/// Finish a legacy `/v1/completions` STREAMING span (called from the terminal
/// `Done` event) and return the report JSON for the terminal usage chunk.
pub fn finish_completion_stream(
    handle: &PowerHandle,
    span: PowerSpan,
    start_tokens: u64,
    completion_tokens: usize,
) -> Option<serde_json::Value> {
    let share = work_share(start_tokens, completion_tokens);
    handle.finish_report(&span, share)
}

/// Current monotonic generated-token count (0 when the counter is untouched).
fn tokens_now() -> u64 {
    crate::metrics::GENERATED_TOKENS.load(Ordering::Relaxed)
}

/// This request's work share = its tokens ÷ all tokens generated during its
/// window. `None` when nothing was counted (avoids a divide-by-zero and lets
/// attribution fall back honestly).
fn work_share(start_tokens: u64, request_tokens: usize) -> Option<f64> {
    let window = tokens_now().saturating_sub(start_tokens);
    if window == 0 || request_tokens == 0 {
        return None;
    }
    Some((request_tokens as f64 / window as f64).clamp(0.0, 1.0))
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

/// Attach an opt-in power report to a completed response.
///
/// - **Blocking**: finish the span now and stash the report on the IR.
/// - **Streaming**: wrap the delta stream so the span is finished as the
///   terminal `Finish` delta flows out, and the report rides the final usage
///   chunk. If the stream is dropped early (client disconnect, error) the
///   span drops with the wrapper and the in-flight count still balances.
/// - **HTTP**: the span is simply dropped.
pub fn attach_power(
    outcome: ChatOutcome,
    span: Option<(Arc<PowerHandle>, PowerSpan, u64)>,
) -> ChatOutcome {
    let Some((handle, span, start_tokens)) = span else {
        return outcome;
    };
    match outcome {
        ChatOutcome::Blocking(mut ir) => {
            let share = work_share(start_tokens, ir.usage.completion_tokens);
            if let Some(v) = handle.finish_report(&span, share) {
                ir.power = Some(v);
            }
            ChatOutcome::Blocking(ir)
        }
        ChatOutcome::Streaming(deltas) => {
            let mut span = Some(span);
            let wrapped = deltas.map(move |delta| match delta {
                crate::ir::StreamDelta::Finish {
                    reason,
                    usage,
                    token_ids,
                    power: _,
                } => {
                    let power = span.take().and_then(|sp| {
                        let share = work_share(start_tokens, usage.completion_tokens);
                        handle.finish_report(&sp, share)
                    });
                    crate::ir::StreamDelta::Finish {
                        reason,
                        usage,
                        token_ids,
                        power,
                    }
                }
                other => other,
            });
            ChatOutcome::Streaming(Box::pin(wrapped))
        }
        // HTTP (error/fast-fail) — span drops here, balancing in-flight.
        other => other,
    }
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
    fn work_share_math() {
        // request generated 40 of 100 tokens in its window → 0.4 share.
        crate::metrics::GENERATED_TOKENS.store(100, Ordering::Relaxed);
        assert_eq!(work_share(0, 40), Some(0.4));
        // no tokens counted in the window → no share (honest fallback).
        crate::metrics::GENERATED_TOKENS.store(60, Ordering::Relaxed);
        assert_eq!(work_share(60, 40), None);
        // request with zero output tokens → no share.
        crate::metrics::GENERATED_TOKENS.store(100, Ordering::Relaxed);
        assert_eq!(work_share(60, 0), None);
        // share is clamped to 1.0 even if bookkeeping undercounts the window.
        crate::metrics::GENERATED_TOKENS.store(70, Ordering::Relaxed);
        assert_eq!(work_share(60, 40), Some(1.0));
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
