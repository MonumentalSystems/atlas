// SPDX-License-Identifier: AGPL-3.0-only

//! POC-0: env-gated scheduler gap-timing instrumentation.
//!
//! DIAG (ATLAS_GAP_TIMING=1): decompose the per-tick scheduler wall so we can
//! attribute the C=8 TTFT/throughput gap to prefill-stall vs decode. Mirrors
//! the `decode_logits_step::decode_timing_record` idiom: a `OnceLock<bool>`
//! gate + `AtomicU64` rolling counters, emitting one summary every N decode
//! ticks. Zero-cost and behavior-identical when the env var is unset (every
//! public fn early-returns after a single OnceLock load).
//!
//! Counters are CUMULATIVE (not reset between reports). This is simpler than
//! the decode_logits_step swap-to-zero idiom and is fine for a bench: each
//! report shows the running means/ratios over the whole run, which is exactly
//! what we want when attributing a steady-state C=8 gap.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

static ENABLED: OnceLock<bool> = OnceLock::new();

/// True when ATLAS_GAP_TIMING=1. Single OnceLock load → zero cost when off.
#[inline]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("ATLAS_GAP_TIMING").as_deref() == Ok("1"))
}

static QUEUE_WAIT_US: AtomicU64 = AtomicU64::new(0);
static PREFILL_COMPUTE_US: AtomicU64 = AtomicU64::new(0);
static PREFILL_TICKS: AtomicU64 = AtomicU64::new(0);
static DECODE_COMPUTE_US: AtomicU64 = AtomicU64::new(0);
static DECODE_TICKS: AtomicU64 = AtomicU64::new(0);
static DECODE_BATCH_SUM: AtomicU64 = AtomicU64::new(0);
static DECODE_TOKENS: AtomicU64 = AtomicU64::new(0);
static STALL_TICKS: AtomicU64 = AtomicU64::new(0);
static REPORT_CNT: AtomicU64 = AtomicU64::new(0);

/// Record one decode tick: `n` = batch size (sequences decoded this tick),
/// `us` = wall microseconds for the decode (forward + logits processing).
pub fn record_decode(n: usize, us: u64) {
    if !enabled() {
        return;
    }
    DECODE_TICKS.fetch_add(1, Ordering::Relaxed);
    DECODE_BATCH_SUM.fetch_add(n as u64, Ordering::Relaxed);
    DECODE_TOKENS.fetch_add(n as u64, Ordering::Relaxed);
    DECODE_COMPUTE_US.fetch_add(us, Ordering::Relaxed);
    maybe_report();
}

/// Record one prefill compute contribution: `us` = wall microseconds spent in
/// the prefill forward (chunk or batched).
pub fn record_prefill(us: u64) {
    if !enabled() {
        return;
    }
    PREFILL_TICKS.fetch_add(1, Ordering::Relaxed);
    PREFILL_COMPUTE_US.fetch_add(us, Ordering::Relaxed);
}

/// Record queue-wait: `us` = microseconds a request sat in the pending queue
/// between enqueue (receiver-thread push) and dequeue (drain_pending_requests).
pub fn record_queue_wait(us: u64) {
    if !enabled() {
        return;
    }
    QUEUE_WAIT_US.fetch_add(us, Ordering::Relaxed);
}

/// Note a tick where a prefill was in progress WHILE active decode lanes
/// existed — the prefill-stall signal (decode waiting on prefill).
pub fn note_stall_tick() {
    if !enabled() {
        return;
    }
    STALL_TICKS.fetch_add(1, Ordering::Relaxed);
}

/// Emit one rolling-summary line every 200 decode ticks. Cumulative counters.
fn maybe_report() {
    let n = REPORT_CNT.fetch_add(1, Ordering::Relaxed) + 1;
    if !n.is_multiple_of(200) {
        return;
    }
    let decode_ticks = DECODE_TICKS.load(Ordering::Relaxed);
    let prefill_ticks = PREFILL_TICKS.load(Ordering::Relaxed);
    let batch_sum = DECODE_BATCH_SUM.load(Ordering::Relaxed);
    let decode_tokens = DECODE_TOKENS.load(Ordering::Relaxed);
    let decode_us = DECODE_COMPUTE_US.load(Ordering::Relaxed);
    let prefill_us = PREFILL_COMPUTE_US.load(Ordering::Relaxed);
    let queue_us = QUEUE_WAIT_US.load(Ordering::Relaxed);
    let stall_ticks = STALL_TICKS.load(Ordering::Relaxed);

    let mean_batch = if decode_ticks > 0 {
        batch_sum as f64 / decode_ticks as f64
    } else {
        0.0
    };
    // decode-only token/s: tokens generated per second of pure decode compute.
    let decode_only_tok_s = if decode_us > 0 {
        decode_tokens as f64 * 1e6 / decode_us as f64
    } else {
        0.0
    };
    let total_ticks = prefill_ticks + decode_ticks;
    let stall_ratio = if total_ticks > 0 {
        stall_ticks as f64 / total_ticks as f64
    } else {
        0.0
    };

    tracing::info!(
        "GAP: mean_batch={:.2} decode_only_tok_s={:.1} prefill_compute_ms={:.1} \
         queue_wait_ms={:.1} stall_ratio={:.3} (decode_ticks={} prefill_ticks={})",
        mean_batch,
        decode_only_tok_s,
        prefill_us as f64 / 1000.0,
        queue_us as f64 / 1000.0,
        stall_ratio,
        decode_ticks,
        prefill_ticks,
    );
}
