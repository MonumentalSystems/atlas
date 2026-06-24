// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduling policy trait (SDD: FIFO vs SLAI).
//!
//! Controls two decisions in the scheduler loop:
//! 1. Whether to accept new prefills or prioritize decode (TBT deadline).
//! 2. Which pending requests to prefill and in what order.
//!
//! Implementations:
//! - [`FifoPolicy`]: always prefill, take first N from queue (current behavior).
//! - [`SlaiPolicy`]: skip prefills when active sequences approach TBT deadline,
//!   select shortest prompts first from ALL pending (SLAI — arXiv:2407.08353).

use std::time::{Duration, Instant};

/// Metadata about a pending request for selection decisions.
pub struct PendingRequestInfo {
    /// Number of prompt tokens (determines prefill cost).
    pub prompt_len: usize,
    /// Index into the full pending requests vec.
    pub index: usize,
}

/// Per-sequence timing for decode urgency decisions.
pub struct ActiveSeqTiming {
    /// When the last token was emitted for this sequence.
    pub last_token_time: Instant,
}

/// Scheduling policy controlling prefill admission and ordering.
pub trait SchedulingPolicy: Send {
    /// Whether to accept new prefills this iteration.
    ///
    /// Returns `false` to skip prefill and proceed directly to decode
    /// (e.g., when active sequences approach their TBT deadline).
    fn should_prefill(&self, active_timings: &[ActiveSeqTiming]) -> bool;

    /// Select up to `capacity` requests from ALL pending, in prefill order.
    ///
    /// Returns indices into `requests` for the selected items, ordered
    /// by desired prefill execution order. FIFO takes the first N;
    /// SLAI picks the N shortest prompts.
    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize>;

    /// Number of prefill tokens to inject this iteration when fusing a
    /// prefill chunk into a decode step ("always-mixed" path).
    ///
    /// Returns a token budget in `[0, full_chunk]`:
    /// - `full_chunk` when no decode is active (no TBT pressure) or the
    ///   policy has no opinion (FIFO / default).
    /// - A clamped, WY4-aligned slice (≥ `MIN_PREFILL_SLICE`) under
    ///   moderate decode pressure so the fused step stays under the TBT
    ///   target.
    /// - `0` ONLY as a hard suppress when a decode has already blown its
    ///   TBT deadline — the caller must then run decode-only this tick.
    ///
    /// This is driven by MEASURED step cost (seeded from conservative
    /// constants), NOT by `now − last_token_time` (which the fused step
    /// self-resets every iteration — see spec blocker B2).
    fn prefill_slice_budget(&self, active_timings: &[ActiveSeqTiming], full_chunk: usize) -> usize {
        // Default (FIFO / unaware): inject the full chunk — same as today.
        let _ = active_timings;
        full_chunk
    }

    /// Policy name for logging.
    fn name(&self) -> &str;
}

/// Minimum prefill slice (tokens) injected into a fused mixed step.
/// WY4-aligned; large enough to make forward progress on a long prompt,
/// small enough to keep the fused step under the TBT target.
pub const MIN_PREFILL_SLICE: usize = 32;

/// FIFO scheduling: always prefill, take first N from queue.
pub struct FifoPolicy;

impl SchedulingPolicy for FifoPolicy {
    fn should_prefill(&self, _active_timings: &[ActiveSeqTiming]) -> bool {
        true
    }

    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        // First N in queue order (FIFO).
        (0..requests.len().min(capacity)).collect()
    }

    fn name(&self) -> &str {
        "fifo"
    }
}

/// SLO-aware scheduling (SLAI-inspired).
///
/// - Skips prefills when any active sequence waited > 80% of `tbt_deadline`
///   since its last token emission (decode-first priority).
/// - Selects the N shortest prompts from ALL pending (reduces median TTFT).
pub struct SlaiPolicy {
    tbt_deadline: Duration,
    /// Minimum fused prefill slice (tokens), WY4-aligned, read once from
    /// `ATLAS_SLAI_MIN_PREFILL_SLICE` (default [`MIN_PREFILL_SLICE`]).
    min_prefill_slice: usize,
    /// Conservative seed for the decode-region step cost (ms) used to size
    /// the prefill slice. TODO(holo-mixed Step 3): replace with an EWMA
    /// updated from measured fused-step durations (a `record_step` hook
    /// wired from the scheduler). Seeded high so the first slices are
    /// conservative (small) until measurement lands.
    t_dec_ms_seed: f64,
    /// Conservative seed for marginal prefill cost (ms / token). TODO as
    /// above — currently a constant. Tuned to GB10 Holo prefill (~0.6 ms
    /// per WY4-aligned token at the resting chunk cap).
    cost_per_tok_ms_seed: f64,
}

impl SlaiPolicy {
    pub fn new(tbt_deadline_ms: u64) -> Self {
        // Read the minimum slice once; clamp ≥4 and WY4-align.
        let min_prefill_slice = std::env::var("ATLAS_SLAI_MIN_PREFILL_SLICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| {
                let v = v.max(4);
                (v / 4) * 4
            })
            .unwrap_or(MIN_PREFILL_SLICE);
        Self {
            tbt_deadline: Duration::from_millis(tbt_deadline_ms),
            min_prefill_slice,
            // Conservative constants (see field docs). Order of magnitude:
            // a small decode batch costs a few ms/step; prefill marginal
            // cost is sub-ms/token at the resting cap.
            t_dec_ms_seed: 5.0,
            cost_per_tok_ms_seed: 0.6,
        }
    }
}

impl SchedulingPolicy for SlaiPolicy {
    fn should_prefill(&self, active_timings: &[ActiveSeqTiming]) -> bool {
        if active_timings.is_empty() {
            return true;
        }
        let now = Instant::now();
        let margin = self.tbt_deadline.mul_f64(0.8);
        for timing in active_timings {
            if now.duration_since(timing.last_token_time) >= margin {
                return false;
            }
        }
        true
    }

    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        // Sort ALL pending by prompt_len, pick shortest N.
        let mut indices: Vec<usize> = (0..requests.len()).collect();
        indices.sort_by_key(|&i| requests[i].prompt_len);
        indices.truncate(capacity);
        indices
    }

    fn prefill_slice_budget(&self, active_timings: &[ActiveSeqTiming], full_chunk: usize) -> usize {
        // No decode active → no TBT pressure → inject the full chunk.
        if active_timings.is_empty() {
            return full_chunk;
        }

        // Hard suppress: if ANY decode has already blown its TBT deadline,
        // return 0 so the caller runs decode-only this tick. This is the
        // ONLY case that returns 0 (spec B2 late-decode protection). Note
        // `worst` is used ONLY as this hard guard, never to size the slice
        // (the fused step self-resets last_token_time, so it cannot drive
        // the budget — that is what makes B2's open-loop design fail).
        let now = Instant::now();
        let worst = active_timings
            .iter()
            .map(|t| now.duration_since(t.last_token_time))
            .max()
            .unwrap_or_default();
        if worst >= self.tbt_deadline {
            return 0;
        }

        // Cost-driven slice (spec B2 fix): size the prefill region from the
        // measured (here: seeded) step cost so the fused step fits the TBT
        // target. budget = (TBT_target − T_dec) / cost_per_tok.
        let tbt_target_ms = self.tbt_deadline.as_secs_f64() * 1_000.0;
        let headroom_ms = (tbt_target_ms - self.t_dec_ms_seed).max(0.0);
        let raw = (headroom_ms / self.cost_per_tok_ms_seed).floor() as i64;
        let raw = raw.clamp(0, full_chunk as i64) as usize;

        // Clamp to [min_prefill_slice, full_chunk], WY4-align, floor at 4.
        let budget = raw.clamp(self.min_prefill_slice.min(full_chunk), full_chunk);
        let budget = (budget / 4) * 4;
        budget.max(4).min(full_chunk)
    }

    fn name(&self) -> &str {
        "slai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_always_prefills() {
        let policy = FifoPolicy;
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        assert!(policy.should_prefill(&timings));
        assert!(policy.should_prefill(&[]));
    }

    #[test]
    fn fifo_selects_first_n() {
        let policy = FifoPolicy;
        let requests = vec![
            PendingRequestInfo {
                prompt_len: 100,
                index: 0,
            },
            PendingRequestInfo {
                prompt_len: 10,
                index: 1,
            },
            PendingRequestInfo {
                prompt_len: 50,
                index: 2,
            },
            PendingRequestInfo {
                prompt_len: 200,
                index: 3,
            },
        ];
        assert_eq!(policy.select_prefills(&requests, 2), vec![0, 1]);
        assert_eq!(policy.select_prefills(&requests, 10), vec![0, 1, 2, 3]);
    }

    #[test]
    fn slai_prefills_when_no_active() {
        let policy = SlaiPolicy::new(100);
        assert!(policy.should_prefill(&[]));
    }

    #[test]
    fn slai_prefills_when_fresh() {
        let policy = SlaiPolicy::new(100);
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        assert!(policy.should_prefill(&timings));
    }

    #[test]
    fn slai_skips_prefill_near_deadline() {
        let policy = SlaiPolicy::new(100); // 80ms margin
        let old_time = Instant::now() - Duration::from_millis(85);
        let timings = vec![ActiveSeqTiming {
            last_token_time: old_time,
        }];
        assert!(!policy.should_prefill(&timings));
    }

    #[test]
    fn slai_prefills_within_margin() {
        let policy = SlaiPolicy::new(100); // 80ms margin
        let recent = Instant::now() - Duration::from_millis(50);
        let timings = vec![ActiveSeqTiming {
            last_token_time: recent,
        }];
        assert!(policy.should_prefill(&timings));
    }

    #[test]
    fn slai_one_urgent_blocks_prefill() {
        let policy = SlaiPolicy::new(100);
        let now = Instant::now();
        let timings = vec![
            ActiveSeqTiming {
                last_token_time: now,
            },
            ActiveSeqTiming {
                last_token_time: now - Duration::from_millis(90),
            },
        ];
        assert!(!policy.should_prefill(&timings));
    }

    #[test]
    fn slai_selects_shortest_from_all() {
        let policy = SlaiPolicy::new(100);
        let requests = vec![
            PendingRequestInfo {
                prompt_len: 500,
                index: 0,
            },
            PendingRequestInfo {
                prompt_len: 10,
                index: 1,
            },
            PendingRequestInfo {
                prompt_len: 200,
                index: 2,
            },
            PendingRequestInfo {
                prompt_len: 50,
                index: 3,
            },
            PendingRequestInfo {
                prompt_len: 300,
                index: 4,
            },
        ];
        // Capacity 3: picks shortest 3 → indices 1(10), 3(50), 2(200)
        assert_eq!(policy.select_prefills(&requests, 3), vec![1, 3, 2]);
    }

    #[test]
    fn slai_selects_all_when_capacity_exceeds() {
        let policy = SlaiPolicy::new(100);
        let requests = vec![
            PendingRequestInfo {
                prompt_len: 100,
                index: 0,
            },
            PendingRequestInfo {
                prompt_len: 10,
                index: 1,
            },
        ];
        // Capacity 10 > 2 requests: returns all sorted
        assert_eq!(policy.select_prefills(&requests, 10), vec![1, 0]);
    }

    #[test]
    fn slai_stable_order_for_equal_lengths() {
        let policy = SlaiPolicy::new(100);
        let requests = vec![
            PendingRequestInfo {
                prompt_len: 50,
                index: 0,
            },
            PendingRequestInfo {
                prompt_len: 50,
                index: 1,
            },
            PendingRequestInfo {
                prompt_len: 50,
                index: 2,
            },
        ];
        assert_eq!(policy.select_prefills(&requests, 3), vec![0, 1, 2]);
    }

    #[test]
    fn select_prefills_empty() {
        assert!(FifoPolicy.select_prefills(&[], 5).is_empty());
        assert!(SlaiPolicy::new(100).select_prefills(&[], 5).is_empty());
    }

    #[test]
    fn fifo_slice_budget_is_full_chunk() {
        // Default trait impl: FIFO always injects the full chunk.
        let policy = FifoPolicy;
        assert_eq!(policy.prefill_slice_budget(&[], 4080), 4080);
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        assert_eq!(policy.prefill_slice_budget(&timings, 4080), 4080);
    }

    #[test]
    fn slai_slice_budget_full_when_no_active() {
        let policy = SlaiPolicy::new(100);
        assert_eq!(policy.prefill_slice_budget(&[], 4080), 4080);
    }

    #[test]
    fn slai_slice_budget_zero_past_deadline() {
        // worst >= tbt_deadline → hard suppress (0), decode-only this tick.
        let policy = SlaiPolicy::new(100);
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now() - Duration::from_millis(120),
        }];
        assert_eq!(policy.prefill_slice_budget(&timings, 4080), 0);
    }

    #[test]
    fn slai_slice_budget_bounded_and_wy4_aligned() {
        // Fresh decode, under deadline → a positive, WY4-aligned slice in
        // [min, full_chunk], never 0.
        let policy = SlaiPolicy::new(100);
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        let b = policy.prefill_slice_budget(&timings, 4080);
        assert!(b > 0, "non-deadline budget must be > 0");
        assert!(b <= 4080, "must never exceed full_chunk");
        assert_eq!(b % 4, 0, "must be WY4-aligned");
    }

    #[test]
    fn slai_slice_budget_never_exceeds_small_full_chunk() {
        // Small chunk cap must clamp the slice (and stay WY4-aligned).
        let policy = SlaiPolicy::new(100);
        let timings = vec![ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        let b = policy.prefill_slice_budget(&timings, 64);
        assert!(b <= 64);
        assert_eq!(b % 4, 0);
    }
}
