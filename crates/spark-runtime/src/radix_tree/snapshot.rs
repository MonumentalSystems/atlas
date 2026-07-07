// SPDX-License-Identifier: AGPL-3.0-only

//! SSM snapshot LRU index — independent of the token-radix structure.
//!
//! Snapshots are keyed by (session_hash, token_count, prefix_hash) so the
//! same prompt across requests can hit a cached SSM state without going
//! through the radix tree.

use super::hash_token_prefix;

pub(super) struct SnapshotEntry {
    snapshot_id: usize,
    session_hash: u64,
    token_count: usize,
    prefix_hash: u64,
    last_access: u64,
    /// Cumulative hits over the entry's lifetime — combined with
    /// `last_access` in eviction to approximate the forecast-based
    /// policy from the Marconi paper §4 (B.4, 2026-04-25). Hot
    /// prefixes (high hit count) survive longer than cold ones at
    /// the same age.
    hit_count: u32,
}

pub(super) struct SsmSnapshotIndex {
    pub(super) entries: Vec<SnapshotEntry>,
    pub(super) access_counter: u64,
    /// Session of the most recent `lookup` — the live conversation. Its
    /// DEEPEST snapshot is the one its next warm turn will restore from, so
    /// `evict_lru` protects it (ATLAS_SSM_TAIL_PROTECT, ported from #278).
    /// Tracks the running tip: recomputed each eviction from `token_count`,
    /// never a pinned slot. Complements the session-aware victim ranking
    /// below — session-aware protects the live session vs *dormant* ones;
    /// tail-protect protects the deep tail *within* the live session (the
    /// single/dominant-conversation case session-freshness can't see).
    last_lookup_session: u64,
    /// Phase-0 measurement counters (ATLAS_SSM_SNAP_STATS). All aggregate,
    /// off the hot path's critical decisions — they only observe. The residual
    /// `recompute_tokens_on_hit` after tail-protect + a large pool is exactly
    /// what Phase 1 (spill-not-drop) converts from recompute → fault-in.
    stats: SnapshotStats,
}

/// Aggregate SSM-snapshot cache telemetry (Phase 0). Summarised via
/// `log_stats_if_due` when `ATLAS_SSM_SNAP_STATS` is set.
#[derive(Default, Clone, Copy)]
pub(super) struct SnapshotStats {
    /// Snapshots registered (new prefix inserted, not an overwrite).
    pub saves: u64,
    /// Prefix lookups attempted.
    pub lookups: u64,
    /// Lookups that restored *some* anchor (deep or shallow).
    pub hits: u64,
    /// Σ restored-anchor depth over hits — mean anchor = this / hits.
    pub anchor_depth_sum: u64,
    /// Σ (matched_tokens − anchor_depth) over hits: the SSM tokens that still
    /// had to be recomputed because the deep tail was not resident. This is the
    /// #278 metric ("mean recompute 4438→262 tok") and Phase 1's target.
    pub recompute_tokens_on_hit: u64,
    /// Σ matched_tokens over *misses* (no anchor at all → full recompute).
    pub recompute_tokens_on_miss: u64,
    /// Snapshot slots freed by `evict_lru` — today a DROP (state discarded);
    /// Phase 1 makes each of these a spill-to-cascade instead.
    pub evictions: u64,
}

impl SsmSnapshotIndex {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            access_counter: 0,
            last_lookup_session: 0,
            stats: SnapshotStats::default(),
        }
    }

    pub(super) fn insert(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                let old = entry.snapshot_id;
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return Some(old);
            }
        }
        self.access_counter += 1;
        self.stats.saves += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            hit_count: 0,
        });
        None
    }

    /// Find deepest snapshot matching session within matched_tokens range.
    /// Task #24: `adapter_id` is folded into the recomputed prefix hash so a
    /// snapshot registered under one adapter never matches another's lookup.
    pub(super) fn lookup(
        &mut self,
        tokens: &[u32],
        matched_tokens: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Option<(usize, usize)> {
        // Track the live conversation so eviction can protect its deep tail
        // (ATLAS_SSM_TAIL_PROTECT).
        if session_hash != 0 {
            self.last_lookup_session = session_hash;
        }
        let mut best: Option<(usize, usize)> = None; // (snapshot_id, token_count)
        for entry in &mut self.entries {
            if entry.token_count > matched_tokens {
                continue;
            }
            if session_hash != 0 && entry.session_hash != 0 && entry.session_hash != session_hash {
                continue;
            }
            let h = hash_token_prefix(tokens, entry.token_count, adapter_id);
            if h != entry.prefix_hash {
                continue;
            }
            if best.is_none() || entry.token_count > best.unwrap().1 {
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                entry.hit_count = entry.hit_count.saturating_add(1);
                best = Some((entry.snapshot_id, entry.token_count));
            }
        }
        // Phase-0 telemetry: hit-rate + recompute distance. `recompute` is the
        // SSM prefix that still had to be re-run because no deeper anchor was
        // resident — the exact loss tail-protect shrinks and Phase 1 removes.
        self.stats.lookups += 1;
        match best {
            Some((_, anchor)) => {
                self.stats.hits += 1;
                self.stats.anchor_depth_sum += anchor as u64;
                self.stats.recompute_tokens_on_hit += matched_tokens.saturating_sub(anchor) as u64;
            }
            None => {
                self.stats.recompute_tokens_on_miss += matched_tokens as u64;
            }
        }
        if std::env::var("ATLAS_SNAP_LOOKUP_DBG").is_ok() {
            let mut cands: Vec<usize> = self.entries.iter().map(|e| e.token_count).collect();
            cands.sort_unstable();
            tracing::info!(
                "snap-lookup: matched={matched_tokens} selected={:?} n_entries={} token_counts={:?}",
                best.map(|b| b.1),
                self.entries.len(),
                cands,
            );
        }
        self.log_stats_if_due();
        best
    }

    /// Emit an aggregate SSM-snapshot cache summary every 64 lookups when
    /// `ATLAS_SSM_SNAP_STATS` is set. Off-by-default and read-only, so it never
    /// perturbs serving; the line is the Phase-0 measurement surface (hit-rate,
    /// mean restore anchor, mean recompute tok/turn — the #278 metrics).
    fn log_stats_if_due(&self) {
        if self.stats.lookups % 64 != 0 || std::env::var_os("ATLAS_SSM_SNAP_STATS").is_none() {
            return;
        }
        let s = &self.stats;
        let hit_rate = s.hits as f64 / s.lookups.max(1) as f64;
        let mean_anchor = s.anchor_depth_sum as f64 / s.hits.max(1) as f64;
        let mean_recompute_hit = s.recompute_tokens_on_hit as f64 / s.hits.max(1) as f64;
        tracing::info!(
            "ssm-snap-stats: lookups={} hits={} hit_rate={:.2} saves={} evictions(drops)={} \
             mean_anchor={:.0}tok mean_recompute_on_hit={:.0}tok recompute_on_miss={}tok resident={}",
            s.lookups,
            s.hits,
            hit_rate,
            s.saves,
            s.evictions,
            mean_anchor,
            mean_recompute_hit,
            s.recompute_tokens_on_miss,
            self.entries.len(),
        );
    }

    pub(super) fn evict_lru(&mut self) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        // Per-entry forecast score (B.4, Marconi paper §4): old AND cold first.
        // last_access * (1 + hit_count) — recent/hot survive. #155 fixed the
        // inverted (÷) form that evicted just-selected snapshots.
        let escore = |e: &SnapshotEntry| e.last_access.saturating_mul(1 + e.hit_count as u64);

        // SESSION-AWARE eviction (default ON; ATLAS_SNAP_EVICT_LEGACY=1 → old
        // per-entry policy). The agentic workload interleaves ~20 multi-turn
        // conversations; per-entry LRU evicts the active conversation's OWN deep
        // checkpoints whenever it goes briefly dormant (its unique deep snapshots
        // have hit_count=0 and a stale last_access vs another conversation's fresh
        // ones), so its next warm turn full-recomputes the SSM state (TTFT 1s→50s).
        // Fix: evict from the STALEST conversation first — rank by the session's
        // freshness (max last_access over its entries), so the active conversation's
        // ENTIRE deep checkpoint chain stays resident until every other (completed/
        // dormant) conversation is gone. Within the victim session, drop its lowest
        // forecast-score entry. This is "prefix caching like llama" for SSM state:
        // the live conversation never re-recomputes what it already computed.
        // Selecting a different victim is correctness-safe — restore re-validates
        // (session_hash + prefix_hash) before using any snapshot; eviction only
        // frees a slot.
        if std::env::var_os("ATLAS_SNAP_EVICT_LEGACY").is_none() {
            let tail_protect = self.last_lookup_session != 0
                && std::env::var_os("ATLAS_SSM_TAIL_PROTECT").is_some();
            let victim_idx = self.session_aware_victim(tail_protect);
            let entry = self.entries.swap_remove(victim_idx);
            self.stats.evictions += 1;
            return Some(entry.snapshot_id);
        }

        let mut victim_idx = 0;
        let mut victim_score = u64::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            let score = escore(entry);
            if score < victim_score {
                victim_score = score;
                victim_idx = i;
            }
        }
        let entry = self.entries.swap_remove(victim_idx);
        self.stats.evictions += 1;
        Some(entry.snapshot_id)
    }

    /// Pure victim selection for the session-aware eviction policy (default
    /// path). Returns the index into `self.entries` to drop. Split out so it
    /// is unit-testable without mutating process env.
    ///
    /// Ranking: evict the STALEST conversation first (by session freshness =
    /// max `last_access` over its entries), then the lowest per-entry forecast
    /// score within it — the live conversation's whole deep chain stays
    /// resident until every dormant conversation is gone.
    ///
    /// `tail_protect` (ATLAS_SSM_TAIL_PROTECT, ported from #278): additionally
    /// exempt the live conversation's DEEPEST snapshot — its next warm turn
    /// restores from it. Session-aware ranking alone only protects the live
    /// session vs *dormant* ones; within a single/dominant session it falls
    /// through to lowest-escore = the just-aged deep tail (hit_count=0), which
    /// the hot token-8192 anchor out-scores, so warm restore falls back to 8192
    /// and recomputes thousands of SSM tokens (~4400 tok, ~7.6s TTFT/turn).
    /// Exactly ONE entry is exempted, scoped to the active session, so any pool
    /// >=2 always has a victim and never deadlocks. Correctness-safe: restore
    /// re-validates session_hash+prefix_hash+depth, so changing the victim
    /// cannot cause a wrong-position restore.
    fn session_aware_victim(&self, tail_protect: bool) -> usize {
        let escore = |e: &SnapshotEntry| e.last_access.saturating_mul(1 + e.hit_count as u64);
        // session freshness = max last_access among that session's entries.
        let mut session_fresh: std::collections::HashMap<u64, u64> =
            std::collections::HashMap::with_capacity(self.entries.len());
        for e in &self.entries {
            let f = session_fresh.entry(e.session_hash).or_insert(0);
            if e.last_access > *f {
                *f = e.last_access;
            }
        }
        let n = self.entries.len();
        let protected_idx: Option<usize> = if tail_protect {
            self.entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.session_hash == self.last_lookup_session)
                .max_by_key(|(_, e)| e.token_count)
                .map(|(i, _)| i)
        } else {
            None
        };
        // Seed the victim at a non-protected index so a pool of exactly the one
        // protected entry (n==1) still yields a victim, and n>=2 never returns
        // the protected slot even if the loop matches nothing.
        let mut victim_idx = protected_idx.map_or(0, |p| if p == 0 && n > 1 { 1 } else { 0 });
        // (stalest session first, then lowest entry score within it)
        let mut victim_key = (u64::MAX, u64::MAX);
        for (i, e) in self.entries.iter().enumerate() {
            if Some(i) == protected_idx && n > 1 {
                continue; // never evict the live session's deepest tail
            }
            let sf = *session_fresh.get(&e.session_hash).unwrap_or(&0);
            let key = (sf, escore(e));
            if key < victim_key {
                victim_key = key;
                victim_idx = i;
            }
        }
        victim_idx
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an entry with an explicit forecast profile. `snapshot_id` doubles
    /// as a stable identity we assert on (independent of Vec index).
    fn entry(
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
        last_access: u64,
        hit_count: u32,
    ) -> SnapshotEntry {
        SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash: snapshot_id as u64, // unique, irrelevant to victim choice
            last_access,
            hit_count,
        }
    }

    fn index(entries: Vec<SnapshotEntry>, live: u64) -> SsmSnapshotIndex {
        SsmSnapshotIndex {
            entries,
            access_counter: 1000,
            last_lookup_session: live,
            stats: SnapshotStats::default(),
        }
    }

    /// The deep-tail eviction inversion (#278 root cause), reproduced against
    /// the session-aware policy: within a SINGLE live session the hot 8192
    /// anchor (self-reinforced hit_count) out-scores the just-aged deep tail
    /// (hit_count=0), so without tail-protect the tail is the victim.
    #[test]
    fn deep_tail_evicted_without_tail_protect() {
        let idx = index(
            vec![
                entry(/*id*/ 7, /*sess*/ 1, /*tok*/ 8192, /*last*/ 100, /*hits*/ 10),
                entry(/*id*/ 9, /*sess*/ 1, /*tok*/ 16000, /*last*/ 50, /*hits*/ 0),
            ],
            1,
        );
        // Victim is the deep tail (id 9) — the pathology.
        let v = idx.session_aware_victim(false);
        assert_eq!(idx.entries[v].snapshot_id, 9);
    }

    /// With tail-protect the live session's DEEPEST snapshot is exempt, so the
    /// hot anchor is evicted instead and the warm-turn restore anchor survives.
    #[test]
    fn deep_tail_survives_with_tail_protect() {
        let idx = index(
            vec![
                entry(7, 1, 8192, 100, 10),
                entry(9, 1, 16000, 50, 0),
            ],
            1,
        );
        let v = idx.session_aware_victim(true);
        // Victim must NOT be the protected deep tail (id 9); it is the anchor.
        assert_eq!(idx.entries[v].snapshot_id, 7);
    }

    /// Tail-protect only shields the LIVE conversation's tail; a dormant
    /// session's deep tail is still evictable (correct — session-aware ranking
    /// evicts the stalest conversation first).
    #[test]
    fn dormant_session_tail_not_protected() {
        // session 2 is live; session 1 is dormant (older last_access).
        let idx = index(
            vec![
                entry(1, 1, 20000, 10, 0),  // dormant deep tail — should die first
                entry(2, 2, 4000, 90, 0),   // live shallow
                entry(3, 2, 12000, 95, 0),  // live deep tail — protected
            ],
            2,
        );
        let v = idx.session_aware_victim(true);
        assert_eq!(idx.entries[v].snapshot_id, 1, "stalest (dormant) session evicted first");
    }

    /// A pool of exactly one entry must still yield that entry as victim even
    /// when it is the protected tail — otherwise `save` can never reclaim and
    /// the cache deadlocks.
    #[test]
    fn single_protected_entry_still_evictable() {
        let idx = index(vec![entry(5, 1, 16000, 50, 0)], 1);
        let v = idx.session_aware_victim(true);
        assert_eq!(idx.entries[v].snapshot_id, 5);
    }

    /// `lookup` records the live session so a later eviction protects the right
    /// conversation's tail.
    #[test]
    fn lookup_tracks_live_session() {
        let mut idx = SsmSnapshotIndex::new();
        assert_eq!(idx.last_lookup_session, 0);
        // No matching entries — lookup returns None but must still latch the session.
        let _ = idx.lookup(&[1, 2, 3], 3, /*session*/ 42, /*adapter*/ 0);
        assert_eq!(idx.last_lookup_session, 42);
    }

    /// Telemetry: a miss records full-recompute; a hit records the residual
    /// distance between the match point and the restored anchor.
    #[test]
    fn stats_track_hits_and_recompute() {
        let mut idx = SsmSnapshotIndex::new();
        let toks: Vec<u32> = (0..100).collect();

        // Cold miss over 100 matched tokens → full recompute counted.
        assert!(idx.lookup(&toks, 100, /*session*/ 7, /*adapter*/ 0).is_none());
        // Register an anchor at depth 40 (hash must line up with lookup's recompute).
        let ph = super::hash_token_prefix(&toks, 40, 0);
        assert!(idx.insert(ph, /*snap*/ 3, /*session*/ 7, /*token_count*/ 40).is_none());
        // Warm turn: match 100 tokens, restore the depth-40 anchor → 60 recompute.
        let hit = idx.lookup(&toks, 100, 7, 0);
        assert_eq!(hit, Some((3, 40)));

        let s = idx.stats; // child module: private field is in scope
        assert_eq!(s.lookups, 2);
        assert_eq!(s.hits, 1);
        assert_eq!(s.saves, 1);
        assert_eq!(s.anchor_depth_sum, 40);
        assert_eq!(s.recompute_tokens_on_hit, 60, "matched(100) - anchor(40)");
        assert_eq!(s.recompute_tokens_on_miss, 100, "cold miss = full recompute");
    }
}
