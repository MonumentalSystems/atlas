// SPDX-License-Identifier: AGPL-3.0-only

//! Pure residency state machine for the ROLLING decode-rollback tier
//! (`ATLAS_SSM_DECODE_RING_ROLL`).
//!
//! The decode-rollback ring keeps `DECODE_ROLLBACK_RING_SLOTS(8)` boundary
//! snapshots per active sequence. Today all 8 are pure HBM (~32 GB at C=64).
//! The rolling tier keeps only the `hot_lanes` most-recent boundaries per
//! sequence HBM-resident (plus `DECODE_SPILL_MARGIN` async-drain lane(s)) and
//! spills the deeper ones to a cold tier (local NVMe / RDMA peer). This module
//! owns the **decision logic only** — no GPU, no I/O — so every rule (hot/cold
//! lane selection, cold-key namespacing, fault-before-read ordering, the
//! spill-completes-after-truncate epoch guard) is unit-tested in isolation. The
//! pool ([`super::ssm_snapshot::SsmSnapshotPool`]) turns each returned plan into
//! device copies + store ops.
//!
//! ## Invariants proven here
//! - A logical slot is always exactly one of Absent / Resident / Spilling / Cold.
//! - A live rollback target (Resident, Spilling, or Cold) is NEVER lost: a
//!   Spilling slot restores from its still-pinned lane (no fault, no wait); a
//!   Cold slot faults back before the restore read.
//! - `resident ≤ hot_lanes` and `resident + spilling ≤ l_phys` at all times.
//! - An epoch guard makes a spill that completes AFTER its slot was overwritten
//!   a no-op-plus-remove, never a stale Cold commit.

use std::collections::VecDeque;

/// Residency of one logical ring slot `(seq, logical)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Residency {
    /// No snapshot has ever been written to this logical slot.
    Absent,
    /// HBM-resident in physical `lane` (of this seq's `l_phys` lanes).
    Resident { lane: usize },
    /// Being drained to the cold tier; `lane` still holds the valid bytes
    /// (pinned) until the spill's `store.put` commits under `epoch`.
    Spilling { lane: usize, epoch: u64 },
    /// Bytes live only in the cold tier under `cold_key`, committed at `epoch`.
    Cold { epoch: u64 },
}

/// A cold-tier spill the pool must drive (async, off the decode path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpillReq {
    pub seq: usize,
    pub logical: usize,
    /// Physical lane holding the bytes to gather (pinned until commit).
    pub lane: usize,
    /// Epoch stamped on the `Spilling` slot; the completion callback must pass
    /// it back so a stale commit (slot overwritten meanwhile) is caught.
    pub epoch: u64,
    /// Namespaced cold-tier key this blob is stored under.
    pub cold_key: u64,
}

/// The pool's plan for one boundary save.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SaveDecision {
    /// Overwrite the resident lane in place (the common per-boundary write of an
    /// already-hot slot) — no spill, no eviction.
    InPlace { lane: usize },
    /// Write into a freshly-claimed lane; if `spill` is `Some`, the displaced
    /// LRU-hot slot must be drained async first-come.
    Fresh { lane: usize, spill: Option<SpillReq> },
    /// No free lane — the pool must synchronously finish these in-flight spills
    /// (freeing their lanes) then re-plan. The common path never hits this
    /// (`DECODE_SPILL_MARGIN` keeps a free lane); it is the bounded-queue
    /// backpressure valve at high C.
    Backpressure { drain: Vec<SpillReq> },
}

/// The pool's plan for one rollback restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RestoreDecision {
    /// Restore directly from a pinned HBM `lane` (Resident, or Spilling whose
    /// bytes are still valid) — a plain D2D, no fault.
    FromLane { lane: usize },
    /// Cold: fault `cold_key` into `scratch_lane` (H2D + synchronize) BEFORE the
    /// D2D restore reads it. The pool releases the scratch lane after.
    FaultThenRestore { scratch_lane: usize, cold_key: u64 },
    /// No live snapshot for this position — the caller must decline the rollback.
    Decline,
}

/// Result of a spill completion callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpillCommit {
    /// Epoch matched: slot is now Cold, its lane freed.
    Committed,
    /// Epoch mismatched (overwritten/truncated mid-spill): the just-written cold
    /// blob is stale and must be removed; the lane was already reclaimed by the
    /// superseding save.
    Cancelled { remove_cold_key: u64 },
}

/// Per-sequence lane bookkeeping.
#[derive(Debug, Clone)]
struct SeqLanes {
    /// `logical (0..ring_slots)` → residency.
    slots: Vec<Residency>,
    /// `lane (0..l_phys)` → logical occupant when Resident/Spilling.
    lane_occupant: Vec<Option<usize>>,
    /// Resident logical slots, LRU at the front (spill victim), MRU at the back.
    lru: VecDeque<usize>,
    /// Monotone per-logical epoch; bumped on every fresh incarnation so a late
    /// spill of a prior incarnation cannot commit stale bytes.
    epoch: Vec<u64>,
}

impl SeqLanes {
    fn new(ring_slots: usize, l_phys: usize) -> Self {
        Self {
            slots: vec![Residency::Absent; ring_slots],
            lane_occupant: vec![None; l_phys],
            lru: VecDeque::with_capacity(l_phys),
            epoch: vec![0; ring_slots],
        }
    }
    fn resident_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|r| matches!(r, Residency::Resident { .. }))
            .count()
    }
    fn free_lane(&self) -> Option<usize> {
        self.lane_occupant.iter().position(|o| o.is_none())
    }
    fn touch_mru(&mut self, logical: usize) {
        self.lru.retain(|&l| l != logical);
        self.lru.push_back(logical);
    }
}

/// The rolling-tier residency manager (whole decode region).
#[derive(Debug, Clone)]
pub(crate) struct DecodeRingManager {
    ring_slots: usize,
    l_phys: usize,
    hot_lanes: usize,
    max_seqs: usize,
    fault_scratch: usize,
    namespace: u64,
    seqs: Vec<SeqLanes>,
    scratch_free: Vec<usize>,
}

impl DecodeRingManager {
    pub(crate) fn new(
        ring_slots: usize,
        hot_lanes: usize,
        spill_margin: usize,
        fault_scratch: usize,
        max_seqs: usize,
        namespace: u64,
    ) -> Self {
        let l_phys = hot_lanes + spill_margin;
        Self {
            ring_slots,
            l_phys,
            hot_lanes,
            max_seqs,
            fault_scratch,
            namespace,
            seqs: (0..max_seqs)
                .map(|_| SeqLanes::new(ring_slots, l_phys))
                .collect(),
            scratch_free: (0..fault_scratch).rev().collect(),
        }
    }

    #[inline]
    pub(crate) fn l_phys(&self) -> usize {
        self.l_phys
    }
    #[inline]
    pub(crate) fn fault_scratch(&self) -> usize {
        self.fault_scratch
    }

    /// Physical frame index (into the decode region, `l_phys × max_seqs + scratch`
    /// frames per layer) for a per-seq lane. The pool multiplies by the per-frame
    /// byte stride.
    #[inline]
    pub(crate) fn lane_frame(&self, seq: usize, lane: usize) -> usize {
        seq * self.l_phys + lane
    }
    /// Physical frame index for a shared scratch lane (appended after all seq
    /// lanes).
    #[inline]
    pub(crate) fn scratch_frame(&self, scratch_lane: usize) -> usize {
        self.max_seqs * self.l_phys + scratch_lane
    }
    /// Total physical frames the decode region must allocate per layer.
    #[inline]
    pub(crate) fn total_frames(&self) -> usize {
        self.max_seqs * self.l_phys + self.fault_scratch
    }

    /// Namespaced cold-tier key for `(seq, logical)` — a SLOT COORDINATE, not
    /// content. Keyed by the *logical slot* (not token position) so keys are
    /// REUSED across a sequence's lifetime and across seq-slot recycling — the
    /// store never grows past `max_seqs × ring_slots` entries and no per-entry
    /// `remove` is needed on truncate. The `DECODE_DOMAIN` fold (via
    /// `namespace`) separates decode keys from Marconi prefix-hash keys on a
    /// shared store/peer — and that is ALL it does: the key is a pure function
    /// of (slot coords, namespace), identical across same-model processes, so
    /// it does NOT make keys client-unique. Cross-CLIENT privacy comes from
    /// the per-process client salt folded into the store-level namespace
    /// (`ssm_tier::fingerprint::resolve_decode_ns`) — the paging peer being
    /// the only cross-process-shared decode store.
    pub(crate) fn cold_key(&self, seq: usize, logical: usize) -> u64 {
        // SSOT: VALUE-IDENTICAL to the historical hand-rolled three-input
        // splitmix64 fold (pinned in `decode_ring_manager_key_tests.rs`).
        // mix64(a, b) = finalize(a ^ b·GOLDEN); regroup the old pre-mix
        // `seq·GOLDEN ^ logical·P2 ^ ns` as a = logical·P2 ^ ns, b = seq.
        // P2 stays a LOCAL const (not promoted into atlas_tier::hash): decode
        // keys are ephemeral; the durable-key API should not widen for them.
        const LOGICAL_SPREAD: u64 = 0xC2B2_AE3D_27D4_EB4F; // xxHash64 prime-2
        atlas_tier::hash::mix64(
            (logical as u64).wrapping_mul(LOGICAL_SPREAD) ^ self.namespace,
            seq as u64,
        )
    }

    /// Current residency of `(seq, logical)` — for tests / assertions.
    pub(crate) fn residency(&self, seq: usize, logical: usize) -> Residency {
        self.seqs[seq].slots[logical]
    }

    /// Plan the per-boundary save of `(seq, logical)`. Mutates state to reflect a
    /// plan the pool WILL carry out — except `Backpressure`, which leaves state
    /// unchanged (the pool drains, then re-calls).
    pub(crate) fn plan_save(&mut self, seq: usize, logical: usize) -> SaveDecision {
        let l_phys = self.l_phys;
        let hot = self.hot_lanes;
        // Resident in place → cheapest path (overwrite, no eviction, no epoch
        // bump: same logical incarnation, freshest bytes replace older bytes).
        if let Residency::Resident { lane } = self.seqs[seq].slots[logical] {
            self.seqs[seq].touch_mru(logical);
            return SaveDecision::InPlace { lane };
        }

        // Need a fresh lane. If none is free, all lanes are Resident+Spilling and
        // the spill drains haven't completed → backpressure.
        let Some(lane) = self.seqs[seq].free_lane() else {
            let drain: Vec<SpillReq> = self.seqs[seq]
                .slots
                .iter()
                .enumerate()
                .filter_map(|(lg, r)| match *r {
                    Residency::Spilling { lane, epoch } => Some(SpillReq {
                        seq,
                        logical: lg,
                        lane,
                        epoch,
                        cold_key: self.cold_key(seq, lg),
                    }),
                    _ => None,
                })
                .collect();
            return SaveDecision::Backpressure { drain };
        };

        // Bump epoch: this is a new incarnation of `logical`. Any in-flight spill
        // of a PRIOR incarnation of this same logical slot will now epoch-mismatch
        // on commit (Cancelled) instead of overwriting our fresh Cold bytes.
        self.seqs[seq].epoch[logical] += 1;

        // Claim the lane, make logical Resident.
        self.seqs[seq].lane_occupant[lane] = Some(logical);
        self.seqs[seq].slots[logical] = Residency::Resident { lane };
        self.seqs[seq].touch_mru(logical);

        // Keep resident ≤ hot_lanes: if we just exceeded it, evict the LRU
        // resident (never `logical`) to Spilling.
        let mut spill = None;
        if self.seqs[seq].resident_count() > hot {
            // Find LRU resident that is not `logical`.
            let victim = loop {
                let Some(cand) = self.seqs[seq].lru.pop_front() else {
                    break None;
                };
                if cand == logical {
                    // Re-queue as MRU and keep looking (shouldn't evict the slot
                    // we just wrote).
                    self.seqs[seq].lru.push_back(cand);
                    continue;
                }
                if let Residency::Resident { lane: vlane } = self.seqs[seq].slots[cand] {
                    break Some((cand, vlane));
                }
                // Stale lru entry (already spilled) — drop and continue.
            };
            if let Some((vlogical, vlane)) = victim {
                let epoch = self.seqs[seq].epoch[vlogical];
                self.seqs[seq].slots[vlogical] = Residency::Spilling { lane: vlane, epoch };
                spill = Some(SpillReq {
                    seq,
                    logical: vlogical,
                    lane: vlane,
                    epoch,
                    cold_key: self.cold_key(seq, vlogical),
                });
            }
        }
        debug_assert!(self.seqs[seq].resident_count() <= hot.max(1));
        debug_assert!(l_phys >= self.lanes_in_use(seq));
        SaveDecision::Fresh { lane, spill }
    }

    fn lanes_in_use(&self, seq: usize) -> usize {
        self.seqs[seq]
            .lane_occupant
            .iter()
            .filter(|o| o.is_some())
            .count()
    }

    /// Plan the rare rollback restore of `(seq, logical)`.
    pub(crate) fn plan_restore(&mut self, seq: usize, logical: usize) -> RestoreDecision {
        match self.seqs[seq].slots[logical] {
            Residency::Resident { lane } | Residency::Spilling { lane, .. } => {
                RestoreDecision::FromLane { lane }
            }
            Residency::Cold { .. } => {
                let cold_key = self.cold_key(seq, logical);
                match self.scratch_free.pop() {
                    Some(scratch_lane) => {
                        RestoreDecision::FaultThenRestore { scratch_lane, cold_key }
                    }
                    // No scratch lane free (≥ ROLLBACK_RESTEER_CAP provisioned, so
                    // a well-behaved caller never starves) — decline rather than
                    // corrupt.
                    None => RestoreDecision::Decline,
                }
            }
            Residency::Absent => RestoreDecision::Decline,
        }
    }

    /// Release a scratch lane claimed by a `FaultThenRestore`, after the D2D
    /// restore has read it.
    pub(crate) fn release_scratch(&mut self, scratch_lane: usize) {
        if !self.scratch_free.contains(&scratch_lane) {
            self.scratch_free.push(scratch_lane);
        }
    }

    /// Commit (or cancel) a spill whose `store.put` just returned. The epoch
    /// guard closes the spill-completes-after-truncate/overwrite race.
    pub(crate) fn complete_spill(&mut self, seq: usize, logical: usize, epoch: u64) -> SpillCommit {
        match self.seqs[seq].slots[logical] {
            Residency::Spilling { lane, epoch: e } if e == epoch => {
                self.seqs[seq].lane_occupant[lane] = None;
                self.seqs[seq].slots[logical] = Residency::Cold { epoch };
                SpillCommit::Committed
            }
            _ => SpillCommit::Cancelled {
                remove_cold_key: self.cold_key(seq, logical),
            },
        }
    }

    /// Reset an ENTIRE sequence slot to a fresh state — a new sequence has
    /// reused this seq-slot (cross-sequence recycling) or the whole ring was
    /// truncated. Without this the new incarnation inherits the prior sequence's
    /// lane occupancy / LRU / residency, corrupting eviction + restore. Returns
    /// the cold keys of every Cold/Spilling slot so the pool can `store.remove`
    /// them; every epoch is bumped so any in-flight spill of the old incarnation
    /// cancels on commit.
    pub(crate) fn reset_seq(&mut self, seq: usize) -> Vec<u64> {
        let mut keys = Vec::new();
        for logical in 0..self.ring_slots {
            if let Some(k) = self.drop_slot(seq, logical) {
                keys.push(k);
            }
        }
        // `drop_slot` already resets slots/lanes/lru/epoch per logical; the seq
        // is now byte-for-byte a fresh `SeqLanes` (all Absent, no lanes held).
        keys
    }

    /// Drop a logical slot (ring `truncate_after` / seq teardown): bump its epoch
    /// so any in-flight spill cancels on commit, free its lane if resident/
    /// spilling, and return its cold key so the pool can `store.remove` it.
    pub(crate) fn drop_slot(&mut self, seq: usize, logical: usize) -> Option<u64> {
        let key = self.cold_key(seq, logical);
        let had_cold = matches!(
            self.seqs[seq].slots[logical],
            Residency::Cold { .. } | Residency::Spilling { .. }
        );
        if let Residency::Resident { lane } | Residency::Spilling { lane, .. } =
            self.seqs[seq].slots[logical]
        {
            self.seqs[seq].lane_occupant[lane] = None;
        }
        self.seqs[seq].epoch[logical] += 1;
        self.seqs[seq].slots[logical] = Residency::Absent;
        self.seqs[seq].lru.retain(|&l| l != logical);
        had_cold.then_some(key)
    }
}

/// Deterministic sequence → spill-worker mapping for the multi-worker decode-spill
/// pool. Coherence rests on this being a pure function of `seq` ONLY: because
/// [`DecodeRingManager::cold_key`] is a pure function of `(seq, logical)`, routing
/// every job for a given sequence to ONE worker keeps all same-`cold_key` spill
/// incarnations (epoch e, e+1, …) FIFO on a single channel — byte-reproducing the
/// single-consumer `put → complete_spill → remove` order the epoch guard was
/// designed around. NEVER route by arrival/round-robin/logical: that would let a
/// stale-epoch completion `store.remove` a freshly committed blob under the shared
/// key. A splitmix step de-clusters the low bits of `seq` so `n` workers get an
/// even share (raw `seq % n` is equally correct — the only invariant is that the
/// index depends on `seq` and `n` alone). `n == 1` always returns 0 (byte-identical
/// to the single-worker build).
pub(crate) fn spill_worker_index(seq: usize, n: usize) -> usize {
    debug_assert!(n >= 1, "worker pool must have at least one worker");
    if n <= 1 {
        return 0;
    }
    // SSOT: `mix64(seq, 0)` IS the bare splitmix64 finalizer over `seq` (its
    // pre-mix `a ^ 0·GOLDEN` is the identity) — value-identical to the old
    // hand-rolled mixer, pinned in `decode_ring_manager_key_tests.rs`.
    (atlas_tier::hash::mix64(seq as u64, 0) % n as u64) as usize
}

#[cfg(test)]
#[path = "decode_ring_manager_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "decode_ring_manager_key_tests.rs"]
mod key_tests;
