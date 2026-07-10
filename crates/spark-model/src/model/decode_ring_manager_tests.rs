// SPDX-License-Identifier: AGPL-3.0-only

//! Residency state-machine tests for [`DecodeRingManager`] — extracted from
//! the inline `mod tests` per the repo convention (tests in their own files;
//! this also brings `decode_ring_manager.rs` back under the 500 LoC cap).

use super::*;

// hot_lanes=2, margin=1 → l_phys=3; ring=8; scratch=4.
fn mgr() -> DecodeRingManager {
    DecodeRingManager::new(8, 2, 1, 4, /*max_seqs*/ 2, /*ns*/ 0xABCD)
}

#[test]
fn cold_key_namespacing_is_domain_separated_and_deterministic() {
    let m = mgr();
    // Deterministic.
    assert_eq!(m.cold_key(0, 3), m.cold_key(0, 3));
    // Distinct per (seq, logical).
    assert_ne!(m.cold_key(0, 3), m.cold_key(1, 3));
    assert_ne!(m.cold_key(0, 3), m.cold_key(0, 4));
    // Namespace changes the key space (no collision with a ns=0 store).
    let m2 = DecodeRingManager::new(8, 2, 1, 4, 2, 0);
    assert_ne!(m.cold_key(0, 3), m2.cold_key(0, 3));
}

#[test]
fn frame_layout_is_contiguous_then_scratch() {
    let m = mgr();
    assert_eq!(m.lane_frame(0, 0), 0);
    assert_eq!(m.lane_frame(0, 2), 2);
    assert_eq!(m.lane_frame(1, 0), 3); // seq1 starts after seq0's 3 lanes
    assert_eq!(m.scratch_frame(0), 2 * 3); // after all seq lanes
    assert_eq!(m.total_frames(), 2 * 3 + 4);
}

#[test]
fn first_two_saves_stay_hot_no_spill() {
    let mut m = mgr();
    let d0 = m.plan_save(0, 0);
    assert!(matches!(d0, SaveDecision::Fresh { spill: None, .. }));
    let d1 = m.plan_save(0, 1);
    assert!(matches!(d1, SaveDecision::Fresh { spill: None, .. }));
    assert!(matches!(m.residency(0, 0), Residency::Resident { .. }));
    assert!(matches!(m.residency(0, 1), Residency::Resident { .. }));
}

#[test]
fn third_distinct_save_spills_the_lru() {
    let mut m = mgr();
    m.plan_save(0, 0); // resident, LRU
    m.plan_save(0, 1); // resident
    let d2 = m.plan_save(0, 2); // exceeds hot=2 → evict LRU (slot 0)
    let SaveDecision::Fresh {
        spill: Some(req), ..
    } = d2
    else {
        panic!("expected a spill of the LRU slot, got {d2:?}");
    };
    assert_eq!(req.logical, 0, "LRU (slot 0) is the spill victim");
    assert!(matches!(m.residency(0, 0), Residency::Spilling { .. }));
    assert!(matches!(m.residency(0, 2), Residency::Resident { .. }));
    assert_eq!(req.cold_key, m.cold_key(0, 0));
}

#[test]
fn in_place_overwrite_of_hot_slot_no_spill() {
    let mut m = mgr();
    m.plan_save(0, 0);
    let again = m.plan_save(0, 0);
    assert!(matches!(again, SaveDecision::InPlace { .. }));
}

#[test]
fn spill_then_commit_makes_cold_and_frees_lane() {
    let mut m = mgr();
    m.plan_save(0, 0);
    m.plan_save(0, 1);
    let SaveDecision::Fresh {
        spill: Some(req), ..
    } = m.plan_save(0, 2)
    else {
        panic!("expected spill");
    };
    assert_eq!(
        m.complete_spill(req.seq, req.logical, req.epoch),
        SpillCommit::Committed
    );
    assert!(matches!(m.residency(0, 0), Residency::Cold { .. }));
    // The freed lane is now reusable: a 4th distinct save finds a free lane.
    let d3 = m.plan_save(0, 3);
    assert!(
        matches!(d3, SaveDecision::Fresh { .. }),
        "freed lane reused, got {d3:?}"
    );
}

#[test]
fn restore_of_spilling_slot_reads_the_pinned_lane_no_fault() {
    // The subtlest invariant: a rollback landing between spill-enqueue and
    // completion restores DIRECTLY from the pinned lane — no fault, no wait.
    let mut m = mgr();
    m.plan_save(0, 0);
    m.plan_save(0, 1);
    let SaveDecision::Fresh {
        spill: Some(req), ..
    } = m.plan_save(0, 2)
    else {
        panic!("expected spill");
    };
    // Slot 0 is now Spilling but NOT yet committed.
    assert!(matches!(m.residency(0, 0), Residency::Spilling { .. }));
    match m.plan_restore(0, 0) {
        RestoreDecision::FromLane { lane } => assert_eq!(lane, req.lane),
        other => panic!("Spilling slot must restore from its pinned lane, got {other:?}"),
    }
}

#[test]
fn restore_of_cold_slot_faults_before_read() {
    let mut m = mgr();
    m.plan_save(0, 0);
    m.plan_save(0, 1);
    let SaveDecision::Fresh {
        spill: Some(req), ..
    } = m.plan_save(0, 2)
    else {
        panic!("expected spill");
    };
    m.complete_spill(req.seq, req.logical, req.epoch); // slot 0 now Cold
    match m.plan_restore(0, 0) {
        RestoreDecision::FaultThenRestore { cold_key, .. } => {
            assert_eq!(cold_key, m.cold_key(0, 0));
        }
        other => panic!("Cold slot must fault before restore, got {other:?}"),
    }
}

#[test]
fn spill_completing_after_overwrite_is_cancelled_not_stale_commit() {
    // The epoch guard: slot 0 spills, then is re-saved (fresh incarnation)
    // BEFORE the spill's put returns. The late commit must Cancel + remove,
    // never clobber the fresh incarnation with a stale Cold. margin=2 leaves a
    // free lane so the re-save takes the Fresh (epoch-bumping) path rather than
    // backpressuring (which would leave the spill legitimately valid).
    let mut m = DecodeRingManager::new(8, /*hot*/ 2, /*margin*/ 2, 4, 2, 0xABCD);
    m.plan_save(0, 0);
    m.plan_save(0, 1);
    let SaveDecision::Fresh {
        spill: Some(old), ..
    } = m.plan_save(0, 2)
    else {
        panic!("expected spill of slot 0");
    };
    assert_eq!(old.logical, 0);
    assert!(matches!(m.residency(0, 0), Residency::Spilling { .. }));
    // Re-save slot 0 while it is still Spilling — new incarnation, epoch bumps.
    let redo = m.plan_save(0, 0);
    assert!(
        matches!(redo, SaveDecision::Fresh { .. }),
        "free lane → Fresh, got {redo:?}"
    );
    assert!(matches!(m.residency(0, 0), Residency::Resident { .. }));
    // The OLD spill (old.epoch) now completes late — must Cancel, not Commit.
    match m.complete_spill(old.seq, old.logical, old.epoch) {
        SpillCommit::Cancelled { remove_cold_key } => {
            assert_eq!(remove_cold_key, m.cold_key(0, 0));
        }
        SpillCommit::Committed => panic!("stale-epoch spill must NOT commit Cold"),
    }
    // And slot 0's fresh incarnation is untouched (still Resident).
    assert!(matches!(m.residency(0, 0), Residency::Resident { .. }));
}

#[test]
fn drop_slot_frees_lane_bumps_epoch_and_cancels_inflight() {
    let mut m = mgr();
    m.plan_save(0, 0);
    m.plan_save(0, 1);
    let SaveDecision::Fresh {
        spill: Some(req), ..
    } = m.plan_save(0, 2)
    else {
        panic!("expected spill");
    };
    // Drop the still-spilling slot 0 (a truncate_after in the tail).
    let removed = m.drop_slot(0, 0);
    assert_eq!(
        removed,
        Some(m.cold_key(0, 0)),
        "spilling slot's cold key returned for removal"
    );
    assert_eq!(m.residency(0, 0), Residency::Absent);
    // The now-late spill commit cancels (epoch bumped by drop).
    assert!(matches!(
        m.complete_spill(req.seq, req.logical, req.epoch),
        SpillCommit::Cancelled { .. }
    ));
}

#[test]
fn reset_seq_clears_stale_state_for_a_recycled_slot() {
    // Sequence A fills the ring (some Resident, one spilled Cold), then the
    // seq-slot is recycled by sequence B → reset_seq must hand back a fresh
    // SeqLanes so B's first save doesn't inherit A's lanes/LRU/residency.
    let mut m = mgr(); // ring=8, hot=2, l_phys=3
    m.plan_save(0, 0);
    m.plan_save(0, 1);
    let SaveDecision::Fresh {
        spill: Some(req), ..
    } = m.plan_save(0, 2)
    else {
        panic!("expected spill");
    };
    m.complete_spill(req.seq, req.logical, req.epoch); // slot 0 Cold
    // seq 0 now holds lanes + a cold blob.
    let keys = m.reset_seq(0);
    assert_eq!(
        keys,
        vec![m.cold_key(0, 0)],
        "the Cold slot's key is returned for removal"
    );
    for lg in 0..8 {
        assert_eq!(m.residency(0, lg), Residency::Absent, "all slots reset");
    }
    // A fresh save on the recycled seq behaves exactly like a brand-new seq:
    // first two stay hot, no spill, no backpressure.
    let d0 = m.plan_save(0, 0);
    assert!(
        matches!(d0, SaveDecision::Fresh { spill: None, .. }),
        "got {d0:?}"
    );
    let d1 = m.plan_save(0, 1);
    assert!(
        matches!(d1, SaveDecision::Fresh { spill: None, .. }),
        "got {d1:?}"
    );
    // And an in-flight spill of A's old incarnation now cancels (epoch bumped).
    assert!(matches!(
        m.complete_spill(req.seq, req.logical, req.epoch),
        SpillCommit::Cancelled { .. }
    ));
}

#[test]
fn backpressure_when_all_lanes_busy_then_recovers() {
    // hot=2, margin=1, l_phys=3. Save 3 distinct slots without ever
    // completing the spill → the drain lane fills and the 4th distinct save
    // must backpressure.
    let mut m = mgr();
    m.plan_save(0, 0);
    m.plan_save(0, 1);
    let s2 = m.plan_save(0, 2); // spills slot0 (Spilling, lane pinned)
    assert!(matches!(s2, SaveDecision::Fresh { spill: Some(_), .. }));
    // Now lanes: slot1 Resident, slot2 Resident, slot0 Spilling → all 3 busy.
    let s3 = m.plan_save(0, 3);
    let SaveDecision::Backpressure { drain } = s3 else {
        panic!("expected backpressure with all lanes busy, got {s3:?}");
    };
    assert_eq!(drain.len(), 1, "one in-flight spill to drain");
    // Pool drains it → lane frees → re-plan succeeds.
    let d = drain[0];
    m.complete_spill(d.seq, d.logical, d.epoch);
    let retry = m.plan_save(0, 3);
    assert!(
        matches!(retry, SaveDecision::Fresh { .. }),
        "recovers after drain, got {retry:?}"
    );
}

#[test]
fn seqs_are_independent() {
    let mut m = mgr();
    m.plan_save(0, 0);
    m.plan_save(0, 1);
    m.plan_save(0, 2); // spill in seq 0
    // seq 1 untouched.
    assert_eq!(m.residency(1, 0), Residency::Absent);
    let d = m.plan_save(1, 0);
    assert!(matches!(d, SaveDecision::Fresh { spill: None, .. }));
}

// ── Multi-worker spill-pool dispatch (GPU-free). The coherence proof is that
// the worker index depends ONLY on `seq`, so every job sharing a `cold_key`
// (which is a pure fn of `(seq, logical)`) lands on one worker's FIFO channel.

#[test]
fn spill_worker_index_n1_is_always_zero() {
    // N == 1 is byte-identical to the single-worker build.
    for seq in 0..64 {
        assert_eq!(spill_worker_index(seq, 1), 0);
    }
}

#[test]
fn spill_worker_index_depends_only_on_seq() {
    // For any pool size, the index is a stable pure function of `seq`; it does
    // NOT depend on logical/epoch (those aren't even inputs) and repeated calls
    // agree. Different seqs may (and generally do) map elsewhere.
    for &n in &[1usize, 2, 4, 8] {
        for seq in 0..128 {
            let w = spill_worker_index(seq, n);
            assert!(w < n, "index {w} out of range for n={n}");
            assert_eq!(w, spill_worker_index(seq, n), "must be deterministic");
        }
        if n > 1 {
            // Sanity: the mapping is not degenerate (not all seqs to worker 0).
            let distinct: std::collections::HashSet<usize> =
                (0..128).map(|s| spill_worker_index(s, n)).collect();
            assert!(
                distinct.len() > 1,
                "n={n} collapsed every seq onto one worker"
            );
        }
    }
}

#[test]
fn same_seq_multi_incarnation_jobs_route_to_one_worker() {
    // Reproduce the headline same-cold_key hazard shape: (seq,logical) spills
    // at epoch e1, is re-saved (fresh incarnation, epoch e2) then evicted again
    // → two SpillReqs sharing ONE cold_key but different epochs. Both MUST route
    // to the same worker so the pool preserves the single-consumer FIFO
    // put→commit→remove order (else a stale-epoch remove erases the fresh blob).
    let mut m = DecodeRingManager::new(8, /*hot*/ 2, /*margin*/ 2, 4, 2, 0xABCD);
    m.plan_save(0, 0);
    m.plan_save(0, 1);
    // First incarnation of slot 0 is evicted → a real SpillReq at epoch e1.
    let SaveDecision::Fresh {
        spill: Some(e1), ..
    } = m.plan_save(0, 2)
    else {
        panic!("expected first spill of slot 0");
    };
    assert_eq!(e1.logical, 0);
    // The SECOND incarnation's spill: same (seq, logical) ⇒ SAME cold_key, a
    // bumped epoch (this is exactly the shape `plan_save`'s epoch bump + a later
    // eviction produce; constructed directly to avoid depending on LRU victim
    // order). This is the headline hazard: two SpillReqs on ONE cold_key.
    let e2 = SpillReq {
        seq: e1.seq,
        logical: e1.logical,
        lane: e1.lane,
        epoch: e1.epoch + 1,
        cold_key: m.cold_key(0, 0),
    };
    assert_eq!(
        e1.cold_key, e2.cold_key,
        "same (seq,logical) ⇒ shared cold_key"
    );
    assert_ne!(
        e1.epoch, e2.epoch,
        "distinct incarnations ⇒ distinct epochs"
    );
    // Coherence: both same-key jobs route to ONE worker for every pool size, so
    // the pool reproduces the single-consumer FIFO put→commit→remove order.
    for &n in &[1usize, 2, 4, 8] {
        assert_eq!(
            spill_worker_index(e1.seq, n),
            spill_worker_index(e2.seq, n),
            "same-key incarnations must share a worker (n={n})",
        );
    }
}

#[test]
fn higher_margin_absorbs_more_inflight_before_backpressure() {
    // Pure-logic proof that the runtime margin is the backpressure lever the
    // worker pool pairs with: margin=1 (l_phys=3) backpressures on the 4th
    // distinct uncompleted save; margin=2 (l_phys=4) defers it by exactly one.
    let mut m1 = DecodeRingManager::new(8, 2, /*margin*/ 1, 4, 2, 0xABCD);
    m1.plan_save(0, 0);
    m1.plan_save(0, 1);
    assert!(matches!(
        m1.plan_save(0, 2),
        SaveDecision::Fresh { spill: Some(_), .. }
    ));
    assert!(
        matches!(m1.plan_save(0, 3), SaveDecision::Backpressure { .. }),
        "margin=1 backpressures on the 4th distinct uncompleted save",
    );

    let mut m2 = DecodeRingManager::new(8, 2, /*margin*/ 2, 4, 2, 0xABCD);
    m2.plan_save(0, 0);
    m2.plan_save(0, 1);
    assert!(matches!(
        m2.plan_save(0, 2),
        SaveDecision::Fresh { spill: Some(_), .. }
    ));
    assert!(
        matches!(m2.plan_save(0, 3), SaveDecision::Fresh { .. }),
        "margin=2 has a free drain lane → no backpressure on the 4th",
    );
    assert!(
        matches!(m2.plan_save(0, 4), SaveDecision::Backpressure { .. }),
        "margin=2 backpressures one save later than margin=1",
    );
}
