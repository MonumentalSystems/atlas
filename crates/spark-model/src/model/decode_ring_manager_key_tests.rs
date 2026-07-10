// SPDX-License-Identifier: AGPL-3.0-only

//! SSOT equivalence pins for `cold_key` and `spill_worker_index`.
//!
//! Both were the LAST hand-transcriptions of the splitmix64 finalizer
//! constants (`0xBF58_476D_1CE4_E5B9` / `0x94D0_49BB_1331_11EB`) outside
//! `atlas_tier::hash`. They now route through the one true `mix64` via
//! operand regrouping: `mix64(a, b) = finalize(a ^ b·GOLDEN)`, so
//! `cold_key(seq, logical) == mix64(logical·P2 ^ ns, seq)` and the worker
//! mixer `== mix64(seq, 0)`. These tests freeze both equivalences against a
//! verbatim copy of the OLD hand-rolled folds, so the SSOT rewrite can never
//! silently drift. Decode keys are ephemeral and MAY rotate — but a rotation
//! must be a deliberate change that updates these references, never a
//! refactor accident.

use super::spill_worker_index;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F; // xxHash64 prime-2 (`LOGICAL_SPREAD`)

/// The OLD hand-rolled three-input fold, verbatim (pre-SSOT reference).
fn old_cold_key(seq: usize, logical: usize, namespace: u64) -> u64 {
    let mut h = (seq as u64).wrapping_mul(GOLDEN) ^ (logical as u64).wrapping_mul(P2) ^ namespace;
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// The OLD hand-rolled worker mixer, verbatim (pre-SSOT reference).
fn old_spill_worker_mixer(seq: usize) -> u64 {
    let mut h = seq as u64;
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}

fn mgr_with_ns(ns: u64) -> super::DecodeRingManager {
    // Geometry is irrelevant to the key fold; keep the fixture tiny.
    super::DecodeRingManager::new(8, 2, 1, 4, /*max_seqs*/ 2, ns)
}

#[test]
fn cold_key_is_value_identical_to_the_old_hand_rolled_fold() {
    let namespaces = [
        atlas_kernels::DECODE_DOMAIN,
        0xABCD,
        0,
        u64::MAX,
        0x1234_5678_9ABC_DEF0,
    ];
    for &ns in &namespaces {
        let m = mgr_with_ns(ns);
        for seq in 0..64usize {
            for logical in 0..64usize {
                assert_eq!(
                    m.cold_key(seq, logical),
                    old_cold_key(seq, logical, ns),
                    "cold_key drifted at (seq={seq}, logical={logical}, ns={ns:#x})"
                );
            }
        }
        // Extremes: `cold_key` never indexes by seq/logical, only mixes them.
        for &(seq, logical) in &[(usize::MAX, usize::MAX), (0, usize::MAX), (usize::MAX, 0)] {
            assert_eq!(
                m.cold_key(seq, logical),
                old_cold_key(seq, logical, ns),
                "cold_key drifted at extreme (seq={seq}, logical={logical}, ns={ns:#x})"
            );
        }
    }
}

#[test]
fn spill_worker_index_is_value_identical_to_the_old_mixer() {
    for &n in &[2usize, 3, 4, 8, 17] {
        for seq in 0..1000usize {
            assert_eq!(
                spill_worker_index(seq, n),
                (old_spill_worker_mixer(seq) % n as u64) as usize,
                "spill_worker_index drifted at (seq={seq}, n={n})"
            );
        }
        assert_eq!(
            spill_worker_index(usize::MAX, n),
            (old_spill_worker_mixer(usize::MAX) % n as u64) as usize,
        );
    }
    // n == 1 stays the early-return 0 (byte-identical single-worker build).
    assert_eq!(spill_worker_index(12345, 1), 0);
}
