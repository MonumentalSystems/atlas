// SPDX-License-Identifier: AGPL-3.0-only

//! THE cross-CLIENT decode-tier regression tests — hardware-free.
//!
//! Decode cold keys are SLOT COORDINATES: `DecodeRingManager::cold_key(seq,
//! logical)` hashes client-LOCAL scheduler/ring indices plus the process- and
//! model-invariant `DECODE_DOMAIN`, so two same-model processes derive
//! byte-identical keys for unrelated recurrent states. The per-model
//! fingerprint namespace fixed cross-MODEL isolation but not cross-CLIENT:
//! on one shared paging peer, client B could fault (read) or remove (delete)
//! client A's rollback blobs under the same wire key — silent corruption plus
//! spurious victim-side "cold MISS on live target" crashes. The fix is the
//! per-process client salt folded by `derive_decode_ns_salted`; these tests
//! drive two `PagingSnapshotStore`s over ONE shared mock peer and pin BOTH
//! directions: distinct salts isolate (get AND remove), the unsalted/equal
//! namespace collides (the pinned old bug), and the content-keyed Marconi
//! tier keeps sharing (the warm cache is a feature there, not a bug).

use std::num::NonZeroU64;
use std::sync::Arc;

use atlas_core::config::ModelConfig;

use super::super::SnapshotBlobStore;
use super::super::fingerprint::{
    ModelFingerprint, derive_decode_ns_salted, mix64, parse_u64_strict, resolve_decode_ns,
};
use super::PagingSnapshotStore;
use super::paging_isolation_tests::MockPagingPeer;
use crate::model::decode_ring_manager::DecodeRingManager;

const BLOB: usize = 64;
const SALT_A: u64 = 0x0000_0000_0000_0001;
const SALT_B: u64 = 0x0000_0000_0000_0002;

fn fp() -> ModelFingerprint {
    ModelFingerprint::derive_with_id(&ModelConfig::qwen3_next_80b_nvfp4(), BLOB, "").unwrap()
}

fn store(peer: &Arc<MockPagingPeer>, ns: NonZeroU64) -> PagingSnapshotStore {
    PagingSnapshotStore::new(Box::new(peer.clone()), BLOB, ns)
}

/// The bug's precondition, reproduced with the REAL manager: two same-model
/// clients (each its own `DecodeRingManager`, both on the production
/// `DECODE_DOMAIN` manager namespace) derive the SAME cold key for the same
/// client-local slot coordinate.
fn shared_cold_key() -> u64 {
    let a = DecodeRingManager::new(8, 2, 1, 4, 4, atlas_kernels::DECODE_DOMAIN);
    let b = DecodeRingManager::new(8, 2, 1, 4, 4, atlas_kernels::DECODE_DOMAIN);
    let k = a.cold_key(3, 2);
    assert_eq!(
        k,
        b.cold_key(3, 2),
        "cold_key is a pure slot coordinate — identical across clients by construction"
    );
    k
}

/// The pre-salt derived namespace (what `resolve_decode_ns` used to return).
fn unsalted_ns(f: ModelFingerprint) -> NonZeroU64 {
    NonZeroU64::new(mix64(f.get(), atlas_kernels::DECODE_DOMAIN)).unwrap()
}

// ── T1 (the headline regression): distinct client salts isolate ──────────
// MUST call the SHIPPED `derive_decode_ns_salted` — do NOT "simplify" this to
// a test-local ns computation: revert-sensitivity depends on the production
// fold. If the client-salt fold is removed, ns_a == ns_b and this test FAILS
// (that failure is the guard working).

#[test]
fn two_clients_same_model_distinct_salts_do_not_cross_serve() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let f = fp();
    let a = store(&peer, derive_decode_ns_salted(f.get(), SALT_A));
    let b = store(&peer, derive_decode_ns_salted(f.get(), SALT_B));
    let k = shared_cold_key();
    a.put(k, &[0xAA; BLOB]).unwrap();
    let mut out = [0u8; BLOB];
    assert!(
        !b.get(k, &mut out).unwrap(),
        "client B must MISS client A's rollback blob for the same slot coordinate"
    );
    assert_eq!(out, [0u8; BLOB], "a miss must leave `out` untouched");
    // …while client A still hits its own blob (isolation, not amnesia).
    assert!(a.get(k, &mut out).unwrap());
    assert_eq!(out, [0xAA; BLOB]);
}

/// The env path folds the per-process salt: `resolve_decode_ns` must NOT
/// return the bare `mix64(fp, DECODE_DOMAIN)` any more. Fails if the salt is
/// reverted at the `resolve_decode_ns` seam (even with the salted core fn
/// left in place). Assumes ATLAS_SSM_DECODE_NS is unset in the test env (the
/// suite never sets it; process-global env is not touched here).
#[test]
fn resolve_decode_ns_folds_the_per_process_client_salt() {
    let f = fp();
    let ns = resolve_decode_ns(f).unwrap();
    assert_ne!(
        ns,
        unsalted_ns(f),
        "resolve_decode_ns returned the UNSALTED namespace — cross-client decode isolation \
         is gone (p = 2^-64 false alarm from a randomly-colliding salt)"
    );
    // And deterministic within the process (OnceLock salt): every decode
    // store this process builds folds the same salt.
    assert_eq!(resolve_decode_ns(f).unwrap(), ns);
}

// ── T2: pin the OLD bug so it cannot silently return ──────────────────────

#[test]
fn two_clients_same_model_unsalted_share_wire_keys_the_pinned_bug() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let f = fp();
    let a = store(&peer, unsalted_ns(f));
    let b = store(&peer, unsalted_ns(f));
    let k = shared_cold_key();
    assert_eq!(
        a.wire(k),
        b.wire(k),
        "same model, no salt ⇒ same wire key — the pinned cross-client bug"
    );
    a.put(k, &[0xAA; BLOB]).unwrap();
    let mut out = [0u8; BLOB];
    assert!(
        b.get(k, &mut out).unwrap(),
        "equal namespaces DO collide — this is the pinned bug"
    );
    assert_eq!(
        out, [0xAA; BLOB],
        "client B silently served client A's recurrent state as a HIT"
    );
}

// ── T3: the DELETE hazard, both directions ────────────────────────────────
// `complete_spill` Cancelled and `drop_slot` both `store.remove` by cold key;
// cross-client that deletes the VICTIM's live blob → a later rollback fault
// hard-bails ("cold MISS on live target"). A wrong-bytes assertion alone
// would miss this interleaving.

#[test]
fn cross_client_remove_unsalted_deletes_the_victims_live_blob() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let f = fp();
    let a = store(&peer, unsalted_ns(f));
    let b = store(&peer, unsalted_ns(f));
    let k = shared_cold_key();
    a.put(k, &[0xAA; BLOB]).unwrap();
    b.remove(k); // e.g. B's drop_slot / cancelled-spill cleanup
    let mut out = [0u8; BLOB];
    assert!(
        !a.get(k, &mut out).unwrap(),
        "pinned: B's remove evicted A's LIVE rollback target (victim-side fatal MISS)"
    );
}

#[test]
fn cross_client_remove_salted_is_client_scoped() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let f = fp();
    let a = store(&peer, derive_decode_ns_salted(f.get(), SALT_A));
    let b = store(&peer, derive_decode_ns_salted(f.get(), SALT_B));
    let k = shared_cold_key();
    a.put(k, &[0xAA; BLOB]).unwrap();
    b.remove(k); // must not touch A's blob
    let mut out = [0u8; BLOB];
    assert!(a.get(k, &mut out).unwrap(), "B's remove must not evict A");
    assert_eq!(out, [0xAA; BLOB]);
}

// ── T4: same pinned salt round-trips (the reproduction path works) ────────
// ATLAS_SSM_DECODE_CLIENT_ID pins the salt; a restart-in-place (same pin)
// finds its own keys, and the hardware harness's same-salt control hits.

#[test]
fn same_client_salt_round_trips_across_store_instances() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let f = fp();
    let ns = derive_decode_ns_salted(f.get(), SALT_A);
    let c1 = store(&peer, ns);
    let c2 = store(&peer, derive_decode_ns_salted(f.get(), SALT_A));
    let k = shared_cold_key();
    let blob: Vec<u8> = (0..BLOB as u8).collect();
    c1.put(k, &blob).unwrap();
    let mut out = vec![0u8; BLOB];
    assert!(
        c2.get(k, &mut out).unwrap(),
        "same model + same pinned salt: the deliberate shared escape works"
    );
    assert_eq!(out, blob, "bit-identical restore");
}

// ── T5: the Marconi/swap tier still SHARES across clients ─────────────────
// Marconi keys are CONTENT hashes (prefix_hash over tokens), so two same-
// model clients SHOULD hit each other's entries — that is the warm cache
// working. The client salt must never leak into the swap namespace (`fp`
// bare, per `resolve_swap_ns`).

#[test]
fn marconi_swap_tier_still_shares_across_clients() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let f = fp();
    let c1 = store(&peer, f.nonzero()); // swap ns = the fingerprint itself
    let c2 = store(&peer, f.nonzero());
    const PREFIX_HASH: u64 = 0x5EED_F00D_CAFE_D00D; // content key, not a coordinate
    let blob: Vec<u8> = (0..BLOB as u8).rev().collect();
    c1.put(PREFIX_HASH, &blob).unwrap();
    let mut out = vec![0u8; BLOB];
    assert!(
        c2.get(PREFIX_HASH, &mut out).unwrap(),
        "content-keyed Marconi warm hit across clients must keep working"
    );
    assert_eq!(out, blob);
    // And the salted decode namespaces stay off the swap namespace entirely.
    for salt in [SALT_A, SALT_B] {
        assert_ne!(derive_decode_ns_salted(f.get(), salt), f.nonzero());
    }
}

// ── T6: salted-namespace properties (env-free core) ───────────────────────

#[test]
fn salted_decode_ns_properties() {
    let f = fp().get();
    let base = mix64(f, atlas_kernels::DECODE_DOMAIN);
    for salt in [0u64, SALT_A, SALT_B, u64::MAX] {
        let ns = derive_decode_ns_salted(f, salt);
        // Deterministic per (fp, salt).
        assert_eq!(ns, derive_decode_ns_salted(f, salt));
        // Never the Marconi swap ns (fp bare), never the model-blind domain
        // constant, never the pre-salt derived value.
        assert_ne!(ns.get(), f, "aliased the Marconi swap namespace");
        assert_ne!(ns.get(), atlas_kernels::DECODE_DOMAIN, "model-blind");
        assert_ne!(ns.get(), base, "salt {salt:#x} did not rotate the ns");
    }
    // Distinct salts ⇒ distinct namespaces; distinct models ⇒ distinct too.
    assert_ne!(
        derive_decode_ns_salted(f, SALT_A),
        derive_decode_ns_salted(f, SALT_B)
    );
    assert_ne!(
        derive_decode_ns_salted(f, SALT_A),
        derive_decode_ns_salted(f ^ 1, SALT_A)
    );
}

#[test]
fn client_id_parses_strictly_and_zero_is_permitted() {
    // 0 is key material here, not a sentinel (contrast parse_ns).
    assert_eq!(parse_u64_strict("V", "0").unwrap(), 0);
    assert_eq!(parse_u64_strict("V", "42").unwrap(), 42);
    assert_eq!(parse_u64_strict("V", "0x2A").unwrap(), 42);
    assert_eq!(parse_u64_strict("V", " 0X2a ").unwrap(), 42);
    // Junk/overflow is a startup ERROR, never a silent random fallback (PCND).
    assert!(parse_u64_strict("V", "banana").is_err());
    assert!(parse_u64_strict("V", "-1").is_err());
    assert!(parse_u64_strict("V", "18446744073709551616").is_err());
}
