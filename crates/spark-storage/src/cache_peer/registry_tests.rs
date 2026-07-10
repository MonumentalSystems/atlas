// SPDX-License-Identifier: AGPL-3.0-only
//
// `carve_disk_slots` precedence pins — moved to their own file (SDD test
// convention) when `cache_peer.rs` split into `cache_peer/`; body unchanged.

use super::carve_disk_slots;

#[test]
fn carve_disk_slots_precedence() {
    let bb = 4u64; // tiny blob
    // Per-kind override: fixed budget, shared remainder UNTOUCHED (no starve).
    let (slots, rem) = carve_disk_slots(Some(40), 100, 100, bb);
    assert_eq!(slots, 10);
    assert_eq!(
        rem, 100,
        "per-kind override must not consume the shared remainder"
    );
    // Per-kind 0 = unbounded for that kind, remainder untouched.
    assert_eq!(carve_disk_slots(Some(0), 100, 100, bb), (0, 100));
    // No override, shared cap set: claim the remainder (and it drops).
    let (slots, rem) = carve_disk_slots(None, 100, 100, bb);
    assert_eq!(slots, 25);
    assert_eq!(rem, 0, "shared carve consumes the remainder");
    // No override, shared cap 0 = unbounded.
    assert_eq!(carve_disk_slots(None, 0, 0, bb), (0, 0));
    // Starved shared remainder still floors at 1 record (never 0=unbounded).
    assert_eq!(carve_disk_slots(None, 100, 0, bb), (1, 0));
}
