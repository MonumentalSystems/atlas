// SPDX-License-Identifier: AGPL-3.0-only

// Codec round-trip / validation tests, moved WITH the codecs from
// `spark-storage/src/expert_peer.rs` and `cache_peer.rs` (RailSet extraction,
// Step B). The exact byte layouts are pinned separately by the golden vectors
// in `spark-storage/tests/rdma_wire_golden.rs` (committed in Step A against
// the pre-move codecs) — these tests cover reader/writer agreement and the
// validation bails. Un-gated: pure `std::io`, runs on the ATLAS_SKIP_BUILD path.

use atlas_rdma::wire::{
    CacheServerParams, VerbsClientParams, VerbsServerParams, read_server_rails, write_server_rails,
};

#[test]
fn verbs_server_params_round_trip() {
    let sp = VerbsServerParams {
        qpn: 0x1234,
        psn: 0x00ab_cdef & 0xff_ffff,
        gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 178, 12],
        layers: vec![(0x7f00_0000_0000, 1001), (0x7f00_0100_0000, 1002)],
    };
    let mut buf = Vec::new();
    sp.write_to(&mut buf).unwrap();
    let back = VerbsServerParams::read_from(&mut &buf[..]).unwrap();
    assert_eq!(sp, back);
}

#[test]
fn verbs_client_params_round_trip() {
    let cp = VerbsClientParams {
        qpn: 0x9999,
        psn: 0x0055_5555,
        gid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    };
    let mut buf = Vec::new();
    cp.write_to(&mut buf).unwrap();
    let back = VerbsClientParams::read_from(&mut &buf[..]).unwrap();
    assert_eq!(cp, back);
}

#[test]
fn server_rails_round_trip() {
    // The dual-rail framing: N `VerbsServerParams` with a leading count.
    let mk = |qpn| VerbsServerParams {
        qpn,
        psn: 0x0012_3456 & 0xff_ffff,
        gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 178, 12],
        // Same base VA across rails (shared pages), distinct rkeys per rail.
        layers: vec![
            (0x7f00_0000_0000, 1001 + qpn),
            (0x7f00_0100_0000, 1002 + qpn),
        ],
    };
    let rails = vec![mk(0x1111), mk(0x2222)];
    let mut buf = Vec::new();
    write_server_rails(&mut buf, &rails).unwrap();
    let back = read_server_rails(&mut &buf[..], 2).unwrap();
    assert_eq!(rails, back);
}

#[test]
fn single_rail_framing_round_trips() {
    // n == 1 is the default path; must round-trip through the framing.
    let sp = VerbsServerParams {
        qpn: 7,
        psn: 9,
        gid: [1u8; 16],
        layers: vec![(0x1000, 42)],
    };
    let mut buf = Vec::new();
    write_server_rails(&mut buf, std::slice::from_ref(&sp)).unwrap();
    // Leading count byte then the single params struct.
    assert_eq!(buf[0], 1);
    let back = read_server_rails(&mut &buf[..], 1).unwrap();
    assert_eq!(back, vec![sp]);
}

#[test]
fn read_server_rails_rejects_mismatch() {
    // Framed count (1) != what the caller negotiated (2) -> protocol error.
    let sp = VerbsServerParams {
        qpn: 1,
        psn: 2,
        gid: [0u8; 16],
        layers: vec![(0x1000, 7)],
    };
    let mut buf = Vec::new();
    write_server_rails(&mut buf, std::slice::from_ref(&sp)).unwrap();
    assert!(read_server_rails(&mut &buf[..], 2).is_err());
}

#[test]
fn read_server_rails_rejects_zero_count() {
    // A zero rail count is a corrupt/hostile frame.
    let buf = [0u8; 1];
    assert!(read_server_rails(&mut &buf[..], 1).is_err());
}

#[test]
fn write_server_rails_rejects_empty() {
    let mut buf = Vec::new();
    assert!(write_server_rails(&mut buf, &[]).is_err());
}

#[test]
fn verbs_server_params_reject_absurd_layer_count() {
    // A corrupt/hostile count must Err, not attempt a huge allocation.
    let mut buf = Vec::new();
    buf.extend_from_slice(&1u32.to_le_bytes()); // qpn
    buf.extend_from_slice(&2u32.to_le_bytes()); // psn
    buf.extend_from_slice(&[0u8; 16]); // gid
    buf.extend_from_slice(&99_999u32.to_le_bytes()); // n_layers (absurd)
    assert!(VerbsServerParams::read_from(&mut &buf[..]).is_err());
}

#[test]
fn kv_server_params_round_trip() {
    let sp = CacheServerParams {
        qpn: 0x4242,
        psn: 0x0012_3456 & 0xff_ffff,
        gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 178, 12],
        base_addr: 0x7f00_1234_0000,
        rkey: 0xdead_beef,
    };
    let mut buf = Vec::new();
    sp.write_to(&mut buf).unwrap();
    assert_eq!(CacheServerParams::read_from(&mut &buf[..]).unwrap(), sp);
}
