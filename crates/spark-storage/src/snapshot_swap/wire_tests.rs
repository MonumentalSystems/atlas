// SPDX-License-Identifier: AGPL-3.0-only

//! Protocol tests: header parse, stripe plan, wire-golden bytes,
//! dispatch/loop/codec, connection-scoped pins.
//!
//! The residency-semantics test suite (disk cap, infinite depth, read-pins,
//! reserved-slot pinning, overwrite/remove, size laws) moved to atlas-tier
//! with the core; the peer-specific MmapSlotArena test lives next to
//! `mmap_arena.rs`.

use super::super::{MemSwapStore, VecSlotArena};
use super::*;

type TestResidency = SnapshotResidency<VecSlotArena, MemSwapStore>;

const B: usize = 8; // tiny blob for tests

fn blob(tag: u8) -> Vec<u8> {
    vec![tag; B]
}

/// Client-side helper: alloc → write bytes into the arena slot → commit.
fn put(r: &mut TestResidency, key: u64, tag: u8) {
    let slot = r.alloc(key).unwrap();
    r.arena_mut().write_slot(slot, &blob(tag)).unwrap();
    r.commit(key).unwrap();
}
fn get(r: &mut TestResidency, key: u64) -> Option<Vec<u8>> {
    r.locate(key).unwrap().map(|slot| {
        let mut out = vec![0u8; B];
        r.arena().read_slot(slot, &mut out).unwrap();
        out
    })
}

fn residency(slots: usize) -> TestResidency {
    SnapshotResidency::new(VecSlotArena::new(B, slots), MemSwapStore::new(B)).unwrap()
}

/// The connection-scoped auto-release: `handle_paging_op` pins a GET hit and
/// releases it on the SAME connection's next op — no new opcode, no client
/// change. During the window the slot survives a concurrent ALLOC.
#[test]
fn handle_paging_op_pins_get_and_auto_releases() {
    let mut r = residency(2);
    put(&mut r, 0, 0);
    put(&mut r, 1, 1);
    let mut pinned: Option<u64> = None;

    // Connection A: GET 0 → pinned.
    let reply = handle_paging_op(&mut r, OP_GET, 0, &mut pinned);
    assert!(matches!(reply, PagingReply::Located(_)));
    assert_eq!(pinned, Some(0));
    assert_eq!(r.read_pin_count(0), 1);

    // Concurrent ALLOC (another connection) evicts the unpinned key 1, not 0.
    put(&mut r, 2, 2);
    assert_eq!(
        get(&mut r, 0),
        Some(blob(0)),
        "pinned GET slot survived the ALLOC"
    );

    // Connection A's NEXT op releases the pin (its RDMA read has drained).
    handle_paging_op(&mut r, OP_REMOVE, 99, &mut pinned);
    assert_eq!(pinned, None);
    assert_eq!(r.read_pin_count(0), 0, "pin auto-released on next op");
}

// ─────────────────────────── protocol tests ───────────────────────────

#[test]
// Deliberate constant asserts: these are load-bearing protocol pins (the shared
// port disambiguates on them) and must stay visible runtime test failures.
#[allow(clippy::assertions_on_constants)]
fn magic_above_legacy_range() {
    // A legacy client's total_bytes is validated <= 1<<42; both paging magics
    // must be strictly above so neither is mistaken for a size, and distinct.
    assert!(PAGING_MAGIC > (1u64 << 42));
    assert!(PAGING_MAGIC_V2 > (1u64 << 42));
    assert_ne!(PAGING_MAGIC, PAGING_MAGIC_V2);
}

#[test]
fn stripe_plan_covers_every_byte_once() {
    for (blob, chunk, rails) in [
        (64usize, 16usize, 2usize), // even split
        (70, 16, 2),                // tail remainder
        (64, 16, 1),                // single rail
        (10, 64, 2),                // chunk > blob → one chunk
        (64, 64, 2),                // chunk == blob
        (66846720, 1048576, 2),     // real 66MB SSM blob, 1MiB chunks, dual-rail
    ] {
        let plan = stripe_plan(blob, chunk, rails);
        assert_eq!(plan.len(), rails.max(1));
        // Flatten and assert every byte [0,blob) is covered exactly once.
        let mut covered = vec![0u8; blob];
        for rail in &plan {
            for &(off, len) in rail {
                assert!(
                    len <= chunk && off + len <= blob,
                    "chunk oob {off}+{len}>{blob}"
                );
                for b in &mut covered[off..off + len] {
                    assert_eq!(*b, 0, "byte {off} double-covered");
                    *b = 1;
                }
            }
        }
        assert!(
            covered.iter().all(|&b| b == 1),
            "gap in coverage blob={blob}"
        );
    }
    // Zero blob → empty plan (no chunks).
    assert!(stripe_plan(0, 16, 2).iter().all(|r| r.is_empty()));
}

#[test]
fn paging_header_v1_v2_legacy_and_reject() {
    // v1: no kind byte on the wire → SSM.
    let mut body = Vec::new();
    body.extend_from_slice(&0x1000u64.to_le_bytes()); // arena
    body.extend_from_slice(&0x40u64.to_le_bytes()); // blob
    let mut c = std::io::Cursor::new(body);
    assert_eq!(
        parse_paging_header(PAGING_MAGIC, &mut c).unwrap(),
        Some((PagingKind::SSM, 0x1000, 0x40))
    );
    // v2: [kind][arena][blob].
    let mut body = vec![PagingKind::KV.0];
    body.extend_from_slice(&0x2000u64.to_le_bytes());
    body.extend_from_slice(&0x80u64.to_le_bytes());
    let mut c = std::io::Cursor::new(body);
    assert_eq!(
        parse_paging_header(PAGING_MAGIC_V2, &mut c).unwrap(),
        Some((PagingKind::KV, 0x2000, 0x80))
    );
    // legacy KV bare total_bytes → None (dumb path).
    let mut c = std::io::Cursor::new(Vec::new());
    assert_eq!(parse_paging_header(12345, &mut c).unwrap(), None);
    // unsupported kind (RO tier) → hard error, never a bogus arena.
    let mut body = vec![3u8];
    body.extend_from_slice(&[0u8; 16]);
    let mut c = std::io::Cursor::new(body);
    assert!(parse_paging_header(PAGING_MAGIC_V2, &mut c).is_err());
}

/// WIRE-GOLDEN (verify's ask): freeze the exact v1 handshake byte layout so a
/// future codec edit can't silently shift it under a running fleet peer.
#[test]
fn v1_handshake_wire_golden() {
    // What connect_paging emits for arena=0x1000, blob=0x40, 1 rail:
    //   [PAGING_MAGIC u64 le][arena u64 le][blob u64 le][n_rails u8]
    let mut w = Vec::new();
    w.extend_from_slice(&PAGING_MAGIC.to_le_bytes());
    w.extend_from_slice(&0x1000u64.to_le_bytes());
    w.extend_from_slice(&0x40u64.to_le_bytes());
    w.push(1u8);
    assert_eq!(
        w,
        vec![
            0x01, 0x00, 0x00, 0x00, 0x45, 0x47, 0x41, 0x50, // PAGING_MAGIC LE
            0x00, 0x10, 0, 0, 0, 0, 0, 0, // arena 0x1000
            0x40, 0, 0, 0, 0, 0, 0, 0,    // blob 0x40
            0x01, // n_rails
        ],
        "v1 handshake bytes must never shift while any in-repo client sends them (Step C retires v1)"
    );
}

/// One PUT (alloc→arena write→commit) then GET, driven through `dispatch`,
/// with the caller's RDMA-write emulated by writing the returned slot.
#[test]
fn dispatch_put_then_get_roundtrips() {
    let mut r = residency(4);
    // ALLOC key 7
    let PagingReply::Located(off) = dispatch(&mut r, OP_ALLOC, 7) else {
        panic!("alloc reply")
    };
    let slot = (off as usize) / B;
    // client RDMA-WRITE emulation
    r.arena_mut().write_slot(slot, &blob(0xAB)).unwrap();
    assert_eq!(dispatch(&mut r, OP_COMMIT, 7), PagingReply::Ok);
    // GET key 7
    let PagingReply::Located(goff) = dispatch(&mut r, OP_GET, 7) else {
        panic!("get reply")
    };
    let mut out = vec![0u8; B];
    r.arena().read_slot((goff as usize) / B, &mut out).unwrap();
    assert_eq!(out, blob(0xAB));
    // unknown key → miss
    assert_eq!(dispatch(&mut r, OP_GET, 999), PagingReply::Miss);
}

/// Fake bidirectional stream: scripted input, captured output.
struct Duplex {
    inp: std::io::Cursor<Vec<u8>>,
    out: Vec<u8>,
}
impl Read for Duplex {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        self.inp.read(b)
    }
}
impl Write for Duplex {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.out.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn req(op: u8, key: u64) -> Vec<u8> {
    let mut v = vec![op];
    v.extend_from_slice(&key.to_le_bytes());
    v
}

/// Drive the full `run_paging_loop` over a scripted request stream and
/// assert the reply bytes — the protocol end-to-end (sans RDMA data plane).
#[test]
fn run_paging_loop_scripts_ok() {
    let mut r = residency(4);
    // ALLOC 1 → we need its offset before we can COMMIT meaningfully, but
    // run_paging_loop reads all input at once; emulate the client by first
    // allocating via dispatch to learn the slot, writing bytes, THEN scripting
    // commit+get through the loop. (Mirrors real client ordering.)
    let PagingReply::Located(off) = dispatch(&mut r, OP_ALLOC, 1) else {
        panic!()
    };
    r.arena_mut()
        .write_slot((off as usize) / B, &blob(0x5A))
        .unwrap();

    let mut script = Vec::new();
    script.extend(req(OP_COMMIT, 1));
    script.extend(req(OP_GET, 1));
    script.extend(req(OP_GET, 42)); // miss
    script.extend(req(OP_REMOVE, 1));
    script.extend(req(OP_BYE, 0));
    let mut dx = Duplex {
        inp: std::io::Cursor::new(script),
        out: Vec::new(),
    };
    run_paging_loop(&mut dx, &mut r).unwrap();

    // Expected replies: COMMIT→[OK]; GET1→[OK][off]; GET42→[MISS]; REMOVE→[OK].
    let mut exp = Vec::new();
    exp.push(ST_OK); // commit
    exp.push(ST_OK);
    exp.extend_from_slice(&off.to_le_bytes()); // get 1 (same slot/offset)
    exp.push(ST_MISS); // get 42
    exp.push(ST_OK); // remove
    assert_eq!(dx.out, exp);
}

/// Client codec: request bytes on the wire + reply decode.
#[test]
fn client_codec_alloc_get_miss() {
    // ALLOC → [ST_OK][offset 0x40]
    let mut reply = vec![ST_OK];
    reply.extend_from_slice(&0x40u64.to_le_bytes());
    let mut dx = Duplex {
        inp: std::io::Cursor::new(reply),
        out: Vec::new(),
    };
    let off = client_alloc(&mut dx, 0xAB).unwrap();
    assert_eq!(off, 0x40);
    assert_eq!(dx.out, req(OP_ALLOC, 0xAB), "request bytes on the wire");

    // GET miss → [ST_MISS]
    let mut dx = Duplex {
        inp: std::io::Cursor::new(vec![ST_MISS]),
        out: Vec::new(),
    };
    assert_eq!(client_get(&mut dx, 7).unwrap(), None);

    // COMMIT ok → [ST_OK]
    let mut dx = Duplex {
        inp: std::io::Cursor::new(vec![ST_OK]),
        out: Vec::new(),
    };
    client_commit(&mut dx, 9).unwrap();
    assert_eq!(dx.out, req(OP_COMMIT, 9));
}

/// End-to-end loopback: the client codec's request bytes feed the peer
/// `dispatch`; the peer's reply bytes feed the client codec — the two halves
/// agree on the wire and a PUT→GET round-trips a blob byte-identical (the
/// RDMA data plane emulated via direct arena writes at the returned offset).
#[test]
fn client_peer_loopback_roundtrip() {
    let mut r = residency(4);
    // Run one client request through the peer and return the client-decoded
    // reply channel (a cursor over the peer's reply bytes).
    fn peer_roundtrip(r: &mut TestResidency, req_bytes: &[u8]) -> std::io::Cursor<Vec<u8>> {
        let op = req_bytes[0];
        let key = u64::from_le_bytes(req_bytes[1..9].try_into().unwrap());
        let mut reply = Vec::new();
        write_reply(&mut reply, &dispatch(r, op, key)).unwrap();
        std::io::Cursor::new(reply)
    }

    // PUT key 3: ALLOC → emulate RDMA write → COMMIT.
    let mut wire = Vec::new();
    send_req(&mut wire, OP_ALLOC, 3).unwrap();
    let mut rep = peer_roundtrip(&mut r, &wire);
    assert_eq!(read_status(&mut rep).unwrap(), ST_OK);
    let off = read_offset(&mut rep).unwrap();
    r.arena_mut()
        .write_slot((off as usize) / B, &blob(0x77))
        .unwrap();

    wire.clear();
    send_req(&mut wire, OP_COMMIT, 3).unwrap();
    assert_eq!(
        read_status(&mut peer_roundtrip(&mut r, &wire)).unwrap(),
        ST_OK
    );

    // GET key 3 → read back byte-identical.
    wire.clear();
    send_req(&mut wire, OP_GET, 3).unwrap();
    let mut rep = peer_roundtrip(&mut r, &wire);
    assert_eq!(read_status(&mut rep).unwrap(), ST_OK);
    let goff = read_offset(&mut rep).unwrap();
    let mut out = vec![0u8; B];
    r.arena().read_slot((goff as usize) / B, &mut out).unwrap();
    assert_eq!(
        out,
        blob(0x77),
        "PUT→GET round-trips byte-identical over the protocol"
    );
}
