// SPDX-License-Identifier: AGPL-3.0-only

// Golden BYTE-VECTOR pins for the RDMA handshake wire format (RailSet
// extraction, Step A — committed against the PRE-refactor codecs on purpose).
//
// These byte layouts are frozen — they are what the fleet peer binary speaks.
// Round-trip tests are structurally blind to a symmetric field reorder
// (writer and reader move together in a refactor); only hand-written byte
// vectors catch one. Every field gets DISTINCT byte values so any swap or
// endianness change is visible. All integers little-endian.
//
// Step B (the RailSet migration) must leave every vector in this file
// unchanged — that invariance, plus the byte-identical rdma_shim.c move, is
// the wire-identity evidence (no contact with the production peer needed).

use spark_storage::cache_peer::CacheServerParams;
use spark_storage::expert_peer::{
    MODE_TCP, MODE_VERBS, SHUTDOWN_MARKER, STATUS_ERR, STATUS_OK, VerbsClientParams,
    VerbsServerParams, decode_request, encode_request, read_server_rails, write_server_rails,
};
use spark_storage::snapshot_swap::{
    OP_ALLOC, OP_BYE, OP_COMMIT, OP_GET, OP_REMOVE, PAGING_MAGIC_V2, ST_ERR, ST_MISS, ST_OK,
};
use spark_storage::weight_peer::{
    MODEL_REQUEST_MAX, WeightManifest, WeightTensorRecord, read_model_request, read_weight_manifest,
    write_model_request, write_weight_manifest,
};

fn gid(start: u8) -> [u8; 16] {
    core::array::from_fn(|i| start + i as u8)
}

/// `[u32 qpn][u32 psn][16B gid]` — 24 bytes exactly.
#[test]
fn verbs_client_params_golden() {
    let p = VerbsClientParams {
        qpn: 0x0403_0201,
        psn: 0x0807_0605,
        gid: gid(0x10),
    };
    let mut buf = Vec::new();
    p.write_to(&mut buf).unwrap();
    #[rustfmt::skip]
    let expect: [u8; 24] = [
        0x01, 0x02, 0x03, 0x04,                         // qpn LE
        0x05, 0x06, 0x07, 0x08,                         // psn LE
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, // gid, verbatim
        0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    ];
    assert_eq!(buf, expect);
    // The reader must accept exactly these bytes back.
    let back = VerbsClientParams::read_from(&mut &expect[..]).unwrap();
    assert_eq!(back, p);
}

/// `[u32 qpn][u32 psn][16B gid][u32 n]{[u64 base][u32 rkey]}*` — 2 layers =
/// 52 bytes exactly.
#[test]
fn verbs_server_params_golden() {
    let p = VerbsServerParams {
        qpn: 0x0403_0201,
        psn: 0x0807_0605,
        gid: gid(0x20),
        layers: vec![
            (0x1122_3344_5566_7788, 0x99AA_BBCC),
            (0xDEAD_BEEF_0BAD_F00D, 0x0102_0304),
        ],
    };
    let mut buf = Vec::new();
    p.write_to(&mut buf).unwrap();
    #[rustfmt::skip]
    let expect: [u8; 52] = [
        0x01, 0x02, 0x03, 0x04,                         // qpn LE
        0x05, 0x06, 0x07, 0x08,                         // psn LE
        0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, // gid, verbatim
        0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
        0x02, 0x00, 0x00, 0x00,                         // n_layers = 2 LE
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // layer0 base LE
        0xCC, 0xBB, 0xAA, 0x99,                         // layer0 rkey LE
        0x0D, 0xF0, 0xAD, 0x0B, 0xEF, 0xBE, 0xAD, 0xDE, // layer1 base LE
        0x04, 0x03, 0x02, 0x01,                         // layer1 rkey LE
    ];
    assert_eq!(buf, expect);
    let back = VerbsServerParams::read_from(&mut &expect[..]).unwrap();
    assert_eq!(back, p);
}

/// Rails framing: a leading `[u8 n_rails]` then each rail's params. The
/// reader bails when the framed count differs from the negotiated `want`,
/// and bounds n to 1..=8.
#[test]
fn server_rails_framing_golden() {
    let rail = |seed: u8| VerbsServerParams {
        qpn: 0x0403_0201,
        psn: 0x0807_0605,
        gid: gid(seed),
        layers: vec![(0x1122_3344_5566_7788, 0x99AA_BBCC)],
    };
    let rails = vec![rail(0x30), rail(0x40)];
    let mut buf = Vec::new();
    write_server_rails(&mut buf, &rails).unwrap();
    // [u8 2] + 2 × 40-byte single-layer params.
    assert_eq!(buf.len(), 1 + 2 * 40);
    assert_eq!(buf[0], 0x02);
    let mut one = Vec::new();
    rails[0].write_to(&mut one).unwrap();
    assert_eq!(&buf[1..41], &one[..]);
    let mut two = Vec::new();
    rails[1].write_to(&mut two).unwrap();
    assert_eq!(&buf[41..], &two[..]);

    let back = read_server_rails(&mut &buf[..], 2).unwrap();
    assert_eq!(back, rails);
    // Negotiated-count mismatch is a protocol error.
    assert!(read_server_rails(&mut &buf[..], 1).is_err());
    // Zero rails / >8 rails are implausible on both ends.
    assert!(write_server_rails(&mut Vec::new(), &[]).is_err());
    assert!(read_server_rails(&mut &[0u8][..], 0).is_err());
    assert!(read_server_rails(&mut &[9u8][..], 9).is_err());
}

/// `[u32 qpn][u32 psn][16B gid][u64 base][u32 rkey]` — 36 bytes exactly
/// (the RW-blade dialect: rdma_kv_backend + rdma_snapshot).
#[test]
fn cache_server_params_golden() {
    let p = CacheServerParams {
        qpn: 0x0403_0201,
        psn: 0x0807_0605,
        gid: gid(0x50),
        base_addr: 0x1122_3344_5566_7788,
        rkey: 0x99AA_BBCC,
    };
    let mut buf = Vec::new();
    p.write_to(&mut buf).unwrap();
    #[rustfmt::skip]
    let expect: [u8; 36] = [
        0x01, 0x02, 0x03, 0x04,                         // qpn LE
        0x05, 0x06, 0x07, 0x08,                         // psn LE
        0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, // gid, verbatim
        0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F,
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // base_addr LE
        0xCC, 0xBB, 0xAA, 0x99,                         // rkey LE
    ];
    assert_eq!(buf, expect);
    let back = CacheServerParams::read_from(&mut &expect[..]).unwrap();
    assert_eq!(back, p);
}

/// Expert fetch request: `[u32 layer][u32 expert]`, plus the shutdown
/// sentinel (layer == u32::MAX).
#[test]
fn expert_request_golden() {
    assert_eq!(
        encode_request(0x0102_0304, 0x0506_0708),
        [0x04, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05]
    );
    assert_eq!(
        decode_request(&[0x04, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05]),
        (0x0102_0304, 0x0506_0708)
    );
    assert_eq!(
        encode_request(SHUTDOWN_MARKER, 0),
        [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00]
    );
}

/// Model/adapter request: `[u32 len][len bytes UTF-8]`, len in 1..=8192.
#[test]
fn model_request_framing_golden() {
    let mut buf = Vec::new();
    write_model_request(&mut buf, "m7").unwrap();
    assert_eq!(buf, [0x02, 0x00, 0x00, 0x00, b'm', b'7']);
    assert_eq!(read_model_request(&mut &buf[..]).unwrap(), "m7");
    assert!(write_model_request(&mut Vec::new(), "").is_err());
    assert_eq!(MODEL_REQUEST_MAX, 8192);
}

/// A FIXED weight manifest, frozen byte-for-byte. Frame is `[u32 LE len][len
/// bytes JSON]`. The round-trip test (`manifest_round_trips`) is blind to a
/// coordinated serde field reorder (writer and reader move together); this pins
/// the WRITER's exact bytes — what the deployed `atlas-weight-peer` daemon
/// speaks — and the READER against the same frozen frame.
#[test]
fn weight_manifest_framing_golden() {
    let m = WeightManifest {
        version: WeightManifest::VERSION,
        model_id: "m7".to_string(),
        shard_files: vec!["s0.safetensors".to_string()],
        shard_lens: vec![0x0102_0304],
        tensors: vec![WeightTensorRecord {
            name: "blk.0.attn.weight".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![2, 3],
            offset_in_shard: 0x1000,
            len: 0x24,
            shard_index: 0,
            extra: true,
        }],
    };
    let mut buf = Vec::new();
    write_weight_manifest(&mut buf, &m).unwrap();
    #[rustfmt::skip]
    let expect: Vec<u8> = {
        let json = br#"{"version":1,"model_id":"m7","shard_files":["s0.safetensors"],"shard_lens":[16909060],"tensors":[{"name":"blk.0.attn.weight","dtype":"BF16","shape":[2,3],"offset_in_shard":4096,"len":36,"shard_index":0,"extra":true}]}"#;
        let mut v = Vec::with_capacity(4 + json.len());
        v.extend_from_slice(&(json.len() as u32).to_le_bytes());
        v.extend_from_slice(json);
        v
    };
    assert_eq!(buf, expect, "weight manifest wire bytes drifted");
    // The reader must accept exactly these frozen bytes back, structurally ==.
    let back = read_weight_manifest(&mut &expect[..]).unwrap();
    assert_eq!(back, m);
}

/// Every protocol constant the fleet peer binary speaks, frozen. v2-only
/// since Step C: PAGING_MAGIC_V2 is the ONLY accepted first u64 on the RW
/// paging port (the retired v1 magic is affirmatively rejected — pinned in
/// wire_tests::v1_magic_is_affirmatively_rejected — and the bare legacy
/// `total_bytes` handshake is gone, so no <= 1<<42 disambiguation remains;
/// the peer keeps an explicit arena sanity bound instead).
#[test]
fn protocol_consts_frozen() {
    assert_eq!(STATUS_OK, 0);
    assert_eq!(STATUS_ERR, 1);
    assert_eq!(MODE_TCP, 0);
    assert_eq!(MODE_VERBS, 1);
    assert_eq!(SHUTDOWN_MARKER, u32::MAX);
    assert_eq!(PAGING_MAGIC_V2, 0x5041_4745_0000_0002);
    assert_eq!(
        (OP_BYE, OP_ALLOC, OP_COMMIT, OP_GET, OP_REMOVE),
        (0, 1, 2, 3, 4)
    );
    assert_eq!((ST_OK, ST_MISS, ST_ERR), (0, 1, 2));
}
