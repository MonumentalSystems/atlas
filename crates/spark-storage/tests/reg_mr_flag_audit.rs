// SPDX-License-Identifier: AGPL-3.0-only

// reg_mr ACCESS-FLAG AUDIT (RailSet extraction, Step B) — the flag is a
// SECURITY/CORRECTNESS INVARIANT and must stay a per-caller parameter:
//
//   * `reg_mr(.., false)` → IBV_ACCESS_LOCAL_WRITE only (client landing /
//     bounce buffers — the NIC may DMA in, zero remote access): exactly 8
//     sites across the five clients.
//   * `reg_mr(.., true)`  → IBV_ACCESS_REMOTE_READ ONLY, deliberately without
//     LOCAL_WRITE so PROT_READ mmaps register: exactly 2 server sites
//     (expert_peer, weight_peer).
//   * `reg_mr_rw`         → REMOTE_READ|REMOTE_WRITE|LOCAL_WRITE: exactly 1
//     site (cache_peer's RW blade arena).
//
// This test scans the SOURCE of the involved files so a refactor that
// homogenizes, defaults, or moves a flag fails loudly. Line numbers may
// shift; counts and literal flags may not.

/// `(reg_mr_false, reg_mr_true, reg_mr_rw)` call-site counts in one file.
fn scan(file: &str) -> (usize, usize, usize) {
    let path = format!("{}/src/{}", env!("CARGO_MANIFEST_DIR"), file);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    // Strip whole-line comments so prose mentioning reg_mr can't count.
    let code: String = src
        .lines()
        .map(|l| {
            if l.trim_start().starts_with("//") {
                ""
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let rw = code.match_indices(".reg_mr_rw(").count();
    let bytes = code.as_bytes();
    let (mut n_false, mut n_true) = (0usize, 0usize);
    for (i, _) in code.match_indices(".reg_mr(") {
        // Walk to the matching close paren, then inspect the LAST argument —
        // it must be a literal `false` or `true`, never a variable/default.
        let start = i + ".reg_mr(".len();
        let mut depth = 1usize;
        let mut j = start;
        while depth > 0 {
            match bytes[j] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            j += 1;
        }
        let last_arg = code[start..j - 1].rsplit(',').next().unwrap().trim();
        match last_arg {
            "false" => n_false += 1,
            "true" => n_true += 1,
            other => panic!("{file}: reg_mr flag must be a literal bool, found {other:?}"),
        }
    }
    (n_false, n_true, rw)
}

#[test]
fn client_landing_mrs_are_local_write_only() {
    // The five RailSet clients: every registration is `remote_read == false`.
    assert_eq!(scan("rdma_kv_backend.rs"), (3, 0, 0)); // region + zero-copy dst + bounce ring
    assert_eq!(scan("expert_tier_rdma.rs"), (1, 0, 0)); // whole arena per rail
    assert_eq!(scan("weight_tier_rdma.rs"), (1, 0, 0)); // bounce per rail
    assert_eq!(scan("weight_lora_rdma.rs"), (1, 0, 0)); // single bounce
    assert_eq!(scan("rdma_snapshot.rs"), (2, 0, 0)); // bounce + shared staging
}

#[test]
fn server_store_mrs_keep_their_flags() {
    // Untouched by Step B: the RO stores are REMOTE_READ only (PROT_READ
    // mmaps), the KV blade is the single RW registration.
    assert_eq!(scan("expert_peer.rs"), (0, 1, 0));
    assert_eq!(scan("weight_peer.rs"), (0, 1, 0));
    assert_eq!(scan("cache_peer.rs"), (0, 0, 1));
}

#[test]
fn totals_are_eight_false_two_true_one_rw() {
    let files = [
        "rdma_kv_backend.rs",
        "expert_tier_rdma.rs",
        "weight_tier_rdma.rs",
        "weight_lora_rdma.rs",
        "rdma_snapshot.rs",
        "expert_peer.rs",
        "weight_peer.rs",
        "cache_peer.rs",
    ];
    let (mut f, mut t, mut rw) = (0, 0, 0);
    for file in files {
        let (a, b, c) = scan(file);
        f += a;
        t += b;
        rw += c;
    }
    assert_eq!((f, t, rw), (8, 2, 1), "reg_mr access-flag census changed");
}
