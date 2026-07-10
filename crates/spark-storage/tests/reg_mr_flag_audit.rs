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
// This test scans the SOURCE so a refactor that homogenizes, defaults, or moves
// a flag fails loudly. Line numbers may shift; counts and literal flags may not.
//
// The census walks the WHOLE `src/` tree rather than a fixed file list: a
// module split (e.g. `rdma_kv_backend.rs` → `rdma_kv_backend/rail.rs`) must not
// silently drop call sites from the audit, and a `reg_mr` added in a brand-new
// file must not go uncounted. `every_reg_mr_site_is_accounted_for` pins that.

use std::path::{Path, PathBuf};

type Census = (usize, usize, usize); // (reg_mr_false, reg_mr_true, reg_mr_rw)

fn src_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` under `src/`, recursively.
fn all_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
        let p = e.expect("dir entry").path();
        if p.is_dir() {
            all_rs(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// `(false, true, rw)` call-site counts in one file.
fn scan_file(path: &Path) -> Census {
    let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
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
            other => panic!("{path:?}: reg_mr flag must be a literal bool, found {other:?}"),
        }
    }
    (n_false, n_true, rw)
}

/// Sum a module: `<name>.rs` plus everything under a sibling `<name>/` dir.
fn scan_module(name: &str) -> Census {
    let root = src_root();
    let mut files = vec![];
    let flat = root.join(format!("{name}.rs"));
    if flat.exists() {
        files.push(flat);
    }
    let dir = root.join(name);
    if dir.is_dir() {
        all_rs(&dir, &mut files);
    }
    assert!(!files.is_empty(), "module {name} has no sources");
    files
        .iter()
        .map(|p| scan_file(p))
        .fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
}

fn scan_tree() -> Census {
    let mut files = vec![];
    all_rs(&src_root(), &mut files);
    files
        .iter()
        .map(|p| scan_file(p))
        .fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
}

/// The modules that are *allowed* to register memory regions.
const REG_MR_MODULES: [&str; 8] = [
    "rdma_kv_backend",
    "expert_tier_rdma",
    "weight_tier_rdma",
    "weight_lora_rdma",
    "rdma_snapshot",
    "expert_peer",
    "weight_peer",
    "cache_peer",
];

#[test]
fn client_landing_mrs_are_local_write_only() {
    // The five RailSet clients: every registration is `remote_read == false`.
    // `rdma_kv_backend` is a directory module (rail.rs holds the ring's two).
    assert_eq!(scan_module("rdma_kv_backend"), (3, 0, 0)); // region + zero-copy dst + bounce ring
    assert_eq!(scan_module("expert_tier_rdma"), (1, 0, 0)); // whole arena per rail
    assert_eq!(scan_module("weight_tier_rdma"), (1, 0, 0)); // bounce per rail
    assert_eq!(scan_module("weight_lora_rdma"), (1, 0, 0)); // single bounce
    assert_eq!(scan_module("rdma_snapshot"), (2, 0, 0)); // bounce + shared staging
}

#[test]
fn server_store_mrs_keep_their_flags() {
    // Untouched by Step B: the RO stores are REMOTE_READ only (PROT_READ
    // mmaps), the KV blade is the single RW registration.
    assert_eq!(scan_module("expert_peer"), (0, 1, 0));
    assert_eq!(scan_module("weight_peer"), (0, 1, 0));
    assert_eq!(scan_module("cache_peer"), (0, 0, 1));
}

#[test]
fn totals_are_eight_false_two_true_one_rw() {
    let (f, t, rw) = REG_MR_MODULES
        .iter()
        .map(|m| scan_module(m))
        .fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    assert_eq!((f, t, rw), (8, 2, 1), "reg_mr access-flag census changed");
}

#[test]
fn every_reg_mr_site_is_accounted_for() {
    // Walk the whole crate: the tree total must equal the sum over the declared
    // modules. If they diverge, a `reg_mr` call was added in a file this audit
    // does not cover — which is precisely how an access flag would slip through.
    let tree = scan_tree();
    let declared = REG_MR_MODULES
        .iter()
        .map(|m| scan_module(m))
        .fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    assert_eq!(
        tree, declared,
        "a reg_mr call exists outside {REG_MR_MODULES:?} — add the module to the audit"
    );
}
