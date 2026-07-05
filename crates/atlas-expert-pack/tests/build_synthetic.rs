// SPDX-License-Identifier: AGPL-3.0-only
//
// End-to-end build test on a tiny synthetic NVFP4 MoE checkpoint — exercises the
// whole pipeline (safetensors parse -> geometry discovery -> transpose -> pack ->
// write -> read-back verify) with no GPU and no multi-GB checkpoint, so it runs
// in CI. The synthetic file mimics the compressed-tensors expert naming the real
// qwen3.5 checkpoints use.

use std::io::Write;
use std::path::{Path, PathBuf};

use atlas_expert_pack::build::{BuildOpts, discover_geometry, run_build};
use atlas_expert_pack::checkpoint::Checkpoint;

const BASE: &str = "model.language_model";
const LAYERS: u32 = 2;
const EXPERTS: u32 = 3;
const INTER: u64 = 16;
const HIDDEN: u64 = 32;
const GS: u64 = 16;

struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

fn tname(layer: u32, expert: u32, proj: &str, field: &str) -> String {
    format!("{BASE}.layers.{layer}.mlp.experts.{expert}.{proj}_proj.{field}")
}

/// Deterministic pseudo-random bytes so the round-trip check is meaningful.
fn fill(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s & 0xFF) as u8
        })
        .collect()
}

fn write_synthetic_checkpoint(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let mut tensors: Vec<Tensor> = Vec::new();
    // Add a couple of non-expert tensors to prove they're ignored.
    tensors.push(Tensor {
        name: format!("{BASE}.embed_tokens.weight"),
        dtype: "BF16",
        shape: vec![4, HIDDEN],
        bytes: fill((4 * HIDDEN * 2) as usize, 999),
    });

    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            for (proj, n, k) in [
                ("gate", INTER, HIDDEN),
                ("up", INTER, HIDDEN),
                ("down", HIDDEN, INTER),
            ] {
                let seed = (layer as u64) << 20 | (expert as u64) << 8 | proj.len() as u64;
                tensors.push(Tensor {
                    name: tname(layer, expert, proj, "weight_packed"),
                    dtype: "U8",
                    shape: vec![n, k / 2],
                    bytes: fill((n * k / 2) as usize, seed),
                });
                tensors.push(Tensor {
                    name: tname(layer, expert, proj, "weight_scale"),
                    dtype: "F8_E4M3",
                    shape: vec![n, k / GS],
                    bytes: fill((n * k / GS) as usize, seed ^ 0xABCD),
                });
                tensors.push(Tensor {
                    name: tname(layer, expert, proj, "weight_global_scale"),
                    dtype: "F32",
                    shape: vec![1],
                    // A plausible reciprocal-convention global scale.
                    bytes: (2.0f32 + expert as f32).to_le_bytes().to_vec(),
                });
                tensors.push(Tensor {
                    name: tname(layer, expert, proj, "input_global_scale"),
                    dtype: "F32",
                    shape: vec![1],
                    bytes: (0.5f32 + proj.len() as f32).to_le_bytes().to_vec(),
                });
            }
        }
    }

    // Assemble safetensors: 8-byte LE header len, JSON header, then data blob.
    let mut blob = Vec::new();
    let mut header = serde_json::Map::new();
    for t in &tensors {
        let begin = blob.len() as u64;
        blob.extend_from_slice(&t.bytes);
        let end = blob.len() as u64;
        let mut m = serde_json::Map::new();
        m.insert("dtype".into(), serde_json::Value::String(t.dtype.into()));
        m.insert(
            "shape".into(),
            serde_json::Value::Array(t.shape.iter().map(|&s| s.into()).collect()),
        );
        m.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![begin.into(), end.into()]),
        );
        header.insert(t.name.clone(), serde_json::Value::Object(m));
    }
    let hdr_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();

    let mut f = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    f.write_all(&(hdr_json.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&hdr_json).unwrap();
    f.write_all(&blob).unwrap();
    f.sync_all().unwrap();
}

fn tmpdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "atlas-xpr-it-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn discovers_geometry_from_shapes() {
    let ckpt_dir = tmpdir("disc");
    write_synthetic_checkpoint(&ckpt_dir);
    let ckpt = Checkpoint::open(&ckpt_dir).unwrap();
    let geo = discover_geometry(&ckpt).unwrap();
    assert_eq!(geo.base_prefix, BASE);
    assert_eq!(geo.moe_layers, vec![0, 1]);
    assert_eq!(geo.num_experts, EXPERTS);
    assert_eq!(geo.inter, INTER);
    assert_eq!(geo.hidden, HIDDEN);
    assert_eq!(geo.group_size, GS);
    assert!(geo.has_input_scale);
    std::fs::remove_dir_all(&ckpt_dir).ok();
}

#[test]
fn full_build_round_trips_and_verifies() {
    let ckpt_dir = tmpdir("build-ck");
    let out_dir = tmpdir("build-out");
    write_synthetic_checkpoint(&ckpt_dir);

    let report = run_build(
        &ckpt_dir,
        &out_dir,
        BuildOpts {
            max_layers: 0,
            verify_samples: LAYERS * EXPERTS, // verify every record
        },
    )
    .expect("build succeeds");

    assert_eq!(report.layers_built, LAYERS);
    assert_eq!(report.experts_built, (LAYERS * EXPERTS) as u64);
    assert_eq!(report.verified, LAYERS * EXPERTS);
    // Payload matches the plan formula for these dims.
    assert_eq!(report.bytes_per_expert_payload, 3 * INTER * HIDDEN * 9 / 16);
    assert_eq!(report.record_stride % 4096, 0);

    // Manifest is present and self-consistent.
    let manifest = std::fs::read_to_string(out_dir.join("manifest.json")).unwrap();
    assert!(manifest.contains("\"num_moe_layers\": 2"));
    assert!(out_dir.join("experts_00000.xpr").exists());
    assert!(out_dir.join("experts_00001.xpr").exists());

    std::fs::remove_dir_all(&ckpt_dir).ok();
    std::fs::remove_dir_all(&out_dir).ok();
}

#[test]
fn max_layers_limits_build() {
    let ckpt_dir = tmpdir("cap-ck");
    let out_dir = tmpdir("cap-out");
    write_synthetic_checkpoint(&ckpt_dir);
    let report = run_build(
        &ckpt_dir,
        &out_dir,
        BuildOpts {
            max_layers: 1,
            verify_samples: 4,
        },
    )
    .unwrap();
    assert_eq!(report.layers_built, 1);
    assert_eq!(report.experts_built, EXPERTS as u64);
    assert!(!out_dir.join("experts_00001.xpr").exists());
    std::fs::remove_dir_all(&ckpt_dir).ok();
    std::fs::remove_dir_all(&out_dir).ok();
}
