// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;
use crate::weights::WeightLoader;

fn push_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_str(b: &mut Vec<u8>, s: &str) {
    push_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}

/// Minimal valid GGUF v3: one F32 1-D tensor + a `general.alignment` KV.
fn build_single_f32_gguf(name: &str, vals: &[f32]) -> Vec<u8> {
    let mut b = Vec::new();
    push_u32(&mut b, 0x4655_4747); // "GGUF"
    push_u32(&mut b, 3); // version
    push_u64(&mut b, 1); // tensor_count
    push_u64(&mut b, 1); // kv_count
    push_str(&mut b, "general.alignment");
    push_u32(&mut b, 4); // UINT32
    push_u32(&mut b, 32);
    push_str(&mut b, name);
    push_u32(&mut b, 1); // n_dims
    push_u64(&mut b, vals.len() as u64); // dims[0]
    push_u32(&mut b, 0); // ggml_type F32
    push_u64(&mut b, 0); // offset
    let pad = (32 - (b.len() % 32)) % 32;
    b.extend(std::iter::repeat_n(0u8, pad));
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// Minimal Laguna GGUF with one `[2, 32]` Q8_0 embedding matrix.
fn build_laguna_q8_0_gguf(raw: &[u8]) -> Vec<u8> {
    assert_eq!(raw.len(), 2 * 34);
    let mut b = Vec::new();
    push_u32(&mut b, 0x4655_4747);
    push_u32(&mut b, 3);
    push_u64(&mut b, 1); // tensor_count
    push_u64(&mut b, 2); // kv_count
    push_str(&mut b, "general.alignment");
    push_u32(&mut b, 4); // UINT32
    push_u32(&mut b, 32);
    push_str(&mut b, "general.architecture");
    push_u32(&mut b, 8); // STRING
    push_str(&mut b, "laguna");
    push_str(&mut b, "token_embd.weight");
    push_u32(&mut b, 2); // n_dims
    push_u64(&mut b, 32); // ggml fastest dimension K
    push_u64(&mut b, 2); // rows N
    push_u32(&mut b, 8); // Q8_0
    push_u64(&mut b, 0);
    let pad = (32 - (b.len() % 32)) % 32;
    b.extend(std::iter::repeat_n(0u8, pad));
    b.extend_from_slice(raw);
    b
}

#[test]
fn loads_single_tensor_cpu_fallback() {
    // Mock cannot execute kernels, so force the CPU reference dequant path.
    unsafe { std::env::set_var("ATLAS_GGUF_FORCE_CPU", "1") };

    let vals = [1.0f32, -2.0, 3.5, 0.0, 7.0, -0.25];
    let bytes = build_single_f32_gguf("token_embd.weight", &vals);

    let dir = std::env::temp_dir().join(format!("atlas_gguf_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.gguf"), &bytes).unwrap();

    let gpu = MockGpuBackend::new();
    let store = GgufLoader::new()
        .load(&dir, &gpu, 1024 * 1024)
        .expect("GGUF load");

    assert_eq!(store.len(), 1);
    assert!(store.contains("model.embed_tokens.weight"));
    let t = store.get("model.embed_tokens.weight").unwrap();
    assert_eq!(t.shape, vec![6]);
    assert_eq!(t.dtype, WeightDtype::BF16);

    let raw = gpu.read_alloc(t.ptr).expect("bf16 bytes present");
    assert_eq!(raw.len(), 6 * WeightDtype::BF16.byte_size());
    let got: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();
    assert_eq!(got, vals.to_vec());

    std::fs::remove_dir_all(&dir).ok();
    unsafe { std::env::remove_var("ATLAS_GGUF_FORCE_CPU") };
}

#[test]
fn laguna_rank2_q8_0_stays_packed() {
    let raw: Vec<u8> = (0..68).map(|i| (i * 7) as u8).collect();
    let bytes = build_laguna_q8_0_gguf(&raw);
    let dir =
        std::env::temp_dir().join(format!("atlas_gguf_laguna_q8_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.gguf"), &bytes).unwrap();

    let gpu = MockGpuBackend::new();
    let store = GgufLoader::new()
        .load(&dir, &gpu, 1024 * 1024)
        .expect("Laguna GGUF load");
    let tensor = store.get("model.embed_tokens.weight").unwrap();
    assert_eq!(tensor.shape, vec![2, 32]);
    assert_eq!(tensor.dtype, WeightDtype::PackedQ8_0);
    assert_eq!(tensor.byte_size(), 68);
    assert_eq!(gpu.read_alloc(tensor.ptr).unwrap(), raw);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn laguna_memory_footprint_preserves_packed_q8_bytes() {
    let raw = vec![0u8; 2 * 34];
    let bytes = build_laguna_q8_0_gguf(&raw);
    let gguf = container::GgufFile::parse(&bytes).expect("parse synthetic Laguna GGUF");
    let footprint = sidecar::memory_footprint(&gguf, "laguna", true, container::Q2Group::G128)
        .expect("compute exact footprint");

    assert_eq!(footprint.resident_bytes, raw.len());
    assert_eq!(footprint.max_tensor_transient_bytes, raw.len());
}

#[test]
fn memory_footprint_accounts_for_bf16_resident_and_raw_transient() {
    let bytes = build_single_f32_gguf("token_embd.weight", &[1.0, 2.0, 3.0]);
    let gguf = container::GgufFile::parse(&bytes).expect("parse synthetic F32 GGUF");
    let footprint = sidecar::memory_footprint(&gguf, "llama", false, container::Q2Group::G128)
        .expect("compute exact footprint");

    assert_eq!(footprint.resident_bytes, 3 * WeightDtype::BF16.byte_size());
    assert_eq!(
        footprint.max_tensor_transient_bytes,
        3 * std::mem::size_of::<f32>()
    );
}

#[test]
fn exact_preflight_includes_largest_tensor_and_guard() {
    const GIB: usize = 1024 * 1024 * 1024;
    let gpu = MockGpuBackend::new(); // reports 120 GiB allocation headroom

    memory::preflight_oom(&gpu, 100 * GIB, 10 * GIB, 5 * GIB)
        .expect("115 GiB exact peak should fit");
    let err = memory::preflight_oom(&gpu, 100 * GIB, 16 * GIB, 5 * GIB)
        .expect_err("121 GiB exact peak must fail before loading");
    let message = err.to_string();
    assert!(message.contains("100.00 GiB resident weights"));
    assert!(message.contains("16.00 GiB max tensor transient"));
}

#[test]
fn laguna_enforces_five_gib_minimum_guard() {
    const GIB: usize = 1024 * 1024 * 1024;
    assert_eq!(memory::effective_guard_bytes("laguna", 4 * GIB), 5 * GIB);
    assert_eq!(memory::effective_guard_bytes("laguna", 7 * GIB), 7 * GIB);
    assert_eq!(memory::effective_guard_bytes("llama", 4 * GIB), 4 * GIB);
}

#[test]
fn find_gguf_picks_first() {
    let dir = std::env::temp_dir().join(format!("atlas_gguf_find_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("b.gguf"), b"x").unwrap();
    std::fs::write(dir.join("a.gguf"), b"x").unwrap();
    std::fs::write(dir.join("notes.txt"), b"x").unwrap();
    let found = find_gguf(&dir).unwrap();
    assert_eq!(found.file_name().unwrap().to_str().unwrap(), "a.gguf");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn find_gguf_skips_mmproj_and_find_mmproj_pairs() {
    let dir = std::env::temp_dir().join(format!("atlas_gguf_mmproj_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // The mmproj sorts lexicographically FIRST ('B' < 'T'), so a naive
    // first-file pick would wrongly select the sidecar as the backbone.
    std::fs::write(dir.join("Bonsai-mmproj-Q8_0.gguf"), b"x").unwrap();
    std::fs::write(dir.join("Ternary-Bonsai-27B-Q2_0.gguf"), b"x").unwrap();

    let backbone = find_gguf(&dir).unwrap();
    assert_eq!(
        backbone.file_name().unwrap().to_str().unwrap(),
        "Ternary-Bonsai-27B-Q2_0.gguf"
    );
    let mmproj = sidecar::find_mmproj(&dir, &backbone).unwrap();
    assert_eq!(
        mmproj.file_name().unwrap().to_str().unwrap(),
        "Bonsai-mmproj-Q8_0.gguf"
    );

    // A text-only dir yields no sidecar.
    let dir2 = std::env::temp_dir().join(format!("atlas_gguf_textonly_{}", std::process::id()));
    std::fs::create_dir_all(&dir2).unwrap();
    std::fs::write(dir2.join("model-Q2_0.gguf"), b"x").unwrap();
    let bb2 = find_gguf(&dir2).unwrap();
    assert!(sidecar::find_mmproj(&dir2, &bb2).is_none());

    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&dir2).ok();
}
