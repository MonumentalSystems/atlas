// SPDX-License-Identifier: AGPL-3.0-only
//! Native GGUF Q8_0 embedding, GEMV, and GEMM parity.

use super::super::*;
use super::helpers::*;
use crate::gpu::KernelArg;

fn fixture(n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % 32, 0);
    let mut packed = Vec::with_capacity(n * k / 32 * 34);
    let mut dense = Vec::with_capacity(n * k);
    for row in 0..n {
        for block in 0..k / 32 {
            let scale = half::f16::from_f32(0.015625 * (1 + (row + block) % 3) as f32);
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..32 {
                let q = ((row * 11 + block * 5 + i * 3) % 31) as i8 - 15;
                packed.push(q as u8);
                dense.push(scale.to_f32() * q as f32);
            }
        }
    }
    (packed, dense)
}

fn assert_close(actual: &[half::bf16], expected: &[half::bf16], label: &str) {
    let max_diff = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a.to_f32() - e.to_f32()).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff <= 0.0625, "{label}: max abs diff {max_diff}");
}

#[test]
fn metal_gguf_q8_0_gemv_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let (n, k) = (7u32, 64u32);
    let (packed, dense) = fixture(n as usize, k as usize);
    let x: Vec<half::bf16> = (0..k)
        .map(|i| half::bf16::from_f32((i as f32 - 20.0) / 64.0))
        .collect();
    let expected: Vec<half::bf16> = (0..n as usize)
        .map(|row| {
            half::bf16::from_f32(
                (0..k as usize)
                    .map(|col| dense[row * k as usize + col] * x[col].to_f32())
                    .sum(),
            )
        })
        .collect();
    let w = backend.alloc(packed.len()).unwrap();
    let xp = backend.alloc(x.len() * 2).unwrap();
    let y = backend.alloc(n as usize * 2).unwrap();
    backend.copy_h2d(&packed, w).unwrap();
    backend.copy_h2d(&bf16_slice_to_bytes(&x), xp).unwrap();
    let kernel = backend.kernel("gguf_q8_0_gemv", "gguf_q8_0_gemv").unwrap();
    backend
        .launch_typed(
            kernel,
            [n.div_ceil(4), 1, 1],
            [128, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(xp),
                KernelArg::Buffer(w),
                KernelArg::Buffer(y),
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(0).unwrap();
    let mut raw = vec![0; n as usize * 2];
    backend.copy_d2h(y, &mut raw).unwrap();
    assert_close(&bytes_to_bf16_vec(&raw), &expected, "q8 gemv");
}

#[test]
fn metal_gguf_q8_0_gemm_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let (m, n, k) = (3u32, 5u32, 64u32);
    let (packed, dense) = fixture(n as usize, k as usize);
    let x: Vec<half::bf16> = (0..m * k)
        .map(|i| half::bf16::from_f32(((i % k) as f32 - 10.0 + (i / k) as f32) / 80.0))
        .collect();
    let mut expected = vec![half::bf16::ZERO; (m * n) as usize];
    for mi in 0..m as usize {
        for ni in 0..n as usize {
            let sum = (0..k as usize)
                .map(|ki| x[mi * k as usize + ki].to_f32() * dense[ni * k as usize + ki])
                .sum();
            expected[mi * n as usize + ni] = half::bf16::from_f32(sum);
        }
    }
    let w = backend.alloc(packed.len()).unwrap();
    let xp = backend.alloc(x.len() * 2).unwrap();
    let y = backend.alloc((m * n) as usize * 2).unwrap();
    backend.copy_h2d(&packed, w).unwrap();
    backend.copy_h2d(&bf16_slice_to_bytes(&x), xp).unwrap();
    let kernel = backend.kernel("gguf_q8_0_gemm", "gguf_q8_0_gemm").unwrap();
    backend
        .launch_typed(
            kernel,
            [n.div_ceil(16), m.div_ceil(16), 1],
            [16, 16, 1],
            0,
            0,
            &[
                KernelArg::Buffer(xp),
                KernelArg::Buffer(w),
                KernelArg::Buffer(y),
                KernelArg::Bytes(&m.to_le_bytes()),
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(0).unwrap();
    let mut raw = vec![0; (m * n) as usize * 2];
    backend.copy_d2h(y, &mut raw).unwrap();
    assert_close(&bytes_to_bf16_vec(&raw), &expected, "q8 gemm");
}

#[test]
fn metal_gguf_q8_0_embedding_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let (vocab, hidden, num_tokens) = (6u32, 64u32, 3u32);
    let tokens = [4u32, 1, 99];
    let (packed, dense) = fixture(vocab as usize, hidden as usize);
    let mut expected = vec![half::bf16::ZERO; (num_tokens * hidden) as usize];
    for (ti, &token) in tokens.iter().enumerate() {
        if token < vocab {
            for h in 0..hidden as usize {
                expected[ti * hidden as usize + h] =
                    half::bf16::from_f32(dense[token as usize * hidden as usize + h]);
            }
        }
    }
    let w = backend.alloc(packed.len()).unwrap();
    let ids = backend.alloc(tokens.len() * 4).unwrap();
    let out = backend.alloc(expected.len() * 2).unwrap();
    backend.copy_h2d(&packed, w).unwrap();
    backend.copy_h2d(&u32_slice_to_bytes(&tokens), ids).unwrap();
    let kernel = backend
        .kernel("gguf_q8_0_embedding", "gguf_q8_0_embedding")
        .unwrap();
    backend
        .launch_typed(
            kernel,
            [hidden.div_ceil(16), num_tokens.div_ceil(16), 1],
            [16, 16, 1],
            0,
            0,
            &[
                KernelArg::Buffer(ids),
                KernelArg::Buffer(w),
                KernelArg::Buffer(out),
                KernelArg::Bytes(&num_tokens.to_le_bytes()),
                KernelArg::Bytes(&hidden.to_le_bytes()),
                KernelArg::Bytes(&vocab.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(0).unwrap();
    let mut raw = vec![0; expected.len() * 2];
    backend.copy_d2h(out, &mut raw).unwrap();
    assert_close(&bytes_to_bf16_vec(&raw), &expected, "q8 embedding");
}
