// SPDX-License-Identifier: AGPL-3.0-only
//! Native GGUF Q4_K matrix and contiguous expert-stack parity.

use super::super::*;
use super::helpers::*;
use crate::gpu::KernelArg;

const QK: usize = 256;
const BLOCK_BYTES: usize = 144;

fn pack_scales(scales: [u8; 8], mins: [u8; 8]) -> [u8; 12] {
    let mut packed = [0u8; 12];
    for j in 0..4 {
        packed[j] = (scales[j] & 0x3f) | ((scales[j + 4] >> 4) << 6);
        packed[j + 4] = (mins[j] & 0x3f) | ((mins[j + 4] >> 4) << 6);
        packed[j + 8] = (scales[j + 4] & 0x0f) | ((mins[j + 4] & 0x0f) << 4);
    }
    packed
}

fn block_fixture(seed: usize) -> (Vec<u8>, Vec<f32>) {
    let d = half::f16::from_f32(0.0078125 * (1 + seed % 2) as f32);
    let dmin = half::f16::from_f32(0.00390625);
    let scales = std::array::from_fn(|i| 1 + ((seed + i * 3) % 11) as u8);
    let mins = std::array::from_fn(|i| ((seed + i * 2) % 5) as u8);
    let mut packed = Vec::with_capacity(BLOCK_BYTES);
    packed.extend_from_slice(&d.to_le_bytes());
    packed.extend_from_slice(&dmin.to_le_bytes());
    packed.extend_from_slice(&pack_scales(scales, mins));
    let mut dense = vec![0.0f32; QK];
    for group in 0..4 {
        for lane in 0..32 {
            let lo = ((seed * 5 + group * 7 + lane * 3) % 16) as u8;
            let hi = ((seed * 9 + group * 3 + lane * 5 + 1) % 16) as u8;
            packed.push(lo | (hi << 4));
            let even = group * 2;
            let odd = even + 1;
            dense[even * 32 + lane] =
                d.to_f32() * scales[even] as f32 * lo as f32 - dmin.to_f32() * mins[even] as f32;
            dense[odd * 32 + lane] =
                d.to_f32() * scales[odd] as f32 * hi as f32 - dmin.to_f32() * mins[odd] as f32;
        }
    }
    assert_eq!(packed.len(), BLOCK_BYTES);
    (packed, dense)
}

fn matrix_fixture(n: usize, k: usize, seed: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % QK, 0);
    let mut packed = Vec::with_capacity(n * k / QK * BLOCK_BYTES);
    let mut dense = Vec::with_capacity(n * k);
    for row in 0..n {
        for block in 0..k / QK {
            let (p, d) = block_fixture(seed + row * 13 + block);
            packed.extend_from_slice(&p);
            dense.extend_from_slice(&d);
        }
    }
    (packed, dense)
}

fn reference(
    input: &[half::bf16],
    weights: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<half::bf16> {
    let mut out = vec![half::bf16::ZERO; m * n];
    for row in 0..m {
        for col in 0..n {
            let sum: f32 = (0..k)
                .map(|i| input[row * k + i].to_f32() * weights[col * k + i])
                .sum();
            out[row * n + col] = half::bf16::from_f32(sum);
        }
    }
    out
}

fn assert_close(actual: &[half::bf16], expected: &[half::bf16], label: &str) {
    let max_diff = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a.to_f32() - e.to_f32()).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff <= 0.0625, "{label}: max abs diff {max_diff}");
}

fn i32_slice_to_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn metal_gguf_q4_k_gemv_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let (n, k) = (7u32, 512u32);
    let (packed, dense) = matrix_fixture(n as usize, k as usize, 2);
    let input: Vec<half::bf16> = (0..k)
        .map(|i| half::bf16::from_f32((i as f32 % 37.0 - 18.0) / 96.0))
        .collect();
    let expected = reference(&input, &dense, 1, n as usize, k as usize);
    let w = backend.alloc(packed.len()).unwrap();
    let x = backend.alloc(input.len() * 2).unwrap();
    let y = backend.alloc(n as usize * 2).unwrap();
    backend.copy_h2d(&packed, w).unwrap();
    backend.copy_h2d(&bf16_slice_to_bytes(&input), x).unwrap();
    let kernel = backend.kernel("gguf_q4_k_gemv", "gguf_q4_k_gemv").unwrap();
    backend
        .launch_typed(
            kernel,
            [n.div_ceil(4), 1, 1],
            [128, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(x),
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
    assert_close(&bytes_to_bf16_vec(&raw), &expected, "q4_k gemv");
}

#[test]
fn metal_gguf_q4_k_grouped_gemm_uses_contiguous_expert_stride() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    // More than 32 slots selects the production prefill tile. Expert runs
    // deliberately cross 16-slot boundaries to cover the mixed-expert
    // fallback as well as weight reuse within a sorted run.
    let (experts, slots, n, k) = (3usize, 35usize, 5usize, 256usize);
    let expert_ids: Vec<i32> = (0..slots)
        .map(|slot| match slot {
            0..11 => 0,
            11..29 => 1,
            _ => 2,
        })
        .collect();
    let input: Vec<half::bf16> = (0..slots * k)
        .map(|i| half::bf16::from_f32(((i % k) as f32 % 29.0 - 14.0) / 80.0))
        .collect();
    let mut packed = Vec::new();
    let mut dense = Vec::new();
    for expert in 0..experts {
        let (p, d) = matrix_fixture(n, k, 10 + expert * 17);
        packed.extend_from_slice(&p);
        dense.push(d);
    }
    let mut expected = vec![half::bf16::ZERO; slots * n];
    for slot in 0..slots {
        if let Ok(expert) = usize::try_from(expert_ids[slot])
            && expert < experts
        {
            expected[slot * n..(slot + 1) * n].copy_from_slice(&reference(
                &input[slot * k..(slot + 1) * k],
                &dense[expert],
                1,
                n,
                k,
            ));
        }
    }
    let w = backend.alloc(packed.len()).unwrap();
    let x = backend.alloc(input.len() * 2).unwrap();
    let ids = backend.alloc(expert_ids.len() * 4).unwrap();
    let y = backend.alloc(expected.len() * 2).unwrap();
    backend.copy_h2d(&packed, w).unwrap();
    backend.copy_h2d(&bf16_slice_to_bytes(&input), x).unwrap();
    backend
        .copy_h2d(&i32_slice_to_bytes(&expert_ids), ids)
        .unwrap();
    let kernel = backend
        .kernel("gguf_q4_k_grouped_gemm", "gguf_q4_k_grouped_gemm")
        .unwrap();
    let slots_u32 = slots as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let slots_per_tg = 16u32;
    let expert_stride = (n * k / QK * BLOCK_BYTES) as u64;
    backend
        .launch_typed(
            kernel,
            [slots_u32.div_ceil(slots_per_tg), n_u32.div_ceil(16), 1],
            [128, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(x),
                KernelArg::Buffer(w),
                KernelArg::Buffer(ids),
                KernelArg::Buffer(y),
                KernelArg::Bytes(&slots_u32.to_le_bytes()),
                KernelArg::Bytes(&n_u32.to_le_bytes()),
                KernelArg::Bytes(&k_u32.to_le_bytes()),
                KernelArg::Bytes(&(experts as u32).to_le_bytes()),
                KernelArg::Bytes(&expert_stride.to_le_bytes()),
                KernelArg::Bytes(&slots_per_tg.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(0).unwrap();
    let mut raw = vec![0; expected.len() * 2];
    backend.copy_d2h(y, &mut raw).unwrap();
    assert_close(&bytes_to_bf16_vec(&raw), &expected, "q4_k grouped gemm");
}

#[test]
#[ignore = "manual production-shape Metal microbenchmark"]
fn metal_gguf_q4_k_grouped_gemm_production_tile_bench() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let (slots, n, k) = (160usize, 1024usize, 3072usize);
    let (mut packed, _) = matrix_fixture(n, k, 31);
    let expert_stride = packed.len() as u64;
    packed.extend_from_slice(&matrix_fixture(n, k, 47).0);
    let input: Vec<half::bf16> = (0..slots * k)
        .map(|i| half::bf16::from_f32(((i % k) as f32 % 29.0 - 14.0) / 80.0))
        .collect();
    let expert_ids = vec![0i32; slots];
    let w = backend.alloc(packed.len()).unwrap();
    let x = backend.alloc(input.len() * 2).unwrap();
    let ids = backend.alloc(expert_ids.len() * 4).unwrap();
    let y = backend.alloc(slots * n * 2).unwrap();
    backend.copy_h2d(&packed, w).unwrap();
    backend.copy_h2d(&bf16_slice_to_bytes(&input), x).unwrap();
    backend
        .copy_h2d(&i32_slice_to_bytes(&expert_ids), ids)
        .unwrap();
    let kernel = backend
        .kernel("gguf_q4_k_grouped_gemm", "gguf_q4_k_grouped_gemm")
        .unwrap();
    let (slots_u32, n_u32, k_u32) = (slots as u32, n as u32, k as u32);
    let slots_per_tg = 16u32;
    let launch = || {
        backend
            .launch_typed(
                kernel,
                [slots_u32.div_ceil(slots_per_tg), n_u32.div_ceil(16), 1],
                [128, 1, 1],
                0,
                0,
                &[
                    KernelArg::Buffer(x),
                    KernelArg::Buffer(w),
                    KernelArg::Buffer(ids),
                    KernelArg::Buffer(y),
                    KernelArg::Bytes(&slots_u32.to_le_bytes()),
                    KernelArg::Bytes(&n_u32.to_le_bytes()),
                    KernelArg::Bytes(&k_u32.to_le_bytes()),
                    KernelArg::Bytes(&2u32.to_le_bytes()),
                    KernelArg::Bytes(&expert_stride.to_le_bytes()),
                    KernelArg::Bytes(&slots_per_tg.to_le_bytes()),
                ],
            )
            .unwrap();
    };
    launch();
    backend.synchronize(0).unwrap();
    let iterations = 10;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        launch();
    }
    backend.synchronize(0).unwrap();
    eprintln!(
        "Q4_K grouped production tile (single expert): {:.3} ms/launch",
        start.elapsed().as_secs_f64() * 1_000.0 / f64::from(iterations)
    );

    let mixed_ids: Vec<i32> = (0..slots).map(|slot| i32::from(slot % 16 >= 8)).collect();
    backend
        .copy_h2d(&i32_slice_to_bytes(&mixed_ids), ids)
        .unwrap();
    launch();
    backend.synchronize(0).unwrap();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        launch();
    }
    backend.synchronize(0).unwrap();
    eprintln!(
        "Q4_K grouped production tile (mixed experts): {:.3} ms/launch",
        start.elapsed().as_secs_f64() * 1_000.0 / f64::from(iterations)
    );
}
