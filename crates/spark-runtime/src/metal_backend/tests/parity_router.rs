// SPDX-License-Identifier: AGPL-3.0-only
//! Laguna sigmoid router and routed-output assembly parity.

use super::super::*;
use super::helpers::*;
use crate::gpu::{GpuBackend, KernelArg};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn read_u32(backend: &MetalGpuBackend, ptr: DevicePtr, len: usize) -> Vec<u32> {
    let mut bytes = vec![0; len * 4];
    backend.copy_d2h(ptr, &mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

fn read_i32(backend: &MetalGpuBackend, ptr: DevicePtr, len: usize) -> Vec<i32> {
    read_u32(backend, ptr, len)
        .into_iter()
        .map(|value| value as i32)
        .collect()
}

fn read_f32(backend: &MetalGpuBackend, ptr: DevicePtr, len: usize) -> Vec<f32> {
    let mut bytes = vec![0; len * 4];
    backend.copy_d2h(ptr, &mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

#[test]
fn metal_laguna_dense_router_gemv_covers_all_experts() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    // Laguna's real router is [256, 3072]. Use that K and one extra output
    // row so the final partially-active 4-row threadgroup is covered too.
    let (experts, hidden) = (257u32, 3072u32);
    let input: Vec<half::bf16> = (0..hidden)
        .map(|i| half::bf16::from_f32(((i % 63) as f32 - 31.0) / 64.0))
        .collect();
    let weights: Vec<half::bf16> = (0..experts * hidden)
        .map(|i| half::bf16::from_f32(((i % 29) as f32 - 14.0) / 128.0))
        .collect();
    let expected: Vec<half::bf16> = (0..experts as usize)
        .map(|row| {
            half::bf16::from_f32(
                (0..hidden as usize)
                    .map(|col| input[col].to_f32() * weights[row * hidden as usize + col].to_f32())
                    .sum(),
            )
        })
        .collect();
    let input_ptr = backend.alloc(input.len() * 2).unwrap();
    let weight_ptr = backend.alloc(weights.len() * 2).unwrap();
    let output_ptr = backend.alloc(experts as usize * 2).unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&input), input_ptr)
        .unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&weights), weight_ptr)
        .unwrap();
    backend
        .memset(output_ptr, 0x7f, experts as usize * 2)
        .unwrap();
    let kernel = backend.kernel("gemv", "dense_gemv_bf16").unwrap();
    backend
        .launch_typed(
            kernel,
            [experts.div_ceil(4), 1, 1],
            [256, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(input_ptr),
                KernelArg::Buffer(weight_ptr),
                KernelArg::Buffer(output_ptr),
                KernelArg::Bytes(&experts.to_le_bytes()),
                KernelArg::Bytes(&hidden.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(0).unwrap();
    let mut raw = vec![0; experts as usize * 2];
    backend.copy_d2h(output_ptr, &mut raw).unwrap();
    let actual = bytes_to_bf16_vec(&raw);
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual.to_f32() - expected.to_f32()).abs() <= 0.125,
            "router row {row} mismatch: actual={actual:?} expected={expected:?}"
        );
    }
}

#[test]
fn metal_laguna_correction_biased_sigmoid_topk_matches_cpu() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let logits = [0.0f32, 1.0, -1.0, 0.5, 1.5, -0.5, 0.25, -1.5];
    let logits_bf16 = logits.map(half::bf16::from_f32);
    let bias = [0.0f32, 0.2, 1.0, -0.2];
    let (tokens, experts, topk) = (2u32, 4u32, 2u32);
    let scaling = 0.7f32;
    let logits_ptr = backend.alloc(logits_bf16.len() * 2).unwrap();
    let bias_ptr = backend.alloc(bias.len() * 4).unwrap();
    let ids_ptr = backend.alloc(tokens as usize * topk as usize * 4).unwrap();
    let weights_ptr = backend.alloc(tokens as usize * topk as usize * 4).unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&logits_bf16), logits_ptr)
        .unwrap();
    backend.copy_h2d(&f32_bytes(&bias), bias_ptr).unwrap();
    let kernel = backend
        .kernel("moe_topk_sig", "moe_topk_sigmoid_batched")
        .unwrap();
    backend
        .launch_typed(
            kernel,
            [tokens, 1, 1],
            [256, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(logits_ptr),
                KernelArg::Buffer(bias_ptr),
                KernelArg::Buffer(ids_ptr),
                KernelArg::Buffer(weights_ptr),
                KernelArg::Bytes(&experts.to_le_bytes()),
                KernelArg::Bytes(&topk.to_le_bytes()),
                KernelArg::Bytes(&1u32.to_le_bytes()),
                KernelArg::Bytes(&scaling.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(0).unwrap();

    let ids = read_u32(&backend, ids_ptr, 4);
    let actual_weights = read_f32(&backend, weights_ptr, 4);
    let mut expected_ids = Vec::new();
    let mut expected_weights = Vec::new();
    for token in 0..tokens as usize {
        let scores: Vec<f32> = (0..experts as usize)
            .map(|expert| {
                1.0 / (1.0 + (-logits_bf16[token * experts as usize + expert].to_f32()).exp())
            })
            .collect();
        let mut ranked: Vec<usize> = (0..experts as usize).collect();
        ranked.sort_by(|a, b| {
            (scores[*b] + bias[*b])
                .partial_cmp(&(scores[*a] + bias[*a]))
                .unwrap()
        });
        let selected = &ranked[..topk as usize];
        let sum: f32 = selected.iter().map(|expert| scores[*expert]).sum();
        for expert in selected {
            expected_ids.push(*expert as u32);
            expected_weights.push(scores[*expert] * scaling / sum);
        }
    }
    assert_eq!(ids, expected_ids);
    for (actual, expected) in actual_weights.iter().zip(expected_weights) {
        assert!((actual - expected).abs() < 1.0e-6);
    }
}

#[test]
fn metal_laguna_sort_unpermute_and_ungated_shared_blend_match_cpu() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let ids = [2u32, 0, 1, 2];
    let weights = [0.25f32, 0.75, 0.6, 0.4];
    let expert_output = [
        10.0f32, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 5.0, 10.0, 15.0,
    ]
    .map(half::bf16::from_f32);
    let shared = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0].map(half::bf16::from_f32);
    let (tokens, experts, topk, hidden) = (2u32, 3u32, 2u32, 3u32);
    let total = tokens * topk;

    let ids_ptr = backend.alloc(ids.len() * 4).unwrap();
    let sorted_tokens = backend.alloc(total as usize * 4).unwrap();
    let sorted_experts = backend.alloc(total as usize * 4).unwrap();
    let offsets = backend.alloc((experts as usize + 1) * 4).unwrap();
    let reverse = backend.alloc(total as usize * 4).unwrap();
    let expert_ptr = backend.alloc(expert_output.len() * 2).unwrap();
    let weights_ptr = backend.alloc(weights.len() * 4).unwrap();
    let output = backend
        .alloc(tokens as usize * hidden as usize * 2)
        .unwrap();
    let shared_ptr = backend.alloc(shared.len() * 2).unwrap();
    backend
        .copy_h2d(
            &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            ids_ptr,
        )
        .unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&expert_output), expert_ptr)
        .unwrap();
    backend.copy_h2d(&f32_bytes(&weights), weights_ptr).unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&shared), shared_ptr)
        .unwrap();

    let sort = backend.kernel("moe", "moe_sort_by_expert").unwrap();
    backend
        .launch_typed(
            sort,
            [1, 1, 1],
            [256, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(ids_ptr),
                KernelArg::Buffer(sorted_tokens),
                KernelArg::Buffer(sorted_experts),
                KernelArg::Buffer(offsets),
                KernelArg::Buffer(reverse),
                KernelArg::Bytes(&total.to_le_bytes()),
                KernelArg::Bytes(&experts.to_le_bytes()),
                KernelArg::Bytes(&topk.to_le_bytes()),
            ],
        )
        .unwrap();
    let unpermute = backend
        .kernel("moe", "moe_unpermute_reduce_indexed")
        .unwrap();
    backend
        .launch_typed(
            unpermute,
            [tokens, 1, 1],
            [256, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(expert_ptr),
                KernelArg::Buffer(output),
                KernelArg::Buffer(reverse),
                KernelArg::Buffer(weights_ptr),
                KernelArg::Bytes(&hidden.to_le_bytes()),
                KernelArg::Bytes(&tokens.to_le_bytes()),
                KernelArg::Bytes(&topk.to_le_bytes()),
            ],
        )
        .unwrap();
    let blend = backend.kernel("moe", "moe_batched_blend").unwrap();
    backend
        .launch_typed(
            blend,
            [tokens, 1, 1],
            [256, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(output),
                KernelArg::Buffer(shared_ptr),
                KernelArg::Buffer(shared_ptr),
                KernelArg::Buffer(shared_ptr),
                KernelArg::Bytes(&hidden.to_le_bytes()),
                KernelArg::Bytes(&tokens.to_le_bytes()),
                KernelArg::Bytes(&0u32.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(0).unwrap();

    assert_eq!(read_i32(&backend, sorted_tokens, 4), [0, 1, 0, 1]);
    assert_eq!(read_i32(&backend, sorted_experts, 4), [0, 1, 2, 2]);
    assert_eq!(read_i32(&backend, offsets, 4), [0, 1, 2, 4]);
    assert_eq!(read_i32(&backend, reverse, 4), [2, 0, 1, 3]);
    let mut raw = vec![0; tokens as usize * hidden as usize * 2];
    backend.copy_d2h(output, &mut raw).unwrap();
    let actual = bytes_to_bf16_vec(&raw);
    let expected = [9.0f32, 18.0, 27.0, 4.6, 7.7, 10.8].map(half::bf16::from_f32);
    assert_eq!(actual, expected);
}
