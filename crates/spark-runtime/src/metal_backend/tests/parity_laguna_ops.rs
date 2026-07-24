// SPDX-License-Identifier: AGPL-3.0-only
//! Laguna-specific YaRN RoPE and per-head softplus gate parity.

use super::super::*;
use super::helpers::*;
use crate::gpu::KernelArg;

fn f32_slice_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn assert_bf16_close(actual: &[half::bf16], expected: &[half::bf16], tolerance: f32) {
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual.to_f32() - expected.to_f32()).abs();
        assert!(
            difference <= tolerance,
            "element {index}: {} != {} (difference {difference})",
            actual.to_f32(),
            expected.to_f32()
        );
    }
}

#[test]
fn metal_laguna_yarn_rope_matches_cpu() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let (seq_len, q_heads, kv_heads, head_dim, rotary_dim) = (2u32, 2u32, 1u32, 8u32, 4u32);
    let positions = [3u32, 7u32];
    let inv_freq = [0.5f32, 0.1f32];
    let factor = 1.25f32;
    let q: Vec<half::bf16> = (0..seq_len * q_heads * head_dim)
        .map(|index| half::bf16::from_f32(index as f32 / 32.0 - 0.5))
        .collect();
    let k: Vec<half::bf16> = (0..seq_len * kv_heads * head_dim)
        .map(|index| half::bf16::from_f32(0.75 - index as f32 / 24.0))
        .collect();
    let mut expected_q = q.clone();
    let mut expected_k = k.clone();
    for (values, heads) in [(&mut expected_q, q_heads), (&mut expected_k, kv_heads)] {
        for token in 0..seq_len as usize {
            for head in 0..heads as usize {
                let base = (token * heads as usize + head) * head_dim as usize;
                for pair in 0..(rotary_dim / 2) as usize {
                    let angle = positions[token] as f32 * inv_freq[pair];
                    let cosine = angle.cos() * factor;
                    let sine = angle.sin() * factor;
                    let first = values[base + pair].to_f32();
                    let second = values[base + pair + (rotary_dim / 2) as usize].to_f32();
                    values[base + pair] = half::bf16::from_f32(first * cosine - second * sine);
                    values[base + pair + (rotary_dim / 2) as usize] =
                        half::bf16::from_f32(second * cosine + first * sine);
                }
            }
        }
    }

    let q_ptr = backend.alloc(q.len() * 2).unwrap();
    let k_ptr = backend.alloc(k.len() * 2).unwrap();
    let positions_ptr = backend.alloc(positions.len() * 4).unwrap();
    let inv_freq_ptr = backend.alloc(inv_freq.len() * 4).unwrap();
    backend.copy_h2d(&bf16_slice_to_bytes(&q), q_ptr).unwrap();
    backend.copy_h2d(&bf16_slice_to_bytes(&k), k_ptr).unwrap();
    backend
        .copy_h2d(&u32_slice_to_bytes(&positions), positions_ptr)
        .unwrap();
    backend
        .copy_h2d(&f32_slice_to_bytes(&inv_freq), inv_freq_ptr)
        .unwrap();
    let kernel = backend.kernel("rope", "rope_forward_yarn_scaled").unwrap();
    backend
        .launch_typed(
            kernel,
            [q_heads + kv_heads, 1, 1],
            [128, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(q_ptr),
                KernelArg::Buffer(k_ptr),
                KernelArg::Buffer(positions_ptr),
                KernelArg::Bytes(&seq_len.to_le_bytes()),
                KernelArg::Bytes(&q_heads.to_le_bytes()),
                KernelArg::Bytes(&kv_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&rotary_dim.to_le_bytes()),
                KernelArg::Buffer(inv_freq_ptr),
                KernelArg::Bytes(&factor.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(0).unwrap();
    let mut q_raw = vec![0u8; q.len() * 2];
    let mut k_raw = vec![0u8; k.len() * 2];
    backend.copy_d2h(q_ptr, &mut q_raw).unwrap();
    backend.copy_d2h(k_ptr, &mut k_raw).unwrap();
    assert_bf16_close(&bytes_to_bf16_vec(&q_raw), &expected_q, 0.015625);
    assert_bf16_close(&bytes_to_bf16_vec(&k_raw), &expected_k, 0.015625);
}

#[test]
fn metal_laguna_softplus_head_gate_matches_cpu() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let (tokens, heads, head_dim) = (2u32, 3u32, 4u32);
    let total = tokens * heads * head_dim;
    let input: Vec<half::bf16> = (0..total)
        .map(|index| half::bf16::from_f32(index as f32 / 20.0 - 0.4))
        .collect();
    let gate: Vec<half::bf16> = [-2.0f32, -0.5, 0.25, 1.0, 4.0, 24.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let expected: Vec<half::bf16> = input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let token = index / (heads * head_dim) as usize;
            let head = (index / head_dim as usize) % heads as usize;
            let gate = gate[token * heads as usize + head].to_f32();
            let softplus = if gate > 20.0 {
                gate
            } else {
                (1.0 + gate.exp()).ln()
            };
            half::bf16::from_f32(value.to_f32() * softplus)
        })
        .collect();
    let input_ptr = backend.alloc(input.len() * 2).unwrap();
    let gate_ptr = backend.alloc(gate.len() * 2).unwrap();
    let output_ptr = backend.alloc(input.len() * 2).unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&input), input_ptr)
        .unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&gate), gate_ptr)
        .unwrap();
    let kernel = backend
        .kernel("residual_add", "softplus_gate_mul_head_broadcast")
        .unwrap();
    backend
        .launch_typed(
            kernel,
            [total.div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(input_ptr),
                KernelArg::Buffer(gate_ptr),
                KernelArg::Buffer(output_ptr),
                KernelArg::Bytes(&heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&total.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(0).unwrap();
    let mut raw = vec![0u8; input.len() * 2];
    backend.copy_d2h(output_ptr, &mut raw).unwrap();
    assert_bf16_close(&bytes_to_bf16_vec(&raw), &expected, 0.03125);
}
