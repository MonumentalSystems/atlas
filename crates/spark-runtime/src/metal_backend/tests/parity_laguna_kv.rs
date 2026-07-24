// SPDX-License-Identifier: AGPL-3.0-only
//! Production-ABI BF16 paged KV ring and long-context decode parity.

use super::helpers::*;
use crate::gpu::{GpuBackend, KernelArg};

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn metal_laguna_bf16_paged_ring_append_and_decode_match_cpu() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let (tokens, heads, head_dim, block_size, physical_blocks) = (8u32, 1u32, 32u32, 2u32, 3u32);
    let elements = (tokens * heads * head_dim) as usize;
    let mut keys = vec![half::bf16::ZERO; elements];
    let mut values = vec![half::bf16::ZERO; elements];
    for token in 0..tokens as usize {
        keys[token * head_dim as usize] = half::bf16::from_f32((token as f32 - 4.0) * 0.5);
        for dimension in 0..head_dim as usize {
            values[token * head_dim as usize + dimension] = half::bf16::from_f32(token as f32);
        }
    }
    let slots: Vec<i64> = (0..tokens as i64).collect();
    let cache_elements = (physical_blocks * block_size * heads * head_dim) as usize;
    let key_ptr = backend.alloc(elements * 2).unwrap();
    let value_ptr = backend.alloc(elements * 2).unwrap();
    let slots_ptr = backend.alloc(slots.len() * 8).unwrap();
    let k_cache = backend.alloc(cache_elements * 2).unwrap();
    let v_cache = backend.alloc(cache_elements * 2).unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&keys), key_ptr)
        .unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&values), value_ptr)
        .unwrap();
    backend.copy_h2d(&i64_bytes(&slots), slots_ptr).unwrap();
    let append = backend
        .kernel("reshape_and_cache", "reshape_and_cache_flash")
        .unwrap();
    // Two ordered chunks exercise ring wrap without racing two threadgroups
    // that intentionally target the same physical row in one dispatch.
    for chunk in 0..2usize {
        let chunk_tokens = tokens / 2;
        let element_offset = chunk * chunk_tokens as usize * head_dim as usize * 2;
        backend
            .launch_typed(
                append,
                [chunk_tokens, 1, 1],
                [256, 1, 1],
                0,
                0,
                &[
                    KernelArg::Buffer(key_ptr.offset(element_offset)),
                    KernelArg::Buffer(value_ptr.offset(element_offset)),
                    KernelArg::Buffer(k_cache),
                    KernelArg::Buffer(v_cache),
                    KernelArg::Buffer(slots_ptr.offset(chunk * chunk_tokens as usize * 8)),
                    KernelArg::Bytes(&heads.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&block_size.to_le_bytes()),
                    KernelArg::Bytes(&(heads * head_dim).to_le_bytes()),
                    KernelArg::Bytes(&(heads * head_dim).to_le_bytes()),
                    KernelArg::Bytes(&physical_blocks.to_le_bytes()),
                ],
            )
            .unwrap();
    }

    let query = vec![half::bf16::from_f32(1.0); head_dim as usize];
    let q_ptr = backend.alloc(query.len() * 2).unwrap();
    let output = backend.alloc(query.len() * 2).unwrap();
    let table = [0i32, 1, 2, 3];
    let table_ptr = backend.alloc(table.len() * 4).unwrap();
    let seq_len_ptr = backend.alloc(4).unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&query), q_ptr)
        .unwrap();
    backend.copy_h2d(&i32_bytes(&table), table_ptr).unwrap();
    backend
        .copy_h2d(&i32_bytes(&[tokens as i32]), seq_len_ptr)
        .unwrap();
    let decode = backend.kernel("paged_decode", "paged_decode_attn").unwrap();
    let one = 1u32;
    let sliding_window = 3u32;
    let scale = 1.0f32;
    backend
        .launch_typed(
            decode,
            [heads, one, 1],
            [256, 1, 1],
            0,
            0,
            &[
                KernelArg::Buffer(q_ptr),
                KernelArg::Buffer(k_cache),
                KernelArg::Buffer(v_cache),
                KernelArg::Buffer(output),
                KernelArg::Buffer(table_ptr),
                KernelArg::Buffer(seq_len_ptr),
                KernelArg::Bytes(&(table.len() as u32).to_le_bytes()),
                KernelArg::Bytes(&heads.to_le_bytes()),
                KernelArg::Bytes(&heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&block_size.to_le_bytes()),
                KernelArg::Bytes(&scale.to_le_bytes()),
                KernelArg::Bytes(&(heads * head_dim).to_le_bytes()),
                KernelArg::Bytes(&sliding_window.to_le_bytes()),
                KernelArg::Bytes(&physical_blocks.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(0).unwrap();

    let scores = [0.5f32, 1.0, 1.5];
    let max_score = scores[2];
    let exps = scores.map(|score| (score - max_score).exp());
    let expected = (5.0 * exps[0] + 6.0 * exps[1] + 7.0 * exps[2]) / exps.iter().sum::<f32>();
    let mut raw = vec![0; head_dim as usize * 2];
    backend.copy_d2h(output, &mut raw).unwrap();
    for (dimension, actual) in bytes_to_bf16_vec(&raw).iter().enumerate() {
        assert!(
            (actual.to_f32() - expected).abs() < 0.03,
            "dimension {dimension}: {} != {expected}",
            actual.to_f32()
        );
    }
}
