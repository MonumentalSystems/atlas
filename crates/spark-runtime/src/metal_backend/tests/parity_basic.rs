// SPDX-License-Identifier: AGPL-3.0-only
//! Smoke + simple element-wise kernel parity (alloc, bf16_add, sigmoid_gate).

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

#[test]
fn metal_kernel_handle_zero_is_reserved() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let first = backend
        .kernel("noop_smoke", "noop_smoke")
        .expect("kernel lookup");
    assert_ne!(first.0, 0, "Metal must reserve KernelHandle(0)");

    let err = backend
        .launch_typed(
            KernelHandle(0),
            [1, 1, 1],
            [1, 1, 1],
            0,
            backend.default_stream(),
            &[],
        )
        .expect_err("reserved kernel handle must not launch");
    assert!(err.to_string().contains("invalid kernel handle 0"));
}

#[test]
fn metal_launch_resolves_only_referenced_buffers() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let referenced = backend.alloc(64).expect("referenced alloc");
    let unrelated = backend.alloc(64).expect("unrelated alloc");
    let scalar = 7u32.to_le_bytes();
    let args = [
        KernelArg::Bytes(&scalar),
        KernelArg::Buffer(referenced.offset(16)),
    ];
    let resolved = backend
        .resolve_buffer_args(&args)
        .expect("resolve buffer args");

    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].0, 1);
    assert_eq!(resolved[0].1.gpuAddress(), referenced.0);
    assert_eq!(resolved[0].2, 16);
    drop(resolved);

    backend.free(referenced).expect("free referenced");
    backend.free(unrelated).expect("free unrelated");
}

#[test]
fn metal_memory_info_separates_physical_and_working_set_capacity() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let info = backend.memory_info();
    assert!(info.recommended_max_working_set_bytes > 0);
    assert_eq!(
        backend.allocation_capacity().expect("allocation capacity"),
        info.recommended_max_working_set_bytes
    );
    assert_eq!(
        info.recommended_headroom_bytes(),
        info.recommended_max_working_set_bytes
            .saturating_sub(info.current_allocated_bytes)
    );
    if let Some(physical) = info.physical_memory_bytes {
        assert!(physical > 0);
    }
    assert!(!backend.supports_graph_capture());
}

#[test]
fn metal_vanilla_residual_add_rms_norm_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let hidden_size = 4u32;
    let eps = 1e-6f32;
    let hidden: Vec<half::bf16> = [1.0, 2.0, 3.0, 4.0].map(half::bf16::from_f32).into();
    let src: Vec<half::bf16> = [0.5, -0.5, 1.0, -1.0].map(half::bf16::from_f32).into();
    let weight: Vec<half::bf16> = [1.0, 2.0, 0.5, 1.5].map(half::bf16::from_f32).into();
    let updated: Vec<half::bf16> = hidden
        .iter()
        .zip(&src)
        .map(|(h, s)| half::bf16::from_f32(h.to_f32() + s.to_f32()))
        .collect();
    let mean_sq = updated.iter().map(|v| v.to_f32().powi(2)).sum::<f32>() / hidden_size as f32;
    let inv = (mean_sq + eps).sqrt().recip();
    let expected: Vec<half::bf16> = updated
        .iter()
        .zip(&weight)
        .map(|(v, w)| half::bf16::from_f32(v.to_f32() * w.to_f32() * inv))
        .collect();

    let bytes = hidden.len() * 2;
    let hidden_ptr = backend.alloc(bytes).unwrap();
    let src_ptr = backend.alloc(bytes).unwrap();
    let weight_ptr = backend.alloc(bytes).unwrap();
    let output_ptr = backend.alloc(bytes).unwrap();
    let residual_ptr = backend.alloc(bytes).unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&hidden), hidden_ptr)
        .unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&src), src_ptr)
        .unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&weight), weight_ptr)
        .unwrap();
    let kernel = backend
        .kernel("rms_norm_vanilla", "residual_add_rms_norm_vanilla")
        .unwrap();
    backend
        .launch_typed(
            kernel,
            [1, 1, 1],
            [hidden_size, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Buffer(hidden_ptr),
                KernelArg::Buffer(src_ptr),
                KernelArg::Buffer(weight_ptr),
                KernelArg::Buffer(output_ptr),
                KernelArg::Buffer(residual_ptr),
                KernelArg::Bytes(&hidden_size.to_le_bytes()),
                KernelArg::Bytes(&eps.to_le_bytes()),
            ],
        )
        .unwrap();
    backend.synchronize(backend.default_stream()).unwrap();

    let mut output_bytes = vec![0u8; bytes];
    let mut residual_bytes = vec![0u8; bytes];
    backend.copy_d2h(output_ptr, &mut output_bytes).unwrap();
    backend.copy_d2h(residual_ptr, &mut residual_bytes).unwrap();
    assert_eq!(bytes_to_bf16_vec(&residual_bytes), updated);
    assert_eq!(bytes_to_bf16_vec(&output_bytes), expected);
}

#[test]
fn metal_async_h2d_preserves_stream_order_when_destination_is_reused() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let first = [1u8, 2, 3, 4];
    let second = [5u8, 6, 7, 8];
    let shared = backend.alloc(first.len()).unwrap();
    let first_out = backend.alloc(first.len()).unwrap();
    let second_out = backend.alloc(first.len()).unwrap();
    let stream = backend.default_stream();

    backend.copy_h2d_async(&first, shared, stream).unwrap();
    backend
        .copy_d2d_async(shared, first_out, first.len(), stream)
        .unwrap();
    backend.copy_h2d_async(&second, shared, stream).unwrap();
    backend
        .copy_d2d_async(shared, second_out, second.len(), stream)
        .unwrap();

    let mut first_readback = [0u8; 4];
    let mut second_readback = [0u8; 4];
    backend.copy_d2h(first_out, &mut first_readback).unwrap();
    backend.copy_d2h(second_out, &mut second_readback).unwrap();
    assert_eq!(first_readback, first);
    assert_eq!(second_readback, second);
}

/// End-to-end check: alloc → memcpy → kernel launch → memcpy back.
/// The kernel is `noop_smoke` from `kernels/metal/common/`. It
/// writes 0.0 to the first `n` floats of `out`, so after launching
/// with `n=4` the first 4 floats should be exactly zero regardless
/// of what we initialised the buffer with.
#[test]
fn metal_alloc_copy_launch_roundtrip() {
    // Pull the metallib bytes the build script embedded; skip the
    // test gracefully when no Metal device is available (CI runner).
    let Some(backend) = maybe_backend() else {
        return;
    };

    // Round-trip a known byte pattern through alloc/copy_h2d/copy_d2h.
    let bytes = 64;
    let ptr = backend.alloc(bytes).expect("alloc");
    let pattern: Vec<u8> = (0..bytes as u8).collect();
    backend.copy_h2d(&pattern, ptr).expect("copy_h2d");
    let mut readback = vec![0u8; bytes];
    backend.copy_d2h(ptr, &mut readback).expect("copy_d2h");
    assert_eq!(pattern, readback, "h2d/d2h round-trip mismatch");

    // Zero the first 4 floats via the noop_smoke kernel.
    let n: u32 = 4;
    let kernel = backend
        .kernel("noop_smoke", "noop_smoke")
        .expect("kernel lookup");
    backend
        .launch_typed(
            kernel,
            [1, 1, 1],
            [n, 1, 1],
            0,
            backend.default_stream(),
            &[KernelArg::Buffer(ptr), KernelArg::Bytes(&n.to_le_bytes())],
        )
        .expect("launch_typed");
    backend
        .synchronize(backend.default_stream())
        .expect("synchronize");

    // First 16 bytes should now be all-zero floats; the rest of
    // the buffer should retain the original pattern.
    let mut after = vec![0u8; bytes];
    backend
        .copy_d2h(ptr, &mut after)
        .expect("copy_d2h post-launch");
    assert_eq!(&after[..16], &[0u8; 16], "kernel did not zero out[0..4]");
    assert_eq!(
        &after[16..],
        &pattern[16..],
        "kernel touched out-of-range bytes"
    );

    backend.free(ptr).expect("free");
}

/// `bf16_add` parity. Trivial element-wise check — the kernel
/// is one line of math but it's the residual primitive every
/// transformer block uses, so a regression here would silently
/// blow up every layer's output.
#[test]
fn metal_bf16_add_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = 257; // odd to verify bounds-check on tail thread
    let a: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(0.1 + 0.001 * i as f32))
        .collect();
    let b: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(-0.05 + 0.0007 * i as f32))
        .collect();

    let mut expected = vec![half::bf16::ZERO; n as usize];
    for i in 0..n as usize {
        expected[i] = half::bf16::from_f32(a[i].to_f32() + b[i].to_f32());
    }

    let a_bytes = bf16_slice_to_bytes(&a);
    let b_bytes = bf16_slice_to_bytes(&b);
    let a_ptr = backend.alloc(a_bytes.len()).unwrap();
    let b_ptr = backend.alloc(b_bytes.len()).unwrap();
    let out_ptr = backend.alloc(a_bytes.len()).unwrap();
    backend.copy_h2d(&a_bytes, a_ptr).unwrap();
    backend.copy_h2d(&b_bytes, b_ptr).unwrap();

    let kernel = backend.kernel("bf16_add", "bf16_add").unwrap();
    let block: u32 = 64;
    backend
        .launch_typed(
            kernel,
            [n.div_ceil(block), 1, 1],
            [block, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Buffer(a_ptr),
                KernelArg::Buffer(b_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch bf16_add");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; a_bytes.len()];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    for i in 0..n as usize {
        assert!(
            (expected[i].to_f32() - actual[i].to_f32()).abs() < 1e-4,
            "bf16_add mismatch at idx {i}"
        );
    }

    backend.free(a_ptr).unwrap();
    backend.free(b_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}

/// `sigmoid_gate` parity. `out = sigmoid(gate) * x`. Distinct
/// from `silu_gate` (which is `gate * sigmoid(gate) * up`) —
/// Qwen3.5 uses this for `attn_output_gate`.
#[test]
fn metal_sigmoid_gate_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = 128;
    let gate: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(-3.0 + 6.0 * i as f32 / (n - 1) as f32))
        .collect();
    let x: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(0.5 + 0.01 * i as f32))
        .collect();

    let mut expected = vec![half::bf16::ZERO; n as usize];
    for i in 0..n as usize {
        let g = gate[i].to_f32();
        let v = x[i].to_f32();
        let sig = 1.0 / (1.0 + (-g).exp());
        expected[i] = half::bf16::from_f32(sig * v);
    }

    let g_bytes = bf16_slice_to_bytes(&gate);
    let x_bytes = bf16_slice_to_bytes(&x);
    let g_ptr = backend.alloc(g_bytes.len()).unwrap();
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let out_ptr = backend.alloc(g_bytes.len()).unwrap();
    backend.copy_h2d(&g_bytes, g_ptr).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();

    let kernel = backend.kernel("sigmoid_gate", "sigmoid_gate").unwrap();
    backend
        .launch_typed(
            kernel,
            [n.div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Buffer(g_ptr),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch sigmoid_gate");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; g_bytes.len()];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..n as usize {
        let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 0.02,
        "sigmoid_gate: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(g_ptr).unwrap();
    backend.free(x_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}
