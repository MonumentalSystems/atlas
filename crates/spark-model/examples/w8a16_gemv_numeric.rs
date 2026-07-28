// SPDX-License-Identifier: AGPL-3.0-only

//! NUMERIC correctness check for `w8a16_gemv` at the Nemotron-Puzzle SSM shapes.
//!
//! The sibling `gemv_fp4_vs_fp8_microtest` is timing-only — it never checks an
//! output value. That left a real gap: enabling native FP8 for the Puzzle-75B
//! Mamba2 projections made the model measurably WORSE (correct "Alice" became an
//! invented "Charlie"), and a bisect pinned it to the DECODE path, i.e. this
//! kernel — yet the scale value, scale shape, broadcast, kernel indexing, LUT and
//! call-site argument order all inspect as correct. So test the arithmetic.
//!
//! Construction is chosen so the expected answer is exact and obvious:
//!   A[k]         = 1.0        for all k
//!   B[n,k]       = 0x38       = 1.0 in E4M3 (sign 0, exp 0111 -> 2^0, mantissa 0)
//!   block_scale  = S          constant over every block
//! then  C[n] = sum_k 1.0 * 1.0 * S = K * S  for every n.
//!
//! A per-block vs per-row scale-layout mix-up still yields the right answer under
//! a CONSTANT scale, so the run also repeats with a scale that VARIES per block —
//! `scale[nb, kb] = 1 + nb + 100*kb` — where the two layouts disagree and the
//! expected value per row is a closed form: C[n] = 128 * sum_kb (1 + nb + 100*kb).
//!
//! Usage: cargo run --release -p spark-model --example w8a16_gemv_numeric -- [N] [K]
//! Default N=18048 K=4096 (Puzzle SSM in_proj). Also try 4096 8192 (out_proj).

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;

const FP8_BLOCK: usize = 128;

fn bf16_bytes(v: f32) -> [u8; 2] {
    // Round-to-nearest-even truncation of f32 -> bf16, as the loaders do.
    let bits = v.to_bits();
    let rounded = ((bits >> 16) & 1).wrapping_add(0x7fff).wrapping_add(bits);
    ((rounded >> 16) as u16).to_le_bytes()
}

fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((u16::from_le_bytes([lo, hi])) as u32) << 16)
}

fn run(gpu: &dyn GpuBackend, stream: u64, n: u32, k: u32, vary: bool) -> Result<()> {
    let nb = (n as usize).div_ceil(FP8_BLOCK);
    let kb = (k as usize).div_ceil(FP8_BLOCK);

    // A = all ones (BF16)
    let a_host: Vec<u8> = (0..k).flat_map(|_| bf16_bytes(1.0)).collect();
    let a = gpu.alloc(a_host.len())?;
    gpu.copy_h2d(&a_host, a)?;

    // B = all 0x38 (= 1.0 in E4M3), [N, K] row-major
    let b_host = vec![0x38u8; (n as usize) * (k as usize)];
    let b = gpu.alloc(b_host.len())?;
    gpu.copy_h2d(&b_host, b)?;

    // block_scale [ceil(N/128), ceil(K/128)] FP32, row-major — the layout the
    // kernel documents and the loader materializes.
    let mut s_host = Vec::with_capacity(nb * kb * 4);
    for i in 0..nb {
        for j in 0..kb {
            let s = if vary {
                1.0f32 + i as f32 + 100.0 * j as f32
            } else {
                0.00088065f32 // the real Puzzle L0 in_proj weight_scale
            };
            s_host.extend_from_slice(&s.to_le_bytes());
        }
    }
    let s = gpu.alloc(s_host.len())?;
    gpu.copy_h2d(&s_host, s)?;

    let c = gpu.alloc((n as usize) * 2)?;
    let kern = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    KernelLaunch::new(gpu, kern)
        .grid([n.div_ceil(4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(s)
        .arg_ptr(c)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    let mut c_host = vec![0u8; (n as usize) * 2];
    gpu.copy_d2h(c, &mut c_host)?;

    // Expected, per row n: sum over k of scale[n/128, k/128].
    // Each k-block contributes min(128, K - kb*128) terms.
    let expect = |row: usize| -> f32 {
        let i = row / FP8_BLOCK;
        (0..kb)
            .map(|j| {
                let width = ((k as usize) - j * FP8_BLOCK).min(FP8_BLOCK) as f32;
                let sv = if vary {
                    1.0f32 + i as f32 + 100.0 * j as f32
                } else {
                    0.00088065f32
                };
                sv * width
            })
            .sum()
    };

    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    let mut first_bad: Option<(usize, f32, f32)> = None;
    for row in 0..(n as usize) {
        let got = bf16_to_f32(c_host[row * 2], c_host[row * 2 + 1]);
        let want = expect(row);
        let rel = if want.abs() > 0.0 {
            ((got - want) / want).abs()
        } else {
            got.abs()
        };
        if rel > worst {
            worst = rel;
            worst_row = row;
        }
        // BF16 carries ~3 decimal digits; 2% covers accumulation order effects.
        if rel > 0.02 && first_bad.is_none() {
            first_bad = Some((row, got, want));
        }
    }

    let label = if vary { "VARYING block scale" } else { "constant scale" };
    println!("  {label}: N={n} K={k} blocks=[{nb},{kb}]");
    println!(
        "    row0: got={:.6} want={:.6}",
        bf16_to_f32(c_host[0], c_host[1]),
        expect(0)
    );
    if let Some((row, got, want)) = first_bad {
        println!("    FAIL first bad row {row}: got={got:.6} want={want:.6}");
        println!("    worst rel err {:.4} at row {worst_row}", worst);
    } else {
        println!("    PASS (worst rel err {:.6} at row {worst_row})", worst);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n: u32 = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(18048);
    let k: u32 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4096);
    if k % 16 != 0 {
        bail!("w8a16_gemv requires K%16==0");
    }
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    println!("w8a16_gemv numeric check");
    run(&gpu, stream, n, k, false)?;
    run(&gpu, stream, n, k, true)?;
    Ok(())
}
