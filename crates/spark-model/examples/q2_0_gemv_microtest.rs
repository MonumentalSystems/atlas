// SPDX-License-Identifier: AGPL-3.0-only
//! Numeric + bandwidth microbench for the native keep-packed ternary Q2_0
//! decode GEMV (`q2_0_gemv` / `q2_0_gemv_batchm`).
//!
//! Builds a synthetic packed Q2_0 weight `[N, K]` (random codes 0..3, random
//! fp16 per-group scales) and a random BF16 activation `[M, K]`, runs the
//! on-device kernel, and compares against an independent CPU oracle that
//! dequantizes each block (`value = (code-1)*d`) and does a dense f32 dot with
//! the bf16 activation. This proves the kernel independent of the full model.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL='*' \
//!     cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example q2_0_gemv_microtest

use anyhow::Result;
use half::{bf16, f16};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const N: usize = 2048;
const K: usize = 4096;
const GROUP: usize = 128;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f(&mut self) -> f32 {
        (((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn code(&mut self) -> u8 {
        (self.next_u64() >> 40) as u8 & 3
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn up_u8(g: &dyn GpuBackend, d: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(d.len().max(1))?;
    g.copy_h2d(d, p)?;
    Ok(p)
}
fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

/// Pack one weight row (`K` codes + `K/GROUP` fp16 scales) into contiguous
/// `block_q2_0` bytes: `[fp16 d][GROUP/4 bytes, 4 codes/byte, low-bits-first]`.
fn pack_row(codes: &[u8], scales: &[f32]) -> Vec<u8> {
    let block_bytes = 2 + GROUP / 4;
    let mut out = Vec::with_capacity((K / GROUP) * block_bytes);
    for (b, &d) in scales.iter().enumerate() {
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let blk = &codes[b * GROUP..(b + 1) * GROUP];
        for chunk in blk.chunks(4) {
            let mut byte = 0u8;
            for (t, &c) in chunk.iter().enumerate() {
                byte |= (c & 3) << (2 * t as u8);
            }
            out.push(byte);
        }
    }
    out
}

/// Independent CPU oracle: dequant `(code-1)*d` per element then a dense f32
/// dot with the bf16-rounded activation. `out[m,n] = sum_k a[m,k]*(code-1)*d`.
fn oracle(codes: &[u8], scales: &[f32], a: &[bf16], m: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * N];
    for row in 0..m {
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                let code = codes[n * K + k] as i32;
                let d = scales[n * (K / GROUP) + k / GROUP];
                acc += a[row * K + k].to_f32() * (code - 1) as f32 * d;
            }
            out[row * N + n] = acc;
        }
    }
    out
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let mut max = 0f32;
    let mut denom = 0f32;
    for w in want {
        denom = denom.max(w.abs());
    }
    let denom = denom.max(1e-6);
    for (g, w) in got.iter().zip(want) {
        max = max.max((g - w).abs() / denom);
    }
    max
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([div_ceil(N as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(N as u32)
        .arg_u32(K as u32)
        .arg_u32(GROUP as u32)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn launch_batchm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([div_ceil(N as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(N as u32)
        .arg_u32(K as u32)
        .arg_u32(GROUP as u32)
        .arg_u32(m)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let gemv_k = g.kernel("q2_0_gemv", "q2_0_gemv")?;
    let batchm_k = g.kernel("q2_0_gemv", "q2_0_gemv_batchm")?;

    let mut rng = Lcg(0x715f_2b30_9f01_0001);
    let codes: Vec<u8> = (0..N * K).map(|_| rng.code()).collect();
    let scales: Vec<f32> = (0..N * (K / GROUP)).map(|_| rng.r(-0.06, 0.06)).collect();
    // Pack row-major [N, K] into contiguous block_q2_0 bytes.
    let mut packed = Vec::with_capacity(N * (K / GROUP) * (2 + GROUP / 4));
    for n in 0..N {
        packed.extend(pack_row(
            &codes[n * K..(n + 1) * K],
            &scales[n * (K / GROUP)..(n + 1) * (K / GROUP)],
        ));
    }
    let weight_bytes = packed.len();

    const MAXM: usize = 8;
    let a: Vec<bf16> = (0..MAXM * K).map(|_| bf16::from_f32(rng.r(-1.0, 1.0))).collect();

    let a_d = up_bf16(g, &a)?;
    let w_d = up_u8(g, &packed)?;
    let c_d = g.alloc(MAXM * N * 2)?;

    let mut all_pass = true;

    // --- M=1 decode GEMV ---
    launch_gemv(g, gemv_k, a_d, w_d, c_d)?;
    g.synchronize(0)?;
    let got = dn_bf16(g, c_d, N)?;
    let want = oracle(&codes, &scales, &a, 1);
    let err = rel_err(&got, &want);
    let pass = err < 1e-2;
    all_pass &= pass;
    println!("q2_0_gemv M=1: max_rel_err={err:.5} {}", if pass { "PASS" } else { "FAIL" });

    // --- Batched M=2,4,8 ---
    for m in [2usize, 4, 8] {
        launch_batchm(g, batchm_k, a_d, w_d, c_d, m as u32)?;
        g.synchronize(0)?;
        let got = dn_bf16(g, c_d, m * N)?;
        let want = oracle(&codes, &scales, &a, m);
        let err = rel_err(&got, &want);
        let pass = err < 1e-2;
        all_pass &= pass;
        println!(
            "q2_0_gemv_batchm M={m}: max_rel_err={err:.5} {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // --- Bandwidth (M=1 decode, weight-byte-bound) ---
    let iters = 200;
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        launch_gemv(g, gemv_k, a_d, w_d, c_d)?;
    }
    g.synchronize(0)?;
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let gbps = weight_bytes as f64 / dt / 1e9;
    println!(
        "q2_0_gemv M=1: {:.1} us/call, {:.1} GB/s weight-read ({} weight bytes, N={N} K={K})",
        dt * 1e6,
        gbps,
        weight_bytes
    );

    if all_pass {
        println!("ALL PASS");
        Ok(())
    } else {
        std::process::exit(1);
    }
}
