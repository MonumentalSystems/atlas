// SPDX-License-Identifier: AGPL-3.0-only
//! Correctness + speed for `int8_gemm_t_m128` (W4A8 prefill core).
//! Correctness: small shape vs an exact host reference of the per-block-scaled
//! int8 GEMM (cosine ~1.0 proves the MMA + dequant indexing). Speed: prefill shapes.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn i8(&mut self) -> i8 {
        (self.next_u64() % 255) as i8 - 127
    }
    fn pos_scale(&mut self) -> f32 {
        0.001 + (self.next_u64() % 1000) as f32 * 0.0005
    }
}

fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn run(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    asc: DevicePtr,
    bsc: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(a).arg_ptr(b).arg_ptr(asc).arg_ptr(bsc).arg_ptr(c)
        .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32)
        .launch(stream)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let h = gpu.kernel("w4a16", "int8_gemm_t_m128")?;
    let h64 = gpu.kernel("w4a16", "int8_gemm_t_m64")?;
    let hk64 = gpu.kernel("w4a16", "int8_gemm_t_m128_k64")?;
    let h8w = gpu.kernel("w4a16", "int8_gemm_8w")?;
    let h8w3 = gpu.kernel("w4a16", "int8_gemm_8w3")?;
    let h8wl = gpu.kernel("w4a16", "int8_gemm_8w_ldm")?;
    let h8wi = gpu.kernel("w4a16", "int8_gemm_8w_ilp")?;
    let h8wab = gpu.kernel("w4a16", "int8_gemm_8w_ldmab")?;
    let hpipe = gpu.kernel("w4a16", "int8_gemm_8w_pipe")?;
    let hpada = gpu.kernel("w4a16", "int8_gemm_padA")?;
    let hfaith = gpu.kernel("w4a16", "int8_gemm_faith")?;
    let hfaith2 = gpu.kernel("w4a16", "int8_gemm_faith2")?;
    let hmmq = gpu.kernel("w4a16", "int8_gemm_mmq")?;
    let hsk = gpu.kernel("w4a16", "int8_gemm_splitk")?;
    let hred = gpu.kernel("w4a16", "int8_splitk_reduce")?;

    // ---- correctness: small shape, exact host reference ----
    let (m, n, k) = (128usize, 256usize, 512usize);
    let nb = k / 32;
    let mut rng = Rng(0xC0FFEE);
    let a_i8: Vec<i8> = (0..m * k).map(|_| rng.i8()).collect();
    let b_i8: Vec<i8> = (0..n * k).map(|_| rng.i8()).collect();
    let a_sc: Vec<f32> = (0..m * nb).map(|_| rng.pos_scale()).collect();
    let b_sc: Vec<f32> = (0..n * nb).map(|_| rng.pos_scale()).collect();

    // host ref: C[m,n] = sum_blk ( sum_{k in blk} A[m,k]*B[n,k] ) * As[m,blk]*Bs[n,blk]
    let mut c_ref = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for blk in 0..nb {
                let mut s = 0i32;
                for kk in 0..32 {
                    let ki = blk * 32 + kk;
                    s += a_i8[mi * k + ki] as i32 * b_i8[ni * k + ki] as i32;
                }
                acc += s as f32 * a_sc[mi * nb + blk] * b_sc[ni * nb + blk];
            }
            c_ref[mi * n + ni] = acc;
        }
    }

    let a_p = up(gpu, bytemuck_i8(&a_i8))?;
    let b_p = up(gpu, bytemuck_i8(&b_i8))?;
    let as_p = up(gpu, bytemuck_f32(&a_sc))?;
    let bs_p = up(gpu, bytemuck_f32(&b_sc))?;
    let c_p = gpu.alloc(m * n * 2)?;
    run(gpu, stream, h, a_p, b_p, as_p, bs_p, c_p, m, n, k)?;
    gpu.synchronize(stream)?;
    let mut raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(c_p, &mut raw)?;
    let c_gpu: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let (mut dot, mut na, mut nbb) = (0f64, 0f64, 0f64);
    let mut maxrel = 0f64;
    for i in 0..m * n {
        let (x, y) = (c_ref[i] as f64, c_gpu[i] as f64);
        dot += x * y;
        na += x * x;
        nbb += y * y;
        if x.abs() > 1.0 {
            maxrel = maxrel.max(((x - y).abs() / x.abs()).min(9.9));
        }
    }
    let cos = dot / (na.sqrt() * nbb.sqrt());
    println!("int8_gemm correctness {m}x{n}x{k}: cosine={cos:.6}  max_rel={maxrel:.4}");
    println!(
        "  ref[0..4]={:?}  gpu[0..4]={:?}",
        &c_ref[..4],
        &c_gpu[..4]
    );
    let pass = cos > 0.999;
    println!("  RESULT: {}", if pass { "PASS" } else { "FAIL" });

    // ---- 8w_ldm correctness (ldmatrix fragment must match host ref) ----
    {
        let c2 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, h8wl)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
            .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c2)
            .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)?;
        gpu.synchronize(stream)?;
        let mut raw2 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c2, &mut raw2)?;
        let cg: Vec<f32> = raw2.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n { let (x, y) = (c_ref[i] as f64, cg[i] as f64); d += x*y; nr += x*x; ng += y*y; }
        let cl = d / (nr.sqrt() * ng.sqrt());
        println!("8w_ldm (ldmatrix.x4) correctness: cosine={cl:.6}  RESULT: {}",
            if cl > 0.999 { "PASS" } else { "FAIL" });
        let _ = gpu.free(c2);
    }
    {
        let c3 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hmmq)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
            .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c3)
            .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r3 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c3, &mut r3)?;
        let cg: Vec<f32> = r3.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m*n { let (x,y)=(c_ref[i] as f64, cg[i] as f64); d+=x*y; nr+=x*x; ng+=y*y; }
        println!("MMQ-tile correctness: cosine={:.6}  RESULT: {}", d/(nr.sqrt()*ng.sqrt()),
            if d/(nr.sqrt()*ng.sqrt())>0.999 {"PASS"} else {"FAIL"});
        let _ = gpu.free(c3);
    }
    {
        let c4 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, h8wab)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
            .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c4)
            .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r4 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c4, &mut r4)?;
        let cg: Vec<f32> = r4.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m*n { let (x,y)=(c_ref[i] as f64, cg[i] as f64); d+=x*y; nr+=x*x; ng+=y*y; }
        println!("8w_ldmAB (ldmatrix A+B) correctness: cosine={:.6}  RESULT: {}", d/(nr.sqrt()*ng.sqrt()),
            if d/(nr.sqrt()*ng.sqrt())>0.999 {"PASS"} else {"FAIL"});
        let _ = gpu.free(c4);
    }
    {
        let c5 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hpipe)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([512, 1, 1])
            .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c5)
            .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r5 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c5, &mut r5)?;
        let cg: Vec<f32> = r5.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m*n { let (x,y)=(c_ref[i] as f64, cg[i] as f64); d+=x*y; nr+=x*x; ng+=y*y; }
        println!("8w_pipe (occ 512) correctness: cosine={:.6}  RESULT: {}", d/(nr.sqrt()*ng.sqrt()),
            if d/(nr.sqrt()*ng.sqrt())>0.999 {"PASS"} else {"FAIL"});
        let _ = gpu.free(c5);
    }
    {
        let c6 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hpada)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
            .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c6)
            .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r6 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c6, &mut r6)?;
        let cg: Vec<f32> = r6.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m*n { let (x,y)=(c_ref[i] as f64, cg[i] as f64); d+=x*y; nr+=x*x; ng+=y*y; }
        println!("padA (bank-fix ldmatrix) correctness: cosine={:.6}  RESULT: {}", d/(nr.sqrt()*ng.sqrt()),
            if d/(nr.sqrt()*ng.sqrt())>0.999 {"PASS"} else {"FAIL"});
        let _ = gpu.free(c6);
    }
    {
        let c7 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hfaith)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
            .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c7)
            .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r7 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c7, &mut r7)?;
        let cg: Vec<f32> = r7.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m*n { let (x,y)=(c_ref[i] as f64, cg[i] as f64); d+=x*y; nr+=x*x; ng+=y*y; }
        println!("FAITH (llama-MMQ port) correctness: cosine={:.6}  RESULT: {}", d/(nr.sqrt()*ng.sqrt()),
            if d/(nr.sqrt()*ng.sqrt())>0.999 {"PASS"} else {"FAIL"});
        let _ = gpu.free(c7);
    }
    {
        let c8 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hfaith2)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
            .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c8)
            .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r8 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c8, &mut r8)?;
        let cg: Vec<f32> = r8.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m*n { let (x,y)=(c_ref[i] as f64, cg[i] as f64); d+=x*y; nr+=x*x; ng+=y*y; }
        println!("FAITH2 (big-K rolling) correctness: cosine={:.6}  RESULT: {}", d/(nr.sqrt()*ng.sqrt()),
            if d/(nr.sqrt()*ng.sqrt())>0.999 {"PASS"} else {"FAIL"});
        let _ = gpu.free(c8);
    }

    // ---- speed: prefill shapes ----
    println!("\n=== int8 speed (TFLOP/s) ===");
    for &(label, m, n, k) in &[
        ("gate/up M=4096", 4096usize, 17408usize, 5120usize),
        ("down    M=4096", 4096, 5120, 17408),
    ] {
        let nb = k / 32;
        let mut rng = Rng(7);
        let a_i8: Vec<i8> = (0..m * k).map(|_| rng.i8()).collect();
        let b_i8: Vec<i8> = (0..n * k).map(|_| rng.i8()).collect();
        let a_sc: Vec<f32> = (0..m * nb).map(|_| rng.pos_scale()).collect();
        let b_sc: Vec<f32> = (0..n * nb).map(|_| rng.pos_scale()).collect();
        let a_p = up(gpu, bytemuck_i8(&a_i8))?;
        let b_p = up(gpu, bytemuck_i8(&b_i8))?;
        let as_p = up(gpu, bytemuck_f32(&a_sc))?;
        let bs_p = up(gpu, bytemuck_f32(&b_sc))?;
        let c_p = gpu.alloc(m * n * 2)?;
        for _ in 0..3 {
            run(gpu, stream, h, a_p, b_p, as_p, bs_p, c_p, m, n, k)?;
        }
        gpu.synchronize(stream)?;
        let iters = 30;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        // M128
        let t0 = Instant::now();
        for _ in 0..iters { run(gpu, stream, h, a_p, b_p, as_p, bs_p, c_p, m, n, k)?; }
        gpu.synchronize(stream)?;
        let tf128 = flops / (t0.elapsed().as_secs_f64() / iters as f64) / 1e12;
        // M64 (grid m/64)
        let launch64 = || -> Result<()> {
            KernelLaunch::new(gpu, h64)
                .grid([n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 { launch64()?; }
        gpu.synchronize(stream)?;
        let t1 = Instant::now();
        for _ in 0..iters { launch64()?; }
        gpu.synchronize(stream)?;
        let tf64 = flops / (t1.elapsed().as_secs_f64() / iters as f64) / 1e12;
        // K_STEP=64 (grid m/128)
        let launchk64 = || -> Result<()> {
            KernelLaunch::new(gpu, hk64)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([128, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launchk64()?; }
        gpu.synchronize(stream)?;
        let tk = Instant::now();
        for _ in 0..iters { launchk64()?; }
        gpu.synchronize(stream)?;
        let tfk64 = flops / (tk.elapsed().as_secs_f64() / iters as f64) / 1e12;
        // 8-warp (block 256, grid m/128 n/128)
        let launch8w = || -> Result<()> {
            KernelLaunch::new(gpu, h8w)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launch8w()?; }
        gpu.synchronize(stream)?;
        let t8 = Instant::now();
        for _ in 0..iters { launch8w()?; }
        gpu.synchronize(stream)?;
        let tf8w = flops / (t8.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launch8w3 = || -> Result<()> {
            KernelLaunch::new(gpu, h8w3)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launch8w3()?; }
        gpu.synchronize(stream)?;
        let t83 = Instant::now();
        for _ in 0..iters { launch8w3()?; }
        gpu.synchronize(stream)?;
        let tf8w3 = flops / (t83.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launch8wl = || -> Result<()> {
            KernelLaunch::new(gpu, h8wl)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launch8wl()?; }
        gpu.synchronize(stream)?;
        let tl = Instant::now();
        for _ in 0..iters { launch8wl()?; }
        gpu.synchronize(stream)?;
        let tf8wl = flops / (tl.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launch8wi = || -> Result<()> {
            KernelLaunch::new(gpu, h8wi)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launch8wi()?; }
        gpu.synchronize(stream)?;
        let ti = Instant::now();
        for _ in 0..iters { launch8wi()?; }
        gpu.synchronize(stream)?;
        let tf8wi = flops / (ti.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchmmq = || -> Result<()> {
            KernelLaunch::new(gpu, hmmq)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launchmmq()?; }
        gpu.synchronize(stream)?;
        let tmq = Instant::now();
        for _ in 0..iters { launchmmq()?; }
        gpu.synchronize(stream)?;
        let tfmmq = flops / (tmq.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launch8wab = || -> Result<()> {
            KernelLaunch::new(gpu, h8wab)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launch8wab()?; }
        gpu.synchronize(stream)?;
        let tab = Instant::now();
        for _ in 0..iters { launch8wab()?; }
        gpu.synchronize(stream)?;
        let tf8wab = flops / (tab.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchpipe = || -> Result<()> {
            KernelLaunch::new(gpu, hpipe)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([512, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launchpipe()?; }
        gpu.synchronize(stream)?;
        let tp = Instant::now();
        for _ in 0..iters { launchpipe()?; }
        gpu.synchronize(stream)?;
        let tfpipe = flops / (tp.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchpada = || -> Result<()> {
            KernelLaunch::new(gpu, hpada)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launchpada()?; }
        gpu.synchronize(stream)?;
        let tpa = Instant::now();
        for _ in 0..iters { launchpada()?; }
        gpu.synchronize(stream)?;
        let tfpada = flops / (tpa.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launchfaith()?; }
        gpu.synchronize(stream)?;
        let tfa = Instant::now();
        for _ in 0..iters { launchfaith()?; }
        gpu.synchronize(stream)?;
        let tffaith = flops / (tfa.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith2 = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith2)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]).block([256, 1, 1])
                .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(c_p)
                .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).launch(stream)
        };
        for _ in 0..3 { launchfaith2()?; }
        gpu.synchronize(stream)?;
        let tf2 = Instant::now();
        for _ in 0..iters { launchfaith2()?; }
        gpu.synchronize(stream)?;
        let tffaith2 = flops / (tf2.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let _ = (tf64, tfk64, tf8w3, tf8w, tf8wl, tf8wi, tfpipe, tfpada, tf8wab);
        print!("{label}: M128 {tf128:.2} | padA {tfpada:.2} | FAITH {tffaith:.2} | FAITH2 {tffaith2:.2} | MMQ {tfmmq:.2}  (bf16=30, llama=60)");
        // split-K sweep (partial + reduce)
        for &ks in &[2u32, 4, 8, 16] {
            let cp = gpu.alloc(ks as usize * m * n * 4)?;
            let mn = (m * n) as u32;
            let go = || -> Result<()> {
                KernelLaunch::new(gpu, hsk)
                    .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, ks]).block([128, 1, 1])
                    .arg_ptr(a_p).arg_ptr(b_p).arg_ptr(as_p).arg_ptr(bs_p).arg_ptr(cp)
                    .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32).arg_u32(ks)
                    .launch(stream)?;
                KernelLaunch::new(gpu, hred)
                    .grid([mn.div_ceil(256), 1, 1]).block([256, 1, 1])
                    .arg_ptr(cp).arg_ptr(c_p).arg_u32(m as u32).arg_u32(n as u32).arg_u32(ks)
                    .launch(stream)
            };
            for _ in 0..3 { go()?; }
            gpu.synchronize(stream)?;
            let t = Instant::now();
            for _ in 0..iters { go()?; }
            gpu.synchronize(stream)?;
            let tf = flops / (t.elapsed().as_secs_f64() / iters as f64) / 1e12;
            print!(" | sk{ks}={tf:.1}");
            let _ = gpu.free(cp);
        }
        println!("   (v2 bf16=30)");
        for p in [a_p, b_p, as_p, bs_p, c_p] {
            let _ = gpu.free(p);
        }
    }
    Ok(())
}

fn bytemuck_i8(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
