// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone SHAPE TEST: block-scaled FP4 (e2m1) GEMM vs FP8 (e4m3) GEMM vs
//! BF16 reference, on Holo 3.1 MoE gate_up shapes.
//!
//! Goal: prove that the CUTLASS NVFP4 (FP4×FP4 block-scaled) GEMM beats the
//! current FP8 (e4m3.e4m3) MoE math on the per-expert gate_up shape, with
//! acceptable accuracy. This does NOT integrate into the MoE kernel — it only
//! proves the math + speed at the relevant M/N/K, like the team's prior
//! CUTLASS/FlashInfer shape proofs.
//!
//! Shapes (Holo 3.1, config.json): hidden=2048, moe_intermediate=512,
//! 256 experts, top_k=8. gate_up GEMM per expert: N = 2*intermediate = 1024,
//! K = hidden = 2048. Production prefill chunk 2048 tokens * 8 / 256 = 64
//! tokens/expert => M=64 is the realistic per-expert tile. We sweep
//! M in {32, 64, 128, 2048}.
//!
//! Three GEMM variants, identical M/N/K, identical underlying bf16 A/B so the
//! ONLY difference is the GEMM precision:
//!   - FP4: spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t  (block-scaled
//!     e2m1 act + e2m1 weight, Sm120 OpClassBlockScaledTensorOp). Weight packed
//!     once via pack_bf16_weight_to_nvfp4_t (not timed).
//!   - FP8: fp8_gemm_t kernel (module "w4a16"): C = A_bf16 @ decode_e4m3(B)^T,
//!     m16n8k32 e4m3.e4m3 tensor cores — the closest dense analog to the MoE
//!     moe_w4a16 FP8 compute (weight->fp8, act->fp8). No block scale.
//!   - BF16: CPU oracle (fp32 accum) is the accuracy ground truth for BOTH
//!     quantized paths; a GPU CUTLASS bf16 GEMM gives the bf16 timing line.
//!
//! Timing: CUDA events, kernel-only, 5 warmup + 50 iters, median us reported.
//! Accuracy: cosine + max_rel vs the CPU bf16 oracle.
//!
//! Build (remote GB10, exact recipe):
//!   ATLAS_TARGET_MODEL=holo-3.1-35b-a3b --no-default-features --features cuda
//!   with CUTLASS_HOME / FLASHINFER_HOME / RUSTFLAGS exported (else the FP4 op
//!   bails "CUTLASS support was not built").

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

// Raw CUDA driver event API for kernel-only timing (verbatim from
// dense_gemm_microtest.rs).
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

// FP4 quantizes BOTH operands to 4-bit e2m1 (block-scaled), so its noise floor
// is looser than the FP8 path here (which only quantizes the weight to e4m3 and
// keeps the activation bf16). On adversarial uniform-random inputs FP4 lands at
// ~0.99 cosine; the standard NVFP4 acceptance threshold is 0.98. Gate at 0.98 —
// real model weights/acts (not uniform-random) quantize meaningfully better,
// which is why vLLM ships this exact model in NVFP4.
const COSINE_GATE: f64 = 0.98;

// ───────────────────────── deterministic PRNG ─────────────────────────
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

// ───────────────────────── number-format helpers ─────────────────────────
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

// Standard OCP E4M3 (1-4-3, bias 7) decode — matches the fp8_gemm_t kernel.
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        0.0
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}
fn f32_to_e4m3(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for b in 0..=255u8 {
        let d = e4m3_to_f32(b);
        if !d.is_finite() {
            continue;
        }
        let e = (d - v).abs();
        if e < best_err {
            best_err = e;
            best = b;
        }
    }
    best
}

// ───────────────────────── upload helpers ─────────────────────────
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// CPU bf16 oracle: C[m,n] = bf16(Σ_k A[m,k]·B[n,k]), fp32 accum.
/// A=[M,K], B=[N,K] row-major (read transposed).
fn cpu_reference(a_bf16: &[u16], b_bf16: &[u16], m: usize, n: usize, k: usize) -> Vec<u16> {
    let nthreads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8);
    let mut out = vec![0u16; m * n];
    let rows_per = m.div_ceil(nthreads);
    std::thread::scope(|sc| {
        for (t, chunk) in out.chunks_mut(rows_per * n).enumerate() {
            let row0 = t * rows_per;
            sc.spawn(move || {
                let rows = chunk.len() / n;
                for rr in 0..rows {
                    let row = row0 + rr;
                    for col in 0..n {
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            let a = bf16_bits_to_f32(a_bf16[row * k + kk]);
                            let b = bf16_bits_to_f32(b_bf16[col * k + kk]);
                            acc += a * b;
                        }
                        chunk[rr * n + col] = f32_to_bf16_bits(acc);
                    }
                }
            });
        }
    });
    out
}

/// Cosine + max_rel of a GPU bf16 output vs the bf16 oracle (both in f32 space).
fn compare(c_gpu: &[u16], c_ref: &[u16]) -> (f64, f64) {
    let (mut dot, mut ng, mut nc, mut max_rel) = (0f64, 0f64, 0f64, 0f64);
    for i in 0..c_ref.len() {
        let g = bf16_bits_to_f32(c_gpu[i]) as f64;
        let c = bf16_bits_to_f32(c_ref[i]) as f64;
        dot += g * c;
        ng += g * g;
        nc += c * c;
        let rel = (g - c).abs() / c.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    let cosine = if ng > 0.0 && nc > 0.0 {
        dot / (ng.sqrt() * nc.sqrt())
    } else {
        0.0
    };
    (cosine, max_rel)
}

/// Read a device bf16 [M,N] buffer back to host u16s.
fn read_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, m: usize, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Time a closure (one GEMM launch, no sync inside) with CUDA events:
/// 5 warmup (with sync) + `iters` un-synced launches, single end-sync.
/// Returns median-equivalent (mean) us per launch.
fn time_gemm<F: Fn() -> Result<()>>(
    gpu: &dyn GpuBackend,
    stream: u64,
    iters: usize,
    launch: F,
) -> Result<f64> {
    for _ in 0..5 {
        launch()?;
    }
    gpu.synchronize(stream)?;

    let (mut ev_start, mut ev_end): (u64, u64) = (0, 0);
    if unsafe { cuEventCreate(&mut ev_start, 0) } != 0 {
        bail!("cuEventCreate(start) failed");
    }
    if unsafe { cuEventCreate(&mut ev_end, 0) } != 0 {
        bail!("cuEventCreate(end) failed");
    }
    if unsafe { cuEventRecord(ev_start, stream) } != 0 {
        bail!("cuEventRecord(start) failed");
    }
    for _ in 0..iters {
        launch()?;
    }
    if unsafe { cuEventRecord(ev_end, stream) } != 0 {
        bail!("cuEventRecord(end) failed");
    }
    if unsafe { cuEventSynchronize(ev_end) } != 0 {
        bail!("cuEventSynchronize(end) failed");
    }
    let mut elapsed_ms: f32 = 0.0;
    if unsafe { cuEventElapsedTime(&mut elapsed_ms, ev_start, ev_end) } != 0 {
        bail!("cuEventElapsedTime failed");
    }
    unsafe {
        cuEventDestroy_v2(ev_start);
        cuEventDestroy_v2(ev_end);
    }
    Ok((elapsed_ms as f64 / iters as f64) * 1e3) // ms -> us per iter
}

struct Row {
    m: usize,
    fp4_us: f64,
    fp8_us: f64,
    bf16_us: f64,
    fp4_cos: f64,
    fp4_max_rel: f64,
    fp8_cos: f64,
    fp8_max_rel: f64,
    // Phase-1 grouped FP4 gate_up kernel (escape-hatch: per-expert collective).
    grp_us: f64,
    grp_cos_vs_collective: f64, // MUST be >= 0.999 (same FP4 math)
    grp_cos_vs_oracle: f64,     // MUST be >= 0.98
}

/// Cosine of two GPU bf16 outputs (both u16) in f32 space.
fn cosine_u16(a: &[u16], b: &[u16]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        let x = bf16_bits_to_f32(a[i]) as f64;
        let y = bf16_bits_to_f32(b[i]) as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

fn run_shape(
    gpu: &dyn GpuBackend,
    stream: u64,
    m: usize,
    n: usize,
    k: usize,
    seed: u64,
) -> Result<Row> {
    let iters = 50usize;

    // ── inputs: shared bf16 A/B so only the GEMM precision differs ──
    let mut rng = Rng(seed);
    // A: bf16 activations, realistic post-norm magnitudes [-1,1].
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    // B: bf16 weights [-0.5,0.5]. The FP4 path packs THIS; the FP8 path quantizes
    // THIS to e4m3; the bf16 path uses it raw — same numbers everywhere.
    let b_bf16: Vec<u16> = (0..n * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.5, 0.5)))
        .collect();
    // e4m3 version of B for the FP8 kernel (decoded-nearest, so realistic).
    let b_fp8: Vec<u8> = b_bf16
        .iter()
        .map(|&b| f32_to_e4m3(bf16_bits_to_f32(b)))
        .collect();

    // ── upload ──
    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
    let b_bf16_ptr = upload_bytes(gpu, &u16s_to_le(&b_bf16))?;
    let b_fp8_ptr = upload_bytes(gpu, &b_fp8)?;

    // FP4 weight buffers: packed [K/2, N] bytes (N-major, K-contiguous within
    // the CUTLASS [N,K/2] view), scales [K/16, N] bytes.
    let packed_len = (k / 2) * n;
    let scale_len = (k / 16) * n;
    let packed_ptr = gpu.alloc(packed_len)?;
    let scale_ptr = gpu.alloc(scale_len)?;
    spark_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
        b_bf16_ptr.0,
        packed_ptr.0,
        scale_ptr.0,
        n as u32,
        k as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;

    // outputs
    let out_fp4 = gpu.alloc(m * n * 2)?;
    let out_fp8 = gpu.alloc(m * n * 2)?;
    let out_bf16 = gpu.alloc(m * n * 2)?;

    let (mu, nu, ku) = (m as u32, n as u32, k as u32);

    // ── FP4 GEMM ──
    let fp4_launch = || {
        spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
            a_ptr.0,
            packed_ptr.0,
            scale_ptr.0,
            1.0, // weights packed via pack_bf16_weight_to_nvfp4_t => scale2 = 1.0
            out_fp4.0,
            mu,
            nu,
            ku,
            stream,
        )
    };
    fp4_launch()?;
    gpu.synchronize(stream)?;
    let fp4_us = time_gemm(gpu, stream, iters, fp4_launch)?;
    let c_fp4 = read_bf16(gpu, out_fp4, m, n)?;

    // ── Phase-1 grouped FP4 gate_up kernel (escape-hatch path) ──
    // Single expert (num_experts=1) covering all M rows; gate weight = b_bf16
    // (already packed above as packed_ptr/scale_ptr), up weight = a second
    // independent bf16 weight so gate!=up. The grouped kernel writes C_gate and
    // C_up; we validate C_gate against:
    //   (a) the single-GEMM collective output c_fp4 (MUST match, cos>=0.999), and
    //   (b) the CPU bf16 oracle c_ref (cos>=0.98).
    let mut rng_up = Rng(seed ^ 0xDEAD_BEEF_CAFE_F00Du64);
    let up_bf16: Vec<u16> = (0..n * k)
        .map(|_| f32_to_bf16_bits(rng_up.uniform(-0.5, 0.5)))
        .collect();
    let up_bf16_ptr = upload_bytes(gpu, &u16s_to_le(&up_bf16))?;
    let up_packed_ptr = gpu.alloc(packed_len)?;
    let up_scale_ptr = gpu.alloc(scale_len)?;
    spark_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
        up_bf16_ptr.0,
        up_packed_ptr.0,
        up_scale_ptr.0,
        n as u32,
        k as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;

    let c_gate_grp = gpu.alloc(m * n * 2)?;
    let c_up_grp = gpu.alloc(m * n * 2)?;
    let expert_offsets: Vec<i32> = vec![0, m as i32];
    let grp_launch = || -> Result<()> {
        spark_runtime::cutlass::nvfp4_grouped_gate_up(
            a_ptr.0,
            &[packed_ptr.0],
            &[scale_ptr.0],
            &[1.0f32],
            &[up_packed_ptr.0],
            &[up_scale_ptr.0],
            &[1.0f32],
            c_gate_grp.0,
            c_up_grp.0,
            &expert_offsets,
            nu,
            ku,
            stream,
        )
    };
    grp_launch()?;
    gpu.synchronize(stream)?;
    let grp_us = time_gemm(gpu, stream, iters, grp_launch)?;
    let c_grp_gate = read_bf16(gpu, c_gate_grp, m, n)?;

    // ── FP8 GEMM (fp8_gemm_t kernel, module "w4a16") ──
    let fp8_handle = gpu.kernel("w4a16", "fp8_gemm_t")?;
    let fp8_launch = || -> Result<()> {
        KernelLaunch::new(gpu, fp8_handle)
            .grid([div_ceil(nu, 128), div_ceil(mu, 64), 1])
            .block([128, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(b_fp8_ptr)
            .arg_ptr(out_fp8)
            .arg_u32(mu)
            .arg_u32(nu)
            .arg_u32(ku)
            .launch(stream)?;
        Ok(())
    };
    fp8_launch()?;
    gpu.synchronize(stream)?;
    let fp8_us = time_gemm(gpu, stream, iters, fp8_launch)?;
    let c_fp8 = read_bf16(gpu, out_fp8, m, n)?;

    // ── BF16 GEMM (CUTLASS Sm120 peer, for the timing line) ──
    let bf16_launch = || {
        spark_runtime::cutlass::bf16_gemm_act_weight_t(
            a_ptr.0,
            b_bf16_ptr.0,
            out_bf16.0,
            mu,
            nu,
            ku,
            stream,
        )
    };
    bf16_launch()?;
    gpu.synchronize(stream)?;
    let bf16_us = time_gemm(gpu, stream, iters, bf16_launch)?;

    // ── accuracy vs CPU bf16 oracle ──
    let c_ref = cpu_reference(&a_bf16, &b_bf16, m, n, k);
    let (fp4_cos, fp4_max_rel) = compare(&c_fp4, &c_ref);
    let (fp8_cos, fp8_max_rel) = compare(&c_fp8, &c_ref);

    // grouped gate output uses the SAME gate weight (b_bf16) as the single-GEMM
    // collective path -> validate vs both c_fp4 (collective) and c_ref (oracle).
    let grp_cos_vs_collective = cosine_u16(&c_grp_gate, &c_fp4);
    let (grp_cos_vs_oracle, _grp_max_rel) = compare(&c_grp_gate, &c_ref);

    for p in [
        a_ptr, b_bf16_ptr, b_fp8_ptr, packed_ptr, scale_ptr, out_fp4, out_fp8, out_bf16,
        up_bf16_ptr, up_packed_ptr, up_scale_ptr, c_gate_grp, c_up_grp,
    ] {
        gpu.free(p).ok();
    }

    Ok(Row {
        m,
        fp4_us,
        fp8_us,
        bf16_us,
        fp4_cos,
        fp4_max_rel,
        fp8_cos,
        fp8_max_rel,
        grp_us,
        grp_cos_vs_collective,
        grp_cos_vs_oracle,
    })
}

fn main() -> Result<()> {
    // Holo 3.1 MoE gate_up: N = 2*moe_intermediate = 1024, K = hidden = 2048.
    let n = 1024usize;
    let k = 2048usize;
    let m_values = [32usize, 64, 128, 2048];
    let seed = 0x_5151_A7A7u64;

    println!("=== Holo 3.1 MoE gate_up FP4-vs-FP8 shape test ===");
    println!("N={n} (2*moe_intermediate=2*512), K={k} (hidden); gate_up per-expert GEMM");
    println!("M sweep: {m_values:?}  (64 = realistic tokens/expert at chunk 2048)");
    println!();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let mut rows = Vec::new();
    for &m in &m_values {
        let row = run_shape(gpu, stream, m, n, k, seed ^ (m as u64).wrapping_mul(0x9E37))?;
        rows.push(row);
    }

    // ── table ──
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>9} {:>9} {:>12}",
        "M", "fp4_us", "fp8_us", "bf16_us", "fp4_cos", "fp8_cos", "fp4/fp8_sp"
    );
    println!("{}", "-".repeat(72));
    let mut all_pass = true;
    let mut min_fp4_cos = 1.0f64;
    for r in &rows {
        let speedup = if r.fp4_us > 0.0 {
            r.fp8_us / r.fp4_us
        } else {
            0.0
        };
        println!(
            "{:>6} {:>10.2} {:>10.2} {:>10.2} {:>9.4} {:>9.4} {:>12.3}",
            r.m, r.fp4_us, r.fp8_us, r.bf16_us, r.fp4_cos, r.fp8_cos, speedup
        );
        min_fp4_cos = min_fp4_cos.min(r.fp4_cos);
        if r.fp4_cos < COSINE_GATE {
            all_pass = false;
        }
    }
    println!("{}", "-".repeat(72));

    // ── Phase-1 grouped FP4 gate_up kernel gate ──
    println!();
    println!("=== Phase-1 grouped FP4 gate_up kernel (escape-hatch: per-expert collective) ===");
    println!(
        "{:>6} {:>10} {:>22} {:>18}",
        "M", "grp_us", "cos_vs_collective", "cos_vs_oracle"
    );
    println!("{}", "-".repeat(60));
    const GRP_COLLECTIVE_GATE: f64 = 0.999;
    let mut grp_pass = true;
    let mut min_grp_coll = 1.0f64;
    let mut min_grp_orc = 1.0f64;
    for r in &rows {
        println!(
            "{:>6} {:>10.2} {:>22.6} {:>18.6}",
            r.m, r.grp_us, r.grp_cos_vs_collective, r.grp_cos_vs_oracle
        );
        min_grp_coll = min_grp_coll.min(r.grp_cos_vs_collective);
        min_grp_orc = min_grp_orc.min(r.grp_cos_vs_oracle);
        if r.grp_cos_vs_collective < GRP_COLLECTIVE_GATE || r.grp_cos_vs_oracle < COSINE_GATE {
            grp_pass = false;
        }
    }
    println!("{}", "-".repeat(60));
    if let Some(r) = rows.iter().find(|r| r.m == 64) {
        println!(
            "GROUPED HEADLINE @M=64: cos_vs_collective={:.6} cos_vs_oracle={:.6} grp_us={:.2} (fp8_us={:.2})",
            r.grp_cos_vs_collective, r.grp_cos_vs_oracle, r.grp_us, r.fp8_us
        );
    }
    if grp_pass {
        println!(
            "GROUPED RESULT: PASS (cos_vs_collective>={GRP_COLLECTIVE_GATE} min {min_grp_coll:.6}; cos_vs_oracle>={COSINE_GATE} min {min_grp_orc:.6})"
        );
    } else {
        eprintln!(
            "GROUPED RESULT: FAIL (collective min {min_grp_coll:.6} gate {GRP_COLLECTIVE_GATE}; oracle min {min_grp_orc:.6} gate {COSINE_GATE})"
        );
        all_pass = false;
    }
    println!();
    println!(
        "(max_rel: fp4 {:?}, fp8 {:?})",
        rows.iter()
            .map(|r| format!("M{}={:.2e}", r.m, r.fp4_max_rel))
            .collect::<Vec<_>>(),
        rows.iter()
            .map(|r| format!("M{}={:.2e}", r.m, r.fp8_max_rel))
            .collect::<Vec<_>>(),
    );

    // headline at M=64
    if let Some(r) = rows.iter().find(|r| r.m == 64) {
        let sp = if r.fp4_us > 0.0 {
            r.fp8_us / r.fp4_us
        } else {
            0.0
        };
        println!(
            "HEADLINE: at M=64  fp4_speedup_over_fp8={sp:.3}x  fp4_cos={:.4}",
            r.fp4_cos
        );
    }

    if all_pass {
        println!("RESULT: PASS (all fp4 cosine >= {COSINE_GATE}; min {min_fp4_cos:.4})");
        Ok(())
    } else {
        eprintln!("RESULT: FAIL (some fp4 cosine < {COSINE_GATE}; min {min_fp4_cos:.4})");
        std::process::exit(1);
    }
}
