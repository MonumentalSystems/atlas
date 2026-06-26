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

// ───────────────────────── FP4 e2m1 / ue4m3 host packers ─────────────────────
// Mirror the CUDA pack (float_to_e2m1 round-to-nearest, ue4m3 scale=max_abs/6).
fn f32_to_e2m1(x: f32) -> u8 {
    let sign: u8 = if x < 0.0 { 8 } else { 0 };
    let ax = x.abs();
    let mag: u8 = if ax <= 0.25 {
        0
    } else if ax <= 0.75 {
        1
    } else if ax <= 1.25 {
        2
    } else if ax <= 1.75 {
        3
    } else if ax <= 2.5 {
        4
    } else if ax <= 3.5 {
        5
    } else if ax <= 5.0 {
        6
    } else {
        7
    };
    sign | mag
}
// ue4m3 (unsigned e4m3 magnitude byte) of a non-negative scale: same bit pattern
// as the standard e4m3 of |scale| with the sign bit clear (scale>=0).
fn f32_to_ue4m3(scale: f32) -> u8 {
    f32_to_e4m3(scale) & 0x7F
}
fn ue4m3_to_f32(byte: u8) -> f32 {
    e4m3_to_f32(byte & 0x7F)
}

/// Pack a bf16 weight `[N,K]` into the FP8-fused-kernel layout:
/// packed `[K/2, N]` (K-major) nibbles + scales `[K/16, N]` ue4m3, per the
/// production `moe_w4a16_fused_gate_up_t_k64` kernel's B_expert indexing
/// (`B[(k/2)*N + n]`, `S[(k/16)*N + n]`). Returns (packed, scales).
fn pack_weight_kmajor(b_bf16: &[u16], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let half_k = k / 2;
    let groups = k / 16;
    let mut packed = vec![0u8; half_k * n];
    let mut scales = vec![0u8; groups * n];
    for col in 0..n {
        for g in 0..groups {
            let base = g * 16;
            let mut max_abs = 0.0f32;
            for i in 0..16 {
                let v = bf16_bits_to_f32(b_bf16[col * k + base + i]);
                max_abs = max_abs.max(v.abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 };
            let sf = f32_to_ue4m3(scale);
            scales[g * n + col] = sf;
            let inv = {
                let d = ue4m3_to_f32(sf);
                if d > 0.0 { 1.0 / d } else { 0.0 }
            };
            for i in (0..16).step_by(2) {
                let v0 = bf16_bits_to_f32(b_bf16[col * k + base + i]) * inv;
                let v1 = bf16_bits_to_f32(b_bf16[col * k + base + i + 1]) * inv;
                let kk = base + i;
                packed[(kk / 2) * n + col] = f32_to_e2m1(v0) | (f32_to_e2m1(v1) << 4);
            }
        }
    }
    (packed, scales)
}

/// Byte-transpose GPU-packed NVFP4 nibbles `[N,K/2]` (the
/// `pack_bf16_weight_to_nvfp4_t` layout the CUTLASS collective consumes) into the
/// Atlas shared-table `[K/2,N]` layout (exactly `transpose_for_gemm`:
/// `t[j*n+i] = src[i*half_k+j]`). Pure index remap — preserves each byte's
/// `(2j,2j+1)` e2m1 nibble pair, so NO re-quantization. The ue4m3 group scales are
/// already `[K/16,N]` in BOTH layouts, so the caller reuses the scale pointer
/// unchanged. Feeding these byte-identical nibbles to the rewritten FP4 kernel
/// makes `cos_vs_collective` a pure transpose-correctness gate (no packer drift).
fn packed_nmajor_to_kmajor(
    gpu: &dyn GpuBackend,
    packed_nmajor: DevicePtr,
    n: usize,
    k: usize,
) -> Result<DevicePtr> {
    let half_k = k / 2;
    let mut src = vec![0u8; half_k * n];
    gpu.copy_d2h(packed_nmajor, &mut src)?;
    let mut dst = vec![0u8; half_k * n];
    for i in 0..n {
        for j in 0..half_k {
            dst[j * n + i] = src[i * half_k + j];
        }
    }
    upload_bytes(gpu, &dst)
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
    // Phase-2 FUSED FP4 kernel (cp.async pipelined, single launch, no gather).
    fused_us: f64,
    fused_cos_vs_collective: f64, // MUST be >= 0.999
    fused_cos_vs_oracle: f64,     // MUST be >= 0.98
    // Production FP8 fused kernel (moe_w4a16_fused_gate_up_t_k64) — the REAL A/B.
    fused_fp8_us: f64,
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

/// L2 norm ratio ‖a‖/‖b‖. Unlike cosine (scale-invariant), this is SENSITIVE to a
/// wrong per-group scale (e.g. a scalar-alpha bug that applies the wrong expert's
/// scale2) — the ratio deviates from 1.
fn norm_ratio_u16(a: &[u16], b: &[u16]) -> f64 {
    let (mut na, mut nb) = (0f64, 0f64);
    for i in 0..a.len() {
        let x = bf16_bits_to_f32(a[i]) as f64;
        let y = bf16_bits_to_f32(b[i]) as f64;
        na += x * x;
        nb += y * y;
    }
    if nb > 0.0 { na.sqrt() / nb.sqrt() } else { 0.0 }
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

    // ── Phase-2 FUSED FP4 kernel (moe_w4a16_fused_gate_up_t_k64_fp4) ──
    // Single launch, cp.async pipelined, in-kernel A-quant, no gather. Consumes
    // the SHARED [K/2,N] tables (transpose_for_gemm layout) the rewritten kernel
    // now reads — here the byte-transpose of the collective's [N,K/2] nibbles +
    // the [K/16,N] scales. Single expert covering all M rows; sorted_token_ids =
    // null (identity).
    let fused_handle = gpu.kernel("moe_w4a16", "moe_w4a16_fused_gate_up_t_k64_fp4")?;
    // Device ptr-tables: one expert each (u64 device pointers).
    let mk_ptr_tbl = |p: u64| -> Result<DevicePtr> {
        let bytes = p.to_le_bytes();
        let d = gpu.alloc(8)?;
        gpu.copy_h2d(&bytes, d)?;
        Ok(d)
    };
    // FP4 fused now reads the SHARED [K/2,N] layout: byte-transpose the
    // collective's [N,K/2] nibbles (bit-identical, no re-quant) so cos_vs_collective
    // is a pure transpose-correctness gate. Scales are [K/16,N] in both → reuse.
    let gate_packed_kt = packed_nmajor_to_kmajor(gpu, packed_ptr, n, k)?;
    let up_packed_kt = packed_nmajor_to_kmajor(gpu, up_packed_ptr, n, k)?;
    let gate_packed_tbl = mk_ptr_tbl(gate_packed_kt.0)?;
    let gate_scale_tbl = mk_ptr_tbl(scale_ptr.0)?;
    let up_packed_tbl = mk_ptr_tbl(up_packed_kt.0)?;
    let up_scale_tbl = mk_ptr_tbl(up_scale_ptr.0)?;
    let gate_scale2_tbl = upload_bytes(gpu, &1.0f32.to_le_bytes())?;
    let up_scale2_tbl = upload_bytes(gpu, &1.0f32.to_le_bytes())?;
    let eoff_dev = {
        let mut b = Vec::new();
        b.extend_from_slice(&0i32.to_le_bytes());
        b.extend_from_slice(&(m as i32).to_le_bytes());
        upload_bytes(gpu, &b)?
    };
    let c_gate_fused = gpu.alloc(m * n * 2)?;
    let c_up_fused = gpu.alloc(m * n * 2)?;
    let max_m_tiles = div_ceil(mu, 64).max(1);
    let fused_launch = || -> Result<()> {
        KernelLaunch::new(gpu, fused_handle)
            .grid([div_ceil(2 * nu, 128), max_m_tiles, 1])
            .block([128, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(gate_packed_tbl)
            .arg_ptr(gate_scale_tbl)
            .arg_ptr(gate_scale2_tbl)
            .arg_ptr(up_packed_tbl)
            .arg_ptr(up_scale_tbl)
            .arg_ptr(up_scale2_tbl)
            .arg_ptr(c_gate_fused)
            .arg_ptr(c_up_fused)
            .arg_ptr(eoff_dev)
            .arg_u64(0) // sorted_token_ids = null
            .arg_u32(1) // num_experts
            .arg_u32(nu)
            .arg_u32(ku)
            .launch(stream)?;
        Ok(())
    };
    fused_launch()?;
    gpu.synchronize(stream)?;
    let fused_us = time_gemm(gpu, stream, iters, fused_launch)?;
    let c_fused_gate = read_bf16(gpu, c_gate_fused, m, n)?;

    // ── Production FP8 fused kernel (moe_w4a16_fused_gate_up_t_k64) ──
    // The REAL A/B for the speed signal. Needs K-major [K/2,N] packed +
    // [K/16,N] ue4m3 scales (its own layout, distinct from the FP4 N-major).
    let (gate_pk, gate_sk) = pack_weight_kmajor(&b_bf16, n, k);
    let (up_pk, up_sk) = pack_weight_kmajor(&up_bf16, n, k);
    let gate_pk_ptr = upload_bytes(gpu, &gate_pk)?;
    let gate_sk_ptr = upload_bytes(gpu, &gate_sk)?;
    let up_pk_ptr = upload_bytes(gpu, &up_pk)?;
    let up_sk_ptr = upload_bytes(gpu, &up_sk)?;
    let gate_pk_tbl = mk_ptr_tbl(gate_pk_ptr.0)?;
    let gate_sk_tbl = mk_ptr_tbl(gate_sk_ptr.0)?;
    let up_pk_tbl = mk_ptr_tbl(up_pk_ptr.0)?;
    let up_sk_tbl = mk_ptr_tbl(up_sk_ptr.0)?;
    let fp8_fused_handle = gpu.kernel("moe_w4a16", "moe_w4a16_fused_gate_up_t_k64")?;
    let c_gate_fp8f = gpu.alloc(m * n * 2)?;
    let c_up_fp8f = gpu.alloc(m * n * 2)?;
    let fp8_fused_launch = || -> Result<()> {
        KernelLaunch::new(gpu, fp8_fused_handle)
            .grid([div_ceil(2 * nu, 128), max_m_tiles, 1])
            .block([128, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(gate_pk_tbl)
            .arg_ptr(gate_sk_tbl)
            .arg_ptr(gate_scale2_tbl)
            .arg_ptr(up_pk_tbl)
            .arg_ptr(up_sk_tbl)
            .arg_ptr(up_scale2_tbl)
            .arg_ptr(c_gate_fp8f)
            .arg_ptr(c_up_fp8f)
            .arg_ptr(eoff_dev)
            .arg_u64(0)
            .arg_u32(1)
            .arg_u32(nu)
            .arg_u32(ku)
            .launch(stream)?;
        Ok(())
    };
    fp8_fused_launch()?;
    gpu.synchronize(stream)?;
    let fused_fp8_us = time_gemm(gpu, stream, iters, fp8_fused_launch)?;

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

    // fused FP4 gate output uses the SAME gate weight (b_bf16) -> validate vs
    // the collective (c_fp4) and the bf16 oracle (c_ref).
    let fused_cos_vs_collective = cosine_u16(&c_fused_gate, &c_fp4);
    let (fused_cos_vs_oracle, _ff_max_rel) = compare(&c_fused_gate, &c_ref);

    for p in [
        a_ptr, b_bf16_ptr, b_fp8_ptr, packed_ptr, scale_ptr, out_fp4, out_fp8, out_bf16,
        up_bf16_ptr, up_packed_ptr, up_scale_ptr, c_gate_grp, c_up_grp,
        gate_packed_kt, up_packed_kt,
        gate_packed_tbl, gate_scale_tbl, up_packed_tbl, up_scale_tbl,
        gate_scale2_tbl, up_scale2_tbl, eoff_dev, c_gate_fused, c_up_fused,
        gate_pk_ptr, gate_sk_ptr, up_pk_ptr, up_sk_ptr,
        gate_pk_tbl, gate_sk_tbl, up_pk_tbl, up_sk_tbl, c_gate_fp8f, c_up_fp8f,
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
        fused_us,
        fused_cos_vs_collective,
        fused_cos_vs_oracle,
        fused_fp8_us,
    })
}

/// FP4 DOWN GEMM proof (`moe_w4a16_down_t_k64_fp4`). Single-output grouped
/// kernel at the down shape: A = post-SiLU intermediate `[M, K=inter]`, weight
/// = down_proj `[N=hidden, K=inter]`, output `[M, N]`. Validates the FP4 down
/// kernel vs (a) the CUTLASS collective `nvfp4_gemm_bf16_act_weight_t` on the
/// SAME packed weights (cos >= 0.999, identical FP4 math) and (b) the CPU bf16
/// oracle (cos >= 0.98). Also times it vs the production FP8 down kernel
/// (`moe_w4a16_grouped_gemm_ptrtable_t_k64`) on the REAL bf16 A/B.
fn run_down_shape(
    gpu: &dyn GpuBackend,
    stream: u64,
    m: usize,
    n: usize, // hidden = 2048
    k: usize, // inter = 512
    seed: u64,
) -> Result<(f64, f64, f64, f64, f64)> {
    // (down_us, fp8_down_us, w4a16_down_us, cos_vs_collective, cos_vs_oracle)
    let iters = 50usize;
    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let b_bf16: Vec<u16> = (0..n * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.5, 0.5)))
        .collect();

    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
    let b_bf16_ptr = upload_bytes(gpu, &u16s_to_le(&b_bf16))?;
    let (mu, nu, ku) = (m as u32, n as u32, k as u32);

    // Collective FP4 weight (pack_bf16_weight_to_nvfp4_t: packed [N,K/2], scale
    // [K/16,N]); the FP4 down kernel reads the [K/2,N] byte-transpose of this.
    let packed_len = (k / 2) * n;
    let scale_len = (k / 16) * n;
    let packed_ptr = gpu.alloc(packed_len)?;
    let scale_ptr = gpu.alloc(scale_len)?;
    spark_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
        b_bf16_ptr.0, packed_ptr.0, scale_ptr.0, nu, ku, stream,
    )?;
    gpu.synchronize(stream)?;

    // ── CUTLASS collective FP4 (reference for the same FP4 math) ──
    let out_collective = gpu.alloc(m * n * 2)?;
    let coll_launch = || {
        spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
            a_ptr.0, packed_ptr.0, scale_ptr.0, 1.0, out_collective.0, mu, nu, ku, stream,
        )
    };
    coll_launch()?;
    gpu.synchronize(stream)?;
    let c_collective = read_bf16(gpu, out_collective, m, n)?;

    // ── FP4 down kernel (the one under test) ──
    let down_handle = gpu.kernel("moe_w4a16", "moe_w4a16_down_t_k64_fp4")?;
    let mk_ptr_tbl = |p: u64| -> Result<DevicePtr> {
        let bytes = p.to_le_bytes();
        let d = gpu.alloc(8)?;
        gpu.copy_h2d(&bytes, d)?;
        Ok(d)
    };
    // The rewritten FP4 down kernel reads the SHARED [K/2,N] layout: byte-transpose
    // the collective's [N,K/2] nibbles (bit-identical) so cos_vs_collective is a
    // pure transpose gate. Scales [K/16,N] reused unchanged.
    let packed_kt = packed_nmajor_to_kmajor(gpu, packed_ptr, n, k)?;
    let packed_tbl = mk_ptr_tbl(packed_kt.0)?;
    let scale_tbl = mk_ptr_tbl(scale_ptr.0)?;
    let scale2_tbl = upload_bytes(gpu, &1.0f32.to_le_bytes())?;
    let eoff_dev = {
        let mut b = Vec::new();
        b.extend_from_slice(&0i32.to_le_bytes());
        b.extend_from_slice(&(m as i32).to_le_bytes());
        upload_bytes(gpu, &b)?
    };
    let out_down = gpu.alloc(m * n * 2)?;
    let max_m_tiles = div_ceil(mu, 64).max(1);
    let down_launch = || -> Result<()> {
        KernelLaunch::new(gpu, down_handle)
            .grid([div_ceil(nu, 128), max_m_tiles, 1])
            .block([128, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(packed_tbl)
            .arg_ptr(scale_tbl)
            .arg_ptr(scale2_tbl)
            .arg_ptr(out_down)
            .arg_ptr(eoff_dev)
            .arg_u64(0) // sorted_token_ids = null (identity, as in production down)
            .arg_u32(1) // num_experts
            .arg_u32(nu)
            .arg_u32(ku)
            .launch(stream)?;
        Ok(())
    };
    down_launch()?;
    gpu.synchronize(stream)?;
    let down_us = time_gemm(gpu, stream, iters, down_launch)?;
    let c_down = read_bf16(gpu, out_down, m, n)?;

    // ── production FP8 down kernel (moe_w4a16_grouped_gemm_ptrtable_t_k64) ──
    // K-major [K/2,N] packed + [K/16,N] ue4m3 scales (its own layout).
    let (b_pk, b_sk) = pack_weight_kmajor(&b_bf16, n, k);
    let b_pk_ptr = upload_bytes(gpu, &b_pk)?;
    let b_sk_ptr = upload_bytes(gpu, &b_sk)?;
    let b_pk_tbl = mk_ptr_tbl(b_pk_ptr.0)?;
    let b_sk_tbl = mk_ptr_tbl(b_sk_ptr.0)?;
    let out_w4a16 = gpu.alloc(m * n * 2)?;
    let w4a16_handle = gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_t_k64")?;
    let w4a16_launch = || -> Result<()> {
        KernelLaunch::new(gpu, w4a16_handle)
            .grid([div_ceil(nu, 128), max_m_tiles, 1])
            .block([128, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(b_pk_tbl)
            .arg_ptr(b_sk_tbl)
            .arg_ptr(scale2_tbl)
            .arg_ptr(out_w4a16)
            .arg_ptr(eoff_dev)
            .arg_u64(0)
            .arg_u32(1)
            .arg_u32(nu)
            .arg_u32(ku)
            .launch(stream)?;
        Ok(())
    };
    w4a16_launch()?;
    gpu.synchronize(stream)?;
    let w4a16_down_us = time_gemm(gpu, stream, iters, w4a16_launch)?;

    // ── production FP8 down (fp8 A): bf16_to_fp8 then moe_fp8_grouped_gemm_ptrtable_t ──
    // The currently-shipped down config (ATLAS_MOE_PREFILL_FP8_DOWN=1). Same
    // K-major weights; A pre-converted to e4m3. Timed for the A/B baseline.
    let a_fp8: Vec<u8> = a_bf16
        .iter()
        .map(|&b| f32_to_e4m3(bf16_bits_to_f32(b)))
        .collect();
    let a_fp8_ptr = upload_bytes(gpu, &a_fp8)?;
    let out_fp8 = gpu.alloc(m * n * 2)?;
    let fp8_handle = gpu.kernel("moe_w4a16", "moe_fp8_grouped_gemm_ptrtable_t")?;
    let fp8_launch = || -> Result<()> {
        KernelLaunch::new(gpu, fp8_handle)
            .grid([div_ceil(nu, 128), max_m_tiles, 1])
            .block([128, 1, 1])
            .arg_ptr(a_fp8_ptr)
            .arg_ptr(b_pk_tbl)
            .arg_ptr(b_sk_tbl)
            .arg_ptr(scale2_tbl)
            .arg_ptr(out_fp8)
            .arg_ptr(eoff_dev)
            .arg_u64(0)
            .arg_u32(1)
            .arg_u32(nu)
            .arg_u32(ku)
            .launch(stream)?;
        Ok(())
    };
    let fp8_down_us = if fp8_launch().is_ok() {
        gpu.synchronize(stream)?;
        time_gemm(gpu, stream, iters, fp8_launch).unwrap_or(0.0)
    } else {
        0.0
    };

    let c_ref = cpu_reference(&a_bf16, &b_bf16, m, n, k);
    let cos_vs_collective = cosine_u16(&c_down, &c_collective);
    let (cos_vs_oracle, _) = compare(&c_down, &c_ref);

    for p in [
        a_ptr, b_bf16_ptr, packed_ptr, scale_ptr, out_collective, packed_kt, packed_tbl,
        scale_tbl, scale2_tbl, eoff_dev, out_down, b_pk_ptr, b_sk_ptr, b_pk_tbl, b_sk_tbl,
        out_w4a16,
        a_fp8_ptr, out_fp8,
    ] {
        gpu.free(p).ok();
    }

    Ok((down_us, fp8_down_us, w4a16_down_us, cos_vs_collective, cos_vs_oracle))
}

/// Single-launch grouped NVFP4 gate_up proof (`atlas_cutlass_nvfp4_grouped_gate_up_fused`).
/// Builds a >=2-expert grouped problem with NON-TRIVIAL `expert_offsets` (a
/// permutation of M into uneven per-expert blocks) and distinct per-expert gate/up
/// weights, then launches the SINGLE `GemmUniversalMode::kGrouped` path and asserts:
///   (a) cos >= 0.999 vs the per-expert dense collective `_t` reference (identical FP4
///       math + SF/layout plumbing — any divergence is the grouped single-launch
///       assembly), block-for-block, gate AND up; and
///   (b) cos >= 0.98 vs the CPU bf16 oracle (proves the FP4 numerics).
/// A is pre-sorted expert-contiguous (what `moe_permute_tokens` produces in prod),
/// so the only piece not exercised here is the permute itself.
/// Returns (min_cos_vs_collective, min_cos_vs_oracle) over {gate,up}×{experts}.
fn run_grouped_fused_shape(
    gpu: &dyn GpuBackend,
    stream: u64,
    m: usize,
    n: usize,
    k: usize,
    num_experts: usize,
    seed: u64,
) -> Result<(f64, f64, f64)> {
    assert!(num_experts >= 2, "grouped-fused proof needs >=2 experts");
    let (nu, ku) = (n as u32, k as u32);
    let packed_len = (k / 2) * n; // [N,K/2] bytes, native CUTLASS pack layout
    let scale_len = (k / 16) * n; // [K/16,N] E4M3 bytes (pack output)
    // Swizzled SFB upper-bound bytes: SF atom rounds N->mult of 128, K/16->mult of 4.
    let round_up = |x: usize, a: usize| -> usize { x.div_ceil(a) * a };
    let sfb_len = round_up(n, 128) * round_up(k / 16, 4);

    // ── NON-TRIVIAL expert_offsets: split M into uneven blocks (a permutation of
    //    token rows into experts). For 2 experts: [0, m*3/8, m] (uneven). For >2,
    //    a roughly-even split with a deliberate first-block skew. ──
    let mut expert_offsets: Vec<i32> = Vec::with_capacity(num_experts + 1);
    expert_offsets.push(0);
    if num_experts == 2 {
        expert_offsets.push((m * 3 / 8) as i32); // uneven: e0 gets 3/8, e1 gets 5/8
        expert_offsets.push(m as i32);
    } else {
        // skew the first block smaller, distribute the rest evenly
        let first = (m / (num_experts * 2)).max(1);
        let rest = m - first;
        let per = rest / (num_experts - 1);
        let mut acc = first;
        expert_offsets.push(acc as i32);
        for e in 1..num_experts {
            acc += if e == num_experts - 1 { m - acc } else { per };
            expert_offsets.push(acc as i32);
        }
    }
    // sanity: monotone, covers [0,m]
    assert_eq!(expert_offsets[0], 0);
    assert_eq!(*expert_offsets.last().unwrap(), m as i32);

    // ── A: [M,K] bf16, expert-contiguous (rows of expert e occupy its offset span) ──
    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;

    // ── per-expert distinct gate + up weights, packed to native CUTLASS layout +
    //    swizzled SFB ──
    let mut gate_bf16: Vec<Vec<u16>> = Vec::with_capacity(num_experts);
    let mut up_bf16: Vec<Vec<u16>> = Vec::with_capacity(num_experts);
    let mut gate_packed_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut gate_sfb_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut up_packed_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut up_sfb_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    // collective reference also needs the [K/16,N] scale (NOT the swizzled SFB).
    let mut gate_scale_ptrs: Vec<DevicePtr> = Vec::with_capacity(num_experts);
    let mut up_scale_ptrs: Vec<DevicePtr> = Vec::with_capacity(num_experts);
    // DISTINCT per-expert scale2 (NOT all 1.0): exercises the grouped epilogue's
    // per-group alpha (alpha_ptr_array). A scalar-alpha bug applies expert-0's
    // scale2 to all experts — invisible to cosine (scale-invariant) but caught by
    // the norm-ratio check below.
    let d = (num_experts.max(2) - 1) as f32;
    let gate_scale2: Vec<f32> = (0..num_experts)
        .map(|e| 0.35 + 0.60 * e as f32 / d)
        .collect();
    let up_scale2: Vec<f32> = (0..num_experts)
        .map(|e| 0.90 - 0.50 * e as f32 / d)
        .collect();
    let mut to_free: Vec<DevicePtr> = vec![a_ptr];

    for _e in 0..num_experts {
        let g: Vec<u16> = (0..n * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-0.5, 0.5)))
            .collect();
        let u: Vec<u16> = (0..n * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-0.5, 0.5)))
            .collect();
        let g_bf16_ptr = upload_bytes(gpu, &u16s_to_le(&g))?;
        let u_bf16_ptr = upload_bytes(gpu, &u16s_to_le(&u))?;
        // native CUTLASS pack: packed [N,K/2], scale [K/16,N]
        let g_packed = gpu.alloc(packed_len)?;
        let g_scale = gpu.alloc(scale_len)?;
        let u_packed = gpu.alloc(packed_len)?;
        let u_scale = gpu.alloc(scale_len)?;
        spark_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
            g_bf16_ptr.0, g_packed.0, g_scale.0, nu, ku, stream,
        )?;
        spark_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
            u_bf16_ptr.0, u_packed.0, u_scale.0, nu, ku, stream,
        )?;
        // load-time SFB swizzle repack ([K/16,N] E4M3 -> CUTLASS SFB atom)
        let g_sfb = gpu.alloc(sfb_len)?;
        let u_sfb = gpu.alloc(sfb_len)?;
        spark_runtime::cutlass::pack_weight_sfb(g_scale.0, g_sfb.0, nu, ku, stream)?;
        spark_runtime::cutlass::pack_weight_sfb(u_scale.0, u_sfb.0, nu, ku, stream)?;
        gpu.synchronize(stream)?;

        gate_packed_ptrs.push(g_packed.0);
        gate_sfb_ptrs.push(g_sfb.0);
        up_packed_ptrs.push(u_packed.0);
        up_sfb_ptrs.push(u_sfb.0);
        gate_scale_ptrs.push(g_scale);
        up_scale_ptrs.push(u_scale);
        gate_bf16.push(g);
        up_bf16.push(u);
        to_free.extend_from_slice(&[
            g_bf16_ptr, u_bf16_ptr, g_packed, u_packed, g_sfb, u_sfb,
        ]);
    }

    // ── single-launch grouped gate_up ──
    let c_gate = gpu.alloc(m * n * 2)?;
    let c_up = gpu.alloc(m * n * 2)?;
    spark_runtime::cutlass::nvfp4_grouped_gate_up_fused(
        a_ptr.0,
        0, // sorted_token_ids = null (A is already expert-contiguous in this test)
        &gate_packed_ptrs,
        &gate_sfb_ptrs,
        &gate_scale2,
        &up_packed_ptrs,
        &up_sfb_ptrs,
        &up_scale2,
        c_gate.0,
        c_up.0,
        &expert_offsets,
        nu,
        ku,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let c_gate_grp = read_bf16(gpu, c_gate, m, n)?;
    let c_up_grp = read_bf16(gpu, c_up, m, n)?;
    to_free.extend_from_slice(&[c_gate, c_up]);

    // ── references: per-expert dense collective + CPU bf16 oracle, block-for-block ──
    let mut min_coll = 1.0f64;
    let mut min_orc = 1.0f64;
    let mut max_nr_dev = 0.0f64; // max |‖grouped‖/‖collective‖ - 1| (scale2 check)
    let coll_out = gpu.alloc(m * n * 2)?; // scratch for collective per-block runs
    to_free.push(coll_out);
    for e in 0..num_experts {
        let r0 = expert_offsets[e] as usize;
        let r1 = expert_offsets[e + 1] as usize;
        let m_e = r1 - r0;
        if m_e == 0 {
            continue;
        }
        let a_e: Vec<u16> = a_bf16[r0 * k..r1 * k].to_vec();
        let a_e_ptr = upload_bytes(gpu, &u16s_to_le(&a_e))?;

        for (proj, packed, scale, s2, weight, c_grp) in [
            (
                "gate",
                gate_packed_ptrs[e],
                gate_scale_ptrs[e].0,
                gate_scale2[e],
                &gate_bf16[e],
                &c_gate_grp,
            ),
            (
                "up",
                up_packed_ptrs[e],
                up_scale_ptrs[e].0,
                up_scale2[e],
                &up_bf16[e],
                &c_up_grp,
            ),
        ] {
            // (a) dense collective on this expert's A-rows + packed weight, with the
            //     SAME per-expert scale2 the grouped path applies as alpha.
            spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
                a_e_ptr.0,
                packed,
                scale,
                s2,
                coll_out.0,
                m_e as u32,
                nu,
                ku,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let c_coll = read_bf16(gpu, coll_out, m_e, n)?;
            // corresponding grouped output block for this expert
            let grp_block = &c_grp[r0 * n..r1 * n];
            let cos_coll = cosine_u16(grp_block, &c_coll);
            // SCALE-SENSITIVE: cosine is scale-invariant and can't catch a wrong
            // per-group scale2; the norm ratio must be ~1 (both have scale2 applied).
            max_nr_dev = max_nr_dev.max((norm_ratio_u16(grp_block, &c_coll) - 1.0).abs());
            // (b) CPU bf16 oracle on this expert's block (cosine — direction only).
            let c_ref = cpu_reference(&a_e, weight, m_e, n, k);
            let (cos_orc, _) = compare(grp_block, &c_ref);
            min_coll = min_coll.min(cos_coll);
            min_orc = min_orc.min(cos_orc);
            let _ = proj;
        }
        gpu.free(a_e_ptr).ok();
    }

    for p in to_free.into_iter().chain(gate_scale_ptrs).chain(up_scale_ptrs) {
        gpu.free(p).ok();
    }
    Ok((min_coll, min_orc, max_nr_dev))
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

    // ── Single-launch GROUPED-FUSED FP4 gate_up (kGrouped, >=2 experts) ──
    // Validates atlas_cutlass_nvfp4_grouped_gate_up_fused: one GemmUniversalMode::
    // kGrouped launch over multiple experts with non-trivial expert_offsets, vs the
    // per-expert dense collective (cos>=0.999) and the bf16 oracle (cos>=0.98).
    println!();
    println!("=== Single-launch GROUPED-FUSED FP4 gate_up (kGrouped, >=2 experts) ===");
    println!(
        "{:>6} {:>5} {:>22} {:>18} {:>12}",
        "M", "ne", "cos_vs_collective", "cos_vs_oracle", "scale2_nrdev"
    );
    println!("{}", "-".repeat(70));
    const GFUSED_COLLECTIVE_GATE: f64 = 0.999;
    // Per-group scale2 (alpha_ptr_array) correctness: ‖grouped‖/‖collective‖ must be
    // ~1 per expert. A scalar-alpha bug (all experts use expert-0's scale2) blows
    // this up while leaving cosine at 1.0. FP4 noise keeps it well under 0.05.
    const GFUSED_NR_GATE: f64 = 0.05;
    let mut gfused_pass = true;
    let mut min_gf_coll = 1.0f64;
    let mut min_gf_orc = 1.0f64;
    let mut max_gf_nrdev = 0.0f64;
    // Sweep the realistic per-expert tile sizes; 2 and 4 experts to exercise
    // non-trivial expert_offsets + multi-group assembly. Skip M too small to split.
    for &gm in &[64usize, 128, 2048] {
        for &ne in &[2usize, 4] {
            if gm < ne * 2 {
                continue; // need >=2 rows/expert for a meaningful split
            }
            let (coll, orc, nrdev) = run_grouped_fused_shape(
                gpu,
                stream,
                gm,
                n,
                k,
                ne,
                seed ^ (gm as u64).wrapping_mul(0x1357) ^ (ne as u64).wrapping_mul(0x2468),
            )?;
            println!("{gm:>6} {ne:>5} {coll:>22.6} {orc:>18.6} {nrdev:>12.4}");
            min_gf_coll = min_gf_coll.min(coll);
            min_gf_orc = min_gf_orc.min(orc);
            max_gf_nrdev = max_gf_nrdev.max(nrdev);
            if coll < GFUSED_COLLECTIVE_GATE || orc < COSINE_GATE || nrdev > GFUSED_NR_GATE {
                gfused_pass = false;
            }
        }
    }
    println!("{}", "-".repeat(56));
    if gfused_pass {
        println!(
            "GROUPED-FUSED RESULT: PASS (cos_vs_collective>={GFUSED_COLLECTIVE_GATE} min {min_gf_coll:.6}; cos_vs_oracle>={COSINE_GATE} min {min_gf_orc:.6}; scale2_nrdev<{GFUSED_NR_GATE} max {max_gf_nrdev:.4})"
        );
    } else {
        eprintln!(
            "GROUPED-FUSED RESULT: FAIL (collective min {min_gf_coll:.6} gate {GFUSED_COLLECTIVE_GATE}; oracle min {min_gf_orc:.6} gate {COSINE_GATE}; scale2_nrdev max {max_gf_nrdev:.4} gate {GFUSED_NR_GATE})"
        );
        all_pass = false;
    }

    // ── Phase-2 FUSED FP4 kernel gate + speed signal ──
    println!();
    println!("=== Phase-2 FUSED FP4 gate_up kernel (cp.async pipelined, single launch) ===");
    println!(
        "{:>6} {:>10} {:>22} {:>16} {:>12} {:>12}",
        "M", "fused_us", "cos_vs_collective", "cos_vs_oracle", "fp8fused_us", "fp4/fp8"
    );
    println!("{}", "-".repeat(86));
    let mut fused_pass = true;
    let mut min_fu_coll = 1.0f64;
    let mut min_fu_orc = 1.0f64;
    for r in &rows {
        let sp = if r.fused_us > 0.0 {
            r.fused_fp8_us / r.fused_us
        } else {
            0.0
        };
        println!(
            "{:>6} {:>10.2} {:>22.6} {:>16.6} {:>12.2} {:>12.3}",
            r.m, r.fused_us, r.fused_cos_vs_collective, r.fused_cos_vs_oracle, r.fused_fp8_us, sp
        );
        min_fu_coll = min_fu_coll.min(r.fused_cos_vs_collective);
        min_fu_orc = min_fu_orc.min(r.fused_cos_vs_oracle);
        if r.fused_cos_vs_collective < GRP_COLLECTIVE_GATE || r.fused_cos_vs_oracle < COSINE_GATE {
            fused_pass = false;
        }
    }
    println!("{}", "-".repeat(86));
    if let Some(r) = rows.iter().find(|r| r.m == 64) {
        let sp = if r.fused_us > 0.0 {
            r.fused_fp8_us / r.fused_us
        } else {
            0.0
        };
        println!(
            "FUSED HEADLINE @M=64: cos_vs_collective={:.6} cos_vs_oracle={:.6} fused_us={:.2} fp8fused_us={:.2} fp4/fp8={:.3}x",
            r.fused_cos_vs_collective, r.fused_cos_vs_oracle, r.fused_us, r.fused_fp8_us, sp
        );
    }
    if fused_pass {
        println!(
            "FUSED RESULT: PASS (cos_vs_collective>={GRP_COLLECTIVE_GATE} min {min_fu_coll:.6}; cos_vs_oracle>={COSINE_GATE} min {min_fu_orc:.6})"
        );
    } else {
        eprintln!(
            "FUSED RESULT: FAIL (collective min {min_fu_coll:.6} gate {GRP_COLLECTIVE_GATE}; oracle min {min_fu_orc:.6} gate {COSINE_GATE})"
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

    // ── FP4 DOWN kernel gate (moe_w4a16_down_t_k64_fp4) ──
    // Down shape: N = hidden = 2048, K = inter = 512. Realistic M = tokens/expert.
    println!();
    println!("=== FP4 DOWN kernel (moe_w4a16_down_t_k64_fp4): N=2048 (hidden), K=512 (inter) ===");
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>20} {:>16} {:>12}",
        "M", "down_us", "fp8down_us", "w4a16_us", "cos_vs_collective", "cos_vs_oracle", "fp4/fp8"
    );
    println!("{}", "-".repeat(94));
    const DOWN_COLLECTIVE_GATE: f64 = 0.999;
    let down_n = 2048usize;
    let down_k = 512usize;
    let mut down_pass = true;
    let mut min_dn_coll = 1.0f64;
    let mut min_dn_orc = 1.0f64;
    for &m in &m_values {
        let (down_us, fp8_us, w4a16_us, coll, orc) =
            run_down_shape(gpu, stream, m, down_n, down_k, seed ^ (m as u64).wrapping_mul(0x7C15))?;
        let sp = if down_us > 0.0 && fp8_us > 0.0 {
            fp8_us / down_us
        } else {
            0.0
        };
        println!(
            "{:>6} {:>10.2} {:>12.2} {:>12.2} {:>20.6} {:>16.6} {:>12.3}",
            m, down_us, fp8_us, w4a16_us, coll, orc, sp
        );
        min_dn_coll = min_dn_coll.min(coll);
        min_dn_orc = min_dn_orc.min(orc);
        if coll < DOWN_COLLECTIVE_GATE || orc < COSINE_GATE {
            down_pass = false;
        }
    }
    println!("{}", "-".repeat(94));
    if down_pass {
        println!(
            "DOWN RESULT: PASS (cos_vs_collective>={DOWN_COLLECTIVE_GATE} min {min_dn_coll:.6}; cos_vs_oracle>={COSINE_GATE} min {min_dn_orc:.6})"
        );
    } else {
        eprintln!(
            "DOWN RESULT: FAIL (collective min {min_dn_coll:.6} gate {DOWN_COLLECTIVE_GATE}; oracle min {min_dn_orc:.6} gate {COSINE_GATE})"
        );
        all_pass = false;
    }

    if all_pass {
        println!("RESULT: PASS (all fp4 cosine >= {COSINE_GATE}; min {min_fp4_cos:.4})");
        Ok(())
    } else {
        eprintln!("RESULT: FAIL (some fp4 cosine < {COSINE_GATE}; min {min_fp4_cos:.4})");
        std::process::exit(1);
    }
}
