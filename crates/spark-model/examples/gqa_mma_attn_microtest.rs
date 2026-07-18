// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness oracle for the GQA-group-packed MMA flash-decode kernels against
//! the scalar `paged_decode_attn` golden reference on IDENTICAL paged inputs.
//!
//! Three kernels are compared per seq_len:
//!   * scalar `paged_decode_attn`                          (golden reference)
//!   * `paged_decode_attn_gqa_mma`                         (Inc1, non-split)
//!   * `paged_decode_attn_gqa_mma_splitk` + `..._reduce`   (Inc2/3, num_splits=3)
//! The split path is checked against BOTH the scalar reference AND the non-split
//! MMA — this catches split-range / reduce / workspace-addressing bugs.
//!
//! Model facts (Qwen3.6-35B BF16 KV): head_dim=256, q_heads=16, kv_heads=2,
//! GQA group=8, block_size=16, full attention (sliding=0). For each seq_len in
//! {16, 31, 64, 100} it builds synthetic Q/K/V + a (reversed) block table,
//! launches all kernels, and reports per-(q_head,dim) max-abs error and the
//! argmax-flip count of O.
//!
//! Bit-exactness is impossible (MMA reorders sums; P is bf16). PASS gate (each
//! comparison): max-abs error < 2^-6 relative to max|O_scalar| AND 0 argmax flips.
//!
//! Usage: cargo run --release -p spark-model --example gqa_mma_attn_microtest \
//!          --features cuda,gpu-examples -- [seed]
//! Exit 0 = all PASS, 1 = any FAIL.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const HDIM: usize = 256;
const NQ: usize = 16;
const NKV: usize = 2;
const BLOCK_SIZE: usize = 16;
const REL_GATE: f32 = 1.0 / 64.0; // 2^-6
const NUM_SPLITS_TEST: u32 = 3; // force split-KV (mirrors 48/(2*8) at C=8)
// Dynamic smem for the split-KV kernel: sQ 4224 + m 128 + l 128 + pool 67584.
const GQA_MMA_SPLITK_SMEM: u32 = 72_064;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    ((bits.wrapping_add(0x7FFF + ((bits >> 16) & 1))) >> 16) as u16
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32s_to_le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len())?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}
fn download_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let seed: u64 = a.get(1).map_or(0x6A11, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x6A11)
    });
    let inv_sqrt_d = 1.0f32 / (HDIM as f32).sqrt();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let scalar_k = gpu.kernel("paged_decode", "paged_decode_attn")?;
    let gqa_k = gpu.kernel("paged_decode", "paged_decode_attn_gqa_mma")?;
    let split_k = gpu.kernel("paged_decode", "paged_decode_attn_gqa_mma_splitk")?;
    let reduce_k = gpu.kernel("paged_decode", "paged_decode_attn_reduce")?;

    let mut all_pass = true;
    for &seq_len in &[16usize, 31, 64, 100] {
        let pass = run_one(
            gpu, stream, scalar_k, gqa_k, split_k, reduce_k, seq_len, inv_sqrt_d, seed,
        )?;
        all_pass &= pass;
    }

    if all_pass {
        println!("RESULT: PASS (all seq_len)");
        Ok(())
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
}

/// Per-(q_head,dim) max-abs error, reference magnitude, and argmax-flip count
/// over dims, between two bf16-bit output tensors laid out `[NQ][HDIM]`.
fn compare(reference: &[u16], test: &[u16]) -> (f32, f32, usize) {
    let mut max_abs = 0f32;
    let mut max_ref = 0f32;
    let mut argmax_flips = 0usize;
    for h in 0..NQ {
        let base = h * HDIM;
        let (mut am_r, mut am_t) = (0usize, 0usize);
        let (mut mv_r, mut mv_t) = (f32::MIN, f32::MIN);
        for d in 0..HDIM {
            let rv = bf16_bits_to_f32(reference[base + d]);
            let tv = bf16_bits_to_f32(test[base + d]);
            max_abs = max_abs.max((rv - tv).abs());
            max_ref = max_ref.max(rv.abs());
            if rv > mv_r {
                mv_r = rv;
                am_r = d;
            }
            if tv > mv_t {
                mv_t = tv;
                am_t = d;
            }
        }
        if am_r != am_t {
            argmax_flips += 1;
        }
    }
    (max_abs, max_ref, argmax_flips)
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    gpu: &dyn GpuBackend,
    stream: u64,
    scalar_k: spark_runtime::gpu::KernelHandle,
    gqa_k: spark_runtime::gpu::KernelHandle,
    split_k: spark_runtime::gpu::KernelHandle,
    reduce_k: spark_runtime::gpu::KernelHandle,
    seq_len: usize,
    inv_sqrt_d: f32,
    seed: u64,
) -> Result<bool> {
    let num_seqs = 1usize;
    let num_blocks = seq_len.div_ceil(BLOCK_SIZE);
    let max_blocks_per_seq = num_blocks;
    let q_stride = NQ * HDIM;

    let mut rng = Rng(seed ^ (seq_len as u64).wrapping_mul(0x100000001B3));

    // Q: [num_seqs, NQ, HDIM]
    let q: Vec<u16> = (0..num_seqs * NQ * HDIM)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();

    // Logical K/V: [seq_len, NKV, HDIM]
    let k_log: Vec<u16> = (0..seq_len * NKV * HDIM)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let v_log: Vec<u16> = (0..seq_len * NKV * HDIM)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();

    // Reversed block table (exercises the paged indirection): logical block i
    // -> physical block (num_blocks-1-i).
    let block_table: Vec<i32> = (0..num_blocks).map(|i| (num_blocks - 1 - i) as i32).collect();

    // Scatter logical K/V into the paged NHD pool: [num_blocks, BLOCK_SIZE, NKV, HDIM].
    let pool_elems = num_blocks * BLOCK_SIZE * NKV * HDIM;
    let mut k_cache = vec![0u16; pool_elems];
    let mut v_cache = vec![0u16; pool_elems];
    for pos in 0..seq_len {
        let lb = pos / BLOCK_SIZE;
        let off = pos % BLOCK_SIZE;
        let pb = block_table[lb] as usize;
        for kvh in 0..NKV {
            for d in 0..HDIM {
                let dst = ((pb * BLOCK_SIZE + off) * NKV + kvh) * HDIM + d;
                let src = (pos * NKV + kvh) * HDIM + d;
                k_cache[dst] = k_log[src];
                v_cache[dst] = v_log[src];
            }
        }
    }

    let seq_lens: Vec<i32> = vec![seq_len as i32];

    let qp = upload(gpu, &u16s_to_le(&q))?;
    let kp = upload(gpu, &u16s_to_le(&k_cache))?;
    let vp = upload(gpu, &u16s_to_le(&v_cache))?;
    let btp = upload(gpu, &i32s_to_le(&block_table))?;
    let slp = upload(gpu, &i32s_to_le(&seq_lens))?;
    let o_scalar = gpu.alloc(num_seqs * NQ * HDIM * 2)?;
    let o_gqa = gpu.alloc(num_seqs * NQ * HDIM * 2)?;
    let o_split = gpu.alloc(num_seqs * NQ * HDIM * 2)?;
    // Split-KV workspace: [num_seqs, NQ, num_splits, HDIM+2] f32.
    let ns = NUM_SPLITS_TEST as usize;
    let workspace = gpu.alloc(num_seqs * NQ * ns * (HDIM + 2) * 4)?;

    // Scalar reference: grid (NQ, num_seqs, 1), block (256,1,1).
    KernelLaunch::new(gpu, scalar_k)
        .grid([NQ as u32, num_seqs as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(o_scalar)
        .arg_ptr(btp)
        .arg_ptr(slp)
        .arg_u32(max_blocks_per_seq as u32)
        .arg_u32(NQ as u32)
        .arg_u32(NKV as u32)
        .arg_u32(HDIM as u32)
        .arg_u32(BLOCK_SIZE as u32)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride as u32)
        .arg_u32(0) // sliding_window = 0 (full attention)
        .launch(stream)?;

    // GQA-MMA: grid (NKV, 1, num_seqs), block (128,1,1).
    KernelLaunch::new(gpu, gqa_k)
        .grid([NKV as u32, 1, num_seqs as u32])
        .block([128, 1, 1])
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(o_gqa)
        .arg_ptr(btp)
        .arg_ptr(slp)
        .arg_u32(max_blocks_per_seq as u32)
        .arg_u32(NQ as u32)
        .arg_u32(NKV as u32)
        .arg_u32(HDIM as u32)
        .arg_u32(BLOCK_SIZE as u32)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride as u32)
        .arg_u32(0)
        .launch(stream)?;

    // Split-KV: grid (NKV, num_splits, num_seqs), block (128,1,1), dynamic smem.
    // Writes un-normalized (O,m,l) partials to `workspace`.
    KernelLaunch::new(gpu, split_k)
        .grid([NKV as u32, NUM_SPLITS_TEST, num_seqs as u32])
        .block([128, 1, 1])
        .shared_mem(GQA_MMA_SPLITK_SMEM)
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(o_split) // used only when num_splits==1; harmless here
        .arg_ptr(workspace)
        .arg_ptr(btp)
        .arg_ptr(slp)
        .arg_u32(max_blocks_per_seq as u32)
        .arg_u32(NQ as u32)
        .arg_u32(NKV as u32)
        .arg_u32(HDIM as u32)
        .arg_u32(BLOCK_SIZE as u32)
        .arg_f32(inv_sqrt_d)
        .arg_u32(NUM_SPLITS_TEST)
        .arg_u32(q_stride as u32)
        .arg_u32(0) // sliding_window = 0
        .launch(stream)?;

    // Reduce: grid (NQ, num_seqs, 1), block (32,1,1). Combines splits -> o_split.
    KernelLaunch::new(gpu, reduce_k)
        .grid([NQ as u32, num_seqs as u32, 1])
        .block([32, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(o_split)
        .arg_ptr(slp)
        .arg_u32(NQ as u32)
        .arg_u32(HDIM as u32)
        .arg_u32(NUM_SPLITS_TEST)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    let os = download_bf16(gpu, o_scalar, num_seqs * NQ * HDIM)?;
    let og = download_bf16(gpu, o_gqa, num_seqs * NQ * HDIM)?;
    let osp = download_bf16(gpu, o_split, num_seqs * NQ * HDIM)?;

    // Three comparisons: MMA vs scalar, split vs scalar, split vs non-split MMA.
    let mut pass = true;
    for (label, refo, testo) in [
        ("mma  vs scalar", &os, &og),
        ("split vs scalar", &os, &osp),
        ("split vs mma   ", &og, &osp),
    ] {
        let (max_abs, max_ref, flips) = compare(refo, testo);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
        let ok = rel < REL_GATE && flips == 0;
        pass &= ok;
        println!(
            "seq_len={seq_len:>3} blocks={num_blocks:>2} splits={NUM_SPLITS_TEST}  {label}  max_abs={max_abs:.3e}  rel={rel:.3e} (gate {REL_GATE:.3e})  argmax_flips={flips}/{NQ}  => {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    for p in [qp, kp, vp, btp, slp, o_scalar, o_gqa, o_split, workspace] {
        gpu.free(p).ok();
    }
    Ok(pass)
}
