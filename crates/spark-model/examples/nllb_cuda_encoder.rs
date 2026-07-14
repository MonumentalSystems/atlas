// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone-1 GPU PoC: NLLB-200 / M2M-100 **encoder** forward on CUDA, using
//! self-contained fp32 kernels (`kernels/gb10/common/nllb_encoder.cu`) plus the
//! production `GpuBackend` / `SafetensorsLoader` infra. Validates the encoder
//! output against the bit-faithful CPU reference (`spark-nllb`):
//!   sum(last_hidden_state) == -14.769035  (input "Hello, world. How are you today?")
//!
//! Run (see PR / notes for the NCCL + ATLAS_TARGET env):
//!   ATLAS_NLLB_DIR=/tank/hf/nllb-200-3.3B-st \
//!     cargo run --release -p spark-model --example nllb_cuda_encoder --features gpu-examples

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};
use std::path::Path;

// Encoder input ids (eng_Latn "Hello, world. How are you today?") — matches the
// spark-nllb reference test.
const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const PAD_ID: u32 = 1;
const REFERENCE_SUM: f32 = -14.769035;

fn main() -> Result<()> {
    let dir =
        std::env::var("ATLAS_NLLB_DIR").unwrap_or_else(|_| "/tank/hf/nllb-200-3.3B-st".into());

    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.default_stream();

    // Config (only what the encoder needs).
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        Path::new(&dir).join("config.json"),
    )?)?;
    let d = cfg["d_model"].as_u64().context("d_model")? as usize;
    let heads = cfg["encoder_attention_heads"].as_u64().context("heads")? as usize;
    let ffn = cfg["encoder_ffn_dim"].as_u64().context("ffn")? as usize;
    let n_layers = cfg["encoder_layers"].as_u64().context("layers")? as usize;
    let head_dim = d / heads;
    let seq = INPUT_IDS.len();
    let embed_scale = (d as f32).sqrt();
    let attn_scale = (head_dim as f32).powf(-0.5);
    println!(
        "[nllb-cuda] d={d} heads={heads} head_dim={head_dim} ffn={ffn} layers={n_layers} seq={seq}"
    );

    println!("[nllb-cuda] loading weights → GPU ...");
    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;

    // Kernel handles (all from the self-contained `nllb_encoder` module).
    let k_embed = gpu.kernel("nllb_encoder", "nllb_embed")?;
    let k_scale = gpu.kernel("nllb_encoder", "nllb_scale_inplace")?;
    let k_add = gpu.kernel("nllb_encoder", "nllb_add_inplace")?;
    let k_relu = gpu.kernel("nllb_encoder", "nllb_relu_inplace")?;
    let k_ln = gpu.kernel("nllb_encoder", "nllb_layernorm")?;
    let k_lin = gpu.kernel("nllb_encoder", "nllb_linear")?;
    let k_attn = gpu.kernel("nllb_encoder", "nllb_attention")?;

    let w = |name: &str| -> Result<DevicePtr> { Ok(store.get(name)?.ptr) };

    // Device scratch (fp32).
    let f32b = |elems: usize| -> Result<DevicePtr> { gpu.alloc(elems * 4) };
    let hidden = f32b(seq * d)?;
    let residual = f32b(seq * d)?;
    let normed = f32b(seq * d)?;
    let qb = f32b(seq * d)?;
    let kb = f32b(seq * d)?;
    let vb = f32b(seq * d)?;
    let attn = f32b(seq * d)?;
    let proj = f32b(seq * d)?;
    let ffbuf = f32b(seq * ffn)?;

    // ---- embed + scale + sinusoidal positions ----
    let ids_dev = gpu.alloc(seq * 4)?;
    gpu.copy_h2d(u32_bytes(INPUT_IDS), ids_dev)?;
    let embed_table = w("model.shared.weight")?;
    KernelLaunch::new(gpu, k_embed)
        .grid([seq as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ids_dev)
        .arg_ptr(embed_table)
        .arg_ptr(hidden)
        .arg_u32(d as u32)
        .launch(stream)?;
    launch_1d(gpu, k_scale, seq * d, stream, |kl| {
        kl.arg_ptr(hidden)
            .arg_u32((seq * d) as u32)
            .arg_f32(embed_scale)
    })?;
    // sinusoidal positions (host fp32) → add
    let pos = sinusoid_positions(INPUT_IDS, d, PAD_ID);
    let pos_dev = f32b(seq * d)?;
    gpu.copy_h2d(f32_bytes(&pos), pos_dev)?;
    launch_1d(gpu, k_add, seq * d, stream, |kl| {
        kl.arg_ptr(hidden)
            .arg_ptr(pos_dev)
            .arg_u32((seq * d) as u32)
    })?;

    // ---- encoder layers (pre-norm) ----
    let ln = |x: DevicePtr, wn: DevicePtr, bn: DevicePtr| -> Result<()> {
        KernelLaunch::new(gpu, k_ln)
            .grid([seq as u32, 1, 1])
            .block([256, 1, 1])
            .shared_mem(256 * 4)
            .arg_ptr(x)
            .arg_ptr(wn)
            .arg_ptr(bn)
            .arg_u32(seq as u32)
            .arg_u32(d as u32)
            .arg_f32(1e-5)
            .launch(stream)
    };
    let linear = |a: DevicePtr,
                  wt: DevicePtr,
                  bias: DevicePtr,
                  c: DevicePtr,
                  n_out: usize,
                  k_in: usize|
     -> Result<()> {
        KernelLaunch::new(gpu, k_lin)
            .grid([div_ceil(n_out as u32, 16), div_ceil(seq as u32, 16), 1])
            .block([16, 16, 1])
            .arg_ptr(a)
            .arg_ptr(wt)
            .arg_ptr(bias)
            .arg_ptr(c)
            .arg_u32(seq as u32)
            .arg_u32(n_out as u32)
            .arg_u32(k_in as u32)
            .launch(stream)
    };

    for l in 0..n_layers {
        let p = format!("model.encoder.layers.{l}");
        // self-attention block
        gpu.copy_d2d(hidden, residual, seq * d * 4)?;
        gpu.copy_d2d(hidden, normed, seq * d * 4)?;
        ln(
            normed,
            w(&format!("{p}.self_attn_layer_norm.weight"))?,
            w(&format!("{p}.self_attn_layer_norm.bias"))?,
        )?;
        linear(
            normed,
            w(&format!("{p}.self_attn.q_proj.weight"))?,
            w(&format!("{p}.self_attn.q_proj.bias"))?,
            qb,
            d,
            d,
        )?;
        linear(
            normed,
            w(&format!("{p}.self_attn.k_proj.weight"))?,
            w(&format!("{p}.self_attn.k_proj.bias"))?,
            kb,
            d,
            d,
        )?;
        linear(
            normed,
            w(&format!("{p}.self_attn.v_proj.weight"))?,
            w(&format!("{p}.self_attn.v_proj.bias"))?,
            vb,
            d,
            d,
        )?;
        KernelLaunch::new(gpu, k_attn)
            .grid([(seq * heads) as u32, 1, 1])
            .block([head_dim as u32, 1, 1])
            .shared_mem(((seq + head_dim) * 4) as u32)
            .arg_ptr(qb)
            .arg_ptr(kb)
            .arg_ptr(vb)
            .arg_ptr(attn)
            .arg_u32(seq as u32)
            .arg_u32(heads as u32)
            .arg_u32(head_dim as u32)
            .arg_f32(attn_scale)
            .launch(stream)?;
        linear(
            attn,
            w(&format!("{p}.self_attn.out_proj.weight"))?,
            w(&format!("{p}.self_attn.out_proj.bias"))?,
            proj,
            d,
            d,
        )?;
        launch_1d(gpu, k_add, seq * d, stream, |kl| {
            kl.arg_ptr(proj).arg_ptr(residual).arg_u32((seq * d) as u32)
        })?;
        gpu.copy_d2d(proj, hidden, seq * d * 4)?;

        // FFN block
        gpu.copy_d2d(hidden, residual, seq * d * 4)?;
        gpu.copy_d2d(hidden, normed, seq * d * 4)?;
        ln(
            normed,
            w(&format!("{p}.final_layer_norm.weight"))?,
            w(&format!("{p}.final_layer_norm.bias"))?,
        )?;
        linear(
            normed,
            w(&format!("{p}.fc1.weight"))?,
            w(&format!("{p}.fc1.bias"))?,
            ffbuf,
            ffn,
            d,
        )?;
        launch_1d(gpu, k_relu, seq * ffn, stream, |kl| {
            kl.arg_ptr(ffbuf).arg_u32((seq * ffn) as u32)
        })?;
        linear(
            ffbuf,
            w(&format!("{p}.fc2.weight"))?,
            w(&format!("{p}.fc2.bias"))?,
            proj,
            d,
            ffn,
        )?;
        launch_1d(gpu, k_add, seq * d, stream, |kl| {
            kl.arg_ptr(proj).arg_ptr(residual).arg_u32((seq * d) as u32)
        })?;
        gpu.copy_d2d(proj, hidden, seq * d * 4)?;
    }
    ln(
        hidden,
        w("model.encoder.layer_norm.weight")?,
        w("model.encoder.layer_norm.bias")?,
    )?;

    // ---- checksum vs CPU reference ----
    gpu.synchronize(stream)?;
    let mut host = vec![0u8; seq * d * 4];
    gpu.copy_d2h(hidden, &mut host)?;
    let out: &[f32] = f32_slice(&host);
    let sum: f32 = out.iter().sum();
    println!(
        "[nllb-cuda] encoder out [{seq}, {d}]  first-tok first5 = {:?}",
        &out[..5]
            .iter()
            .map(|v| (v * 1e5).round() / 1e5)
            .collect::<Vec<_>>()
    );
    println!("[nllb-cuda] SUM = {sum:.6}   (CPU reference {REFERENCE_SUM:.6})");
    let err = (sum - REFERENCE_SUM).abs();
    if err < 0.2 {
        println!("[nllb-cuda] PASS  (|Δ| = {err:.4})");
        Ok(())
    } else {
        anyhow::bail!("FAIL: encoder sum {sum} diverged from reference {REFERENCE_SUM} (|Δ|={err})")
    }
}

/// Masked incremental sinusoidal positions (M2M-100 convention: non-pad start
/// at padding_idx+1; pad → zeroed padding_idx row). fp32, `[seq, d]`.
fn sinusoid_positions(ids: &[u32], d: usize, pad: u32) -> Vec<f32> {
    let seq = ids.len();
    let half = d / 2;
    let emb_scale = 10000f32.ln() / (half as f32 - 1.0);
    let mut pos = vec![0f32; seq * d];
    let mut running = 0u32;
    for (i, &id) in ids.iter().enumerate() {
        let p = if id != pad {
            running += 1;
            running + pad
        } else {
            pad
        };
        if p == pad {
            continue;
        }
        for j in 0..half {
            let freq = (-(j as f32) * emb_scale).exp();
            let ang = p as f32 * freq;
            pos[i * d + j] = ang.sin();
            pos[i * d + half + j] = ang.cos();
        }
    }
    pos
}

fn launch_1d(
    gpu: &dyn GpuBackend,
    kernel: spark_runtime::gpu::KernelHandle,
    n: usize,
    stream: u64,
    args: impl FnOnce(KernelLaunch) -> KernelLaunch,
) -> Result<()> {
    let kl = KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n as u32, 256), 1, 1])
        .block([256, 1, 1]);
    args(kl).launch(stream)
}

fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn f32_slice(b: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}
