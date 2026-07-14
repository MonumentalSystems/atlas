// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone-6 GPU PoC: **request-batching** for NLLB-200 on CUDA. B different
//! prompts are translated in one batched, lockstep bf16 tensor-core forward:
//! per-sequence encoders → per-sequence cross-attn K/V, then a batched decoder
//! with per-sequence device KV caches (batch-major [B, MAX, d]), M=B tensor-core
//! GEMM projections, batched attention, and batched on-device argmax. Each
//! sequence stops at its own EOS.
//!
//! Validates every sequence token-exact vs its single-stream HF-bf16 greedy
//! output, and reports batched throughput.
//!
//! Run:
//!   ATLAS_NLLB_DIR=/tank/hf/nllb-200-3.3B-bf16 \
//!     cargo run --release -p spark-model --example nllb_cuda_batch --features gpu-examples

use anyhow::{Context, Result};
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};
use std::path::Path;

#[path = "nllb_cuda_batch/ctx.rs"]
mod ctx;
#[allow(unused_imports)]
use ctx::*;

const FORCED_BOS: u32 = 256057; // fra_Latn
const PAD_ID: u32 = 1;
const MAX_NEW: usize = 96;

struct K {
    embed: KernelHandle,
    scale: KernelHandle,
    add: KernelHandle,
    add_row: KernelHandle,
    relu: KernelHandle,
    ln: KernelHandle,
    bias: KernelHandle,
    attn_enc: KernelHandle,
    attn_dec: KernelHandle,
    scatter: KernelHandle,
    gemm: KernelHandle,
    argmax: KernelHandle,
}

fn main() -> Result<()> {
    let dir =
        std::env::var("ATLAS_NLLB_DIR").unwrap_or_else(|_| "/tank/hf/nllb-200-3.3B-bf16".into());
    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.default_stream();

    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        Path::new(&dir).join("config.json"),
    )?)?;
    let d = cfg["d_model"].as_u64().context("d_model")? as usize;
    let heads = cfg["encoder_attention_heads"].as_u64().context("heads")? as usize;
    let ffn = cfg["encoder_ffn_dim"].as_u64().context("ffn")? as usize;
    let enc_layers = cfg["encoder_layers"].as_u64().context("enc_layers")? as usize;
    let dec_layers = cfg["decoder_layers"].as_u64().context("dec_layers")? as usize;
    let vocab = cfg["vocab_size"].as_u64().context("vocab")? as usize;
    let dec_start = cfg["decoder_start_token_id"].as_u64().unwrap_or(2) as u32;
    let eos = cfg["eos_token_id"].as_u64().unwrap_or(2) as u32;
    let head_dim = d / heads;
    let embed_scale = if cfg["scale_embedding"].as_bool().unwrap_or(true) {
        (d as f32).sqrt()
    } else {
        1.0
    };
    let attn_scale = (head_dim as f32).powf(-0.5);

    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;
    let kh = K {
        embed: gpu.kernel("nllb_encoder", "nllb_embed_bf16")?,
        scale: gpu.kernel("nllb_encoder", "nllb_scale_bf16")?,
        add: gpu.kernel("nllb_encoder", "nllb_add_bf16")?,
        add_row: gpu.kernel("nllb_encoder", "nllb_add_row_bf16")?,
        relu: gpu.kernel("nllb_encoder", "nllb_relu_bf16")?,
        ln: gpu.kernel("nllb_encoder", "nllb_layernorm_bf16")?,
        bias: gpu.kernel("nllb_encoder", "nllb_bias_bf16")?,
        attn_enc: gpu.kernel("nllb_encoder", "nllb_attn_kv_bf16")?,
        attn_dec: gpu.kernel("nllb_encoder", "nllb_attn_bdecode")?,
        scatter: gpu.kernel("nllb_encoder", "nllb_scatter_batched")?,
        gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
        argmax: gpu.kernel("nllb_encoder", "nllb_argmax_batched")?,
    };
    let c = Ctx {
        gpu,
        k: &kh,
        store: &store,
        d,
        heads,
        head_dim,
        ffn,
        vocab,
        dec_layers,
        attn_scale,
        embed_scale,
        embed_table: store.get("model.shared.weight")?.ptr,
        dec_start,
        eos,
        stream,
    };

    // ATLAS_NLLB_REP tiles the prompt set for a throughput sweep (correctness
    // still checked per sequence — identical prompts give identical outputs).
    let rep: usize = std::env::var("ATLAS_NLLB_REP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let prompts: Vec<Vec<u32>> = prompts()
        .into_iter()
        .cycle()
        .take(prompts().len() * rep)
        .collect();
    let oracles: Vec<Vec<u32>> = oracles()
        .into_iter()
        .cycle()
        .take(oracles().len() * rep)
        .collect();
    let b = prompts.len();
    let enc_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    let max_enc = *enc_lens.iter().max().unwrap();
    println!("[nllb-batch] B={b} enc_lens={enc_lens:?} d={d} vocab={vocab}");

    // ---- per-sequence encoder → batch-major cross-attn K/V [B, max_enc, d] ----
    let cross_k: Vec<DevicePtr> = (0..dec_layers)
        .map(|_| c.bf16b(b * max_enc * d))
        .collect::<Result<_>>()?;
    let cross_v: Vec<DevicePtr> = (0..dec_layers)
        .map(|_| c.bf16b(b * max_enc * d))
        .collect::<Result<_>>()?;
    let escr = Scratch::new(&c, max_enc)?;
    for (bi, prompt) in prompts.iter().enumerate() {
        let seq = prompt.len();
        let enc_out = c.bf16b(seq * d)?;
        c.embed_seq(prompt, enc_out)?;
        for l in 0..enc_layers {
            let p = format!("model.encoder.layers.{l}");
            c.enc_self_attn(&p, enc_out, seq, &escr)?;
            c.ffn_block(&p, enc_out, seq, &escr)?;
        }
        c.layer_norm("model.encoder.layer_norm", enc_out, seq)?;
        for l in 0..dec_layers {
            let p = format!("model.decoder.layers.{l}.encoder_attn");
            c.linear(
                &format!("{p}.k_proj"),
                enc_out,
                cross_k[l].offset(bi * max_enc * d * 2),
                seq,
                d,
                d,
            )?;
            c.linear(
                &format!("{p}.v_proj"),
                enc_out,
                cross_v[l].offset(bi * max_enc * d * 2),
                seq,
                d,
                d,
            )?;
        }
        c.gpu.free(enc_out)?;
    }

    // ---- batched greedy decode ----
    let t0 = std::time::Instant::now();
    let outs = c.batched_greedy(b, max_enc, &enc_lens, &cross_k, &cross_v, FORCED_BOS)?;
    let dt = t0.elapsed().as_secs_f64();

    let mut ok = true;
    for bi in 0..b {
        let pass = outs[bi] == oracles[bi];
        ok &= pass;
        println!(
            "[nllb-batch] seq{bi} {} ({} tok)",
            if pass { "PASS" } else { "FAIL" },
            outs[bi].len()
        );
        if !pass {
            println!("   got {:?}\n   exp {:?}", outs[bi], oracles[bi]);
        }
    }
    let total_tok: usize = outs.iter().map(|o| o.len()).sum();
    println!(
        "[nllb-batch] {} sequences, {} tokens in {:.3}s = {:.1} tok/s aggregate ({:.1} tok/s/seq)",
        b,
        total_tok,
        dt,
        total_tok as f64 / dt,
        total_tok as f64 / dt / b as f64
    );
    anyhow::ensure!(
        ok,
        "a batched sequence diverged from its single-stream reference"
    );
    println!("[nllb-batch] ALL PASS — request-batching token-exact vs HF-bf16");
    Ok(())
}

struct Ctx<'a> {
    gpu: &'a dyn GpuBackend,
    k: &'a K,
    store: &'a WeightStore,
    d: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    vocab: usize,
    dec_layers: usize,
    attn_scale: f32,
    embed_scale: f32,
    embed_table: DevicePtr,
    dec_start: u32,
    eos: u32,
    stream: u64,
}

struct Scratch {
    residual: DevicePtr,
    normed: DevicePtr,
    q: DevicePtr,
    kk: DevicePtr,
    v: DevicePtr,
    attn: DevicePtr,
    proj: DevicePtr,
    ff: DevicePtr,
}
