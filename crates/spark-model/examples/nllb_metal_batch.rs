// SPDX-License-Identifier: AGPL-3.0-only

//! Metal BF16 request batching for NLLB-200 / M2M-100 greedy translation.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightDtype, WeightLoader, WeightStore};
use std::path::Path;

#[path = "nllb_metal_batch/batch.rs"]
mod batch;
#[allow(dead_code)]
#[path = "nllb_metal_bf16/ctx.rs"]
mod ctx;

use ctx::{Ctx, Kernels, Scratch};

const FORCED_BOS: u32 = 256057;
pub const MAX_NEW: usize = 96;

fn prompts() -> Vec<Vec<u32>> {
    vec![
        vec![
            256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
        ],
        vec![256047, 1617, 167554, 248, 43978, 248075, 2],
        vec![256047, 138409, 200356, 248, 9, 19450, 5753, 248075, 2],
        vec![
            256047, 117, 9713, 6399, 9, 54445, 452, 121318, 248079, 43205, 248075, 2,
        ],
    ]
}

fn oracles() -> Vec<Vec<u32>> {
    vec![
        vec![
            256057, 17994, 141190, 248079, 25358, 123732, 248105, 30213, 385, 2,
        ],
        vec![256057, 1181, 14183, 613, 84809, 248075, 2],
        vec![
            256057, 1034, 80431, 1590, 88752, 1956, 613, 159, 86106, 80198, 248075, 2,
        ],
        vec![
            256057, 1048, 190412, 3335, 2626, 201, 79, 78752, 248079, 10, 248116, 73, 4255, 161248,
            248075, 2,
        ],
    ]
}

fn main() -> Result<()> {
    let dir =
        std::env::var("ATLAS_NLLB_DIR").unwrap_or_else(|_| "/tank/hf/nllb-200-3.3B-bf16".into());
    let modules = atlas_kernels::metallib_modules();
    if modules.is_empty() {
        bail!(
            "metal kernel registry empty; rebuild with ATLAS_TARGET_HW=metal \
             ATLAS_TARGET_MODEL=nllb-200-3.3b ATLAS_TARGET_QUANT=bf16"
        );
    }
    let backend = MetalGpuBackend::new(0, &modules)?;
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

    println!("[nllb-metal-batch] loading BF16 weights to Metal ...");
    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;
    anyhow::ensure!(
        store.get("model.shared.weight")?.dtype == WeightDtype::BF16,
        "expected a BF16 safetensors checkpoint at {dir}"
    );
    let kernels = load_kernels(gpu)?;
    let base_prompts = prompts();
    let base_oracles = oracles();
    let rep: usize = std::env::var("ATLAS_NLLB_REP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let prompts: Vec<Vec<u32>> = base_prompts
        .iter()
        .cloned()
        .cycle()
        .take(base_prompts.len() * rep)
        .collect();
    let oracles: Vec<Vec<u32>> = base_oracles
        .iter()
        .cloned()
        .cycle()
        .take(base_oracles.len() * rep)
        .collect();
    let b = prompts.len();
    let enc_lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let max_enc = *enc_lens.iter().max().context("empty batch")?;
    println!("[nllb-metal-batch] B={b} enc_lens={enc_lens:?} d={d} vocab={vocab}");

    let ctx = Ctx {
        gpu,
        k: &kernels,
        store: &store,
        d,
        heads,
        head_dim,
        ffn,
        vocab,
        dec_layers,
        enc_len: max_enc,
        attn_scale,
        embed_scale,
        embed_table: store.get("model.shared.weight")?.ptr,
        dec_start,
        eos,
        stream,
    };

    let cross_k: Vec<DevicePtr> = (0..dec_layers)
        .map(|_| ctx.bf16b(b * max_enc * d))
        .collect::<Result<_>>()?;
    let cross_v: Vec<DevicePtr> = (0..dec_layers)
        .map(|_| ctx.bf16b(b * max_enc * d))
        .collect::<Result<_>>()?;
    let escr = Scratch::new(&ctx, max_enc)?;
    for (bi, prompt) in prompts.iter().enumerate() {
        encode_one(
            &ctx, &escr, enc_layers, dec_layers, max_enc, bi, prompt, &cross_k, &cross_v,
        )?;
    }

    let t0 = std::time::Instant::now();
    let outs = ctx.batched_greedy(b, max_enc, &enc_lens, &cross_k, &cross_v, FORCED_BOS)?;
    let dt = t0.elapsed().as_secs_f64();
    let mut ok = true;
    for bi in 0..b {
        let pass = outs[bi] == oracles[bi];
        ok &= pass;
        println!(
            "[nllb-metal-batch] seq{bi} {} ({} tok)",
            if pass { "PASS" } else { "FAIL" },
            outs[bi].len()
        );
        if !pass {
            println!("  got {:?}\n  exp {:?}", outs[bi], oracles[bi]);
        }
    }
    let total_tok: usize = outs.iter().map(Vec::len).sum();
    println!(
        "[nllb-metal-batch] {b} sequences, {total_tok} tokens in {dt:.3}s = {:.1} tok/s aggregate ({:.1} tok/s/seq)",
        total_tok as f64 / dt,
        total_tok as f64 / dt / b as f64
    );
    anyhow::ensure!(ok, "a batched sequence diverged from its reference");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_one(
    ctx: &Ctx,
    scr: &Scratch,
    enc_layers: usize,
    dec_layers: usize,
    max_enc: usize,
    bi: usize,
    prompt: &[u32],
    cross_k: &[DevicePtr],
    cross_v: &[DevicePtr],
) -> Result<()> {
    let seq = prompt.len();
    let enc_out = ctx.bf16b(seq * ctx.d)?;
    ctx.embed_seq(prompt, enc_out)?;
    for l in 0..enc_layers {
        let p = format!("model.encoder.layers.{l}");
        ctx.enc_self_attn(&p, enc_out, seq, scr)?;
        ctx.ffn_block(&p, enc_out, seq, scr)?;
    }
    ctx.layer_norm("model.encoder.layer_norm", enc_out, seq)?;
    for l in 0..dec_layers {
        let p = format!("model.decoder.layers.{l}.encoder_attn");
        let off = bi * max_enc * ctx.d * 2;
        ctx.linear(
            &format!("{p}.k_proj"),
            enc_out,
            cross_k[l].offset(off),
            seq,
            ctx.d,
            ctx.d,
        )?;
        ctx.linear(
            &format!("{p}.v_proj"),
            enc_out,
            cross_v[l].offset(off),
            seq,
            ctx.d,
            ctx.d,
        )?;
    }
    ctx.gpu.free(enc_out)
}

fn load_kernels(gpu: &dyn GpuBackend) -> Result<Kernels> {
    Ok(Kernels {
        embed: gpu.kernel("nllb_encoder", "nllb_embed_bf16")?,
        scale: gpu.kernel("nllb_encoder", "nllb_scale_bf16")?,
        add: gpu.kernel("nllb_encoder", "nllb_add_bf16")?,
        relu: gpu.kernel("nllb_encoder", "nllb_relu_bf16")?,
        ln: gpu.kernel("nllb_encoder", "nllb_layernorm_bf16")?,
        lin: gpu.kernel("nllb_encoder", "nllb_linear_bf16")?,
        lin_no_bias: gpu.kernel("nllb_encoder", "nllb_linear_no_bias_bf16")?,
        gemv: gpu.kernel("nllb_encoder", "nllb_gemv_bf16")?,
        gemv_no_bias: gpu.kernel("nllb_encoder", "nllb_gemv_bf16_no_bias")?,
        attn: gpu.kernel("nllb_encoder", "nllb_attn_kv_bf16")?,
        add_row: gpu.kernel("nllb_encoder", "nllb_add_row_bf16")?,
        scatter: gpu.kernel("nllb_encoder", "nllb_scatter_batched")?,
        attn_decode: gpu.kernel("nllb_encoder", "nllb_attn_bdecode")?,
        argmax_batched: gpu.kernel("nllb_encoder", "nllb_argmax_batched")?,
        argmax: gpu.kernel("argmax_bf16", "argmax_bf16")?,
    })
}
