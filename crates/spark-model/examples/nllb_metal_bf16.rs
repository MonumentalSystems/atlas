// SPDX-License-Identifier: AGPL-3.0-only

//! Metal NLLB-200 / M2M-100 decode validation with BF16 safetensors, device KV
//! cache, greedy decode, and beam search.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightDtype, WeightLoader, WeightStore};
use std::path::Path;

#[path = "nllb_metal_bf16/ctx.rs"]
mod ctx;
#[path = "nllb_metal_bf16/decode.rs"]
mod decode;

use ctx::{Ctx, Kernels, Scratch, bf16_bytes, decoder_pos_table_bf16};
use decode::{DecCtx, DecScratch};

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057;
pub const MAX_NEW: usize = 96;
pub const CACHE_ROWS: usize = MAX_NEW + 2;
const EXPECTED_GREEDY: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 123732, 248105, 30213, 385, 2,
];
const EXPECTED_BEAM5: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 4255, 956, 34821, 248105, 30213, 102506, 248116, 15510,
    385, 2,
];

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
    println!(
        "[nllb-metal-bf16] d={d} heads={heads} ffn={ffn} enc={enc_layers} dec={dec_layers} vocab={vocab}"
    );

    println!("[nllb-metal-bf16] loading BF16 weights to Metal ...");
    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;
    anyhow::ensure!(
        store.get("model.shared.weight")?.dtype == WeightDtype::BF16,
        "expected a BF16 safetensors checkpoint at {dir}"
    );
    let kernels = load_kernels(gpu)?;
    let embed_table = store.get("model.shared.weight")?.ptr;
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
        enc_len: INPUT_IDS.len(),
        attn_scale,
        embed_scale,
        embed_table,
        dec_start,
        eos,
        stream,
    };

    let enc_out = ctx.bf16b(ctx.enc_len * d)?;
    let escr = Scratch::new(&ctx, ctx.enc_len)?;
    ctx.embed_seq(INPUT_IDS, enc_out)?;
    for l in 0..enc_layers {
        let p = format!("model.encoder.layers.{l}");
        ctx.enc_self_attn(&p, enc_out, ctx.enc_len, &escr)?;
        ctx.ffn_block(&p, enc_out, ctx.enc_len, &escr)?;
    }
    ctx.layer_norm("model.encoder.layer_norm", enc_out, ctx.enc_len)?;

    let mut cross_k = Vec::with_capacity(dec_layers);
    let mut cross_v = Vec::with_capacity(dec_layers);
    for l in 0..dec_layers {
        let p = format!("model.decoder.layers.{l}.encoder_attn");
        let ck = ctx.bf16b(ctx.enc_len * d)?;
        let cv = ctx.bf16b(ctx.enc_len * d)?;
        ctx.linear(&format!("{p}.k_proj"), enc_out, ck, ctx.enc_len, d, d)?;
        ctx.linear(&format!("{p}.v_proj"), enc_out, cv, ctx.enc_len, d, d)?;
        cross_k.push(ck);
        cross_v.push(cv);
    }

    let pos_table = ctx.bf16b(CACHE_ROWS * d)?;
    gpu.copy_h2d(
        bf16_bytes(&decoder_pos_table_bf16(CACHE_ROWS, d)),
        pos_table,
    )?;
    let dctx = DecCtx {
        ctx: &ctx,
        cross_k: &cross_k,
        cross_v: &cross_v,
        pos_table,
    };
    let dscr = DecScratch::new(&ctx)?;

    let t0 = std::time::Instant::now();
    let greedy = dctx.greedy(&dscr, FORCED_BOS)?;
    println!("[nllb-metal-bf16] greedy ids = {greedy:?}");
    anyhow::ensure!(greedy == EXPECTED_GREEDY, "BF16 greedy diverged");
    println!(
        "[nllb-metal-bf16] greedy PASS in {:.3}s",
        t0.elapsed().as_secs_f64()
    );

    let t1 = std::time::Instant::now();
    let beam = dctx.beam(&dscr, FORCED_BOS, 5, 1.0)?;
    println!("[nllb-metal-bf16] beam=5 ids = {beam:?}");
    println!(
        "[nllb-metal-bf16] beam=5 {} in {:.3}s",
        if beam == EXPECTED_BEAM5 {
            "matches HF"
        } else {
            "differs from HF"
        },
        t1.elapsed().as_secs_f64()
    );
    Ok(())
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
        gemv_batched: gpu.kernel("nllb_encoder", "nllb_gemv_batched_bf16")?,
        attn: gpu.kernel("nllb_encoder", "nllb_attn_kv_bf16")?,
        add_row: gpu.kernel("nllb_encoder", "nllb_add_row_bf16")?,
        scatter: gpu.kernel("nllb_encoder", "nllb_scatter_batched")?,
        gather: gpu.kernel("nllb_encoder", "nllb_gather_batched")?,
        attn_decode: gpu.kernel("nllb_encoder", "nllb_attn_bdecode")?,
        argmax_batched: gpu.kernel("nllb_encoder", "nllb_argmax_batched")?,
        topk_lse: gpu.kernel("nllb_encoder", "nllb_topk_lse_bf16")?,
        argmax: gpu.kernel("argmax_bf16", "argmax_bf16")?,
    })
}
