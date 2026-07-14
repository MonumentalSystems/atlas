// SPDX-License-Identifier: AGPL-3.0-only

//! Metal BF16 beam batching for NLLB-200 / M2M-100 translation.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightDtype, WeightLoader, WeightStore};
use std::path::Path;

#[path = "nllb_metal_batch/batch.rs"]
#[allow(dead_code)]
mod batch;
#[path = "nllb_metal_beambatch/beam.rs"]
mod beam;
#[allow(dead_code)]
#[path = "nllb_metal_bf16/ctx.rs"]
mod ctx;

use ctx::{Ctx, Kernels, Scratch};

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057;
const NUM_BEAMS: usize = 5;
pub const MAX_NEW: usize = 96;
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

    println!("[nllb-metal-beambatch] loading BF16 weights to Metal ...");
    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;
    anyhow::ensure!(
        store.get("model.shared.weight")?.dtype == WeightDtype::BF16,
        "expected a BF16 safetensors checkpoint at {dir}"
    );
    let kernels = load_kernels(gpu)?;
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
        embed_table: store.get("model.shared.weight")?.ptr,
        dec_start,
        eos,
        stream,
    };

    let enc_len = INPUT_IDS.len();
    let enc_out = ctx.bf16b(enc_len * d)?;
    let escr = Scratch::new(&ctx, enc_len)?;
    ctx.embed_seq(INPUT_IDS, enc_out)?;
    for l in 0..enc_layers {
        let p = format!("model.encoder.layers.{l}");
        ctx.enc_self_attn(&p, enc_out, enc_len, &escr)?;
        ctx.ffn_block(&p, enc_out, enc_len, &escr)?;
    }
    ctx.layer_norm("model.encoder.layer_norm", enc_out, enc_len)?;

    let b = NUM_BEAMS;
    let cross_k: Vec<DevicePtr> = (0..dec_layers)
        .map(|_| ctx.bf16b(b * enc_len * d))
        .collect::<Result<_>>()?;
    let cross_v: Vec<DevicePtr> = (0..dec_layers)
        .map(|_| ctx.bf16b(b * enc_len * d))
        .collect::<Result<_>>()?;
    let tmp = ctx.bf16b(enc_len * d)?;
    for l in 0..dec_layers {
        let p = format!("model.decoder.layers.{l}.encoder_attn");
        ctx.linear(&format!("{p}.k_proj"), enc_out, tmp, enc_len, d, d)?;
        for bi in 0..b {
            ctx.gpu.copy_d2d(
                tmp,
                cross_k[l].offset(bi * enc_len * d * 2),
                enc_len * d * 2,
            )?;
        }
        ctx.linear(&format!("{p}.v_proj"), enc_out, tmp, enc_len, d, d)?;
        for bi in 0..b {
            ctx.gpu.copy_d2d(
                tmp,
                cross_v[l].offset(bi * enc_len * d * 2),
                enc_len * d * 2,
            )?;
        }
    }

    let t0 = std::time::Instant::now();
    let out = ctx.beam_batched(b, &cross_k, &cross_v, FORCED_BOS, 1.0)?;
    let dt = t0.elapsed().as_secs_f64();
    println!("[nllb-metal-beambatch] beam={b} ids = {out:?}");
    anyhow::ensure!(out == EXPECTED_BEAM5, "beam-batched output diverged");
    println!(
        "[nllb-metal-beambatch] PASS - {} tok in {:.3}s = {:.1} tok/s",
        out.len(),
        dt,
        out.len() as f64 / dt
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
