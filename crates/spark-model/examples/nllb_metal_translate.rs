// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone-2 Metal PoC: full NLLB-200 / M2M-100 encoder + decoder +
//! cross-attention + greedy translation using self-contained fp32 Metal
//! kernels plus the production `GpuBackend` / `SafetensorsLoader` infra.
//!
//! Validates the greedy token sequence against HF `transformers`:
//!   input "Hello, world. How are you today?" (eng_Latn) -> fra_Latn
//!   generated ids = [256057, 17994, 141190, 248079, 25358, 123732, 248105,
//!                    30213, 248079, 1724, 25601, 385, 2]
//!
//! Run:
//!   ATLAS_TARGET_HW=metal ATLAS_TARGET_MODEL=nllb-200-3.3b ATLAS_TARGET_QUANT=bf16 \
//!     ATLAS_NLLB_DIR=/path/to/nllb-200-3.3B-st \
//!     cargo run --release -p spark-model --example nllb_metal_translate \
//!       --no-default-features --features metal

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};
use std::path::Path;

#[path = "nllb_metal_translate/ctx.rs"]
mod ctx;
use ctx::{Ctx, Kernels, Scratch, argmax_f32};

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057;
const EXPECTED_GEN: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 123732, 248105, 30213, 248079, 1724, 25601, 385, 2,
];
const MAX_NEW: usize = 64;

fn main() -> Result<()> {
    let dir =
        std::env::var("ATLAS_NLLB_DIR").unwrap_or_else(|_| "/tank/hf/nllb-200-3.3B-st".into());
    let forced_bos: u32 = std::env::var("ATLAS_NLLB_FORCED_BOS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(FORCED_BOS);

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
    let dec_heads = cfg["decoder_attention_heads"]
        .as_u64()
        .unwrap_or(heads as u64) as usize;
    let dec_ffn = cfg["decoder_ffn_dim"].as_u64().unwrap_or(ffn as u64) as usize;
    anyhow::ensure!(
        dec_heads == heads && dec_ffn == ffn,
        "asymmetric enc/dec dims (heads {heads}/{dec_heads}, ffn {ffn}/{dec_ffn}) not supported by this PoC"
    );
    let head_dim = d / heads;
    let embed_scale = if cfg["scale_embedding"].as_bool().unwrap_or(true) {
        (d as f32).sqrt()
    } else {
        1.0
    };
    let attn_scale = (head_dim as f32).powf(-0.5);
    let enc_len = INPUT_IDS.len();
    println!(
        "[nllb-metal] d={d} heads={heads} head_dim={head_dim} ffn={ffn} enc={enc_layers} dec={dec_layers} vocab={vocab}"
    );

    println!("[nllb-metal] loading weights to Metal ...");
    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;
    let k = Kernels::load(gpu)?;
    let embed_table = store.get("model.shared.weight")?.ptr;
    let ctx = Ctx {
        gpu,
        k: &k,
        d,
        heads,
        head_dim,
        ffn,
        attn_scale,
        stream,
    };

    let enc_out = ctx.f32b(enc_len * d)?;
    ctx.embed_and_positions(INPUT_IDS, embed_table, embed_scale, enc_out)?;
    let scr = Scratch::new(&ctx, MAX_NEW)?;
    for l in 0..enc_layers {
        let p = format!("model.encoder.layers.{l}");
        ctx.attn_block(
            &store,
            &p,
            "self_attn",
            enc_out,
            enc_len,
            enc_len,
            false,
            &scr,
        )?;
        ctx.ffn_block(&store, &p, enc_out, enc_len, &scr)?;
    }
    ctx.layer_norm(&store, "model.encoder.layer_norm", enc_out, enc_len)?;

    let mut cross_k = Vec::with_capacity(dec_layers);
    let mut cross_v = Vec::with_capacity(dec_layers);
    for l in 0..dec_layers {
        let p = format!("model.decoder.layers.{l}.encoder_attn");
        let ck = ctx.f32b(enc_len * d)?;
        let cv = ctx.f32b(enc_len * d)?;
        ctx.linear(&store, &format!("{p}.k_proj"), enc_out, ck, enc_len, d, d)?;
        ctx.linear(&store, &format!("{p}.v_proj"), enc_out, cv, enc_len, d, d)?;
        cross_k.push(ck);
        cross_v.push(cv);
    }

    let dh = ctx.f32b(MAX_NEW * d)?;
    let logits_dev = ctx.f32b(vocab)?;
    let mut logits_host = vec![0u8; vocab * 4];
    let mut dec: Vec<u32> = vec![dec_start];
    let mut generated: Vec<u32> = Vec::new();

    for step in 0..MAX_NEW {
        let len = dec.len();
        ctx.embed_and_positions(&dec, embed_table, embed_scale, dh)?;
        for li in 0..dec_layers {
            let p = format!("model.decoder.layers.{li}");
            ctx.attn_block(&store, &p, "self_attn", dh, len, len, true, &scr)?;
            ctx.cross_attn_block(&store, &p, dh, len, enc_len, cross_k[li], cross_v[li], &scr)?;
            ctx.ffn_block(&store, &p, dh, len, &scr)?;
        }
        ctx.layer_norm(&store, "model.decoder.layer_norm", dh, len)?;

        let last = dh.offset((len - 1) * d * 4);
        ctx.linear_no_bias_raw(last, embed_table, logits_dev, 1, vocab, d)?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(logits_dev, &mut logits_host)?;

        let next = if step == 0 {
            forced_bos
        } else {
            argmax_f32(&logits_host) as u32
        };
        generated.push(next);
        dec.push(next);
        if next == eos || dec.len() >= MAX_NEW {
            break;
        }
    }

    println!("[nllb-metal] generated ids = {generated:?}");
    println!("[nllb-metal] expected  ids = {EXPECTED_GEN:?}");
    if generated == EXPECTED_GEN {
        println!("[nllb-metal] PASS - exact token match vs HF greedy reference");
        Ok(())
    } else {
        bail!("FAIL: generated ids diverged from reference")
    }
}
