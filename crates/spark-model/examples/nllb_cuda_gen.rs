// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone-3 GPU PoC: NLLB-200 / M2M-100 translation on CUDA with a device
//! **KV cache** (single-token decode; O(L) instead of O(L²)) and **beam
//! search** (per-beam caches + clone-on-branch, faithful to HF
//! `BeamSearchScorer`). fp32 throughout, reuses the self-contained
//! `nllb_encoder` kernel module — the cache/beam are pure orchestration, no
//! new kernels.
//!
//! Validates BOTH decode modes against HF `transformers` (eng_Latn "Hello,
//! world. How are you today?" → fra_Latn):
//!   greedy  → [256057,17994,141190,248079,25358,123732,248105,30213,248079,1724,25601,385,2]
//!   beam=5  → [256057,17994,141190,248079,25358,4255,956,34821,248105,30213,102506,248116,15510,385,2]
//!
//! Run:
//!   ATLAS_NLLB_DIR=/tank/hf/nllb-200-3.3B-st \
//!     cargo run --release -p spark-model --example nllb_cuda_gen --features gpu-examples

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};
use std::path::Path;

#[path = "nllb_cuda_gen/ctx.rs"]
mod ctx;
#[allow(unused_imports)]
use ctx::*;
#[path = "nllb_cuda_gen/decode.rs"]
mod decode;
#[allow(unused_imports)]
use decode::*;

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057; // fra_Latn
const PAD_ID: u32 = 1;
const MAX_NEW: usize = 96;
// Decoder cache / position-table rows: MAX_NEW generated tokens + the 2 seed
// tokens (decoder_start, forced_bos) the beam loop writes before iterating.
const CACHE_ROWS: usize = MAX_NEW + 2;
const EXPECTED_GREEDY: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 123732, 248105, 30213, 248079, 1724, 25601, 385, 2,
];
const EXPECTED_BEAM5: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 4255, 956, 34821, 248105, 30213, 102506, 248116, 15510,
    385, 2,
];

struct Kernels {
    embed: KernelHandle,
    scale: KernelHandle,
    add: KernelHandle,
    relu: KernelHandle,
    ln: KernelHandle,
    lin: KernelHandle,
    attn: KernelHandle,
}

fn main() -> Result<()> {
    let dir =
        std::env::var("ATLAS_NLLB_DIR").unwrap_or_else(|_| "/tank/hf/nllb-200-3.3B-st".into());

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
    let enc_len = INPUT_IDS.len();
    println!(
        "[nllb] d={d} heads={heads} head_dim={head_dim} ffn={ffn} enc={enc_layers} dec={dec_layers} vocab={vocab}"
    );

    println!("[nllb] loading weights → GPU ...");
    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;
    let k = Kernels {
        embed: gpu.kernel("nllb_encoder", "nllb_embed")?,
        scale: gpu.kernel("nllb_encoder", "nllb_scale_inplace")?,
        add: gpu.kernel("nllb_encoder", "nllb_add_inplace")?,
        relu: gpu.kernel("nllb_encoder", "nllb_relu_inplace")?,
        ln: gpu.kernel("nllb_encoder", "nllb_layernorm")?,
        lin: gpu.kernel("nllb_encoder", "nllb_linear")?,
        attn: gpu.kernel("nllb_encoder", "nllb_attn_kv")?,
    };
    let embed_table = store.get("model.shared.weight")?.ptr;

    let ctx = Ctx {
        gpu,
        k: &k,
        store: &store,
        d,
        heads,
        head_dim,
        ffn,
        vocab,
        dec_layers,
        enc_len,
        attn_scale,
        embed_scale,
        embed_table,
        dec_start,
        eos,
        stream,
    };

    // ---- encoder (once) ----
    let enc_out = ctx.f32b(enc_len * d)?;
    let escr = Scratch::new(&ctx, enc_len)?;
    ctx.embed_seq(INPUT_IDS, enc_out)?;
    for l in 0..enc_layers {
        let p = format!("model.encoder.layers.{l}");
        ctx.enc_self_attn(&p, enc_out, enc_len, &escr)?;
        ctx.ffn_block(&p, enc_out, enc_len, &escr)?;
    }
    ctx.layer_norm("model.encoder.layer_norm", enc_out, enc_len)?;

    // ---- precompute cross-attention K/V per decoder layer (once) ----
    let mut cross_k = Vec::with_capacity(dec_layers);
    let mut cross_v = Vec::with_capacity(dec_layers);
    for l in 0..dec_layers {
        let p = format!("model.decoder.layers.{l}.encoder_attn");
        let ck = ctx.f32b(enc_len * d)?;
        let cv = ctx.f32b(enc_len * d)?;
        ctx.linear(&format!("{p}.k_proj"), enc_out, ck, enc_len, d, d)?;
        ctx.linear(&format!("{p}.v_proj"), enc_out, cv, enc_len, d, d)?;
        cross_k.push(ck);
        cross_v.push(cv);
    }

    // Precomputed decoder position sinusoids (posid = index + 2), rows 0..MAX_NEW.
    let pos_table = ctx.f32b(CACHE_ROWS * d)?;
    let pos_host = decoder_pos_table(CACHE_ROWS, d);
    gpu.copy_h2d(f32_bytes(&pos_host), pos_table)?;

    let dctx = DecCtx {
        ctx: &ctx,
        cross_k: &cross_k,
        cross_v: &cross_v,
        pos_table,
    };
    let dscr = DecScratch::new(&ctx)?;

    // ---- greedy (KV cache) ----
    let t0 = std::time::Instant::now();
    let greedy = dctx.greedy(&dscr, FORCED_BOS)?;
    let gdt = t0.elapsed().as_secs_f64();
    println!("[nllb] greedy ids   = {greedy:?}");
    anyhow::ensure!(
        greedy == EXPECTED_GREEDY,
        "greedy (KV cache) diverged from reference"
    );
    println!(
        "[nllb] greedy PASS (KV cache, token-exact) — {} tok in {:.3}s = {:.1} tok/s",
        greedy.len(),
        gdt,
        greedy.len() as f64 / gdt
    );

    // ---- beam search (KV cache, num_beams=5) ----
    let t1 = std::time::Instant::now();
    let beam = dctx.beam(&dscr, FORCED_BOS, 5, 1.0)?;
    let bdt = t1.elapsed().as_secs_f64();
    println!("[nllb] beam=5 ids   = {beam:?}");
    anyhow::ensure!(
        beam == EXPECTED_BEAM5,
        "beam=5 (KV cache) diverged from reference"
    );
    println!(
        "[nllb] beam=5 PASS (KV cache, token-exact) — {} tok in {:.3}s = {:.1} tok/s",
        beam.len(),
        bdt,
        beam.len() as f64 / bdt
    );

    println!("[nllb] ALL PASS — KV-cache greedy + beam both token-exact vs HF");
    Ok(())
}

// ───────────────────────── core context ─────────────────────────

struct Ctx<'a> {
    gpu: &'a dyn GpuBackend,
    k: &'a Kernels,
    store: &'a WeightStore,
    d: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    vocab: usize,
    dec_layers: usize,
    enc_len: usize,
    attn_scale: f32,
    embed_scale: f32,
    embed_table: DevicePtr,
    dec_start: u32,
    eos: u32,
    stream: u64,
}

/// Scratch for whole-sequence (encoder) forwards.
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

// ───────────────────────── cached decoder ─────────────────────────

struct DecCtx<'a> {
    ctx: &'a Ctx<'a>,
    cross_k: &'a [DevicePtr],
    cross_v: &'a [DevicePtr],
    pos_table: DevicePtr,
}

/// Single-token decoder scratch (1 row of d / ffn / vocab).
struct DecScratch {
    dh: DevicePtr,
    residual: DevicePtr,
    normed: DevicePtr,
    q: DevicePtr,
    attn: DevicePtr,
    proj: DevicePtr,
    ff: DevicePtr,
    logits: DevicePtr,
    id_dev: DevicePtr,
}

/// Per-beam device KV cache: 2×dec_layers buffers of [MAX_NEW, d].
struct KvCache {
    k: Vec<DevicePtr>,
    v: Vec<DevicePtr>,
}

struct Beam {
    tokens: Vec<u32>,
    score: f32,
    cache: KvCache,
    logits: Vec<f32>,
}

/// Bounded pool of the best `num_beams` finished hypotheses (HF BeamHypotheses).
struct BeamHyps {
    num_beams: usize,
    length_penalty: f32,
    beams: Vec<(Vec<u32>, f32)>, // (tokens incl. decoder-start, normalized score)
}
