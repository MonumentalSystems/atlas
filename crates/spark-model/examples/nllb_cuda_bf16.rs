// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone-4/5 GPU PoC: NLLB-200 / M2M-100 translation on CUDA with a **bf16
//! tensor-core** pipeline. The ENCODER's batched projections/FFN/lm_head run on
//! Atlas's shared `dense_gemm_bf16_pipelined` (mma.sync + cp.async) tensor-core
//! kernel; single-token DECODE uses a right-sized `nllb_gemv_bf16` (warp-per-row
//! GEMV, fused bias) since the pipelined GEMM wastes 127/128 of its M-tile on
//! M=1. LayerNorm / attention / elementwise use bf16 variants in the
//! `nllb_encoder` module (bf16 storage, f32 accumulation). Greedy argmax runs
//! on-device (`argmax`) — no per-token 1 MB logits copy. Device KV cache + beam
//! as in milestone 3.
//!
//! Loads the bf16 checkpoint (default `/tank/hf/nllb-200-3.3B-bf16`). bf16
//! introduces small numeric drift vs the fp32 reference, so greedy is
//! hard-checked (robust) and beam is reported (score-sensitive).
//!
//! Run:
//!   ATLAS_NLLB_DIR=/tank/hf/nllb-200-3.3B-bf16 \
//!     cargo run --release -p spark-model --example nllb_cuda_bf16 --features gpu-examples

use anyhow::{Context, Result};
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};
use std::path::Path;

#[path = "nllb_cuda_bf16/ctx.rs"]
mod ctx;
#[allow(unused_imports)]
use ctx::*;
#[path = "nllb_cuda_bf16/decode.rs"]
mod decode;
#[allow(unused_imports)]
use decode::*;
#[path = "nllb_cuda_bf16/util.rs"]
mod util;
#[allow(unused_imports)]
use util::*;

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057; // fra_Latn
const PAD_ID: u32 = 1;
const MAX_NEW: usize = 96;
const CACHE_ROWS: usize = MAX_NEW + 2;
// bf16 references (from HF `transformers` run in bfloat16 — NOT the fp32
// sequence). bf16 greedy gives the cleaner "Bonjour, comment allez-vous ?";
// beam=5 is robust and matches the fp32 result.
const EXPECTED_GREEDY: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 123732, 248105, 30213, 385, 2,
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
    bias: KernelHandle,
    attn: KernelHandle,
    gemm: KernelHandle,
    gemv: KernelHandle,
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
    let enc_len = INPUT_IDS.len();
    println!(
        "[nllb-bf16] d={d} heads={heads} ffn={ffn} enc={enc_layers} dec={dec_layers} vocab={vocab}"
    );

    println!("[nllb-bf16] loading bf16 weights → GPU ...");
    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;
    anyhow::ensure!(
        store.get("model.shared.weight")?.dtype == spark_runtime::weights::WeightDtype::BF16,
        "expected a bf16 checkpoint at {dir}"
    );
    let k = Kernels {
        embed: gpu.kernel("nllb_encoder", "nllb_embed_bf16")?,
        scale: gpu.kernel("nllb_encoder", "nllb_scale_bf16")?,
        add: gpu.kernel("nllb_encoder", "nllb_add_bf16")?,
        relu: gpu.kernel("nllb_encoder", "nllb_relu_bf16")?,
        ln: gpu.kernel("nllb_encoder", "nllb_layernorm_bf16")?,
        bias: gpu.kernel("nllb_encoder", "nllb_bias_bf16")?,
        attn: gpu.kernel("nllb_encoder", "nllb_attn_kv_bf16")?,
        gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
        gemv: gpu.kernel("nllb_encoder", "nllb_gemv_bf16")?,
        argmax: gpu.kernel("argmax", "argmax_bf16")?,
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
    let enc_out = ctx.bf16b(enc_len * d)?;
    let escr = Scratch::new(&ctx, enc_len)?;
    ctx.embed_seq(INPUT_IDS, enc_out)?;
    for l in 0..enc_layers {
        let p = format!("model.encoder.layers.{l}");
        ctx.enc_self_attn(&p, enc_out, enc_len, &escr)?;
        ctx.ffn_block(&p, enc_out, enc_len, &escr)?;
    }
    ctx.layer_norm("model.encoder.layer_norm", enc_out, enc_len)?;

    // ---- cross-attention K/V per decoder layer (once) ----
    let mut cross_k = Vec::with_capacity(dec_layers);
    let mut cross_v = Vec::with_capacity(dec_layers);
    for l in 0..dec_layers {
        let p = format!("model.decoder.layers.{l}.encoder_attn");
        let ck = ctx.bf16b(enc_len * d)?;
        let cv = ctx.bf16b(enc_len * d)?;
        ctx.linear(&format!("{p}.k_proj"), enc_out, ck, enc_len, d, d)?;
        ctx.linear(&format!("{p}.v_proj"), enc_out, cv, enc_len, d, d)?;
        cross_k.push(ck);
        cross_v.push(cv);
    }

    let pos_table = ctx.bf16b(CACHE_ROWS * d)?;
    let pos_host = decoder_pos_table_bf16(CACHE_ROWS, d);
    gpu.copy_h2d(bf16_bytes(&pos_host), pos_table)?;

    let dctx = DecCtx {
        ctx: &ctx,
        cross_k: &cross_k,
        cross_v: &cross_v,
        pos_table,
    };
    let dscr = DecScratch::new(&ctx)?;

    // ---- greedy (bf16, KV cache, on-device argmax) ----
    let t0 = std::time::Instant::now();
    let greedy = dctx.greedy(&dscr, FORCED_BOS)?;
    let gdt = t0.elapsed().as_secs_f64();
    println!("[nllb-bf16] greedy ids = {greedy:?}");
    anyhow::ensure!(
        greedy == EXPECTED_GREEDY,
        "bf16 greedy diverged from reference"
    );
    println!(
        "[nllb-bf16] greedy PASS (token-exact) — {} tok in {:.3}s = {:.1} tok/s",
        greedy.len(),
        gdt,
        greedy.len() as f64 / gdt
    );

    // ---- beam search (bf16, KV cache) ----
    let t1 = std::time::Instant::now();
    let beam = dctx.beam(&dscr, FORCED_BOS, 5, 1.0)?;
    let bdt = t1.elapsed().as_secs_f64();
    println!("[nllb-bf16] beam=5 ids = {beam:?}");
    let beam_match = beam == EXPECTED_BEAM5;
    println!(
        "[nllb-bf16] beam=5 {} — {} tok in {:.3}s = {:.1} tok/s",
        if beam_match {
            "matches HF (token-exact)"
        } else {
            "differs (bf16 score drift — acceptable)"
        },
        beam.len(),
        bdt,
        beam.len() as f64 / bdt
    );

    println!("[nllb-bf16] DONE — greedy token-exact on the bf16 tensor-core path");
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
    argmax_dev: DevicePtr,
}

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

struct BeamHyps {
    num_beams: usize,
    length_penalty: f32,
    beams: Vec<(Vec<u32>, f32)>,
}
