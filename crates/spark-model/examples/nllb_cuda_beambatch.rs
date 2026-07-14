// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone-7 GPU PoC: **beam-batching** for NLLB-200 on CUDA. The `num_beams`
//! beams of a single source are decoded as a batch (B=num_beams) through one
//! bf16 tensor-core forward per step — M=B GEMM projections, per-beam device KV
//! caches, batched attention. Cross-attention K/V is shared (all beams share the
//! source) and replicated across the B slots. Beam selection reorders the
//! per-beam caches via `nllb_gather_batched` (HF `_reorder_cache`); the
//! BeamHypotheses bookkeeping is host-side (faithful to `BeamSearchScorer`).
//!
//! Validates token-exact vs HF-bf16 beam=5 and reports tok/s vs single-stream.
//!
//! Run:
//!   ATLAS_NLLB_DIR=/tank/hf/nllb-200-3.3B-bf16 \
//!     cargo run --release -p spark-model --example nllb_cuda_beambatch --features gpu-examples

use anyhow::{Context, Result};
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};
use std::path::Path;

#[path = "nllb_cuda_beambatch/ctx.rs"]
mod ctx;
#[allow(unused_imports)]
use ctx::*;
#[path = "nllb_cuda_beambatch/decode.rs"]
mod decode;
#[allow(unused_imports)]
use decode::*;

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057;
const PAD_ID: u32 = 1;
const MAX_NEW: usize = 96;
const NUM_BEAMS: usize = 5;
const EXPECTED_BEAM5: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 4255, 956, 34821, 248105, 30213, 102506, 248116, 15510,
    385, 2,
];

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
    gather: KernelHandle,
    gemm: KernelHandle,
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
        gather: gpu.kernel("nllb_encoder", "nllb_gather_batched")?,
        gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
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
        enc_len,
        attn_scale,
        embed_scale,
        embed_table: store.get("model.shared.weight")?.ptr,
        dec_start,
        eos,
        stream,
    };

    // ---- encoder (once) → cross K/V, replicated to B beam slots ----
    let b = NUM_BEAMS;
    let enc_out = c.bf16b(enc_len * d)?;
    let escr = Scratch::new(&c, enc_len)?;
    c.embed_seq(INPUT_IDS, enc_out)?;
    for l in 0..enc_layers {
        let p = format!("model.encoder.layers.{l}");
        c.enc_self_attn(&p, enc_out, enc_len, &escr)?;
        c.ffn_block(&p, enc_out, enc_len, &escr)?;
    }
    c.layer_norm("model.encoder.layer_norm", enc_out, enc_len)?;
    let cross_k: Vec<DevicePtr> = (0..dec_layers)
        .map(|_| c.bf16b(b * enc_len * d))
        .collect::<Result<_>>()?;
    let cross_v: Vec<DevicePtr> = (0..dec_layers)
        .map(|_| c.bf16b(b * enc_len * d))
        .collect::<Result<_>>()?;
    let tmp = c.bf16b(enc_len * d)?;
    for l in 0..dec_layers {
        let p = format!("model.decoder.layers.{l}.encoder_attn");
        c.linear(&format!("{p}.k_proj"), enc_out, tmp, enc_len, d, d)?;
        for bi in 0..b {
            c.gpu.copy_d2d(
                tmp,
                cross_k[l].offset(bi * enc_len * d * 2),
                enc_len * d * 2,
            )?;
        }
        c.linear(&format!("{p}.v_proj"), enc_out, tmp, enc_len, d, d)?;
        for bi in 0..b {
            c.gpu.copy_d2d(
                tmp,
                cross_v[l].offset(bi * enc_len * d * 2),
                enc_len * d * 2,
            )?;
        }
    }

    let t0 = std::time::Instant::now();
    let out = c.beam_batched(b, &cross_k, &cross_v, FORCED_BOS, 1.0)?;
    let dt = t0.elapsed().as_secs_f64();
    println!("[nllb-beambatch] beam={b} ids = {out:?}");
    let pass = out == EXPECTED_BEAM5;
    println!(
        "[nllb-beambatch] beam={b} {} — {} tok in {:.3}s = {:.1} tok/s",
        if pass {
            "PASS (token-exact vs HF-bf16)"
        } else {
            "differs"
        },
        out.len(),
        dt,
        out.len() as f64 / dt
    );
    anyhow::ensure!(pass, "beam-batched output diverged from HF-bf16 reference");
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
    enc_len: usize,
    attn_scale: f32,
    embed_scale: f32,
    embed_table: DevicePtr,
    dec_start: u32,
    eos: u32,
    stream: u64,
}

struct DecBuf {
    dh: DevicePtr,
    residual: DevicePtr,
    normed: DevicePtr,
    q: DevicePtr,
    knew: DevicePtr,
    vnew: DevicePtr,
    attn: DevicePtr,
    proj: DevicePtr,
    ff: DevicePtr,
    logits: DevicePtr,
    id: DevicePtr,
    selftk: DevicePtr,
    crosstk: DevicePtr,
    pos_table: DevicePtr,
}

struct Beam {
    tokens: Vec<u32>,
    score: f32,
    logits: Vec<f32>,
}
struct BeamHyps {
    num_beams: usize,
    lp: f32,
    beams: Vec<(Vec<u32>, f32)>,
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
