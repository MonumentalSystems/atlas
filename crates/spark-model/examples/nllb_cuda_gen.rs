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

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057; // fra_Latn
const PAD_ID: u32 = 1;
const MAX_NEW: usize = 96;
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
    let pos_table = ctx.f32b(MAX_NEW * d)?;
    let pos_host = decoder_pos_table(MAX_NEW, d);
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

impl Ctx<'_> {
    fn f32b(&self, elems: usize) -> Result<DevicePtr> {
        self.gpu.alloc(elems * 4)
    }
    fn w(&self, name: &str) -> Result<DevicePtr> {
        Ok(self.store.get(name)?.ptr)
    }

    fn layer_norm(&self, prefix: &str, x: DevicePtr, rows: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.ln)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .shared_mem(256 * 4)
            .arg_ptr(x)
            .arg_ptr(self.w(&format!("{prefix}.weight"))?)
            .arg_ptr(self.w(&format!("{prefix}.bias"))?)
            .arg_u32(rows as u32)
            .arg_u32(self.d as u32)
            .arg_f32(1e-5)
            .launch(self.stream)
    }

    /// C[rows,n_out] = A[rows,k_in] @ W[n_out,k_in]^T + bias (weight `{prefix}.{weight,bias}`).
    fn linear(
        &self,
        prefix: &str,
        a: DevicePtr,
        c: DevicePtr,
        rows: usize,
        n_out: usize,
        k_in: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.lin)
            .grid([div_ceil(n_out as u32, 16), div_ceil(rows as u32, 16), 1])
            .block([16, 16, 1])
            .arg_ptr(a)
            .arg_ptr(self.w(&format!("{prefix}.weight"))?)
            .arg_ptr(self.w(&format!("{prefix}.bias"))?)
            .arg_ptr(c)
            .arg_u32(rows as u32)
            .arg_u32(n_out as u32)
            .arg_u32(k_in as u32)
            .launch(self.stream)
    }

    /// C[rows,n_out] = A @ table^T (tied lm_head, no bias).
    fn lm_head(&self, a: DevicePtr, c: DevicePtr, rows: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.lin)
            .grid([
                div_ceil(self.vocab as u32, 16),
                div_ceil(rows as u32, 16),
                1,
            ])
            .block([16, 16, 1])
            .arg_ptr(a)
            .arg_ptr(self.embed_table)
            .arg_ptr(DevicePtr(0))
            .arg_ptr(c)
            .arg_u32(rows as u32)
            .arg_u32(self.vocab as u32)
            .arg_u32(self.d as u32)
            .launch(self.stream)
    }

    fn attention(
        &self,
        q: DevicePtr,
        kk: DevicePtr,
        v: DevicePtr,
        out: DevicePtr,
        tq: usize,
        tk: usize,
        causal: bool,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.attn)
            .grid([(tq * self.heads) as u32, 1, 1])
            .block([self.head_dim as u32, 1, 1])
            .shared_mem(((tk + self.head_dim) * 4) as u32)
            .arg_ptr(q)
            .arg_ptr(kk)
            .arg_ptr(v)
            .arg_ptr(out)
            .arg_u32(tq as u32)
            .arg_u32(tk as u32)
            .arg_u32(self.heads as u32)
            .arg_u32(self.head_dim as u32)
            .arg_f32(self.attn_scale)
            .arg_u32(causal as u32)
            .launch(self.stream)
    }

    fn add(&self, dst: DevicePtr, src: DevicePtr, n: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.add)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_u32(n as u32)
            .launch(self.stream)
    }

    /// Embed a whole token sequence (encoder) → out[seq, d], scaled + positions.
    fn embed_seq(&self, ids: &[u32], out: DevicePtr) -> Result<()> {
        let (d, seq) = (self.d, ids.len());
        let ids_dev = self.gpu.alloc(seq * 4)?;
        self.gpu.copy_h2d(u32_bytes(ids), ids_dev)?;
        KernelLaunch::new(self.gpu, self.k.embed)
            .grid([seq as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ids_dev)
            .arg_ptr(self.embed_table)
            .arg_ptr(out)
            .arg_u32(d as u32)
            .launch(self.stream)?;
        KernelLaunch::new(self.gpu, self.k.scale)
            .grid([div_ceil((seq * d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(out)
            .arg_u32((seq * d) as u32)
            .arg_f32(self.embed_scale)
            .launch(self.stream)?;
        let pos = sinusoid_positions(ids, d, PAD_ID);
        let pos_dev = self.f32b(seq * d)?;
        self.gpu.copy_h2d(f32_bytes(&pos), pos_dev)?;
        self.add(out, pos_dev, seq * d)?;
        self.gpu.free(ids_dev)?;
        self.gpu.free(pos_dev)?;
        Ok(())
    }

    /// Encoder self-attention block (pre-norm, bidirectional) over `x` in place.
    fn enc_self_attn(&self, layer: &str, x: DevicePtr, seq: usize, s: &Scratch) -> Result<()> {
        let (d, bytes) = (self.d, seq * self.d * 4);
        let p = format!("{layer}.self_attn");
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(&format!("{layer}.self_attn_layer_norm"), s.normed, seq)?;
        self.linear(&format!("{p}.q_proj"), s.normed, s.q, seq, d, d)?;
        self.linear(&format!("{p}.k_proj"), s.normed, s.kk, seq, d, d)?;
        self.linear(&format!("{p}.v_proj"), s.normed, s.v, seq, d, d)?;
        self.attention(s.q, s.kk, s.v, s.attn, seq, seq, false)?;
        self.linear(&format!("{p}.out_proj"), s.attn, s.proj, seq, d, d)?;
        self.add(s.proj, s.residual, seq * d)?;
        self.gpu.copy_d2d(s.proj, x, bytes)
    }

    /// FFN block (pre-norm, ReLU) over `x[rows,d]` in place.
    fn ffn_block(&self, layer: &str, x: DevicePtr, rows: usize, s: &Scratch) -> Result<()> {
        let (d, ffn, bytes) = (self.d, self.ffn, rows * self.d * 4);
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(&format!("{layer}.final_layer_norm"), s.normed, rows)?;
        self.linear(&format!("{layer}.fc1"), s.normed, s.ff, rows, ffn, d)?;
        KernelLaunch::new(self.gpu, self.k.relu)
            .grid([div_ceil((rows * ffn) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.ff)
            .arg_u32((rows * ffn) as u32)
            .launch(self.stream)?;
        self.linear(&format!("{layer}.fc2"), s.ff, s.proj, rows, d, ffn)?;
        self.add(s.proj, s.residual, rows * d)?;
        self.gpu.copy_d2d(s.proj, x, bytes)
    }
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
impl Scratch {
    fn new(c: &Ctx, rows: usize) -> Result<Self> {
        let d = c.d;
        Ok(Self {
            residual: c.f32b(rows * d)?,
            normed: c.f32b(rows * d)?,
            q: c.f32b(rows * d)?,
            kk: c.f32b(rows * d)?,
            v: c.f32b(rows * d)?,
            attn: c.f32b(rows * d)?,
            proj: c.f32b(rows * d)?,
            ff: c.f32b(rows * c.ffn)?,
        })
    }
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
impl DecScratch {
    fn new(c: &Ctx) -> Result<Self> {
        let d = c.d;
        Ok(Self {
            dh: c.f32b(d)?,
            residual: c.f32b(d)?,
            normed: c.f32b(d)?,
            q: c.f32b(d)?,
            attn: c.f32b(d)?,
            proj: c.f32b(d)?,
            ff: c.f32b(c.ffn)?,
            logits: c.f32b(c.vocab)?,
            id_dev: c.gpu.alloc(4)?,
        })
    }
}

/// Per-beam device KV cache: 2×dec_layers buffers of [MAX_NEW, d].
struct KvCache {
    k: Vec<DevicePtr>,
    v: Vec<DevicePtr>,
}

impl<'a> DecCtx<'a> {
    fn c(&self) -> &Ctx<'a> {
        self.ctx
    }

    fn alloc_cache(&self) -> Result<KvCache> {
        let c = self.c();
        let mut k = Vec::with_capacity(c.dec_layers);
        let mut v = Vec::with_capacity(c.dec_layers);
        for _ in 0..c.dec_layers {
            k.push(c.f32b(MAX_NEW * c.d)?);
            v.push(c.f32b(MAX_NEW * c.d)?);
        }
        Ok(KvCache { k, v })
    }

    /// Clone the first `used` rows of every layer's K/V into a fresh cache.
    fn clone_cache(&self, src: &KvCache, used: usize) -> Result<KvCache> {
        let c = self.c();
        let dst = self.alloc_cache()?;
        let bytes = used * c.d * 4;
        for l in 0..c.dec_layers {
            c.gpu.copy_d2d(src.k[l], dst.k[l], bytes)?;
            c.gpu.copy_d2d(src.v[l], dst.v[l], bytes)?;
        }
        Ok(dst)
    }

    fn free_cache(&self, cache: KvCache) -> Result<()> {
        let c = self.c();
        for l in 0..c.dec_layers {
            c.gpu.free(cache.k[l])?;
            c.gpu.free(cache.v[l])?;
        }
        Ok(())
    }

    /// Forward ONE decoder token at sequence index `pos` (0-based), writing its
    /// self-attn K/V into `cache` row `pos` and attending over rows `0..=pos`.
    /// Returns host logits [vocab] predicting the next token.
    fn forward_one(
        &self,
        tok: u32,
        pos: usize,
        cache: &KvCache,
        s: &DecScratch,
        out_logits: &mut [u8],
    ) -> Result<()> {
        let c = self.c();
        let d = c.d;
        // embed token + scale + position sinusoid (row `pos` of pos_table)
        c.gpu.copy_h2d(u32_bytes(&[tok]), s.id_dev)?;
        KernelLaunch::new(c.gpu, c.k.embed)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.id_dev)
            .arg_ptr(c.embed_table)
            .arg_ptr(s.dh)
            .arg_u32(d as u32)
            .launch(c.stream)?;
        KernelLaunch::new(c.gpu, c.k.scale)
            .grid([div_ceil(d as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.dh)
            .arg_u32(d as u32)
            .arg_f32(c.embed_scale)
            .launch(c.stream)?;
        c.add(s.dh, self.pos_table.offset(pos * d * 4), d)?;

        let off = pos * d * 4;
        let tk = pos + 1;
        for l in 0..c.dec_layers {
            let p = format!("model.decoder.layers.{l}");
            // self-attention (causal via cache slice 0..=pos)
            c.gpu.copy_d2d(s.dh, s.residual, d * 4)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 4)?;
            c.layer_norm(&format!("{p}.self_attn_layer_norm"), s.normed, 1)?;
            c.linear(&format!("{p}.self_attn.q_proj"), s.normed, s.q, 1, d, d)?;
            // write this token's K/V directly into the cache row `pos`
            c.linear(
                &format!("{p}.self_attn.k_proj"),
                s.normed,
                cache.k[l].offset(off),
                1,
                d,
                d,
            )?;
            c.linear(
                &format!("{p}.self_attn.v_proj"),
                s.normed,
                cache.v[l].offset(off),
                1,
                d,
                d,
            )?;
            c.attention(s.q, cache.k[l], cache.v[l], s.attn, 1, tk, false)?;
            c.linear(&format!("{p}.self_attn.out_proj"), s.attn, s.proj, 1, d, d)?;
            c.add(s.proj, s.residual, d)?;
            c.gpu.copy_d2d(s.proj, s.dh, d * 4)?;
            // cross-attention over encoder K/V
            c.gpu.copy_d2d(s.dh, s.residual, d * 4)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 4)?;
            c.layer_norm(&format!("{p}.encoder_attn_layer_norm"), s.normed, 1)?;
            c.linear(&format!("{p}.encoder_attn.q_proj"), s.normed, s.q, 1, d, d)?;
            c.attention(
                s.q,
                self.cross_k[l],
                self.cross_v[l],
                s.attn,
                1,
                c.enc_len,
                false,
            )?;
            c.linear(
                &format!("{p}.encoder_attn.out_proj"),
                s.attn,
                s.proj,
                1,
                d,
                d,
            )?;
            c.add(s.proj, s.residual, d)?;
            c.gpu.copy_d2d(s.proj, s.dh, d * 4)?;
            // FFN
            c.gpu.copy_d2d(s.dh, s.residual, d * 4)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 4)?;
            c.layer_norm(&format!("{p}.final_layer_norm"), s.normed, 1)?;
            c.linear(&format!("{p}.fc1"), s.normed, s.ff, 1, c.ffn, d)?;
            KernelLaunch::new(c.gpu, c.k.relu)
                .grid([div_ceil(c.ffn as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(s.ff)
                .arg_u32(c.ffn as u32)
                .launch(c.stream)?;
            c.linear(&format!("{p}.fc2"), s.ff, s.proj, 1, d, c.ffn)?;
            c.add(s.proj, s.residual, d)?;
            c.gpu.copy_d2d(s.proj, s.dh, d * 4)?;
        }
        c.layer_norm("model.decoder.layer_norm", s.dh, 1)?;
        c.lm_head(s.dh, s.logits, 1)?;
        c.gpu.synchronize(c.stream)?;
        c.gpu.copy_d2h(s.logits, out_logits)?;
        Ok(())
    }

    /// Greedy decode with a single KV cache.
    fn greedy(&self, s: &DecScratch, forced_bos: u32) -> Result<Vec<u32>> {
        let c = self.c();
        let cache = self.alloc_cache()?;
        let mut logits = vec![0u8; c.vocab * 4];
        let mut dec = vec![c.dec_start];
        let mut generated = Vec::new();
        for step in 0..MAX_NEW {
            self.forward_one(dec[step], step, &cache, s, &mut logits)?;
            let next = if step == 0 {
                forced_bos
            } else {
                argmax_f32(&logits) as u32
            };
            generated.push(next);
            dec.push(next);
            if next == c.eos {
                break;
            }
        }
        self.free_cache(cache)?;
        Ok(generated)
    }

    /// Beam search with per-beam KV caches (faithful to HF BeamSearchScorer).
    fn beam(
        &self,
        s: &DecScratch,
        forced_bos: u32,
        num_beams: usize,
        length_penalty: f32,
    ) -> Result<Vec<u32>> {
        let c = self.c();
        let mut logits = vec![0u8; c.vocab * 4];

        // Init: build the [start, forced_bos] cache once, clone to all beams.
        let base = self.alloc_cache()?;
        self.forward_one(c.dec_start, 0, &base, s, &mut logits)?; // fills row 0 (logits discarded)
        self.forward_one(forced_bos, 1, &base, s, &mut logits)?; // fills row 1, logits predict next
        let mut beams: Vec<Beam> = Vec::with_capacity(num_beams);
        for b in 0..num_beams {
            let cache = if b == 0 {
                None
            } else {
                Some(self.clone_cache(&base, 2)?)
            };
            beams.push(Beam {
                tokens: vec![c.dec_start, forced_bos],
                score: if b == 0 { 0.0 } else { f32::NEG_INFINITY },
                cache: cache.unwrap_or_else(|| KvCache {
                    k: vec![],
                    v: vec![],
                }),
                logits: f32_vec(&logits),
            });
        }
        beams[0].cache = base; // beam 0 owns the base cache

        let mut hyps = BeamHyps::new(num_beams, length_penalty);
        for _ in 1..MAX_NEW {
            let cur_len = beams[0].tokens.len();
            // expand: gather top-2*num_beams (score, beam, token) candidates
            let mut cands: Vec<(f32, usize, u32)> = Vec::new();
            for (b, beam) in beams.iter().enumerate() {
                if !beam.score.is_finite() {
                    continue;
                }
                let lse = logsumexp(&beam.logits);
                for (val, tok) in top_k(&beam.logits, 2 * num_beams) {
                    cands.push((beam.score + (val - lse), b, tok as u32));
                }
            }
            cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // select num_beams continuing beams; EOS candidates finalize a hyp
            let mut next: Vec<Beam> = Vec::with_capacity(num_beams);
            for (score, b, tok) in cands {
                if next.len() == num_beams {
                    break;
                }
                if tok == c.eos {
                    hyps.add(beams[b].tokens.clone(), score);
                } else {
                    let cache = self.clone_cache(&beams[b].cache, cur_len)?;
                    self.forward_one(tok, cur_len, &cache, s, &mut logits)?;
                    let mut tokens = beams[b].tokens.clone();
                    tokens.push(tok);
                    next.push(Beam {
                        tokens,
                        score,
                        cache,
                        logits: f32_vec(&logits),
                    });
                }
            }
            let best_running = next[0].score;
            for beam in beams.drain(..) {
                if !beam.cache.k.is_empty() {
                    self.free_cache(beam.cache)?;
                }
            }
            beams = next;
            if hyps.is_done(best_running, cur_len) {
                break;
            }
        }

        // finalize: fold surviving beams into the pool, take the best
        for beam in beams.drain(..) {
            if beam.score.is_finite() {
                hyps.add(beam.tokens.clone(), beam.score);
            }
            if !beam.cache.k.is_empty() {
                self.free_cache(beam.cache)?;
            }
        }
        let mut best = hyps.best().context("no finished hypotheses")?;
        best.push(c.eos);
        Ok(best[1..].to_vec())
    }
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
impl BeamHyps {
    fn new(num_beams: usize, length_penalty: f32) -> Self {
        Self {
            num_beams,
            length_penalty,
            beams: Vec::new(),
        }
    }
    fn worst(&self) -> f32 {
        self.beams
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::INFINITY, f32::min)
    }
    fn add(&mut self, tokens: Vec<u32>, sum_logprob: f32) {
        let score = sum_logprob / (tokens.len() as f32).powf(self.length_penalty);
        if self.beams.len() < self.num_beams || score > self.worst() {
            self.beams.push((tokens, score));
            if self.beams.len() > self.num_beams {
                let (wi, _) = self
                    .beams
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
                    .unwrap();
                self.beams.swap_remove(wi);
            }
        }
    }
    fn is_done(&self, best_running_sum_logprob: f32, cur_len: usize) -> bool {
        if self.beams.len() < self.num_beams {
            return false;
        }
        // early_stopping = false: stop once the worst kept hyp beats the best
        // score any running beam could still reach at this length.
        let highest = best_running_sum_logprob / (cur_len as f32).powf(self.length_penalty);
        self.worst() >= highest
    }
    fn best(&self) -> Option<Vec<u32>> {
        self.beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(t, _)| t.clone())
    }
}

// ───────────────────────── host helpers ─────────────────────────

fn logsumexp(x: &[f32]) -> f32 {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m + x.iter().map(|&v| (v - m).exp()).sum::<f32>().ln()
}

/// The `k` largest (value, index), value-descending (index asc on ties).
fn top_k(x: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, &v) in x.iter().enumerate() {
        if best.len() < k {
            best.push((v, i));
            if best.len() == k {
                best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }
        } else if v > best[k - 1].0 {
            best[k - 1] = (v, i);
            let mut j = k - 1;
            while j > 0 && best[j].0 > best[j - 1].0 {
                best.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    best
}

fn argmax_f32(bytes: &[u8]) -> usize {
    let logits = f32_slice(bytes);
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Decoder position sinusoid table: row `i` = sinusoid for position id `i+2`.
fn decoder_pos_table(max_len: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let emb_scale = 10000f32.ln() / (half as f32 - 1.0);
    let mut t = vec![0f32; max_len * d];
    for i in 0..max_len {
        let p = (i + 2) as f32; // padding_idx=1, offset so index 0 → posid 2
        for j in 0..half {
            let ang = p * (-(j as f32) * emb_scale).exp();
            t[i * d + j] = ang.sin();
            t[i * d + half + j] = ang.cos();
        }
    }
    t
}

fn sinusoid_positions(ids: &[u32], d: usize, pad: u32) -> Vec<f32> {
    let (seq, half) = (ids.len(), d / 2);
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
            let ang = p as f32 * (-(j as f32) * emb_scale).exp();
            pos[i * d + j] = ang.sin();
            pos[i * d + half + j] = ang.cos();
        }
    }
    pos
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
fn f32_vec(b: &[u8]) -> Vec<f32> {
    f32_slice(b).to_vec()
}
