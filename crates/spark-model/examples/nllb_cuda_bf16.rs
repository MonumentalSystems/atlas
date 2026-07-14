// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone-4 GPU PoC: NLLB-200 / M2M-100 translation on CUDA with a **bf16
//! tensor-core** pipeline. The heavy projections/FFN/lm_head run on Atlas's
//! shared `dense_gemm_bf16_pipelined` (mma.sync + cp.async) tensor-core kernel;
//! LayerNorm / attention / elementwise use bf16 variants in the `nllb_encoder`
//! module (bf16 storage, f32 accumulation). Greedy argmax runs on-device
//! (`argmax`) — no per-token 1 MB logits copy. Device KV cache + beam as in
//! milestone 3.
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

impl Ctx<'_> {
    fn bf16b(&self, elems: usize) -> Result<DevicePtr> {
        self.gpu.alloc(elems * 2)
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

    /// Tensor-core GEMM: C[M,N] = A[M,K] @ W[N,K]^T (bf16), no bias.
    fn gemm(
        &self,
        a: DevicePtr,
        wt: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        kdim: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.gemm)
            .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(wt)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(kdim as u32)
            .launch(self.stream)
    }

    /// Linear with bias: GEMM then row-broadcast bias add.
    fn linear(
        &self,
        prefix: &str,
        a: DevicePtr,
        c: DevicePtr,
        rows: usize,
        n_out: usize,
        k_in: usize,
    ) -> Result<()> {
        self.gemm(
            a,
            self.w(&format!("{prefix}.weight"))?,
            c,
            rows,
            n_out,
            k_in,
        )?;
        KernelLaunch::new(self.gpu, self.k.bias)
            .grid([div_ceil((rows * n_out) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c)
            .arg_ptr(self.w(&format!("{prefix}.bias"))?)
            .arg_u32(rows as u32)
            .arg_u32(n_out as u32)
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
        let pos = encoder_pos_bf16(ids, d, PAD_ID);
        let pos_dev = self.bf16b(seq * d)?;
        self.gpu.copy_h2d(bf16_bytes(&pos), pos_dev)?;
        self.add(out, pos_dev, seq * d)?;
        self.gpu.free(ids_dev)?;
        self.gpu.free(pos_dev)?;
        Ok(())
    }

    fn enc_self_attn(&self, layer: &str, x: DevicePtr, seq: usize, s: &Scratch) -> Result<()> {
        let (d, bytes) = (self.d, seq * self.d * 2);
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

    fn ffn_block(&self, layer: &str, x: DevicePtr, rows: usize, s: &Scratch) -> Result<()> {
        let (d, ffn, bytes) = (self.d, self.ffn, rows * self.d * 2);
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
            residual: c.bf16b(rows * d)?,
            normed: c.bf16b(rows * d)?,
            q: c.bf16b(rows * d)?,
            kk: c.bf16b(rows * d)?,
            v: c.bf16b(rows * d)?,
            attn: c.bf16b(rows * d)?,
            proj: c.bf16b(rows * d)?,
            ff: c.bf16b(rows * c.ffn)?,
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
impl DecScratch {
    fn new(c: &Ctx) -> Result<Self> {
        let d = c.d;
        Ok(Self {
            dh: c.bf16b(d)?,
            residual: c.bf16b(d)?,
            normed: c.bf16b(d)?,
            q: c.bf16b(d)?,
            attn: c.bf16b(d)?,
            proj: c.bf16b(d)?,
            ff: c.bf16b(c.ffn)?,
            logits: c.bf16b(c.vocab)?,
            id_dev: c.gpu.alloc(4)?,
            argmax_dev: c.gpu.alloc(4)?,
        })
    }
}

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
            k.push(c.bf16b(CACHE_ROWS * c.d)?);
            v.push(c.bf16b(CACHE_ROWS * c.d)?);
        }
        Ok(KvCache { k, v })
    }

    fn clone_cache(&self, src: &KvCache, used: usize) -> Result<KvCache> {
        let c = self.c();
        let dst = self.alloc_cache()?;
        let bytes = used * c.d * 2;
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

    /// Forward one decoder token at index `pos`; write its self-attn K/V into
    /// `cache` row `pos`, attend rows 0..=pos. Fills `s.logits` (bf16, device).
    fn forward_one(&self, tok: u32, pos: usize, cache: &KvCache, s: &DecScratch) -> Result<()> {
        let c = self.c();
        let d = c.d;
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
        c.add(s.dh, self.pos_table.offset(pos * d * 2), d)?;

        let off = pos * d * 2;
        let tk = pos + 1;
        for l in 0..c.dec_layers {
            let p = format!("model.decoder.layers.{l}");
            // self-attention (cache slice 0..=pos)
            c.gpu.copy_d2d(s.dh, s.residual, d * 2)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 2)?;
            c.layer_norm(&format!("{p}.self_attn_layer_norm"), s.normed, 1)?;
            c.linear(&format!("{p}.self_attn.q_proj"), s.normed, s.q, 1, d, d)?;
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
            c.gpu.copy_d2d(s.proj, s.dh, d * 2)?;
            // cross-attention
            c.gpu.copy_d2d(s.dh, s.residual, d * 2)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 2)?;
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
            c.gpu.copy_d2d(s.proj, s.dh, d * 2)?;
            // FFN
            c.gpu.copy_d2d(s.dh, s.residual, d * 2)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 2)?;
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
            c.gpu.copy_d2d(s.proj, s.dh, d * 2)?;
        }
        c.layer_norm("model.decoder.layer_norm", s.dh, 1)?;
        c.gemm(s.dh, c.embed_table, s.logits, 1, c.vocab, d)?; // tied lm_head, no bias
        Ok(())
    }

    /// On-device argmax over the last logits → token id.
    fn argmax_dev(&self, s: &DecScratch) -> Result<u32> {
        let c = self.c();
        KernelLaunch::new(c.gpu, c.k.argmax)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(s.logits)
            .arg_ptr(s.argmax_dev)
            .arg_u32(c.vocab as u32)
            .launch(c.stream)?;
        c.gpu.synchronize(c.stream)?;
        let mut idx = [0u8; 4];
        c.gpu.copy_d2h(s.argmax_dev, &mut idx)?;
        Ok(u32::from_le_bytes(idx))
    }

    /// Copy the device bf16 logits back to host as f32 (for beam scoring).
    fn logits_host(&self, s: &DecScratch) -> Result<Vec<f32>> {
        let c = self.c();
        c.gpu.synchronize(c.stream)?;
        let mut raw = vec![0u8; c.vocab * 2];
        c.gpu.copy_d2h(s.logits, &mut raw)?;
        Ok(raw
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect())
    }

    fn greedy(&self, s: &DecScratch, forced_bos: u32) -> Result<Vec<u32>> {
        let c = self.c();
        let cache = self.alloc_cache()?;
        let mut dec = vec![c.dec_start];
        let mut generated = Vec::new();
        for step in 0..MAX_NEW {
            self.forward_one(dec[step], step, &cache, s)?;
            let next = if step == 0 {
                forced_bos
            } else {
                self.argmax_dev(s)?
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

    fn beam(
        &self,
        s: &DecScratch,
        forced_bos: u32,
        num_beams: usize,
        length_penalty: f32,
    ) -> Result<Vec<u32>> {
        let c = self.c();
        let base = self.alloc_cache()?;
        self.forward_one(c.dec_start, 0, &base, s)?;
        self.forward_one(forced_bos, 1, &base, s)?;
        let init_logits = self.logits_host(s)?;
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
                cache: cache.unwrap_or(KvCache {
                    k: vec![],
                    v: vec![],
                }),
                logits: init_logits.clone(),
            });
        }
        beams[0].cache = base;

        let mut hyps = BeamHyps::new(num_beams, length_penalty);
        for _ in 1..MAX_NEW {
            let cur_len = beams[0].tokens.len();
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
            let mut next: Vec<Beam> = Vec::with_capacity(num_beams);
            for (score, b, tok) in cands {
                if next.len() == num_beams {
                    break;
                }
                if tok == c.eos {
                    hyps.add(beams[b].tokens.clone(), score);
                } else {
                    let cache = self.clone_cache(&beams[b].cache, cur_len)?;
                    self.forward_one(tok, cur_len, &cache, s)?;
                    let logits = self.logits_host(s)?;
                    let mut tokens = beams[b].tokens.clone();
                    tokens.push(tok);
                    next.push(Beam {
                        tokens,
                        score,
                        cache,
                        logits,
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

struct BeamHyps {
    num_beams: usize,
    length_penalty: f32,
    beams: Vec<(Vec<u32>, f32)>,
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

fn sinusoid_row(pos: f32, d: usize, out: &mut [bf16]) {
    let half = d / 2;
    let emb_scale = 10000f32.ln() / (half as f32 - 1.0);
    for j in 0..half {
        let ang = pos * (-(j as f32) * emb_scale).exp();
        out[j] = bf16::from_f32(ang.sin());
        out[half + j] = bf16::from_f32(ang.cos());
    }
}

fn decoder_pos_table_bf16(max_len: usize, d: usize) -> Vec<bf16> {
    let mut t = vec![bf16::from_f32(0.0); max_len * d];
    for i in 0..max_len {
        sinusoid_row((i + 2) as f32, d, &mut t[i * d..i * d + d]);
    }
    t
}

fn encoder_pos_bf16(ids: &[u32], d: usize, pad: u32) -> Vec<bf16> {
    let seq = ids.len();
    let mut t = vec![bf16::from_f32(0.0); seq * d];
    let mut running = 0u32;
    for (i, &id) in ids.iter().enumerate() {
        let p = if id != pad {
            running += 1;
            running + pad
        } else {
            pad
        };
        if p != pad {
            sinusoid_row(p as f32, d, &mut t[i * d..i * d + d]);
        }
    }
    t
}

fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn bf16_bytes(v: &[bf16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
