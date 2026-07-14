// SPDX-License-Identifier: AGPL-3.0-only

//! Metal NLLB-200 / M2M-100 decode validation with device KV cache and beam
//! search. Mirrors `nllb_cuda_gen` with fp32 Metal kernels.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};
use std::path::Path;

#[path = "nllb_metal_gen/beam.rs"]
mod beam;
#[allow(dead_code)]
#[path = "nllb_metal_translate/ctx.rs"]
mod ctx;
use beam::{Beam, BeamHyps, argmax_f32, decoder_pos_table, logsumexp, top_k};

use ctx::{Ctx, Kernels, Scratch};

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057;
const MAX_NEW: usize = 96;
const CACHE_ROWS: usize = MAX_NEW + 2;
const EXPECTED_GREEDY: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 123732, 248105, 30213, 248079, 1724, 25601, 385, 2,
];
const EXPECTED_BEAM5: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 4255, 956, 34821, 248105, 30213, 102506, 248116, 15510,
    385, 2,
];

fn main() -> Result<()> {
    let dir =
        std::env::var("ATLAS_NLLB_DIR").unwrap_or_else(|_| "/tank/hf/nllb-200-3.3B-st".into());
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
        "[nllb-metal-kv] d={d} heads={heads} head_dim={head_dim} ffn={ffn} enc={enc_layers} dec={dec_layers} vocab={vocab}"
    );

    println!("[nllb-metal-kv] loading weights to Metal ...");
    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;
    let kernels = Kernels::load(gpu)?;
    let embed_table = store.get("model.shared.weight")?.ptr;
    let ctx = Ctx {
        gpu,
        k: &kernels,
        d,
        heads,
        head_dim,
        ffn,
        attn_scale,
        stream,
    };

    let enc_len = INPUT_IDS.len();
    let enc_out = ctx.f32b(enc_len * d)?;
    let escr = Scratch::new(&ctx, enc_len)?;
    ctx.embed_and_positions(INPUT_IDS, embed_table, embed_scale, enc_out)?;
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
            &escr,
        )?;
        ctx.ffn_block(&store, &p, enc_out, enc_len, &escr)?;
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
    let pos_table = ctx.f32b(CACHE_ROWS * d)?;
    gpu.copy_h2d(f32_bytes(&decoder_pos_table(CACHE_ROWS, d)), pos_table)?;

    let dctx = DecCtx {
        ctx: &ctx,
        store: &store,
        cross_k: &cross_k,
        cross_v: &cross_v,
        pos_table,
        embed_table,
        embed_scale,
        vocab,
        dec_layers,
        enc_len,
        dec_start,
        eos,
    };
    let dscr = DecScratch::new(&dctx)?;

    let t0 = std::time::Instant::now();
    let greedy = dctx.greedy(&dscr, FORCED_BOS)?;
    println!("[nllb-metal-kv] greedy ids = {greedy:?}");
    anyhow::ensure!(greedy == EXPECTED_GREEDY, "greedy KV cache diverged");
    println!(
        "[nllb-metal-kv] greedy PASS in {:.3}s",
        t0.elapsed().as_secs_f64()
    );

    let t1 = std::time::Instant::now();
    let beam = dctx.beam(&dscr, FORCED_BOS, 5, 1.0)?;
    println!("[nllb-metal-kv] beam=5 ids = {beam:?}");
    anyhow::ensure!(beam == EXPECTED_BEAM5, "beam=5 KV cache diverged");
    println!(
        "[nllb-metal-kv] beam=5 PASS in {:.3}s",
        t1.elapsed().as_secs_f64()
    );
    Ok(())
}

struct DecCtx<'a> {
    ctx: &'a Ctx<'a>,
    store: &'a WeightStore,
    cross_k: &'a [DevicePtr],
    cross_v: &'a [DevicePtr],
    pos_table: DevicePtr,
    embed_table: DevicePtr,
    embed_scale: f32,
    vocab: usize,
    dec_layers: usize,
    enc_len: usize,
    dec_start: u32,
    eos: u32,
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
}

impl DecScratch {
    fn new(dctx: &DecCtx) -> Result<Self> {
        let c = dctx.ctx;
        Ok(Self {
            dh: c.f32b(c.d)?,
            residual: c.f32b(c.d)?,
            normed: c.f32b(c.d)?,
            q: c.f32b(c.d)?,
            attn: c.f32b(c.d)?,
            proj: c.f32b(c.d)?,
            ff: c.f32b(c.ffn)?,
            logits: c.f32b(dctx.vocab)?,
            id_dev: c.gpu.alloc(4)?,
        })
    }
}

struct KvCache {
    k: Vec<DevicePtr>,
    v: Vec<DevicePtr>,
}

impl<'a> DecCtx<'a> {
    fn alloc_cache(&self) -> Result<KvCache> {
        let c = self.ctx;
        let mut k = Vec::with_capacity(self.dec_layers);
        let mut v = Vec::with_capacity(self.dec_layers);
        for _ in 0..self.dec_layers {
            k.push(c.f32b(CACHE_ROWS * c.d)?);
            v.push(c.f32b(CACHE_ROWS * c.d)?);
        }
        Ok(KvCache { k, v })
    }

    fn clone_cache(&self, src: &KvCache, used: usize) -> Result<KvCache> {
        let c = self.ctx;
        let dst = self.alloc_cache()?;
        let bytes = used * c.d * 4;
        for l in 0..self.dec_layers {
            c.gpu.copy_d2d(src.k[l], dst.k[l], bytes)?;
            c.gpu.copy_d2d(src.v[l], dst.v[l], bytes)?;
        }
        Ok(dst)
    }

    fn free_cache(&self, cache: KvCache) -> Result<()> {
        for l in 0..self.dec_layers {
            self.ctx.gpu.free(cache.k[l])?;
            self.ctx.gpu.free(cache.v[l])?;
        }
        Ok(())
    }

    fn forward_one(
        &self,
        tok: u32,
        pos: usize,
        cache: &KvCache,
        s: &DecScratch,
        out_logits: &mut [u8],
    ) -> Result<()> {
        let c = self.ctx;
        let d = c.d;
        c.gpu.copy_h2d(u32_bytes(&[tok]), s.id_dev)?;
        c.embed_raw(s.id_dev, self.embed_table, s.dh, 1)?;
        c.scale(s.dh, d, self.embed_scale)?;
        c.add(s.dh, self.pos_table.offset(pos * d * 4), d)?;

        let off = pos * d * 4;
        let tk = pos + 1;
        for l in 0..self.dec_layers {
            let p = format!("model.decoder.layers.{l}");
            c.gpu.copy_d2d(s.dh, s.residual, d * 4)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 4)?;
            c.layer_norm(
                self.store,
                &format!("{p}.self_attn_layer_norm"),
                s.normed,
                1,
            )?;
            c.linear(
                self.store,
                &format!("{p}.self_attn.q_proj"),
                s.normed,
                s.q,
                1,
                d,
                d,
            )?;
            c.linear(
                self.store,
                &format!("{p}.self_attn.k_proj"),
                s.normed,
                cache.k[l].offset(off),
                1,
                d,
                d,
            )?;
            c.linear(
                self.store,
                &format!("{p}.self_attn.v_proj"),
                s.normed,
                cache.v[l].offset(off),
                1,
                d,
                d,
            )?;
            c.attention(s.q, cache.k[l], cache.v[l], s.attn, 1, tk, false)?;
            c.linear(
                self.store,
                &format!("{p}.self_attn.out_proj"),
                s.attn,
                s.proj,
                1,
                d,
                d,
            )?;
            c.add(s.proj, s.residual, d)?;
            c.gpu.copy_d2d(s.proj, s.dh, d * 4)?;

            c.gpu.copy_d2d(s.dh, s.residual, d * 4)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 4)?;
            c.layer_norm(
                self.store,
                &format!("{p}.encoder_attn_layer_norm"),
                s.normed,
                1,
            )?;
            c.linear(
                self.store,
                &format!("{p}.encoder_attn.q_proj"),
                s.normed,
                s.q,
                1,
                d,
                d,
            )?;
            c.attention(
                s.q,
                self.cross_k[l],
                self.cross_v[l],
                s.attn,
                1,
                self.enc_len,
                false,
            )?;
            c.linear(
                self.store,
                &format!("{p}.encoder_attn.out_proj"),
                s.attn,
                s.proj,
                1,
                d,
                d,
            )?;
            c.add(s.proj, s.residual, d)?;
            c.gpu.copy_d2d(s.proj, s.dh, d * 4)?;

            c.gpu.copy_d2d(s.dh, s.residual, d * 4)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 4)?;
            c.layer_norm(self.store, &format!("{p}.final_layer_norm"), s.normed, 1)?;
            c.linear(self.store, &format!("{p}.fc1"), s.normed, s.ff, 1, c.ffn, d)?;
            c.launch_relu(s.ff, c.ffn)?;
            c.linear(self.store, &format!("{p}.fc2"), s.ff, s.proj, 1, d, c.ffn)?;
            c.add(s.proj, s.residual, d)?;
            c.gpu.copy_d2d(s.proj, s.dh, d * 4)?;
        }
        c.layer_norm(self.store, "model.decoder.layer_norm", s.dh, 1)?;
        c.linear_no_bias_raw(s.dh, self.embed_table, s.logits, 1, self.vocab, d)?;
        c.gpu.synchronize(c.stream)?;
        c.gpu.copy_d2h(s.logits, out_logits)?;
        Ok(())
    }

    fn greedy(&self, s: &DecScratch, forced_bos: u32) -> Result<Vec<u32>> {
        let cache = self.alloc_cache()?;
        let mut logits = vec![0u8; self.vocab * 4];
        let mut dec = vec![self.dec_start];
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
            if next == self.eos {
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
        let mut logits = vec![0u8; self.vocab * 4];
        let base = self.alloc_cache()?;
        self.forward_one(self.dec_start, 0, &base, s, &mut logits)?;
        self.forward_one(forced_bos, 1, &base, s, &mut logits)?;
        let init_logits = f32_vec(&logits);
        let mut beams = Vec::with_capacity(num_beams);
        for b in 0..num_beams {
            let cache = if b == 0 {
                None
            } else {
                Some(self.clone_cache(&base, 2)?)
            };
            beams.push(Beam {
                tokens: vec![self.dec_start, forced_bos],
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
            let mut cands = Vec::new();
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
            let mut next = Vec::with_capacity(num_beams);
            for (score, b, tok) in cands {
                if next.len() == num_beams {
                    break;
                }
                if tok == self.eos {
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
        for beam in beams.drain(..) {
            if beam.score.is_finite() {
                hyps.add(beam.tokens.clone(), beam.score);
            }
            if !beam.cache.k.is_empty() {
                self.free_cache(beam.cache)?;
            }
        }
        let mut best = hyps.best().context("no finished hypotheses")?;
        best.push(self.eos);
        Ok(best[1..].to_vec())
    }
}

fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn f32_vec(b: &[u8]) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len() / 4) }.to_vec()
}
