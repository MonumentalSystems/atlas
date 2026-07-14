// SPDX-License-Identifier: AGPL-3.0-only

use super::{CACHE_ROWS, MAX_NEW};
use crate::ctx::{Ctx, u32_bytes};
use anyhow::{Context, Result};
use half::bf16;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

pub struct DecCtx<'a> {
    pub ctx: &'a Ctx<'a>,
    pub cross_k: &'a [DevicePtr],
    pub cross_v: &'a [DevicePtr],
    pub pos_table: DevicePtr,
}

pub struct DecScratch {
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
    pub fn new(c: &Ctx) -> Result<Self> {
        Ok(Self {
            dh: c.bf16b(c.d)?,
            residual: c.bf16b(c.d)?,
            normed: c.bf16b(c.d)?,
            q: c.bf16b(c.d)?,
            attn: c.bf16b(c.d)?,
            proj: c.bf16b(c.d)?,
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
    fn alloc_cache(&self) -> Result<KvCache> {
        let c = self.ctx;
        let mut k = Vec::with_capacity(c.dec_layers);
        let mut v = Vec::with_capacity(c.dec_layers);
        for _ in 0..c.dec_layers {
            k.push(c.bf16b(CACHE_ROWS * c.d)?);
            v.push(c.bf16b(CACHE_ROWS * c.d)?);
        }
        Ok(KvCache { k, v })
    }

    fn clone_cache(&self, src: &KvCache, used: usize) -> Result<KvCache> {
        let c = self.ctx;
        let dst = self.alloc_cache()?;
        let bytes = used * c.d * 2;
        for l in 0..c.dec_layers {
            c.gpu.copy_d2d(src.k[l], dst.k[l], bytes)?;
            c.gpu.copy_d2d(src.v[l], dst.v[l], bytes)?;
        }
        Ok(dst)
    }

    fn free_cache(&self, cache: KvCache) -> Result<()> {
        for l in 0..self.ctx.dec_layers {
            self.ctx.gpu.free(cache.k[l])?;
            self.ctx.gpu.free(cache.v[l])?;
        }
        Ok(())
    }

    fn forward_one(&self, tok: u32, pos: usize, cache: &KvCache, s: &DecScratch) -> Result<()> {
        let c = self.ctx;
        let d = c.d;
        c.gpu.copy_h2d(u32_bytes(&[tok]), s.id_dev)?;
        c.embed_one(s.id_dev, s.dh)?;
        c.add(s.dh, self.pos_table.offset(pos * d * 2), d)?;

        let off = pos * d * 2;
        let tk = pos + 1;
        for l in 0..c.dec_layers {
            let p = format!("model.decoder.layers.{l}");
            c.gpu.copy_d2d(s.dh, s.residual, d * 2)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 2)?;
            c.layer_norm(&format!("{p}.self_attn_layer_norm"), s.normed, 1)?;
            c.linear1(&format!("{p}.self_attn.q_proj"), s.normed, s.q, d, d)?;
            c.linear1(
                &format!("{p}.self_attn.k_proj"),
                s.normed,
                cache.k[l].offset(off),
                d,
                d,
            )?;
            c.linear1(
                &format!("{p}.self_attn.v_proj"),
                s.normed,
                cache.v[l].offset(off),
                d,
                d,
            )?;
            c.attention(s.q, cache.k[l], cache.v[l], s.attn, 1, tk, false)?;
            c.linear1(&format!("{p}.self_attn.out_proj"), s.attn, s.proj, d, d)?;
            c.add(s.proj, s.residual, d)?;
            c.gpu.copy_d2d(s.proj, s.dh, d * 2)?;

            c.gpu.copy_d2d(s.dh, s.residual, d * 2)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 2)?;
            c.layer_norm(&format!("{p}.encoder_attn_layer_norm"), s.normed, 1)?;
            c.linear1(&format!("{p}.encoder_attn.q_proj"), s.normed, s.q, d, d)?;
            c.attention(
                s.q,
                self.cross_k[l],
                self.cross_v[l],
                s.attn,
                1,
                c.enc_len,
                false,
            )?;
            c.linear1(&format!("{p}.encoder_attn.out_proj"), s.attn, s.proj, d, d)?;
            c.add(s.proj, s.residual, d)?;
            c.gpu.copy_d2d(s.proj, s.dh, d * 2)?;

            c.gpu.copy_d2d(s.dh, s.residual, d * 2)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 2)?;
            c.layer_norm(&format!("{p}.final_layer_norm"), s.normed, 1)?;
            c.linear1(&format!("{p}.fc1"), s.normed, s.ff, c.ffn, d)?;
            c.relu(s.ff, c.ffn)?;
            c.linear1(&format!("{p}.fc2"), s.ff, s.proj, d, c.ffn)?;
            c.add(s.proj, s.residual, d)?;
            c.gpu.copy_d2d(s.proj, s.dh, d * 2)?;
        }
        c.layer_norm("model.decoder.layer_norm", s.dh, 1)?;
        c.lm_head(s.dh, s.logits)
    }

    fn argmax_dev(&self, s: &DecScratch) -> Result<u32> {
        let c = self.ctx;
        KernelLaunch::new(c.gpu, c.k.argmax)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_u32(c.vocab as u32)
            .arg_ptr(s.logits)
            .arg_ptr(s.argmax_dev)
            .launch(c.stream)?;
        c.gpu.synchronize(c.stream)?;
        let mut idx = [0u8; 4];
        c.gpu.copy_d2h(s.argmax_dev, &mut idx)?;
        Ok(u32::from_le_bytes(idx))
    }

    fn logits_host(&self, s: &DecScratch) -> Result<Vec<f32>> {
        let c = self.ctx;
        c.gpu.synchronize(c.stream)?;
        let mut raw = vec![0u8; c.vocab * 2];
        c.gpu.copy_d2h(s.logits, &mut raw)?;
        Ok(raw
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect())
    }

    pub fn greedy(&self, s: &DecScratch, forced_bos: u32) -> Result<Vec<u32>> {
        let c = self.ctx;
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

    pub fn beam(
        &self,
        s: &DecScratch,
        forced_bos: u32,
        num_beams: usize,
        length_penalty: f32,
    ) -> Result<Vec<u32>> {
        let c = self.ctx;
        let base = self.alloc_cache()?;
        self.forward_one(c.dec_start, 0, &base, s)?;
        self.forward_one(forced_bos, 1, &base, s)?;
        let init_logits = self.logits_host(s)?;
        let mut beams = Vec::with_capacity(num_beams);
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
                if tok == c.eos {
                    hyps.add(beams[b].tokens.clone(), score);
                } else {
                    let cache = self.clone_cache(&beams[b].cache, cur_len)?;
                    self.forward_one(tok, cur_len, &cache, s)?;
                    let mut tokens = beams[b].tokens.clone();
                    tokens.push(tok);
                    next.push(Beam {
                        tokens,
                        score,
                        cache,
                        logits: self.logits_host(s)?,
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
        self.beams.len() >= self.num_beams
            && self.worst() >= best_running_sum_logprob / (cur_len as f32).powf(self.length_penalty)
    }

    fn best(&self) -> Option<Vec<u32>> {
        self.beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(t, _)| t.clone())
    }
}

fn logsumexp(x: &[f32]) -> f32 {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m + x.iter().map(|&v| (v - m).exp()).sum::<f32>().ln()
}

fn top_k(x: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut best = Vec::with_capacity(k + 1);
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
