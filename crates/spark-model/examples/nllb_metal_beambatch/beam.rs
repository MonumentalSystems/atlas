// SPDX-License-Identifier: AGPL-3.0-only

use super::MAX_NEW;
use crate::ctx::{Ctx, bf16_bytes, decoder_pos_table_bf16, u32_bytes};
use anyhow::{Context, Result};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const TOPK: usize = 10;

impl Ctx<'_> {
    pub fn beam_batched(
        &self,
        b: usize,
        cross_k: &[DevicePtr],
        cross_v: &[DevicePtr],
        forced_bos: u32,
        lp: f32,
    ) -> Result<Vec<u32>> {
        let buf = DecBuf::new(self, b)?;
        let mut sk = self.cache_set(b)?;
        let mut sv = self.cache_set(b)?;
        let mut sk2 = self.cache_set(b)?;
        let mut sv2 = self.cache_set(b)?;
        let perm_dev = self.gpu.alloc(b * 4)?;

        self.forward_step(
            &vec![self.dec_start; b],
            0,
            b,
            &sk,
            &sv,
            cross_k,
            cross_v,
            &buf,
        )?;
        self.forward_step(&vec![forced_bos; b], 1, b, &sk, &sv, cross_k, cross_v, &buf)?;
        let init_candidates = self.candidates_host(&buf, b)?;
        let mut beams: Vec<Beam> = (0..b)
            .map(|bi| Beam {
                tokens: vec![self.dec_start, forced_bos],
                score: if bi == 0 { 0.0 } else { f32::NEG_INFINITY },
                candidates: init_candidates[bi].clone(),
            })
            .collect();

        let mut hyps = BeamHyps::new(b, lp);
        for _ in 1..MAX_NEW {
            let cur_len = beams[0].tokens.len();
            let mut cands = Vec::new();
            for (bi, beam) in beams.iter().enumerate() {
                if !beam.score.is_finite() {
                    continue;
                }
                for (&val, &tok) in beam
                    .candidates
                    .vals
                    .iter()
                    .zip(beam.candidates.ids.iter())
                    .take(2 * b)
                {
                    cands.push((beam.score + (val - beam.candidates.lse), bi, tok));
                }
            }
            cands.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());

            let mut perm = vec![0u32; b];
            let mut new_tokens = Vec::with_capacity(b);
            let mut new_scores = Vec::with_capacity(b);
            let mut cur = Vec::with_capacity(b);
            for (score, parent, tok) in cands {
                if new_tokens.len() == b {
                    break;
                }
                if tok == self.eos {
                    hyps.add(beams[parent].tokens.clone(), score);
                } else {
                    let i = new_tokens.len();
                    perm[i] = parent as u32;
                    let mut t = beams[parent].tokens.clone();
                    t.push(tok);
                    new_tokens.push(t);
                    new_scores.push(score);
                    cur.push(tok);
                }
            }
            let best_running = new_scores[0];
            self.gpu.copy_h2d(u32_bytes(&perm), perm_dev)?;
            for l in 0..self.dec_layers {
                self.gather(sk[l], sk2[l], perm_dev, b, cur_len, MAX_NEW, self.d)?;
                self.gather(sv[l], sv2[l], perm_dev, b, cur_len, MAX_NEW, self.d)?;
            }
            std::mem::swap(&mut sk, &mut sk2);
            std::mem::swap(&mut sv, &mut sv2);

            self.forward_step(&cur, cur_len, b, &sk, &sv, cross_k, cross_v, &buf)?;
            let candidates = self.candidates_host(&buf, b)?;
            beams = (0..b)
                .map(|i| Beam {
                    tokens: new_tokens[i].clone(),
                    score: new_scores[i],
                    candidates: candidates[i].clone(),
                })
                .collect();
            if hyps.is_done(best_running, cur_len) {
                break;
            }
        }
        for beam in &beams {
            if beam.score.is_finite() {
                hyps.add(beam.tokens.clone(), beam.score);
            }
        }
        let mut best = hyps.best().context("no finished hypotheses")?;
        best.push(self.eos);
        buf.free(self)?;
        self.free_cache_set(sk)?;
        self.free_cache_set(sv)?;
        self.free_cache_set(sk2)?;
        self.free_cache_set(sv2)?;
        self.gpu.free(perm_dev)?;
        Ok(best[1..].to_vec())
    }

    fn cache_set(&self, b: usize) -> Result<Vec<DevicePtr>> {
        (0..self.dec_layers)
            .map(|_| self.bf16b(b * MAX_NEW * self.d))
            .collect()
    }

    fn free_cache_set(&self, cache: Vec<DevicePtr>) -> Result<()> {
        for ptr in cache {
            self.gpu.free(ptr)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_step(
        &self,
        cur: &[u32],
        pos: usize,
        b: usize,
        sk: &[DevicePtr],
        sv: &[DevicePtr],
        cross_k: &[DevicePtr],
        cross_v: &[DevicePtr],
        buf: &DecBuf,
    ) -> Result<()> {
        self.embed_batch(cur, buf.id, buf.dh)?;
        self.add_row(
            buf.dh,
            buf.pos_table.offset(pos * self.d * 2),
            b * self.d,
            self.d,
        )?;
        self.gpu
            .copy_h2d(u32_bytes(&vec![(pos + 1) as u32; b]), buf.self_tk)?;
        for l in 0..self.dec_layers {
            let p = format!("model.decoder.layers.{l}");
            self.self_attn_batch(
                &p,
                buf.dh,
                buf.residual,
                buf.normed,
                buf.q,
                buf.knew,
                buf.vnew,
                buf.attn,
                buf.proj,
                sk[l],
                sv[l],
                pos,
                buf.self_tk,
                b,
            )?;
            self.cross_attn_batch(
                &p,
                buf.dh,
                buf.residual,
                buf.normed,
                buf.q,
                buf.attn,
                buf.proj,
                cross_k[l],
                cross_v[l],
                buf.cross_tk,
                b,
                self.enc_len,
            )?;
            self.ffn_batch(&p, buf.dh, buf.residual, buf.normed, buf.ff, buf.proj, b)?;
        }
        self.layer_norm("model.decoder.layer_norm", buf.dh, b)?;
        self.lm_head_batch(buf.dh, buf.logits, b)
    }

    fn candidates_host(&self, buf: &DecBuf, b: usize) -> Result<Vec<Candidates>> {
        KernelLaunch::new(self.gpu, self.k.topk_lse)
            .grid([b as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(buf.logits)
            .arg_ptr(buf.top_vals)
            .arg_ptr(buf.top_ids)
            .arg_ptr(buf.lse)
            .arg_u32(b as u32)
            .arg_u32(self.vocab as u32)
            .launch(self.stream)?;
        self.gpu.synchronize(self.stream)?;
        let mut vals_raw = vec![0u8; b * TOPK * 4];
        let mut ids_raw = vec![0u8; b * TOPK * 4];
        let mut lse_raw = vec![0u8; b * 4];
        self.gpu.copy_d2h(buf.top_vals, &mut vals_raw)?;
        self.gpu.copy_d2h(buf.top_ids, &mut ids_raw)?;
        self.gpu.copy_d2h(buf.lse, &mut lse_raw)?;
        let vals = f32_slice(&vals_raw);
        let ids = u32_slice(&ids_raw);
        let lse = f32_slice(&lse_raw);
        Ok((0..b)
            .map(|bi| Candidates {
                vals: vals[bi * TOPK..(bi + 1) * TOPK].to_vec(),
                ids: ids[bi * TOPK..(bi + 1) * TOPK].to_vec(),
                lse: lse[bi],
            })
            .collect())
    }

    fn gather(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        perm: DevicePtr,
        b: usize,
        used: usize,
        stride: usize,
        d: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.gather)
            .grid([div_ceil((b * used * d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_ptr(perm)
            .arg_u32(b as u32)
            .arg_u32(used as u32)
            .arg_u32(stride as u32)
            .arg_u32(d as u32)
            .launch(self.stream)
    }
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
    top_vals: DevicePtr,
    top_ids: DevicePtr,
    lse: DevicePtr,
    id: DevicePtr,
    self_tk: DevicePtr,
    cross_tk: DevicePtr,
    pos_table: DevicePtr,
}

impl DecBuf {
    fn new(c: &Ctx, b: usize) -> Result<Self> {
        let d = c.d;
        let cross_tk = c.gpu.alloc(b * 4)?;
        c.gpu
            .copy_h2d(u32_bytes(&vec![c.enc_len as u32; b]), cross_tk)?;
        let pos_table = c.bf16b(MAX_NEW * d)?;
        c.gpu
            .copy_h2d(bf16_bytes(&decoder_pos_table_bf16(MAX_NEW, d)), pos_table)?;
        Ok(Self {
            dh: c.bf16b(b * d)?,
            residual: c.bf16b(b * d)?,
            normed: c.bf16b(b * d)?,
            q: c.bf16b(b * d)?,
            knew: c.bf16b(b * d)?,
            vnew: c.bf16b(b * d)?,
            attn: c.bf16b(b * d)?,
            proj: c.bf16b(b * d)?,
            ff: c.bf16b(b * c.ffn)?,
            logits: c.bf16b(b * c.vocab)?,
            top_vals: c.gpu.alloc(b * TOPK * 4)?,
            top_ids: c.gpu.alloc(b * TOPK * 4)?,
            lse: c.gpu.alloc(b * 4)?,
            id: c.gpu.alloc(b * 4)?,
            self_tk: c.gpu.alloc(b * 4)?,
            cross_tk,
            pos_table,
        })
    }

    fn free(self, c: &Ctx) -> Result<()> {
        for ptr in [
            self.dh,
            self.residual,
            self.normed,
            self.q,
            self.knew,
            self.vnew,
            self.attn,
            self.proj,
            self.ff,
            self.logits,
            self.top_vals,
            self.top_ids,
            self.lse,
            self.id,
            self.self_tk,
            self.cross_tk,
            self.pos_table,
        ] {
            c.gpu.free(ptr)?;
        }
        Ok(())
    }
}

struct Beam {
    tokens: Vec<u32>,
    score: f32,
    candidates: Candidates,
}

#[derive(Clone)]
struct Candidates {
    vals: Vec<f32>,
    ids: Vec<u32>,
    lse: f32,
}

struct BeamHyps {
    num_beams: usize,
    lp: f32,
    beams: Vec<(Vec<u32>, f32)>,
}

impl BeamHyps {
    fn new(num_beams: usize, lp: f32) -> Self {
        Self {
            num_beams,
            lp,
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
        let score = sum_logprob / (tokens.len() as f32).powf(self.lp);
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

    fn is_done(&self, best_running: f32, cur_len: usize) -> bool {
        self.beams.len() >= self.num_beams
            && self.worst() >= best_running / (cur_len as f32).powf(self.lp)
    }

    fn best(&self) -> Option<Vec<u32>> {
        self.beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(t, _)| t.clone())
    }
}

fn f32_slice(b: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len() / 4) }
}

fn u32_slice(b: &[u8]) -> &[u32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u32>(), b.len() / 4) }
}
