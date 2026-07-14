// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of nllb_cuda_bf16.rs for the 500-LoC cap.

use super::*;

impl Scratch {
    pub fn new(c: &Ctx, rows: usize) -> Result<Self> {
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

impl DecScratch {
    pub fn new(c: &Ctx) -> Result<Self> {
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

impl<'a> DecCtx<'a> {
    pub fn c(&self) -> &Ctx<'a> {
        self.ctx
    }

    pub fn alloc_cache(&self) -> Result<KvCache> {
        let c = self.c();
        let mut k = Vec::with_capacity(c.dec_layers);
        let mut v = Vec::with_capacity(c.dec_layers);
        for _ in 0..c.dec_layers {
            k.push(c.bf16b(CACHE_ROWS * c.d)?);
            v.push(c.bf16b(CACHE_ROWS * c.d)?);
        }
        Ok(KvCache { k, v })
    }

    pub fn clone_cache(&self, src: &KvCache, used: usize) -> Result<KvCache> {
        let c = self.c();
        let dst = self.alloc_cache()?;
        let bytes = used * c.d * 2;
        for l in 0..c.dec_layers {
            c.gpu.copy_d2d(src.k[l], dst.k[l], bytes)?;
            c.gpu.copy_d2d(src.v[l], dst.v[l], bytes)?;
        }
        Ok(dst)
    }

    pub fn free_cache(&self, cache: KvCache) -> Result<()> {
        let c = self.c();
        for l in 0..c.dec_layers {
            c.gpu.free(cache.k[l])?;
            c.gpu.free(cache.v[l])?;
        }
        Ok(())
    }

    /// Forward one decoder token at index `pos`; write its self-attn K/V into
    /// `cache` row `pos`, attend rows 0..=pos. Fills `s.logits` (bf16, device).
    pub fn forward_one(&self, tok: u32, pos: usize, cache: &KvCache, s: &DecScratch) -> Result<()> {
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
            // cross-attention
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
            // FFN
            c.gpu.copy_d2d(s.dh, s.residual, d * 2)?;
            c.gpu.copy_d2d(s.dh, s.normed, d * 2)?;
            c.layer_norm(&format!("{p}.final_layer_norm"), s.normed, 1)?;
            c.linear1(&format!("{p}.fc1"), s.normed, s.ff, c.ffn, d)?;
            KernelLaunch::new(c.gpu, c.k.relu)
                .grid([div_ceil(c.ffn as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(s.ff)
                .arg_u32(c.ffn as u32)
                .launch(c.stream)?;
            c.linear1(&format!("{p}.fc2"), s.ff, s.proj, d, c.ffn)?;
            c.add(s.proj, s.residual, d)?;
            c.gpu.copy_d2d(s.proj, s.dh, d * 2)?;
        }
        c.layer_norm("model.decoder.layer_norm", s.dh, 1)?;
        c.gemv(s.dh, c.embed_table, DevicePtr(0), s.logits, c.vocab, d)?; // tied lm_head, no bias
        Ok(())
    }

    /// On-device argmax over the last logits → token id.
    pub fn argmax_dev(&self, s: &DecScratch) -> Result<u32> {
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
    pub fn logits_host(&self, s: &DecScratch) -> Result<Vec<f32>> {
        let c = self.c();
        c.gpu.synchronize(c.stream)?;
        let mut raw = vec![0u8; c.vocab * 2];
        c.gpu.copy_d2h(s.logits, &mut raw)?;
        Ok(raw
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect())
    }

    pub fn greedy(&self, s: &DecScratch, forced_bos: u32) -> Result<Vec<u32>> {
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

    pub fn beam(
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
