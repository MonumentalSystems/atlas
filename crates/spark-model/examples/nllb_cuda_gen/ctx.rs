// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of nllb_cuda_gen.rs for the 500-LoC cap.

use super::*;

impl Ctx<'_> {
    pub fn f32b(&self, elems: usize) -> Result<DevicePtr> {
        self.gpu.alloc(elems * 4)
    }
    pub fn w(&self, name: &str) -> Result<DevicePtr> {
        Ok(self.store.get(name)?.ptr)
    }

    pub fn layer_norm(&self, prefix: &str, x: DevicePtr, rows: usize) -> Result<()> {
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
    pub fn linear(
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
    pub fn lm_head(&self, a: DevicePtr, c: DevicePtr, rows: usize) -> Result<()> {
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

    pub fn attention(
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

    pub fn add(&self, dst: DevicePtr, src: DevicePtr, n: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.add)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_u32(n as u32)
            .launch(self.stream)
    }

    /// Embed a whole token sequence (encoder) → out[seq, d], scaled + positions.
    pub fn embed_seq(&self, ids: &[u32], out: DevicePtr) -> Result<()> {
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
    pub fn enc_self_attn(&self, layer: &str, x: DevicePtr, seq: usize, s: &Scratch) -> Result<()> {
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
    pub fn ffn_block(&self, layer: &str, x: DevicePtr, rows: usize, s: &Scratch) -> Result<()> {
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

impl Scratch {
    pub fn new(c: &Ctx, rows: usize) -> Result<Self> {
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

impl DecScratch {
    pub fn new(c: &Ctx) -> Result<Self> {
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

impl<'a> DecCtx<'a> {
    pub fn c(&self) -> &Ctx<'a> {
        self.ctx
    }

    pub fn alloc_cache(&self) -> Result<KvCache> {
        let c = self.c();
        let mut k = Vec::with_capacity(c.dec_layers);
        let mut v = Vec::with_capacity(c.dec_layers);
        for _ in 0..c.dec_layers {
            k.push(c.f32b(CACHE_ROWS * c.d)?);
            v.push(c.f32b(CACHE_ROWS * c.d)?);
        }
        Ok(KvCache { k, v })
    }

    /// Clone the first `used` rows of every layer's K/V into a fresh cache.
    pub fn clone_cache(&self, src: &KvCache, used: usize) -> Result<KvCache> {
        let c = self.c();
        let dst = self.alloc_cache()?;
        let bytes = used * c.d * 4;
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

    /// Forward ONE decoder token at sequence index `pos` (0-based), writing its
    /// self-attn K/V into `cache` row `pos` and attending over rows `0..=pos`.
    /// Returns host logits [vocab] predicting the next token.
    pub fn forward_one(
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
    pub fn greedy(&self, s: &DecScratch, forced_bos: u32) -> Result<Vec<u32>> {
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
    pub fn beam(
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
