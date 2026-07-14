// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of nllb_cuda_beambatch.rs for the 500-LoC cap.

use super::*;

impl Ctx<'_> {
    pub fn bf16b(&self, elems: usize) -> Result<DevicePtr> {
        self.gpu.alloc(elems * 2)
    }
    pub fn u32b(&self, elems: usize) -> Result<DevicePtr> {
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
    pub fn gemm(
        &self,
        a: DevicePtr,
        wt: DevicePtr,
        cc: DevicePtr,
        m: usize,
        n: usize,
        kk: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.gemm)
            .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(wt)
            .arg_ptr(cc)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            .launch(self.stream)
    }
    pub fn linear(
        &self,
        prefix: &str,
        a: DevicePtr,
        cc: DevicePtr,
        rows: usize,
        n: usize,
        kk: usize,
    ) -> Result<()> {
        self.gemm(a, self.w(&format!("{prefix}.weight"))?, cc, rows, n, kk)?;
        KernelLaunch::new(self.gpu, self.k.bias)
            .grid([div_ceil((rows * n) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(cc)
            .arg_ptr(self.w(&format!("{prefix}.bias"))?)
            .arg_u32(rows as u32)
            .arg_u32(n as u32)
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
    pub fn embed_seq(&self, ids: &[u32], out: DevicePtr) -> Result<()> {
        let (d, seq) = (self.d, ids.len());
        let ids_dev = self.u32b(seq)?;
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
    pub fn enc_self_attn(&self, layer: &str, x: DevicePtr, seq: usize, s: &Scratch) -> Result<()> {
        let (d, bytes) = (self.d, seq * self.d * 2);
        let p = format!("{layer}.self_attn");
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(&format!("{layer}.self_attn_layer_norm"), s.normed, seq)?;
        self.linear(&format!("{p}.q_proj"), s.normed, s.q, seq, d, d)?;
        self.linear(&format!("{p}.k_proj"), s.normed, s.kk, seq, d, d)?;
        self.linear(&format!("{p}.v_proj"), s.normed, s.v, seq, d, d)?;
        KernelLaunch::new(self.gpu, self.k.attn_enc)
            .grid([(seq * self.heads) as u32, 1, 1])
            .block([self.head_dim as u32, 1, 1])
            .shared_mem(((seq + self.head_dim) * 4) as u32)
            .arg_ptr(s.q)
            .arg_ptr(s.kk)
            .arg_ptr(s.v)
            .arg_ptr(s.attn)
            .arg_u32(seq as u32)
            .arg_u32(seq as u32)
            .arg_u32(self.heads as u32)
            .arg_u32(self.head_dim as u32)
            .arg_f32(self.attn_scale)
            .arg_u32(0)
            .launch(self.stream)?;
        self.linear(&format!("{p}.out_proj"), s.attn, s.proj, seq, d, d)?;
        self.add(s.proj, s.residual, seq * d)?;
        self.gpu.copy_d2d(s.proj, x, bytes)
    }
    pub fn ffn_block(&self, layer: &str, x: DevicePtr, rows: usize, s: &Scratch) -> Result<()> {
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
    pub fn scatter(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        pos: usize,
        b: usize,
        stride: usize,
        d: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.scatter)
            .grid([div_ceil((b * d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(pos as u32)
            .arg_u32(b as u32)
            .arg_u32(stride as u32)
            .arg_u32(d as u32)
            .launch(self.stream)
    }
    pub fn attn_batched(
        &self,
        q: DevicePtr,
        kc: DevicePtr,
        vc: DevicePtr,
        out: DevicePtr,
        b: usize,
        stride: usize,
        tk: DevicePtr,
        sh: u32,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.attn_dec)
            .grid([(b * self.heads) as u32, 1, 1])
            .block([self.head_dim as u32, 1, 1])
            .shared_mem(sh)
            .arg_ptr(q)
            .arg_ptr(kc)
            .arg_ptr(vc)
            .arg_ptr(out)
            .arg_u32(b as u32)
            .arg_u32(stride as u32)
            .arg_ptr(tk)
            .arg_u32(self.heads as u32)
            .arg_u32(self.head_dim as u32)
            .arg_f32(self.attn_scale)
            .launch(self.stream)
    }

    /// One batched decode step: fill self-cache row `pos` for all B beams and
    /// compute `logits[B, vocab]` (bf16, device). `cur` = one token per beam.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step(
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
        let (d, ffn) = (self.d, self.ffn);
        let sh = ((self.head_dim + MAX_NEW) * 4) as u32;
        self.gpu.copy_h2d(u32_bytes(cur), buf.id)?;
        KernelLaunch::new(self.gpu, self.k.embed)
            .grid([b as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(buf.id)
            .arg_ptr(self.embed_table)
            .arg_ptr(buf.dh)
            .arg_u32(d as u32)
            .launch(self.stream)?;
        KernelLaunch::new(self.gpu, self.k.scale)
            .grid([div_ceil((b * d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(buf.dh)
            .arg_u32((b * d) as u32)
            .arg_f32(self.embed_scale)
            .launch(self.stream)?;
        KernelLaunch::new(self.gpu, self.k.add_row)
            .grid([div_ceil((b * d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(buf.dh)
            .arg_ptr(buf.pos_table.offset(pos * d * 2))
            .arg_u32((b * d) as u32)
            .arg_u32(d as u32)
            .launch(self.stream)?;
        self.gpu
            .copy_h2d(u32_bytes(&vec![(pos + 1) as u32; b]), buf.selftk)?;
        for l in 0..self.dec_layers {
            let p = format!("model.decoder.layers.{l}");
            self.gpu.copy_d2d(buf.dh, buf.residual, b * d * 2)?;
            self.gpu.copy_d2d(buf.dh, buf.normed, b * d * 2)?;
            self.layer_norm(&format!("{p}.self_attn_layer_norm"), buf.normed, b)?;
            self.linear(&format!("{p}.self_attn.q_proj"), buf.normed, buf.q, b, d, d)?;
            self.linear(
                &format!("{p}.self_attn.k_proj"),
                buf.normed,
                buf.knew,
                b,
                d,
                d,
            )?;
            self.linear(
                &format!("{p}.self_attn.v_proj"),
                buf.normed,
                buf.vnew,
                b,
                d,
                d,
            )?;
            self.scatter(buf.knew, sk[l], pos, b, MAX_NEW, d)?;
            self.scatter(buf.vnew, sv[l], pos, b, MAX_NEW, d)?;
            self.attn_batched(buf.q, sk[l], sv[l], buf.attn, b, MAX_NEW, buf.selftk, sh)?;
            self.linear(
                &format!("{p}.self_attn.out_proj"),
                buf.attn,
                buf.proj,
                b,
                d,
                d,
            )?;
            self.add(buf.proj, buf.residual, b * d)?;
            self.gpu.copy_d2d(buf.proj, buf.dh, b * d * 2)?;
            self.gpu.copy_d2d(buf.dh, buf.residual, b * d * 2)?;
            self.gpu.copy_d2d(buf.dh, buf.normed, b * d * 2)?;
            self.layer_norm(&format!("{p}.encoder_attn_layer_norm"), buf.normed, b)?;
            self.linear(
                &format!("{p}.encoder_attn.q_proj"),
                buf.normed,
                buf.q,
                b,
                d,
                d,
            )?;
            self.attn_batched(
                buf.q,
                cross_k[l],
                cross_v[l],
                buf.attn,
                b,
                self.enc_len,
                buf.crosstk,
                sh,
            )?;
            self.linear(
                &format!("{p}.encoder_attn.out_proj"),
                buf.attn,
                buf.proj,
                b,
                d,
                d,
            )?;
            self.add(buf.proj, buf.residual, b * d)?;
            self.gpu.copy_d2d(buf.proj, buf.dh, b * d * 2)?;
            self.gpu.copy_d2d(buf.dh, buf.residual, b * d * 2)?;
            self.gpu.copy_d2d(buf.dh, buf.normed, b * d * 2)?;
            self.layer_norm(&format!("{p}.final_layer_norm"), buf.normed, b)?;
            self.linear(&format!("{p}.fc1"), buf.normed, buf.ff, b, ffn, d)?;
            KernelLaunch::new(self.gpu, self.k.relu)
                .grid([div_ceil((b * ffn) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(buf.ff)
                .arg_u32((b * ffn) as u32)
                .launch(self.stream)?;
            self.linear(&format!("{p}.fc2"), buf.ff, buf.proj, b, d, ffn)?;
            self.add(buf.proj, buf.residual, b * d)?;
            self.gpu.copy_d2d(buf.proj, buf.dh, b * d * 2)?;
        }
        self.layer_norm("model.decoder.layer_norm", buf.dh, b)?;
        self.gemm(buf.dh, self.embed_table, buf.logits, b, self.vocab, d)?;
        Ok(())
    }

    pub fn logits_host(&self, buf: &DecBuf, b: usize) -> Result<Vec<Vec<f32>>> {
        self.gpu.synchronize(self.stream)?;
        let mut raw = vec![0u8; b * self.vocab * 2];
        self.gpu.copy_d2h(buf.logits, &mut raw)?;
        Ok((0..b)
            .map(|bi| {
                raw[bi * self.vocab * 2..(bi + 1) * self.vocab * 2]
                    .chunks_exact(2)
                    .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect()
            })
            .collect())
    }

    pub fn beam_batched(
        &self,
        b: usize,
        cross_k: &[DevicePtr],
        cross_v: &[DevicePtr],
        forced_bos: u32,
        lp: f32,
    ) -> Result<Vec<u32>> {
        let d = self.d;
        let buf = DecBuf::new(self, b)?;
        // two ping-pong cache sets for reorder
        let mut sk: Vec<DevicePtr> = (0..self.dec_layers)
            .map(|_| self.bf16b(b * MAX_NEW * d))
            .collect::<Result<_>>()?;
        let mut sv: Vec<DevicePtr> = (0..self.dec_layers)
            .map(|_| self.bf16b(b * MAX_NEW * d))
            .collect::<Result<_>>()?;
        let mut sk2: Vec<DevicePtr> = (0..self.dec_layers)
            .map(|_| self.bf16b(b * MAX_NEW * d))
            .collect::<Result<_>>()?;
        let mut sv2: Vec<DevicePtr> = (0..self.dec_layers)
            .map(|_| self.bf16b(b * MAX_NEW * d))
            .collect::<Result<_>>()?;
        let perm_dev = self.u32b(b)?;

        // init: all beams decode [dec_start, forced_bos] into their slots
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
        let init_logits = self.logits_host(&buf, b)?;
        let mut beams: Vec<Beam> = (0..b)
            .map(|bi| Beam {
                tokens: vec![self.dec_start, forced_bos],
                score: if bi == 0 { 0.0 } else { f32::NEG_INFINITY },
                logits: init_logits[bi].clone(),
            })
            .collect();

        let mut hyps = BeamHyps::new(b, lp);
        for _ in 1..MAX_NEW {
            let cur_len = beams[0].tokens.len();
            let mut cands: Vec<(f32, usize, u32)> = Vec::new();
            for (bi, beam) in beams.iter().enumerate() {
                if !beam.score.is_finite() {
                    continue;
                }
                let lse = logsumexp(&beam.logits);
                for (val, tok) in top_k(&beam.logits, 2 * b) {
                    cands.push((beam.score + (val - lse), bi, tok as u32));
                }
            }
            cands.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());

            let mut perm = vec![0u32; b];
            let mut new_tokens: Vec<Vec<u32>> = Vec::with_capacity(b);
            let mut new_scores: Vec<f32> = Vec::with_capacity(b);
            let mut cur: Vec<u32> = Vec::with_capacity(b);
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

            // reorder caches: sk2[i] = sk[perm[i]] (rows 0..cur_len), then swap
            self.gpu.copy_h2d(u32_bytes(&perm), perm_dev)?;
            for l in 0..self.dec_layers {
                self.gather(sk[l], sk2[l], perm_dev, b, cur_len, MAX_NEW, d)?;
                self.gather(sv[l], sv2[l], perm_dev, b, cur_len, MAX_NEW, d)?;
            }
            std::mem::swap(&mut sk, &mut sk2);
            std::mem::swap(&mut sv, &mut sv2);

            // forward the new token for each beam (writes row cur_len)
            self.forward_step(&cur, cur_len, b, &sk, &sv, cross_k, cross_v, &buf)?;
            let lh = self.logits_host(&buf, b)?;
            beams = (0..b)
                .map(|i| Beam {
                    tokens: new_tokens[i].clone(),
                    score: new_scores[i],
                    logits: lh[i].clone(),
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
        Ok(best[1..].to_vec())
    }

    pub fn gather(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        perm: DevicePtr,
        b: usize,
        used: usize,
        stride: usize,
        d: usize,
    ) -> Result<()> {
        let n = (b * used * d) as u32;
        KernelLaunch::new(self.gpu, self.k.gather)
            .grid([div_ceil(n, 256), 1, 1])
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
