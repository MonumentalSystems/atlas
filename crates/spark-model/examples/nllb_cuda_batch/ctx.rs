// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of nllb_cuda_batch.rs for the 500-LoC cap.

use super::*;

// Batch of eng_Latn prompts (input ids) + their HF-bf16 greedy references.
pub fn prompts() -> Vec<Vec<u32>> {
    vec![
        vec![
            256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
        ],
        vec![256047, 1617, 167554, 248, 43978, 248075, 2],
        vec![256047, 138409, 200356, 248, 9, 19450, 5753, 248075, 2],
        vec![
            256047, 117, 9713, 6399, 9, 54445, 452, 121318, 248079, 43205, 248075, 2,
        ],
    ]
}

pub fn oracles() -> Vec<Vec<u32>> {
    vec![
        vec![
            256057, 17994, 141190, 248079, 25358, 123732, 248105, 30213, 385, 2,
        ],
        vec![256057, 1181, 14183, 613, 84809, 248075, 2],
        vec![
            256057, 1034, 80431, 1590, 88752, 1956, 613, 159, 86106, 80198, 248075, 2,
        ],
        vec![
            256057, 1048, 190412, 3335, 2626, 201, 79, 78752, 248079, 10, 248116, 73, 4255, 161248,
            248075, 2,
        ],
    ]
}

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
        c: DevicePtr,
        m: usize,
        n: usize,
        kk: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.gemm)
            .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(wt)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            .launch(self.stream)
    }
    /// Biased linear via GEMM (M=rows) + row-broadcast bias.
    pub fn linear(
        &self,
        prefix: &str,
        a: DevicePtr,
        c: DevicePtr,
        rows: usize,
        n: usize,
        kk: usize,
    ) -> Result<()> {
        self.gemm(a, self.w(&format!("{prefix}.weight"))?, c, rows, n, kk)?;
        KernelLaunch::new(self.gpu, self.k.bias)
            .grid([div_ceil((rows * n) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c)
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

    // ---- single-sequence encoder (M=seq) ----
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
        // single-sequence dense SDPA (shared K/V) via nllb_attn_kv_bf16
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

    // ---- batched decode ----
    #[allow(clippy::too_many_arguments)]
    pub fn batched_greedy(
        &self,
        b: usize,
        max_enc: usize,
        enc_lens: &[usize],
        cross_k: &[DevicePtr],
        cross_v: &[DevicePtr],
        forced_bos: u32,
    ) -> Result<Vec<Vec<u32>>> {
        let (d, ffn, h, hd) = (self.d, self.ffn, self.heads, self.head_dim);
        let dmodel = d;
        // scratch [B, d] / [B, ffn] / [B, vocab]
        let dh = self.bf16b(b * d)?;
        let residual = self.bf16b(b * d)?;
        let normed = self.bf16b(b * d)?;
        let q = self.bf16b(b * d)?;
        let knew = self.bf16b(b * d)?;
        let vnew = self.bf16b(b * d)?;
        let attn = self.bf16b(b * d)?;
        let proj = self.bf16b(b * d)?;
        let ff = self.bf16b(b * ffn)?;
        let logits = self.bf16b(b * self.vocab)?;
        let self_k: Vec<DevicePtr> = (0..self.dec_layers)
            .map(|_| self.bf16b(b * MAX_NEW * d))
            .collect::<Result<_>>()?;
        let self_v: Vec<DevicePtr> = (0..self.dec_layers)
            .map(|_| self.bf16b(b * MAX_NEW * d))
            .collect::<Result<_>>()?;
        let pos_table = self.bf16b(MAX_NEW * d)?;
        self.gpu
            .copy_h2d(bf16_bytes(&decoder_pos_table_bf16(MAX_NEW, d)), pos_table)?;

        let id_dev = self.u32b(b)?;
        let next_dev = self.u32b(b)?;
        let selftk_dev = self.u32b(b)?;
        let crosstk_dev = self.u32b(b)?;
        self.gpu.copy_h2d(
            u32_bytes(&enc_lens.iter().map(|&x| x as u32).collect::<Vec<_>>()),
            crosstk_dev,
        )?;

        let sh_attn = ((hd + MAX_NEW) * 4) as u32;
        let mut cur: Vec<u32> = vec![self.dec_start; b];
        let mut done = vec![false; b];
        let mut outs: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut next_host = vec![0u8; b * 4];

        for step in 0..MAX_NEW {
            // embed batch of `cur` tokens
            self.gpu.copy_h2d(u32_bytes(&cur), id_dev)?;
            KernelLaunch::new(self.gpu, self.k.embed)
                .grid([b as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(id_dev)
                .arg_ptr(self.embed_table)
                .arg_ptr(dh)
                .arg_u32(d as u32)
                .launch(self.stream)?;
            KernelLaunch::new(self.gpu, self.k.scale)
                .grid([div_ceil((b * d) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(dh)
                .arg_u32((b * d) as u32)
                .arg_f32(self.embed_scale)
                .launch(self.stream)?;
            KernelLaunch::new(self.gpu, self.k.add_row)
                .grid([div_ceil((b * d) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(dh)
                .arg_ptr(pos_table.offset(step * d * 2))
                .arg_u32((b * d) as u32)
                .arg_u32(d as u32)
                .launch(self.stream)?;

            self.gpu
                .copy_h2d(u32_bytes(&vec![(step + 1) as u32; b]), selftk_dev)?;

            for l in 0..self.dec_layers {
                let p = format!("model.decoder.layers.{l}");
                // self-attention (per-seq causal cache)
                self.gpu.copy_d2d(dh, residual, b * d * 2)?;
                self.gpu.copy_d2d(dh, normed, b * d * 2)?;
                self.layer_norm(&format!("{p}.self_attn_layer_norm"), normed, b)?;
                self.linear(&format!("{p}.self_attn.q_proj"), normed, q, b, d, d)?;
                self.linear(&format!("{p}.self_attn.k_proj"), normed, knew, b, d, d)?;
                self.linear(&format!("{p}.self_attn.v_proj"), normed, vnew, b, d, d)?;
                self.scatter(knew, self_k[l], step, b, MAX_NEW, d)?;
                self.scatter(vnew, self_v[l], step, b, MAX_NEW, d)?;
                self.attn_batched(
                    q, self_k[l], self_v[l], attn, b, MAX_NEW, selftk_dev, sh_attn,
                )?;
                self.linear(&format!("{p}.self_attn.out_proj"), attn, proj, b, d, d)?;
                self.add(proj, residual, b * d)?;
                self.gpu.copy_d2d(proj, dh, b * d * 2)?;
                // cross-attention (per-seq encoder cache)
                self.gpu.copy_d2d(dh, residual, b * d * 2)?;
                self.gpu.copy_d2d(dh, normed, b * d * 2)?;
                self.layer_norm(&format!("{p}.encoder_attn_layer_norm"), normed, b)?;
                self.linear(&format!("{p}.encoder_attn.q_proj"), normed, q, b, d, d)?;
                self.attn_batched(
                    q,
                    cross_k[l],
                    cross_v[l],
                    attn,
                    b,
                    max_enc,
                    crosstk_dev,
                    sh_attn,
                )?;
                self.linear(&format!("{p}.encoder_attn.out_proj"), attn, proj, b, d, d)?;
                self.add(proj, residual, b * d)?;
                self.gpu.copy_d2d(proj, dh, b * d * 2)?;
                // FFN
                self.gpu.copy_d2d(dh, residual, b * d * 2)?;
                self.gpu.copy_d2d(dh, normed, b * d * 2)?;
                self.layer_norm(&format!("{p}.final_layer_norm"), normed, b)?;
                self.linear(&format!("{p}.fc1"), normed, ff, b, ffn, d)?;
                KernelLaunch::new(self.gpu, self.k.relu)
                    .grid([div_ceil((b * ffn) as u32, 256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(ff)
                    .arg_u32((b * ffn) as u32)
                    .launch(self.stream)?;
                self.linear(&format!("{p}.fc2"), ff, proj, b, d, ffn)?;
                self.add(proj, residual, b * d)?;
                self.gpu.copy_d2d(proj, dh, b * d * 2)?;
            }
            self.layer_norm("model.decoder.layer_norm", dh, b)?;
            self.gemm(dh, self.embed_table, logits, b, self.vocab, d)?; // batched tied lm_head
            KernelLaunch::new(self.gpu, self.k.argmax)
                .grid([b as u32, 1, 1])
                .block([256, 1, 1])
                .shared_mem(256 * 8)
                .arg_ptr(logits)
                .arg_ptr(next_dev)
                .arg_u32(b as u32)
                .arg_u32(self.vocab as u32)
                .launch(self.stream)?;
            self.gpu.synchronize(self.stream)?;
            self.gpu.copy_d2h(next_dev, &mut next_host)?;

            let mut all_done = true;
            for bi in 0..b {
                let am = u32::from_le_bytes([
                    next_host[bi * 4],
                    next_host[bi * 4 + 1],
                    next_host[bi * 4 + 2],
                    next_host[bi * 4 + 3],
                ]);
                let nx = if step == 0 {
                    forced_bos
                } else if done[bi] {
                    self.eos
                } else {
                    am
                };
                if !done[bi] {
                    outs[bi].push(nx);
                    if nx == self.eos {
                        done[bi] = true;
                    }
                }
                cur[bi] = nx;
                all_done &= done[bi];
            }
            if all_done {
                break;
            }
        }

        // cleanup
        for p in [
            dh,
            residual,
            normed,
            q,
            knew,
            vnew,
            attn,
            proj,
            ff,
            logits,
            pos_table,
            id_dev,
            next_dev,
            selftk_dev,
            crosstk_dev,
        ] {
            self.gpu.free(p)?;
        }
        for l in 0..self.dec_layers {
            self.gpu.free(self_k[l])?;
            self.gpu.free(self_v[l])?;
        }
        let _ = (dmodel, h, hd);
        Ok(outs)
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
}

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

pub fn sinusoid_row(pos: f32, d: usize, out: &mut [bf16]) {
    let half = d / 2;
    let emb_scale = 10000f32.ln() / (half as f32 - 1.0);
    for j in 0..half {
        let ang = pos * (-(j as f32) * emb_scale).exp();
        out[j] = bf16::from_f32(ang.sin());
        out[half + j] = bf16::from_f32(ang.cos());
    }
}

pub fn decoder_pos_table_bf16(max_len: usize, d: usize) -> Vec<bf16> {
    let mut t = vec![bf16::from_f32(0.0); max_len * d];
    for i in 0..max_len {
        sinusoid_row((i + 2) as f32, d, &mut t[i * d..i * d + d]);
    }
    t
}

pub fn encoder_pos_bf16(ids: &[u32], d: usize, pad: u32) -> Vec<bf16> {
    let mut t = vec![bf16::from_f32(0.0); ids.len() * d];
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

pub fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn bf16_bytes(v: &[bf16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
