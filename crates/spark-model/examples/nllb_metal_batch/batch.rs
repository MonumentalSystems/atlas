// SPDX-License-Identifier: AGPL-3.0-only

use super::MAX_NEW;
use crate::ctx::{Ctx, bf16_bytes, decoder_pos_table_bf16, u32_bytes};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

impl Ctx<'_> {
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
        let d = self.d;
        let dh = self.bf16b(b * d)?;
        let residual = self.bf16b(b * d)?;
        let normed = self.bf16b(b * d)?;
        let q = self.bf16b(b * d)?;
        let knew = self.bf16b(b * d)?;
        let vnew = self.bf16b(b * d)?;
        let attn = self.bf16b(b * d)?;
        let proj = self.bf16b(b * d)?;
        let ff = self.bf16b(b * self.ffn)?;
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

        let id_dev = self.gpu.alloc(b * 4)?;
        let next_dev = self.gpu.alloc(b * 4)?;
        let self_tk_dev = self.gpu.alloc(b * 4)?;
        let cross_tk_dev = self.gpu.alloc(b * 4)?;
        self.gpu.copy_h2d(
            u32_bytes(&enc_lens.iter().map(|&x| x as u32).collect::<Vec<_>>()),
            cross_tk_dev,
        )?;

        let mut cur = vec![self.dec_start; b];
        let mut done = vec![false; b];
        let mut outs = vec![Vec::new(); b];
        let mut next_host = vec![0u8; b * 4];
        for step in 0..MAX_NEW {
            self.embed_batch(&cur, id_dev, dh)?;
            self.add_row(dh, pos_table.offset(step * d * 2), b * d, d)?;
            self.gpu
                .copy_h2d(u32_bytes(&vec![(step + 1) as u32; b]), self_tk_dev)?;
            for l in 0..self.dec_layers {
                let p = format!("model.decoder.layers.{l}");
                self.self_attn_batch(
                    &p,
                    dh,
                    residual,
                    normed,
                    q,
                    knew,
                    vnew,
                    attn,
                    proj,
                    self_k[l],
                    self_v[l],
                    step,
                    self_tk_dev,
                    b,
                )?;
                self.cross_attn_batch(
                    &p,
                    dh,
                    residual,
                    normed,
                    q,
                    attn,
                    proj,
                    cross_k[l],
                    cross_v[l],
                    cross_tk_dev,
                    b,
                    max_enc,
                )?;
                self.ffn_batch(&p, dh, residual, normed, ff, proj, b)?;
            }
            self.layer_norm("model.decoder.layer_norm", dh, b)?;
            self.lm_head_batch(dh, logits, b)?;
            self.argmax_batch(logits, next_dev, b)?;
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
            self_tk_dev,
            cross_tk_dev,
        ] {
            self.gpu.free(p)?;
        }
        for l in 0..self.dec_layers {
            self.gpu.free(self_k[l])?;
            self.gpu.free(self_v[l])?;
        }
        Ok(outs)
    }

    fn embed_batch(&self, cur: &[u32], id_dev: DevicePtr, out: DevicePtr) -> Result<()> {
        self.gpu.copy_h2d(u32_bytes(cur), id_dev)?;
        KernelLaunch::new(self.gpu, self.k.embed)
            .grid([cur.len() as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(id_dev)
            .arg_ptr(self.embed_table)
            .arg_ptr(out)
            .arg_u32(self.d as u32)
            .launch(self.stream)?;
        KernelLaunch::new(self.gpu, self.k.scale)
            .grid([div_ceil((cur.len() * self.d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(out)
            .arg_u32((cur.len() * self.d) as u32)
            .arg_f32(self.embed_scale)
            .launch(self.stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn self_attn_batch(
        &self,
        layer: &str,
        dh: DevicePtr,
        residual: DevicePtr,
        normed: DevicePtr,
        q: DevicePtr,
        knew: DevicePtr,
        vnew: DevicePtr,
        attn: DevicePtr,
        proj: DevicePtr,
        self_k: DevicePtr,
        self_v: DevicePtr,
        step: usize,
        tk_dev: DevicePtr,
        b: usize,
    ) -> Result<()> {
        let d = self.d;
        self.gpu.copy_d2d(dh, residual, b * d * 2)?;
        self.gpu.copy_d2d(dh, normed, b * d * 2)?;
        self.layer_norm(&format!("{layer}.self_attn_layer_norm"), normed, b)?;
        self.linear(&format!("{layer}.self_attn.q_proj"), normed, q, b, d, d)?;
        self.linear(&format!("{layer}.self_attn.k_proj"), normed, knew, b, d, d)?;
        self.linear(&format!("{layer}.self_attn.v_proj"), normed, vnew, b, d, d)?;
        self.scatter(knew, self_k, step, b, MAX_NEW, d)?;
        self.scatter(vnew, self_v, step, b, MAX_NEW, d)?;
        self.attn_decode(q, self_k, self_v, attn, b, MAX_NEW, tk_dev)?;
        self.linear(&format!("{layer}.self_attn.out_proj"), attn, proj, b, d, d)?;
        self.add(proj, residual, b * d)?;
        self.gpu.copy_d2d(proj, dh, b * d * 2)
    }

    #[allow(clippy::too_many_arguments)]
    fn cross_attn_batch(
        &self,
        layer: &str,
        dh: DevicePtr,
        residual: DevicePtr,
        normed: DevicePtr,
        q: DevicePtr,
        attn: DevicePtr,
        proj: DevicePtr,
        cross_k: DevicePtr,
        cross_v: DevicePtr,
        tk_dev: DevicePtr,
        b: usize,
        max_enc: usize,
    ) -> Result<()> {
        let d = self.d;
        self.gpu.copy_d2d(dh, residual, b * d * 2)?;
        self.gpu.copy_d2d(dh, normed, b * d * 2)?;
        self.layer_norm(&format!("{layer}.encoder_attn_layer_norm"), normed, b)?;
        self.linear(&format!("{layer}.encoder_attn.q_proj"), normed, q, b, d, d)?;
        self.attn_decode(q, cross_k, cross_v, attn, b, max_enc, tk_dev)?;
        self.linear(
            &format!("{layer}.encoder_attn.out_proj"),
            attn,
            proj,
            b,
            d,
            d,
        )?;
        self.add(proj, residual, b * d)?;
        self.gpu.copy_d2d(proj, dh, b * d * 2)
    }

    fn ffn_batch(
        &self,
        layer: &str,
        dh: DevicePtr,
        residual: DevicePtr,
        normed: DevicePtr,
        ff: DevicePtr,
        proj: DevicePtr,
        b: usize,
    ) -> Result<()> {
        self.gpu.copy_d2d(dh, residual, b * self.d * 2)?;
        self.gpu.copy_d2d(dh, normed, b * self.d * 2)?;
        self.layer_norm(&format!("{layer}.final_layer_norm"), normed, b)?;
        self.linear(&format!("{layer}.fc1"), normed, ff, b, self.ffn, self.d)?;
        self.relu(ff, b * self.ffn)?;
        self.linear(&format!("{layer}.fc2"), ff, proj, b, self.d, self.ffn)?;
        self.add(proj, residual, b * self.d)?;
        self.gpu.copy_d2d(proj, dh, b * self.d * 2)
    }

    fn add_row(&self, dst: DevicePtr, row: DevicePtr, n: usize, d: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.add_row)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(row)
            .arg_u32(n as u32)
            .arg_u32(d as u32)
            .launch(self.stream)
    }

    fn scatter(
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

    fn attn_decode(
        &self,
        q: DevicePtr,
        kc: DevicePtr,
        vc: DevicePtr,
        out: DevicePtr,
        b: usize,
        stride: usize,
        tk: DevicePtr,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.attn_decode)
            .grid([(b * self.heads) as u32, 1, 1])
            .block([self.head_dim as u32, 1, 1])
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

    fn lm_head_batch(&self, a: DevicePtr, c: DevicePtr, b: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.lin_no_bias)
            .grid([div_ceil(self.vocab as u32, 16), div_ceil(b as u32, 16), 1])
            .block([16, 16, 1])
            .arg_ptr(a)
            .arg_ptr(self.embed_table)
            .arg_ptr(c)
            .arg_u32(b as u32)
            .arg_u32(self.vocab as u32)
            .arg_u32(self.d as u32)
            .launch(self.stream)
    }

    fn argmax_batch(&self, logits: DevicePtr, out: DevicePtr, b: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.argmax_batched)
            .grid([b as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(out)
            .arg_u32(b as u32)
            .arg_u32(self.vocab as u32)
            .launch(self.stream)
    }
}
