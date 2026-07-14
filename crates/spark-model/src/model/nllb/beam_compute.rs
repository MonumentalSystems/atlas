// SPDX-License-Identifier: AGPL-3.0-only

//! Batched beam-search GPU primitives: the M=B decode step (all B beams of a
//! request forwarded through one bf16 tensor-core GEMM per projection) plus the
//! batched attention / scatter / gather kernels that back it. Promoted from the
//! milestone-7 `nllb_cuda_beambatch` example. Reuses the multi-row `linear`
//! (which also applies per-request LoRA at M=B), `layer_norm`, `add`, `relu`,
//! `scale`, `gemm` and `embed_rows` from [`super::compute`].

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::NllbGpuModel;
use super::beam::DecBuf;
use super::util::u32_bytes;

impl NllbGpuModel {
    /// Write `src[B,d]` into batch-major cache `[B,stride,d]` at row `pos`.
    fn scatter(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        pos: usize,
        b: usize,
        stride: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.scatter)
            .grid([div_ceil((b * self.d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(pos as u32)
            .arg_u32(b as u32)
            .arg_u32(stride as u32)
            .arg_u32(self.d as u32)
            .launch(self.stream())
    }

    /// Batched attention over B beams; `tk` holds each beam's key length. When
    /// `stride == 0` all beams read the SAME `kc`/`vc` (shared cross-KV).
    #[allow(clippy::too_many_arguments)]
    fn attn_batched(
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
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.attn_bdecode)
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
            .launch(self.stream())
    }

    /// Reorder beam caches: `dst[i] = src[perm[i]]` over rows `0..used`.
    pub(super) fn gather(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        perm: DevicePtr,
        b: usize,
        used: usize,
        stride: usize,
    ) -> Result<()> {
        let n = (b * used * self.d) as u32;
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.gather)
            .grid([div_ceil(n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_ptr(perm)
            .arg_u32(b as u32)
            .arg_u32(used as u32)
            .arg_u32(stride as u32)
            .arg_u32(self.d as u32)
            .launch(self.stream())
    }

    /// One batched decode step: fill self-KV row `pos` for all B beams and write
    /// `logits[B, vocab]` (bf16). `cur` = one token per beam. `cross_k`/`cross_v`
    /// are single `[enc_len,d]` buffers shared across beams (stride 0).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn beam_forward_step(
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
        let sh = ((self.head_dim + buf.cache_rows) * 4) as u32;
        self.gpu.copy_h2d(u32_bytes(cur), buf.id)?;
        self.embed_rows(buf.id, buf.dh, b)?;
        self.scale(buf.dh, b * d)?;
        // add the position row (broadcast across the B batch)
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.add_row)
            .grid([div_ceil((b * d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(buf.dh)
            .arg_ptr(buf.pos_table.offset(pos * d * 2))
            .arg_u32((b * d) as u32)
            .arg_u32(d as u32)
            .launch(self.stream())?;
        self.gpu
            .copy_h2d(u32_bytes(&vec![(pos + 1) as u32; b]), buf.selftk)?;

        for l in 0..self.dec_layers {
            let p = format!("model.decoder.layers.{l}");
            // causal self-attention (batched, per-beam length via selftk)
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
            self.scatter(buf.knew, sk[l], pos, b, buf.cache_rows)?;
            self.scatter(buf.vnew, sv[l], pos, b, buf.cache_rows)?;
            self.attn_batched(
                buf.q,
                sk[l],
                sv[l],
                buf.attn,
                b,
                buf.cache_rows,
                buf.selftk,
                sh,
            )?;
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
            // cross-attention (shared cross-KV, stride 0)
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
                0,
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
            // FFN
            self.gpu.copy_d2d(buf.dh, buf.residual, b * d * 2)?;
            self.gpu.copy_d2d(buf.dh, buf.normed, b * d * 2)?;
            self.layer_norm(&format!("{p}.final_layer_norm"), buf.normed, b)?;
            self.linear(&format!("{p}.fc1"), buf.normed, buf.ff, b, ffn, d)?;
            self.relu(buf.ff, b * ffn)?;
            self.linear(&format!("{p}.fc2"), buf.ff, buf.proj, b, d, ffn)?;
            self.add(buf.proj, buf.residual, b * d)?;
            self.gpu.copy_d2d(buf.proj, buf.dh, b * d * 2)?;
        }
        self.layer_norm("model.decoder.layer_norm", buf.dh, b)?;
        self.gemm(buf.dh, self.embed_table, buf.logits, b, self.vocab, d)?; // tied lm_head
        Ok(())
    }

    /// Copy the device `[B,vocab]` bf16 logits to host as per-beam `f32` rows.
    pub(super) fn beam_logits_host(&self, buf: &DecBuf, b: usize) -> Result<Vec<Vec<f32>>> {
        self.gpu.synchronize(self.stream())?;
        let mut raw = vec![0u8; b * self.vocab * 2];
        self.gpu.copy_d2h(buf.logits, &mut raw)?;
        Ok((0..b)
            .map(|bi| {
                raw[bi * self.vocab * 2..(bi + 1) * self.vocab * 2]
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect()
            })
            .collect())
    }
}
