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
use super::beam::{CrossGroup, DecBuf};
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

    /// One batched decode step over `b` rows (Σ beams of all fused requests):
    /// fill self-KV row `pos` for every row and write `logits[b, vocab]` (bf16).
    /// `cur` = one token per row. Self-attention / projections / FFN / lm_head are
    /// single M=b GEMMs; cross-attention loops the per-request `cross` groups, each
    /// reading its own `[enc_len,d]` buffer shared across its beams (stride 0).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn beam_forward_step(
        &self,
        cur: &[u32],
        pos: usize,
        b: usize,
        sk: &[DevicePtr],
        sv: &[DevicePtr],
        cross: &[CrossGroup],
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
            // Per-request cross-attention: each group reads its own [enc_len,d]
            // cross-KV, shared across its beams (stride 0), into its batch rows.
            for g in cross {
                let sh_cross = ((self.head_dim + g.enc_len) * 4) as u32;
                self.attn_batched(
                    buf.q.offset(g.row_off * d * 2),
                    g.k[l],
                    g.v[l],
                    buf.attn.offset(g.row_off * d * 2),
                    g.n_rows,
                    0,
                    buf.crosstk.offset(g.row_off * 4),
                    sh_cross,
                )?;
            }
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

    /// Phase-d on-device candidate reduction: for each of `rows` logit rows,
    /// return its log-sum-exp over the full vocab and its top-`k` `(value, token)`
    /// pairs (descending, ties by lower id) — the same expansion inputs the host
    /// path derives from a full-vocab D2H, but the D2H shrinks from
    /// `rows*vocab*2` to `rows*k*8` bytes. `k` must be ≤ `NLLB_TOPK_KMAX`.
    pub(super) fn beam_cands_device(
        &self,
        buf: &DecBuf,
        rows: usize,
        k: usize,
    ) -> Result<Vec<(f32, Vec<(f32, u32)>)>> {
        let sh = (128 * k * 8) as u32; // 128 threads · k · (f32 val + u32 id)
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.beam_topk)
            .grid([rows as u32, 1, 1])
            .block([128, 1, 1])
            .shared_mem(sh)
            .arg_ptr(buf.logits)
            .arg_ptr(buf.topk_lse)
            .arg_ptr(buf.topk_val)
            .arg_ptr(buf.topk_idx)
            .arg_u32(rows as u32)
            .arg_u32(self.vocab as u32)
            .arg_u32(k as u32)
            .launch(self.stream())?;
        self.gpu.synchronize(self.stream())?;
        // The kernel packs outputs as [rows, k] (stride k), so the first
        // rows*k elements are contiguous.
        let (mut lse, mut val, mut idx) = (
            vec![0u8; rows * 4],
            vec![0u8; rows * k * 4],
            vec![0u8; rows * k * 4],
        );
        self.gpu.copy_d2h(buf.topk_lse, &mut lse)?;
        self.gpu.copy_d2h(buf.topk_val, &mut val)?;
        self.gpu.copy_d2h(buf.topk_idx, &mut idx)?;
        let f32_at =
            |b: &[u8], i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        let u32_at =
            |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        Ok((0..rows)
            .map(|r| {
                let top = (0..k)
                    .map(|j| (f32_at(&val, r * k + j), u32_at(&idx, r * k + j)))
                    .collect();
                (f32_at(&lse, r), top)
            })
            .collect())
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
