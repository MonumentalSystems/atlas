// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of nllb_cuda_bf16.rs for the 500-LoC cap.

use super::*;

impl Ctx<'_> {
    pub fn bf16b(&self, elems: usize) -> Result<DevicePtr> {
        self.gpu.alloc(elems * 2)
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

    /// Tensor-core GEMM: C[M,N] = A[M,K] @ W[N,K]^T (bf16), no bias.
    pub fn gemm(
        &self,
        a: DevicePtr,
        wt: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        kdim: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.gemm)
            .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(wt)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(kdim as u32)
            .launch(self.stream)
    }

    /// Linear with bias: GEMM then row-broadcast bias add.
    pub fn linear(
        &self,
        prefix: &str,
        a: DevicePtr,
        c: DevicePtr,
        rows: usize,
        n_out: usize,
        k_in: usize,
    ) -> Result<()> {
        self.gemm(
            a,
            self.w(&format!("{prefix}.weight"))?,
            c,
            rows,
            n_out,
            k_in,
        )?;
        KernelLaunch::new(self.gpu, self.k.bias)
            .grid([div_ceil((rows * n_out) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c)
            .arg_ptr(self.w(&format!("{prefix}.bias"))?)
            .arg_u32(rows as u32)
            .arg_u32(n_out as u32)
            .launch(self.stream)
    }

    /// Single-token GEMV: y[N] = W[N,K] @ x[K] + bias (bias may be NULL).
    /// Fuses the bias-add; right-sized for M=1 decode.
    pub fn gemv(
        &self,
        x: DevicePtr,
        wt: DevicePtr,
        bias: DevicePtr,
        y: DevicePtr,
        n: usize,
        kdim: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.gemv)
            .grid([div_ceil(n as u32, 8), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(wt)
            .arg_ptr(bias)
            .arg_ptr(y)
            .arg_u32(n as u32)
            .arg_u32(kdim as u32)
            .launch(self.stream)
    }

    /// M=1 biased linear via GEMV (weight/bias named `{prefix}.{weight,bias}`).
    pub fn linear1(
        &self,
        prefix: &str,
        x: DevicePtr,
        y: DevicePtr,
        n_out: usize,
        k_in: usize,
    ) -> Result<()> {
        self.gemv(
            x,
            self.w(&format!("{prefix}.weight"))?,
            self.w(&format!("{prefix}.bias"))?,
            y,
            n_out,
            k_in,
        )
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
        self.attention(s.q, s.kk, s.v, s.attn, seq, seq, false)?;
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
}
