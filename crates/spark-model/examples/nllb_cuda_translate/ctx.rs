// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of nllb_cuda_translate.rs for the 500-LoC cap.

use super::*;

impl Scratch {
    pub fn new(ctx: &Ctx, max_len: usize) -> Result<Self> {
        let d = ctx.d;
        Ok(Self {
            residual: ctx.f32b(max_len * d)?,
            normed: ctx.f32b(max_len * d)?,
            q: ctx.f32b(max_len * d)?,
            kk: ctx.f32b(max_len * d)?,
            v: ctx.f32b(max_len * d)?,
            attn: ctx.f32b(max_len * d)?,
            proj: ctx.f32b(max_len * d)?,
            ff: ctx.f32b(max_len * ctx.ffn)?,
        })
    }
}

impl Ctx<'_> {
    pub fn f32b(&self, elems: usize) -> Result<DevicePtr> {
        self.gpu.alloc(elems * 4)
    }

    pub fn embed_and_positions(
        &self,
        ids: &[u32],
        table: DevicePtr,
        scale: f32,
        out: DevicePtr,
    ) -> Result<()> {
        let (d, seq) = (self.d, ids.len());
        let ids_dev = self.gpu.alloc(seq * 4)?;
        self.gpu.copy_h2d(u32_bytes(ids), ids_dev)?;
        KernelLaunch::new(self.gpu, self.k.embed)
            .grid([seq as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ids_dev)
            .arg_ptr(table)
            .arg_ptr(out)
            .arg_u32(d as u32)
            .launch(self.stream)?;
        self.launch_1d(self.k.scale, seq * d, |kl| {
            kl.arg_ptr(out).arg_u32((seq * d) as u32).arg_f32(scale)
        })?;
        let pos = sinusoid_positions(ids, d, PAD_ID);
        let pos_dev = self.f32b(seq * d)?;
        self.gpu.copy_h2d(f32_bytes(&pos), pos_dev)?;
        self.launch_1d(self.k.add, seq * d, |kl| {
            kl.arg_ptr(out).arg_ptr(pos_dev).arg_u32((seq * d) as u32)
        })?;
        self.gpu.free(ids_dev)?;
        self.gpu.free(pos_dev)?;
        Ok(())
    }

    pub fn layer_norm(
        &self,
        store: &WeightStore,
        prefix: &str,
        x: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        let (wn, bn) = (
            store.get(&format!("{prefix}.weight"))?.ptr,
            store.get(&format!("{prefix}.bias"))?.ptr,
        );
        KernelLaunch::new(self.gpu, self.k.ln)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .shared_mem(256 * 4)
            .arg_ptr(x)
            .arg_ptr(wn)
            .arg_ptr(bn)
            .arg_u32(rows as u32)
            .arg_u32(self.d as u32)
            .arg_f32(1e-5)
            .launch(self.stream)
    }

    /// C[rows,n_out] = A[rows,k_in] @ W[n_out,k_in]^T + bias, weight/bias named `{prefix}.{weight,bias}`.
    pub fn linear(
        &self,
        store: &WeightStore,
        prefix: &str,
        a: DevicePtr,
        c: DevicePtr,
        rows: usize,
        n_out: usize,
        k_in: usize,
    ) -> Result<()> {
        let wt = store.get(&format!("{prefix}.weight"))?.ptr;
        let bias = store.get(&format!("{prefix}.bias"))?.ptr;
        KernelLaunch::new(self.gpu, self.k.lin)
            .grid([div_ceil(n_out as u32, 16), div_ceil(rows as u32, 16), 1])
            .block([16, 16, 1])
            .arg_ptr(a)
            .arg_ptr(wt)
            .arg_ptr(bias)
            .arg_ptr(c)
            .arg_u32(rows as u32)
            .arg_u32(n_out as u32)
            .arg_u32(k_in as u32)
            .launch(self.stream)
    }

    /// Pre-norm self/cross attention over `x` in place. `sub` is "self_attn" or
    /// "encoder_attn"; K/V come from `x` (self) — see `cross_attn_block` for cross.
    pub fn attn_block(
        &self,
        store: &WeightStore,
        layer: &str,
        sub: &str,
        x: DevicePtr,
        tq: usize,
        tk: usize,
        causal: bool,
        s: &Scratch,
    ) -> Result<()> {
        let (d, bytes) = (self.d, tq * self.d * 4);
        let p = format!("{layer}.{sub}");
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(store, &format!("{layer}.{sub}_layer_norm"), s.normed, tq)?;
        self.linear(store, &format!("{p}.q_proj"), s.normed, s.q, tq, d, d)?;
        self.linear(store, &format!("{p}.k_proj"), s.normed, s.kk, tk, d, d)?;
        self.linear(store, &format!("{p}.v_proj"), s.normed, s.v, tk, d, d)?;
        self.attention(s.q, s.kk, s.v, s.attn, tq, tk, causal)?;
        self.linear(store, &format!("{p}.out_proj"), s.attn, s.proj, tq, d, d)?;
        self.launch_1d(self.k.add, tq * d, |kl| {
            kl.arg_ptr(s.proj)
                .arg_ptr(s.residual)
                .arg_u32((tq * d) as u32)
        })?;
        self.gpu.copy_d2d(s.proj, x, bytes)
    }

    /// Pre-norm cross-attention: Q from decoder `x`, K/V precomputed from encoder.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_attn_block(
        &self,
        store: &WeightStore,
        layer: &str,
        x: DevicePtr,
        tq: usize,
        tk: usize,
        ck: DevicePtr,
        cv: DevicePtr,
        s: &Scratch,
    ) -> Result<()> {
        let (d, bytes) = (self.d, tq * self.d * 4);
        let p = format!("{layer}.encoder_attn");
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(
            store,
            &format!("{layer}.encoder_attn_layer_norm"),
            s.normed,
            tq,
        )?;
        self.linear(store, &format!("{p}.q_proj"), s.normed, s.q, tq, d, d)?;
        self.attention(s.q, ck, cv, s.attn, tq, tk, false)?;
        self.linear(store, &format!("{p}.out_proj"), s.attn, s.proj, tq, d, d)?;
        self.launch_1d(self.k.add, tq * d, |kl| {
            kl.arg_ptr(s.proj)
                .arg_ptr(s.residual)
                .arg_u32((tq * d) as u32)
        })?;
        self.gpu.copy_d2d(s.proj, x, bytes)
    }

    pub fn ffn_block(
        &self,
        store: &WeightStore,
        layer: &str,
        x: DevicePtr,
        rows: usize,
        s: &Scratch,
    ) -> Result<()> {
        let (d, ffn, bytes) = (self.d, self.ffn, rows * self.d * 4);
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(store, &format!("{layer}.final_layer_norm"), s.normed, rows)?;
        self.linear(store, &format!("{layer}.fc1"), s.normed, s.ff, rows, ffn, d)?;
        self.launch_1d(self.k.relu, rows * ffn, |kl| {
            kl.arg_ptr(s.ff).arg_u32((rows * ffn) as u32)
        })?;
        self.linear(store, &format!("{layer}.fc2"), s.ff, s.proj, rows, d, ffn)?;
        self.launch_1d(self.k.add, rows * d, |kl| {
            kl.arg_ptr(s.proj)
                .arg_ptr(s.residual)
                .arg_u32((rows * d) as u32)
        })?;
        self.gpu.copy_d2d(s.proj, x, bytes)
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

    pub fn launch_1d(
        &self,
        kernel: KernelHandle,
        n: usize,
        args: impl FnOnce(KernelLaunch) -> KernelLaunch,
    ) -> Result<()> {
        let kl = KernelLaunch::new(self.gpu, kernel)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1]);
        args(kl).launch(self.stream)
    }
}

pub fn sinusoid_positions(ids: &[u32], d: usize, pad: u32) -> Vec<f32> {
    let (seq, half) = (ids.len(), d / 2);
    let emb_scale = 10000f32.ln() / (half as f32 - 1.0);
    let mut pos = vec![0f32; seq * d];
    let mut running = 0u32;
    for (i, &id) in ids.iter().enumerate() {
        let p = if id != pad {
            running += 1;
            running + pad
        } else {
            pad
        };
        if p == pad {
            continue;
        }
        for j in 0..half {
            let ang = p as f32 * (-(j as f32) * emb_scale).exp();
            pos[i * d + j] = ang.sin();
            pos[i * d + half + j] = ang.cos();
        }
    }
    pos
}

pub fn argmax_f32(bytes: &[u8]) -> usize {
    let logits = f32_slice(bytes);
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

pub fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn f32_slice(b: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}
