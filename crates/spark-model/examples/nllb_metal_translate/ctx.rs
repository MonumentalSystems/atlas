// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::WeightStore;

const PAD_ID: u32 = 1;

pub struct Kernels {
    embed: KernelHandle,
    scale: KernelHandle,
    add: KernelHandle,
    relu: KernelHandle,
    ln: KernelHandle,
    lin: KernelHandle,
    lin_no_bias: KernelHandle,
    attn: KernelHandle,
}

impl Kernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            embed: gpu.kernel("nllb_encoder", "nllb_embed")?,
            scale: gpu.kernel("nllb_encoder", "nllb_scale_inplace")?,
            add: gpu.kernel("nllb_encoder", "nllb_add_inplace")?,
            relu: gpu.kernel("nllb_encoder", "nllb_relu_inplace")?,
            ln: gpu.kernel("nllb_encoder", "nllb_layernorm")?,
            lin: gpu.kernel("nllb_encoder", "nllb_linear")?,
            lin_no_bias: gpu.kernel("nllb_encoder", "nllb_linear_no_bias")?,
            attn: gpu.kernel("nllb_encoder", "nllb_attn_kv")?,
        })
    }
}

pub struct Ctx<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub k: &'a Kernels,
    pub d: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub attn_scale: f32,
    pub stream: u64,
}

pub struct Scratch {
    residual: DevicePtr,
    normed: DevicePtr,
    q: DevicePtr,
    kk: DevicePtr,
    v: DevicePtr,
    attn: DevicePtr,
    proj: DevicePtr,
    ff: DevicePtr,
}

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
        let wn = store.get(&format!("{prefix}.weight"))?.ptr;
        let bn = store.get(&format!("{prefix}.bias"))?.ptr;
        KernelLaunch::new(self.gpu, self.k.ln)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(wn)
            .arg_ptr(bn)
            .arg_u32(rows as u32)
            .arg_u32(self.d as u32)
            .arg_f32(1e-5)
            .launch(self.stream)
    }

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

    pub fn linear_no_bias_raw(
        &self,
        a: DevicePtr,
        wt: DevicePtr,
        c: DevicePtr,
        rows: usize,
        n_out: usize,
        k_in: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.lin_no_bias)
            .grid([div_ceil(n_out as u32, 16), div_ceil(rows as u32, 16), 1])
            .block([16, 16, 1])
            .arg_ptr(a)
            .arg_ptr(wt)
            .arg_ptr(c)
            .arg_u32(rows as u32)
            .arg_u32(n_out as u32)
            .arg_u32(k_in as u32)
            .launch(self.stream)
    }

    #[allow(clippy::too_many_arguments)]
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
        let bytes = tq * self.d * 4;
        let p = format!("{layer}.{sub}");
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(store, &format!("{layer}.{sub}_layer_norm"), s.normed, tq)?;
        self.linear(
            store,
            &format!("{p}.q_proj"),
            s.normed,
            s.q,
            tq,
            self.d,
            self.d,
        )?;
        self.linear(
            store,
            &format!("{p}.k_proj"),
            s.normed,
            s.kk,
            tk,
            self.d,
            self.d,
        )?;
        self.linear(
            store,
            &format!("{p}.v_proj"),
            s.normed,
            s.v,
            tk,
            self.d,
            self.d,
        )?;
        self.attention(s.q, s.kk, s.v, s.attn, tq, tk, causal)?;
        self.linear(
            store,
            &format!("{p}.out_proj"),
            s.attn,
            s.proj,
            tq,
            self.d,
            self.d,
        )?;
        self.launch_1d(self.k.add, tq * self.d, |kl| {
            kl.arg_ptr(s.proj)
                .arg_ptr(s.residual)
                .arg_u32((tq * self.d) as u32)
        })?;
        self.gpu.copy_d2d(s.proj, x, bytes)
    }

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
        let bytes = tq * self.d * 4;
        let p = format!("{layer}.encoder_attn");
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(
            store,
            &format!("{layer}.encoder_attn_layer_norm"),
            s.normed,
            tq,
        )?;
        self.linear(
            store,
            &format!("{p}.q_proj"),
            s.normed,
            s.q,
            tq,
            self.d,
            self.d,
        )?;
        self.attention(s.q, ck, cv, s.attn, tq, tk, false)?;
        self.linear(
            store,
            &format!("{p}.out_proj"),
            s.attn,
            s.proj,
            tq,
            self.d,
            self.d,
        )?;
        self.launch_1d(self.k.add, tq * self.d, |kl| {
            kl.arg_ptr(s.proj)
                .arg_ptr(s.residual)
                .arg_u32((tq * self.d) as u32)
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
        let bytes = rows * self.d * 4;
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(store, &format!("{layer}.final_layer_norm"), s.normed, rows)?;
        self.linear(
            store,
            &format!("{layer}.fc1"),
            s.normed,
            s.ff,
            rows,
            self.ffn,
            self.d,
        )?;
        self.launch_1d(self.k.relu, rows * self.ffn, |kl| {
            kl.arg_ptr(s.ff).arg_u32((rows * self.ffn) as u32)
        })?;
        self.linear(
            store,
            &format!("{layer}.fc2"),
            s.ff,
            s.proj,
            rows,
            self.d,
            self.ffn,
        )?;
        self.launch_1d(self.k.add, rows * self.d, |kl| {
            kl.arg_ptr(s.proj)
                .arg_ptr(s.residual)
                .arg_u32((rows * self.d) as u32)
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
        self.launch_1d(self.k.add, n, |kl| {
            kl.arg_ptr(dst).arg_ptr(src).arg_u32(n as u32)
        })
    }

    pub fn scale(&self, x: DevicePtr, n: usize, scale: f32) -> Result<()> {
        self.launch_1d(self.k.scale, n, |kl| {
            kl.arg_ptr(x).arg_u32(n as u32).arg_f32(scale)
        })
    }

    pub fn launch_relu(&self, x: DevicePtr, n: usize) -> Result<()> {
        self.launch_1d(self.k.relu, n, |kl| kl.arg_ptr(x).arg_u32(n as u32))
    }

    pub fn embed_raw(
        &self,
        ids: DevicePtr,
        table: DevicePtr,
        out: DevicePtr,
        seq: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.embed)
            .grid([seq as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ids)
            .arg_ptr(table)
            .arg_ptr(out)
            .arg_u32(self.d as u32)
            .launch(self.stream)
    }

    fn launch_1d(
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

fn sinusoid_positions(ids: &[u32], d: usize, pad: u32) -> Vec<f32> {
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

fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn f32_slice(b: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len() / 4) }
}
