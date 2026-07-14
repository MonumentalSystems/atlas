// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::WeightStore;

const PAD_ID: u32 = 1;

#[allow(dead_code)]
pub struct Kernels {
    pub embed: KernelHandle,
    pub scale: KernelHandle,
    pub add: KernelHandle,
    pub relu: KernelHandle,
    pub ln: KernelHandle,
    pub lin: KernelHandle,
    pub lin_no_bias: KernelHandle,
    pub gemv: KernelHandle,
    pub gemv_no_bias: KernelHandle,
    pub attn: KernelHandle,
    pub add_row: KernelHandle,
    pub scatter: KernelHandle,
    pub attn_decode: KernelHandle,
    pub argmax_batched: KernelHandle,
    pub argmax: KernelHandle,
}

pub struct Ctx<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub k: &'a Kernels,
    pub store: &'a WeightStore,
    pub d: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub dec_layers: usize,
    pub enc_len: usize,
    pub attn_scale: f32,
    pub embed_scale: f32,
    pub embed_table: DevicePtr,
    pub dec_start: u32,
    pub eos: u32,
    pub stream: u64,
}

impl Ctx<'_> {
    pub fn bf16b(&self, elems: usize) -> Result<DevicePtr> {
        self.gpu.alloc(elems * 2)
    }

    fn w(&self, name: &str) -> Result<DevicePtr> {
        Ok(self.store.get(name)?.ptr)
    }

    pub fn layer_norm(&self, prefix: &str, x: DevicePtr, rows: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.ln)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.w(&format!("{prefix}.weight"))?)
            .arg_ptr(self.w(&format!("{prefix}.bias"))?)
            .arg_u32(rows as u32)
            .arg_u32(self.d as u32)
            .arg_f32(1e-5)
            .launch(self.stream)
    }

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

    pub fn linear1(
        &self,
        prefix: &str,
        x: DevicePtr,
        y: DevicePtr,
        n_out: usize,
        k_in: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.gemv)
            .grid([n_out as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.w(&format!("{prefix}.weight"))?)
            .arg_ptr(self.w(&format!("{prefix}.bias"))?)
            .arg_ptr(y)
            .arg_u32(n_out as u32)
            .arg_u32(k_in as u32)
            .launch(self.stream)
    }

    pub fn lm_head(&self, x: DevicePtr, y: DevicePtr) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.gemv_no_bias)
            .grid([self.vocab as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.embed_table)
            .arg_ptr(y)
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

    pub fn relu(&self, x: DevicePtr, n: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.relu)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_u32(n as u32)
            .launch(self.stream)
    }

    pub fn embed_seq(&self, ids: &[u32], out: DevicePtr) -> Result<()> {
        let ids_dev = self.gpu.alloc(ids.len() * 4)?;
        self.gpu.copy_h2d(u32_bytes(ids), ids_dev)?;
        KernelLaunch::new(self.gpu, self.k.embed)
            .grid([ids.len() as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ids_dev)
            .arg_ptr(self.embed_table)
            .arg_ptr(out)
            .arg_u32(self.d as u32)
            .launch(self.stream)?;
        KernelLaunch::new(self.gpu, self.k.scale)
            .grid([div_ceil((ids.len() * self.d) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(out)
            .arg_u32((ids.len() * self.d) as u32)
            .arg_f32(self.embed_scale)
            .launch(self.stream)?;
        let pos = encoder_pos_bf16(ids, self.d, PAD_ID);
        let pos_dev = self.bf16b(ids.len() * self.d)?;
        self.gpu.copy_h2d(bf16_bytes(&pos), pos_dev)?;
        self.add(out, pos_dev, ids.len() * self.d)?;
        self.gpu.free(ids_dev)?;
        self.gpu.free(pos_dev)?;
        Ok(())
    }

    pub fn embed_one(&self, id_dev: DevicePtr, out: DevicePtr) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.embed)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(id_dev)
            .arg_ptr(self.embed_table)
            .arg_ptr(out)
            .arg_u32(self.d as u32)
            .launch(self.stream)?;
        KernelLaunch::new(self.gpu, self.k.scale)
            .grid([div_ceil(self.d as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(out)
            .arg_u32(self.d as u32)
            .arg_f32(self.embed_scale)
            .launch(self.stream)
    }

    pub fn enc_self_attn(&self, layer: &str, x: DevicePtr, seq: usize, s: &Scratch) -> Result<()> {
        let p = format!("{layer}.self_attn");
        self.gpu.copy_d2d(x, s.residual, seq * self.d * 2)?;
        self.gpu.copy_d2d(x, s.normed, seq * self.d * 2)?;
        self.layer_norm(&format!("{layer}.self_attn_layer_norm"), s.normed, seq)?;
        self.linear(&format!("{p}.q_proj"), s.normed, s.q, seq, self.d, self.d)?;
        self.linear(&format!("{p}.k_proj"), s.normed, s.kk, seq, self.d, self.d)?;
        self.linear(&format!("{p}.v_proj"), s.normed, s.v, seq, self.d, self.d)?;
        self.attention(s.q, s.kk, s.v, s.attn, seq, seq, false)?;
        self.linear(
            &format!("{p}.out_proj"),
            s.attn,
            s.proj,
            seq,
            self.d,
            self.d,
        )?;
        self.add(s.proj, s.residual, seq * self.d)?;
        self.gpu.copy_d2d(s.proj, x, seq * self.d * 2)
    }

    pub fn ffn_block(&self, layer: &str, x: DevicePtr, rows: usize, s: &Scratch) -> Result<()> {
        self.gpu.copy_d2d(x, s.residual, rows * self.d * 2)?;
        self.gpu.copy_d2d(x, s.normed, rows * self.d * 2)?;
        self.layer_norm(&format!("{layer}.final_layer_norm"), s.normed, rows)?;
        self.linear(
            &format!("{layer}.fc1"),
            s.normed,
            s.ff,
            rows,
            self.ffn,
            self.d,
        )?;
        self.relu(s.ff, rows * self.ffn)?;
        self.linear(
            &format!("{layer}.fc2"),
            s.ff,
            s.proj,
            rows,
            self.d,
            self.ffn,
        )?;
        self.add(s.proj, s.residual, rows * self.d)?;
        self.gpu.copy_d2d(s.proj, x, rows * self.d * 2)
    }
}

pub struct Scratch {
    pub residual: DevicePtr,
    pub normed: DevicePtr,
    pub q: DevicePtr,
    pub kk: DevicePtr,
    pub v: DevicePtr,
    pub attn: DevicePtr,
    pub proj: DevicePtr,
    pub ff: DevicePtr,
}

impl Scratch {
    pub fn new(c: &Ctx, rows: usize) -> Result<Self> {
        Ok(Self {
            residual: c.bf16b(rows * c.d)?,
            normed: c.bf16b(rows * c.d)?,
            q: c.bf16b(rows * c.d)?,
            kk: c.bf16b(rows * c.d)?,
            v: c.bf16b(rows * c.d)?,
            attn: c.bf16b(rows * c.d)?,
            proj: c.bf16b(rows * c.d)?,
            ff: c.bf16b(rows * c.ffn)?,
        })
    }
}

fn sinusoid_row(pos: f32, d: usize, out: &mut [bf16]) {
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

fn encoder_pos_bf16(ids: &[u32], d: usize, pad: u32) -> Vec<bf16> {
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
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

pub fn bf16_bytes(v: &[bf16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
