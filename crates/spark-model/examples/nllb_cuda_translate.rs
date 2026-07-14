// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone-2 GPU PoC: full NLLB-200 / M2M-100 **encoder + decoder + cross-
//! attention + greedy** translation on CUDA, using the self-contained fp32
//! kernels in `kernels/gb10/common/nllb_encoder.cu` plus the production
//! `GpuBackend` / `SafetensorsLoader` infra. fp32 throughout → bit-faithful to
//! the `spark-nllb` CPU reference.
//!
//! Validates the greedy token sequence against HF `transformers`:
//!   input "Hello, world. How are you today?" (eng_Latn) → fra_Latn
//!   generated ids = [256057, 17994, 141190, 248079, 25358, 123732, 248105,
//!                    30213, 248079, 1724, 25601, 385, 2]
//!
//! Run:
//!   ATLAS_NLLB_DIR=/tank/hf/nllb-200-3.3B-st \
//!     cargo run --release -p spark-model --example nllb_cuda_translate --features gpu-examples

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};
use std::path::Path;

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057; // fra_Latn
const PAD_ID: u32 = 1;
const EXPECTED_GEN: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 123732, 248105, 30213, 248079, 1724, 25601, 385, 2,
];
const MAX_NEW: usize = 64;

struct Kernels {
    embed: KernelHandle,
    scale: KernelHandle,
    add: KernelHandle,
    relu: KernelHandle,
    ln: KernelHandle,
    lin: KernelHandle,
    attn: KernelHandle,
}

fn main() -> Result<()> {
    let dir =
        std::env::var("ATLAS_NLLB_DIR").unwrap_or_else(|_| "/tank/hf/nllb-200-3.3B-st".into());
    let forced_bos: u32 = std::env::var("ATLAS_NLLB_FORCED_BOS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(FORCED_BOS);

    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.default_stream();

    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        Path::new(&dir).join("config.json"),
    )?)?;
    let d = cfg["d_model"].as_u64().context("d_model")? as usize;
    let heads = cfg["encoder_attention_heads"].as_u64().context("heads")? as usize;
    let ffn = cfg["encoder_ffn_dim"].as_u64().context("ffn")? as usize;
    let enc_layers = cfg["encoder_layers"].as_u64().context("enc_layers")? as usize;
    let dec_layers = cfg["decoder_layers"].as_u64().context("dec_layers")? as usize;
    let vocab = cfg["vocab_size"].as_u64().context("vocab")? as usize;
    let dec_start = cfg["decoder_start_token_id"].as_u64().unwrap_or(2) as u32;
    let eos = cfg["eos_token_id"].as_u64().unwrap_or(2) as u32;
    // This example shares one Ctx (dims) across encoder + decoder, which is
    // only valid when the two stacks match. NLLB/M2M-100 are symmetric; assert
    // it so an asymmetric checkpoint fails loudly instead of mis-shaping.
    let dec_heads = cfg["decoder_attention_heads"]
        .as_u64()
        .unwrap_or(heads as u64) as usize;
    let dec_ffn = cfg["decoder_ffn_dim"].as_u64().unwrap_or(ffn as u64) as usize;
    anyhow::ensure!(
        dec_heads == heads && dec_ffn == ffn,
        "asymmetric enc/dec dims (heads {heads}/{dec_heads}, ffn {ffn}/{dec_ffn}) not supported by this PoC"
    );
    let head_dim = d / heads;
    // Honor scale_embedding (M2M-100/NLLB set it true → sqrt(d_model)).
    let embed_scale = if cfg["scale_embedding"].as_bool().unwrap_or(true) {
        (d as f32).sqrt()
    } else {
        1.0
    };
    let attn_scale = (head_dim as f32).powf(-0.5);
    let enc_len = INPUT_IDS.len();
    println!(
        "[nllb] d={d} heads={heads} head_dim={head_dim} ffn={ffn} enc={enc_layers} dec={dec_layers} vocab={vocab}"
    );

    println!("[nllb] loading weights → GPU ...");
    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;
    let k = Kernels {
        embed: gpu.kernel("nllb_encoder", "nllb_embed")?,
        scale: gpu.kernel("nllb_encoder", "nllb_scale_inplace")?,
        add: gpu.kernel("nllb_encoder", "nllb_add_inplace")?,
        relu: gpu.kernel("nllb_encoder", "nllb_relu_inplace")?,
        ln: gpu.kernel("nllb_encoder", "nllb_layernorm")?,
        lin: gpu.kernel("nllb_encoder", "nllb_linear")?,
        attn: gpu.kernel("nllb_encoder", "nllb_attn_kv")?,
    };
    let embed_table = store.get("model.shared.weight")?.ptr;

    let ctx = Ctx {
        gpu,
        k: &k,
        d,
        heads,
        head_dim,
        ffn,
        attn_scale,
        stream,
    };

    // ---- encoder (once) ----
    let enc_out = ctx.f32b(enc_len * d)?;
    ctx.embed_and_positions(INPUT_IDS, embed_table, embed_scale, enc_out)?;
    let scr = Scratch::new(&ctx, MAX_NEW)?;
    for l in 0..enc_layers {
        let p = format!("model.encoder.layers.{l}");
        ctx.attn_block(
            &store,
            &p,
            "self_attn",
            enc_out,
            enc_len,
            enc_len,
            false,
            &scr,
        )?;
        ctx.ffn_block(&store, &p, enc_out, enc_len, &scr)?;
    }
    ctx.layer_norm(&store, "model.encoder.layer_norm", enc_out, enc_len)?;

    // ---- precompute cross-attention K/V per decoder layer (once) ----
    let mut cross_k = Vec::with_capacity(dec_layers);
    let mut cross_v = Vec::with_capacity(dec_layers);
    for l in 0..dec_layers {
        let p = format!("model.decoder.layers.{l}.encoder_attn");
        let ck = ctx.f32b(enc_len * d)?;
        let cv = ctx.f32b(enc_len * d)?;
        ctx.linear(&store, &format!("{p}.k_proj"), enc_out, ck, enc_len, d, d)?;
        ctx.linear(&store, &format!("{p}.v_proj"), enc_out, cv, enc_len, d, d)?;
        cross_k.push(ck);
        cross_v.push(cv);
    }

    // ---- greedy decode ----
    let dh = ctx.f32b(MAX_NEW * d)?;
    let logits_dev = ctx.f32b(vocab)?;
    let mut logits_host = vec![0u8; vocab * 4];
    let mut dec: Vec<u32> = vec![dec_start];
    let mut generated: Vec<u32> = Vec::new();

    for step in 0..MAX_NEW {
        let l = dec.len();
        ctx.embed_and_positions(&dec, embed_table, embed_scale, dh)?;
        for li in 0..dec_layers {
            let p = format!("model.decoder.layers.{li}");
            ctx.attn_block(&store, &p, "self_attn", dh, l, l, true, &scr)?;
            ctx.cross_attn_block(&store, &p, dh, l, enc_len, cross_k[li], cross_v[li], &scr)?;
            ctx.ffn_block(&store, &p, dh, l, &scr)?;
        }
        ctx.layer_norm(&store, "model.decoder.layer_norm", dh, l)?;

        // lm_head on last position (tied to shared embeddings, no bias)
        let last = dh.offset((l - 1) * d * 4);
        KernelLaunch::new(gpu, k.lin)
            .grid([div_ceil(vocab as u32, 16), 1, 1])
            .block([16, 16, 1])
            .arg_ptr(last)
            .arg_ptr(embed_table)
            .arg_ptr(DevicePtr(0))
            .arg_ptr(logits_dev)
            .arg_u32(1)
            .arg_u32(vocab as u32)
            .arg_u32(d as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(logits_dev, &mut logits_host)?;

        let next = if step == 0 {
            forced_bos
        } else {
            argmax_f32(&logits_host) as u32
        };
        generated.push(next);
        dec.push(next);
        if next == eos || dec.len() >= MAX_NEW {
            break;
        }
    }

    println!("[nllb] generated ids = {generated:?}");
    println!("[nllb] expected  ids = {EXPECTED_GEN:?}");
    if generated == EXPECTED_GEN {
        println!("[nllb] PASS — exact token match vs HF greedy reference");
        Ok(())
    } else {
        anyhow::bail!("FAIL: generated ids diverged from reference")
    }
}

struct Ctx<'a> {
    gpu: &'a dyn GpuBackend,
    k: &'a Kernels,
    d: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    attn_scale: f32,
    stream: u64,
}

/// Per-forward scratch buffers, sized for the longest sequence we process.
struct Scratch {
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
    fn new(ctx: &Ctx, max_len: usize) -> Result<Self> {
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
    fn f32b(&self, elems: usize) -> Result<DevicePtr> {
        self.gpu.alloc(elems * 4)
    }

    fn embed_and_positions(
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

    fn layer_norm(
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
    fn linear(
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
    fn attn_block(
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
    fn cross_attn_block(
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

    fn ffn_block(
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

    fn attention(
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

fn argmax_f32(bytes: &[u8]) -> usize {
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
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn f32_slice(b: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}
