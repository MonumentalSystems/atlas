// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone-7 GPU PoC: **beam-batching** for NLLB-200 on CUDA. The `num_beams`
//! beams of a single source are decoded as a batch (B=num_beams) through one
//! bf16 tensor-core forward per step — M=B GEMM projections, per-beam device KV
//! caches, batched attention. Cross-attention K/V is shared (all beams share the
//! source) and replicated across the B slots. Beam selection reorders the
//! per-beam caches via `nllb_gather_batched` (HF `_reorder_cache`); the
//! BeamHypotheses bookkeeping is host-side (faithful to `BeamSearchScorer`).
//!
//! Validates token-exact vs HF-bf16 beam=5 and reports tok/s vs single-stream.
//!
//! Run:
//!   ATLAS_NLLB_DIR=/tank/hf/nllb-200-3.3B-bf16 \
//!     cargo run --release -p spark-model --example nllb_cuda_beambatch --features gpu-examples

use anyhow::{Context, Result};
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};
use std::path::Path;

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057;
const PAD_ID: u32 = 1;
const MAX_NEW: usize = 96;
const NUM_BEAMS: usize = 5;
const EXPECTED_BEAM5: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 4255, 956, 34821, 248105, 30213, 102506, 248116, 15510,
    385, 2,
];

struct K {
    embed: KernelHandle,
    scale: KernelHandle,
    add: KernelHandle,
    add_row: KernelHandle,
    relu: KernelHandle,
    ln: KernelHandle,
    bias: KernelHandle,
    attn_enc: KernelHandle,
    attn_dec: KernelHandle,
    scatter: KernelHandle,
    gather: KernelHandle,
    gemm: KernelHandle,
}

fn main() -> Result<()> {
    let dir =
        std::env::var("ATLAS_NLLB_DIR").unwrap_or_else(|_| "/tank/hf/nllb-200-3.3B-bf16".into());
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
    let head_dim = d / heads;
    let embed_scale = if cfg["scale_embedding"].as_bool().unwrap_or(true) {
        (d as f32).sqrt()
    } else {
        1.0
    };
    let attn_scale = (head_dim as f32).powf(-0.5);
    let enc_len = INPUT_IDS.len();

    let store: WeightStore = SafetensorsLoader::new().load(Path::new(&dir), gpu, 0)?;
    let kh = K {
        embed: gpu.kernel("nllb_encoder", "nllb_embed_bf16")?,
        scale: gpu.kernel("nllb_encoder", "nllb_scale_bf16")?,
        add: gpu.kernel("nllb_encoder", "nllb_add_bf16")?,
        add_row: gpu.kernel("nllb_encoder", "nllb_add_row_bf16")?,
        relu: gpu.kernel("nllb_encoder", "nllb_relu_bf16")?,
        ln: gpu.kernel("nllb_encoder", "nllb_layernorm_bf16")?,
        bias: gpu.kernel("nllb_encoder", "nllb_bias_bf16")?,
        attn_enc: gpu.kernel("nllb_encoder", "nllb_attn_kv_bf16")?,
        attn_dec: gpu.kernel("nllb_encoder", "nllb_attn_bdecode")?,
        scatter: gpu.kernel("nllb_encoder", "nllb_scatter_batched")?,
        gather: gpu.kernel("nllb_encoder", "nllb_gather_batched")?,
        gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
    };
    let c = Ctx {
        gpu,
        k: &kh,
        store: &store,
        d,
        heads,
        head_dim,
        ffn,
        vocab,
        dec_layers,
        enc_len,
        attn_scale,
        embed_scale,
        embed_table: store.get("model.shared.weight")?.ptr,
        dec_start,
        eos,
        stream,
    };

    // ---- encoder (once) → cross K/V, replicated to B beam slots ----
    let b = NUM_BEAMS;
    let enc_out = c.bf16b(enc_len * d)?;
    let escr = Scratch::new(&c, enc_len)?;
    c.embed_seq(INPUT_IDS, enc_out)?;
    for l in 0..enc_layers {
        let p = format!("model.encoder.layers.{l}");
        c.enc_self_attn(&p, enc_out, enc_len, &escr)?;
        c.ffn_block(&p, enc_out, enc_len, &escr)?;
    }
    c.layer_norm("model.encoder.layer_norm", enc_out, enc_len)?;
    let cross_k: Vec<DevicePtr> = (0..dec_layers)
        .map(|_| c.bf16b(b * enc_len * d))
        .collect::<Result<_>>()?;
    let cross_v: Vec<DevicePtr> = (0..dec_layers)
        .map(|_| c.bf16b(b * enc_len * d))
        .collect::<Result<_>>()?;
    let tmp = c.bf16b(enc_len * d)?;
    for l in 0..dec_layers {
        let p = format!("model.decoder.layers.{l}.encoder_attn");
        c.linear(&format!("{p}.k_proj"), enc_out, tmp, enc_len, d, d)?;
        for bi in 0..b {
            c.gpu.copy_d2d(
                tmp,
                cross_k[l].offset(bi * enc_len * d * 2),
                enc_len * d * 2,
            )?;
        }
        c.linear(&format!("{p}.v_proj"), enc_out, tmp, enc_len, d, d)?;
        for bi in 0..b {
            c.gpu.copy_d2d(
                tmp,
                cross_v[l].offset(bi * enc_len * d * 2),
                enc_len * d * 2,
            )?;
        }
    }

    let t0 = std::time::Instant::now();
    let out = c.beam_batched(b, &cross_k, &cross_v, FORCED_BOS, 1.0)?;
    let dt = t0.elapsed().as_secs_f64();
    println!("[nllb-beambatch] beam={b} ids = {out:?}");
    let pass = out == EXPECTED_BEAM5;
    println!(
        "[nllb-beambatch] beam={b} {} — {} tok in {:.3}s = {:.1} tok/s",
        if pass {
            "PASS (token-exact vs HF-bf16)"
        } else {
            "differs"
        },
        out.len(),
        dt,
        out.len() as f64 / dt
    );
    anyhow::ensure!(pass, "beam-batched output diverged from HF-bf16 reference");
    Ok(())
}

struct Ctx<'a> {
    gpu: &'a dyn GpuBackend,
    k: &'a K,
    store: &'a WeightStore,
    d: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    vocab: usize,
    dec_layers: usize,
    enc_len: usize,
    attn_scale: f32,
    embed_scale: f32,
    embed_table: DevicePtr,
    dec_start: u32,
    eos: u32,
    stream: u64,
}

impl Ctx<'_> {
    fn bf16b(&self, elems: usize) -> Result<DevicePtr> {
        self.gpu.alloc(elems * 2)
    }
    fn u32b(&self, elems: usize) -> Result<DevicePtr> {
        self.gpu.alloc(elems * 4)
    }
    fn w(&self, name: &str) -> Result<DevicePtr> {
        Ok(self.store.get(name)?.ptr)
    }
    fn layer_norm(&self, prefix: &str, x: DevicePtr, rows: usize) -> Result<()> {
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
    fn gemm(
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
    fn linear(
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
    fn add(&self, dst: DevicePtr, src: DevicePtr, n: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.add)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_u32(n as u32)
            .launch(self.stream)
    }
    fn embed_seq(&self, ids: &[u32], out: DevicePtr) -> Result<()> {
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
    fn enc_self_attn(&self, layer: &str, x: DevicePtr, seq: usize, s: &Scratch) -> Result<()> {
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
    fn ffn_block(&self, layer: &str, x: DevicePtr, rows: usize, s: &Scratch) -> Result<()> {
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
    fn forward_step(
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

    fn logits_host(&self, buf: &DecBuf, b: usize) -> Result<Vec<Vec<f32>>> {
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

    fn beam_batched(
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

    fn gather(
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

struct DecBuf {
    dh: DevicePtr,
    residual: DevicePtr,
    normed: DevicePtr,
    q: DevicePtr,
    knew: DevicePtr,
    vnew: DevicePtr,
    attn: DevicePtr,
    proj: DevicePtr,
    ff: DevicePtr,
    logits: DevicePtr,
    id: DevicePtr,
    selftk: DevicePtr,
    crosstk: DevicePtr,
    pos_table: DevicePtr,
}
impl DecBuf {
    fn new(c: &Ctx, b: usize) -> Result<Self> {
        let d = c.d;
        let crosstk = c.u32b(b)?;
        c.gpu
            .copy_h2d(u32_bytes(&vec![c.enc_len as u32; b]), crosstk)?;
        let pos_table = c.bf16b(MAX_NEW * d)?;
        c.gpu
            .copy_h2d(bf16_bytes(&decoder_pos_table_bf16(MAX_NEW, d)), pos_table)?;
        Ok(Self {
            dh: c.bf16b(b * d)?,
            residual: c.bf16b(b * d)?,
            normed: c.bf16b(b * d)?,
            q: c.bf16b(b * d)?,
            knew: c.bf16b(b * d)?,
            vnew: c.bf16b(b * d)?,
            attn: c.bf16b(b * d)?,
            proj: c.bf16b(b * d)?,
            ff: c.bf16b(b * c.ffn)?,
            logits: c.bf16b(b * c.vocab)?,
            id: c.u32b(b)?,
            selftk: c.u32b(b)?,
            crosstk,
            pos_table,
        })
    }
}

struct Beam {
    tokens: Vec<u32>,
    score: f32,
    logits: Vec<f32>,
}
struct BeamHyps {
    num_beams: usize,
    lp: f32,
    beams: Vec<(Vec<u32>, f32)>,
}
impl BeamHyps {
    fn new(num_beams: usize, lp: f32) -> Self {
        Self {
            num_beams,
            lp,
            beams: Vec::new(),
        }
    }
    fn worst(&self) -> f32 {
        self.beams
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::INFINITY, f32::min)
    }
    fn add(&mut self, tokens: Vec<u32>, sum_logprob: f32) {
        let score = sum_logprob / (tokens.len() as f32).powf(self.lp);
        if self.beams.len() < self.num_beams || score > self.worst() {
            self.beams.push((tokens, score));
            if self.beams.len() > self.num_beams {
                let (wi, _) = self
                    .beams
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
                    .unwrap();
                self.beams.swap_remove(wi);
            }
        }
    }
    fn is_done(&self, best_running: f32, cur_len: usize) -> bool {
        if self.beams.len() < self.num_beams {
            return false;
        }
        self.worst() >= best_running / (cur_len as f32).powf(self.lp)
    }
    fn best(&self) -> Option<Vec<u32>> {
        self.beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(t, _)| t.clone())
    }
}

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
    fn new(c: &Ctx, rows: usize) -> Result<Self> {
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

fn logsumexp(x: &[f32]) -> f32 {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m + x.iter().map(|&v| (v - m).exp()).sum::<f32>().ln()
}
fn top_k(x: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, &v) in x.iter().enumerate() {
        if best.len() < k {
            best.push((v, i));
            if best.len() == k {
                best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }
        } else if v > best[k - 1].0 {
            best[k - 1] = (v, i);
            let mut j = k - 1;
            while j > 0 && best[j].0 > best[j - 1].0 {
                best.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    best
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
fn decoder_pos_table_bf16(max_len: usize, d: usize) -> Vec<bf16> {
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
fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn bf16_bytes(v: &[bf16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
