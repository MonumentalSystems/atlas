// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.5/3.6 (GDN-hybrid, `general.architecture = qwen35`) GGUF value
//! transforms.
//!
//! llama.cpp's `qwen35` converter stores three families of GDN / norm tensors
//! with a *different encoding* than the HF checkpoint Atlas's kernels expect.
//! The name map ([`super::names`]) already lands each tensor under its HF name;
//! this module fixes the VALUES (never re-quantizing — every transform runs on
//! the dequantized F32 host values, then rounds to BF16 once at the end, so
//! quant block boundaries are irrelevant and the near-1.0 norm offset keeps full
//! F32 precision). Gated on the `qwen35` arch by the caller; other archs never
//! see it.
//!
//! Three inverse transforms, diagnosed byte-for-byte against
//! `nvidia/Qwen3.6-27B-NVFP4`:
//!
//! 1. **RMSNorm +1 offset.** llama.cpp stores the raw norm weight; Atlas
//!    computes `x*(1+w)`, so it needs `w_hf = w_gguf - 1`. Applied to
//!    `input_layernorm`, `post_attention_layernorm`, `self_attn.q_norm`,
//!    `self_attn.k_norm` and the final `model.norm`. NOT the GDN `linear_attn.
//!    norm` (that one matches the reference untouched).
//!
//! 2. **SSM A recovery.** llama.cpp stores `ssm_a = -exp(A_log)`; Atlas's GDN
//!    wants the raw `A_log`, so `A_log = ln(-ssm_a)` (element-wise; `ssm_a` is
//!    negative). Applied to `linear_attn.A_log`.
//!
//! 3. **GDN value-head reorder.** llama.cpp's `_LinearAttentionVReorderBase`
//!    (conversion/qwen.py, `_reorder_v_heads`) reshapes the HF value-head axis
//!    `(num_k_heads, num_v_per_k, head_dim)` and swaps the first two dims to a
//!    tiled `(num_v_per_k, num_k_heads, head_dim)` order (so ggml can use a cheap
//!    broadcast repeat). We apply the INVERSE — gather HF head `i` from GGUF head
//!    `perm(i)` — to `dt_bias`, `A_log`, the V rows of `in_proj_qkv`, all rows of
//!    `in_proj_z` / `in_proj_a` / `in_proj_b`, the V channels of `conv1d`, and
//!    the V columns of `out_proj`.

use anyhow::{Result, ensure};

use super::container::GgufFile;

/// GDN head geometry, read from the GGUF `*.ssm.*` metadata keys.
#[derive(Clone, Copy, Debug)]
pub struct GdnDims {
    /// Linear-attention key heads (`ssm.group_count`).
    pub num_k_heads: usize,
    /// Linear-attention value heads (`ssm.time_step_rank`).
    pub num_v_heads: usize,
    /// Per-value-head dimension (`ssm.inner_size / num_v_heads`).
    pub value_head_dim: usize,
    /// Per-key-head dimension (`ssm.state_size`); sizes the Q|K partitions of
    /// the fused `in_proj_qkv` / `conv1d` so their V region can be located.
    pub key_head_dim: usize,
}

impl GdnDims {
    /// Value heads per key head (the reorder's inner group factor).
    fn num_v_per_k(&self) -> usize {
        self.num_v_heads / self.num_k_heads
    }

    /// Rows spanned by the fused Q and K partitions of `in_proj_qkv` / channels
    /// of `conv1d` (2 × key heads × key-head-dim). The V region follows.
    fn qk_rows(&self) -> usize {
        2 * self.key_head_dim * self.num_k_heads
    }

    /// Source GGUF value-head index for HF value-head `hf`. HF stores heads
    /// grouped by key head (`hf = k*num_v_per_k + r`); GGUF stores them tiled
    /// (`gguf = r*num_k_heads + k`). This inverts the converter's transpose.
    fn gguf_head(&self, hf: usize) -> usize {
        let vpk = self.num_v_per_k();
        let k = hf / vpk;
        let r = hf % vpk;
        r * self.num_k_heads + k
    }
}

/// Read GDN head geometry from GGUF metadata for architecture `arch`
/// (e.g. `"qwen35"`). Returns `None` if the SSM keys are absent or inconsistent.
pub fn gdn_dims(gguf: &GgufFile, arch: &str) -> Option<GdnDims> {
    let g = |suffix: &str| gguf.get_u64(&format!("{arch}.{suffix}"));
    let num_k_heads = g("ssm.group_count")? as usize;
    let num_v_heads = g("ssm.time_step_rank")? as usize;
    let inner_size = g("ssm.inner_size")? as usize;
    let key_head_dim = g("ssm.state_size")? as usize;
    if num_k_heads == 0
        || num_v_heads == 0
        || !num_v_heads.is_multiple_of(num_k_heads)
        || inner_size == 0
        || !inner_size.is_multiple_of(num_v_heads)
    {
        return None;
    }
    Some(GdnDims {
        num_k_heads,
        num_v_heads,
        value_head_dim: inner_size / num_v_heads,
        key_head_dim,
    })
}

/// True if `arch` is a Qwen3.5/3.6 GDN-hybrid architecture whose GGUF encoding
/// needs the value transforms in this module.
pub fn is_qwen35(arch: &str) -> bool {
    matches!(arch, "qwen35" | "qwen35moe" | "qwen3_5")
}

/// The per-tensor transform selected by (mapped) HF name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// Subtract 1.0 from every element (RMSNorm +1 offset).
    NormOffset,
    /// `A_log = ln(-ssm_a)` then value-head reorder along dim 0.
    ALog,
    /// Value-head reorder along the row axis. `after_qk` = the reordered region
    /// starts after the fused Q|K partition (`in_proj_qkv` / `conv1d`);
    /// otherwise the whole tensor is value heads. `head_dim_rows` = rows per
    /// value head (`false` → one, `true` → `value_head_dim`).
    ReorderRows {
        after_qk: bool,
        head_dim_rows: bool,
    },
    /// Value-head reorder along the column (input) axis of `out_proj`.
    ReorderOutCols,
}

/// Classify a mapped HF tensor name into its transform, or `None` if untouched.
fn classify(hf: &str) -> Option<Op> {
    if hf == "model.norm.weight"
        || hf.ends_with(".input_layernorm.weight")
        || hf.ends_with(".post_attention_layernorm.weight")
        || hf.ends_with(".self_attn.q_norm.weight")
        || hf.ends_with(".self_attn.k_norm.weight")
    {
        return Some(Op::NormOffset);
    }
    if hf.ends_with(".linear_attn.A_log") {
        return Some(Op::ALog);
    }
    if hf.ends_with(".linear_attn.dt_bias")
        || hf.ends_with(".linear_attn.in_proj_a.weight")
        || hf.ends_with(".linear_attn.in_proj_b.weight")
    {
        return Some(Op::ReorderRows {
            after_qk: false,
            head_dim_rows: false,
        });
    }
    if hf.ends_with(".linear_attn.in_proj_z.weight") {
        return Some(Op::ReorderRows {
            after_qk: false,
            head_dim_rows: true,
        });
    }
    if hf.ends_with(".linear_attn.in_proj_qkv.weight")
        || hf.ends_with(".linear_attn.conv1d.weight")
    {
        return Some(Op::ReorderRows {
            after_qk: true,
            head_dim_rows: true,
        });
    }
    if hf.ends_with(".linear_attn.out_proj.weight") {
        return Some(Op::ReorderOutCols);
    }
    None
}

/// True if the loader must route this tensor through the CPU dequant + transform
/// path (i.e. its VALUES need fixing for the `qwen35` arch).
pub fn needs(hf_name: &str) -> bool {
    classify(hf_name).is_some()
}

/// Apply the qwen35 value transform (if any) to the dequantized F32 values
/// `buf` in place. `hf_shape` is the tensor's HF (row-major) shape. No-op for
/// names [`classify`] does not recognize.
pub fn apply(hf_name: &str, buf: &mut [f32], hf_shape: &[usize], dims: &GdnDims) -> Result<()> {
    let Some(op) = classify(hf_name) else {
        return Ok(());
    };
    match op {
        Op::NormOffset => buf.iter_mut().for_each(|x| *x -= 1.0),
        Op::ALog => {
            buf.iter_mut().for_each(|x| *x = (-*x).ln());
            reorder_rows(buf, hf_name, hf_shape, dims, false, false)?;
        }
        Op::ReorderRows {
            after_qk,
            head_dim_rows,
        } => reorder_rows(buf, hf_name, hf_shape, dims, after_qk, head_dim_rows)?,
        Op::ReorderOutCols => reorder_out_cols(buf, hf_name, hf_shape, dims)?,
    }
    Ok(())
}

/// Round an F32 slice to little-endian BF16 host bytes (the loader's upload
/// form). Round-to-nearest-even, NaN quieted.
pub fn to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    out
}

/// Elements produced by one row of a row-major tensor of shape `hf_shape` (the
/// product of every dim after the first; 1 for a 1-D tensor).
fn row_width(hf_shape: &[usize]) -> usize {
    hf_shape.iter().skip(1).product::<usize>().max(1)
}

/// Reorder value-head blocks along the outer (row) axis from GGUF tiled order to
/// HF grouped order. `after_qk` = the reordered region starts after the fused
/// Q|K partition (`in_proj_qkv` / `conv1d`); otherwise the whole tensor is value
/// heads. `head_dim_rows` = each value head spans `value_head_dim` rows (else 1).
fn reorder_rows(
    buf: &mut [f32],
    hf_name: &str,
    hf_shape: &[usize],
    dims: &GdnDims,
    after_qk: bool,
    head_dim_rows: bool,
) -> Result<()> {
    let rw = row_width(hf_shape);
    let hd = if head_dim_rows { dims.value_head_dim } else { 1 };
    let base = if after_qk { dims.qk_rows() * rw } else { 0 };
    let blk = hd * rw; // elements per value head
    let region = dims.num_v_heads * blk;
    ensure!(
        base + region <= buf.len(),
        "qwen35 reorder {hf_name}: V region [{base}..{}] exceeds {} elems",
        base + region,
        buf.len()
    );
    let src = buf[base..base + region].to_vec();
    for hf in 0..dims.num_v_heads {
        let g = dims.gguf_head(hf);
        let (d, s) = (base + hf * blk, g * blk);
        buf[d..d + blk].copy_from_slice(&src[s..s + blk]);
    }
    Ok(())
}

/// Reorder value-head column blocks (input dim) of `out_proj`, per output row.
/// `hf_shape = [out_dim, value_dim]`, `value_dim = num_v_heads * value_head_dim`.
fn reorder_out_cols(
    buf: &mut [f32],
    hf_name: &str,
    hf_shape: &[usize],
    dims: &GdnDims,
) -> Result<()> {
    ensure!(
        hf_shape.len() == 2,
        "qwen35 reorder {hf_name}: out_proj expects 2-D shape, got {hf_shape:?}"
    );
    let out_rows = hf_shape[0];
    let value_dim = hf_shape[1];
    ensure!(
        value_dim == dims.num_v_heads * dims.value_head_dim,
        "qwen35 reorder {hf_name}: value_dim {value_dim} != num_v_heads*value_head_dim {}",
        dims.num_v_heads * dims.value_head_dim
    );
    let cb = dims.value_head_dim; // elements per value-head column block
    ensure!(
        out_rows * value_dim <= buf.len(),
        "qwen35 reorder {hf_name}: {out_rows}x{value_dim} exceeds {} elems",
        buf.len()
    );
    let mut tmp = vec![0f32; value_dim];
    for row in 0..out_rows {
        let ro = row * value_dim;
        tmp.copy_from_slice(&buf[ro..ro + value_dim]);
        for hf in 0..dims.num_v_heads {
            let g = dims.gguf_head(hf);
            buf[ro + hf * cb..ro + hf * cb + cb].copy_from_slice(&tmp[g * cb..g * cb + cb]);
        }
    }
    Ok(())
}

/// f32 → bf16 bits, round-to-nearest-even (NaN quieted).
#[inline]
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if f.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

#[cfg(test)]
mod tests;
