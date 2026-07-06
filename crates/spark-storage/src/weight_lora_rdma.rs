// SPDX-License-Identifier: AGPL-3.0-only
//
// RdmaLoraLoader — RDMA-stage a PEFT adapter's A/B tensors straight into a
// resident LoRA pool SLOT for fast rotation (vs a disk reload).
//
// This is the client half for LoRA rotation over the RDMA weight tier. It
// reuses the SAME `weight_peer` wire protocol + verbs stack as
// `weight_tier_rdma`: connect, request an adapter dir by id/path, read the
// manifest, then one-sided RDMA-READ each `lora_A/lora_B` tensor's bytes into a
// pinned bounce. The ONLY difference from `weight_tier_rdma` is the landing:
// instead of a fresh per-tensor GPU buffer, each tensor lands into a caller-
// computed pool-slot SUB-REGION (`LoraLandTarget.dst`) after the SAME host
// F16/F32→BF16 conversion the disk adapter loader does
// (`spark-runtime .../adapter.rs`) and the SAME B row-repack (stride r →
// max_rank) the pool pack does (`spark-model .../lora/mod.rs`). Landing bytes
// are therefore byte-identical to the disk pack — post_read-into-bounce simply
// replaces the disk loader's copy_d2h.
//
// The plan (which tensor lands where, with what geometry) is computed by
// spark-model (`lora::rdma_stage`, the only place `classify_key` + slot offsets
// live) and passed in as `&[LoraLandTarget]`, keeping this crate free of any
// spark-model dependency (spark-model → spark-storage is the acyclic direction).
//
// Like `weight_tier_rdma`, the verbs data path is gated on `atlas_rdma_verbs`;
// without rdma-core the loader compiles but `stage_into_slot` returns a clear
// runtime error.

use anyhow::Result;

/// Which half of a LoRA pair a target lands. A is copied contiguous into the
/// head of the padded `[max_rank, in]` region; B is row-repacked from stride
/// `r` to stride `max_rank` into `[out, max_rank]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoraAbKind {
    A,
    B,
}

/// One landing instruction: the adapter tensor `tensor_name` (a manifest key),
/// which half it is, the device destination address (a pool-slot sub-region
/// base, already `pool + slot*slot_bytes + a_off|b_off`), and the geometry the
/// convert/repack needs. `rank` is the adapter's real r; `max_rank` the pool's
/// padded rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraLandTarget {
    pub tensor_name: String,
    pub kind: LoraAbKind,
    /// Destination device address (`DevicePtr.0`) of the slot sub-region.
    pub dst: u64,
    /// Output dim (B rows). Unused for A.
    pub out_dim: usize,
    /// Input dim (A cols). Unused for B.
    pub in_dim: usize,
    /// Adapter real rank r (A rows / B cols).
    pub rank: usize,
    /// Pool padded rank (B destination row stride).
    pub max_rank: usize,
}

const BF16_BYTES: usize = 2;

/// Host F16/F32/BF16 → BF16, byte-for-byte matching the disk adapter loader
/// (`load_adapter_safetensors`): `half::bf16::from_f32` (round-to-nearest-even)
/// for the float conversions. Any other dtype is a hard error — a PEFT adapter
/// is only ever F32 (default), F16, or BF16.
pub fn convert_to_bf16(raw: &[u8], dtype: &str) -> Result<Vec<u8>> {
    use half::{bf16, f16};
    Ok(match dtype {
        "BF16" => raw.to_vec(),
        "F16" => raw
            .chunks_exact(2)
            .flat_map(|c| bf16::from_f32(f16::from_le_bytes([c[0], c[1]]).to_f32()).to_le_bytes())
            .collect(),
        "F32" => raw
            .chunks_exact(4)
            .flat_map(|c| {
                bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes()
            })
            .collect(),
        other => anyhow::bail!(
            "REJECT[lora-rdma-dtype]: adapter tensor dtype '{other}' (want F32/F16/BF16)"
        ),
    })
}

/// Row-repack a BF16 B tensor `[out_dim, r]` into the pool's padded
/// `[out_dim, max_rank]` layout: per row, `r` BF16 elements copied from stride
/// `r` to stride `max_rank`; pad columns stay zero. Byte-identical to the disk
/// pack's B repack. Returns `out_dim * max_rank * 2` bytes.
pub fn repack_b_to_padded(src_bf16: &[u8], out_dim: usize, r: usize, max_rank: usize) -> Vec<u8> {
    let mut dst = vec![0u8; out_dim * max_rank * BF16_BYTES];
    for row in 0..out_dim {
        let d = row * max_rank * BF16_BYTES;
        let s = row * r * BF16_BYTES;
        dst[d..d + r * BF16_BYTES].copy_from_slice(&src_bf16[s..s + r * BF16_BYTES]);
    }
    dst
}

/// The final host bytes to `copy_h2d` for a target, given the raw on-wire
/// tensor bytes (as landed in the bounce). A: convert only. B: convert + repack.
/// Factored out (un-gated) so the byte-identity logic is unit-testable off the
/// RDMA path — the verbs loop calls this so tested and shipped logic agree.
pub fn land_bytes_for_target(target: &LoraLandTarget, raw: &[u8], dtype: &str) -> Result<Vec<u8>> {
    let bf16 = convert_to_bf16(raw, dtype)?;
    Ok(match target.kind {
        LoraAbKind::A => bf16, // [r, in] contiguous → head of padded region
        LoraAbKind::B => repack_b_to_padded(&bf16, target.out_dim, target.rank, target.max_rank),
    })
}

/// RDMA-stage a named adapter's A/B into pre-computed pool-slot sub-regions.
pub struct RdmaLoraLoader {
    /// `host:port` of the weight peer (from `$ATLAS_LORA_PEER`).
    pub peer_addr: String,
    /// Adapter dir id/path the peer staged (its `adapter_model.safetensors`).
    pub adapter_id: String,
}

impl RdmaLoraLoader {
    pub fn new(peer_addr: String, adapter_id: String) -> Self {
        Self {
            peer_addr,
            adapter_id,
        }
    }
}

#[cfg(all(feature = "cuda", not(atlas_rdma_verbs)))]
impl RdmaLoraLoader {
    /// Stub: no rdma-core in this build.
    pub fn stage_into_slot(
        &self,
        _gpu: &dyn spark_runtime::gpu::GpuBackend,
        _targets: &[LoraLandTarget],
    ) -> Result<()> {
        anyhow::bail!(
            "$ATLAS_LORA_PEER is set but this build has no rdma-core (atlas_rdma_verbs \
             cfg); rebuild with rdma-core, or unset ATLAS_LORA_PEER to rotate from disk"
        )
    }
}

#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
impl RdmaLoraLoader {
    /// Connect, request the adapter, and RDMA-READ each `lora_A/lora_B` tensor
    /// into its pool-slot sub-region (single rail — an adapter is one small
    /// shard). Convert + (B) repack land bytes byte-identical to the disk pack.
    pub fn stage_into_slot(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        targets: &[LoraLandTarget],
    ) -> Result<()> {
        use std::collections::HashMap;
        use std::ffi::c_void;
        use std::io::{Read, Write};
        use std::net::TcpStream;

        use anyhow::{Context, bail};

        use crate::expert_peer::{MODE_VERBS, STATUS_OK, VerbsClientParams, read_server_rails};
        use crate::rdma_verbs::Verbs;
        use crate::weight_peer::{read_weight_manifest, tensor_remote_addr, write_model_request};

        let by_name: HashMap<&str, &LoraLandTarget> = targets
            .iter()
            .map(|t| (t.tensor_name.as_str(), t))
            .collect();

        // 1. Connect + request the adapter + read the manifest.
        let mut stream = TcpStream::connect(&self.peer_addr)
            .with_context(|| format!("connect lora peer {}", self.peer_addr))?;
        stream.set_nodelay(true).ok();
        write_model_request(&mut stream, &self.adapter_id).context("send adapter request")?;
        let manifest = read_weight_manifest(&mut stream).context("read adapter manifest")?;
        let num_shards = manifest.num_shards();

        // Only the tensors we have a landing target for (all lora_A/lora_B).
        let retained: Vec<&crate::weight_peer::WeightTensorRecord> = manifest
            .tensors
            .iter()
            .filter(|t| by_name.contains_key(t.name.as_str()))
            .collect();
        if retained.is_empty() {
            bail!(
                "lora peer manifest matched none of the {} land targets",
                targets.len()
            );
        }

        // 2. Single-rail verbs handshake (adapter = one shard, few MB).
        let dev = std::env::var("ATLAS_LORA_RDMA_DEV")
            .or_else(|_| std::env::var("ATLAS_WEIGHT_RDMA_DEV"))
            .or_else(|_| std::env::var("ATLAS_EXPERT_RDMA_DEV"))
            .unwrap_or_else(|_| "roceP2p1s0f1".to_string());
        let gid: u32 = std::env::var("ATLAS_LORA_RDMA_GID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let n_rails = 1usize;

        stream.write_all(&[MODE_VERBS]).context("send verbs mode")?;
        stream.write_all(&[n_rails as u8]).context("send n_rails")?;

        let max_len = retained.iter().map(|t| t.len).max().unwrap_or(0);
        if max_len > u32::MAX as u64 {
            bail!("adapter tensor {max_len} bytes exceeds single-WR RDMA READ limit");
        }
        let bounce_len = (max_len as usize).max(1);

        let psn = rand::random::<u32>() & 0xff_ffff;
        let mut verbs = Verbs::create(&dev, gid, psn)?;
        let bounce = gpu
            .alloc_host_pinned(bounce_len)
            .context("alloc pinned RDMA landing bounce")?;
        // SAFETY: `bounce` backs `bounce_len` pinned bytes that outlive the MR.
        let keys = unsafe { verbs.reg_mr(bounce as *mut c_void, bounce_len, false) }
            .context("register RDMA landing bounce")?;

        let server = read_server_rails(&mut stream, n_rails).context("read verbs server params")?;
        let sp = &server[0];
        if sp.layers.len() != num_shards {
            bail!(
                "peer published {} shard MRs but manifest has {num_shards}",
                sp.layers.len()
            );
        }

        stream
            .write_all(&[n_rails as u8])
            .context("send client n_rails")?;
        VerbsClientParams {
            qpn: verbs.qpn(),
            psn: verbs.psn(),
            gid: verbs.gid(),
        }
        .write_to(&mut stream)
        .context("send verbs client params")?;
        verbs.connect(sp.qpn, sp.psn, &sp.gid)?;
        let mut ack = [0u8; 1];
        stream
            .read_exact(&mut ack)
            .context("read verbs ready ack")?;
        if ack[0] != STATUS_OK {
            bail!("lora peer refused connection (ack {})", ack[0]);
        }

        // 3. Pull each tensor into the bounce, convert/repack, land into slot.
        for (idx, rec) in retained.iter().enumerate() {
            let (shard_base, rkey) = *sp
                .layers
                .get(rec.shard_index as usize)
                .with_context(|| format!("no shard MR {} for {}", rec.shard_index, rec.name))?;
            let remote_addr = tensor_remote_addr(shard_base, rec.offset_in_shard);
            let len = rec.len as usize;
            let wr_id = idx as u64;
            // SAFETY: bounce backs >= len pinned bytes in this MR; remote_addr/
            // rkey address the peer's shard MR; len <= u32::MAX.
            unsafe {
                verbs
                    .post_read(
                        bounce as *mut c_void,
                        keys.lkey,
                        remote_addr,
                        rkey,
                        len as u32,
                        wr_id,
                    )
                    .with_context(|| format!("post_read {}", rec.name))?;
            }
            match verbs.poll() {
                Ok(got) if got == wr_id => {}
                Ok(got) => bail!("completion wr_id {got:#x} != {wr_id:#x} ({})", rec.name),
                Err(e) => return Err(e).with_context(|| format!("poll {}", rec.name)),
            }
            // SAFETY: bounce now holds `len` valid bytes from the READ.
            let raw = unsafe { std::slice::from_raw_parts(bounce, len) };
            let target = by_name[rec.name.as_str()];
            let host = land_bytes_for_target(target, raw, &rec.dtype)?;
            gpu.copy_h2d(&host, spark_runtime::gpu::DevicePtr(target.dst))?;
        }

        drop(verbs);
        let _ = gpu.free_host_pinned(bounce, bounce_len);
        tracing::info!(
            "RDMA-staged adapter '{}' into {} slot targets",
            manifest.model_id,
            retained.len(),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};

    #[test]
    fn convert_f32_halves_len_and_rounds_like_disk() {
        // Two f32 values → 4 BF16 bytes, matching half::bf16::from_f32.
        let vals = [1.5f32, -2.25f32];
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = convert_to_bf16(&raw, "F32").unwrap();
        assert_eq!(out.len(), 4, "F32 [n] → BF16 [n] halves the byte count");
        let expect: Vec<u8> = vals
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
            .collect();
        assert_eq!(out, expect);
    }

    #[test]
    fn convert_f16_preserves_len() {
        let raw: Vec<u8> = [f16::from_f32(0.5), f16::from_f32(3.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = convert_to_bf16(&raw, "F16").unwrap();
        assert_eq!(out.len(), 4, "F16 [n] → BF16 [n] same byte count");
    }

    #[test]
    fn convert_bf16_is_identity() {
        let raw: Vec<u8> = (0..8).collect();
        assert_eq!(convert_to_bf16(&raw, "BF16").unwrap(), raw);
    }

    #[test]
    fn convert_rejects_unknown_dtype() {
        assert!(convert_to_bf16(&[0u8; 4], "F8_E4M3").is_err());
    }

    #[test]
    fn repack_b_pads_columns_zero_and_preserves_rows() {
        // out_dim=2, r=1, max_rank=3. src = 2 rows × 1 bf16 = [A0,A1].
        // dst = 2 rows × 3 bf16, real col at head, pads zero.
        let src: Vec<u8> = vec![0xAA, 0xBB, 0xCC, 0xDD]; // row0=[AA BB], row1=[CC DD]
        let dst = repack_b_to_padded(&src, 2, 1, 3);
        assert_eq!(dst.len(), 2 * 3 * 2);
        assert_eq!(&dst[0..2], &[0xAA, 0xBB]); // row0 real
        assert_eq!(&dst[2..6], &[0, 0, 0, 0]); // row0 pad cols
        assert_eq!(&dst[6..8], &[0xCC, 0xDD]); // row1 real
        assert_eq!(&dst[8..12], &[0, 0, 0, 0]); // row1 pad cols
    }

    #[test]
    fn land_a_is_convert_only() {
        let t = LoraLandTarget {
            tensor_name: "x".into(),
            kind: LoraAbKind::A,
            dst: 0,
            out_dim: 0,
            in_dim: 2,
            rank: 1,
            max_rank: 4,
        };
        let raw: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = land_bytes_for_target(&t, &raw, "F32").unwrap();
        assert_eq!(out.len(), 4); // r*in*2 = 1*2*2, no padding for A columns
    }

    #[test]
    fn land_b_converts_then_repacks() {
        let t = LoraLandTarget {
            tensor_name: "y".into(),
            kind: LoraAbKind::B,
            dst: 0,
            out_dim: 2,
            in_dim: 0,
            rank: 1,
            max_rank: 3,
        };
        // B = [out=2, r=1] f32 → convert → repack to [2,3] bf16.
        let raw: Vec<u8> = [5.0f32, 6.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = land_bytes_for_target(&t, &raw, "F32").unwrap();
        assert_eq!(out.len(), 2 * 3 * 2);
        // Row real cols == bf16 of the source values.
        assert_eq!(&out[0..2], &bf16::from_f32(5.0).to_le_bytes());
        assert_eq!(&out[6..8], &bf16::from_f32(6.0).to_le_bytes());
    }
}
