// SPDX-License-Identifier: AGPL-3.0-only
//
// RdmaWeightLoader — the client half of the RDMA weight-staging tier.
//
// A `spark_runtime::weights::WeightLoader` (the same trait the disk loaders
// impl) whose source is a peer's RAM blade (`weight_peer`) over one-sided RDMA
// instead of the local SSD. For FAST MODEL SWAPS: connect, request a model by
// id/path, read the peer's manifest, then RDMA-READ every resident tensor's
// bytes straight out of the peer's shard MRs into a pinned bounce and
// `copy_h2d` to a freshly-alloc'd GPU buffer — one buffer per tensor, keyed by
// the exact safetensors name, byte-identical to the disk path.
//
// It composes with expert streaming: `stream_all_experts` / EP filtering skip
// the routed experts (served separately by `expert_peer`), so this loads only
// the resident set (attention, router gate, shared expert, norms, embed,
// lm_head, MTP) — the exact `should_skip_tensor` predicate the disk loaders use.
//
// Bounce path (option a): tensors are MB-sized and bandwidth-bound, so the
// bounce + copy_h2d overhead is negligible (unlike the 8 KiB KV groups that
// needed zero-copy). Reuses the dual-rail striping template (tensor % n_rails).
//
// Like `expert_tier_rdma`, the verbs data path is gated on `atlas_rdma_verbs`;
// without rdma-core the loader compiles but `load` returns a clear runtime error
// (the selection lives in the server's `load_weight_store`, keyed on
// `$ATLAS_WEIGHT_PEER`).

use anyhow::Result;
use std::path::Path;

use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightLoader, WeightStore, parse_expert_index};

use crate::weight_peer::WeightTensorRecord;

/// Loads a model's resident weights from a `weight_peer` over one-sided RDMA.
pub struct RdmaWeightLoader {
    /// `host:port` of the weight peer (from `$ATLAS_WEIGHT_PEER`).
    pub peer_addr: String,
    /// Model id/path to request. When `None`, the loader sends the `model_dir`
    /// path passed to `load` (so the client and peer agree on the local path).
    pub model_id: Option<String>,
    pub ep_rank: usize,
    pub ep_world_size: usize,
    pub num_experts: usize,
    /// Expert streaming: skip ALL routed-expert tensors (served from the expert
    /// peer). Set from `config.expert_streaming` at the call site.
    pub stream_all_experts: bool,
    /// Pre-flight OOM multiplier override (advisory; parity with disk loaders).
    pub peak_memory_multiplier: Option<f64>,
}

impl RdmaWeightLoader {
    pub fn new(peer_addr: String) -> Self {
        Self {
            peer_addr,
            model_id: None,
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            stream_all_experts: false,
            peak_memory_multiplier: None,
        }
    }

    pub fn with_ep(peer_addr: String, ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            peer_addr,
            model_id: None,
            ep_rank,
            ep_world_size,
            num_experts,
            stream_all_experts: false,
            peak_memory_multiplier: None,
        }
    }

    /// The exact skip predicate the disk loaders use (`SafetensorsLoader` /
    /// `FastSafetensorsLoader::should_skip_tensor`), applied to a manifest
    /// record. `extra_weights` tensors (`rec.extra`) are NEVER skipped, matching
    /// the disk path's no-skip pass for `extra_weights.safetensors`.
    // Only reached from the `atlas_rdma_verbs` load path; on a cuda host without
    // rdma-core the whole data path runtime-bails, leaving this unreferenced.
    #[cfg_attr(not(atlas_rdma_verbs), allow(dead_code))]
    fn should_skip_tensor(&self, rec: &WeightTensorRecord) -> bool {
        if rec.extra {
            return false;
        }
        // MTP head experts are small — always replicate, never skip.
        if rec.name.starts_with("mtp.") {
            return false;
        }
        // Expert streaming: routed experts stream from the expert peer.
        if self.stream_all_experts && parse_expert_index(&rec.name).is_some() {
            return true;
        }
        if self.ep_world_size <= 1 {
            return false;
        }
        if let Some(idx) = parse_expert_index(&rec.name) {
            let per_rank = self.num_experts / self.ep_world_size;
            let local_start = self.ep_rank * per_rank;
            let local_end = if self.ep_rank == self.ep_world_size - 1 {
                self.num_experts
            } else {
                local_start + per_rank
            };
            idx < local_start || idx >= local_end
        } else {
            false
        }
    }
}

impl WeightLoader for RdmaWeightLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        self.load_impl(model_dir, gpu, oom_reserve_bytes)
    }
}

#[cfg(not(atlas_rdma_verbs))]
impl RdmaWeightLoader {
    fn load_impl(
        &self,
        _model_dir: &Path,
        _gpu: &dyn GpuBackend,
        _oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        anyhow::bail!(
            "$ATLAS_WEIGHT_PEER is set but this build has no rdma-core (atlas_rdma_verbs \
             cfg); rebuild with rdma-core, or unset ATLAS_WEIGHT_PEER to load from disk"
        )
    }
}

#[cfg(atlas_rdma_verbs)]
fn env_str(keys: &[&str], default: &str) -> String {
    for k in keys {
        if let Ok(v) = std::env::var(k) {
            if !v.is_empty() {
                return v;
            }
        }
    }
    default.to_string()
}

#[cfg(atlas_rdma_verbs)]
fn env_u32(keys: &[&str], default: u32) -> u32 {
    for k in keys {
        if let Some(v) = std::env::var(k).ok().and_then(|s| s.parse().ok()) {
            return v;
        }
    }
    default
}

#[cfg(atlas_rdma_verbs)]
impl RdmaWeightLoader {
    fn load_impl(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        use std::collections::HashMap;
        use std::ffi::c_void;
        use std::io::{Read, Write};
        use std::net::TcpStream;

        use anyhow::{Context, bail};

        use crate::expert_peer::{MODE_VERBS, STATUS_OK, VerbsClientParams, read_server_rails};
        use crate::rdma_verbs::Verbs;
        use crate::weight_peer::{
            rail_for_tensor, read_weight_manifest, tensor_remote_addr, write_model_request,
        };
        use spark_runtime::weights::{WeightDtype, WeightTensor};

        // 1. Connect + request the model + read the manifest.
        let mut stream = TcpStream::connect(&self.peer_addr)
            .with_context(|| format!("connect weight peer {}", self.peer_addr))?;
        stream.set_nodelay(true).ok();
        let model_id = self
            .model_id
            .clone()
            .unwrap_or_else(|| model_dir.to_string_lossy().into_owned());
        write_model_request(&mut stream, &model_id).context("send model request")?;
        let manifest = read_weight_manifest(&mut stream).context("read weight manifest")?;
        let num_shards = manifest.num_shards();

        // 2. Filter to the resident set (skip streamed/EP experts). extra_weights
        // tensors are always kept (should_skip_tensor honors rec.extra).
        let retained: Vec<&WeightTensorRecord> = manifest
            .tensors
            .iter()
            .filter(|t| !self.should_skip_tensor(t))
            .collect();

        // 3. Advisory OOM pre-flight (parity with the disk loaders — estimate
        // from the manifest, not local headers).
        {
            let est: u64 = retained.iter().map(|t| t.len).sum();
            let fp8: u64 = retained
                .iter()
                .filter(|t| t.dtype == "F8_E4M3")
                .map(|t| t.len)
                .sum();
            let fp8_frac = if est > 0 { fp8 as f64 / est as f64 } else { 0.0 };
            let mult = self
                .peak_memory_multiplier
                .unwrap_or(if fp8_frac > 0.5 { 1.5 } else { 1.3 });
            let peak = (est as f64 * mult) as usize;
            let free = gpu.free_memory()?;
            let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!(
                "RDMA weight load pre-flight: {:.2} GB manifest, {:.1}x = {:.2} GB peak, \
                 {:.2} GB free, {:.1} GB reserve (FP8 {:.0}%)",
                gib(est as usize),
                mult,
                gib(peak),
                gib(free),
                gib(oom_reserve_bytes),
                fp8_frac * 100.0,
            );
            if peak + oom_reserve_bytes > free {
                bail!(
                    "OOM pre-flight (RDMA weight peer): peak {:.2} GB + {:.2} GB reserve > {:.2} GB free",
                    gib(peak),
                    gib(oom_reserve_bytes),
                    gib(free),
                );
            }
        }

        // 4. Verbs handshake. Rail 0 defaults to the shared expert CX7 link;
        // dual-rail is opt-in (ATLAS_WEIGHT_DUAL_RAIL=1). ATLAS_WEIGHT_* overrides
        // fall back to the ATLAS_EXPERT_* names so a single fabric config serves
        // both tiers.
        let dev0 = env_str(&["ATLAS_WEIGHT_RDMA_DEV", "ATLAS_EXPERT_RDMA_DEV"], "roceP2p1s0f1");
        let gid0 = env_u32(&["ATLAS_WEIGHT_RDMA_GID", "ATLAS_EXPERT_RDMA_GID"], 3);
        let dev1 = env_str(&["ATLAS_WEIGHT_RAIL2_DEV", "ATLAS_EXPERT_RAIL2_DEV"], "rocep1s0f1");
        let gid1 = env_u32(&["ATLAS_WEIGHT_RAIL2_GID", "ATLAS_EXPERT_RAIL2_GID"], 3);
        let dual = std::env::var("ATLAS_WEIGHT_DUAL_RAIL").ok().as_deref() == Some("1");
        let rail_devs: Vec<(String, u32)> = if dual {
            vec![(dev0, gid0), (dev1, gid1)]
        } else {
            vec![(dev0, gid0)]
        };
        let n_rails = rail_devs.len();

        stream.write_all(&[MODE_VERBS]).context("send verbs mode")?;
        stream.write_all(&[n_rails as u8]).context("send n_rails")?;

        // One pinned, registered bounce per rail, sized to the largest retained
        // tensor. Tensors are processed serially per rail (post → poll), so one
        // bounce per rail suffices; pipelining is deferred (bandwidth-bound).
        let max_len = retained.iter().map(|t| t.len).max().unwrap_or(0);
        if max_len > u32::MAX as u64 {
            bail!(
                "tensor of {} bytes exceeds the 4 GiB single-WR RDMA READ limit \
                 (per-tensor chunking not implemented)",
                max_len
            );
        }
        let bounce_len = (max_len as usize).max(1);

        struct Rail {
            verbs: Verbs,
            bounce_ptr: *mut u8,
            bounce_lkey: u32,
        }
        let mut rails: Vec<Rail> = Vec::with_capacity(n_rails);
        // Track pinned allocations to free AFTER the rails (MRs) are dropped.
        let mut pinned: Vec<*mut u8> = Vec::with_capacity(n_rails);
        for (dev, gid) in &rail_devs {
            let psn = rand::random::<u32>() & 0xff_ffff;
            let mut v = Verbs::create(dev, *gid, psn)?;
            let ptr = gpu
                .alloc_host_pinned(bounce_len)
                .context("alloc pinned RDMA landing bounce")?;
            // SAFETY: ptr backs `bounce_len` pinned bytes that outlive the MR
            // (freed after the rails are dropped below).
            let keys = unsafe { v.reg_mr(ptr as *mut c_void, bounce_len, false) }
                .context("register RDMA landing bounce")?;
            pinned.push(ptr);
            rails.push(Rail {
                verbs: v,
                bounce_ptr: ptr,
                bounce_lkey: keys.lkey,
            });
        }

        // Peer publishes per-rail per-SHARD (base, rkey). Validate shard counts.
        let server = read_server_rails(&mut stream, n_rails).context("read verbs server params")?;
        for sp in &server {
            if sp.layers.len() != num_shards {
                bail!(
                    "peer published {} shard MRs but manifest has {num_shards} shards",
                    sp.layers.len()
                );
            }
        }

        // Reply with our QP params, connect each rail, await the ready ack.
        stream.write_all(&[n_rails as u8]).context("send client n_rails")?;
        for r in &rails {
            VerbsClientParams {
                qpn: r.verbs.qpn(),
                psn: r.verbs.psn(),
                gid: r.verbs.gid(),
            }
            .write_to(&mut stream)
            .context("send verbs client params")?;
        }
        for (i, r) in rails.iter_mut().enumerate() {
            let sp = &server[i];
            r.verbs.connect(sp.qpn, sp.psn, &sp.gid)?;
        }
        let mut ack = [0u8; 1];
        stream.read_exact(&mut ack).context("read verbs ready ack")?;
        if ack[0] != STATUS_OK {
            bail!("weight peer refused connection (ack {})", ack[0]);
        }
        tracing::info!(
            "RDMA weight loader connected to {} ({} shards, {} resident tensors, {n_rails} rail(s))",
            manifest.model_id,
            num_shards,
            retained.len(),
        );

        // 5. RDMA-READ each resident tensor into its rail's bounce, then copy_h2d
        // into a fresh per-tensor GPU buffer. Byte-identical: the manifest offset
        // is absolute (shard_base + offset reads the raw data slice), `len` is
        // authoritative, dtype/shape come from the header verbatim.
        let mut weights: HashMap<String, WeightTensor> = HashMap::new();
        let mut offload_logged = false;
        for (idx, rec) in retained.iter().enumerate() {
            let ri = rail_for_tensor(idx, n_rails);
            let sp = &server[ri];
            let (shard_base, rkey) = *sp
                .layers
                .get(rec.shard_index as usize)
                .with_context(|| format!("no shard MR {} for {}", rec.shard_index, rec.name))?;
            let remote_addr = tensor_remote_addr(shard_base, rec.offset_in_shard);
            let len = rec.len as usize;
            let wr_id = idx as u64;

            let rail = &mut rails[ri];
            // SAFETY: bounce_ptr backs `bounce_len >= len` pinned bytes in this
            // rail's MR; remote_addr/rkey address the peer's shard MR on this
            // same rail; len <= u32::MAX (checked above).
            unsafe {
                rail.verbs
                    .post_read(
                        rail.bounce_ptr as *mut c_void,
                        rail.bounce_lkey,
                        remote_addr,
                        rkey,
                        len as u32,
                        wr_id,
                    )
                    .with_context(|| format!("post_read {}", rec.name))?;
            }
            match rail.verbs.poll() {
                Ok(got) if got == wr_id => {}
                Ok(got) => bail!("completion wr_id {got:#x} != expected {wr_id:#x} ({})", rec.name),
                Err(e) => return Err(e).with_context(|| format!("poll {}", rec.name)),
            }

            // SAFETY: the bounce now holds `len` valid bytes landed by the READ.
            let src = unsafe { std::slice::from_raw_parts(rail.bounce_ptr, len) };
            let dtype = WeightDtype::from_safetensors_str(&rec.dtype)
                .with_context(|| format!("tensor {}", rec.name))?;
            let shape: Vec<usize> = rec.shape.iter().map(|&d| d as usize).collect();

            let ptr = match gpu.alloc(len) {
                Ok(p) => {
                    gpu.copy_h2d(src, p)?;
                    p
                }
                Err(_) => {
                    if !offload_logged {
                        tracing::warn!(
                            "GPU alloc failed for {} ({len} bytes) — switching to managed (UVM) memory",
                            rec.name
                        );
                        offload_logged = true;
                    }
                    let p = gpu.alloc_managed(len)?;
                    // SAFETY: managed ptr is host-addressable UVM of `len` bytes;
                    // src is the pinned bounce of `len` bytes. Matches the disk
                    // loaders' CPU-memcpy fallback.
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), p.0 as *mut u8, len);
                    }
                    p
                }
            };
            weights.insert(rec.name.clone(), WeightTensor { ptr, shape, dtype });
        }

        // 6. Tear down: drop the rails (dereg MRs) BEFORE freeing the pinned
        // bounces they registered, then release the pinned host memory.
        drop(rails);
        for ptr in pinned {
            let _ = gpu.free_host_pinned(ptr, bounce_len);
        }

        tracing::info!("RDMA-loaded {} weight tensors", weights.len());
        Ok(WeightStore::from_map(weights))
    }
}
