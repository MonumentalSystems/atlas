// SPDX-License-Identifier: AGPL-3.0-only
//
// Weight-serving peer + wire protocol — the RDMA weight-staging tier.
//
// Generalizes `expert_peer` from (layer, expert) expert records to ALL of a
// model's safetensors tensors, for FAST MODEL SWAPS. A peer holds one or more
// staged models' shard files mmap'd + `ibv_reg_mr`'d REMOTE_READ in its RAM; a
// client (`weight_tier_rdma::RdmaWeightLoader`) requests a model by id/path,
// reads the peer's MANIFEST, then one-sided RDMA-READs each tensor's bytes
// straight out of the shard MRs (~24 GB/s dual-rail) instead of the ~2 GB/s USB
// SSD. Weights are READ-ONLY → one-sided READ, no coherence — the exact
// expert-tier pattern.
//
// It's a CACHE: the FIRST stage of a model into the blade faults its pages in
// from disk (slow); every later swap reads them out of the peer's warm RAM
// (fast). Pre-warm the rotation set by connecting once.
//
// Wire protocol (little-endian), connection-oriented, server responds to the
// client's model choice first:
//   1. Client sends the model request: `[u32 len][len bytes of model id/path]`.
//   2. Server stages that model (mmap + parse headers, cached across
//      connections) and sends the manifest: `[u32 len][len bytes of JSON]`
//      (`WeightManifest` — per-tensor {name,dtype,shape,offset,len,shard}).
//   3. Client sends `[u8 transport_mode]` (only `MODE_VERBS` is served).
//   4. Verbs handshake (reused verbatim from `expert_peer`): `[u8 n_rails]`,
//      then per rail a `VerbsServerParams` whose `layers` vector carries this
//      model's per-SHARD `(mr_base, rkey)` (shards play the role experts' layer
//      files do). The client replies with its QP params, the server connects
//      and idles — the client pulls all tensor bytes one-sided.
//
// Per-tensor geometry rides the JSON manifest (like `ExpertIndex`); only the
// per-shard `(base, rkey)` rides `VerbsServerParams` (like the expert peer's
// per-layer `(base, rkey)`) — keeping shard counts well under the 4096/8 wire
// caps and the 512-MR-per-QP shim limit (real models have tens of shards).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// One tensor's placement inside a staged model, mirroring the safetensors
/// header exactly: `offset_in_shard` is the ABSOLUTE file offset (8-byte size
/// prefix + header + the tensor's data-section start), `len` is the raw
/// contiguous byte count (`data_offsets[1] - data_offsets[0]`). The client
/// RDMA-READs exactly `[shard_base + offset_in_shard .. + len)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightTensorRecord {
    /// HuggingFace tensor name — the `WeightStore` key, verbatim.
    pub name: String,
    /// Raw safetensors dtype string (`"BF16"`, `"F8_E4M3"`, `"I8"`, …). The
    /// client maps it via `WeightDtype::from_safetensors_str` — the same closed
    /// mapping the disk loaders use.
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Absolute byte offset of the tensor's first byte within its shard file.
    pub offset_in_shard: u64,
    /// Tensor byte length (authoritative — do NOT recompute from shape; packed
    /// NVFP4 lengths differ from `numel * byte_size`).
    pub len: u64,
    /// Index into [`WeightManifest::shard_files`].
    pub shard_index: u32,
    /// True for tensors from `extra_weights.safetensors` (grafted MTP etc.):
    /// loaded with NO expert-skip filter, exactly like the disk loaders.
    pub extra: bool,
}

/// A staged model's manifest: the geometry the client needs to reconstruct a
/// byte-identical `WeightStore`. Published as length-prefixed JSON right after
/// the client's model request. The per-shard `(base, rkey)` MR handles ride the
/// verbs handshake separately (see the module doc).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightManifest {
    pub version: u32,
    /// The resolved model id/path the peer staged (echo for the client's log).
    pub model_id: String,
    /// Shard file names, shard-indexed. `shard_index` in each tensor record
    /// indexes this list; the verbs `layers` vector is per-shard in this order.
    pub shard_files: Vec<String>,
    /// Byte length of each shard file, shard-indexed (parallels `shard_files`).
    pub shard_lens: Vec<u64>,
    pub tensors: Vec<WeightTensorRecord>,
}

impl WeightManifest {
    pub const VERSION: u32 = 1;

    /// Number of shard files (== the per-rail `layers` MR count the peer
    /// publishes and the client validates against).
    pub fn num_shards(&self) -> usize {
        self.shard_files.len()
    }

    /// Total registered bytes across all shards (the whole-file MRs) — the
    /// figure charged against the blade `CommitLedger` once per staged model.
    pub fn total_shard_bytes(&self) -> u64 {
        self.shard_lens.iter().sum()
    }
}

/// Rail selection for a tensor under dual-rail striping: tensor `N` is served
/// over rail `N % n_rails`. Factored out (un-gated) so the striping is unit-
/// testable off the RDMA path — the client's read loop calls this so the tested
/// logic and the shipped logic are the same. `n_rails` is clamped to `>= 1`.
pub fn rail_for_tensor(tensor_index: usize, n_rails: usize) -> usize {
    tensor_index % n_rails.max(1)
}

/// Absolute peer virtual address of a tensor's first byte: the shard's whole-
/// file REMOTE_READ MR base plus the tensor's ABSOLUTE in-shard offset (the
/// safetensors data-section offset, which already includes the 8-byte size
/// prefix + header). The client RDMA-READs `[addr .. addr + len)`. Factored out
/// (un-gated) so the address math is unit-testable off the RDMA path.
pub fn tensor_remote_addr(shard_base: u64, offset_in_shard: u64) -> u64 {
    shard_base + offset_in_shard
}

fn read_u32<R: std::io::Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).context("read u32")?;
    Ok(u32::from_le_bytes(b))
}

/// Longest model id/path the wire accepts (a corrupt/hostile length must not
/// trigger a huge allocation).
pub const MODEL_REQUEST_MAX: usize = 8192;

/// Wire form of the model request: `[u32 len][len bytes UTF-8 id/path]`.
pub fn write_model_request<W: std::io::Write>(w: &mut W, id: &str) -> Result<()> {
    let bytes = id.as_bytes();
    if bytes.is_empty() || bytes.len() > MODEL_REQUEST_MAX {
        bail!("implausible model request length: {}", bytes.len());
    }
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(())
}

/// Read a model request written by [`write_model_request`].
pub fn read_model_request<R: std::io::Read>(r: &mut R) -> Result<String> {
    let len = read_u32(r)? as usize;
    if len == 0 || len > MODEL_REQUEST_MAX {
        bail!("implausible model request length: {len}");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).context("read model request body")?;
    String::from_utf8(buf).context("model request is not valid UTF-8")
}

/// Serialize + frame a manifest as `[u32 len][len bytes JSON]`.
pub fn write_weight_manifest<W: std::io::Write>(w: &mut W, m: &WeightManifest) -> Result<()> {
    let json = serde_json::to_vec(m).context("serialize weight manifest")?;
    w.write_all(&(json.len() as u32).to_le_bytes())?;
    w.write_all(&json)?;
    Ok(())
}

/// Read + parse a length-prefixed manifest. Shared by the client tier.
pub fn read_weight_manifest<R: std::io::Read>(r: &mut R) -> Result<WeightManifest> {
    let len = read_u32(r)? as usize;
    if len == 0 || len > 256 * 1024 * 1024 {
        bail!("implausible weight manifest length: {len}");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .context("read weight manifest json")?;
    let m: WeightManifest = serde_json::from_slice(&buf).context("parse weight manifest json")?;
    if m.version != WeightManifest::VERSION {
        bail!(
            "weight manifest version {} != supported {}",
            m.version,
            WeightManifest::VERSION
        );
    }
    if m.shard_files.len() != m.shard_lens.len() {
        bail!(
            "manifest shard_files ({}) / shard_lens ({}) length mismatch",
            m.shard_files.len(),
            m.shard_lens.len()
        );
    }
    Ok(m)
}

#[cfg(unix)]
pub use server_impl::{WeightPeerConfig, serve};

#[cfg(unix)]
mod server_impl {
    use super::*;
    use crate::expert_peer::MODE_VERBS;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::io::Read;
    use std::net::{TcpListener, TcpStream, ToSocketAddrs};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    /// Peer configuration: the RDMA rails, the memory ceiling, and how the peer
    /// resolves a model request to a directory.
    #[derive(Clone, Debug)]
    pub struct WeightPeerConfig {
        /// `(device, gid_idx)` per rail, in link order (rail 0 = the cabled
        /// link). Mirrors `expert_peer::RdmaConfig::rails`.
        pub rails: Vec<(String, u32)>,
        /// Ceiling on total registered (staged) RAM in bytes across ALL staged
        /// models. `0` = unlimited. Each model is charged its shard bytes once,
        /// at first stage (the per-connection MRs share the same warm pages).
        pub max_blade_bytes: u64,
        /// Model directories to pre-stage at startup. Also the allow-list when
        /// `allow_any_path` is false: a client may only request a model whose
        /// resolved path matches one of these (or its basename).
        pub staged_dirs: Vec<PathBuf>,
        /// When true, a client may request ANY filesystem path and the peer
        /// stages it on demand (convenient for a trusted LAN; off by default).
        pub allow_any_path: bool,
    }

    impl Default for WeightPeerConfig {
        fn default() -> Self {
            Self {
                rails: vec![("roceP2p1s0f1".into(), 3)],
                max_blade_bytes: 0,
                staged_dirs: Vec::new(),
                allow_any_path: false,
            }
        }
    }

    /// A staged model held resident: the persistent shard mmaps (kept mapped so
    /// their pages stay warm in RAM across connections), the manifest, and the
    /// ledger reservation released when the model is dropped. Per-connection
    /// `reg_mr` re-registers these same base VAs on each client QP's PD.
    struct StagedModel {
        // Read only by the verbs serve path (reg_mr each shard); on a build
        // without rdma-core the mmaps still hold pages warm but aren't iterated.
        #[cfg_attr(not(atlas_rdma_verbs), allow(dead_code))]
        shard_mmaps: Vec<Mmap>,
        manifest: WeightManifest,
        _reservation: crate::blade_cap::Reservation,
    }

    type StagedMap = Arc<Mutex<HashMap<String, Arc<StagedModel>>>>;

    /// Serve staged models on `addr` until interrupted. One thread per
    /// connection; blocking. Intended to run as its own process
    /// (`atlas-weight-peer`).
    pub fn serve<A: ToSocketAddrs>(addr: A, cfg: WeightPeerConfig) -> Result<()> {
        let cfg = Arc::new(cfg);
        let ledger = Arc::new(crate::blade_cap::CommitLedger::new(cfg.max_blade_bytes));
        let staged: StagedMap = Arc::new(Mutex::new(HashMap::new()));

        // Pre-stage the configured directories (first stage is the slow one —
        // do it up front so the first client swap is already warm).
        for dir in &cfg.staged_dirs {
            match stage_model(&staged, &ledger, dir) {
                Ok(m) => tracing::info!(
                    "weight-peer pre-staged {} ({} shards, {} tensors, {:.1} GiB)",
                    m.manifest.model_id,
                    m.manifest.num_shards(),
                    m.manifest.tensors.len(),
                    m.manifest.total_shard_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                ),
                Err(e) => tracing::warn!("weight-peer pre-stage {} failed: {e}", dir.display()),
            }
        }

        let listener = TcpListener::bind(addr).context("bind weight-peer listener")?;
        let local = listener.local_addr().ok();
        tracing::info!(
            "weight-peer serving on {:?} (verbs rails {:?}, cap {}, allow_any_path {})",
            local,
            cfg.rails,
            if cfg.max_blade_bytes == 0 {
                "unlimited".to_string()
            } else {
                format!(
                    "{:.1} GiB",
                    cfg.max_blade_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                )
            },
            cfg.allow_any_path,
        );

        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("weight-peer accept error: {e}");
                    continue;
                }
            };
            let cfg = cfg.clone();
            let ledger = ledger.clone();
            let staged = staged.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_conn(stream, &cfg, &ledger, &staged) {
                    tracing::warn!("weight-peer connection ended: {e}");
                }
            });
        }
        Ok(())
    }

    fn handle_conn(
        mut stream: TcpStream,
        cfg: &WeightPeerConfig,
        ledger: &Arc<crate::blade_cap::CommitLedger>,
        staged: &StagedMap,
    ) -> Result<()> {
        stream.set_nodelay(true).ok();

        // 1. Client tells us which model it wants.
        let request = read_model_request(&mut stream)?;
        let dir = resolve_request(cfg, &request)?;

        // 2. Stage it (or reuse the warm one) and publish the manifest.
        let model = stage_model(staged, ledger, &dir)?;
        write_weight_manifest(&mut stream, &model.manifest).context("send manifest")?;

        // 3. Transport selection. Only verbs is served for weights.
        let mut mode = [0u8; 1];
        stream
            .read_exact(&mut mode)
            .context("read transport mode")?;
        match mode[0] {
            MODE_VERBS => serve_verbs(stream, &model, cfg),
            other => bail!("weight-peer only serves verbs; client asked for mode {other}"),
        }
    }

    /// Resolve a client's model request string to a directory, honoring the
    /// allow-list / `allow_any_path` policy.
    fn resolve_request(cfg: &WeightPeerConfig, request: &str) -> Result<PathBuf> {
        let req = Path::new(request);
        // Exact path match against a staged dir, or basename match.
        for d in &cfg.staged_dirs {
            if d == req
                || d.file_name().and_then(|n| n.to_str()) == Some(request)
                || d.to_string_lossy() == request
            {
                return Ok(d.clone());
            }
        }
        if cfg.allow_any_path && req.is_dir() {
            return Ok(req.to_path_buf());
        }
        bail!(
            "model '{request}' is not staged (and allow_any_path is off); \
             pass it to the peer with --stage <dir>"
        );
    }

    /// Look up a warm staged model or stage it now (mmap shards + parse headers
    /// + charge the ledger). Idempotent per resolved-path key.
    fn stage_model(
        staged: &StagedMap,
        ledger: &Arc<crate::blade_cap::CommitLedger>,
        dir: &Path,
    ) -> Result<Arc<StagedModel>> {
        let key = dir.to_string_lossy().into_owned();
        {
            let map = staged.lock().unwrap();
            if let Some(m) = map.get(&key) {
                return Ok(m.clone());
            }
        }

        // Build the manifest (resolve shards + parse each shard's header) and
        // mmap every shard REMOTE-read + warm.
        let (shard_paths, manifest) = build_manifest(dir, &key)?;
        // Charge the ledger BEFORE pinning any pages; the RAII guard lives in
        // the StagedModel and releases if we bail below or when it's dropped.
        let reservation = ledger
            .try_reserve(manifest.total_shard_bytes())
            .context("weight blade cap")?;

        let mut shard_mmaps = Vec::with_capacity(shard_paths.len());
        for p in &shard_paths {
            shard_mmaps.push(Mmap::open_ro(p).with_context(|| format!("mmap {}", p.display()))?);
        }

        let model = Arc::new(StagedModel {
            shard_mmaps,
            manifest,
            _reservation: reservation,
        });
        let mut map = staged.lock().unwrap();
        // Another thread may have staged it while we worked; prefer the existing
        // one (drops ours, releasing its reservation).
        Ok(map.entry(key).or_insert(model).clone())
    }

    /// One-sided RDMA READ weight serving. Registers each shard mmap REMOTE_READ
    /// on every rail, publishes the per-shard `(base, rkey)`, connects to the
    /// client's QPs, then idles — the client pulls all tensor bytes one-sided.
    #[cfg(not(atlas_rdma_verbs))]
    fn serve_verbs(
        _stream: TcpStream,
        _model: &Arc<StagedModel>,
        _cfg: &WeightPeerConfig,
    ) -> Result<()> {
        bail!("client requested verbs transport but this peer was built without rdma-core");
    }

    #[cfg(atlas_rdma_verbs)]
    fn serve_verbs(
        mut stream: TcpStream,
        model: &Arc<StagedModel>,
        cfg: &WeightPeerConfig,
    ) -> Result<()> {
        use crate::expert_peer::{
            STATUS_OK, VerbsClientParams, VerbsServerParams, write_server_rails,
        };
        use crate::rdma_verbs::Verbs;
        use std::io::Write;

        let num_shards = model.shard_mmaps.len();

        // Negotiate the rail count.
        let mut b1 = [0u8; 1];
        stream.read_exact(&mut b1).context("read n_rails")?;
        let n_rails = b1[0] as usize;
        if n_rails == 0 || n_rails > cfg.rails.len() {
            bail!(
                "client asked for {n_rails} rails; peer has {}",
                cfg.rails.len()
            );
        }

        // One QP per rail (distinct per-rail PSN so successive clients don't
        // collide). No ledger charge here — staging already charged the pages;
        // the N per-rail MRs share those same refcounted mmap pages.
        let pid = std::process::id();
        let mut rails: Vec<Verbs> = Vec::with_capacity(n_rails);
        for (i, (dev, gid)) in cfg.rails.iter().take(n_rails).enumerate() {
            let psn = (0x77_7777 ^ pid ^ ((i as u32) << 20)) & 0xff_ffff;
            rails.push(Verbs::create(dev, *gid, psn)?);
        }

        // Register each shard mmap (REMOTE_READ) on EVERY rail's PD — one rkey
        // per (rail, shard), identical base VA, shared physical pages. The mmaps
        // live in the persistent StagedModel, so they outlive these MRs.
        let mut per_rail_shards: Vec<Vec<(u64, u32)>> = (0..n_rails)
            .map(|_| Vec::with_capacity(num_shards))
            .collect();
        for m in &model.shard_mmaps {
            for (ri, v) in rails.iter_mut().enumerate() {
                // SAFETY: the mapping covers `m.len` bytes at `m.addr` and lives
                // in the StagedModel Arc, which outlives every rail here.
                let keys = unsafe { v.reg_mr(m.addr as *mut _, m.len, true)? };
                per_rail_shards[ri].push((m.addr as u64, keys.rkey));
            }
        }

        // Publish one VerbsServerParams per rail; `layers` carries per-SHARD
        // (base, rkey) in shard order (shards play experts' per-layer role).
        let sp: Vec<VerbsServerParams> = rails
            .iter()
            .enumerate()
            .map(|(ri, v)| VerbsServerParams {
                qpn: v.qpn(),
                psn: v.psn(),
                gid: v.gid(),
                layers: std::mem::take(&mut per_rail_shards[ri]),
            })
            .collect();
        write_server_rails(&mut stream, &sp).context("send verbs server params")?;

        // Learn each client rail's QP, connect, ack.
        stream.read_exact(&mut b1).context("read client n_rails")?;
        if b1[0] as usize != n_rails {
            bail!("client rail count mismatch");
        }
        for v in rails.iter_mut() {
            let cp =
                VerbsClientParams::read_from(&mut stream).context("read verbs client params")?;
            v.connect(cp.qpn, cp.psn, &cp.gid)?;
        }
        stream
            .write_all(&[STATUS_OK])
            .context("send verbs ready ack")?;
        tracing::info!(
            "weight-peer verbs client connected to {} ({n_rails} rail(s), {num_shards} shard MRs/rail)",
            model.manifest.model_id,
        );

        // Idle until the client hangs up. All movement is one-sided RDMA READ.
        let mut sink = [0u8; 8];
        loop {
            match stream.read(&mut sink) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        // Drop rails (dereg MRs) BEFORE the StagedModel Arc frees anything — the
        // mmaps persist in the map regardless, but dropping rails first keeps
        // dereg strictly over live mappings.
        drop(rails);
        Ok(())
    }

    /// Resolve shard files (index / single / glob), parse each shard's header,
    /// and assemble the [`WeightManifest`]. Mirrors the resolution order in
    /// `spark_runtime::fast_weights::header::resolve_shards`.
    fn build_manifest(dir: &Path, model_id: &str) -> Result<(Vec<PathBuf>, WeightManifest)> {
        let (shard_paths, weight_map) = resolve_shards(dir)?;

        let mut shard_files = Vec::with_capacity(shard_paths.len());
        let mut shard_lens = Vec::with_capacity(shard_paths.len());
        let mut tensors: Vec<WeightTensorRecord> = Vec::new();

        for (idx, path) in shard_paths.iter().enumerate() {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            let len = std::fs::metadata(path)
                .with_context(|| format!("stat {}", path.display()))?
                .len();
            // Only publish tensors the index actually routes (weight_map). A
            // shard header may list orphan/tied/aux tensors absent from the
            // index; the disk loaders iterate weight_map keys and never load
            // those, so filtering here keeps the RDMA store byte-identical
            // (same key set, not a superset). No index (single/glob) => keep all.
            parse_shard_header(path, idx as u32, false, weight_map.as_ref(), &mut tensors)?;
            shard_files.push(name);
            shard_lens.push(len);
        }

        // extra_weights.safetensors: an extra shard whose tensors are NEVER
        // expert-skipped (grafted MTP etc.), exactly like the disk loaders.
        let extra = dir.join("extra_weights.safetensors");
        if extra.exists() {
            let idx = shard_paths.len();
            let mut shard_paths = shard_paths.clone();
            let len = std::fs::metadata(&extra)?.len();
            // extra_weights.safetensors is never in the index weight_map; its
            // tensors are always fully published (extra=true), so pass no filter.
            parse_shard_header(&extra, idx as u32, true, None, &mut tensors)?;
            shard_files.push("extra_weights.safetensors".to_string());
            shard_lens.push(len);
            shard_paths.push(extra);
            let manifest = WeightManifest {
                version: WeightManifest::VERSION,
                model_id: model_id.to_string(),
                shard_files,
                shard_lens,
                tensors,
            };
            return Ok((shard_paths, manifest));
        }

        let manifest = WeightManifest {
            version: WeightManifest::VERSION,
            model_id: model_id.to_string(),
            shard_files,
            shard_lens,
            tensors,
        };
        Ok((shard_paths, manifest))
    }

    /// Resolved shard set: the shard file paths (shard-index order) plus the
    /// index `weight_map` (tensor name -> shard file) when an index exists, or
    /// `None` for a single-file / glob checkpoint (keep every header tensor).
    type ShardResolution = (Vec<PathBuf>, Option<HashMap<String, String>>);

    /// Shard discovery, resolution order identical to the disk loaders:
    /// (1) model.safetensors.index.json, else consolidated.safetensors.index.json;
    /// (2) single model.safetensors; (3) glob model.safetensors-* / consolidated-*.
    fn resolve_shards(dir: &Path) -> Result<ShardResolution> {
        let index = dir.join("model.safetensors.index.json");
        let consolidated = dir.join("consolidated.safetensors.index.json");
        let actual = if index.exists() {
            Some(index)
        } else if consolidated.exists() {
            Some(consolidated)
        } else {
            None
        };

        if let Some(ip) = actual {
            let json =
                std::fs::read_to_string(&ip).with_context(|| format!("read {}", ip.display()))?;
            let v: Value = serde_json::from_str(&json)?;
            let map = v
                .get("weight_map")
                .and_then(|m| m.as_object())
                .context("index json missing weight_map object")?;
            let weight_map: HashMap<String, String> = map
                .iter()
                .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_string())))
                .collect();
            let mut shards: Vec<String> = weight_map
                .values()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            shards.sort();
            let files = shards.iter().map(|s| dir.join(s)).collect();
            return Ok((files, Some(weight_map)));
        }

        let single = dir.join("model.safetensors");
        if single.exists() {
            return Ok((vec![single], None));
        }

        // PEFT adapter dir: a single `adapter_model.safetensors` (its keys are
        // the lora_A/lora_B tensors, classified client-side). Staged for LoRA
        // rotation over the RDMA tier (`weight_lora_rdma`).
        let adapter = dir.join("adapter_model.safetensors");
        if adapter.exists() {
            return Ok((vec![adapter], None));
        }

        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    (n.starts_with("model.safetensors-") || n.starts_with("consolidated-"))
                        && n.ends_with(".safetensors")
                })
            })
            .collect();
        shards.sort();
        if shards.is_empty() {
            bail!("no safetensor files found in {}", dir.display());
        }
        Ok((shards, None))
    }

    /// Parse one shard's safetensors header, pushing a [`WeightTensorRecord`]
    /// per tensor. Byte layout: `[u64 LE header_size][header_size JSON]`; data
    /// section starts at `8 + header_size`; each tensor's `data_offsets` are
    /// relative to that. We publish ABSOLUTE offsets so the client's
    /// `remote_addr = shard_base + offset_in_shard` reads out of the whole-file
    /// MR directly. Validates dtype against the disk loaders' closed set.
    fn parse_shard_header(
        path: &Path,
        shard_index: u32,
        extra: bool,
        weight_map: Option<&HashMap<String, String>>,
        out: &mut Vec<WeightTensorRecord>,
    ) -> Result<()> {
        let mut f =
            std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut size_buf = [0u8; 8];
        f.read_exact(&mut size_buf)
            .with_context(|| format!("read header size of {}", path.display()))?;
        let header_size = u64::from_le_bytes(size_buf) as usize;
        if header_size > 64 * 1024 * 1024 {
            bail!(
                "{}: safetensor header too large ({header_size} bytes)",
                path.display()
            );
        }
        let mut header_buf = vec![0u8; header_size];
        f.read_exact(&mut header_buf)
            .with_context(|| format!("read header of {}", path.display()))?;
        let data_start = 8 + header_size as u64;

        let json: Value = serde_json::from_slice(&header_buf)?;
        let obj = json
            .as_object()
            .with_context(|| format!("{}: header is not a JSON object", path.display()))?;

        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            // Skip header tensors the index doesn't route (orphan/tied/aux) so the
            // published set matches the disk loaders' weight_map iteration exactly.
            if let Some(map) = weight_map
                && !map.contains_key(name)
            {
                continue;
            }
            let dtype = info["dtype"].as_str().unwrap_or("BF16").to_string();
            validate_dtype(&dtype, name)?;
            let shape: Vec<u64> = info["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
                .unwrap_or_default();
            let offsets = info["data_offsets"]
                .as_array()
                .with_context(|| format!("tensor {name} missing data_offsets"))?;
            let rel_start = offsets[0].as_u64().context("bad data_offsets[0]")?;
            let rel_end = offsets[1].as_u64().context("bad data_offsets[1]")?;
            out.push(WeightTensorRecord {
                name: name.clone(),
                dtype,
                shape,
                offset_in_shard: data_start + rel_start,
                len: rel_end - rel_start,
                shard_index,
                extra,
            });
        }
        Ok(())
    }

    /// The disk loaders' closed dtype set (mirrors
    /// `WeightDtype::from_safetensors_str`). Fail at STAGE time on anything
    /// unsupported rather than shipping a bad manifest.
    fn validate_dtype(dtype: &str, tensor: &str) -> Result<()> {
        match dtype {
            // F16 is accepted for PEFT LoRA adapter shards (default PEFT save
            // is F32, some export F16); the `weight_lora_rdma` client converts
            // F16/F32 → BF16 host-side before landing. The base-weight disk
            // loaders never see F16 (they load model.safetensors, not adapters).
            "F32" | "F16" | "BF16" | "U8" | "I8" | "F8_E4M3" | "F8_E8M0" | "I64" => Ok(()),
            other => bail!("unsupported safetensors dtype '{other}' for tensor {tensor}"),
        }
    }

    /// A read-only whole-file `mmap`, warmed (MADV_WILLNEED) and unmapped on
    /// drop. Held persistently in a `StagedModel` so pages stay resident across
    /// connections (the weight-cache property). `Send`/`Sync`: the raw pointer
    /// is only ever handed to `ibv_reg_mr` as a base and read by the NIC; the
    /// Rust side never dereferences it.
    struct Mmap {
        addr: *mut libc::c_void,
        len: usize,
    }

    // SAFETY: see the doc above — addr/len are an immutable mapping description;
    // the memory is read only by the HCA, never mutated through this pointer.
    unsafe impl Send for Mmap {}
    unsafe impl Sync for Mmap {}

    impl Mmap {
        fn open_ro(path: &Path) -> Result<Self> {
            use std::os::fd::AsRawFd;
            let f = std::fs::File::open(path)?;
            let len = f.metadata()?.len() as usize;
            if len == 0 {
                bail!("empty shard file {}", path.display());
            }
            // SAFETY: fd is a valid open RO file; MAP_SHARED read mapping of
            // `len` bytes. The kernel keeps the mapping valid after the fd
            // closes.
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    f.as_raw_fd(),
                    0,
                )
            };
            if addr == libc::MAP_FAILED {
                bail!(
                    "mmap {} failed: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                );
            }
            // Warm the pages into RAM so the first (and every) RDMA read hits
            // resident memory — the cache property. Best-effort.
            // SAFETY: addr/len came from the successful mmap above.
            unsafe { libc::posix_madvise(addr, len, libc::POSIX_MADV_WILLNEED) };
            Ok(Self { addr, len })
        }
    }

    impl Drop for Mmap {
        fn drop(&mut self) {
            // SAFETY: addr/len came from a successful mmap and are unmapped once.
            unsafe { libc::munmap(self.addr, self.len) };
        }
    }

    #[cfg(test)]
    mod server_tests {
        use super::*;
        use std::io::Write;

        /// Write a minimal safetensors file: `[u64 LE header_len][header][data]`.
        fn write_st(path: &Path, header: &str, data: &[u8]) {
            let hb = header.as_bytes();
            let mut f = std::fs::File::create(path).unwrap();
            f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
            f.write_all(hb).unwrap();
            f.write_all(data).unwrap();
        }

        /// A shard header may list tensors the index `weight_map` does not route
        /// (orphan/tied/aux). The disk loaders iterate weight_map keys and never
        /// load those; `build_manifest` must filter identically so the RDMA store
        /// is byte-identical (same key set, not a superset).
        #[test]
        fn build_manifest_filters_orphan_tensors() {
            let dir = std::env::temp_dir().join(format!("wpeer-orphan-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let shard = "model-00001.safetensors";
            // Two f32[2] tensors in the header; only `a.weight` is in the index.
            let header = r#"{"a.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"orphan.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
            write_st(&dir.join(shard), header, &[0u8; 16]);
            std::fs::write(
                dir.join("model.safetensors.index.json"),
                format!(r#"{{"weight_map":{{"a.weight":"{shard}"}}}}"#),
            )
            .unwrap();

            let (_paths, manifest) = build_manifest(&dir, "test").unwrap();
            let names: Vec<&str> = manifest.tensors.iter().map(|t| t.name.as_str()).collect();
            assert_eq!(
                names,
                vec!["a.weight"],
                "orphan tensor (not in weight_map) must not be published"
            );

            std::fs::remove_dir_all(&dir).ok();
        }

        /// With no index (single-file checkpoint) every header tensor is kept —
        /// there is no weight_map to filter against.
        #[test]
        fn build_manifest_keeps_all_when_no_index() {
            let dir = std::env::temp_dir().join(format!("wpeer-single-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let header = r#"{"a.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
            write_st(&dir.join("model.safetensors"), header, &[0u8; 16]);

            let (_paths, manifest) = build_manifest(&dir, "test").unwrap();
            let mut names: Vec<&str> = manifest.tensors.iter().map(|t| t.name.as_str()).collect();
            names.sort();
            assert_eq!(names, vec!["a.weight", "b.weight"]);

            std::fs::remove_dir_all(&dir).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> WeightManifest {
        WeightManifest {
            version: WeightManifest::VERSION,
            model_id: "qwen3.6-35b-a3b".to_string(),
            shard_files: vec![
                "model.safetensors-00001-of-00002.safetensors".to_string(),
                "model.safetensors-00002-of-00002.safetensors".to_string(),
            ],
            shard_lens: vec![1_000_000, 2_000_000],
            tensors: vec![
                WeightTensorRecord {
                    name: "model.embed_tokens.weight".to_string(),
                    dtype: "BF16".to_string(),
                    shape: vec![152064, 4096],
                    offset_in_shard: 4096,
                    len: 152064 * 4096 * 2,
                    shard_index: 0,
                    extra: false,
                },
                WeightTensorRecord {
                    name: "mtp.experts.5.gate_proj.weight_packed".to_string(),
                    dtype: "I8".to_string(),
                    shape: vec![2048, 1024],
                    offset_in_shard: 8192,
                    len: 2048 * 1024,
                    shard_index: 1,
                    extra: true,
                },
            ],
        }
    }

    #[test]
    fn manifest_round_trips() {
        let m = sample_manifest();
        let mut buf = Vec::new();
        write_weight_manifest(&mut buf, &m).unwrap();
        let back = read_weight_manifest(&mut &buf[..]).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn manifest_rejects_bad_version() {
        let mut m = sample_manifest();
        m.version = 999;
        let mut buf = Vec::new();
        write_weight_manifest(&mut buf, &m).unwrap();
        assert!(read_weight_manifest(&mut &buf[..]).is_err());
    }

    #[test]
    fn manifest_rejects_shard_len_mismatch() {
        let mut m = sample_manifest();
        m.shard_lens.pop(); // now 2 files, 1 len
        let mut buf = Vec::new();
        write_weight_manifest(&mut buf, &m).unwrap();
        assert!(read_weight_manifest(&mut &buf[..]).is_err());
    }

    #[test]
    fn model_request_round_trips() {
        let mut buf = Vec::new();
        write_model_request(&mut buf, "/tank/models/qwen3.6-35b-a3b").unwrap();
        let back = read_model_request(&mut &buf[..]).unwrap();
        assert_eq!(back, "/tank/models/qwen3.6-35b-a3b");
    }

    #[test]
    fn model_request_rejects_empty() {
        let mut buf = Vec::new();
        assert!(write_model_request(&mut buf, "").is_err());
    }

    #[test]
    fn model_request_rejects_oversize_read() {
        // A hostile length prefix must not attempt a giant allocation.
        let mut buf = Vec::new();
        buf.extend_from_slice(&(MODEL_REQUEST_MAX as u32 + 1).to_le_bytes());
        assert!(read_model_request(&mut &buf[..]).is_err());
    }

    #[test]
    fn total_shard_bytes_sums_lens() {
        let m = sample_manifest();
        assert_eq!(m.total_shard_bytes(), 3_000_000);
        assert_eq!(m.num_shards(), 2);
    }

    #[test]
    fn tensor_remote_addr_adds_absolute_offset() {
        // remote_addr = shard MR base + the tensor's ABSOLUTE in-shard offset.
        // The offset already includes the 8-byte size prefix + header, so no
        // extra rebasing — a bug here reads header bytes off the shard front.
        assert_eq!(tensor_remote_addr(0x1_0000_0000, 4096), 0x1_0000_1000);
        assert_eq!(tensor_remote_addr(0, 0), 0);
        // Matches the manifest records verbatim (offset_in_shard is absolute).
        let m = sample_manifest();
        let base = 0xdead_0000u64;
        let t = &m.tensors[0];
        assert_eq!(tensor_remote_addr(base, t.offset_in_shard), base + 4096);
    }

    #[test]
    fn rail_for_tensor_stripes_round_robin() {
        // Single rail: everything on rail 0.
        for i in 0..5 {
            assert_eq!(rail_for_tensor(i, 1), 0);
        }
        // Dual rail: even → 0, odd → 1.
        assert_eq!(rail_for_tensor(0, 2), 0);
        assert_eq!(rail_for_tensor(1, 2), 1);
        assert_eq!(rail_for_tensor(2, 2), 0);
        assert_eq!(rail_for_tensor(3, 2), 1);
        // n_rails == 0 must not divide-by-zero (clamped to 1).
        assert_eq!(rail_for_tensor(7, 0), 0);
    }
}
