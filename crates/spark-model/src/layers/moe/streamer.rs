// SPDX-License-Identifier: AGPL-3.0-only
//
// Model-wide expert streamer shared by every MoE layer (Arc).
//
// Stage 3 (async prefetch overlap): a background worker thread fetches the NEXT
// MoE layer's local experts into its ring slab while the GPU computes the
// CURRENT layer, so the fetch I/O (NVMe / peer) hides under compute. The main
// (compute) thread only ever patches pointer tables from residency ADDRESSES —
// it never touches arena bytes — so there is no CPU/CPU data race; the worker
// owns the tier exclusively.
//
// Deferred-free (invariant C): the arena is a ring of K = expert_arena_layers
// slabs. When K < num_moe_layers a slab is reused. Before the worker overwrites
// a slab, the compute thread waits on a per-slab CUDA event recorded AFTER the
// previous occupant's grouped GEMM — so the GPU has finished reading a slab
// before the worker refills it. All event record/sync happens on the compute
// thread; the worker only does CPU I/O.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result, bail};
use atlas_core::config::ModelConfig;

use spark_storage::CudaEvent;
use spark_storage::ExpertIndex;
use spark_storage::expert::ExpertKey;
use spark_storage::expert_tier::{ArenaSlot, ExpertResidency, TierKind, open_tier};

/// A prefetch job for dense MoE layer `dense`.
///
/// `experts == None` is the DENSE job: fetch the whole local range `[lo, hi)`
/// (the layer-ahead overlap path). `experts == Some(ids)` is the REACTIVE
/// (expert-granular) job: fetch ONLY those global expert ids, each into its
/// identity slot `slot = e - lo` (invariant E: every id must lie in `[lo, hi)`).
struct Job {
    dense: u32,
    lo: u32,
    hi: u32,
    experts: Option<Vec<u32>>,
}

/// dense-layer -> the fetched residencies (indexed by `expert - lo`), or an error.
type DoneMap = HashMap<u32, std::result::Result<Vec<ExpertResidency>, String>>;

pub struct ExpertStreamerShared {
    tx: Sender<Job>,
    done: Arc<(Mutex<DoneMap>, Condvar)>,
    requested: Mutex<HashSet<u32>>,
    /// WS3 persistent residency: dense-layer -> its cached arena residencies.
    /// A layer stays here across prefills until its slab is reused by another
    /// layer. When the working set fits (num_slabs >= num_moe_layers), warm
    /// prefills are pure cache hits — zero expert I/O.
    resident_cache: Mutex<HashMap<u32, Vec<ExpertResidency>>>,
    /// Which dense layer currently owns each slab (arena occupancy). A cached
    /// layer's arena bytes are valid iff it still owns its slab.
    slab_occupant: Mutex<Vec<Option<u32>>>,
    /// Per-slab consumer events (compute-thread only). `slab_events[s]` marks
    /// the GEMM of slab `s`'s current occupant complete.
    slab_events: Vec<CudaEvent>,
    num_slabs: u32,
    num_moe_layers: u32,
    kind: TierKind,
    // Worker handle kept alive for the streamer's lifetime; dropping the Sender
    // ends the worker loop.
    _worker: std::thread::JoinHandle<()>,
}

impl ExpertStreamerShared {
    /// Open the store, size the arena ring, spawn the fetch worker.
    pub fn open(config: &ModelConfig) -> Result<Self> {
        let dir = config
            .expert_store_dir
            .clone()
            .context("expert_streaming set but --stream-experts dir is None")?;
        let index = ExpertIndex::load(&dir)?;
        if index.num_experts as usize != config.num_experts {
            bail!(
                "expert store has {} experts but model config has {}",
                index.num_experts,
                config.num_experts
            );
        }
        let (lo, hi) = config.local_expert_range();
        let slots_per_slab = (hi - lo) as u32;
        if slots_per_slab == 0 {
            bail!("expert streaming: local_expert_range is empty");
        }
        let num_moe_layers = index.num_moe_layers;
        let num_slabs = if config.expert_arena_layers == 0 {
            num_moe_layers
        } else {
            (config.expert_arena_layers as u32).clamp(1, num_moe_layers)
        };
        let backend = config.expert_backend.clone();

        // The worker owns the tier (single-threaded tier access).
        let mut tier = open_tier(&backend, &dir, num_slabs, slots_per_slab)?;
        let kind = tier.kind();
        tracing::info!(
            "expert streamer: backend={:?} store={} experts={} local={} slabs={} \
             (arena_layers={}) async-prefetch record_stride={}",
            kind,
            dir.display(),
            index.num_experts,
            slots_per_slab,
            num_slabs,
            config.expert_arena_layers,
            index.record_stride,
        );

        let slab_events = (0..num_slabs)
            .map(|_| CudaEvent::new())
            .collect::<Result<Vec<_>>>()?;

        let (tx, rx) = channel::<Job>();
        let done: Arc<(Mutex<DoneMap>, Condvar)> =
            Arc::new((Mutex::new(HashMap::new()), Condvar::new()));
        let worker_done = done.clone();
        let worker = std::thread::Builder::new()
            .name("expert-prefetch".into())
            .spawn(move || {
                // NULL stream for the worker's tier ops (Posix's copy_h2d syncs
                // internally; UMA/RDMA ignore the stream — pure CPU I/O).
                let stream = 0u64;
                while let Ok(job) = rx.recv() {
                    // Dense job → the whole local range; reactive job → exactly the
                    // active ids (same order the caller passed, so `take_active`'s
                    // residency vec zips positionally with that id list).
                    let ids: Vec<u32> = match &job.experts {
                        Some(v) => v.clone(),
                        None => (job.lo..job.hi).collect(),
                    };
                    let mut out = Vec::with_capacity(ids.len());
                    let mut err: Option<String> = None;
                    for &e in &ids {
                        let slot = ArenaSlot::new(job.dense % num_slabs, e - job.lo);
                        match tier.fetch(ExpertKey::new(job.dense, e), slot, stream) {
                            Ok(res) => out.push(res),
                            Err(e2) => {
                                err = Some(format!("layer {} expert {e}: {e2:#}", job.dense));
                                break;
                            }
                        }
                    }
                    let (lock, cv) = &*worker_done;
                    let mut map = lock.lock().expect("prefetch done mutex");
                    map.insert(job.dense, err.map_or(Ok(out), Err));
                    cv.notify_all();
                }
            })
            .context("spawn expert-prefetch worker")?;

        Ok(Self {
            tx,
            done,
            requested: Mutex::new(HashSet::new()),
            resident_cache: Mutex::new(HashMap::new()),
            slab_occupant: Mutex::new(vec![None; num_slabs as usize]),
            slab_events,
            num_slabs,
            num_moe_layers,
            kind,
            _worker: worker,
        })
    }

    pub fn num_moe_layers(&self) -> u32 {
        self.num_moe_layers
    }
    #[allow(dead_code)]
    pub fn num_slabs(&self) -> u32 {
        self.num_slabs
    }
    #[allow(dead_code)]
    pub fn kind(&self) -> TierKind {
        self.kind
    }

    /// Request a prefetch of dense layer `dense`. WS3: a cache hit (the layer is
    /// still resident in its slab) enqueues NO fetch — only a miss/eviction does.
    pub fn prefetch(&self, dense: u32, lo: u32, hi: u32) {
        if dense >= self.num_moe_layers {
            return;
        }
        let slab = (dense % self.num_slabs) as usize;
        let mut occ = self.slab_occupant.lock().expect("occupant mutex");
        let cache = self.resident_cache.lock().expect("cache mutex");
        // Resident hit: still owns its slab and cached → arena bytes are valid,
        // no I/O needed.
        if occ[slab] == Some(dense) && cache.contains_key(&dense) {
            return;
        }
        drop(cache);
        // Miss/eviction: evict the current occupant's cache entry (its arena
        // bytes are about to be overwritten) and claim the slab.
        if let Some(old) = occ[slab]
            && old != dense
        {
            self.resident_cache
                .lock()
                .expect("cache mutex")
                .remove(&old);
        }
        occ[slab] = Some(dense);
        drop(occ);
        let mut req = self.requested.lock().expect("requested mutex");
        if req.insert(dense) {
            let _ = self.tx.send(Job {
                dense,
                lo,
                hi,
                experts: None,
            });
        }
    }

    /// REACTIVE (expert-granular) fetch: enqueue a fetch of ONLY `active` (the
    /// global expert ids with routed token count > 0) for dense layer `dense`.
    /// Each id lands in its identity slot `slot = e - lo`; inactive slots keep
    /// their prior bytes (the grouped GEMM never indexes a count==0 expert).
    ///
    /// This BYPASSES the WS3 resident_cache (per-token active sets vary) and
    /// INVALIDATES it: the arena slab is about to be partially overwritten, so
    /// any later DENSE prefetch of this layer must re-fetch the full range.
    /// Correctness: the caller must `wait_slab_free(dense)` before this (the
    /// prior GEMM reading this slab must complete) and pair each reactive
    /// install with a `take_active(dense)`.
    pub fn prefetch_sparse(&self, dense: u32, active: &[u32], lo: u32, hi: u32) {
        if dense >= self.num_moe_layers {
            return;
        }
        let slab = (dense % self.num_slabs) as usize;
        // Invalidate WS3 state for this slab. The sparse write is about to
        // overwrite the arena bytes of WHATEVER layer currently occupies this
        // slab — which, when num_slabs < num_moe_layers, is a DIFFERENT layer
        // than `dense`. Clear the occupant unconditionally and drop the cached
        // full-layer residency of BOTH that prior occupant and `dense`; else a
        // later dense prefetch of the evicted layer would see occ[slab]==Some(it)
        // + a cache hit and skip the re-fetch, reading the overwritten bytes.
        let prev = {
            let mut occ = self.slab_occupant.lock().expect("occupant mutex");
            occ[slab].take()
        };
        {
            let mut cache = self.resident_cache.lock().expect("cache mutex");
            if let Some(p) = prev {
                cache.remove(&p);
            }
            cache.remove(&dense);
        }
        // Direct send (no `requested` dedup): reactive fetches are per-forward
        // and consumed immediately by `take_active`, which clears DoneMap[dense].
        let _ = self.tx.send(Job {
            dense,
            lo,
            hi,
            experts: Some(active.to_vec()),
        });
    }

    /// Block until the reactive fetch for `dense` lands; return its residencies
    /// in the SAME order as the `active` id slice passed to `prefetch_sparse`.
    /// Unlike `take`, this never consults / promotes the resident_cache.
    pub fn take_active(&self, dense: u32) -> Result<Vec<ExpertResidency>> {
        let (lock, cv) = &*self.done;
        let mut map = lock.lock().expect("done mutex");
        loop {
            if let Some(entry) = map.remove(&dense) {
                return entry.map_err(|e| anyhow::anyhow!(e));
            }
            map = cv.wait(map).expect("done condvar");
        }
    }

    /// Return dense layer `dense`'s residencies. WS3: a resident cache hit
    /// returns immediately; otherwise block on the worker fetch and promote the
    /// result into the persistent cache.
    pub fn take(&self, dense: u32) -> Result<Vec<ExpertResidency>> {
        // Fast path: resident in cache.
        if let Some(res) = self.resident_cache.lock().expect("cache mutex").get(&dense) {
            return Ok(res.clone());
        }
        // Slow path: wait for the worker fetch, then cache it.
        let (lock, cv) = &*self.done;
        let mut map = lock.lock().expect("done mutex");
        loop {
            if let Some(entry) = map.remove(&dense) {
                self.requested
                    .lock()
                    .expect("requested mutex")
                    .remove(&dense);
                let res = entry.map_err(|e| anyhow::anyhow!(e))?;
                self.resident_cache
                    .lock()
                    .expect("cache mutex")
                    .insert(dense, res.clone());
                return Ok(res);
            }
            map = cv.wait(map).expect("done condvar");
        }
    }

    /// Record (on the compute `stream`, after the GEMM) that dense layer
    /// `dense`'s slab has been consumed by the GPU.
    pub fn record_consumed(&self, dense: u32, stream: u64) -> Result<()> {
        self.slab_events[(dense % self.num_slabs) as usize].record(stream)
    }

    /// Block the compute thread until the slab that dense layer `dense` will
    /// land in has had its previous occupant's GEMM complete (deferred-free).
    /// A never-recorded event (first use of a slab) returns immediately.
    pub fn wait_slab_free(&self, dense: u32) -> Result<()> {
        self.slab_events[(dense % self.num_slabs) as usize].sync()
    }
}
