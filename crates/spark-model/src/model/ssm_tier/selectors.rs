// SPDX-License-Identifier: AGPL-3.0-only

//! Env-driven selectors: which store backs the Marconi spill tier and the
//! decode rolling tier.

use anyhow::{Result, bail};

use super::fingerprint::ModelFingerprint;
use super::unified::{build_unified_swap, unified_hot_slots};
use super::{
    ArenaSnapshotStore, FileSnapshotArena, MemBlobStore, SnapshotBlobStore, UnifiedSnapshotStore,
    ssm_tier_unified,
};

/// Whether the SSM spill tier is engaged (`ATLAS_SSM_TIER`). Default off ⇒
/// eviction drops exactly as before ⇒ byte-identical to a pre-tier build.
pub(crate) fn ssm_tier_enabled() -> bool {
    std::env::var_os("ATLAS_SSM_TIER").is_some()
}

/// Build the SSM spill-tier store (called only when `ssm_tier_enabled()`).
/// Local backends: the unified host-RAM [`UnifiedSnapshotStore`] when
/// `ATLAS_SSM_TIER_UNIFIED` is set (logging and falling back to host-RAM on a
/// build error — the tier is optional, never a hard model-init error),
/// otherwise `MemBlobStore::new(0)` — the byte-identical default. The RDMA arena
/// (`ATLAS_SSM_RDMA_TIER` / `ATLAS_SSM_SWAP`) binding lands in a follow-up PR.
pub(crate) fn build_tier_store(
    _fp: ModelFingerprint,
    blob_bytes: usize,
) -> Result<std::sync::Arc<dyn SnapshotBlobStore>> {
    use std::sync::Arc;
    if ssm_tier_unified() {
        // §4 fix (host-RAM arm): one policy core instead of the FIFO
        // MemBlobStore — a bounded LRU hot arena that spills (never rejects)
        // into the swap tier. NOTE: unlike today's lazily-growing unbounded
        // store, the hot arena is allocated up front (slots × blob_bytes).
        let hot_slots = unified_hot_slots();
        let hot = Box::new(atlas_tier::VecSlotArena::new(blob_bytes, hot_slots));
        let swap = build_unified_swap(blob_bytes, "marconi-host");
        match UnifiedSnapshotStore::new(hot, swap, blob_bytes) {
            Ok(s) => {
                tracing::info!(
                    "SSM spill tier = UNIFIED residency in host RAM ({hot_slots} hot slots × \
                     {blob_bytes} B, LRU spill, never rejects)"
                );
                return Ok(Arc::new(s));
            }
            Err(e) => tracing::warn!(
                "SSM unified residency init failed ({e:#}); falling back to host-RAM store"
            ),
        }
    }
    Ok(Arc::new(MemBlobStore::new(0)))
}

/// Build the **decode rolling-tier** cold store (a SEPARATE instance from the
/// Marconi `build_tier_store`, its own `ATLAS_SSM_DECODE_*` env namespace so keys
/// and budgets never collide). Non-dropping is a HARD requirement: a dropped
/// decode blob is a lost rollback target = corrupt restore (unlike Marconi's
/// miss→recompute). `min_slots` = `(ring_slots − hot_lanes) × max_batch_size` is
/// the worst-case cold residency; the local NVMe arena is sized ≥ that and its
/// undersizing is a preflight ERROR, never a warn.
///
/// Selection (`ATLAS_SSM_DECODE_TIER`):
///   - `nvme` + `ATLAS_SSM_DECODE_NVME_DIR=<dir>` → [`FileSnapshotArena`] behind
///     [`ArenaSnapshotStore`], provably sized ≥ `min_slots`.
///   - unset / anything else → unbounded host-RAM `MemBlobStore::new(0)`.
///
/// The `peer` (RDMA) decode tier binding lands in a follow-up PR.
pub(crate) fn build_decode_tier_store(
    _fp: ModelFingerprint,
    blob_bytes: usize,
    min_slots: usize,
) -> Result<std::sync::Arc<dyn SnapshotBlobStore>> {
    use std::sync::Arc;
    match std::env::var("ATLAS_SSM_DECODE_TIER").ok().as_deref() {
        Some("nvme") => {
            let dir = std::env::var("ATLAS_SSM_DECODE_NVME_DIR")
                .ok()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "ATLAS_SSM_DECODE_TIER=nvme requires ATLAS_SSM_DECODE_NVME_DIR=<dir>"
                    )
                })?;
            if ssm_tier_unified() && blob_bytes > 0 && blob_bytes.is_multiple_of(4096) {
                // §4 unification: a RAM hot cache over the lifted O_DIRECT swap
                // file. The uncapped disk tier (max_disk_slots = 0) makes
                // NON-DROPPING hold BY CONSTRUCTION instead of by arena sizing
                // — a decode rollback target can never be refused or dropped.
                std::fs::create_dir_all(&dir)?;
                let path = std::path::Path::new(&dir)
                    .join(format!("atlas-decode-ring.{}.swap", std::process::id()));
                let swap = atlas_tier::DirectSwapFile::create(&path, blob_bytes)?;
                let hot_slots = unified_hot_slots().min(min_slots + 1);
                let hot = Box::new(atlas_tier::VecSlotArena::new(blob_bytes, hot_slots));
                let store = UnifiedSnapshotStore::new(hot, Box::new(swap), blob_bytes)?;
                tracing::info!(
                    "SSM decode cold tier = UNIFIED residency ({hot_slots} hot RAM slots + \
                     O_DIRECT swap in {dir}; non-dropping by construction ≥ min_slots \
                     {min_slots})"
                );
                return Ok(Arc::new(store));
            }
            if ssm_tier_unified() {
                tracing::info!(
                    "SSM decode cold tier: ATLAS_SSM_TIER_UNIFIED set but blob_bytes \
                     {blob_bytes} is not a 4 KiB multiple (O_DIRECT stride); keeping the \
                     sized arena store"
                );
            }
            // Provision to the worst-case cold residency + headroom slot so the
            // non-dropping invariant holds by construction (an undersized arena
            // would return Ok(false) on a live target = corruption).
            let slots = min_slots + 1;
            let capacity = slots as u64 * blob_bytes as u64;
            let arena = FileSnapshotArena::create(&dir, capacity)?;
            tracing::info!(
                "SSM decode cold tier = LOCAL NVMe {dir} ({slots} slots × {blob_bytes} B = \
                 {:.2} GiB, non-dropping ≥ min_slots {min_slots})",
                capacity as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            Ok(Arc::new(ArenaSnapshotStore::new(
                Box::new(arena),
                blob_bytes,
                slots,
            )))
        }
        // Unset = the documented default: unbounded, non-dropping host RAM.
        None => {
            tracing::info!("SSM decode cold tier = host-RAM (unbounded, non-dropping)");
            Ok(Arc::new(MemBlobStore::new(0)))
        }
        // PCND: a typo ("nmve", "peer ", "") must never silently defeat the
        // tiering intent by falling through to unbounded host RAM, where decode
        // spills accumulate until OOM on a long session. Fail fast, name the
        // variable, the bad value, and the accepted values — mirroring the strict
        // `parse_ns` this chunk introduced one match arm away.
        Some(other) => bail!(
            "ATLAS_SSM_DECODE_TIER={other:?} is not recognized (accepted: \"nvme\", \"peer\", or \
             unset for unbounded host-RAM). Refusing to silently fall back to host-RAM."
        ),
    }
}

#[cfg(test)]
#[path = "selectors_tests.rs"]
mod tests;
