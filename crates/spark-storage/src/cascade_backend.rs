// SPDX-License-Identifier: AGPL-3.0-only
//
// CascadeBackend — a T1 local pinned-LPDDR write-back cache in front of any
// `StorageBackend` backing tier (the RDMA peer, or the SSD io_uring backend).
//
// The KV cache overflow already spills to `backing`; this inserts the handoff's
// missing middle tier: hot groups live in a bounded pinned-host cache (fast,
// GPU-addressable, no RDMA), and only evicted groups flush DOWN to `backing`.
// A restore hits T1 (a local copy_h2d) or falls through to `backing`. Purely a
// placement layer — no tier transforms bytes, so the composite is bit-identical
// to the backing alone (the group-id -> address bijection is the same on every
// tier). Enabled by $ATLAS_KV_LOCAL_GB; 0 (default) leaves the path untouched.

use std::ffi::c_void;

use anyhow::{Context, Result};

use crate::backend::{ReadRequest, StorageBackend};
use crate::cascade_policy::SlotCache;
use crate::cuda_min::{PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout};

/// The T1 byte store: one pinned buffer of `cap_slots * group_bytes`, slot `i`
/// at `ptr + i*group_bytes`. Pinned so the flush-read is a plain host slice and
/// the restore copy_h2d is fast (same LPDDR is GPU-addressable on GB10).
struct PinnedStore {
    buf: PinnedBuffer,
    group_bytes: usize,
}

impl PinnedStore {
    fn new(cap_slots: u32, group_bytes: usize) -> Result<Self> {
        let bytes = (cap_slots as usize)
            .checked_mul(group_bytes)
            .context("CascadeBackend T1 size overflow")?;
        Ok(Self {
            buf: PinnedBuffer::new(bytes).context("alloc T1 pinned store")?,
            group_bytes,
        })
    }
    #[inline]
    fn slot_host_ptr(&self, slot: u32) -> *const c_void {
        // SAFETY: slot < cap_slots (SlotCache invariant); offset within the buf.
        unsafe {
            (self.buf.ptr as *const u8).add(slot as usize * self.group_bytes) as *const c_void
        }
    }
    /// Copy the group bytes for `slot` out into a fresh Vec (releases the borrow
    /// so the flush can call `&mut backing`).
    fn slot_bytes(&self, slot: u32) -> Vec<u8> {
        // SAFETY: as above; the slot holds `group_bytes` valid bytes.
        unsafe {
            std::slice::from_raw_parts(
                (self.buf.ptr as *const u8).add(slot as usize * self.group_bytes),
                self.group_bytes,
            )
            .to_vec()
        }
    }
    fn write_slot(&mut self, slot: u32, src: &[u8]) {
        debug_assert_eq!(src.len(), self.group_bytes);
        // SAFETY: slot in range; src is exactly group_bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                (self.buf.ptr as *mut u8).add(slot as usize * self.group_bytes),
                self.group_bytes,
            );
        }
    }
}

pub struct CascadeBackend {
    hot: SlotCache,
    store: PinnedStore,
    backing: Box<dyn StorageBackend>,
    group_bytes: usize,
}

// Single-owner rationale identical to RdmaKvBackend: both trait methods take
// `&mut self`, no `&self` method touches shared state, and HighSpeedSwap owns it
// single-threaded. The pinned store's raw ptr is only used under `&mut self`.
unsafe impl Sync for CascadeBackend {}

impl CascadeBackend {
    pub fn new(
        backing: Box<dyn StorageBackend>,
        layout: GroupLayout,
        cap_slots: u32,
    ) -> Result<Self> {
        let group_bytes = layout.group_bytes() as usize;
        tracing::info!(
            "high-speed-swap: T1 cascade cache = {cap_slots} slots × {group_bytes} B = {:.1} GiB local pinned RAM, backing below",
            (cap_slots as f64 * group_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
        );
        Ok(Self {
            hot: SlotCache::new(cap_slots),
            store: PinnedStore::new(cap_slots, group_bytes)?,
            backing,
            group_bytes,
        })
    }

    /// Flush every resident T1 group down to backing (durability on teardown).
    fn flush_all(&mut self) -> Result<()> {
        for (key, slot) in self.hot.residents() {
            let bytes = self.store.slot_bytes(slot);
            self.backing.write_from_host(key, &bytes)?;
        }
        Ok(())
    }
}

impl StorageBackend for CascadeBackend {
    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let plan = self.hot.plan_write(key);
        // Evicted a resident group → flush its (still-in-slot) bytes DOWN first.
        if let Some((victim_key, victim_slot)) = plan.flush_victim {
            let victim_bytes = self.store.slot_bytes(victim_slot);
            self.backing
                .write_from_host(victim_key, &victim_bytes)
                .context("cascade: flush T1 victim to backing")?;
        }
        // Then cache the new group in T1 (write-back — lives here until evicted).
        self.store.write_slot(plan.slot, src);
        Ok(())
    }

    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        let keys: Vec<GroupKey> = requests.iter().map(|r| r.group).collect();
        let (hits, misses) = self.hot.plan_read(&keys);
        // T1 hits: local copy_h2d straight from the pinned slot into HBM.
        for (i, slot) in &hits {
            let src = self.store.slot_host_ptr(*slot);
            copy_h_to_d_async(requests[*i].dst_dev_ptr, src, self.group_bytes, stream)?;
            self.hot.touch(*slot);
        }
        // Misses fall through to backing (peer RDMA or SSD). Non-promoting: a
        // miss is NOT pulled up into T1 (smaller correctness surface; write-back
        // populates T1). backing.read syncs the stream for its own dsts; the
        // trailing stream_sync also covers the hit copy_h2d above.
        if !misses.is_empty() {
            let miss_reqs: Vec<ReadRequest> = misses.iter().map(|&i| requests[i]).collect();
            self.backing.read(&miss_reqs, stream)?;
        }
        stream_sync(stream)?;
        Ok(())
    }

    fn register_landing_region(&mut self, base: u64, len: usize) -> Result<()> {
        // Forward to backing so RDMA zero-copy restore of MISSES still lands
        // directly into the UMA pool. (T1 hits copy_h2d locally regardless.)
        self.backing.register_landing_region(base, len)
    }
}

impl Drop for CascadeBackend {
    fn drop(&mut self) {
        // Durability: push any T1-resident groups down to backing before the
        // pinned store frees. Best-effort on teardown.
        let _ = self.flush_all();
    }
}
