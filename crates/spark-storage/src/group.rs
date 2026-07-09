// SPDX-License-Identifier: AGPL-3.0-only
//
// Group identity for the high-speed-swap layer.
//
// A *group* is the unit of NVMe ↔ HBM movement: one (layer, block, kv_head)
// tuple's K-stripe or V-stripe. Each group is `block_size × head_dim ×
// elem_bytes` bytes contiguous on disk, sized to round up to the device's
// optimal I/O block (typically 4 KiB).
//
// The bijection (layer, block, kv_head) ⇆ group_id is computed deterministically
// from the dimensions; we never store the inverse mapping. Group IDs are dense
// 64-bit integers.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroupId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvKind {
    K = 0,
    V = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupKey {
    pub layer: u32,
    pub block: u32,
    pub kv_head: u16,
    pub kv_kind: u8, // KvKind as u8 (no Hash on enum w/o derive on payload)
}

impl GroupKey {
    pub fn new(layer: u32, block: u32, kv_head: u16, kv_kind: KvKind) -> Self {
        Self {
            layer,
            block,
            kv_head,
            kv_kind: kv_kind as u8,
        }
    }
    pub fn kind(self) -> KvKind {
        match self.kv_kind {
            0 => KvKind::K,
            _ => KvKind::V,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GroupLayout {
    pub num_layers: u32,
    pub num_blocks: u32,
    pub num_kv_heads: u16,
    /// `block_size × head_dim × elem_bytes`, rounded up to fs_block_size.
    pub group_stride: u64,
    /// Filesystem block size (4 KiB on most NVMe). Group stride is a multiple.
    pub fs_block_size: u64,
}

impl GroupLayout {
    pub fn new(
        num_layers: u32,
        num_blocks: u32,
        num_kv_heads: u16,
        block_size: u32,
        head_dim: u32,
        elem_bytes: u32,
        fs_block_size: u64,
    ) -> Self {
        let raw = (block_size as u64) * (head_dim as u64) * (elem_bytes as u64);
        let group_stride = raw.div_ceil(fs_block_size) * fs_block_size;
        Self {
            num_layers,
            num_blocks,
            num_kv_heads,
            group_stride,
            fs_block_size,
        }
    }

    /// Bytes occupied by one full layer in its file (K + V across all blocks).
    pub fn bytes_per_layer(&self) -> u64 {
        2 * (self.num_blocks as u64) * (self.num_kv_heads as u64) * self.group_stride
    }

    /// File offset for `key` within its layer's file.
    pub fn file_offset(&self, key: GroupKey) -> u64 {
        debug_assert!(key.block < self.num_blocks);
        debug_assert!(key.kv_head < self.num_kv_heads);
        let kv_stride = (self.num_kv_heads as u64) * self.group_stride;
        (key.block as u64) * (2 * kv_stride)
            + (key.kv_kind as u64) * kv_stride
            + (key.kv_head as u64) * self.group_stride
    }

    /// Dense `GroupId` for `key`.
    pub fn group_id(&self, key: GroupKey) -> GroupId {
        let per_layer = 2 * (self.num_blocks as u64) * (self.num_kv_heads as u64);
        let per_block = 2 * (self.num_kv_heads as u64);
        GroupId(
            (key.layer as u64) * per_layer
                + (key.block as u64) * per_block
                + (key.kv_kind as u64) * (self.num_kv_heads as u64)
                + (key.kv_head as u64),
        )
    }

    /// Number of bytes a single group occupies on disk (== group_stride).
    pub fn group_bytes(&self) -> u64 {
        self.group_stride
    }

    /// File offset of the BASE of `block` within its layer's file — i.e. the
    /// start of the block's contiguous `[K0,K1,…,K(nkv-1),V0,…,V(nkv-1)]` span.
    ///
    /// Equal by construction to `file_offset(GroupKey::new(_, block, 0, K))`,
    /// but computed WITHOUT routing through kv_head/kind: a coalesced (whole-
    /// block) I/O op targets this base and covers `block_bytes()` — provably the
    /// union of the 2·nkv per-head group offsets (`base + i·group_stride` for
    /// `i = kind·nkv + kv_head`). Used by the ATLAS_HSS_COALESCE_BLOCKS path.
    pub fn block_offset(&self, block: u32) -> u64 {
        debug_assert!(block < self.num_blocks);
        let kv_stride = (self.num_kv_heads as u64) * self.group_stride;
        (block as u64) * (2 * kv_stride)
    }

    /// Bytes one whole block occupies on disk: `2 · num_kv_heads · group_stride`
    /// (all kv_heads, K and V). Byte-for-byte equal to the device side's
    /// `ScratchDims::slot_bytes()`, so a single op of this size at
    /// `block_offset ↔ slot_dev_ptr` reproduces the 2·nkv per-head copies.
    pub fn block_bytes(&self) -> u64 {
        2 * (self.num_kv_heads as u64) * self.group_stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_and_id_are_consistent() {
        let l = GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096);
        // BF16: 16 * 128 * 2 = 4096 bytes — already aligned.
        assert_eq!(l.group_stride, 4096);
        let k0 = GroupKey::new(0, 0, 0, KvKind::K);
        assert_eq!(l.file_offset(k0), 0);
        let k1 = GroupKey::new(0, 0, 0, KvKind::V);
        assert_eq!(l.file_offset(k1), (8u64) * 4096);
        let k2 = GroupKey::new(0, 1, 0, KvKind::K);
        assert_eq!(l.file_offset(k2), 2 * 8 * 4096);
        let k3 = GroupKey::new(0, 0, 7, KvKind::V);
        assert_eq!(l.file_offset(k3), 8 * 4096 + 7 * 4096);
    }

    #[test]
    fn rounds_up_to_fs_block() {
        // Hypothetical odd shape: block_size=16, head_dim=96, BF16 → 3 KiB raw.
        let l = GroupLayout::new(1, 1, 1, 16, 96, 2, 4096);
        assert_eq!(l.group_stride, 4096);
    }

    #[test]
    fn bytes_per_layer_correct() {
        let l = GroupLayout::new(1, 4, 2, 16, 128, 2, 4096);
        // 4 blocks * 2 kv_heads * 4096 bytes * 2 (K+V) = 65536
        assert_eq!(l.bytes_per_layer(), 65536);
    }

    // ── ATLAS_HSS_COALESCE_BLOCKS: block-granular offset/size math ──────────

    /// block_offset(b) is exactly the kh=0/K base and the closed-form
    /// b·2·nkv·group_stride, for the first, second, and last block.
    #[test]
    fn block_offset_is_the_k0_base() {
        let l = GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096);
        let nkv = l.num_kv_heads as u64;
        for &b in &[0u32, 1, l.num_blocks - 1] {
            assert_eq!(
                l.block_offset(b),
                l.file_offset(GroupKey::new(3, b, 0, KvKind::K)),
                "block_offset must equal the kh=0/K per-head offset"
            );
            assert_eq!(l.block_offset(b), (b as u64) * 2 * nkv * l.group_stride);
        }
    }

    /// block_bytes == 2·nkv·group_stride == 2·kv_stride.
    #[test]
    fn block_bytes_is_two_kv_stride() {
        let l = GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096);
        let kv_stride = (l.num_kv_heads as u64) * l.group_stride;
        assert_eq!(l.block_bytes(), 2 * (l.num_kv_heads as u64) * l.group_stride);
        assert_eq!(l.block_bytes(), 2 * kv_stride);
    }

    /// LOAD-BEARING span coverage: the 2·nkv per-head group offsets of a fixed
    /// block, sorted, start at block_offset, step by exactly group_stride with
    /// no gap/overlap/dup, and end at block_offset + block_bytes. This is the
    /// proof that ONE contiguous block_bytes op covers exactly the union of the
    /// per-head spans. Also pins the device index map: the i-th sorted per-head
    /// span sits at block_offset + i·group_stride, matching the slot pointers
    /// (K at slot_base + kh·gs, V at slot_base + (nkv+kh)·gs).
    #[test]
    fn per_head_spans_tile_the_block_exactly() {
        let l = GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096);
        let block = 5u32;
        let nkv = l.num_kv_heads;
        let gs = l.group_stride;
        let mut offs: Vec<u64> = Vec::new();
        for kind in [KvKind::K, KvKind::V] {
            for kh in 0..nkv {
                offs.push(l.file_offset(GroupKey::new(0, block, kh, kind)));
            }
        }
        offs.sort_unstable();
        // Dense, gapless, dup-free tiling from the block base.
        assert_eq!(*offs.first().unwrap(), l.block_offset(block));
        for w in offs.windows(2) {
            assert_eq!(w[1] - w[0], gs, "per-head spans must be contiguous (no gap/overlap)");
        }
        assert_eq!(
            *offs.last().unwrap() + gs,
            l.block_offset(block) + l.block_bytes(),
            "sorted spans must end exactly at block_offset + block_bytes"
        );
        assert_eq!(offs.len() as u64, l.block_bytes() / gs);
        // Index map: i = kind*nkv + kh ⇒ offset - block_offset == i*gs.
        for kh in 0..nkv {
            let ik = 0u64 * nkv as u64 + kh as u64;
            let iv = 1u64 * nkv as u64 + kh as u64;
            assert_eq!(
                l.file_offset(GroupKey::new(0, block, kh, KvKind::K)) - l.block_offset(block),
                ik * gs
            );
            assert_eq!(
                l.file_offset(GroupKey::new(0, block, kh, KvKind::V)) - l.block_offset(block),
                iv * gs
            );
        }
    }

    /// Blocks tile the layer file with no inter-block gap (consistent with
    /// bytes_per_layer): block_offset(b+1) - block_offset(b) == block_bytes.
    #[test]
    fn blocks_tile_the_file() {
        let l = GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096);
        for b in 0..8u32 {
            assert_eq!(l.block_offset(b + 1) - l.block_offset(b), l.block_bytes());
        }
        // And the whole file is num_blocks worth of block_bytes.
        assert_eq!(
            (l.num_blocks as u64) * l.block_bytes(),
            l.bytes_per_layer()
        );
    }

    /// Padded shape (head_dim=96 → raw 3072 rounds up to group_stride 4096):
    /// block_offset and block_bytes stay 4096-aligned (O_DIRECT safe) and the
    /// per-head spans still tile the block exactly with the tail padding.
    #[test]
    fn padded_head_dim_stays_aligned_and_tiled() {
        let l = GroupLayout::new(4, 16, 3, 16, 96, 2, 4096);
        assert_eq!(l.group_stride, 4096, "3 KiB raw rounds up to one fs block");
        let block = 2u32;
        assert_eq!(l.block_offset(block) % 4096, 0, "block base O_DIRECT-aligned");
        assert_eq!(l.block_bytes() % 4096, 0, "block size O_DIRECT-aligned");
        let mut offs: Vec<u64> = Vec::new();
        for kind in [KvKind::K, KvKind::V] {
            for kh in 0..l.num_kv_heads {
                offs.push(l.file_offset(GroupKey::new(0, block, kh, kind)));
            }
        }
        offs.sort_unstable();
        assert_eq!(*offs.first().unwrap(), l.block_offset(block));
        for w in offs.windows(2) {
            assert_eq!(w[1] - w[0], l.group_stride);
        }
        assert_eq!(*offs.last().unwrap() + l.group_stride, l.block_offset(block) + l.block_bytes());
    }
}
