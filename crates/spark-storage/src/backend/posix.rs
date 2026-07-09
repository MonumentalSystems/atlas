// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-2 POSIX reference backend. Single pinned bounce buffer, `pread` +
// `cuMemcpyHtoDAsync`, stream-sync after every memcpy to avoid the next
// pread overwriting in-flight DMA. Slow-but-deterministic; used by tests as
// the oracle the io_uring backend is compared against.

use anyhow::{Context, Result, bail};
use std::ffi::c_void;

use super::{BlockReadRequest, ReadRequest, StorageBackend};
use crate::cuda_min::{PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout};
use crate::layout::Layout;

pub struct PosixBackend {
    layout: Layout,
    bounce: PinnedBuffer,
    /// Bytes one whole block occupies (== device slot_bytes). The block methods
    /// (ATLAS_HSS_COALESCE_BLOCKS) stage through `bounce`, so when `coalesce`
    /// was requested the bounce is sized to THIS instead of group_bytes.
    block_bytes: usize,
    /// ATLAS_HSS_COALESCE_RUNS (Tier-2): max consecutive-id blocks one merged
    /// pread may cover (== bounce bytes / block_bytes). `1` = the Tier-1 per-block
    /// path (run merging OFF); `read_blocks` dispatches on `r_max > 1`. When `> 1`
    /// the bounce is sized to `r_max · block_bytes`.
    r_max: usize,
}

impl PosixBackend {
    /// Un-coalesced backend (group-sized bounce) — the default, byte-identical
    /// to before ATLAS_HSS_COALESCE_BLOCKS existed.
    pub fn new(layout: Layout) -> Result<Self> {
        Self::new_with(layout, false)
    }

    /// `coalesce` sizes the pinned bounce to `block_bytes` so the block
    /// read/write methods can pread/pwrite one contiguous span. Set it iff the
    /// caller drives the block methods (flag threaded from `HighSpeedSwap::new`).
    pub fn new_with(layout: Layout, coalesce: bool) -> Result<Self> {
        Self::new_with_run_cap(layout, coalesce, 0)
    }

    /// Tier-2 (ATLAS_HSS_COALESCE_RUNS) constructor. `run_cap_bytes` bounds one
    /// merged pread (~1 MiB): `r_max = if coalesce && run_cap_bytes > block_bytes
    /// { max(1, run_cap_bytes / block_bytes) } else { 1 }`, so the pinned bounce
    /// is sized to `r_max · block_bytes`. `run_cap_bytes == 0` (the delegating
    /// `new`/`new_with`) ⇒ `r_max == 1`, byte- AND op-identical to Tier-1.
    pub fn new_with_run_cap(layout: Layout, coalesce: bool, run_cap_bytes: usize) -> Result<Self> {
        let group_bytes = layout.group_bytes() as usize;
        let block_bytes = layout.block_bytes() as usize;
        let r_max = if coalesce && run_cap_bytes > block_bytes {
            (run_cap_bytes / block_bytes).max(1)
        } else {
            1
        };
        let buf_bytes = if coalesce {
            block_bytes * r_max
        } else {
            group_bytes
        };
        let bounce = PinnedBuffer::new(buf_bytes).context("alloc pinned bounce buffer")?;
        Ok(Self {
            layout,
            bounce,
            block_bytes,
            r_max,
        })
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Tier-2 (ATLAS_HSS_COALESCE_RUNS) run-merged read: `plan_runs` splits the
    /// requests into maximal strictly-consecutive same-layer runs (capped at
    /// `r_max`); each run is ONE pread of `len·block_bytes` at
    /// `block_offset(run_start)`, then its `len` blocks are SCATTERED per-block
    /// from `bounce + i·block_bytes` to `sorted[start+i].dst_dev_ptr`, then ONE
    /// `stream_sync` per run (the shared bounce must not be overwritten by the
    /// next pread until all `len` H2Ds land). Byte-identical to Tier-1's `len`
    /// per-block reads — same bytes, same slots, one disk op instead of `len`.
    fn read_runs(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        let block_bytes = self.block_bytes;
        let (sorted, runs) = super::plan_runs(requests, self.r_max);
        let bounce_ptr = self.bounce.ptr;
        for (start, len) in runs {
            let read_len = len * block_bytes;
            if self.bounce.bytes < read_len {
                bail!(
                    "posix backend not built for run coalescing (bounce {} < run bytes {} = {} \
                     blocks × {}); construct with new_with_run_cap(.., run_cap_bytes)",
                    self.bounce.bytes,
                    read_len,
                    len,
                    block_bytes
                );
            }
            let rs = &sorted[start];
            let fd = self.layout.fd(rs.base_key.layer);
            let off = self
                .layout
                .block_offset(rs.base_key.layer, rs.base_key.block) as i64;
            let n = unsafe { libc::pread(fd, bounce_ptr, read_len, off) };
            if n != read_len as isize {
                bail!(
                    "run pread {read_len}@{off} returned {n}, errno {}",
                    std::io::Error::last_os_error()
                );
            }
            for i in 0..len {
                let dst = sorted[start + i].dst_dev_ptr;
                let src = unsafe { (bounce_ptr as *const u8).add(i * block_bytes) };
                copy_h_to_d_async(dst, src as *const c_void, block_bytes, stream)?;
            }
            stream_sync(stream)?;
        }
        Ok(())
    }
}

impl StorageBackend for PosixBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        let bounce_ptr = self.bounce.ptr;
        for req in requests {
            let fd = self.layout.fd(req.group.layer);
            let off = self.layout.offset(req.group) as i64;
            let n = unsafe { libc::pread(fd, bounce_ptr, bytes, off) };
            if n != bytes as isize {
                bail!(
                    "pread {bytes}@{off} returned {n}, errno {}",
                    std::io::Error::last_os_error()
                );
            }
            // The pinned bounce buffer is shared across all requests in this
            // call; we must let the H→D DMA complete before the next pread
            // overwrites the buffer, otherwise the second cuMemcpyHtoDAsync
            // will read partial / stale bytes. Phase-3 io_uring backend uses
            // multiple registered buffers and avoids this serialization.
            copy_h_to_d_async(req.dst_dev_ptr, bounce_ptr as *const c_void, bytes, stream)?;
            stream_sync(stream)?;
        }
        Ok(())
    }

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        if src.len() != bytes {
            bail!(
                "write_from_host: src len {} != group bytes {bytes}",
                src.len()
            );
        }
        // O_DIRECT requires page-aligned source. Stage through the pinned
        // bounce buffer (which is page-aligned per cuMemAllocHost contract).
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.bounce.ptr as *mut u8, bytes);
        }
        let fd = self.layout.fd(key.layer);
        let off = self.layout.offset(key) as i64;
        let n = unsafe { libc::pwrite(fd, self.bounce.ptr, bytes, off) };
        if n != bytes as isize {
            bail!(
                "pwrite {bytes}@{off} returned {n}, errno {}",
                std::io::Error::last_os_error()
            );
        }
        // fsync would be needed for crash durability; skipped for the test
        // path where the file is single-process / single-run.
        let _ = fd;
        Ok(())
    }

    fn group_layout(&self) -> GroupLayout {
        self.layout.spec
    }

    fn read_blocks(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        // Tier-2 (ATLAS_HSS_COALESCE_RUNS): merge consecutive-id runs into one
        // pread each. r_max == 1 (flag off) falls through to the Tier-1 per-block
        // body below, byte- AND op-identical.
        if self.r_max > 1 {
            return self.read_runs(requests, stream);
        }
        // ONE pread + ONE copy_h_to_d + ONE stream_sync PER BLOCK (2·nkv fewer
        // syncs than the per-head path — the QD=1 serialisation win).
        let bytes = self.block_bytes;
        if self.bounce.bytes < bytes {
            bail!(
                "posix backend not built for block coalescing (bounce {} < block_bytes {}); \
                 construct with new_with(.., coalesce=true)",
                self.bounce.bytes,
                bytes
            );
        }
        let bounce_ptr = self.bounce.ptr;
        for req in requests {
            let fd = self.layout.fd(req.base_key.layer);
            let off = self
                .layout
                .block_offset(req.base_key.layer, req.base_key.block) as i64;
            let n = unsafe { libc::pread(fd, bounce_ptr, bytes, off) };
            if n != bytes as isize {
                bail!(
                    "block pread {bytes}@{off} returned {n}, errno {}",
                    std::io::Error::last_os_error()
                );
            }
            copy_h_to_d_async(req.dst_dev_ptr, bounce_ptr as *const c_void, bytes, stream)?;
            stream_sync(stream)?;
        }
        Ok(())
    }

    fn read_blocks_async(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        // Posix has no true async (single shared bounce); same body as
        // read_blocks, mirroring `read_async` delegating to `read`.
        self.read_blocks(requests, stream)
    }

    fn write_block_from_host(&mut self, base_key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.block_bytes;
        if src.len() != bytes {
            bail!(
                "write_block_from_host: src len {} != block bytes {bytes}",
                src.len()
            );
        }
        if self.bounce.bytes < bytes {
            bail!(
                "posix backend not built for block coalescing (bounce {} < block_bytes {}); \
                 construct with new_with(.., coalesce=true)",
                self.bounce.bytes,
                bytes
            );
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.bounce.ptr as *mut u8, bytes);
        }
        let fd = self.layout.fd(base_key.layer);
        let off = self.layout.block_offset(base_key.layer, base_key.block) as i64;
        let n = unsafe { libc::pwrite(fd, self.bounce.ptr, bytes, off) };
        if n != bytes as isize {
            bail!(
                "block pwrite {bytes}@{off} returned {n}, errno {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(())
    }

    fn supports_write_run_coalescing(&self) -> bool {
        // bounce is r_max·block_bytes when built with a run cap (Tier-2).
        self.r_max > 1
    }

    fn write_blocks_run(&mut self, base_key: GroupKey, run_len: usize, src: &[u8]) -> Result<()> {
        // Tier-2-WRITE (ATLAS_HSS_COALESCE_WRITE_RUNS): ONE wide sync pwrite of
        // run_len·block_bytes at block_offset(run_start) through the page-aligned
        // bounce. Byte-identical to run_len per-block pwrites.
        let block_bytes = self.block_bytes;
        let run_bytes = run_len * block_bytes;
        if src.len() != run_bytes {
            bail!(
                "write_blocks_run: src len {} != run bytes {run_bytes} ({run_len} × {block_bytes})",
                src.len()
            );
        }
        if self.bounce.bytes < run_bytes {
            bail!(
                "posix backend not built for write-run coalescing (bounce {} < run bytes {} = \
                 {run_len} × {block_bytes}); construct with new_with_run_cap(.., run_cap_bytes)",
                self.bounce.bytes,
                run_bytes
            );
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.bounce.ptr as *mut u8, run_bytes);
        }
        let fd = self.layout.fd(base_key.layer);
        let off = self.layout.block_offset(base_key.layer, base_key.block) as i64;
        let n = unsafe { libc::pwrite(fd, self.bounce.ptr, run_bytes, off) };
        if n != run_bytes as isize {
            bail!(
                "run pwrite {run_bytes}@{off} returned {n}, errno {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(())
    }
}

impl PosixBackend {
    /// Test helper: drop the page cache for the layer files so subsequent
    /// reads actually hit NVMe (otherwise small tests trivially read from RAM).
    pub fn drop_pagecache(&self) {
        for layer in 0..self.layout.spec.num_layers {
            let fd = self.layout.fd(layer);
            unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::{GroupLayout, KvKind};

    fn tempdir(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("atlas-storage-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    #[ignore = "requires GPU"]
    fn write_then_read_round_trip() {
        // CUDA must be initialised before any pinned-host allocation.
        let _ctx = crate::cuda_min::CudaCtx::new(0).expect("cuda init");
        let dir = tempdir("rt");
        let spec = GroupLayout::new(1, 2, 1, 16, 128, 2, 4096);
        let layout = Layout::create(&dir, spec).unwrap();
        let mut backend = PosixBackend::new(layout).unwrap();
        let bytes = backend.layout().group_bytes() as usize;
        let pat: Vec<u8> = (0..bytes).map(|i| (i & 0xFF) as u8).collect();
        let key = GroupKey::new(0, 1, 0, KvKind::V);
        backend.write_from_host(key, &pat).unwrap();
        backend.drop_pagecache();

        let dev = crate::cuda_min::DeviceBuffer::new(bytes).unwrap();
        let req = ReadRequest {
            group: key,
            dst_dev_ptr: dev.ptr,
        };
        // Construct a stream from the (already-existing) ctx to satisfy the
        // backend signature.
        backend.read(&[req], _ctx.stream).unwrap();
        let mut host_back = vec![0_u8; bytes];
        crate::cuda_min::copy_d_to_h_async(
            host_back.as_mut_ptr() as *mut c_void,
            dev.ptr,
            bytes,
            _ctx.stream,
        )
        .unwrap();
        crate::cuda_min::stream_sync(_ctx.stream).unwrap();
        assert_eq!(host_back, pat);
        std::fs::remove_dir_all(&dir).ok();
    }
}
