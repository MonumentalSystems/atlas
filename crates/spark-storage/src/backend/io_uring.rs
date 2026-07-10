// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-3 production storage backend: `io_uring` (IORING_SETUP_SQPOLL +
// IORING_REGISTER_BUFFERS) + per-buffer `CudaEvent` for safe reuse across
// async H→D DMAs. Per-buffer events let us keep QD≥8 in flight without the
// per-op `cuStreamSynchronize` that throttled the POSIX backend to QD=1.

use anyhow::{Context, Result, bail};
use io_uring::{IoUring, opcode, types};
use std::ffi::c_void;
use std::os::fd::RawFd;

use super::{BlockReadRequest, ReadRequest, StorageBackend};
use crate::cuda_min::{CudaEvent, PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout};
use crate::layout::Layout;

pub struct IoUringBackend {
    layout: Layout,
    ring: IoUring,
    buffers: Vec<PinnedBuffer>,
    events: Vec<Option<CudaEvent>>, // event per buffer, None = idle
    qd: usize,
    /// Bytes one whole block occupies (== device slot_bytes). Cached for the
    /// ATLAS_HSS_COALESCE_BLOCKS single-op read/write. When `coalesce` is set
    /// the pinned buffers below are sized to THIS (not group_bytes) so a
    /// block ReadFixed / pwrite fits — a block op into a group-sized registered
    /// iovec would be silent corruption, so the block methods hard-check it.
    block_bytes: usize,
    /// ATLAS_HSS_COALESCE_RUNS (Tier-2): max consecutive-id blocks one merged
    /// ReadFixed may cover (== registered buffer bytes / block_bytes). `1` = the
    /// Tier-1 per-block path (run merging OFF); the block read methods dispatch on
    /// `r_max > 1`. When `> 1` the registered buffers below are sized to
    /// `r_max · block_bytes` so a run ReadFixed of that length fits.
    r_max: usize,
}

impl IoUringBackend {
    /// Un-coalesced backend (group-sized buffers) — the default, byte-identical
    /// to before ATLAS_HSS_COALESCE_BLOCKS existed.
    pub fn new(layout: Layout, qd: usize) -> Result<Self> {
        Self::new_with(layout, qd, false)
    }

    /// `coalesce` sizes the pinned/registered buffers to `block_bytes` (=
    /// `2·nkv·group_stride`) so the block read/write methods can issue ONE
    /// contiguous op. Set it iff the caller will drive the block methods
    /// (ATLAS_HSS_COALESCE_BLOCKS on); the flag is threaded from
    /// `HighSpeedSwap::new` so buffer sizing and the caller's dispatch agree.
    pub fn new_with(layout: Layout, qd: usize, coalesce: bool) -> Result<Self> {
        Self::new_with_run_cap(layout, qd, coalesce, 0)
    }

    /// Tier-2 (ATLAS_HSS_COALESCE_RUNS) constructor. `run_cap_bytes` bounds one
    /// merged read (~1 MiB): `r_max = if coalesce && run_cap_bytes > block_bytes
    /// { max(1, run_cap_bytes / block_bytes) } else { 1 }`, so the registered
    /// pinned buffers are sized to `r_max · block_bytes`. `run_cap_bytes == 0`
    /// (the delegating `new`/`new_with`) ⇒ `r_max == 1`, byte- AND op-identical to
    /// the Tier-1 per-block path (the run body is dead code). Threaded from
    /// `HighSpeedSwap::new` so buffer sizing and the flag agree.
    pub fn new_with_run_cap(
        layout: Layout,
        qd: usize,
        coalesce: bool,
        run_cap_bytes: usize,
    ) -> Result<Self> {
        if qd == 0 {
            bail!("queue depth must be ≥ 1");
        }
        // SQPOLL: kernel polls SQ; idle 2s before parking.
        let ring = IoUring::builder()
            .setup_sqpoll(2_000)
            .build(qd as u32)
            .context("io_uring build")?;

        let group_bytes = layout.group_bytes() as usize;
        let block_bytes = layout.block_bytes() as usize;
        // Tier-2 run merging presupposes Tier-1 block coalescing (block-sized or
        // wider buffers). r_max blocks per merged ReadFixed; 1 = Tier-1 per-block.
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
        let mut buffers = Vec::with_capacity(qd);
        for _ in 0..qd {
            buffers.push(PinnedBuffer::new(buf_bytes)?);
        }
        // Register the pinned host buffers with io_uring for zero-copy
        // direct-IO. After this, ReadFixed at index `i` lands in `buffers[i]`.
        let iovecs: Vec<libc::iovec> = buffers
            .iter()
            .map(|b| libc::iovec {
                iov_base: b.ptr,
                iov_len: b.bytes,
            })
            .collect();
        unsafe {
            ring.submitter()
                .register_buffers(&iovecs)
                .context("register_buffers")?;
        }
        let events: Vec<Option<CudaEvent>> = (0..qd).map(|_| None).collect();
        Ok(Self {
            layout,
            ring,
            buffers,
            events,
            qd,
            block_bytes,
            r_max,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Test helper: drop the page cache for the layer files so subsequent
    /// reads actually hit NVMe.
    pub fn drop_pagecache(&self) {
        for layer in 0..self.layout.spec.num_layers {
            let fd = self.layout.fd(layer);
            unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }

    /// Wait for the previous DMA out of `buf_idx` to complete (if any) so
    /// we can reuse the buffer for a new io_uring read.
    fn wait_buffer_free(&mut self, buf_idx: usize) -> Result<()> {
        if let Some(ev) = self.events[buf_idx].take() {
            ev.sync()?;
        }
        Ok(())
    }

    /// Submit one read request into `buf_idx` and return its user_data tag.
    fn submit_read(
        &mut self,
        fd: RawFd,
        offset: u64,
        bytes: u32,
        buf_idx: u16,
        user_data: u64,
    ) -> Result<()> {
        let buf_ptr = self.buffers[buf_idx as usize].ptr as *mut u8;
        let read_e = opcode::ReadFixed::new(types::Fd(fd), buf_ptr, bytes, buf_idx)
            .offset(offset)
            .build()
            .user_data(user_data);
        unsafe {
            self.ring
                .submission()
                .push(&read_e)
                .map_err(|_| anyhow::anyhow!("io_uring SQ full"))?;
        }
        Ok(())
    }
}

impl IoUringBackend {
    /// Body shared by `read` (sync) and `read_async`. When `sync_at_end` is
    /// true it is textually the pre-#11-refinement `read`: terminal `stream_sync`
    /// \+ event wipe. When false (async prefetch), it drops BOTH — the persistent
    /// per-buffer `self.events` are kept so `wait_buffer_free` gates cross-call
    /// bounce reuse device-side without a host sync, and mirror-RAW is closed by
    /// the caller's `kv_prefetch_done`.
    fn read_inner(
        &mut self,
        requests: &[ReadRequest],
        stream: u64,
        sync_at_end: bool,
    ) -> Result<()> {
        let bytes = self.layout.group_bytes() as u32;
        // user_data layout: high 16 bits = req index, low 16 bits = buf index.
        // (We never submit > 65k requests in one batch.)
        if requests.len() > u16::MAX as usize {
            bail!("io_uring batch too large: {}", requests.len());
        }

        let mut next_submit = 0;
        let mut completed = 0;
        // Buffer ownership: free buffers form a stack; busy ones are claimed
        // by an in-flight read until its CQE arrives.
        let mut free_bufs: Vec<u16> = (0..self.qd as u16).rev().collect();

        while completed < requests.len() {
            // Submit while we have a free buffer and pending requests.
            while next_submit < requests.len() {
                let Some(&buf_idx) = free_bufs.last() else {
                    break;
                };
                self.wait_buffer_free(buf_idx as usize)?;
                free_bufs.pop();
                let req = &requests[next_submit];
                let fd = self.layout.fd(req.group.layer);
                let off = self.layout.offset(req.group);
                let user = ((next_submit as u64) << 16) | (buf_idx as u64);
                self.submit_read(fd, off, bytes, buf_idx, user)?;
                next_submit += 1;
            }
            // Submit and wait for at least one completion.
            self.ring
                .submit_and_wait(1)
                .context("io_uring submit_and_wait")?;
            // Drain everything that's ready.
            let cq = self.ring.completion();
            for cqe in cq {
                let user = cqe.user_data();
                let buf_idx = (user & 0xFFFF) as u16;
                let req_idx = (user >> 16) as usize;
                let result = cqe.result();
                if result < 0 {
                    bail!("io_uring read failed for req {req_idx}: errno {}", -result);
                }
                if result as u32 != bytes {
                    bail!("io_uring short read: req {req_idx} got {result}, expected {bytes}");
                }
                let req = &requests[req_idx];
                let buf = &self.buffers[buf_idx as usize];
                copy_h_to_d_async(
                    req.dst_dev_ptr,
                    buf.ptr as *const c_void,
                    bytes as usize,
                    stream,
                )?;
                let ev = CudaEvent::new()?;
                ev.record(stream)?;
                self.events[buf_idx as usize] = Some(ev);
                free_bufs.push(buf_idx);
                completed += 1;
            }
        }
        if sync_at_end {
            // After all reads have produced device data, finalise the stream
            // (matches PosixBackend semantics: at return, the stream is synced).
            stream_sync(stream)?;
            // Drop now-completed events; they are useful only across calls.
            // (Async path KEEPS them so wait_buffer_free gates the next call's
            // bounce reuse without a host sync.)
            for slot in self.events.iter_mut() {
                *slot = None;
            }
        }
        Ok(())
    }

    /// ATLAS_HSS_COALESCE_BLOCKS block-read body — structurally the per-head
    /// `read_inner` (same QD ring, free-buf stack, per-buffer CudaEvent reuse,
    /// terminal-sync discipline) but each op moves ONE contiguous `block_bytes`
    /// span at the block base offset into the slot base. Kept SEPARATE from
    /// `read_inner` so the per-head flag-OFF path stays textually + op-identical.
    fn read_block_inner(
        &mut self,
        requests: &[BlockReadRequest],
        stream: u64,
        sync_at_end: bool,
    ) -> Result<()> {
        // Guard the sizing coupling: a block ReadFixed into a group-sized
        // registered iovec is silent corruption. Fail loud instead.
        if self.buffers[0].bytes < self.block_bytes {
            bail!(
                "io_uring backend not built for block coalescing (buffer {} < block_bytes {}); \
                 construct with new_with(.., coalesce=true) — the ATLAS_HSS_COALESCE_BLOCKS flag \
                 must be set before backend construction",
                self.buffers[0].bytes,
                self.block_bytes
            );
        }
        let bytes = self.block_bytes as u32;
        if requests.len() > u16::MAX as usize {
            bail!("io_uring batch too large: {}", requests.len());
        }
        let mut next_submit = 0;
        let mut completed = 0;
        let mut free_bufs: Vec<u16> = (0..self.qd as u16).rev().collect();

        while completed < requests.len() {
            while next_submit < requests.len() {
                let Some(&buf_idx) = free_bufs.last() else {
                    break;
                };
                self.wait_buffer_free(buf_idx as usize)?;
                free_bufs.pop();
                let req = &requests[next_submit];
                let fd = self.layout.fd(req.base_key.layer);
                let off = self
                    .layout
                    .block_offset(req.base_key.layer, req.base_key.block);
                let user = ((next_submit as u64) << 16) | (buf_idx as u64);
                self.submit_read(fd, off, bytes, buf_idx, user)?;
                next_submit += 1;
            }
            self.ring
                .submit_and_wait(1)
                .context("io_uring submit_and_wait")?;
            let cq = self.ring.completion();
            for cqe in cq {
                let user = cqe.user_data();
                let buf_idx = (user & 0xFFFF) as u16;
                let req_idx = (user >> 16) as usize;
                let result = cqe.result();
                if result < 0 {
                    bail!("io_uring block read failed for req {req_idx}: errno {}", -result);
                }
                if result as u32 != bytes {
                    bail!("io_uring short block read: req {req_idx} got {result}, expected {bytes}");
                }
                let req = &requests[req_idx];
                let buf = &self.buffers[buf_idx as usize];
                copy_h_to_d_async(
                    req.dst_dev_ptr,
                    buf.ptr as *const c_void,
                    bytes as usize,
                    stream,
                )?;
                let ev = CudaEvent::new()?;
                ev.record(stream)?;
                self.events[buf_idx as usize] = Some(ev);
                free_bufs.push(buf_idx);
                completed += 1;
            }
        }
        if sync_at_end {
            stream_sync(stream)?;
            for slot in self.events.iter_mut() {
                *slot = None;
            }
        }
        Ok(())
    }

    /// ATLAS_HSS_COALESCE_RUNS (Tier-2) run-merged block-read body — a SEPARATE
    /// sibling of `read_block_inner` so the flag-OFF (`r_max == 1`) path stays
    /// textually + op-identical to Tier-1. `plan_runs` sorts a copy of the
    /// requests by (layer, block) and splits it into maximal strictly-consecutive
    /// same-layer runs capped at `r_max`. Each run is ONE `ReadFixed` of
    /// `len·block_bytes` at `block_offset(run_start)`; on completion its
    /// `len` blocks are SCATTERED per-block from `buf + i·block_bytes` to
    /// `sorted[start+i].dst_dev_ptr` (the dst rides inside the sorted request, so
    /// it can never desync from its block id). The single per-buffer `CudaEvent`
    /// is recorded AFTER the whole scatter loop — the buffer must survive all
    /// `len` H2Ds before `wait_buffer_free` lets a later ReadFixed clobber it.
    fn read_run_inner(
        &mut self,
        requests: &[BlockReadRequest],
        stream: u64,
        sync_at_end: bool,
    ) -> Result<()> {
        let block_bytes = self.block_bytes;
        let (sorted, runs) = super::plan_runs(requests, self.r_max);
        // user_data high 16 bits = run index; we never plan > 65k runs per batch.
        if runs.len() > u16::MAX as usize {
            bail!("io_uring run batch too large: {}", runs.len());
        }
        let mut next_submit = 0;
        let mut completed = 0;
        let mut free_bufs: Vec<u16> = (0..self.qd as u16).rev().collect();

        while completed < runs.len() {
            while next_submit < runs.len() {
                let Some(&buf_idx) = free_bufs.last() else {
                    break;
                };
                let (start, len) = runs[next_submit];
                let read_len = len * block_bytes;
                // Sizing coupling guard: a run ReadFixed into a too-small
                // registered iovec is silent corruption (it does NOT spill into
                // the next buffer — the kernel bounds-checks the iovec). Fail loud.
                if self.buffers[buf_idx as usize].bytes < read_len {
                    bail!(
                        "io_uring backend not built for run coalescing (buffer {} < run bytes {} \
                         = {} blocks × {}); construct with new_with_run_cap(.., run_cap_bytes) — \
                         ATLAS_HSS_COALESCE_RUNS must be set before backend construction",
                        self.buffers[buf_idx as usize].bytes,
                        read_len,
                        len,
                        block_bytes
                    );
                }
                self.wait_buffer_free(buf_idx as usize)?;
                free_bufs.pop();
                let rs = &sorted[start];
                let fd = self.layout.fd(rs.base_key.layer);
                let off = self
                    .layout
                    .block_offset(rs.base_key.layer, rs.base_key.block);
                let user = ((next_submit as u64) << 16) | (buf_idx as u64);
                self.submit_read(fd, off, read_len as u32, buf_idx, user)?;
                next_submit += 1;
            }
            self.ring
                .submit_and_wait(1)
                .context("io_uring submit_and_wait")?;
            let cq = self.ring.completion();
            for cqe in cq {
                let user = cqe.user_data();
                let buf_idx = (user & 0xFFFF) as u16;
                let run_idx = (user >> 16) as usize;
                let result = cqe.result();
                let (start, len) = runs[run_idx];
                let read_len = len * block_bytes;
                if result < 0 {
                    bail!("io_uring run read failed for run {run_idx}: errno {}", -result);
                }
                if result as usize != read_len {
                    bail!(
                        "io_uring short run read: run {run_idx} got {result}, expected {read_len}"
                    );
                }
                let buf = &self.buffers[buf_idx as usize];
                // Scatter: one contiguous disk read → `len` per-block H2Ds, each
                // to its own slot. Byte-identical to Tier-1's `len` per-block
                // reads (same bytes, same per-block offset, same dst slot).
                for i in 0..len {
                    let dst = sorted[start + i].dst_dev_ptr;
                    let src = unsafe { (buf.ptr as *const u8).add(i * block_bytes) };
                    copy_h_to_d_async(dst, src as *const c_void, block_bytes, stream)?;
                }
                // ONE event AFTER the full scatter loop (buffer reuse safety).
                let ev = CudaEvent::new()?;
                ev.record(stream)?;
                self.events[buf_idx as usize] = Some(ev);
                free_bufs.push(buf_idx);
                completed += 1;
            }
        }
        if sync_at_end {
            stream_sync(stream)?;
            for slot in self.events.iter_mut() {
                *slot = None;
            }
        }
        Ok(())
    }
}

impl StorageBackend for IoUringBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        // Byte-identical to the pre-refinement path: records per-buffer events,
        // terminal stream_sync, then wipes the events.
        self.read_inner(requests, stream, true)
    }

    fn read_async(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        // No terminal host sync: the H2Ds stay in flight on `stream`, mirror-RAW
        // is closed by the caller's `kv_prefetch_done`, and cross-call bounce
        // reuse is gated by the persisted per-buffer events via wait_buffer_free.
        self.read_inner(requests, stream, false)
    }

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        if src.len() != bytes {
            bail!(
                "write_from_host: src len {} != group bytes {bytes}",
                src.len()
            );
        }
        // Stage through buffer 0 — pinned + page-aligned for O_DIRECT.
        self.wait_buffer_free(0)?;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.buffers[0].ptr as *mut u8, bytes);
        }
        let fd = self.layout.fd(key.layer);
        let off = self.layout.offset(key) as i64;
        let n = unsafe { libc::pwrite(fd, self.buffers[0].ptr, bytes, off) };
        if n != bytes as isize {
            bail!(
                "pwrite {bytes}@{off} returned {n}, errno {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(())
    }

    fn group_layout(&self) -> GroupLayout {
        self.layout.spec
    }

    fn read_blocks(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        // Tier-2 (ATLAS_HSS_COALESCE_RUNS) when r_max > 1; otherwise the Tier-1
        // per-block path, textually unchanged. Both terminal stream_sync.
        if self.r_max > 1 {
            self.read_run_inner(requests, stream, true)
        } else {
            self.read_block_inner(requests, stream, true)
        }
    }

    fn read_blocks_async(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        // No terminal host sync (prefetch path) — mirrors `read_async`.
        if self.r_max > 1 {
            self.read_run_inner(requests, stream, false)
        } else {
            self.read_block_inner(requests, stream, false)
        }
    }

    fn write_block_from_host(&mut self, base_key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.block_bytes;
        if src.len() != bytes {
            bail!(
                "write_block_from_host: src len {} != block bytes {bytes}",
                src.len()
            );
        }
        // buffers[0] must be block-sized (the write path stages through it too).
        if self.buffers[0].bytes < bytes {
            bail!(
                "io_uring backend not built for block coalescing (buffer {} < block_bytes {}); \
                 construct with new_with(.., coalesce=true)",
                self.buffers[0].bytes,
                bytes
            );
        }
        self.wait_buffer_free(0)?;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.buffers[0].ptr as *mut u8, bytes);
        }
        let fd = self.layout.fd(base_key.layer);
        let off = self.layout.block_offset(base_key.layer, base_key.block) as i64;
        let n = unsafe { libc::pwrite(fd, self.buffers[0].ptr, bytes, off) };
        if n != bytes as isize {
            bail!(
                "block pwrite {bytes}@{off} returned {n}, errno {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(())
    }

    fn supports_write_run_coalescing(&self) -> bool {
        // buffers[0] is r_max·block_bytes when built with a run cap (Tier-2);
        // r_max > 1 means a multi-block wide pwrite fits.
        self.r_max > 1
    }

    fn write_blocks_run(&mut self, base_key: GroupKey, run_len: usize, src: &[u8]) -> Result<()> {
        // Tier-2-WRITE (ATLAS_HSS_COALESCE_WRITE_RUNS): ONE wide sync pwrite of
        // run_len·block_bytes at block_offset(run_start), staged through the
        // registered pinned buffers[0]. Byte-identical to run_len per-block
        // pwrites (block_offset is linear/gapless) — only the op count collapses.
        let block_bytes = self.block_bytes;
        let run_bytes = run_len * block_bytes;
        if src.len() != run_bytes {
            bail!(
                "write_blocks_run: src len {} != run bytes {run_bytes} ({run_len} × {block_bytes})",
                src.len()
            );
        }
        // buffers[0] must hold the whole run (sized r_max·block_bytes when the
        // run cap was threaded in). Bail loud rather than truncate/overrun.
        if self.buffers[0].bytes < run_bytes {
            bail!(
                "io_uring backend not built for write-run coalescing (buffer {} < run bytes {} = \
                 {run_len} × {block_bytes}); construct with new_with_run_cap(.., run_cap_bytes)",
                self.buffers[0].bytes,
                run_bytes
            );
        }
        self.wait_buffer_free(0)?;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.buffers[0].ptr as *mut u8, run_bytes);
        }
        let fd = self.layout.fd(base_key.layer);
        let off = self.layout.block_offset(base_key.layer, base_key.block) as i64;
        let n = unsafe { libc::pwrite(fd, self.buffers[0].ptr, run_bytes, off) };
        if n != run_bytes as isize {
            bail!(
                "run pwrite {run_bytes}@{off} returned {n}, errno {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_min::{CudaCtx, DeviceBuffer, copy_d_to_h_async};
    use crate::group::{GroupKey, GroupLayout, KvKind};
    use std::path::PathBuf;

    fn tempdir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("atlas-iouring-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    #[ignore = "requires GPU"]
    fn write_then_read_round_trip() {
        let _ctx = CudaCtx::new(0).expect("cuda init");
        let dir = tempdir("rt");
        let spec = GroupLayout::new(1, 4, 1, 16, 128, 2, 4096);
        let layout = Layout::create(&dir, spec).unwrap();
        let mut backend = IoUringBackend::new(layout, 4).unwrap();
        let bytes = backend.layout().group_bytes() as usize;
        // Three different patterns at three different keys to exercise SQ depth.
        let patterns: Vec<(GroupKey, Vec<u8>)> = (0..4u32)
            .map(|b| {
                let k = GroupKey::new(0, b, 0, KvKind::K);
                let pat: Vec<u8> = (0..bytes)
                    .map(|i| ((i + b as usize) & 0xFF) as u8)
                    .collect();
                (k, pat)
            })
            .collect();
        for (k, p) in &patterns {
            backend.write_from_host(*k, p).unwrap();
        }
        backend.drop_pagecache();
        let dev: Vec<DeviceBuffer> = patterns
            .iter()
            .map(|_| DeviceBuffer::new(bytes).unwrap())
            .collect();
        let reqs: Vec<ReadRequest> = patterns
            .iter()
            .zip(&dev)
            .map(|((k, _), d)| ReadRequest {
                group: *k,
                dst_dev_ptr: d.ptr,
            })
            .collect();
        backend.read(&reqs, _ctx.stream).unwrap();
        for ((_, expected), d) in patterns.iter().zip(&dev) {
            let mut got = vec![0_u8; bytes];
            copy_d_to_h_async(got.as_mut_ptr() as *mut c_void, d.ptr, bytes, _ctx.stream).unwrap();
            stream_sync(_ctx.stream).unwrap();
            assert_eq!(&got, expected);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// #11-refinement: `read_async` (no terminal host `stream_sync`) must land
    /// byte-identical data to `read`, AND its persisted per-buffer events must
    /// gate cross-call bounce reuse without a host sync. We force reuse by using
    /// qd (2) < requests (4), then issue TWO back-to-back `read_async` calls (the
    /// second reuses buffers the first left in flight — `wait_buffer_free` on the
    /// persisted events is the only thing preventing an H2D-vs-refill race). A
    /// caller-side `stream_sync` (standing in for the HSS `kv_prefetch_done`
    /// consumer) precedes each readback. Any missing reuse gate → corruption.
    #[test]
    #[ignore = "requires GPU"]
    fn read_async_matches_read_and_reuses_bounces_safely() {
        let _ctx = CudaCtx::new(0).expect("cuda init");
        let dir = tempdir("async-rt");
        // 4 blocks, qd=2 so every batch laps the 2-buffer ring at least twice.
        let spec = GroupLayout::new(1, 4, 1, 16, 128, 2, 4096);
        let layout = Layout::create(&dir, spec).unwrap();
        let mut backend = IoUringBackend::new(layout, 2).unwrap();
        let bytes = backend.layout().group_bytes() as usize;
        let patterns: Vec<(GroupKey, Vec<u8>)> = (0..4u32)
            .map(|b| {
                let k = GroupKey::new(0, b, 0, KvKind::K);
                let pat: Vec<u8> = (0..bytes).map(|i| ((i * 3 + b as usize) & 0xFF) as u8).collect();
                (k, pat)
            })
            .collect();
        for (k, p) in &patterns {
            backend.write_from_host(*k, p).unwrap();
        }
        backend.drop_pagecache();

        let readback = |backend: &mut IoUringBackend, dev: &[DeviceBuffer], async_mode: bool| {
            let reqs: Vec<ReadRequest> = patterns
                .iter()
                .zip(dev)
                .map(|((k, _), d)| ReadRequest {
                    group: *k,
                    dst_dev_ptr: d.ptr,
                })
                .collect();
            if async_mode {
                backend.read_async(&reqs, _ctx.stream).unwrap();
                // Stand in for the HSS kv_prefetch_done consumer: the async path
                // does NOT sync, so the caller must before reading device memory.
                stream_sync(_ctx.stream).unwrap();
            } else {
                backend.read(&reqs, _ctx.stream).unwrap();
            }
            for ((_, expected), d) in patterns.iter().zip(dev) {
                let mut got = vec![0_u8; bytes];
                copy_d_to_h_async(got.as_mut_ptr() as *mut c_void, d.ptr, bytes, _ctx.stream)
                    .unwrap();
                stream_sync(_ctx.stream).unwrap();
                assert_eq!(&got, expected, "read_async landed wrong bytes");
            }
        };

        let dev_sync: Vec<DeviceBuffer> = patterns.iter().map(|_| DeviceBuffer::new(bytes).unwrap()).collect();
        let dev_a: Vec<DeviceBuffer> = patterns.iter().map(|_| DeviceBuffer::new(bytes).unwrap()).collect();
        let dev_b: Vec<DeviceBuffer> = patterns.iter().map(|_| DeviceBuffer::new(bytes).unwrap()).collect();

        readback(&mut backend, &dev_sync, false); // sync oracle
        readback(&mut backend, &dev_a, true); // async, first call (persists events)
        readback(&mut backend, &dev_b, true); // async, reuses buffers across calls
        std::fs::remove_dir_all(&dir).ok();
    }
}
