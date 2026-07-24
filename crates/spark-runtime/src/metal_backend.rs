// SPDX-License-Identifier: AGPL-3.0-only

//! Apple Metal GPU backend.
//!
//! Implements [`GpuBackend`] on top of the Metal framework via the
//! `objc2-metal` bindings. Apple Silicon is unified-memory (UMA), so
//! every `MTLBuffer` is allocated with `StorageModeShared` and host
//! `memcpy` against `buffer.contents()` is the canonical H2D/D2H path
//! — no PCIe staging, no pinned-host bounce.
//!
//! # Pointer model
//!
//! `DevicePtr` carries a real GPU virtual address obtained from
//! `MTLBuffer::gpuAddress()` (Metal 3+, native to all Apple Silicon).
//! That makes pointer arithmetic (`DevicePtr::offset`) a plain integer
//! add — no buffer/offset pair to thread through. To recover the
//! owning `MTLBuffer` for `free` / blit-copy / `setBuffer:`, we keep
//! a side table `BTreeMap<base_gpu_address, MTLBuffer>` and look up
//! the largest key ≤ ptr. The buffer's gpuAddress range is
//! `[base, base + length)`, so a binary search is enough.
//!
//! # Streams
//!
//! A stream handle indexes a slab of `MetalStream { queue, in_flight }`.
//! Handle 0 is the default stream and is lazily created on first use.
//! `synchronize(stream)` commits the in-flight `MTLCommandBuffer` and
//! `waitUntilCompleted()`s; the next encoder opens on a fresh buffer.
//!
//! # Kernel handles
//!
//! `KernelHandle` indexes a slab of `MTLComputePipelineState`. The
//! library cache (one `MTLLibrary` per `metallib_modules()` entry) is
//! built once at construction and never mutated; pipeline lookups go
//! through the slab + a `(module, fn_name)` HashMap so repeated
//! `kernel()` calls are O(1) cached.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLLibrary, MTLResource, MTLResourceOptions,
    MTLSharedEvent, MTLSize,
};
use parking_lot::Mutex;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

// ── Internal type aliases (Retained<ProtocolObject<dyn _>> is verbose) ────

type ObjDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type ObjBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type ObjQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type ObjCmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type ObjLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;
type ObjPipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type ObjSharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;

/// Metal UMA capacity snapshot.
///
/// Physical RAM and Metal's recommended working set are deliberately separate:
/// the former describes the machine, while the latter is the allocation budget
/// applications should use for admission control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalMemoryInfo {
    pub physical_memory_bytes: Option<usize>,
    pub recommended_max_working_set_bytes: usize,
    pub current_allocated_bytes: usize,
}

impl MetalMemoryInfo {
    pub fn recommended_headroom_bytes(self) -> usize {
        self.recommended_max_working_set_bytes
            .saturating_sub(self.current_allocated_bytes)
    }
}

// ── Stream + slab types ──────────────────────────────────────────────────

struct MetalStream {
    queue: ObjQueue,
    /// In-flight command buffer accumulating encoded work. Committed +
    /// waited on by `synchronize()`; replaced by a fresh buffer on
    /// next encoder open.
    in_flight: Option<ObjCmdBuf>,
}

/// Tracks one outstanding shared event so `record_event` can write the
/// next signal value and `stream_wait_event` can wait on the same
/// counter.
struct EventSlot {
    event: ObjSharedEvent,
    /// Monotonic value sequence — record_event signals `next`, then
    /// increments. stream_wait_event waits on `next - 1` (the most
    /// recently recorded value). Atomic via the surrounding Mutex.
    next: u64,
}

/// Key for the pipeline cache. Stored as owned strings because the
/// `&str` arguments to `kernel()` come from arbitrary call sites.
type PipelineKey = (String, String);

// ── MetalGpuBackend struct + state ───────────────────────────────────────

pub struct MetalGpuBackend {
    device: ObjDevice,
    /// Side table mapping a buffer's base gpuAddress to the owning
    /// `MTLBuffer`. BTreeMap so we can find the buffer containing an
    /// arbitrary `DevicePtr` via `range(..=ptr).next_back()`.
    allocations: Arc<Mutex<BTreeMap<u64, ObjBuffer>>>,
    /// Stream slab. Indexed by `stream_handle - 1`; handle 0 is the
    /// implicit default stream materialized lazily into slot 0.
    streams: Arc<Mutex<Vec<MetalStream>>>,
    /// Loaded metallibs keyed by module name.
    libraries: HashMap<String, ObjLibrary>,
    /// Pipeline-state cache + slab. The HashMap maps `(module, fn)` to
    /// the slab index; the slab owns the `MTLComputePipelineState`.
    /// Both are mutexed so `kernel()` can be called from any thread.
    pipeline_cache: Arc<Mutex<HashMap<PipelineKey, KernelHandle>>>,
    pipeline_slab: Arc<Mutex<Vec<ObjPipeline>>>,
    /// Packed Laguna deliberately ships a sparse active kernel set; CUDA-only
    /// lookups resolve to the invalid sentinel and are never materialized.
    sparse_kernel_set: bool,
    /// Shared-event slab for cross-stream synchronization.
    events: Arc<Mutex<Vec<EventSlot>>>,
}

unsafe impl Send for MetalGpuBackend {}
unsafe impl Sync for MetalGpuBackend {}

impl MetalGpuBackend {
    /// Initialize the Metal backend with the embedded metallib modules.
    ///
    /// `kernel_modules` is the `metallib_modules()` slice produced by
    /// `atlas-kernels`' build script — `(module_name, metallib_bytes)`.
    /// Each entry is loaded into its own `MTLLibrary` via
    /// `newLibraryWithData_error:`. The default stream (handle 0) is
    /// materialized eagerly so the first launch doesn't pay queue-
    /// creation latency.
    pub fn new(ordinal: usize, kernel_modules: &[(&'static str, &'static [u8])]) -> Result<Self> {
        if ordinal != 0 {
            bail!(
                "Metal: only ordinal 0 is supported (Apple Silicon has one \
                 system default device); requested ordinal {ordinal}"
            );
        }
        let device: ObjDevice = MTLCreateSystemDefaultDevice().ok_or_else(|| {
            anyhow!("MTLCreateSystemDefaultDevice returned null — no Metal-capable GPU")
        })?;

        // Build the library cache up-front. `newLibraryWithData_error`
        // takes a `DispatchData`, which is libdispatch's reference-
        // counted byte container. We wrap the &'static slice via
        // dispatch2::DispatchData (zero-copy) — the metallibs are
        // embedded by include_bytes! and outlive the backend.
        let mut libraries: HashMap<String, ObjLibrary> = HashMap::new();
        for (name, bytes) in kernel_modules {
            let data = dispatch2::DispatchData::from_static_bytes(bytes);
            let lib = device.newLibraryWithData_error(&data).map_err(|e| {
                anyhow!(
                    "newLibraryWithData failed for module '{name}': {}",
                    e.localizedDescription()
                )
            })?;
            libraries.insert((*name).to_string(), lib);
        }

        // Materialize the default stream eagerly (slot 0 = handle 0).
        let default_queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null on default device"))?;
        let streams = vec![MetalStream {
            queue: default_queue,
            in_flight: None,
        }];

        tracing::info!(
            device = %device.name().to_string(),
            metallib_modules = libraries.len(),
            physical_memory_bytes = ?sysctl_memsize(),
            recommended_max_working_set_bytes = device.recommendedMaxWorkingSetSize(),
            current_allocated_bytes = device.currentAllocatedSize(),
            "MetalGpuBackend initialized"
        );

        let sparse_kernel_set = libraries.contains_key("laguna_marker");
        Ok(Self {
            device,
            allocations: Arc::new(Mutex::new(BTreeMap::new())),
            streams: Arc::new(Mutex::new(streams)),
            libraries,
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
            pipeline_slab: Arc::new(Mutex::new(Vec::new())),
            sparse_kernel_set,
            events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Return the underlying `MTLDevice` (escape hatch for advanced
    /// use cases — graph capture, custom resource creation, etc.).
    pub fn raw_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Snapshot physical UMA capacity and Metal's recommended allocation budget.
    pub fn memory_info(&self) -> MetalMemoryInfo {
        MetalMemoryInfo {
            physical_memory_bytes: sysctl_memsize(),
            recommended_max_working_set_bytes: self.device.recommendedMaxWorkingSetSize() as usize,
            current_allocated_bytes: self.device.currentAllocatedSize(),
        }
    }

    // ── Internal helpers ────────────────────────────────────────────

    /// Look up the `MTLBuffer` owning `ptr` and the byte offset of
    /// `ptr` within it. Returns `None` if no allocation contains it.
    fn find_buffer(
        allocs: &BTreeMap<u64, ObjBuffer>,
        ptr: DevicePtr,
    ) -> Option<(ObjBuffer, usize)> {
        let (base, buf) = allocs.range(..=ptr.0).next_back()?;
        let offset = (ptr.0 - *base) as usize;
        if offset > buf.length() {
            return None;
        }
        Some((buf.clone(), offset))
    }

    /// Resolve a stream handle to its slab index. Handle 0 → slot 0
    /// (default stream); other handles index `handle - 1`.
    fn stream_index(handle: u64, slab: &[MetalStream]) -> Result<usize> {
        let idx = if handle == 0 {
            0
        } else {
            (handle - 1) as usize
        };
        if idx >= slab.len() {
            bail!("Metal: invalid stream handle {handle}");
        }
        Ok(idx)
    }

    /// Borrow (or open) the in-flight command buffer on the given
    /// stream. Returns a clone of the `Retained` so the caller can
    /// encode without holding the streams mutex across encoder calls.
    fn current_cmd_buf(&self, stream_handle: u64) -> Result<ObjCmdBuf> {
        let mut slab = self.streams.lock();
        let idx = Self::stream_index(stream_handle, &slab)?;
        let s = &mut slab[idx];
        if let Some(ref cb) = s.in_flight {
            return Ok(cb.clone());
        }
        let cb = s
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("commandBuffer returned null on stream {stream_handle}"))?;
        s.in_flight = Some(cb.clone());
        Ok(cb)
    }

    /// Resolve only buffer arguments referenced by one launch. This keeps the
    /// allocation registry lock out of encoder calls without cloning/scanning
    /// unrelated model allocations.
    fn resolve_buffer_args(
        &self,
        args: &[KernelArg<'_>],
    ) -> Result<Vec<(usize, ObjBuffer, usize)>> {
        let allocs = self.allocations.lock();
        args.iter()
            .enumerate()
            .filter_map(|(idx, arg)| match arg {
                KernelArg::Buffer(ptr) => Some(
                    Self::find_buffer(&allocs, *ptr)
                        .map(|(buffer, offset)| (idx, buffer, offset))
                        .ok_or_else(|| anyhow!("launch_typed: arg #{idx} ptr {ptr} not allocated")),
                ),
                KernelArg::Bytes(_) => None,
            })
            .collect()
    }

    /// Commit the in-flight buffer on `stream_handle` (no wait). Used
    /// internally by `synchronize` and `record_event`. Returns the
    /// committed buffer so callers that need to `waitUntilCompleted()`
    /// can.
    fn commit_in_flight(&self, stream_handle: u64) -> Result<Option<ObjCmdBuf>> {
        let mut slab = self.streams.lock();
        let idx = Self::stream_index(stream_handle, &slab)?;
        let s = &mut slab[idx];
        let Some(cb) = s.in_flight.take() else {
            return Ok(None);
        };
        cb.commit();
        Ok(Some(cb))
    }
}

// ── GpuBackend impl ──────────────────────────────────────────────────────

impl GpuBackend for MetalGpuBackend {
    fn supports_bounded_sliding_kv(&self) -> bool {
        true
    }
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        // StorageModeShared is the UMA-friendly mode: `contents()`
        // returns a CPU-mappable pointer that aliases GPU memory.
        let buf: ObjBuffer = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("newBufferWithLength failed for {bytes} bytes"))?;
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("MTLBuffer::gpuAddress returned 0 — Metal 3 / macOS 13 required");
        }
        self.allocations.lock().insert(addr, buf);
        Ok(DevicePtr(addr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        // Apple Silicon UMA: managed and shared are the same thing.
        // No paged virtual memory swap mechanism (cuMemAllocManaged on
        // GB10) — Metal lets the OS handle pressure via its memory
        // pool. Defer to plain alloc.
        self.alloc(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // Removing the entry drops the last `Retained` reference; the
        // ObjC runtime releases the underlying MTLBuffer.
        self.allocations.lock().remove(&ptr.0);
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        // A synchronous host write must not race queued GPU readers/writers of
        // the same shared UMA allocation.
        self.synchronize(0)?;
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_h2d: ptr {dst} not in any allocation"))?;
        if offset + src.len() > buf.length() {
            bail!(
                "copy_h2d: write overflows buffer ({} + {} > {})",
                offset,
                src.len(),
                buf.length()
            );
        }
        let contents: NonNull<c_void> = buf.contents();
        unsafe {
            let dst_ptr = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
        }
        Ok(())
    }

    fn copy_h2d_async(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_h2d_async: dst {dst} not allocated"))?;
        if dst_off + src.len() > dst_buf.length() {
            bail!(
                "copy_h2d_async: write overflows buffer ({} + {} > {})",
                dst_off,
                src.len(),
                dst_buf.length()
            );
        }
        drop(allocs);

        // `newBufferWithBytes` snapshots stack/pinned input immediately. The
        // command buffer retains this staging resource until its ordered blit
        // completes, so callers need not extend the host slice lifetime.
        let source = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(src.as_ptr() as *mut c_void).expect("non-empty source"),
                src.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| anyhow!("newBufferWithBytes failed for {} bytes", src.len()))?;
        let cmd_buf = self.current_cmd_buf(stream)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &source,
                0,
                &dst_buf,
                dst_off,
                src.len(),
            );
        }
        enc.endEncoding();
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        // Match the synchronous CUDA copy contract: commit and wait for prior
        // work on the default stream before reading shared UMA bytes on CPU.
        self.synchronize(0)?;
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2h: ptr {src} not in any allocation"))?;
        if offset + dst.len() > buf.length() {
            bail!(
                "copy_d2h: read overflows buffer ({} + {} > {})",
                offset,
                dst.len(),
                buf.length()
            );
        }
        let contents: NonNull<c_void> = buf.contents();
        unsafe {
            let src_ptr = (contents.as_ptr() as *const u8).add(offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // UMA: synchronize the stream so prior kernels have written
        // their bytes back through the cache, then memcpy.
        self.synchronize(stream)?;
        self.copy_d2h(src, dst)
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (src_buf, src_off) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2d: src ptr {src} not allocated"))?;
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_d2d: dst ptr {dst} not allocated"))?;
        drop(allocs);

        let cmd_buf = self.current_cmd_buf(0)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buf, src_off, &dst_buf, dst_off, bytes,
            );
        }
        enc.endEncoding();
        // Synchronize so the d2d behaves like CUDA's synchronous variant.
        if let Some(cb) = self.commit_in_flight(0)? {
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (src_buf, src_off) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2d_async: src {src} not allocated"))?;
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_d2d_async: dst {dst} not allocated"))?;
        drop(allocs);
        let cmd_buf = self.current_cmd_buf(stream)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buf, src_off, &dst_buf, dst_off, bytes,
            );
        }
        enc.endEncoding();
        Ok(())
    }

    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        // Metal can't safely interpret untyped `*mut c_void` slots as
        // either buffers or scalars (CUDA gets away with this because
        // the driver cross-references the kernel signature). Callers
        // must use `launch_typed`; the cuda-style untyped path is
        // intentionally unsupported here.
        bail!(
            "Metal backend: launch() requires typed args. Use launch_typed() \
             with KernelArg::Buffer / KernelArg::Bytes — see KernelLaunch builder."
        );
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        // Resolve the pipeline state.
        let pipeline = {
            let slab = self.pipeline_slab.lock();
            let slab_idx = func
                .0
                .checked_sub(1)
                .ok_or_else(|| anyhow!("launch_typed: invalid kernel handle 0"))?;
            slab.get(slab_idx as usize)
                .cloned()
                .ok_or_else(|| anyhow!("launch_typed: unknown kernel handle {}", func.0))?
        };

        let referenced_buffers = self.resolve_buffer_args(args)?;

        let cmd_buf = self.current_cmd_buf(stream)?;
        let enc = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("computeCommandEncoder returned null"))?;
        enc.setComputePipelineState(&pipeline);

        // Mark only resources referenced by this dispatch. `setBuffer` retains
        // the same objects for command execution; `useResource` supplies the
        // conservative read/write residency and hazard declaration.
        for (_, buf, _) in &referenced_buffers {
            let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**buf);
            enc.useResource_usage(
                resource,
                objc2_metal::MTLResourceUsage::Read | objc2_metal::MTLResourceUsage::Write,
            );
        }

        let mut referenced_buffers = referenced_buffers.into_iter();
        // Bind each typed arg to its index.
        for (idx, arg) in args.iter().enumerate() {
            match arg {
                KernelArg::Buffer(_) => {
                    let (resolved_idx, buf, offset) =
                        referenced_buffers.next().ok_or_else(|| {
                            anyhow!("launch_typed: missing resolved buffer for arg #{idx}")
                        })?;
                    debug_assert_eq!(resolved_idx, idx);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&buf), offset, idx);
                    }
                }
                KernelArg::Bytes(b) => {
                    let ptr = NonNull::new(b.as_ptr() as *mut c_void)
                        .ok_or_else(|| anyhow!("launch_typed: arg #{idx} bytes is null"))?;
                    unsafe {
                        enc.setBytes_length_atIndex(ptr, b.len(), idx);
                    }
                }
            }
        }

        let threadgroups = MTLSize {
            width: grid[0] as usize,
            height: grid[1] as usize,
            depth: grid[2] as usize,
        };
        let threads_per_tg = MTLSize {
            width: block[0] as usize,
            height: block[1] as usize,
            depth: block[2] as usize,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        enc.endEncoding();
        Ok(())
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        if let Some(cb) = self.commit_in_flight(stream)? {
            cb.waitUntilCompleted();
            if cb.status() == MTLCommandBufferStatus::Error {
                let message = cb
                    .error()
                    .map(|error| error.localizedDescription().to_string())
                    .unwrap_or_else(|| "unknown Metal command-buffer error".to_string());
                bail!("Metal command buffer failed on stream {stream}: {message}");
            }
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        let key: PipelineKey = (module.to_string(), func_name.to_string());
        if let Some(handle) = self.pipeline_cache.lock().get(&key) {
            return Ok(*handle);
        }
        let Some(lib) = self.libraries.get(module) else {
            if self.sparse_kernel_set {
                return Ok(KernelHandle(0));
            }
            return Err(anyhow!("Metal: unknown module '{module}'"));
        };
        let ns_name = NSString::from_str(func_name);
        let Some(function) = lib.newFunctionWithName(&ns_name) else {
            if self.sparse_kernel_set {
                return Ok(KernelHandle(0));
            }
            return Err(anyhow!(
                "Metal: function '{func_name}' not found in module '{module}'"
            ));
        };
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                anyhow!(
                    "newComputePipelineStateWithFunction failed for '{func_name}': {}",
                    e.localizedDescription()
                )
            })?;
        let mut slab = self.pipeline_slab.lock();
        // KernelHandle(0) is the cross-backend sentinel for "disabled".
        // Metal slab indices are therefore encoded as index + 1.
        let handle = KernelHandle(slab.len() as u64 + 1);
        slab.push(pipeline);
        drop(slab);
        self.pipeline_cache.lock().insert(key, handle);
        Ok(handle)
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // On UMA we can write through `contents()` directly when we
        // own the whole range — much cheaper than a blit fillBuffer.
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, ptr)
            .ok_or_else(|| anyhow!("memset: ptr {ptr} not allocated"))?;
        if offset + bytes > buf.length() {
            bail!(
                "memset: range overflows buffer ({} + {} > {})",
                offset,
                bytes,
                buf.length()
            );
        }
        let contents = buf.contents();
        unsafe {
            let dst = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::write_bytes(dst, value, bytes);
        }
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        // UMA + StorageModeShared makes the synchronous memset semantically
        // equivalent (no host/device cache split to flush).
        self.memset(ptr, value, bytes)
    }

    fn total_memory(&self) -> Result<usize> {
        // On Apple Silicon UMA, "device memory" = system RAM. Probe
        // hw.memsize via sysctl for the authoritative number; fall
        // back to MTLDevice.recommendedMaxWorkingSetSize otherwise.
        let info = self.memory_info();
        Ok(info
            .physical_memory_bytes
            .unwrap_or(info.recommended_max_working_set_bytes))
    }

    fn allocation_capacity(&self) -> Result<usize> {
        Ok(self.memory_info().recommended_max_working_set_bytes)
    }

    fn free_memory(&self) -> Result<usize> {
        // No direct API for "free GPU memory" on UMA. Approximate via
        // `recommendedMaxWorkingSetSize - currentAllocatedSize`,
        // which matches the headroom Metal will let us allocate
        // before performance degrades.
        Ok(self.memory_info().recommended_headroom_bytes())
    }

    fn create_stream(&self) -> Result<u64> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null"))?;
        let mut slab = self.streams.lock();
        slab.push(MetalStream {
            queue,
            in_flight: None,
        });
        // Handle = slab index + 1 so handle 0 stays reserved for
        // the default stream.
        Ok(slab.len() as u64)
    }

    fn bind_to_thread(&self) -> Result<()> {
        // Metal devices/queues are thread-safe; no binding required.
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| anyhow!("newSharedEvent returned null"))?;
        let mut slab = self.events.lock();
        slab.push(EventSlot { event, next: 1 });
        // Handle = slab index + 1 (0 reserved for "no event").
        Ok(slab.len() as u64)
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        let value = {
            let mut slab = self.events.lock();
            let idx = (event as usize)
                .checked_sub(1)
                .ok_or_else(|| anyhow!("record_event: invalid event handle {event}"))?;
            let slot = slab
                .get_mut(idx)
                .ok_or_else(|| anyhow!("record_event: event handle {event} out of range"))?;
            let v = slot.next;
            slot.next += 1;
            v
        };
        let cmd_buf = self.current_cmd_buf(stream)?;
        let event_obj = {
            let slab = self.events.lock();
            slab[(event - 1) as usize].event.clone()
        };
        // Encode the signal on the active command buffer. Metal will
        // signal value=`value` once everything queued on this buffer
        // up to this point has completed.
        let proto: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event_obj);
        cmd_buf.encodeSignalEvent_value(proto, value);
        // A signal on an uncommitted command buffer can never become visible.
        // The consumer stream would wait forever and eventually trip macOS's
        // GPU watchdog. Publish the producer buffer after encoding the signal;
        // the consumer's encoded wait preserves cross-stream ordering.
        self.commit_in_flight(stream)?;
        Ok(())
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        let (event_obj, value) = {
            let slab = self.events.lock();
            let idx = (event as usize)
                .checked_sub(1)
                .ok_or_else(|| anyhow!("stream_wait_event: invalid event handle {event}"))?;
            let slot = slab
                .get(idx)
                .ok_or_else(|| anyhow!("stream_wait_event: event handle {event} out of range"))?;
            // Wait on the most-recently-recorded value (next - 1).
            // If nothing has been recorded yet, slot.next is 1 and
            // value is 0 — Metal treats wait-for-0 as a no-op.
            (slot.event.clone(), slot.next.saturating_sub(1))
        };
        let cmd_buf = self.current_cmd_buf(stream)?;
        let proto: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event_obj);
        cmd_buf.encodeWaitForEvent_value(proto, value);
        Ok(())
    }

    fn destroy_event(&self, event: u64) -> Result<()> {
        if event == 0 {
            return Ok(());
        }
        let mut slab = self.events.lock();
        let idx = (event - 1) as usize;
        if let Some(slot) = slab.get_mut(idx) {
            // Replace with a fresh dummy event so the slab indices
            // stay stable across destroys (matches the cuda backend
            // semantics — handles are not reused).
            slot.next = 0;
        }
        Ok(())
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        // UMA: a Shared MTLBuffer's contents() pointer IS host-pinned
        // memory from the GPU's perspective. We park the buffer in
        // the alloc table keyed by gpuAddress, then return the host
        // pointer. `free_host_pinned` looks the buffer up by host
        // pointer to release it.
        let buf = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("alloc_host_pinned: newBufferWithLength failed"))?;
        let host_ptr = buf.contents().as_ptr() as *mut u8;
        // Stash by gpuAddress so plain `free()` on the DevicePtr would
        // also work; the host-pinned variant is purely a CPU view.
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("alloc_host_pinned: gpuAddress returned 0");
        }
        self.allocations.lock().insert(addr, buf);
        Ok(host_ptr)
    }

    fn free_host_pinned(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // Find the buffer whose contents() pointer matches.
        let mut allocs = self.allocations.lock();
        let target_addr = allocs.iter().find_map(|(addr, buf)| {
            let host = buf.contents().as_ptr() as *mut u8;
            if host == ptr { Some(*addr) } else { None }
        });
        if let Some(addr) = target_addr {
            allocs.remove(&addr);
        }
        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Probe `hw.memsize` via libc::sysctl on macOS. Returns the total
/// system RAM in bytes — on Apple Silicon UMA this is also the upper
/// bound on Metal-addressable memory.
fn sysctl_memsize() -> Option<usize> {
    use std::ffi::CString;
    let name = CString::new("hw.memsize").ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut u64 as *mut c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 { Some(value as usize) } else { None }
}

// ── Smoke test ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
