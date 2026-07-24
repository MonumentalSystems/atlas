// SPDX-License-Identifier: AGPL-3.0-only

//! Page-aligned O_DIRECT I/O primitives for [`super::FastSafetensorsLoader`].

use anyhow::{Result, bail};
use std::alloc::{Layout, alloc, dealloc};
use std::fs::File;
use std::path::Path;

/// O_DIRECT requires offsets, sizes, and buffers to be aligned to the
/// filesystem block size. 4 KiB is the upper bound on every Linux arch we
/// target and satisfies every fs we've seen.
pub(crate) const O_DIRECT_ALIGN: usize = 4096;

/// Heap buffer aligned to [`O_DIRECT_ALIGN`]. `Send` so it can cross a channel
/// from the reader thread to the copier thread.
pub(crate) struct AlignedBuffer {
    ptr: *mut u8,
    cap: usize,
    layout: Layout,
}

// SAFETY: `AlignedBuffer` owns the allocation pointed to by `ptr`; the
// pointer is created by `std::alloc::alloc` with the recorded `layout` and
// is freed in `Drop` with the matching layout. The struct holds no shared
// references and exposes no `&self` API that aliases the buffer, so moving
// it between threads only moves the unique owner of the allocation. We do
// not implement `Sync`: concurrent `&AlignedBuffer` readers are not a
// pattern Atlas uses (each shard is owned by a single reader thread).
unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    fn new(cap_bytes: usize) -> Self {
        let cap = cap_bytes.max(1).div_ceil(O_DIRECT_ALIGN) * O_DIRECT_ALIGN;
        let layout = Layout::from_size_align(cap, O_DIRECT_ALIGN).expect("valid layout");
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, cap, layout }
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.cap) }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn open_direct(path: &Path) -> std::io::Result<File> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::FromRawFd;
    let cstr = CString::new(path.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let flags = libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC;
    let fd = unsafe { libc::open(cstr.as_ptr(), flags) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(target_os = "macos")]
pub(crate) fn open_direct(path: &Path) -> std::io::Result<File> {
    use std::os::unix::io::AsRawFd;

    let file = File::open(path)?;
    // Apple has no O_DIRECT. F_NOCACHE provides the equivalent property Atlas
    // needs here: bounded pread data bypasses the unified file cache instead of
    // displacing Metal allocations and other applications during a large load.
    let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    if rc == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(file)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn open_direct(_path: &Path) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "O_DIRECT only supported on Linux",
    ))
}

/// Read a tensor's bytes into an aligned buffer. Returns the buffer and the
/// offset within it where the tensor's `len` bytes start.
///
/// With `using_direct=true`, the read window is widened to the nearest 4 KiB
/// boundaries so the offset/size/buffer alignment constraints are met. If the
/// aligned window extends past end-of-file, we let the kernel return a short
/// read for the trailing fragment (Linux ≥ 2.6 accepts this for O_DIRECT on
/// mainstream filesystems — we only require that the tensor's exact bytes
/// have been populated, which the post-loop check enforces).
pub(crate) fn read_tensor_aligned(
    fd: std::os::unix::io::RawFd,
    abs_offset: u64,
    len: usize,
    using_direct: bool,
) -> Result<(AlignedBuffer, usize)> {
    let (window_start, window_len, slice_off) = if using_direct {
        let ws = abs_offset - (abs_offset % O_DIRECT_ALIGN as u64);
        let unaligned_end = abs_offset + len as u64;
        let aligned_end = unaligned_end.div_ceil(O_DIRECT_ALIGN as u64) * O_DIRECT_ALIGN as u64;
        let wl = (aligned_end - ws) as usize;
        (ws, wl, (abs_offset - ws) as usize)
    } else {
        (abs_offset, len, 0usize)
    };

    let mut buf = AlignedBuffer::new(window_len);
    let mut filled = 0usize;
    while filled < window_len {
        let ret = unsafe {
            libc::pread(
                fd,
                buf.as_mut_ptr().add(filled) as *mut libc::c_void,
                window_len - filled,
                (window_start + filled as u64) as libc::off_t,
            )
        };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            // On O_DIRECT, a non-aligned tail at EOF may return EINVAL on
            // some filesystems. Accept the short read as long as we've
            // already covered the tensor's exact range.
            if filled >= slice_off + len {
                break;
            }
            bail!(
                "pread failed at offset {}: {err}",
                window_start + filled as u64
            );
        }
        if ret == 0 {
            break; // EOF
        }
        filled += ret as usize;
    }
    if filled < slice_off + len {
        bail!(
            "short read: got {} bytes, need at least {} (tensor spans offset {}..{})",
            filled,
            slice_off + len,
            abs_offset,
            abs_offset + len as u64
        );
    }
    Ok((buf, slice_off))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

    #[test]
    fn macos_uncached_file_supports_bounded_pread() {
        let path = std::env::temp_dir().join(format!(
            "atlas-uncached-read-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let bytes: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
        let mut writer = std::fs::File::create(&path).expect("create fixture");
        writer.write_all(&bytes).expect("write fixture");
        writer.sync_all().expect("sync fixture");
        drop(writer);

        let file = open_direct(&path).expect("open uncached data file");
        let (buf, offset) =
            read_tensor_aligned(file.as_raw_fd(), 123, 4097, true).expect("bounded uncached pread");
        assert_eq!(&buf.as_slice()[offset..offset + 4097], &bytes[123..4220]);

        drop(file);
        std::fs::remove_file(path).expect("remove fixture");
    }
}
