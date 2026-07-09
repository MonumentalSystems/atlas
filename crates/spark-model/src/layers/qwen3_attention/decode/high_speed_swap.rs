// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `super::super::decode.rs` for file-size budget.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};
use spark_runtime::kv_dequant::{
    NVFP4_E2M1_LUT, TURBO4_LUT, dequant_4bit_block_to_bf16, dequant_fp8_to_bf16,
    dequant_turbo3_block_to_bf16, dequant_turbo8_block_to_bf16,
};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Opt-in (`ATLAS_HSS_PINNED_OFFLOAD`): stage the per-block K/V D2H copies of the
/// high-speed-swap offload through a REUSED PINNED host buffer and FUSE the two
/// per-block copies (K then V) into a single `synchronize` instead of one implicit
/// sync per copy. Mirrors the pinned+fused decode-ring spill gather lever
/// (`ATLAS_SSM_DECODE_FUSED_GATHER`): D2H into pageable host memory stages through
/// CUDA's small internal bounce buffer (implicitly synchronous, ~0.17 GB/s), so the
/// pageable per-copy path throttles the offload; a pinned dst makes
/// `cuMemcpyDtoHAsync` truly async + full-bandwidth and lets the two copies share
/// one sync.
///
/// DEFAULT OFF: unset (or any non-truthy value) keeps the exact current pageable
/// per-copy-synced path, byte-for-byte. This gate exists so the pinned path stays
/// inert until GPU-validated — the offloaded KV is rehydrated and read back for
/// attention, so the disk image MUST stay bit-identical across all 5 dtype arms.
/// Set `ATLAS_HSS_PINNED_OFFLOAD=1` to engage.
static HSS_PINNED_OFFLOAD: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    parse_hss_pinned_flag(std::env::var("ATLAS_HSS_PINNED_OFFLOAD").ok().as_deref())
});

/// Pure parse of `ATLAS_HSS_PINNED_OFFLOAD`: engaged only for the truthy tokens
/// `1`/`true`/`on`/`yes`. Everything else (unset, `0`, `false`, `off`, `no`,
/// garbage) is OFF = the current pageable per-copy-synced path. Extracted so the
/// flag semantics are unit-testable without touching process env.
pub(crate) fn parse_hss_pinned_flag(v: Option<&str>) -> bool {
    matches!(v, Some("1") | Some("true") | Some("on") | Some("yes"))
}

/// Which host buffer the D2H copy writes into, per dtype arm. Pure — mirrors the
/// arm dispatch in `high_speed_swap_offload_new_blocks` so a unit test can pin the
/// which-buffer-is-the-D2H-dst mapping (the misattribution guard: pinning the wrong
/// buffer in a quant arm would silently no-op the optimization).
// Wired into a production `debug_assert!` inside the offload loop (see the match
// on `layer_dtype`) so the unit-tested mapping guards the real arm dispatch and
// cannot silently drift from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HssD2hDstKind {
    /// BF16 arm: the D2H dst IS the final BF16 payload (`k_host`/`v_host`).
    Bf16Payload,
    /// Quantized arms: the D2H dst is RAW quantized bytes (`k_raw`/`v_raw`); a
    /// host-side dequant then produces the BF16 payload (never a D2H dst).
    RawQuant,
}

/// Classify the D2H destination buffer for `dtype`. Bf16 (incl. its K-turbo
/// composites) stage the payload directly; every quantized arm stages raw bytes.
pub(crate) fn hss_d2h_dst_kind(dtype: KvCacheDtype) -> HssD2hDstKind {
    match dtype {
        KvCacheDtype::Bf16
        | KvCacheDtype::Bf16KTurbo4V
        | KvCacheDtype::Bf16KTurbo3V
        | KvCacheDtype::Bf16KTurbo2V => HssD2hDstKind::Bf16Payload,
        _ => HssD2hDstKind::RawQuant,
    }
}

/// Exact byte count the D2H copy writes for `dtype`, and thus the size the reused
/// pinned staging buffer must be for this layer. bf16 = `block_floats * 2` (the
/// payload); fp8 = `block_floats` (1 raw byte / element); nvfp4/turbo{3,4,8} =
/// `layer_block_bytes` (the packed device stride). Pure — unit-tested.
pub(crate) fn hss_d2h_dst_bytes(
    dtype: KvCacheDtype,
    block_floats: usize,
    layer_block_bytes: usize,
) -> usize {
    match dtype {
        KvCacheDtype::Bf16
        | KvCacheDtype::Bf16KTurbo4V
        | KvCacheDtype::Bf16KTurbo3V
        | KvCacheDtype::Bf16KTurbo2V => block_floats * 2,
        KvCacheDtype::Fp8
        | KvCacheDtype::Fp8KTurbo4V
        | KvCacheDtype::Fp8KTurbo3V
        | KvCacheDtype::Fp8KTurbo2V => block_floats,
        _ => layer_block_bytes,
    }
}

/// RAII guard for the per-invocation pinned K/V staging pair. Frees BOTH pinned
/// regions on drop — including on the mid-loop `anyhow::bail!` (issue #31 eviction
/// path), so the pinned allocation never leaks.
struct HssPinnedPair<'g> {
    gpu: &'g dyn spark_runtime::gpu::GpuBackend,
    k: *mut u8,
    v: *mut u8,
    bytes: usize,
}

impl Drop for HssPinnedPair<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free_host_pinned(self.k, self.bytes);
        let _ = self.gpu.free_host_pinned(self.v, self.bytes);
    }
}

impl Qwen3AttentionLayer {
    pub(in super::super) fn high_speed_swap_engaged(
        &self,
        kv_cache: &spark_runtime::kv_cache::PagedKvCache,
    ) -> bool {
        if kv_cache.config().cache_blocks_per_seq.is_none() {
            return false;
        }
        if !spark_storage::local_installed() {
            return false;
        }
        // Phase 6.2.c — proper: every quantization variant now has a host-side
        // dequant path that produces BF16 for the orchestrator's tiled-attention
        // kernel. For Turbo3/4/8 the cache stores WHT(K)/WHT(V) which round-trip
        // correctly because production already applies WHT(Q) and iWHT(out)
        // around the orchestrator call (decode.rs WHT bookends).
        matches!(
            kv_cache.dtype_for_layer(self.attn_layer_idx),
            KvCacheDtype::Bf16
                | KvCacheDtype::Fp8
                | KvCacheDtype::Nvfp4
                | KvCacheDtype::Turbo3
                | KvCacheDtype::Turbo4
                | KvCacheDtype::Turbo8,
        )
    }

    /// Catch up: alloc a disk_block_id and offload K/V to disk for every
    /// block_table entry that doesn't yet have a disk_id. Idempotent —
    /// re-running on a sequence that's fully caught up is a no-op.
    /// Called after every K/V write from both decode and prefill paths.
    pub(in super::super) fn high_speed_swap_offload_new_blocks(
        &self,
        kv_cache: &mut PagedKvCache,
        block_table: &Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        nkv: u32,
        hd: u32,
        bs: usize,
    ) -> Result<()> {
        if !self.high_speed_swap_engaged(kv_cache) {
            return Ok(());
        }
        let layer_u32 = self.attn_layer_idx as u32;
        let block_floats = bs * (nkv as usize) * (hd as usize);

        // Phase 6.3: disk_block_ids growth lives in the alloc helper
        // (`model.rs::ensure_blocks_through_decode` / `_prefill`). Here we
        // assert the invariant `disk_block_ids.len() == hss_window_start +
        // block_table.len()` (window_start derivable from the lengths) and
        // proceed straight to the per-layer K/V offload.
        debug_assert!(
            disk_block_ids.len() >= block_table.len(),
            "Phase 6.3 invariant: alloc helper must keep disk_block_ids ≥ block_table.len() \
             (got disk={} bt={})",
            disk_block_ids.len(),
            block_table.len()
        );

        // Step 2: per-layer catch-up. THIS layer's offloaded count
        // (`disk_last_offloaded_per_layer[L]`) lags `disk_block_ids.len()`
        // by however many new blocks have been allocated since this layer
        // last ran. For each missing block, the layer's K/V is currently
        // in the production HBM cache at `block_table[bt_idx]` for
        // `bt_idx = logical_pos - window_start`. Read it back, push to disk.
        if disk_last_offloaded_per_layer.len() <= self.attn_layer_idx {
            // Defensive: caller didn't size the vec. Resize on first hit.
            disk_last_offloaded_per_layer.resize(self.attn_layer_idx + 1, 0);
        }
        let last = disk_last_offloaded_per_layer[self.attn_layer_idx] as usize;
        let total = disk_block_ids.len();
        if total == 0 {
            return Ok(());
        }
        // Window of HBM-resident blocks: block_table[0..block_table.len()]
        // covers logical positions [total - block_table.len(), total).
        let window_start = total.saturating_sub(block_table.len());
        // Always re-offload the BOUNDARY block (one before `last`) on every
        // call, in addition to all blocks in `last..total`. Two cases:
        //
        // (1) Decode case: `last == total` (no new block this step). Slots in
        //     the active block keep getting written one-per-step without
        //     `disk_block_ids.len()` growing. `start = total - 1` ensures the
        //     active block is re-pushed every step. Without this the streaming
        //     kernel reads stale (zero-init) bytes for the unwritten slots
        //     → degenerate attention → "the the the" loop.
        //
        // (2) Chunked-prefill boundary case (issue #31, follow-up to PR #37):
        //     `last < total` after a new chunk advanced `disk_block_ids`. The
        //     PREVIOUS chunk's last block (`last - 1`) typically has unwritten
        //     tail slots — `reshape_and_cache_flash` writes only the chunk's
        //     own token slots, so when chunk N ended mid-block it left the
        //     tail slots zero on disk after the post-chunk-N offload. Chunk
        //     N+1 fills those tail slots in HBM but the offload's `start =
        //     last` skipped re-pushing the boundary block, so disk's
        //     boundary-block tail stays permanently zeroed. Decode reads the
        //     full history from disk via `attend_layer_on_stream`, so the
        //     zeroed slots silently corrupt attention for the chunk-boundary
        //     positions (manifests as needle-in-haystack precision loss in
        //     long-context recall — see issue #31 differential tests).
        //
        // `last.saturating_sub(1).min(total - 1)` covers both cases at the
        // cost of ~one extra D2H per layer per chunk (negligible).
        let start = last.saturating_sub(1).min(total - 1);

        // Layer dtype and packed block stride are constant across this call (one
        // layer). Hoist them so the pinned staging buffer can be sized ONCE.
        let layer_dtype = kv_cache.dtype_for_layer(self.attn_layer_idx);
        let layer_block_bytes = kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx);
        let d2h_dst_bytes = hss_d2h_dst_bytes(layer_dtype, block_floats, layer_block_bytes);

        // Opt-in pinned staging (`ATLAS_HSS_PINNED_OFFLOAD`). Allocate the K and V
        // pinned regions ONCE and reuse them across every block in the loop. K and
        // V are DISJOINT regions — both are live simultaneously (dequant reads
        // k_dst AND v_dst; offload reads k_host AND v_host together), so a single
        // shared region would let V's copy clobber K before K is consumed. On any
        // alloc failure we warn once and fall back to the pageable per-copy path
        // (still correct, just not pinned). NOTE: the block loop below is fully
        // SERIAL (and multi-seq wraps it in a serial per-seq pass), so one reused
        // pair per invocation is safe; if that ever parallelizes this must become
        // per-worker.
        let pinned: Option<HssPinnedPair> = if *HSS_PINNED_OFFLOAD {
            match ctx.gpu.alloc_host_pinned(d2h_dst_bytes) {
                Ok(k) => match ctx.gpu.alloc_host_pinned(d2h_dst_bytes) {
                    Ok(v) => {
                        static ACTIVE_LOG: std::sync::Once = std::sync::Once::new();
                        ACTIVE_LOG.call_once(|| {
                            tracing::info!(
                                "HSS pinned offload ACTIVE (ATLAS_HSS_PINNED_OFFLOAD): reused \
                                 pinned K/V pair, {d2h_dst_bytes} B each, fused D2H"
                            );
                        });
                        Some(HssPinnedPair {
                            gpu: ctx.gpu,
                            k,
                            v,
                            bytes: d2h_dst_bytes,
                        })
                    }
                    Err(e) => {
                        let _ = ctx.gpu.free_host_pinned(k, d2h_dst_bytes);
                        tracing::warn!(
                            "ATLAS_HSS_PINNED_OFFLOAD: V pinned alloc({d2h_dst_bytes}) failed \
                             ({e:#}); falling back to pageable per-copy offload"
                        );
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!(
                        "ATLAS_HSS_PINNED_OFFLOAD: K pinned alloc({d2h_dst_bytes}) failed \
                         ({e:#}); falling back to pageable per-copy offload"
                    );
                    None
                }
            }
        } else {
            None
        };
        // `fused` == pinned engaged for THIS invocation: async-enqueue both copies
        // then ONE trailing synchronize, versus the legacy per-copy synchronous
        // path. Kept in lockstep with `pinned` so the OFF path is byte-identical.
        let fused = pinned.is_some();
        // One D2H copy, async when fused (caller MUST synchronize before reading
        // dst) else the legacy self-syncing `copy_d2h_on_stream`.
        let d2h = |src: u64, dst: &mut [u8]| -> Result<()> {
            if fused {
                ctx.gpu
                    .copy_d2h_on_stream_async(DevicePtr(src), dst, stream)
            } else {
                ctx.gpu.copy_d2h_on_stream(DevicePtr(src), dst, stream)
            }
        };

        for logical_pos in start..total {
            if logical_pos < window_start {
                // Issue #31: the slide-before-alloc loop in
                // `block_mgmt::ensure_blocks_through_prefill` advanced the
                // sliding window past `logical_pos` before this layer got
                // a chance to offload. The invariant declared at line 122-127
                // of `block_mgmt.rs` (every attention layer must catch up
                // its offloads before any slide) is debug-asserted only —
                // release builds silently let the slide proceed, then this
                // check trips at the next offload pass.
                //
                // Practical fix until Phase 6.2.b lands chunked-prefill
                // reads through the HSS orchestrator: ensure
                // `--high-speed-swap-cache-blocks-per-seq × --block-size`
                // is large enough that the per-chunk prefill never grows
                // `disk_block_ids` past `block_table.len()` faster than the
                // per-layer offload can keep up. Drop --high-speed-swap if
                // KV fits HBM at this batch size.
                anyhow::bail!(
                    "high-speed-swap: layer {} block {} was evicted before this layer offloaded \
                     it (issue #31). \n\
                     Diagnostic state: attn_layer_idx={}, logical_pos={}, \
                     window_start={}, total=disk_block_ids.len()={}, \
                     block_table.len()={}, this_layer.last_offloaded={}, \
                     all_layer_cursors={:?}.\n\
                     This means the sliding-window eviction loop advanced past disk slot \
                     {} before attention layer {} could push its K/V there. The slide-before-alloc \
                     invariant in block_mgmt.rs (every attention layer must offload before any \
                     slide) is debug-asserted only — release builds skip it.\n\
                     Workaround: raise --high-speed-swap-cache-blocks-per-seq so \
                     `cap × block_size` ≥ your largest prompt, OR drop --high-speed-swap \
                     entirely if KV fits HBM at this batch/quant.",
                    self.attn_layer_idx,
                    logical_pos,
                    self.attn_layer_idx,
                    logical_pos,
                    window_start,
                    total,
                    block_table.len(),
                    last,
                    disk_last_offloaded_per_layer,
                    logical_pos,
                    self.attn_layer_idx,
                );
            }
            let bt_idx = logical_pos - window_start;
            let phys_blk = block_table[bt_idx];
            let disk_id = disk_block_ids[logical_pos];
            let k_block_dev = kv_cache.k_cache_ptr(self.attn_layer_idx, phys_blk).0;
            let v_block_dev = kv_cache.v_cache_ptr(self.attn_layer_idx, phys_blk).0;
            let mut k_host = vec![half::bf16::from_f32(0.0); block_floats];
            let mut v_host = vec![half::bf16::from_f32(0.0); block_floats];
            // Phase 6.2.c proper — dispatch on layer dtype. BF16 streams the
            // bytes directly; quantized variants read raw bytes then dequant
            // on the host before disk-write (the streaming kernel reads BF16).
            //
            // `d2h` is async when the pinned path is engaged (ATLAS_HSS_PINNED_OFFLOAD),
            // so EVERY arm must `ctx.gpu.synchronize(stream)?` after the two enqueues
            // and BEFORE any CPU read of the dst (the bf16 payload-copy or the quant
            // dequant). When not fused, `d2h` is the legacy self-syncing copy and the
            // trailing `synchronize` is skipped — byte-identical to the old path.
            let bs_us = bs;
            let nkv_us = nkv as usize;
            let hd_us = hd as usize;
            match layer_dtype {
                KvCacheDtype::Bf16
                | KvCacheDtype::Bf16KTurbo4V
                | KvCacheDtype::Bf16KTurbo3V
                | KvCacheDtype::Bf16KTurbo2V => {
                    // Guard: the tested mapping must agree that this arm's D2H dst
                    // IS the payload (k_host/v_host), not a raw quant buffer.
                    debug_assert_eq!(
                        hss_d2h_dst_kind(layer_dtype),
                        HssD2hDstKind::Bf16Payload,
                        "bf16 arm took a dtype the classifier calls RawQuant"
                    );
                    // copy_d2h_on_stream[_async]: orders the D2H after
                    // WHT+reshape_and_cache on the production stream. copy_d2h would
                    // race (default-stream sync only) and read torn bytes — Turbo8
                    // race fix, 2026-04-28. For bf16 the D2H dst IS the payload.
                    match &pinned {
                        Some(p) => {
                            // Stage into the reused pinned pair, then materialize
                            // the payload into k_host/v_host for the shared offload
                            // below. SAFETY: p.k/p.v each own `d2h_dst_bytes`
                            // (== block_floats*2 here) of page-locked memory, touched
                            // only by this thread, serially per block.
                            let k_dst = unsafe {
                                std::slice::from_raw_parts_mut(p.k, block_floats * 2)
                            };
                            let v_dst = unsafe {
                                std::slice::from_raw_parts_mut(p.v, block_floats * 2)
                            };
                            d2h(k_block_dev, k_dst)?;
                            d2h(v_block_dev, v_dst)?;
                            ctx.gpu.synchronize(stream)?;
                            let k_src = unsafe {
                                std::slice::from_raw_parts(
                                    p.k as *const half::bf16,
                                    block_floats,
                                )
                            };
                            let v_src = unsafe {
                                std::slice::from_raw_parts(
                                    p.v as *const half::bf16,
                                    block_floats,
                                )
                            };
                            k_host.copy_from_slice(k_src);
                            v_host.copy_from_slice(v_src);
                        }
                        None => {
                            // Legacy pageable path: D2H directly into k_host/v_host,
                            // self-syncing per copy. Byte-identical to pre-flag code.
                            d2h(k_block_dev, unsafe {
                                std::slice::from_raw_parts_mut(
                                    k_host.as_mut_ptr() as *mut u8,
                                    block_floats * 2,
                                )
                            })?;
                            d2h(v_block_dev, unsafe {
                                std::slice::from_raw_parts_mut(
                                    v_host.as_mut_ptr() as *mut u8,
                                    block_floats * 2,
                                )
                            })?;
                        }
                    }
                }
                KvCacheDtype::Fp8
                | KvCacheDtype::Fp8KTurbo4V
                | KvCacheDtype::Fp8KTurbo3V
                | KvCacheDtype::Fp8KTurbo2V => {
                    debug_assert_eq!(
                        hss_d2h_dst_kind(layer_dtype),
                        HssD2hDstKind::RawQuant,
                        "fp8 arm took a dtype the classifier calls Bf16Payload"
                    );
                    let mut k_raw_vec: Vec<u8>;
                    let mut v_raw_vec: Vec<u8>;
                    let (k_raw, v_raw): (&mut [u8], &mut [u8]) = match &pinned {
                        // SAFETY: pinned pair sized to d2h_dst_bytes == block_floats.
                        Some(p) => unsafe {
                            (
                                std::slice::from_raw_parts_mut(p.k, block_floats),
                                std::slice::from_raw_parts_mut(p.v, block_floats),
                            )
                        },
                        None => {
                            k_raw_vec = vec![0u8; block_floats];
                            v_raw_vec = vec![0u8; block_floats];
                            (&mut k_raw_vec[..], &mut v_raw_vec[..])
                        }
                    };
                    d2h(k_block_dev, k_raw)?;
                    d2h(v_block_dev, v_raw)?;
                    if fused {
                        ctx.gpu.synchronize(stream)?;
                    }
                    let (k_scale, v_scale) = self.effective_fp8_scales();
                    dequant_fp8_to_bf16(k_raw, k_scale, &mut k_host);
                    dequant_fp8_to_bf16(v_raw, v_scale, &mut v_host);
                }
                KvCacheDtype::Nvfp4
                | KvCacheDtype::Turbo4
                | KvCacheDtype::Turbo4KTurbo3V
                | KvCacheDtype::Turbo4KTurbo8V => {
                    debug_assert_eq!(
                        hss_d2h_dst_kind(layer_dtype),
                        HssD2hDstKind::RawQuant,
                        "nvfp4/turbo4 arm took a dtype the classifier calls Bf16Payload"
                    );
                    let mut k_raw_vec: Vec<u8>;
                    let mut v_raw_vec: Vec<u8>;
                    let (k_raw, v_raw): (&mut [u8], &mut [u8]) = match &pinned {
                        // SAFETY: pinned pair sized to d2h_dst_bytes == layer_block_bytes.
                        Some(p) => unsafe {
                            (
                                std::slice::from_raw_parts_mut(p.k, layer_block_bytes),
                                std::slice::from_raw_parts_mut(p.v, layer_block_bytes),
                            )
                        },
                        None => {
                            k_raw_vec = vec![0u8; layer_block_bytes];
                            v_raw_vec = vec![0u8; layer_block_bytes];
                            (&mut k_raw_vec[..], &mut v_raw_vec[..])
                        }
                    };
                    d2h(k_block_dev, k_raw)?;
                    d2h(v_block_dev, v_raw)?;
                    if fused {
                        ctx.gpu.synchronize(stream)?;
                    }
                    let lut = if layer_dtype == KvCacheDtype::Nvfp4 {
                        &NVFP4_E2M1_LUT
                    } else {
                        &TURBO4_LUT
                    };
                    dequant_4bit_block_to_bf16(k_raw, bs_us, nkv_us, hd_us, lut, &mut k_host);
                    dequant_4bit_block_to_bf16(v_raw, bs_us, nkv_us, hd_us, lut, &mut v_host);
                }
                KvCacheDtype::Turbo3 | KvCacheDtype::Turbo3KTurbo8V | KvCacheDtype::Turbo2 => {
                    debug_assert_eq!(
                        hss_d2h_dst_kind(layer_dtype),
                        HssD2hDstKind::RawQuant,
                        "turbo3 arm took a dtype the classifier calls Bf16Payload"
                    );
                    let mut k_raw_vec: Vec<u8>;
                    let mut v_raw_vec: Vec<u8>;
                    let (k_raw, v_raw): (&mut [u8], &mut [u8]) = match &pinned {
                        // SAFETY: pinned pair sized to d2h_dst_bytes == layer_block_bytes.
                        Some(p) => unsafe {
                            (
                                std::slice::from_raw_parts_mut(p.k, layer_block_bytes),
                                std::slice::from_raw_parts_mut(p.v, layer_block_bytes),
                            )
                        },
                        None => {
                            k_raw_vec = vec![0u8; layer_block_bytes];
                            v_raw_vec = vec![0u8; layer_block_bytes];
                            (&mut k_raw_vec[..], &mut v_raw_vec[..])
                        }
                    };
                    d2h(k_block_dev, k_raw)?;
                    d2h(v_block_dev, v_raw)?;
                    if fused {
                        ctx.gpu.synchronize(stream)?;
                    }
                    dequant_turbo3_block_to_bf16(k_raw, bs_us, nkv_us, hd_us, &mut k_host);
                    dequant_turbo3_block_to_bf16(v_raw, bs_us, nkv_us, hd_us, &mut v_host);
                }
                KvCacheDtype::Turbo8 => {
                    debug_assert_eq!(
                        hss_d2h_dst_kind(layer_dtype),
                        HssD2hDstKind::RawQuant,
                        "turbo8 arm took a dtype the classifier calls Bf16Payload"
                    );
                    let mut k_raw_vec: Vec<u8>;
                    let mut v_raw_vec: Vec<u8>;
                    let (k_raw, v_raw): (&mut [u8], &mut [u8]) = match &pinned {
                        // SAFETY: pinned pair sized to d2h_dst_bytes == layer_block_bytes.
                        Some(p) => unsafe {
                            (
                                std::slice::from_raw_parts_mut(p.k, layer_block_bytes),
                                std::slice::from_raw_parts_mut(p.v, layer_block_bytes),
                            )
                        },
                        None => {
                            k_raw_vec = vec![0u8; layer_block_bytes];
                            v_raw_vec = vec![0u8; layer_block_bytes];
                            (&mut k_raw_vec[..], &mut v_raw_vec[..])
                        }
                    };
                    d2h(k_block_dev, k_raw)?;
                    d2h(v_block_dev, v_raw)?;
                    if fused {
                        ctx.gpu.synchronize(stream)?;
                    }
                    dequant_turbo8_block_to_bf16(k_raw, bs_us, nkv_us, hd_us, &mut k_host);
                    dequant_turbo8_block_to_bf16(v_raw, bs_us, nkv_us, hd_us, &mut v_host);
                }
            }
            spark_storage::with_local(|hss| {
                match layer_dtype {
                    KvCacheDtype::Bf16
                    | KvCacheDtype::Bf16KTurbo4V
                    | KvCacheDtype::Bf16KTurbo3V
                    | KvCacheDtype::Bf16KTurbo2V => hss.offload_block_on_stream(
                        stream,
                        layer_u32,
                        disk_id,
                        k_block_dev,
                        &k_host,
                        &v_host,
                    ),
                    // Quantized: skip predictor projection — the BF16 kernel
                    // would OOB-read on a non-BF16 layout. Eviction degrades
                    // to LRU for these blocks; correctness preserved.
                    _ => hss.offload_block_no_predict_on_stream(
                        stream, layer_u32, disk_id, &k_host, &v_host,
                    ),
                }
            })
            .expect("local_installed checked in high_speed_swap_engaged")?;
        }
        disk_last_offloaded_per_layer[self.attn_layer_idx] = total as u32;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{HssD2hDstKind, hss_d2h_dst_bytes, hss_d2h_dst_kind, parse_hss_pinned_flag};
    use spark_runtime::kv_cache::KvCacheDtype;

    // bs*nkv*hd = 16*8*128 per the offload loop.
    const BLOCK_FLOATS: usize = 16 * 8 * 128; // 16384
    // Arbitrary packed device stride distinct from BLOCK_FLOATS and BLOCK_FLOATS*2
    // to prove the quant arms select layer_block_bytes (not a fixed size).
    const LAYER_BLOCK_BYTES: usize = 12288;

    /// bf16 (and its K-turbo composites) stage the payload directly; every other
    /// engaged dtype stages RAW quant bytes. Guards against the misattribution
    /// bug of pinning k_host (the dequant output) instead of k_raw.
    #[test]
    fn dst_kind_bf16_is_payload_quant_is_raw() {
        for dt in [
            KvCacheDtype::Bf16,
            KvCacheDtype::Bf16KTurbo4V,
            KvCacheDtype::Bf16KTurbo3V,
            KvCacheDtype::Bf16KTurbo2V,
        ] {
            assert_eq!(hss_d2h_dst_kind(dt), HssD2hDstKind::Bf16Payload, "{dt:?}");
        }
        for dt in [
            KvCacheDtype::Fp8,
            KvCacheDtype::Fp8KTurbo4V,
            KvCacheDtype::Fp8KTurbo3V,
            KvCacheDtype::Fp8KTurbo2V,
            KvCacheDtype::Nvfp4,
            KvCacheDtype::Turbo4,
            KvCacheDtype::Turbo4KTurbo3V,
            KvCacheDtype::Turbo4KTurbo8V,
            KvCacheDtype::Turbo3,
            KvCacheDtype::Turbo3KTurbo8V,
            KvCacheDtype::Turbo2,
            KvCacheDtype::Turbo8,
        ] {
            assert_eq!(hss_d2h_dst_kind(dt), HssD2hDstKind::RawQuant, "{dt:?}");
        }
    }

    /// bf16 dst == block_floats*2 (the payload bytes); fp8 dst == block_floats
    /// (1 raw byte/elem); nvfp4/turbo{3,4,8} dst == layer_block_bytes (packed).
    /// Sizing the pinned staging buffer wrong (e.g. layer_block_bytes for a bf16
    /// layer) would OOB the D2H copy.
    #[test]
    fn dst_bytes_per_arm_size() {
        // bf16 group -> block_floats * 2
        for dt in [
            KvCacheDtype::Bf16,
            KvCacheDtype::Bf16KTurbo4V,
            KvCacheDtype::Bf16KTurbo3V,
            KvCacheDtype::Bf16KTurbo2V,
        ] {
            assert_eq!(
                hss_d2h_dst_bytes(dt, BLOCK_FLOATS, LAYER_BLOCK_BYTES),
                BLOCK_FLOATS * 2,
                "{dt:?}"
            );
        }
        // fp8 group -> block_floats (1 raw byte per element)
        for dt in [
            KvCacheDtype::Fp8,
            KvCacheDtype::Fp8KTurbo4V,
            KvCacheDtype::Fp8KTurbo3V,
            KvCacheDtype::Fp8KTurbo2V,
        ] {
            assert_eq!(
                hss_d2h_dst_bytes(dt, BLOCK_FLOATS, LAYER_BLOCK_BYTES),
                BLOCK_FLOATS,
                "{dt:?}"
            );
        }
        // nvfp4/turbo group -> layer_block_bytes (packed device stride)
        for dt in [
            KvCacheDtype::Nvfp4,
            KvCacheDtype::Turbo4,
            KvCacheDtype::Turbo4KTurbo3V,
            KvCacheDtype::Turbo4KTurbo8V,
            KvCacheDtype::Turbo3,
            KvCacheDtype::Turbo3KTurbo8V,
            KvCacheDtype::Turbo2,
            KvCacheDtype::Turbo8,
        ] {
            assert_eq!(
                hss_d2h_dst_bytes(dt, BLOCK_FLOATS, LAYER_BLOCK_BYTES),
                LAYER_BLOCK_BYTES,
                "{dt:?}"
            );
        }
    }

    /// The pinned staging buffer must cover the D2H dst for the layer's dtype.
    /// bf16 is the largest fixed case (block_floats*2); the buffer sized from
    /// hss_d2h_dst_bytes must never be smaller than what the copy writes.
    #[test]
    fn buffer_covers_copy_length() {
        let cases = [
            (KvCacheDtype::Bf16, BLOCK_FLOATS * 2),
            (KvCacheDtype::Fp8, BLOCK_FLOATS),
            (KvCacheDtype::Turbo8, LAYER_BLOCK_BYTES),
            (KvCacheDtype::Nvfp4, LAYER_BLOCK_BYTES),
        ];
        for (dt, copy_len) in cases {
            let buf = hss_d2h_dst_bytes(dt, BLOCK_FLOATS, LAYER_BLOCK_BYTES);
            assert!(buf >= copy_len, "{dt:?}: buf {buf} < copy_len {copy_len}");
        }
    }

    /// Only `1`/`true`/`on`/`yes` engage the pinned path; unset and every other
    /// token (incl. `0`/`false`/`off`/`no`/garbage) stay OFF = the byte-identical
    /// pageable default.
    #[test]
    fn flag_parse_truthy_only() {
        for on in ["1", "true", "on", "yes"] {
            assert!(parse_hss_pinned_flag(Some(on)), "{on:?} should engage");
        }
        assert!(!parse_hss_pinned_flag(None), "unset must be OFF");
        for off in ["0", "false", "off", "no", "", "TRUE", "1 ", "enable", "y"] {
            assert!(!parse_hss_pinned_flag(Some(off)), "{off:?} should be OFF");
        }
    }

    /// The pinned K and V regions are each sized to exactly `d2h_dst_bytes` and are
    /// two separate allocations (disjoint by construction — never a shared buffer).
    /// For a bf16 layer the K region reinterpreted as bf16 is exactly `block_floats`
    /// elements, matching the payload the offload consumes.
    #[test]
    fn pinned_pair_sizing_and_bf16_view() {
        // bf16 layer: dst bytes == block_floats*2, so a bf16 view is block_floats.
        let dst = hss_d2h_dst_bytes(KvCacheDtype::Bf16, BLOCK_FLOATS, LAYER_BLOCK_BYTES);
        assert_eq!(dst, BLOCK_FLOATS * 2);
        assert_eq!(dst / std::mem::size_of::<half::bf16>(), BLOCK_FLOATS);
        // Two independently-allocated Vecs of `dst` bytes model the disjoint K/V
        // pinned regions: each is exactly `dst` bytes and their address ranges do
        // not overlap.
        let k = vec![0u8; dst];
        let v = vec![0u8; dst];
        assert_eq!(k.len(), dst);
        assert_eq!(v.len(), dst);
        let k_range = k.as_ptr() as usize..k.as_ptr() as usize + dst;
        let v_range = v.as_ptr() as usize..v.as_ptr() as usize + dst;
        assert!(
            k_range.end <= v_range.start || v_range.end <= k_range.start,
            "K and V staging regions must be disjoint"
        );
    }
}
