I now have complete ground truth. The MRoPE consumer also resets `img_idx=0` per chunk and reads `chunk_tokens` only. Both consumers confirm Review 2's BROKEN finding precisely: **the existing single-stream multi-chunk image path already resets to row 0 / grid 0 per chunk**. This makes the safest correct scope clear: gate vision co-dispatch to single-chunk-fit image prompts (and document that multi-chunk image prefill is a pre-existing limitation).

I have everything I need to write the final synthesized plan.

---

# FINAL IMPLEMENTATION PLAN — Batched Qwen3-VL ViT (cross-image co-dispatch)

**Goal:** stop the ViT from running once per image. Batch all concurrent images into ONE ViT forward so each block's GEMM weights (qkv/proj/fc1/fc2/patch_embed/merger) are read once over `M = Σpatches`. Win target: image TTFT C8 drops from ~6.5s toward ~1s.

**Scope decision (load-bearing, from REVIEW 2 Issue #1 — BROKEN otherwise):** Vision co-dispatch batches only image prompts that **fit entirely in chunk 0**. Both the splice (`embed_chunk.rs:123-124`) and MRoPE (`upload_meta.rs:119-122`) reset `img_idx=0` per chunk and index only `chunk_tokens` — so a pad run straddling a chunk boundary is **already broken in the existing single-stream code**. We do NOT fix that pre-existing bug here; we gate around it. This makes the per-stream base a chunk-0-only concern, eliminating the `PrefillInProgress`/`run_standard.rs` re-base plumbing that REVIEW 2 said was required for the unbounded version.

---

## PART 1 — `vision_encoder.rs` batched forward

Files: all under `crates/spark-model/src/layers/vision_encoder/enc_impl/`.

### 1.1 `forward.rs` — `forward()` becomes a 1-image shim; add `forward_batched`

**BEFORE** (forward.rs:12-125, the entire `impl VisionEncoder` block). The current `forward` writes the final merger to `buf_out.offset(0)` on every call (line 103) — the N>1 overwrite bug from MAP C §1.

**AFTER** — replace the body:

```rust
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};   // ADD DevicePtr to the import (currently only GpuBackend)

use super::super::VisionEncoder;

impl VisionEncoder {
    /// Single-image forward (back-compat shim). For N=1 this issues the SAME
    /// kernels with the SAME args in the SAME order as the old per-image path
    /// → byte-identical output. Returns total_rows = (1+n_deepstack)*merged_p,
    /// the OLD return value (only ever tested `> 0` downstream).
    pub fn forward(
        &self,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<usize> {
        let images = [(pixels, grid_h, grid_w)];
        let per_image = self.forward_batched(&images, gpu, stream)?;
        let merged_p = per_image[0].2;
        Ok((1 + self.deepstack_indexes.len()) * merged_p)
    }

    /// Batched forward over N images. M-agnostic ops (patch_embed + all 27
    /// blocks' GEMMs/norms/gelu/residuals) run ONCE over M=Σpᵢ; per-image-
    /// geometry stages (host pos/rope prep, attention, mergers) loop per image.
    ///
    /// buf_out layout (rows of out_hidden_size BF16):
    ///   [0 .. Σmerged_p)                = final merger, IMAGE-ORDER packed   ← splicer reads here
    ///   [(k+1)*Σmerged_p .. )           = deepstack-k, image-order packed     ← LLM-unused
    ///
    /// IN-BOUNDS INVARIANT (REVIEW 1 Issue #1): the deepstack high-water row is
    /// 4·Σmerged_p = Σp ≤ p_max. This holds ONLY because all four mergers emit
    /// exactly merged_p rows each. buf_out is p_max rows → no realloc, no overrun.
    ///
    /// Returns per-image (post_h, post_w, merged_p) in image order.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        let sms2 = self.spatial_merge_size * self.spatial_merge_size;
        let sms = self.spatial_merge_size.max(1);
        let n_img = images.len();

        let mut p_i = Vec::with_capacity(n_img);
        let mut p_off = Vec::with_capacity(n_img);
        let mut mp_i = Vec::with_capacity(n_img);
        let mut mp_off = Vec::with_capacity(n_img);
        let (mut p_total, mut mp_total) = (0usize, 0usize);
        for (_px, gh, gw) in images.iter() {
            let p = gh * gw;
            let mp = p / sms2;
            p_off.push(p_total);
            mp_off.push(mp_total);
            p_i.push(p);
            mp_i.push(mp);
            p_total += p;
            mp_total += mp;
        }

        // Caller (scheduler + prepare_vision_embed_dispatch) must cap Σp ≤ p_max.
        // Defend here: oversized → per-image fallback that still packs buf_out.
        if p_total > self.p_max {
            return self.forward_oversized_fallback(images, &p_i, &mp_i, &mp_off, sms, gpu, stream);
        }

        // 1. Per-image host prep, packed into the SHARED buffers at p_off[i].
        let pos_interp_on = std::env::var("ATLAS_VISION_POSINTERP")
            .map(|v| v != "0").unwrap_or(true);
        for (i, (_px, gh, gw)) in images.iter().enumerate() {
            let p = p_i[i];
            let pos_dst = self.buf_pos_resampled.offset(p_off[i] * self.hidden_size * 2);
            if pos_interp_on {
                self.resample_pos_embed_into(*gh, *gw, pos_dst, gpu, stream)?;
            } else {
                self.gpu_copy_bf16(gpu, self.pos_embed, pos_dst, p * self.hidden_size * 2, stream)?;
            }
            let cos_dst = self.buf_rope_cos.offset(p_off[i] * self.head_dim * 2);
            let sin_dst = self.buf_rope_sin.offset(p_off[i] * self.head_dim * 2);
            self.build_rope_cossin_into(*gh, *gw, cos_dst, sin_dst, gpu, stream)?;
        }

        // 2. Patch embed over M=Σp.
        self.patch_embed_batched(images, &p_off, p_total, gpu, stream)?;
        Self::maybe_dump_buf(gpu, self.buf_h1, p_total * self.hidden_size, "patch_embed", stream)?;

        // 3. 27 blocks: M-agnostic ops once, attention per image, deepstack per image.
        let mut deepstack_iter = self.deepstack_indexes.iter().enumerate();
        let mut next_ds = deepstack_iter.next();
        for (block_idx, blk) in self.blocks.iter().enumerate() {
            self.vit_block_batched(blk, p_total, &p_i, &p_off, gpu, stream)?;
            Self::maybe_dump_buf(gpu, self.buf_h1, p_total * self.hidden_size,
                                 &format!("block{block_idx:02}"), stream)?;
            if let Some((ds_idx, &ds_block)) = next_ds
                && block_idx + 1 == ds_block
            {
                let n_h_bytes = p_total * self.hidden_size * 2;
                self.gpu_copy_bf16(gpu, self.buf_h1, self.buf_h2, n_h_bytes, stream)?;
                let ds_region_base = (ds_idx + 1) * mp_total;   // = mp_total + ds_idx*mp_total
                for (i, (_px, gh, gw)) in images.iter().enumerate() {
                    let src = self.buf_h2.offset(p_off[i] * self.hidden_size * 2);
                    let out_rows = ds_region_base + mp_off[i];
                    let out_slice = self.buf_out.offset(out_rows * self.out_hidden_size * 2);
                    self.apply_merger(&self.deepstack[ds_idx], p_i[i], *gh, *gw, src, out_slice, gpu, stream)?;
                }
                next_ds = deepstack_iter.next();
            }
        }

        // 4. Final merger per image → packed [0..Σmerged_p).
        for (i, (_px, gh, gw)) in images.iter().enumerate() {
            let src = self.buf_h1.offset(p_off[i] * self.hidden_size * 2);
            let out_slice = self.buf_out.offset(mp_off[i] * self.out_hidden_size * 2);
            self.apply_merger(&self.merger, p_i[i], *gh, *gw, src, out_slice, gpu, stream)?;
        }
        Self::maybe_dump_buf(gpu, self.buf_out, mp_total * self.out_hidden_size, "final", stream)?;

        Ok(images.iter().map(|(_px, gh, gw)| (gh / sms, gw / sms, (gh * gw) / sms2)).collect())
    }
```

**N=1 byte-identity** (REVIEW 1 confirmed SOUND): `p_off=[0]`, `mp_off=[0]`, so every `_into`/`_batched` helper resolves to the original base address and M=p. Host prep, patch_embed, blocks, deepstack tap (`ds_region_base = (ds_idx+1)*merged_p` matches old line 86), and final merger (`offset(0)` matches old line 103) issue the identical kernel stream.

### 1.2 `forward.rs` — oversized fallback (REVIEW 1 Issue #3 fixes folded in)

REVIEW 1 found the original `park` deepstack write corrupts the final region and `Σmerged_p` can exceed `buf_out`. **Fixes folded in:** drop the deepstack write entirely (LLM-unused), add a bounds `debug_assert`, and rely on the scheduler cap so this path is effectively dead. Append to forward.rs:

```rust
    /// Fallback for Σp > p_max: encode each image alone (full single-image
    /// kernel sequence) writing its final-merger rows into the PACKED buf_out
    /// at mp_off[i]. NO deepstack write (LLM-unused; the old `park` offset
    /// corrupted the final region — REVIEW 1 #3). The scheduler caps Σp ≤ p_max
    /// so this is normally unreachable; it exists purely as a correctness guard.
    fn forward_oversized_fallback(
        &self,
        images: &[(&[f32], usize, usize)],
        p_i: &[usize],
        mp_i: &[usize],
        mp_off: &[usize],
        sms: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        // buf_out holds p_max rows; the packed final region needs Σmerged_p. If
        // a caller ignores the cap and Σmerged_p > p_max we cannot pack safely.
        debug_assert!(
            mp_off.last().map(|o| o + mp_i.last().unwrap()).unwrap_or(0) <= self.p_max,
            "oversized vision batch: Σmerged_p exceeds buf_out rows"
        );
        let pos_interp_on = std::env::var("ATLAS_VISION_POSINTERP")
            .map(|v| v != "0").unwrap_or(true);
        for (i, (pixels, gh, gw)) in images.iter().enumerate() {
            let p = p_i[i];
            if pos_interp_on {
                self.resample_pos_embed(*gh, *gw, gpu, stream)?;
            } else {
                self.gpu_copy_bf16(gpu, self.pos_embed, self.buf_pos_resampled,
                                   p * self.hidden_size * 2, stream)?;
            }
            self.build_rope_cossin(*gh, *gw, gpu, stream)?;
            self.patch_embed(pixels, p, gpu, stream)?;
            // No deepstack tap in the fallback (correctness-only, LLM-unused).
            for blk in self.blocks.iter() {
                self.vit_block(blk, p, gpu, stream)?;
            }
            let out_slice = self.buf_out.offset(mp_off[i] * self.out_hidden_size * 2);
            self.apply_merger(&self.merger, p, *gh, *gw, self.buf_h1, out_slice, gpu, stream)?;
        }
        Ok(images.iter().map(|(_px, gh, gw)| (gh / sms, gw / sms, (gh * gw) / (sms * sms))).collect())
    }
}
```

### 1.3 `pos_embed.rs` — `_into` variants (REVIEW 1: SOUND)

Add `use spark_runtime::gpu::DevicePtr;`. Rename `resample_pos_embed`→`resample_pos_embed_into(..., dst: DevicePtr, ...)`, changing ONLY the final `copy_h2d_async(bytes, dst, stream)` target; add a zero-offset shim `resample_pos_embed` calling `..._into(.., self.buf_pos_resampled, ..)`. Same for `build_rope_cossin`→`build_rope_cossin_into(..., cos_dst, sin_dst, ...)` (final two uploads target `cos_dst`/`sin_dst`), with a shim. Bodies otherwise character-identical → guarantees per-image rows are byte-identical to single-image rows.

### 1.4 `patch_embed.rs` — `patch_embed_batched` (REVIEW: SOUND)

Add `patch_embed_batched(images, p_off, p_total, gpu, stream)`: loop-upload each image's f32 pixels into `buf_f32.offset(p_off[i]*1536*4)`; then ONE `k_f32_bf16` over `p_total*1536` → `buf_wide`, ONE `vit_gemm_bias` at M=`p_total` → `buf_h1`, ONE `k_add` (+pos) over `p_total*hidden`. `DevicePtr`/`div_ceil` already in scope. N=1 = identical kernel stream.

### 1.5 `vit_block.rs` — `vit_block_batched` (REVIEW 1 Issue #4: SOUND)

Add `vit_block_batched(blk, p_total, p_i, p_off, gpu, stream)`: identical to `vit_block` except (a) all element/GEMM counts use `p_total`/`pt`, and (b) **attention loops per image**:

```rust
        // QKV GEMM (step 3) populates buf_wide for the WHOLE batch first.
        // ... then attention, PER IMAGE over its disjoint slice:
        for (i, &p) in p_i.iter().enumerate() {
            let p32 = p as u32;
            let sm_bytes = ((p + self.head_dim) * std::mem::size_of::<f32>()) as u32;
            let qkv = self.buf_wide.offset(p_off[i] * qkv_n as usize * 2);   // qkv_n=3456
            let o   = self.buf_h1.offset(p_off[i] * self.hidden_size * 2);
            let cos = self.buf_rope_cos.offset(p_off[i] * self.head_dim * 2);
            let sin = self.buf_rope_sin.offset(p_off[i] * self.head_dim * 2);
            KernelLaunch::new(gpu, self.k_attn)
                .grid([p32, self.num_heads as u32, 1]).block([32, 1, 1])
                .shared_mem(sm_bytes)
                .arg_ptr(qkv).arg_ptr(o).arg_ptr(cos).arg_ptr(sin)
                .arg_u32(p32).arg_u32(self.num_heads as u32).arg_u32(self.head_dim as u32)
                .launch(stream)?;
        }
```

Per MAP B §1: the kernel indexes `qi/kj/vj ∈ [0, seq)` relative to the passed bases → SDPA never crosses image boundaries. `buf_wide` (QKV) is read-only during the loop; each image writes a disjoint `buf_h1` row range; proj (step 5) reads the fully-populated `buf_h1` at M=`p_total`. No hazard (same-stream ordering).

### 1.6 `prepare_vision_embed_dispatch` → `forward_batched` (prefill_a.rs:38-63)

This fixes the MAP C §1 N>1 overwrite bug for the non-co-dispatched path too. **AFTER**:

```rust
    pub(super) fn prepare_vision_embed_dispatch(
        &self,
        images: &[(Vec<f32>, usize, usize)],
    ) -> Result<()> {
        let ve = match &self.vision_encoder { Some(ve) => ve, None => return Ok(()) };
        let stream = self.gpu.default_stream();
        let img_refs: Vec<(&[f32], usize, usize)> =
            images.iter().map(|(px, gh, gw)| (px.as_slice(), *gh, *gw)).collect();
        let per_image = ve.forward_batched(&img_refs, self.gpu.as_ref(), stream)?;
        let grids: Vec<(usize, usize)> = per_image.iter().map(|(h, w, _)| (*h, *w)).collect();
        let total_merged: usize = per_image.iter().map(|(_, _, mp)| mp).sum();
        *self.vision_embed_patches.lock() = total_merged;   // any > 0 satisfies the flag
        *self.vision_image_grids.lock() = grids;
        Ok(())
    }
```

### 1.7 `apply_merger` per-image serialization (REVIEW 1 Issue #6 — perf follow-up, NOT a blocker)

The N final + 3N deepstack mergers share `buf_merge_in`/`buf_merge_fc1` at offset 0, so they serialize and re-read merger weights N times. **Not a correctness bug** (single-stream, in-order). Leave as-is for v1. Documented follow-up: pack `vision_spatial_merge` per-image into `buf_merge_in[Σmerged_p, 4608]` (fits: ≤ p_max/4 rows), then ONE fc1/gelu/fc2 over Σmerged_p.

### 1.8 Files touched (Part 1)

`forward.rs` (shim + `forward_batched` + `forward_oversized_fallback`, add `DevicePtr` import), `pos_embed.rs` (`_into` + shims, add `DevicePtr` import), `patch_embed.rs` (`patch_embed_batched`), `vit_block.rs` (`vit_block_batched`), `prefill_a.rs` (`prepare_vision_embed_dispatch`). **No changes** to `merger.rs`, `utils.rs`, `init.rs` — `apply_merger` is offset-clean; all buffers are p_max-sized.

### 1.9 Single-image byte-identity validation (the gate before Part 2)

`ATLAS_DUMP_VIT=<dir>` (utils.rs) dumps `patch_embed`, `block00..block26`, `final` BF16 buffers.
1. On `main` (old `forward`): one image, `ATLAS_DUMP_VIT=/tmp/vit_old`.
2. On this branch (shim→`forward_batched`): same image, `ATLAS_DUMP_VIT=/tmp/vit_new`.
3. `cmp -l /tmp/vit_old/final.bin /tmp/vit_new/final.bin` and every `block*.bin`/`patch_embed.bin` must report **zero differing bytes**. Any diff localizes the regression to one stage.
4. Saturn smoke test still passes single-image.

---

## PART 2 — Scheduler vision co-dispatch

Gate: **single-chunk-fit image prompts only** (REVIEW 2 Issue #1). This removes the `PrefillInProgress`/`run_standard.rs` re-base plumbing the unbounded version needed — the per-stream base is set once before the chunk-0 `prefill_chunk` and reset after.

### 2.1 `VisionSlice` descriptor + `image_pixels_ref` accessor

`phase_start_prefills.rs` (top, after `use`):
```rust
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VisionSlice {
    pub patch_row_offset: usize,   // first buf_out row this request owns
    pub grid_index_offset: usize,  // first vision_image_grids index
    pub num_images: usize,
    pub patch_row_count: usize,
}
```
`api/inference_impl.rs` (next to `has_image_pixels`):
```rust
pub fn image_pixels_ref(&self) -> &[(Vec<f32>, usize, usize)] {
    match self {
        InferenceRequest::Streaming { image_pixels, .. }
        | InferenceRequest::Blocking { image_pixels, .. } => image_pixels.as_slice(),
    }
}
```

### 2.2 Batched model entry point

`traits/model.rs` (after `prepare_vision_embed`):
```rust
fn prepare_vision_embed_batched(
    &self,
    per_request: &[Vec<(Vec<f32>, usize, usize)>],
) -> Result<Vec<(usize, usize, usize, usize)>> {   // (row_off, grid_off, n_img, row_cnt) per request
    let _ = per_request; Ok(Vec::new())
}
/// Per-stream chunk-0 base for the NEXT prefill_chunk's splice + MRoPE.
fn set_vision_slice_base(&self, _row_base: usize, _grid_base: usize, _owned_images: usize) {}
```
`trait_impl/mod.rs` overrides both (the second sets the three new mutex fields). `trait_impl/prefill_a.rs` adds `prepare_vision_embed_batched_dispatch`:
```rust
pub(super) fn prepare_vision_embed_batched_dispatch(
    &self,
    per_request: &[Vec<(Vec<f32>, usize, usize)>],
) -> Result<Vec<(usize, usize, usize, usize)>> {
    let ve = match &self.vision_encoder { Some(ve) => ve, None => return Ok(Vec::new()) };
    let stream = self.gpu.default_stream();
    let mut flat: Vec<(&[f32], usize, usize)> = Vec::new();
    let mut req_bounds = Vec::with_capacity(per_request.len());
    for imgs in per_request {
        req_bounds.push((flat.len(), imgs.len()));
        for (px, gh, gw) in imgs { flat.push((px.as_slice(), *gh, *gw)); }
    }
    let per_image = ve.forward_batched(&flat, self.gpu.as_ref(), stream)?; // Vec<(ph,pw,merged_p)>
    let mut grids = Vec::with_capacity(per_image.len());
    let mut total_rows = 0usize;
    for (ph, pw, mp) in &per_image { grids.push((*ph, *pw)); total_rows += *mp; }
    *self.vision_embed_patches.lock() = total_rows;
    *self.vision_image_grids.lock() = grids;
    let mut out = Vec::with_capacity(per_request.len());
    let mut row_cursor = 0usize;
    for (img_start, n_img) in req_bounds {
        let row_count: usize = per_image[img_start..img_start + n_img].iter().map(|(_,_,mp)| *mp).sum();
        out.push((row_cursor, img_start, n_img, row_count));
        row_cursor += row_count;
    }
    Ok(out)
}
```
New mutex fields in `model/types.rs` (init 0): `vision_row_base`, `vision_grid_base`, `vision_owned_images` (`parking_lot::Mutex<usize>`).

### 2.3 Scheduler gather pre-pass (`phase_start_prefills.rs`, between L55-56)

Inserted before the `for req in new_reqs` loop. **REVIEW 2 fixes folded in:** (a) cap-overflow **fully disables** batching for the tick (not partial-batch + per-request remainder — eliminates the `buf_out`-reuse race, REVIEW 2 Additional fix); (b) **single-chunk-fit gate** so multi-chunk image prompts fall back to per-request encode (REVIEW 2 Issue #1).

```rust
let vision_codispatch_enabled = std::env::var("ATLAS_VISION_CODISPATCH")
    .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false"))).unwrap_or(true);
const VISION_P_MAX: usize = 6400;   // Part-1 scratch cap (Σ pre-merge patches)

let mut vision_slices: Vec<VisionSlice> = vec![VisionSlice::default(); new_reqs.len()];
if vision_codispatch_enabled && chunked {
    // chunk-0 budget for THIS tick (mirrors the budget below).
    let chunk0_budget = if active.is_empty() && prefilling.is_empty() { max_batch_tokens } else { max_prefill_tokens };

    let mut batched_req_idx: Vec<usize> = Vec::new();
    let mut per_request_imgs: Vec<Vec<(Vec<f32>, usize, usize)>> = Vec::new();
    let mut running_patches = 0usize;
    let mut overflow = false;
    for (k, req) in new_reqs.iter().enumerate() {
        if !req.has_image_pixels() { continue; }
        // SINGLE-CHUNK-FIT GATE (REVIEW 2 #1): the splice/MRoPE reset img_idx=0
        // per chunk, so a pad run straddling a chunk boundary is wrong. Only
        // co-dispatch prompts whose whole token stream fits chunk 0.
        if req.prompt_len() > chunk0_budget { continue; }   // → self-encodes per-request
        let imgs = req.image_pixels_ref();
        let req_patches: usize = imgs.iter().map(|(_, gh, gw)| gh * gw).sum();
        if running_patches + req_patches > VISION_P_MAX { overflow = true; break; }
        running_patches += req_patches;
        batched_req_idx.push(k);
        per_request_imgs.push(imgs.to_vec());
    }
    if overflow {                       // FULL-DISABLE for the tick (race fix)
        batched_req_idx.clear(); per_request_imgs.clear();
    }
    if batched_req_idx.len() >= 2 {
        match model.prepare_vision_embed_batched(&per_request_imgs) {
            Ok(descs) if descs.len() == batched_req_idx.len() => {
                for (slot, (row_off, grid_off, n_img, row_cnt)) in
                    batched_req_idx.iter().zip(descs.into_iter())
                {
                    vision_slices[*slot] = VisionSlice {
                        patch_row_offset: row_off, grid_index_offset: grid_off,
                        num_images: n_img, patch_row_count: row_cnt,
                    };
                }
                // ONE fence for the whole batch (mirrors prefill_a_step.rs:207-208).
                if let Err(e) = model.record_event(prefill_event, model.default_stream())
                    .and_then(|_| model.stream_wait_event(prefill_stream, prefill_event))
                { tracing::error!("vision co-dispatch fence failed: {e:#}"); }
            }
            Ok(_) | Err(_) => {
                // Per-stream fallback: leave all slices Default → each image
                // request self-encodes in its own start_chunked_prefill,
                // reporting any genuine error to its own sink.
                tracing::warn!("vision batched encode failed/mismatched; per-request fallback this tick");
            }
        }
    }
}
```
Loop header: `for req in new_reqs` → `for (req_idx, req) in new_reqs.into_iter().enumerate()`. (`prompt_len()` — add a trivial accessor if absent; it must equal the token count `start_chunked_prefill` uses for `prompt_tokens.len()`. Verify against how prompt tokens are derived in `start_chunked_prefill`; if tokenization happens inside it, instead gate on the post-tokenization length there and treat the pre-pass gate as best-effort — see Risk R4.)

### 2.4 Plumb `VisionSlice` into `start_chunked_prefill`

`prefill_a_step.rs` signature: add `vision_slice: Option<VisionSlice>` after `defer: bool`. Call site (2.3 loop):
```rust
let slice = vision_slices[req_idx];
let slice_opt = if slice.num_images > 0 { Some(slice) } else { None };
// ... pass slice_opt as the new last arg to start_chunked_prefill(...)
```
Skip per-request encode+fence when pre-encoded (replace prefill_a_step.rs:196-209):
```rust
if vision_slice.is_none() && !image_pixels.is_empty() {
    model.prepare_vision_embed(&image_pixels)?;
    model.record_event(prefill_event, model.default_stream())?;
    model.stream_wait_event(prefill_stream, prefill_event)?;
}
```
Set/reset the per-stream base around the chunk-0 `prefill_chunk` (prefill_a_step.rs:222). Because the gate guarantees single-chunk-fit, this stream finishes prefill in chunk 0 — `set_vision_slice_base` then reset is correct and complete (REVIEW 2 #2 satisfied by the single-chunk gate, no `PrefillInProgress` carry needed):
```rust
if let Some(s) = vision_slice {
    model.set_vision_slice_base(s.patch_row_offset, s.grid_index_offset, s.num_images);
}
let chunk_res = model.prefill_chunk(&prompt_tokens, &mut seq, 0, chunk_len, is_last, prefill_stream);
if vision_slice.is_some() { model.set_vision_slice_base(0, 0, 0); }
chunk_res
```

### 2.5 Consumers read the base

`embed_chunk.rs:123-130` — splice from `row_base + img_idx`:
```rust
let row_base = *self.vision_row_base.lock();   // 0 for legacy single encode
let mut img_idx = 0usize;
for (i, &tok) in chunk_tokens.iter().enumerate() {
    if tok == pad_id {
        let src = ve.buf_out.offset((row_base + img_idx) * ve.out_hidden_size * 2);
        let dst = hidden_dst.offset(i * h * elem_bytes);
        self.gpu.copy_d2d_async(src, dst, ve.out_hidden_size * 2, stream)?;
        img_idx += 1;
    }
}
```
Comment to add at the `pending > 0` gate (embed_chunk.rs:111, REVIEW 2 #4): the total is now shared across streams; a text-only co-tenant stream is safe only because its `chunk_tokens` carry no pad tokens (loop is a no-op).

`upload_meta.rs:105,119,122` — MRoPE starts at `grid_base`, bounded by `grid_base+owned`:
```rust
let grids = self.vision_image_grids.lock().clone();
let grid_base = *self.vision_grid_base.lock();
let owned = *self.vision_owned_images.lock();
let grid_hi = if owned > 0 { (grid_base + owned).min(grids.len()) } else { grids.len() };
// ...
let mut img_idx = grid_base;
// loop guard: while i < chunk_tokens.len() { if chunk_tokens[i]==pad_id && img_idx < grid_hi { ... } }
```

### 2.6 Untouched paths (REVIEW 2: SOUND)

Non-chunked path (`prefill_b_step.rs`) — pre-pass is gated `chunked`, `vision_slice` never `Some`, bases stay 0, `prefill_c.rs` reads row 0 as today. Mixed batch (some streams text-only) — text-only get `VisionSlice::default()` → `None` → legacy branch (no-op since `image_pixels.is_empty()`). Prefix cache — no gate needed (image prompts bypass prefix cache at `prefill_a.rs:106` via `tokens_have_vision_pad`).

### 2.7 Files touched (Part 2)

`phase_start_prefills.rs` (`VisionSlice`, pre-pass, `.enumerate()`, pass `slice_opt`), `prefill_a_step.rs` (`vision_slice` param, skip encode+fence, base set/reset), `scheduler/mod.rs` (re-export `VisionSlice` if `super::*` doesn't surface it), `api/inference_impl.rs` (`image_pixels_ref`, `prompt_len` if needed), `traits/model.rs` (two trait methods), `trait_impl/mod.rs` (overrides), `trait_impl/prefill_a.rs` (`prepare_vision_embed_batched_dispatch`), `model/types.rs` (three mutex fields), `embed_chunk.rs` (row_base), `upload_meta.rs` (grid_base/grid_hi).

---

## BUILD

Use the EXACT CLAUDE.md remote build on `gx10-9959` (separate filesystem — `rsync` edits over first):
```bash
ssh gx10-9959 'cd ~/atlas && source ~/.cargo/env
  export PATH=/usr/local/cuda/bin:$PATH
  export CUTLASS_HOME=$HOME/cutlass
  export FLASHINFER_HOME=$HOME/flashinfer
  export RUSTFLAGS="-L/home/ms/nccl/build/lib -L/usr/local/cuda/lib64"
  export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4
  cargo build --release -p spark-server --bin spark --no-default-features --features cuda'
```
Build log must say `compiled N kernels for target 0 (gb10, holo-3.1-35b-a3b, nvfp4)`. This change is host-side Rust only (no `.cu` edits), but the full env is still required or the binary loads-but-fails. Serve with `bash scripts/holo_serve.sh /tmp/holo.log` (uses prebuilt `target/release/spark`, does not build).

---

## VALIDATION PLAN

**(a) Single-image byte-identical (gate before anything else).** Per Part 1.9: `ATLAS_DUMP_VIT` dumps on `main` vs branch for the same image; `cmp -l` on `patch_embed.bin`, every `block*.bin`, `final.bin` must show **zero differing bytes**. Then Saturn smoke test single-image → "a planet with rings, resembling Saturn". **Pass bar: zero byte diffs + correct caption.**

**(b) Multi-image single request correct.** One request with 2 images (the previously-broken N>1 case — MAP C §1 overwrite). Verify each image's final rows are distinct/non-zero: `cmp` rows `[0, mp0)` vs `[mp0, mp0+mp1)` of `final.bin` differ. Greedy output must reference both images correctly. **Pass bar: both images described correctly; the two row ranges differ.**

**(c) Concurrent image TTFT flattens (the win).** Concurrent image requests (C1, C2, C4, C8), each carrying a single chunk-0-fit image, with `ATLAS_VISION_CODISPATCH=1`. Measure per-request image TTFT (prefill via `max_ttft`, not wall; `max_tokens>=250`). Baseline today: ~0.86→6.53s (C1→C8). **Pass bar: C8 image TTFT drops from ~6.5s toward ~1s** (i.e. roughly flat across C1→C8, like vLLM's 0.16→0.66s). Cross-check log: exactly ONE "Vision (batched)" / one batched-encode per tick for the co-dispatched images (not N encodes). Also A/B `ATLAS_VISION_CODISPATCH=0` to confirm regression back to ~6.5s (isolates the win to this path).

**(d) Mixed-batch correctness.** One tick with 2 image streams + 1 text-only stream + 1 oversized/multi-chunk image stream: image streams batched + correct, text stream unaffected, oversized stream self-encodes correctly. **Pass bar: all four produce correct output; no cross-tenant contamination.**

---

## RISK / ROLLBACK

**Env gate (rollback):** `ATLAS_VISION_CODISPATCH=0` disables the entire Part-2 pre-pass → every image request self-encodes per-request via the Part-1 `forward_batched` (which is still correct for N≥1 and byte-identical for N=1). Part 1 has no separate gate but is byte-identical single-image, so it is always safe to ship; the only behavioral switch is the scheduler co-dispatch.

**What could still go wrong / reviewer-flagged residuals:**
- **R1 — Multi-chunk image prefill is NOT fixed (REVIEW 2 Issue #1, pre-existing).** The splice/MRoPE reset `img_idx=0` per chunk and index only `chunk_tokens` — a pad run straddling a chunk boundary is wrong **in the existing single-stream code today**. We do NOT fix it; we gate co-dispatch to single-chunk-fit prompts (2.3). Multi-chunk image prompts fall back to the unchanged per-request path (same correctness as `main`). If the single-chunk gate is bypassed/mis-sized, a multi-chunk co-dispatched image silently mis-splices. **Mitigation: the gate is the firewall; verify `prompt_len()` matches the actual chunk-0 token count (R4).**
- **R2 — `buf_out`-reuse race on overflow (REVIEW 2 Additional fix, RESOLVED).** Adopted full-disable on cap-overflow (2.3), so no mixed batched+fallback `buf_out` reuse within a tick. If anyone reverts to partial-batch, the race returns (a default-stream fallback encode can overwrite `buf_out` before an earlier prefill-stream injection executes).
- **R3 — `forward_oversized_fallback` is correctness-only and unreachable under the cap (REVIEW 1 Issue #3, RESOLVED).** Deepstack `park` write removed; bounds `debug_assert` added. If a caller ever ignores the cap AND `Σmerged_p > p_max`, the assert fires in debug; release would corrupt — but the scheduler cap (`VISION_P_MAX=6400`) plus the in-`forward_batched` guard make this unreachable in practice.
- **R4 — `prompt_len()` accessor must equal the token count used downstream.** If prompt tokenization happens inside `start_chunked_prefill` (not before the pre-pass), the pre-pass gate is approximate. Verify where `prompt_tokens` is built; if it's inside, either expose the same count to the pre-pass or move the single-chunk-fit check into `start_chunked_prefill` and have it self-encode when the prompt won't fit chunk 0 (treat the pre-pass `VisionSlice` as advisory and ignore it when oversized).
- **R5 — Merger serialization (REVIEW 1 Issue #6).** Mergers still re-read weights N times (perf only, not correctness). Acceptable for v1; the GEMM win is in the 27 blocks (the dominant cost). Follow-up to batch the merger is documented in 1.7.
- **R6 — Adversarial DoS (REVIEW 2 Issue #4).** One malformed image forces the whole tick to per-request fallback (redundant encodes). Acceptable; isolation > throughput under attack.
- **R7 — Confirm remaining `forward()` callers (REVIEW 1 Issue #7).** Grep `\.forward(` on `VisionEncoder` to confirm every remaining single-image caller still relies on the `total_rows` return contract the shim reproduces. The dispatch path (1.6) now uses `forward_batched` directly; any other caller must be enumerated before merge.

**Nothing the reviewers marked BROKEN is left unresolved:** REVIEW 1 #1 (buffer-sizing) → real invariant `4·Σmerged_p = Σp ≤ p_max` documented in the `forward_batched` doc-comment with the "all four mergers emit merged_p rows" assumption made explicit; REVIEW 1 #3 (oversized fallback) → deepstack park removed + assert; REVIEW 2 #1 (multi-chunk) → single-chunk-fit gate; REVIEW 2 race → full-disable on overflow.