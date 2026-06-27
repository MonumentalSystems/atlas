All confirmed. `offset` takes bytes. Now I'll write the final implementation doc.

---

# FINAL IMPLEMENTATION PLAN — Replace `vision_attention_rope`

## Decision: GEMM-based (Approach 1), with all Review-1 fixes folded in. Fused-flash (Approach 2) is REJECTED.

**Rationale (one paragraph):** The GEMM path is buildable entirely from a kernel already trusted *in this exact file* — `dense_gemm_bf16_pipelined` is what `vit_gemm_bias` already launches for QKV/proj (init.rs:86, vit_block.rs:31), and `dense_gemm_bf16_f32out` is its f32-output sibling in the same module `"gemm"`. Both **already zero-mask the K/N tail** (dense_gemm_bf16.cu:100-108 for f32out, :385-415 for pipelined), so `head_dim=72` needs zero special handling — it is the single biggest risk-eliminator and it is the property that kills the fused path: Approach 2's only viable reuse target (`inferspark_prefill`) bakes `HDIM` as a compile-time constant with an even warp-pair split that 72 breaks (MAP B), so a fused kernel must be hand-written. Review 2 then found that hand-written kernel has a real local-memory-spill cliff (`acc[d]`/`q_reg[d]` indexed by a runtime loop variable → off-chip local memory on the hottest loop) that **plausibly negates the entire speedup** and is unverifiable without ncu, plus a `VIT_QTILE`/`qi` correctness landmine. The GEMM path moves the O(seq²·D) work onto BF16 tensor cores with no register-array hazard, the only numeric cost is two extra bf16 roundings (Qr/Kr and P) that are standard SDPA-as-GEMM and validate at rel-err ~1e-2. Expected: 398 ms → ~10-25 ms (Review-1's tempered 16-40×, not the optimistic 50×). We ship it behind `ATLAS_VISION_ATTN_LEGACY=1` for instant A/B rollback.

**Review-1 fixes folded in (all 5):** BUG1 GEMM1 grid/block = 16 not 32; BUG3 `buf_o_stage` allocated; BUG4 `k_scatter_head` handle added; 5b `k_gemm_f32` via hard `gpu.kernel(...)?`; `debug_assert!(seq<=1024)` overflow guard. GEMM2 grid was already correct (N-first), kept.

**One BROKEN item to flag up front:** the legacy `vision_attention_rope` kernel comment claims smem `[seq+2*D]` but **both launch sites allocate only `(seq+D)*4` bytes** (vit_block.rs:79, :231) — MAP A confirmed this is harmless (the kernel only ever writes `q_rope[0..D)`). The new GEMM path uses **no dynamic smem at the call site** (softmax's tiny 32 B is static `__shared__`), so this discrepancy is sidestepped entirely. No action, noted so the implementer doesn't copy the `sm_bytes` arg into the new launches.

---

## 1. The `.cu` kernels

**Target file (verified symlink → physical):** `/home/ms/atlas/kernels/gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu`
(holo-3.1-35b-a3b/nvfp4/vision_encoder.cu → ../../qwen3.6-35b-a3b/nvfp4/vision_encoder.cu; editing this changes holo-3.1 AND qwen3.6 targets only.)

**Append after the close of `vision_attention_rope` (after line 273), keeping the legacy kernel intact.** Uses existing `bf16_to_f32`/`f32_to_bf16` (lines 14-19); no new includes.

```cpp
// ─────────────────────────────────────────────────────────────────────────────
// GEMM-based ViT SDPA (replacement for vision_attention_rope).
// Pipeline per image: vit_rope_deinterleave → [per head: GEMM1 QKᵀ (f32out) →
// vit_softmax_rows → GEMM2 P·V (pipelined) → vit_scatter_head].
// Semantics IDENTICAL to vision_attention_rope: interleaved QKV strides,
// rotate-half 2D RoPE on Q,K only, scale=rsqrt(D), non-causal max-subtracted
// softmax. The two GEMMs run on BF16 tensor cores; rope/softmax/scatter are
// memory-bound element passes.
// ─────────────────────────────────────────────────────────────────────────────

// (A) Deinterleave QKV[seq,3*H*D] → head-contiguous rotated Qr,Kr [H,seq,D] and
//     TRANSPOSED Vt [H,D,seq]. V is pre-transposed so GEMM2 (which computes
//     A·Bᵀ) yields P·V. One thread per (token,head,d) element.
//     grid = ( ceil(seq*D/256), H, 1 ), block = (256,1,1)
extern "C" __global__
void vit_rope_deinterleave(
    const __nv_bfloat16* __restrict__ QKV,      // [seq, 3*H*D] interleaved
    __nv_bfloat16* __restrict__ Qr,             // [H, seq, D]
    __nv_bfloat16* __restrict__ Kr,             // [H, seq, D]
    __nv_bfloat16* __restrict__ Vt,             // [H, D, seq]  (transposed!)
    const __nv_bfloat16* __restrict__ rope_cos, // [seq, D]
    const __nv_bfloat16* __restrict__ rope_sin, // [seq, D]
    unsigned int seq, unsigned int H, unsigned int D)
{
    unsigned int h   = blockIdx.y;
    unsigned int lin = blockIdx.x * blockDim.x + threadIdx.x; // tok*D + d
    if (h >= H || lin >= seq * D) return;
    unsigned int tok = lin / D;
    unsigned int d   = lin % D;
    unsigned int half_D = D / 2;

    unsigned int stride_qkv = 3u * H * D;
    const __nv_bfloat16* Q_row = QKV + (size_t)tok * stride_qkv + 0u * H * D + h * D;
    const __nv_bfloat16* K_row = QKV + (size_t)tok * stride_qkv + 1u * H * D + h * D;
    const __nv_bfloat16* V_row = QKV + (size_t)tok * stride_qkv + 2u * H * D + h * D;

    float cos_v = bf16_to_f32(rope_cos[(size_t)tok * D + d]);
    float sin_v = bf16_to_f32(rope_sin[(size_t)tok * D + d]);

    // Q rotate-half (verbatim lines 219-242 formula)
    float qv    = bf16_to_f32(Q_row[d]);
    float qpart = (d < half_D) ? bf16_to_f32(Q_row[d + half_D])
                               : bf16_to_f32(Q_row[d - half_D]);
    float qrot  = (d < half_D) ? -qpart : qpart;
    float q_r   = qv * cos_v + qrot * sin_v;

    // K rotate-half
    float kv    = bf16_to_f32(K_row[d]);
    float kpart = (d < half_D) ? bf16_to_f32(K_row[d + half_D])
                               : bf16_to_f32(K_row[d - half_D]);
    float krot  = (d < half_D) ? -kpart : kpart;
    float k_r   = kv * cos_v + krot * sin_v;

    // V un-rotated
    float v_v   = bf16_to_f32(V_row[d]);

    size_t hc = (size_t)h * seq * D + (size_t)tok * D + d;   // Qr/Kr [H,seq,D]
    Qr[hc] = f32_to_bf16(q_r);
    Kr[hc] = f32_to_bf16(k_r);
    // Vt [H,D,seq]: index h*D*seq + d*seq + tok  (d,tok swapped = transpose)
    Vt[(size_t)h * D * seq + (size_t)d * seq + (size_t)tok] = f32_to_bf16(v_v);
}

// (B) Row softmax over raw scores S[seq,seq] (f32) → probs P[seq,seq] (bf16).
//     scale = rsqrtf(D) folded in here (GEMM1 produced raw Q·Kᵀ). Parallel
//     block-per-row, 3-pass (max / sumexp / normalize). Replaces the legacy
//     single-thread softmax.
//     grid = (seq,1,1), block = (256,1,1)
//     NOTE: uses expf() (not __expf) to match the legacy kernel during
//     byte-validation. Swap to __expf for a small speed gain after parity is
//     confirmed and Saturn still reads correct.
extern "C" __global__
void vit_softmax_rows(
    const float* __restrict__ S,        // [seq, seq] f32 raw scores
    __nv_bfloat16* __restrict__ P,      // [seq, seq] bf16 probs
    unsigned int seq, unsigned int D)
{
    unsigned int row = blockIdx.x;
    if (row >= seq) return;
    const float* srow = S + (size_t)row * seq;
    __nv_bfloat16* prow = P + (size_t)row * seq;
    float scale = rsqrtf((float)D);

    __shared__ float red[256 / 32];     // one slot per warp
    unsigned int tid  = threadIdx.x;
    unsigned int lane = tid & 31u, warp = tid >> 5;

    // pass 1: row max
    float m = -1e30f;
    for (unsigned int j = tid; j < seq; j += blockDim.x)
        m = fmaxf(m, srow[j] * scale);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = -1e30f;
        for (unsigned int w = 0; w < blockDim.x / 32; ++w) mm = fmaxf(mm, red[w]);
        red[0] = mm;
    }
    __syncthreads();
    float row_max = red[0];

    // pass 2: sum exp
    float s = 0.0f;
    for (unsigned int j = tid; j < seq; j += blockDim.x)
        s += expf(srow[j] * scale - row_max);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (unsigned int w = 0; w < blockDim.x / 32; ++w) ss += red[w];
        red[0] = (ss > 0.0f) ? (1.0f / ss) : 0.0f;
    }
    __syncthreads();
    float inv_sum = red[0];

    // pass 3: normalize + store bf16
    for (unsigned int j = tid; j < seq; j += blockDim.x)
        prow[j] = f32_to_bf16(expf(srow[j] * scale - row_max) * inv_sum);
}

// (C) Scatter contiguous Oh[seq,D] → interleaved O[seq, dst_stride] head slot.
//     grid = ( ceil(seq*D/256), 1, 1 ), block = (256,1,1)
extern "C" __global__
void vit_scatter_head(
    const __nv_bfloat16* __restrict__ Oh,  // [seq, D] contiguous
    __nv_bfloat16* __restrict__ O,          // [seq, dst_stride], head-slot base
    unsigned int seq, unsigned int D, unsigned int dst_stride)
{
    unsigned int lin = blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= seq * D) return;
    unsigned int tok = lin / D, d = lin % D;
    O[(size_t)tok * dst_stride + d] = Oh[(size_t)tok * D + d];
}
```

**Why pre-transpose V (not a separate transpose pass):** `dense_gemm_bf16_pipelined` computes `C = A·Bᵀ` (dense_gemm_bf16.cu:5). GEMM2 wants `O = P·V`, `P[seq,seq]`, `V[seq,D]`. Passing `B = Vt[D,seq]` (row-major) gives `Bᵀ = V[seq,D]`, so `A·B�at = P·V`. The transpose is one strided store in kernel (A), free. Confirmed against dense_gemm_bf16.cu:5-10.

---

## 2. Struct fields + buffers + handles

### 2a. `vision_encoder.rs` — struct fields (add near line 54, after `k_attn`)

```rust
    k_attn: KernelHandle,           // vision_attention_rope (legacy / fallback)
    k_rope_deint: KernelHandle,     // vit_rope_deinterleave
    k_softmax: KernelHandle,        // vit_softmax_rows
    k_scatter_head: KernelHandle,   // vit_scatter_head
    k_gemm_f32: KernelHandle,       // dense_gemm_bf16_f32out (f32 scores)
```
(`k_gemm_pipelined` already exists at the GEMM2 reuse — no new field; it's already `crate::layers::try_kernel(gpu,"gemm","dense_gemm_bf16_pipelined")`.)

And the scratch buffer fields (add near the other `buf_*` near line 69):
```rust
    pub buf_qr: DevicePtr,        // [H, seq, D] rotated Q, head-contiguous
    pub buf_kr: DevicePtr,        // [H, seq, D] rotated K, head-contiguous
    pub buf_vt: DevicePtr,        // [H, D, seq] transposed V, head-contiguous
    pub buf_scores: DevicePtr,    // [seq, seq] f32, per-head reuse
    pub buf_probs: DevicePtr,     // [seq, seq] bf16, per-head reuse
    pub buf_o_stage: DevicePtr,   // [seq, D] bf16 GEMM2 staging, per-head reuse
```

### 2b. `init.rs` — buffer alloc (add after line ~49, next to `buf_rope_sin`)

```rust
        // ── ViT GEMM-attention scratch (head-contiguous Q/K/V + per-head scores).
        //    Q/K/V sized to p_max (matches existing convention; ~44 MB total).
        //    scores/probs sized to the per-IMAGE SDPA cap attn_max=1024 (the
        //    score matrix is seq², and seq is image-capped at 1024 — a p_max²
        //    matrix would be 164 MB/head and is never reached). Reused across
        //    the 16-head loop (each head computed then consumed before next).
        let attn_max = 1024usize;
        let qkv_head_elems = p_max * num_heads * head_dim;        // 6400*16*72
        let buf_qr = gpu.alloc(qkv_head_elems * 2)?;             // [H,seq,D] bf16
        let buf_kr = gpu.alloc(qkv_head_elems * 2)?;             // [H,seq,D] bf16
        let buf_vt = gpu.alloc(qkv_head_elems * 2)?;             // [H,D,seq] bf16
        let buf_scores  = gpu.alloc(attn_max * attn_max * 4)?;   // [seq,seq] f32
        let buf_probs   = gpu.alloc(attn_max * attn_max * 2)?;   // [seq,seq] bf16
        let buf_o_stage = gpu.alloc(p_max * head_dim * 2)?;      // [seq,D] bf16
```
Total new scratch ≈ 3×6400×72×16×2 + 1024²×4 + 1024²×2 + 6400×72×2 ≈ **44.2 + 4 + 2 + 0.9 = ~51 MB.** Fine on the 121 GB box.

### 2c. `init.rs` — handle lookup (add in `Ok(Self { .. })`, near line 91 next to `k_attn:`)

```rust
            k_attn: gpu.kernel("vision_encoder", "vision_attention_rope")?,
            k_rope_deint: gpu.kernel("vision_encoder", "vit_rope_deinterleave")?,
            k_softmax: gpu.kernel("vision_encoder", "vit_softmax_rows")?,
            k_scatter_head: gpu.kernel("vision_encoder", "vit_scatter_head")?,
            // f32-out GEMM for raw QKᵀ scores. Module "gemm" (same as
            // k_gemm_pipelined at init.rs:86 — the dense_gemm_bf16 stem is
            // remapped to module "gemm"). Hard-required (Review-1 fix 5b): it
            // definitely exists, and a null handle would silently launch nothing.
            k_gemm_f32: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
```
And add the 6 `buf_*` to the same `Ok(Self{..})` block.

---

## 3. `vit_block.rs` — the `vit_attention_gemm` helper + call-site swaps

Add this method to `impl VisionEncoder` in vit_block.rs:

```rust
    /// GEMM-based SDPA for one image's [seq, 3*H*D] QKV slice → O[seq, H*D].
    /// Replaces the warp-per-query vision_attention_rope. Per head:
    ///   GEMM1 QKᵀ → softmax → GEMM2 P·V → scatter. rope-deinterleave runs once
    ///   for all heads. `qkv/o/cos/sin` are per-image base pointers (caller
    ///   already offset them). seq ≤ 1024 (buf_scores/probs are [1024,1024]).
    /// IMPORTANT: all launches share one `stream`, so GEMM1(h)→softmax(h)→
    ///   GEMM2(h)→scatter(h) are ordered before head h+1 reuses buf_scores/
    ///   probs/o_stage. Do NOT parallelize heads onto multiple streams without
    ///   per-head score buffers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vit_attention_gemm(
        &self,
        gpu: &dyn GpuBackend,
        qkv: DevicePtr,   // [seq, 3*H*D]
        o: DevicePtr,     // [seq, H*D]
        cos: DevicePtr,   // [seq, D]
        sin: DevicePtr,   // [seq, D]
        seq: u32,
        stream: u64,
    ) -> Result<()> {
        debug_assert!(seq <= 1024, "ViT SDPA seq {seq} exceeds buf_scores cap (1024)");
        let h_n = self.num_heads as u32;
        let d   = self.head_dim as u32;       // 72
        let hd  = self.hidden_size as u32;    // H*D = 1152

        // (1) rope + deinterleave + V-transpose → buf_qr/buf_kr/buf_vt (all heads)
        KernelLaunch::new(gpu, self.k_rope_deint)
            .grid([div_ceil(seq * d, 256), h_n, 1])
            .block([256, 1, 1])
            .arg_ptr(qkv).arg_ptr(self.buf_qr).arg_ptr(self.buf_kr).arg_ptr(self.buf_vt)
            .arg_ptr(cos).arg_ptr(sin)
            .arg_u32(seq).arg_u32(h_n).arg_u32(d)
            .launch(stream)?;

        let qk_head = (seq * d) as usize;     // Qr/Kr head stride (seq*D elems)
        let v_head  = (d * seq) as usize;     // Vt head stride (D*seq elems)
        for head in 0..self.num_heads {
            let qr_h = self.buf_qr.offset(head * qk_head * 2);   // ×2 bytes (bf16)
            let kr_h = self.buf_kr.offset(head * qk_head * 2);
            let vt_h = self.buf_vt.offset(head * v_head * 2);
            let o_h  = o.offset(head * self.head_dim * 2);       // O[seq,H*D] slot

            // (2) GEMM1: S[seq,seq] = Qr_h[seq,D] @ Kr_h[seq,D]ᵀ  (f32 out, raw).
            //     FIX (Review-1 BUG1): dense_gemm_bf16_f32out is TILE=16,
            //     block (16,16), grid (ceil(N/16),ceil(M/16)). N=seq, M=seq.
            KernelLaunch::new(gpu, self.k_gemm_f32)
                .grid([div_ceil(seq, 16), div_ceil(seq, 16), 1])
                .block([16, 16, 1])
                .arg_ptr(qr_h).arg_ptr(kr_h).arg_ptr(self.buf_scores)
                .arg_u32(seq).arg_u32(seq).arg_u32(d)   // M, N, K=72
                .launch(stream)?;

            // (3) softmax rows (scale folded) → buf_probs[seq,seq] bf16
            KernelLaunch::new(gpu, self.k_softmax)
                .grid([seq, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_scores).arg_ptr(self.buf_probs)
                .arg_u32(seq).arg_u32(d)
                .launch(stream)?;

            // (4) GEMM2: O_stage[seq,D] = P[seq,seq] @ Vt_h[D,seq]ᵀ = P·V.
            //     pipelined GEMM grid is (ceil(N/128),ceil(M/128)): N=d=72→1
            //     tile, M=seq. (N-first — confirmed dense_gemm_bf16.cu + the
            //     trusted vit_gemm_bias launch at line 33.)
            KernelLaunch::new(gpu, self.k_gemm_pipelined)
                .grid([div_ceil(d, 128), div_ceil(seq, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_probs).arg_ptr(vt_h).arg_ptr(self.buf_o_stage)
                .arg_u32(seq).arg_u32(d).arg_u32(seq)   // M=seq, N=d, K=seq
                .launch(stream)?;

            // (5) scatter contiguous O_stage[seq,D] → interleaved o head slot
            KernelLaunch::new(gpu, self.k_scatter_head)
                .grid([div_ceil(seq * d, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_o_stage).arg_ptr(o_h)
                .arg_u32(seq).arg_u32(d).arg_u32(hd)    // dst row stride = H*D
                .launch(stream)?;
        }
        Ok(())
    }
```

### 3a. `vit_block` — replace the `k_attn` launch (lines 105–119) with:

```rust
        // 4. Attention. GEMM-based SDPA unless ATLAS_VISION_ATTN_LEGACY=1.
        if std::env::var("ATLAS_VISION_ATTN_LEGACY").is_ok() {
            KernelLaunch::new(gpu, self.k_attn)
                .grid([p32, self.num_heads as u32, 1])
                .block([32, 1, 1])
                .shared_mem(sm_bytes as u32)
                .arg_ptr(self.buf_wide).arg_ptr(self.buf_h1)
                .arg_ptr(self.buf_rope_cos).arg_ptr(self.buf_rope_sin)
                .arg_u32(p32).arg_u32(self.num_heads as u32).arg_u32(self.head_dim as u32)
                .launch(stream)?;
        } else {
            self.vit_attention_gemm(gpu, self.buf_wide, self.buf_h1,
                self.buf_rope_cos, self.buf_rope_sin, p32, stream)?;
        }
```
(`sm_bytes` at line 79 is now only used by the legacy branch — keep it; if clippy warns when the env is unset at compile time it won't, the branch references it at runtime.)

### 3b. `vit_block_batched` — replace the per-image `k_attn` launch (lines 235–248, inside the `for` loop) with:

```rust
            if std::env::var("ATLAS_VISION_ATTN_LEGACY").is_ok() {
                KernelLaunch::new(gpu, self.k_attn)
                    .grid([p32, self.num_heads as u32, 1])
                    .block([32, 1, 1])
                    .shared_mem(sm_bytes)
                    .arg_ptr(qkv).arg_ptr(o).arg_ptr(cos).arg_ptr(sin)
                    .arg_u32(p32).arg_u32(self.num_heads as u32).arg_u32(self.head_dim as u32)
                    .launch(stream)?;
            } else {
                self.vit_attention_gemm(gpu, qkv, o, cos, sin, p32, stream)?;
            }
```
The per-image offsets `qkv/o/cos/sin` (lines 231–234) and `sm_bytes` (line 230) are unchanged — `vit_attention_gemm` takes the same per-image base pointers. Because images are processed **serially on one stream** and `buf_scores/probs/o_stage` are reused, there is no cross-image aliasing (one image's loop fully consumes them before the next).

**`use` note:** vit_block.rs already imports `DevicePtr`, `GpuBackend`, `KernelLaunch`, `div_ceil` (lines 7-8). `std::env` is path-qualified inline. No new imports.

---

## 4. BUILD + VALIDATION

### Build (the CLAUDE.md remote block — edits must be rsync'd to gx10-9959 first)

```bash
rsync -az --delete /home/ms/atlas/kernels/ gx10-9959:atlas/kernels/
rsync -az /home/ms/atlas/crates/spark-model/src/layers/vision_encoder.rs \
      gx10-9959:atlas/crates/spark-model/src/layers/vision_encoder.rs
rsync -az /home/ms/atlas/crates/spark-model/src/layers/vision_encoder/enc_impl/ \
      gx10-9959:atlas/crates/spark-model/src/layers/vision_encoder/enc_impl/

ssh gx10-9959 'cd ~/atlas && source ~/.cargo/env
  export PATH=/usr/local/cuda/bin:$PATH
  export CUTLASS_HOME=$HOME/cutlass FLASHINFER_HOME=$HOME/flashinfer
  export RUSTFLAGS="-L/home/ms/nccl/build/lib -L/usr/local/cuda/lib64"
  export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4
  cargo build --release -p spark-server --bin spark --no-default-features --features cuda'
```
**Build log must read** `compiled N kernels for target 0 (gb10, holo-3.1-35b-a3b, nvfp4)` with N up by **3** (vit_rope_deinterleave, vit_softmax_rows, vit_scatter_head). If it instead says a qwen3-next target or "CUTLASS support was not built", a build env var was dropped — re-check the block.

### Validation (in order — each gates the next)

1. **Single-head GEMM1 corner (catches BUG1-class transpose/stride/grid bugs immediately).** With `ATLAS_DUMP_VIT`, dump `buf_scores` after GEMM1 for head 0 and host-recompute `Qr_h @ Kr_hᵀ` for a 4×4 corner (`S[i,j] = Σ_d Qr[0,i,d]·Kr[0,j,d]`). Must match to f32. A wrong block/grid (the 32-vs-16 bug) corrupts this even though it neither crashes nor fails compilation — this is the gating check.
2. **Block-by-block parity.** `ATLAS_DUMP_VIT` dumps each block's post-attention bf16 buffer. Run `ATLAS_VISION_ATTN_LEGACY=1` (oracle) vs default (new) and compare per block at **relative-error ≤ 1e-2** (the new path adds bf16 rounding on Qr/Kr and P — not bit-exact by design, this is the standard SDPA-as-GEMM tradeoff). First divergent block localizes the bug: rope → GEMM1 → softmax → GEMM2 → scatter. Keep `expf` (not `__expf`) in the softmax kernel during this pass.
3. **End-to-end Saturn.** Saturn image → must produce "a planet with rings, resembling Saturn" / "Saturn". This is the real correctness bar.
4. **Timing.** With `ATLAS_VISION_TIMING`, the VIT_SEC (patch + 27 blocks) should drop from **~425 ms toward <100 ms** (NOATTN floor was 27 ms; attention was ~398 ms of the 425). Per Review-1's tempered estimate expect the attention portion at **~10-25 ms**, so VIT_SEC ~40-55 ms. Cross-check on `/tmp/real_bench.py` image c1: per-image ViT encode 426 → ~40 ms, image TTFT ~700 → ~310 ms.

After parity + Saturn pass, optionally swap softmax `expf` → `__expf` and re-confirm Saturn for a small speed gain.

---

## 5. RISK / ROLLBACK

**Rollback:** the legacy `vision_attention_rope` stays registered (`k_attn`); `ATLAS_VISION_ATTN_LEGACY=1` restores it at both call sites with zero rebuild. A/B is a single env var. If the GEMM path is wrong or slow in production, set the flag and you are back to the known-good kernel instantly.

**What could still go wrong:**
- **GEMM2 N=72 underfills the 128-N-tile** (44% of every N-tile is zero-masked columns; only 1 N-tile × ceil(seq/128) M-tiles → 8 CTAs at seq=1024 on 48 SMs, ~6× underutilized). Correct, just leaves perf on the table. *Mitigation if timing disappoints:* the zero-copy strided-output variant (§deferred below) or a CUDA-graph capture of the 64-launch sequence.
- **Per-image launch count** = 1 + 16×4 = **65 launches/image** (~5 µs each ≈ 0.3 ms overhead). Negligible vs 398 ms but it is the floor the GEMM-math can't go below without graph capture.
- **bf16 rounding drift** on Qr/Kr/P could in principle flip a token on a borderline image. Saturn passing + rel-err ≤1e-2 across all 27 blocks is the guard; if a different image regresses, fall back via the flag and investigate that image's logits.
- **seq > 1024 overflow:** if any single image ever exceeds 1024 patches per image, `buf_scores`/`buf_probs` (sized [1024,1024]) OOB-write. The `debug_assert!` catches it in debug builds; in release it would silently corrupt. The patch-cap pipeline keeps per-image seq ≤1024, but if that cap is ever raised, bump `attn_max` in init.rs in lockstep.
- **The legacy-kernel smem discrepancy** (`[seq+2*D]` comment vs `(seq+D)*4` alloc) is inherited only by the fallback branch and is pre-existing/harmless (MAP A) — the new path uses no call-site dynamic smem.

**Deferred optimization (do NOT ship first):** add an `ldc` (output row-stride) arg to `dense_gemm_bf16_pipelined` so GEMM2 writes straight into the interleaved O head slot (stride H·D), eliminating `buf_o_stage` + `vit_scatter_head`. This touches a GEMM shared by `vit_gemm_bias` and likely MoE — not a 3-line risk-free edit. Ship the scatter version (isolated, no other call-site risk), prove correctness, then optimize.

**BROKEN items:** none blocking. One pre-existing cosmetic discrepancy noted (legacy smem comment) — no action.