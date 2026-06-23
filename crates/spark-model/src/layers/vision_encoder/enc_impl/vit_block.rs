// SPDX-License-Identifier: AGPL-3.0-only

//! Single ViT block (norm → QKV → RoPE attention → proj → +residual →
//! norm → fc1 → GELU → fc2 → +residual).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{ViTBlock, VisionEncoder};

impl VisionEncoder {
    /// ViT GEMM with bias: C[m,n] = A[m,k] @ B[n,k]^T + bias[n] (BF16).
    /// Prefers the tensor-core `dense_gemm_bf16_pipelined` (~40× the scalar
    /// `vision_gemm_bias` on the ViT's large-M shapes) + a fused bias-add; falls
    /// back to the scalar fused kernel if either handle is unavailable. The ViT
    /// GEMMs dominate image prefill (~5s/image on the scalar path).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vit_gemm_bias(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        b: DevicePtr,
        bias: DevicePtr,
        c: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if self.k_gemm_pipelined.0 != 0 && self.k_add_bias.0 != 0 {
            KernelLaunch::new(gpu, self.k_gemm_pipelined)
                .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(a)
                .arg_ptr(b)
                .arg_ptr(c)
                .arg_u32(m)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)?;
            KernelLaunch::new(gpu, self.k_add_bias)
                .grid([div_ceil(m * n, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(c)
                .arg_ptr(bias)
                .arg_u32(m)
                .arg_u32(n)
                .launch(stream)
        } else {
            KernelLaunch::new(gpu, self.k_gemm)
                .grid([div_ceil(n, 32), div_ceil(m, 32), 1])
                .block([32, 32, 1])
                .arg_ptr(a)
                .arg_ptr(b)
                .arg_ptr(bias)
                .arg_ptr(c)
                .arg_u32(m)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
        }
    }

    /// Run one ViT block (in-place on buf_h1; buf_h2 and buf_wide are scratch).
    pub(super) fn vit_block(
        &self,
        blk: &ViTBlock,
        p: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let p32 = p as u32;
        let qkv_n = (3 * self.num_heads * self.head_dim) as u32; // 3456
        let inter = self.intermediate_size as u32; // 4304
        let n_h = p * self.hidden_size;
        // Attention-kernel shared memory: scores[p] + q_rope[head_dim].
        let sm_bytes = (p + self.head_dim) * std::mem::size_of::<f32>();

        // --- Attention sub-block ---
        // 1. save residual
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        // 2. norm1 in-place
        KernelLaunch::new(gpu, self.k_norm)
            .grid([p32, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(blk.norm1_w)
            .arg_ptr(blk.norm1_b)
            .arg_u32(p32)
            .arg_u32(h)
            .arg_f32(1e-6)
            .launch(stream)?;
        // 3. QKV GEMM → buf_wide
        self.vit_gemm_bias(gpu, self.buf_h1, blk.qkv_w, blk.qkv_b, self.buf_wide, p32, qkv_n, h, stream)?;
        // 4. Attention with 2D rotary pos emb applied inline to Q/K
        //    (blockDim=32 for correct warp reduction; rope buffers already
        //    uploaded once per image by `build_rope_cossin`).
        KernelLaunch::new(gpu, self.k_attn)
            .grid([p32, self.num_heads as u32, 1])
            .block([32, 1, 1])
            .shared_mem(sm_bytes as u32)
            .arg_ptr(self.buf_wide)
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_rope_cos)
            .arg_ptr(self.buf_rope_sin)
            .arg_u32(p32)
            .arg_u32(self.num_heads as u32)
            .arg_u32(self.head_dim as u32)
            .launch(stream)?;
        // 5. proj GEMM → buf_wide (reuse)
        self.vit_gemm_bias(gpu, self.buf_h1, blk.proj_w, blk.proj_b, self.buf_wide, p32, h, h, stream)?;
        // 6. residual add: buf_wide += buf_h2
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_wide)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        // 7. copy post-attn back to buf_h1
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_wide)
            .arg_ptr(self.buf_h1)
            .arg_u32(n_h as u32)
            .launch(stream)?;

        // --- FFN sub-block ---
        // 8. save residual
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        // 9. norm2 in-place
        KernelLaunch::new(gpu, self.k_norm)
            .grid([p32, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(blk.norm2_w)
            .arg_ptr(blk.norm2_b)
            .arg_u32(p32)
            .arg_u32(h)
            .arg_f32(1e-6)
            .launch(stream)?;
        // 10. fc1 GEMM → buf_wide
        self.vit_gemm_bias(gpu, self.buf_h1, blk.fc1_w, blk.fc1_b, self.buf_wide, p32, inter, h, stream)?;
        // 11. GELU in-place on buf_wide
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(p32 * inter, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_wide)
            .arg_u32(p32 * inter)
            .launch(stream)?;
        // 12. fc2 GEMM → buf_h1 (overwrites normed hidden, OK — normed already consumed by fc1)
        self.vit_gemm_bias(gpu, self.buf_wide, blk.fc2_w, blk.fc2_b, self.buf_h1, p32, h, inter, stream)?;
        // 13. residual add: buf_h1 += buf_h2
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)
    }
}
