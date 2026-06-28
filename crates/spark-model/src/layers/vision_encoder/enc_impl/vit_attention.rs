// SPDX-License-Identifier: AGPL-3.0-only

//! GEMM-based ViT self-attention (SDPA) for one image, split out of
//! `vit_block.rs` to keep each enc_impl sibling ≤500 LoC. The per-head
//! GEMM1→softmax→GEMM2→scatter launch order and the shared
//! buf_scores/buf_probs/buf_o_stage reuse are correctness-critical and read
//! top-to-bottom (see the doc comment on `vit_attention_gemm`).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::VisionEncoder;

impl VisionEncoder {
    /// GEMM-based SDPA for one image's [seq, 3*H*D] QKV slice → O[seq, H*D].
    /// Fast replacement for the warp-per-query `vision_attention_rope`. Once per
    /// call: rope + deinterleave + V-transpose (all heads). Then per head:
    ///   GEMM1 raw QKᵀ (f32 out) → row softmax (scale folded) → GEMM2 P·V →
    ///   scatter into the interleaved O head slot.
    /// `qkv/o/cos/sin` are per-image base pointers (caller already offset them).
    /// All launches share `stream`, so each head's GEMM1→softmax→GEMM2→scatter is
    /// ordered before head h+1 reuses buf_scores/buf_probs/buf_o_stage — do NOT
    /// split heads across streams without per-head score buffers. seq ≤ 1024
    /// (buf_scores/probs are [1024,1024]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vit_attention_gemm(
        &self,
        gpu: &dyn GpuBackend,
        qkv: DevicePtr, // [seq, 3*H*D]
        o: DevicePtr,   // [seq, H*D]
        cos: DevicePtr, // [seq, D]
        sin: DevicePtr, // [seq, D]
        seq: u32,
        stream: u64,
    ) -> Result<()> {
        debug_assert!(
            seq <= 1024,
            "ViT SDPA seq {seq} exceeds buf_scores cap (1024)"
        );
        let h_n = self.num_heads as u32;
        let d = self.head_dim as u32; // 72
        let hd = self.hidden_size as u32; // H*D = 1152

        // (1) rope + deinterleave + V-transpose → buf_qr/buf_kr/buf_vt (all heads)
        KernelLaunch::new(gpu, self.k_rope_deint)
            .grid([div_ceil(seq * d, 256), h_n, 1])
            .block([256, 1, 1])
            .arg_ptr(qkv)
            .arg_ptr(self.buf_qr)
            .arg_ptr(self.buf_kr)
            .arg_ptr(self.buf_vt)
            .arg_ptr(cos)
            .arg_ptr(sin)
            .arg_u32(seq)
            .arg_u32(h_n)
            .arg_u32(d)
            .launch(stream)?;

        let qk_head = (seq * d) as usize; // Qr/Kr head stride (seq*D elems)
        let v_head = (d * seq) as usize; // Vt head stride (D*seq elems)
        for head in 0..self.num_heads {
            let qr_h = self.buf_qr.offset(head * qk_head * 2); // ×2 bytes (bf16)
            let kr_h = self.buf_kr.offset(head * qk_head * 2);
            let vt_h = self.buf_vt.offset(head * v_head * 2);
            let o_h = o.offset(head * self.head_dim * 2); // O[seq,H*D] head slot

            // (2) GEMM1: S[seq,seq] = Qr_h[seq,D] @ Kr_h[seq,D]ᵀ (raw, f32 out).
            //     f32out is TILE=16: block (16,16), grid (ceil(N/16),ceil(M/16)).
            KernelLaunch::new(gpu, self.k_gemm_f32)
                .grid([div_ceil(seq, 16), div_ceil(seq, 16), 1])
                .block([16, 16, 1])
                .arg_ptr(qr_h)
                .arg_ptr(kr_h)
                .arg_ptr(self.buf_scores)
                .arg_u32(seq) // M
                .arg_u32(seq) // N
                .arg_u32(d) // K = 72
                .launch(stream)?;

            // (3) row softmax (scale folded) → buf_probs[seq,seq] bf16
            KernelLaunch::new(gpu, self.k_softmax)
                .grid([seq, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_scores)
                .arg_ptr(self.buf_probs)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;

            // (4) GEMM2: O_stage[seq,D] = P[seq,seq] @ Vt_h[D,seq]ᵀ = P·V.
            //     pipelined grid is (ceil(N/128),ceil(M/128)): N=d→1 tile, M=seq.
            KernelLaunch::new(gpu, self.k_gemm_pipelined)
                .grid([div_ceil(d, 128), div_ceil(seq, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_probs)
                .arg_ptr(vt_h)
                .arg_ptr(self.buf_o_stage)
                .arg_u32(seq) // M
                .arg_u32(d) // N
                .arg_u32(seq) // K
                .launch(stream)?;

            // (5) scatter contiguous O_stage[seq,D] → interleaved o head slot
            KernelLaunch::new(gpu, self.k_scatter_head)
                .grid([div_ceil(seq * d, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_o_stage)
                .arg_ptr(o_h)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(hd) // dst row stride = H*D
                .launch(stream)?;
        }
        Ok(())
    }
}
