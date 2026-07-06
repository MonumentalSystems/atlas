// SPDX-License-Identifier: AGPL-3.0-only

//! Pointer/size accessors for [`BufferArena`], split out of `buffers.rs`
//! to keep it under the 500-LoC cap. Each getter returns a `DevicePtr` into
//! the arena, a byte size, or the `BufferSizes`. As a child module of `buffers`
//! this can read `BufferArena`'s private fields; the second `impl` block is
//! merged into the type's inherent API.

use super::{BufferArena, BufferSizes};
use crate::gpu::DevicePtr;

impl BufferArena {
    pub fn hidden_states(&self) -> DevicePtr {
        self.hidden_states
    }
    pub fn residual(&self) -> DevicePtr {
        self.residual
    }
    pub fn norm_output(&self) -> DevicePtr {
        self.norm_output
    }
    pub fn qkv_output(&self) -> DevicePtr {
        self.qkv_output
    }
    pub fn attn_output(&self) -> DevicePtr {
        self.attn_output
    }
    pub fn gate_logits(&self) -> DevicePtr {
        self.gate_logits
    }
    pub fn gate_logits_f32(&self) -> DevicePtr {
        self.gate_logits_f32
    }
    pub fn moe_router_in_f32(&self) -> DevicePtr {
        self.moe_router_in_f32
    }
    pub fn moe_output(&self) -> DevicePtr {
        self.moe_output
    }
    pub fn logits(&self) -> DevicePtr {
        self.logits
    }
    pub fn ssm_qkvz(&self) -> DevicePtr {
        self.ssm_qkvz
    }
    pub fn ssm_ba(&self) -> DevicePtr {
        self.ssm_ba
    }
    /// Sequential [Q|K|V|Z] after deinterleaving.
    pub fn ssm_deinterleaved(&self) -> DevicePtr {
        self.ssm_deinterleaved
    }
    /// FP32 [gate, beta] for GDN (num_v_heads * 2 floats).
    pub fn ssm_gates(&self) -> DevicePtr {
        self.ssm_gates
    }
    /// FP32 conv1d output for SSM recurrent path (prevents BF16 precision drift).
    pub fn ssm_conv_out_f32(&self) -> DevicePtr {
        self.ssm_conv_out_f32
    }
    /// Scratch buffer for MoE routing + kernel metadata uploads.
    pub fn scratch(&self) -> DevicePtr {
        self.scratch
    }
    /// Token IDs `[M]` u32 — stable across the layer loop (DeepSeek-V4 hash-MoE
    /// reads `tid2eid[token_id]`). Upload the pass's token IDs here before the
    /// layer loop; under CUDA-graph decode upload before each replay.
    pub fn token_ids(&self) -> DevicePtr {
        self.token_ids
    }
    /// Allocated byte size of the scratch buffer (#110: bounds-check
    /// batched metadata-staging uploads against this).
    pub fn scratch_bytes(&self) -> usize {
        self.sizes.scratch
    }
    /// Batched expert gate projection output.
    pub fn expert_gate_out(&self) -> DevicePtr {
        self.expert_gate_out
    }
    /// Batched expert up projection output.
    pub fn expert_up_out(&self) -> DevicePtr {
        self.expert_up_out
    }
    /// Batched expert down projection output.
    pub fn expert_down_out(&self) -> DevicePtr {
        self.expert_down_out
    }
    /// Split-K decode attention workspace (F32 partials).
    /// GDN FLA chunked-prefill scratch base (W|U|S|uc sub-divided by the caller).
    /// `DevicePtr::NULL` unless this is a 128-dim-linear-head GDN model.
    pub fn gdn_fla_scratch(&self) -> DevicePtr {
        self.gdn_fla_scratch
    }
    /// Shared dense-FFN q8_1 activation scratch (Q4_K MMQ gate/up). NULL for MoE.
    pub fn ffn_act_q8(&self) -> DevicePtr {
        self.ffn_act_q8
    }
    /// Shared dense-FFN int8/NVFP4 activation scratch (a_i8 / packed). NULL for MoE.
    pub fn ffn_act_a(&self) -> DevicePtr {
        self.ffn_act_a
    }
    /// Shared dense-FFN int8/NVFP4 activation-scale scratch. NULL for MoE.
    pub fn ffn_act_scale(&self) -> DevicePtr {
        self.ffn_act_scale
    }
    /// Persistent FP8 block-scaled activation scratch for prefill projections.
    /// Replaces a per-projection alloc/sync/free in the W8A8+FP32-epilogue path.
    pub fn fp8_act(&self) -> DevicePtr {
        self.fp8_act
    }
    /// Allocated byte size of `fp8_act` (debug bounds-check at call sites).
    pub fn fp8_act_bytes(&self) -> usize {
        self.sizes.fp8_act
    }
    /// Persistent per-128-block FP32 scales paired with `fp8_act`.
    pub fn fp8_act_scale(&self) -> DevicePtr {
        self.fp8_act_scale
    }
    /// LoRA shrink scratch `xa = x@Aᵀ` [M, adapter_max_rank] BF16.
    /// `DevicePtr::NULL` when no adapter is configured.
    pub fn lora_xa(&self) -> DevicePtr {
        self.lora_xa
    }
    /// Allocated byte size of `lora_xa` (0 when no adapter).
    pub fn lora_xa_bytes(&self) -> usize {
        self.sizes.lora_xa
    }
    /// LoRA expand scratch `delta = xa@Bᵀ` [M, max(hidden, intermediate)]
    /// BF16. `DevicePtr::NULL` when no adapter is configured.
    pub fn lora_delta(&self) -> DevicePtr {
        self.lora_delta
    }
    /// Allocated byte size of `lora_delta` (0 when no adapter).
    pub fn lora_delta_bytes(&self) -> usize {
        self.sizes.lora_delta
    }
    /// LoRA hidden-activation scratch [M, intermediate_size] BF16 for the
    /// runtime FFN delta path. `DevicePtr::NULL` when no adapter.
    pub fn lora_hact(&self) -> DevicePtr {
        self.lora_hact
    }
    /// Allocated byte size of `lora_hact` (0 when no adapter).
    pub fn lora_hact_bytes(&self) -> usize {
        self.sizes.lora_hact
    }
    pub fn splitk_workspace(&self) -> DevicePtr {
        self.splitk_workspace
    }
    /// Grouped O-projection latent [M, o_groups*o_lora_rank] BF16 (V4-Flash).
    pub fn o_latent(&self) -> DevicePtr {
        self.o_latent
    }
    /// All-ones BF16 vector (max_dim) — weight for unweighted RMSNorm (q_b_norm).
    pub fn norm_unit_w(&self) -> DevicePtr {
        self.norm_unit_w
    }
    /// HC residual streams [M, hc_mult, hidden] BF16 (DeepSeek-V4 mHC).
    pub fn hc_streams(&self) -> DevicePtr {
        self.hc_streams
    }
    /// HC `post` mixing weights [M, hc_mult] F32.
    pub fn hc_post(&self) -> DevicePtr {
        self.hc_post
    }
    /// HC `comb` Sinkhorn matrix [M, hc_mult, hc_mult] F32.
    pub fn hc_comb(&self) -> DevicePtr {
        self.hc_comb
    }
    pub fn max_batch_tokens(&self) -> usize {
        self.max_batch_tokens
    }
    pub fn sizes(&self) -> &BufferSizes {
        &self.sizes
    }
}
