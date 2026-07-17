// SPDX-License-Identifier: AGPL-3.0-only

//! b12x fused-MoE dispatch gate for prefill (`ATLAS_MOE_B12X`). Hoisted here to
//! keep `forward_prefill.rs` under the ≤500 LoC cap. The gate is airtight: it returns
//! `Ok(false)` (⇒ grouped-CUTLASS fallback runs, byte-unchanged) unless EVERY condition
//! holds. When it fires, it writes the routed-expert result straight into `output`
//! (bf16); the shared-expert blend + EP all-reduce tail in `forward_prefill` still run.

use super::*;

impl MoeLayer {
    /// Try b12x for a concurrent decode batch. Eligibility is resolved before any
    /// GPU work so an ineligible route can safely continue through token-major MoE.
    pub fn try_b12x_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if num_tokens < 4
            || std::env::var("ATLAS_MOE_B12X_DECODE").ok().as_deref() != Some("1")
            || self.lora.is_some()
            || !self.b12x_eligible(num_tokens as u32, ctx)
        {
            return Ok(false);
        }
        self.forward_prefill(input, num_tokens, ctx, stream)?;
        Ok(true)
    }

    fn b12x_eligible(&self, n: u32, ctx: &ForwardContext) -> bool {
        ops::b12x_flashinfer::available()
            && self.b12x.as_ref().is_some_and(|w| n <= w.max_tokens)
            && ctx.comm.is_none_or(|c| c.world_size() <= 1)
            && self.pre_expert_norm.is_none()
            && self.lora.is_none()
    }

    /// Try the b12x fused-MoE path. Returns `Ok(true)` iff b12x ran (caller then skips
    /// the grouped sort→GEMM→unpermute block); `Ok(false)` ⇒ grouped fallback.
    ///
    /// The all-experts-resident invariant is already encoded in `self.b12x.is_some()`
    /// (the load-time repack refuses EP / null-expert configs). Streaming-experts does
    /// not exist on this branch, so there is no streamer to re-check.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_b12x_prefill(
        &self,
        input: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        output: DevicePtr,
        n: u32,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !self.b12x_eligible(n, ctx) {
            return Ok(false);
        }
        let Some(b12x) = self.b12x.as_ref() else {
            return Ok(false);
        };

        ops::b12x_flashinfer::b12x_moe_prefill(
            ctx.gpu,
            input,
            indices_dev,
            weights_dev,
            output,
            b12x,
            n,
            stream,
        )?;
        tracing::debug!(
            "ATLAS_MOE_B12X: N={num_tokens} routed experts via one resident b12x launch"
        );
        Ok(true)
    }
}
