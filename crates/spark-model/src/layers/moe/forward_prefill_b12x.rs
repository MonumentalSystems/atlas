// SPDX-License-Identifier: AGPL-3.0-only

//! b12x fused-MoE dispatch gate for prefill (`ATLAS_LAGUNA_MOE_B12X`). Hoisted here to
//! keep `forward_prefill.rs` under the ≤500 LoC cap. The gate is airtight: it returns
//! `Ok(false)` (⇒ grouped-CUTLASS fallback runs, byte-unchanged) unless EVERY condition
//! holds. When it fires, it writes the routed-expert result straight into `output`
//! (bf16); the shared-expert blend + EP all-reduce tail in `forward_prefill` still run.

use super::*;

impl MoeLayer {
    /// Try the b12x fused-MoE path. Returns `Ok(true)` iff b12x ran (caller then skips
    /// the grouped sort→GEMM→unpermute block); `Ok(false)` ⇒ grouped fallback.
    ///
    /// The all-experts-resident invariant is already encoded in `self.b12x.is_some()`
    /// (the load-time repack refuses EP / null-expert configs). The EP re-check here is
    /// belt-and-braces against a world_size>1 run reaching this layer.
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
        // env flag + lib loaded
        if !ops::b12x_flashinfer::available() {
            return Ok(false);
        }
        // load-time repack succeeded (⇒ no null experts ∧ ep≤1)
        let Some(b12x) = self.b12x.as_ref() else {
            return Ok(false);
        };
        // no EP all-reduce fan-in for the routed path
        if ctx.comm.is_some() && ctx.config.ep_world_size > 1 {
            return Ok(false);
        }
        // b12x has no pre-expert-norm hook (Gemma-4 style) — not applicable to Laguna
        if self.pre_expert_norm.is_some() {
            return Ok(false);
        }
        // shim workspace capacity
        if n > b12x.max_tokens {
            return Ok(false);
        }

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
            "ATLAS_LAGUNA_MOE_B12X: N={num_tokens} routed experts via one resident b12x launch"
        );
        Ok(true)
    }
}
