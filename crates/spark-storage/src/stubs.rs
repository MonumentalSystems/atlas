// SPDX-License-Identifier: AGPL-3.0-only

//! Non-cuda stubs for the high-speed-swap orchestrator surface.
//!
//! When the crate is built without the `cuda` feature (e.g. on Apple
//! Silicon under `--features metal`), `HighSpeedSwap` and its
//! `with_local` / `local_installed` / `install_local` helpers don't
//! exist as their real implementations — they're CUDA + GDS + io_uring
//! end-to-end. But spark-model code calls into this surface from
//! several decode/prefill paths and KV-cache-eviction helpers; making
//! those compile on macOS without rewriting every call site means
//! providing a stub that:
//!
//! 1. Has the same public type and method names so the closure bodies
//!    inside `with_local(|hss| { hss.dec_disk_ref(...); ... })`
//!    type-check cleanly.
//! 2. Reports "not installed" so callers gracefully skip the swap
//!    path. `local_installed()` returns `false`; `with_local()`
//!    returns `None` without ever invoking the closure.
//! 3. Bails with a clear error from `install_local` so a misconfigured
//!    metal-feature build that *tries* to install the orchestrator
//!    fails fast instead of silently no-op-ing.
//!
//! The method bodies are `unreachable!()` because `with_local` never
//! calls the closure on non-cuda — they exist purely to satisfy the
//! type checker.

use crate::config::HighSpeedSwapConfig;
use crate::model_dims::ModelDims;

pub struct HighSpeedSwap;

/// Mirror of `high_speed_swap::AttendSeqReq` (Phase 5 batched attend) so
/// non-cuda builds of the spark-model call sites type-check.
pub struct AttendSeqReq<'a> {
    pub seq_slot: usize,
    pub seq_block_ids: &'a [u32],
    pub q_dev: u64,
    pub output_dev: u64,
}

#[allow(unused_variables)]
impl HighSpeedSwap {
    pub fn alloc_disk_block_id(&mut self) -> Option<u32> {
        None
    }

    pub fn inc_disk_ref(&mut self, id: u32) {}

    pub fn dec_disk_ref(&mut self, id: u32) -> u32 {
        0
    }

    pub fn offload_block_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> anyhow::Result<()> {
        unreachable!("HighSpeedSwap stub: cuda feature is off")
    }

    pub fn offload_block_no_predict_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> anyhow::Result<()> {
        unreachable!()
    }

    pub fn attend_layer_on_stream(
        &mut self,
        seq_slot: usize,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
    ) -> anyhow::Result<()> {
        unreachable!()
    }

    pub fn attend_layer_batch_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        seqs: &[AttendSeqReq<'_>],
    ) -> anyhow::Result<()> {
        unreachable!()
    }

    pub fn attend_layer_on_stream_with_q_pos(
        &mut self,
        seq_slot: usize,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
        last_block_valid_slots: i32,
    ) -> anyhow::Result<()> {
        unreachable!()
    }

    // Phase-3 prefetch surface, referenced by decode_a2.rs's `with_local`
    // closure whose body is type-checked against this stub on non-cuda builds.
    pub fn prefetch_layer(&mut self, _layer: u32, _seq_block_ids: &[u32]) -> anyhow::Result<()> {
        unreachable!()
    }

    // #11: WAR-fence record, ditto.
    pub fn record_kv_read_event(&self, _stream: u64) -> anyhow::Result<()> {
        unreachable!()
    }
}

pub fn local_installed() -> bool {
    false
}

pub fn with_local<R>(
    _f: impl FnOnce(&mut HighSpeedSwap) -> anyhow::Result<R>,
) -> Option<anyhow::Result<R>> {
    None
}

pub fn install_local(
    _stream: u64,
    _cfg: HighSpeedSwapConfig,
    _model: ModelDims,
) -> anyhow::Result<()> {
    anyhow::bail!("HighSpeedSwap unavailable: spark-storage built without cuda feature")
}
