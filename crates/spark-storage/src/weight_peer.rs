// SPDX-License-Identifier: AGPL-3.0-only
//
// Weight-serving peer + wire protocol — the RDMA weight-staging tier.
//
// Generalizes `expert_peer` from (layer, expert) expert records to ALL of a
// model's safetensors tensors, for FAST MODEL SWAPS. A peer holds one or more
// staged models' shard files mmap'd + `ibv_reg_mr`'d REMOTE_READ in its RAM; a
// client (`weight_tier_rdma::RdmaWeightLoader`) requests a model by id/path,
// reads the peer's MANIFEST, then one-sided RDMA-READs each tensor's bytes
// straight out of the shard MRs (~24 GB/s dual-rail) instead of the ~2 GB/s USB
// SSD. Weights are READ-ONLY → one-sided READ, no coherence — the exact
// expert-tier pattern.
//
// It's a CACHE: the FIRST stage of a model into the blade faults its pages in
// from disk (slow); every later swap reads them out of the peer's warm RAM
// (fast). Pre-warm the rotation set by connecting once.
//
// Wire protocol (little-endian), connection-oriented, server responds to the
// client's model choice first:
//   1. Client sends the model request: `[u32 len][len bytes of model id/path]`.
//   2. Server stages that model (mmap + parse headers, cached across
//      connections) and sends the manifest: `[u32 len][len bytes of JSON]`
//      (`WeightManifest` — per-tensor {name,dtype,shape,offset,len,shard}).
//   3. Client sends `[u8 transport_mode]` (only `MODE_VERBS` is served).
//   4. Verbs handshake (reused verbatim from `expert_peer`): `[u8 n_rails]`,
//      then per rail a `VerbsServerParams` whose `layers` vector carries this
//      model's per-SHARD `(mr_base, rkey)` (shards play the role experts' layer
//      files do). The client replies with its QP params, the server connects
//      and idles — the client pulls all tensor bytes one-sided.
//
// Per-tensor geometry rides the JSON manifest (like `ExpertIndex`); only the
// per-shard `(base, rkey)` rides `VerbsServerParams` (like the expert peer's
// per-layer `(base, rkey)`) — keeping shard counts well under the 4096/8 wire
// caps and the 512-MR-per-QP shim limit (real models have tens of shards).
//
// Module layout (SDD split — the client-facing half is un-gated + verbs-free so
// LoRA (chunk 5) lifts it cleanly):
//   * `manifest` — the manifest types + address/rail math (un-gated).
//   * `wire`     — the length-prefixed model-request / manifest codec (un-gated).
//   * `serve`    — the `atlas-weight-peer` daemon (unix; holds the reg_mr true).
//   * `shard`    — shard resolution + safetensors parse + the warm RO mmap (unix).

mod manifest;
mod wire;
#[cfg(unix)]
mod serve;
#[cfg(unix)]
mod shard;

pub use manifest::{WeightManifest, WeightTensorRecord, rail_for_tensor, tensor_remote_addr};
pub use wire::{
    MODEL_REQUEST_MAX, read_model_request, read_weight_manifest, write_model_request,
    write_weight_manifest,
};
#[cfg(unix)]
pub use serve::{WeightPeerConfig, serve};
