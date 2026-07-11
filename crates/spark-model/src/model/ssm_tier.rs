// SPDX-License-Identifier: AGPL-3.0-only

//! Model-safety contract for the SSM snapshot tiers: the stable, model-agnostic
//! durable-key [`ModelFingerprint`] and the [`ensure_ssm_tier_capability`]
//! gate. Both are pure `ModelConfig`-derived leaves — no HBM, RDMA, or paging
//! machinery — so a caching tier can key entries so they never collide across
//! models and can refuse a model that has no recurrent state. The paging/spill
//! machinery that consumes these lands separately.

mod capability;
mod fingerprint;

pub(crate) use capability::ensure_ssm_tier_capability;
pub(crate) use fingerprint::ModelFingerprint;
