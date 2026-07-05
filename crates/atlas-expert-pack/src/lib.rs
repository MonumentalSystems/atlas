// SPDX-License-Identifier: AGPL-3.0-only
//
// atlas-expert-pack: offline converter from an NVFP4 MoE checkpoint to the
// resident-layout expert store consumed by the (Phase 2) expert streamer.
//
// See `docs/streaming-experts/` for the plan. This crate is the Phase 1
// "expert-file builder": it never touches a GPU, so it runs on any host and can
// pre-stage the ~200 GB 397B store on the box that will later stream it.

pub mod build;
pub mod checkpoint;
pub mod safetensors_min;
pub mod transpose;
