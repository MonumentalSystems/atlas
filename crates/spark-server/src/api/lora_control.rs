// SPDX-License-Identifier: AGPL-3.0-only

//! LoRA adapter rotation control plane: `POST /v1/lora/active`.
//!
//! Selects the globally-active resident adapter at runtime (eager-on-rotate).
//! The request is forwarded to the scheduler over the rotation channel and
//! applied at a QUIESCENT point (no in-flight decode), so the re-point never
//! races a live delta read or a CUDA-graph replay. Batch-1 honest: rotation
//! changes the adapter applied to ALL subsequent requests (per-request adapter
//! routing is a future increment).

use std::sync::Arc;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use crate::AppState;
use crate::api::compact::openai_error_response;

#[derive(Deserialize)]
pub struct SetActiveLoraRequest {
    /// The resident adapter NAME to activate (as advertised by `/v1/models`).
    pub adapter: String,
}

#[derive(Serialize)]
struct SetActiveLoraResponse {
    object: &'static str,
    active: String,
}

/// POST /v1/lora/active  `{"adapter": "NAME"}`
pub async fn set_active_lora(
    State(state): State<Arc<AppState>>,
    body: Result<Json<SetActiveLoraRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return openai_error_response(StatusCode::BAD_REQUEST, format!("invalid body: {e}"));
        }
    };

    let Some(ref tx) = state.rotation_tx else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "no LoRA adapter is loaded (start with --lora-adapter NAME=PATH)".to_string(),
        );
    };

    if !state.adapter_names.iter().any(|n| n == &req.adapter) {
        return openai_error_response(
            StatusCode::NOT_FOUND,
            format!(
                "adapter '{}' is not resident (resident: [{}])",
                req.adapter,
                state.adapter_names.join(", ")
            ),
        );
    }

    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
    if tx
        .send((
            crate::scheduler::LoraCommand::Rotate(req.adapter.clone()),
            ack_tx,
        ))
        .await
        .is_err()
    {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler rotation channel closed".to_string(),
        );
    }
    match ack_rx.await {
        Ok(Ok(_)) => {
            // Optimistic status mirror (the scheduler's model owns the truth).
            if let Ok(mut a) = state.active_adapter.lock() {
                *a = Some(req.adapter.clone());
            }
            Json(SetActiveLoraResponse {
                object: "lora.active",
                active: req.adapter,
            })
            .into_response()
        }
        Ok(Err(reason)) => openai_error_response(StatusCode::BAD_REQUEST, reason),
        Err(_) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler dropped the rotation ack (shutting down?)".to_string(),
        ),
    }
}

#[derive(Deserialize)]
pub struct LoadLoraRequest {
    /// Name to stamp on the loaded adapter (its `/v1/models` id after the swap).
    pub name: String,
    /// Filesystem path to the PEFT adapter dir (contains adapter_config.json +
    /// adapter_model.safetensors).
    pub path: String,
    /// Pool slot to load into (default 0 — the single slot under --max-loras 1).
    #[serde(default)]
    pub slot: usize,
}

#[derive(Serialize)]
struct LoadLoraResponse {
    object: &'static str,
    loaded: String,
    slot: usize,
}

/// POST /v1/lora/load  `{"name": "vega", "path": "/dir", "slot": 0}`
///
/// Dynamically loads a DIFFERENT adapter from disk into a pool slot at runtime
/// — the pool-size-1 demonstration of per-request weight change: with
/// `--max-loras 1` only one adapter is resident, and this swaps the single
/// slot's contents on demand (needs `ATLAS_LORA_ROTATE=1` so decode is eager).
pub async fn load_lora_into_slot(
    State(state): State<Arc<AppState>>,
    body: Result<Json<LoadLoraRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return openai_error_response(StatusCode::BAD_REQUEST, format!("invalid body: {e}"));
        }
    };

    let Some(ref tx) = state.rotation_tx else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "no LoRA adapter pool is loaded (start with --lora-adapter NAME=PATH)".to_string(),
        );
    };

    let dir = std::path::PathBuf::from(&req.path);
    if !dir.join("adapter_config.json").exists() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("no adapter_config.json under path '{}'", req.path),
        );
    }

    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
    let cmd = crate::scheduler::LoraCommand::LoadIntoSlot {
        name: req.name.clone(),
        dir,
        slot: req.slot,
    };
    if tx.send((cmd, ack_tx)).await.is_err() {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler rotation channel closed".to_string(),
        );
    }
    match ack_rx.await {
        Ok(Ok(_)) => {
            // The swapped slot becomes the served adapter; mirror it.
            if let Ok(mut a) = state.active_adapter.lock() {
                *a = Some(req.name.clone());
            }
            Json(LoadLoraResponse {
                object: "lora.loaded",
                loaded: req.name,
                slot: req.slot,
            })
            .into_response()
        }
        Ok(Err(reason)) => openai_error_response(StatusCode::BAD_REQUEST, reason),
        Err(_) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler dropped the load ack (shutting down?)".to_string(),
        ),
    }
}
