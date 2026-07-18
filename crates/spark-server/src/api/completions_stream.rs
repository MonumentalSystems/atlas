// SPDX-License-Identifier: AGPL-3.0-only

//! SSE streaming path for legacy `/v1/completions`.
//!
//! Extracted from `completions.rs` (2026-07-18) to keep both files under the
//! 500-LoC cap and to give the opt-in power-attribution wiring a home. The
//! blocking path stays in `completions_exec.rs`; the handler + request
//! validation stay in `completions.rs`.

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{CompletionChunk, CompletionRequest, Usage};

use super::completions_exec::CompletionParams;
use super::inference_types::{InferenceRequest, StreamEvent};

/// SSE streaming path for legacy completions (single prompt, n=1, handler-
/// guarded). Echo: prompt text (+ logprobs when set) is the FIRST chunk; with
/// `stream_options.include_usage` a `choices: []` usage chunk precedes `[DONE]`.
pub(super) async fn completions_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    req: CompletionRequest,
    p: CompletionParams,
) -> Result<Response, (StatusCode, String)> {
    // Opt-in power window (span + generated-token baseline). Extracted before
    // `p` is consumed building the request; finished on the terminal `Done`
    // event. Dropping the stream early drops the span, balancing in-flight.
    #[cfg(feature = "power_attribution")]
    let mut power = p.power_ctx;

    // Match chat_stream/mod.rs sizing; see comment there.
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let prompt_len = prompt_tokens.len();
    let echo = req.echo;
    let logprobs_k = p.logprobs_k;
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    // Echo needs the prompt tokens after the request consumes them.
    let echo_prompt = if echo {
        Some(prompt_tokens.clone())
    } else {
        None
    };
    // Echo WITHOUT logprobs has no PromptLogprobs event to hook — the
    // prompt text chunk is prepended client-side; decode it now, before
    // the request takes ownership of the tokens.
    let echo_only_text = if echo && logprobs_k.is_none() {
        Some(state.tokenizer.decode(&prompt_tokens).unwrap_or_default())
    } else {
        None
    };

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let request = InferenceRequest::Streaming {
        prompt_tokens: std::sync::Arc::new(prompt_tokens),
        session_hash,
        adapter_slot: p.adapter_slot,
        src_lang_id: p.src_lang_id,
        tgt_lang_id: p.tgt_lang_id,
        num_beams: p.num_beams,
        length_penalty: p.length_penalty,
        early_stopping: p.early_stopping,
        image_pixels: Vec::new(),
        max_tokens: req.max_tokens,
        min_tokens: 0,
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        top_n_sigma: p.top_n_sigma,
        min_p: p.min_p,
        repetition_penalty: p.repetition_penalty,
        presence_penalty: p.presence_penalty,
        frequency_penalty: p.frequency_penalty,
        // Legacy /v1/completions path doesn't have tool semantics.
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias: p.logit_bias,
        stop_tokens: p.stop_tokens,
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: p.repetition_detection,
        require_tool_call: false,
        // Completions API defines no tools — multi-tool-call continuation off.
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec: None,
        seed: req.seed,
        top_logprobs: logprobs_k,
        prompt_logprobs: if echo { logprobs_k } else { None },
        echo,
        timeout_at: None,
        token_tx,
        // /v1/completions has no guard pipeline yet — the flag is
        // created so the scheduler's emit_step type-checks cleanly,
        // but never flipped.
        cancel_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };

    state.request_tx.send(request).await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        )
    })?;

    let chunk_id = crate::openai::new_completion_id();
    let model_name = state.model_name.clone();

    let model = model_name.clone();
    let id = chunk_id.clone();
    let mut all_toks: Vec<u32> = Vec::new();
    let mut emitted: usize = 0;
    // Incremental-detokenizer state (see `ChatTokenizer::incremental_decode`):
    // extends `content_decoded` a bounded suffix at a time instead of
    // re-decoding all_toks every token (O(n²) → O(n)).
    let mut content_decoded = String::new();
    let mut detok_prefix_offset: usize = 0;
    let mut detok_read_offset: usize = 0;
    let token_stream = ReceiverStream::new(token_rx).flat_map(move |event| {
        let events: Vec<Result<Event, std::convert::Infallible>> = match event {
            // Echo + logprobs: prompt text and its logprobs, before any
            // generated token (the scheduler emits this exactly once).
            StreamEvent::PromptLogprobs(lps) => {
                let prompt_toks = echo_prompt.clone().unwrap_or_default();
                let text = state.tokenizer.decode(&prompt_toks).unwrap_or_default();
                let decode = |tid: u32| state.tokenizer.decode(&[tid]).unwrap_or_default();
                let lp = super::completions_logprobs::build_completion_logprobs(
                    &decode,
                    true,
                    &prompt_toks,
                    &lps,
                    &[],
                    &[],
                );
                let chunk = CompletionChunk::echo_chunk(&model, &id, text, Some(lp));
                vec![Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                )]
            }
            StreamEvent::Token(tok) | StreamEvent::TokenWithLogprobs(tok, _) => {
                all_toks.push(tok);
                content_decoded.push_str(&state.tokenizer.incremental_decode(
                    &all_toks,
                    &mut detok_prefix_offset,
                    &mut detok_read_offset,
                ));
                let stable_end = content_decoded.len();
                let delta = if stable_end <= emitted {
                    String::new()
                } else {
                    let d = content_decoded[emitted..stable_end].to_string();
                    emitted = stable_end;
                    d
                };
                let chunk = CompletionChunk::text_chunk(&model, &id, delta);
                vec![Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                )]
            }
            StreamEvent::Done {
                finish_reason,
                prompt_tokens: _,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                reasoning_tokens,
                cached_prompt_tokens,
                guard_stop: _,
            } => {
                let tps = if decode_time_ms > 0.0 {
                    completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
                } else {
                    0.0
                };
                let usage = Usage {
                    prompt_tokens: prompt_len,
                    completion_tokens,
                    total_tokens: prompt_len + completion_tokens,
                    prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
                        cached_tokens: cached_prompt_tokens as usize,
                        audio_tokens: 0,
                    }),
                    completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
                        reasoning_tokens: reasoning_tokens as usize,
                        audio_tokens: 0,
                        accepted_prediction_tokens: 0,
                        rejected_prediction_tokens: 0,
                    }),
                    time_to_first_token_ms,
                    response_tokens_per_second: tps,
                };
                // Opt-in power: finish the span with the final token count as
                // the work-share numerator; rides the terminal usage chunk.
                #[cfg(feature = "power_attribution")]
                let power_json = power.take().and_then(|(handle, span, start_tokens)| {
                    crate::power::finish_completion_stream(
                        &handle,
                        span,
                        start_tokens,
                        completion_tokens,
                    )
                });
                #[cfg(not(feature = "power_attribution"))]
                let power_json: Option<serde_json::Value> = None;
                if include_usage {
                    // Chat parity: finish chunk without usage, then a
                    // choices:[] usage-only chunk (power rides the usage chunk).
                    let fin = CompletionChunk::finish_chunk_no_usage(&model, &id, &finish_reason);
                    let usage_chunk =
                        CompletionChunk::usage_only_chunk(&model, &id, usage).with_power(power_json);
                    vec![
                        Ok(Event::default().data(serde_json::to_string(&fin).unwrap_or_default())),
                        Ok(Event::default()
                            .data(serde_json::to_string(&usage_chunk).unwrap_or_default())),
                    ]
                } else {
                    let chunk = CompletionChunk::done_chunk(&model, &id, &finish_reason, usage)
                        .with_power(power_json);
                    vec![Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                    )]
                }
            }
            StreamEvent::Error(msg) => {
                vec![Ok(Event::default().data(format!(r#"{{"error":"{msg}"}}"#)))]
            }
        };
        futures::stream::iter(events)
    });

    // Echo WITHOUT logprobs: the scheduler emits no PromptLogprobs event
    // (nothing to collect), so prepend the prompt text chunk directly.
    let echo_prefix: Option<Event> = echo_only_text.map(|text| {
        let chunk = CompletionChunk::echo_chunk(&model_name, &chunk_id, text, None);
        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
    });

    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    let prefix = futures::stream::iter(
        echo_prefix
            .into_iter()
            .map(Ok::<_, std::convert::Infallible>),
    );
    let full_stream = prefix.chain(token_stream).chain(done_event);

    Ok(Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}
