// SPDX-License-Identifier: AGPL-3.0-only

//! K=4 verify step and its shared result-commit routine.

use super::*;

const K4_SUMMARY_PERIOD: u64 = 100;
static K4_ACCEPT_3: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K4_ACCEPT_2: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K4_ACCEPT_1: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K4_ACCEPT_0: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
fn k4_record_outcome(num_accepted: usize, seq_len: usize) {
    let counter = match num_accepted {
        3 => &K4_ACCEPT_3,
        2 => &K4_ACCEPT_2,
        1 => &K4_ACCEPT_1,
        _ => &K4_ACCEPT_0,
    };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let total = K4_ACCEPT_3.load(std::sync::atomic::Ordering::Relaxed)
        + K4_ACCEPT_2.load(std::sync::atomic::Ordering::Relaxed)
        + K4_ACCEPT_1.load(std::sync::atomic::Ordering::Relaxed)
        + K4_ACCEPT_0.load(std::sync::atomic::Ordering::Relaxed);
    if total >= K4_SUMMARY_PERIOD {
        let a3 = K4_ACCEPT_3.swap(0, std::sync::atomic::Ordering::Relaxed);
        let a2 = K4_ACCEPT_2.swap(0, std::sync::atomic::Ordering::Relaxed);
        let a1 = K4_ACCEPT_1.swap(0, std::sync::atomic::Ordering::Relaxed);
        let a0 = K4_ACCEPT_0.swap(0, std::sync::atomic::Ordering::Relaxed);
        let total = (a3 + a2 + a1 + a0).max(1);
        let mean = (3 * a3 + 2 * a2 + a1) as f64 / total as f64;
        tracing::info!(
            "K4 summary: {a3} accept-3 / {a2} accept-2 / {a1} accept-1 / {a0} reject in last {total} steps (mean accepted={mean:.2}) seq_len={seq_len}"
        );
    }
}

/// K=4 verify: [last_token, draft1, draft2, draft3] → [v0, v1, v2, v3].
pub fn step_verify_k4(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }
    let tokens = [a.last_token, drafts[0], drafts[1], drafts[2]];
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF4) {
        tracing::error!("EP broadcast verify_k4 cmd: {e:#}");
        a.finished = true;
        return;
    }
    for &token in &tokens {
        if let Err(e) = model.ep_broadcast_cmd(token) {
            tracing::error!("EP broadcast verify_k4 token: {e:#}");
            a.finished = true;
            return;
        }
    }
    let t_verify = Instant::now();
    let result = if dflash_verify_raw_argmax && !model.is_ep() {
        model.decode_and_verify_fused(&tokens, &mut a.seq, 0)
    } else {
        model
            .decode_verify_graphed_k4(&tokens, &mut a.seq, 0)
            .map(|v| v.to_vec())
    };
    let result = match result {
        Ok(result) if result.len() == 4 => [result[0], result[1], result[2], result[3]],
        Ok(_) => {
            tracing::error!("K4 verifier returned wrong result length");
            a.finished = true;
            return;
        }
        Err(e) => {
            tracing::error!("K4 verifier failed: {e:#}");
            a.finished = true;
            return;
        }
    };
    finish_k4_verify(
        model,
        a,
        drafts,
        num_drafts,
        verify_ctx,
        dflash_verify_raw_argmax,
        tokens,
        result,
        t_verify.elapsed().as_micros(),
    );
}

/// Complete one K=4 verification from already-computed raw verifier argmaxes.
/// Shared by the singleton graphed path and the C×4 batched path.
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_k4_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
    tokens: [u32; 4],
    raw: [u32; 4],
    verify_us: u128,
) {
    a.last_token_time = Instant::now();
    let verified = if dflash_verify_raw_argmax
        && !crate::scheduler::verify_pipeline_helper::dflash_masked_verify_enabled()
    {
        raw
    } else {
        let picked = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model, &raw, a, verify_ctx,
        );
        [
            picked.first().copied().unwrap_or(raw[0]),
            picked.get(1).copied().unwrap_or(raw[1]),
            picked.get(2).copied().unwrap_or(raw[2]),
            picked.get(3).copied().unwrap_or(raw[3]),
        ]
    };
    let num_accepted = drafts
        .iter()
        .zip(verified.iter())
        .take_while(|(draft, verified)| draft == verified)
        .count();
    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &verified, top_logprobs)
    } else {
        Vec::new()
    };
    if let Err(e) = model.ep_broadcast_cmd(num_accepted as u32) {
        tracing::error!("EP broadcast verify_k4 result: {e:#}");
        a.finished = true;
        return;
    }
    tracing::debug!(
        "K4 verify: tokens={tokens:?} raw={raw:?} drafts={drafts:?} accepted={num_accepted} seq_len={}",
        a.seq.seq_len
    );

    let committed = num_accepted + 1;
    let discard = 4 - committed;
    if discard > 0 {
        a.seq.seq_len -= discard;
        a.seq.tokens.truncate(a.seq.tokens.len() - discard);
        if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
    }
    for i in 0..num_accepted {
        emit_token(a, drafts[i], verify_lps.get(i).cloned());
        if a.finished {
            return;
        }
    }
    emit_token(
        a,
        verified[num_accepted],
        verify_lps.get(num_accepted).cloned(),
    );
    if a.finished {
        return;
    }
    a.last_token = verified[num_accepted];
    if let Err(e) = model.commit_verify_state_async(&mut a.seq, committed, 4) {
        tracing::error!("commit_verify_state_async (K=4): {e:#}");
        a.finished = true;
        return;
    }
    if let Err(e) = model.save_hidden_for_mtp(num_accepted, 0) {
        tracing::error!("save_hidden_for_mtp({num_accepted}): {e:#}");
        return;
    }
    if num_accepted == 3
        && let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0)
    {
        tracing::error!("trim_proposer_state: {e:#}");
    }
    let t_propose = Instant::now();
    let grammar_mask = mtp_grammar_mask_for(a);
    match model.run_mtp_propose_multi(
        a.last_token,
        a.seq.seq_len,
        num_drafts,
        &mut a.seq,
        0,
        grammar_mask.as_deref(),
    ) {
        Ok(drafts) if !drafts.is_empty() => a.pending_drafts = drafts,
        Ok(_) => {}
        Err(e) => tracing::error!("run_mtp_propose_multi: {e:#}"),
    }
    tracing::debug!(
        "K4 accept-{num_accepted}: verify={verify_us}μs propose={}μs seq_len={}",
        t_propose.elapsed().as_micros(),
        a.seq.seq_len
    );
    k4_record_outcome(num_accepted, a.seq.seq_len);
}
