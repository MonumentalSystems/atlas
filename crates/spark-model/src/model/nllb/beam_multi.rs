// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-request (C×B) batched beam co-dispatch (Phase c). Runs the C requests'
//! encoders sequentially into per-request cross-KV, then decodes ALL Σ beams as
//! one M=(Σ beams) batch per step — every projection / self-attention / FFN /
//! lm_head is a single tensor-core GEMM, and only cross-attention loops per
//! request (each reading its own encoder output, [`super::beam::CrossGroup`]).
//! Per-request candidate pruning, hypothesis pools, per-request `max_new`, and
//! staggered completion run on the host, so each request's winner is token-exact
//! versus running it alone. All fused requests must share one LoRA adapter (the
//! model's single global gate); the scheduler groups by adapter before calling in
//! and this driver falls back to serial if a mixed-adapter batch slips through.

use anyhow::{Context, Result};

use super::NllbGpuModel;
use super::beam::{BeamHyps, CrossGroup, DecBuf, NLLB_TOPK_KMAX, logsumexp, top_k};
use super::kv::NllbSeqKv;
use super::util::u32_bytes;
use crate::traits::BeamReq;

/// One beam carrying its precomputed candidate expansion: the log-sum-exp over
/// the full vocab (`lse`) and the top `(value, token)` pairs — from the on-device
/// top-k kernel (Phase d) or the host fallback. The next step's candidates are
/// `score + (value − lse)` over these, so no per-step full-vocab host scan.
struct MBeam {
    tokens: Vec<u32>,
    score: f32,
    lse: f32,
    top: Vec<(f32, u32)>,
}

/// Per-request decode state within the fused C×B batch.
struct ReqState {
    b: usize,        // num_beams
    row_off: usize,  // first batch row for this request
    forced_bos: u32, // tgt-lang token seeded at decoder step 1
    enc_len: usize,  // this request's encoder length (its cross key length)
    max_new: usize,  // this request's own generation cap
    early_stopping: bool,
    beams: Vec<MBeam>,
    hyps: BeamHyps,
    active: bool,
}

/// New beams staged for one request after a prune, applied post-forward.
struct Pending {
    new_tokens: Vec<Vec<u32>>,
    new_scores: Vec<f32>,
    best_running: f32,
}

impl NllbGpuModel {
    /// Co-dispatch beam search for several requests as one fused batch. Returns
    /// one winning hypothesis per request, in request order (decoder-start
    /// stripped, EOS-terminated) — identical to `generate_beam_one` per request.
    pub(super) fn beam_batched_multi(&self, reqs: &[BeamReq]) -> Result<Vec<Vec<u32>>> {
        if reqs.len() <= 1 {
            return reqs.iter().map(|r| self.generate_beam_one(r)).collect();
        }
        // The fused M=(Σ beams) forward uses one global LoRA gate, so every
        // request in the batch must share an adapter. Heterogeneous adapters
        // fall back to serial (the scheduler normally groups by adapter first).
        let slot0 = reqs[0].adapter_slot;
        if reqs.iter().any(|r| r.adapter_slot != slot0) {
            return reqs.iter().map(|r| self.generate_beam_one(r)).collect();
        }
        self.set_lora_active(slot0);

        let gpu = self.gpu.as_ref();
        // 1) Run each encoder into its own cross-KV (kept alive across decode).
        let mut kvs: Vec<NllbSeqKv> = Vec::with_capacity(reqs.len());
        let mut states: Vec<ReqState> = Vec::with_capacity(reqs.len());
        let mut row_off = 0usize;
        let mut max_new = 2usize;
        for req in reqs {
            let src_id = if req.src_lang_id != 0 {
                req.src_lang_id
            } else {
                self.lang.src_lang_id
            };
            let tgt_id = if req.tgt_lang_id != 0 {
                req.tgt_lang_id
            } else {
                self.lang.tgt_lang_id
            };
            let src = self.lang.encoder_input_with(src_id, &req.prompt_tokens);
            let mut kv = NllbSeqKv::new(gpu, self.dec_layers, 2, self.d)?;
            self.run_encoder(&src, &mut kv)?;
            let b = req.num_beams.max(1);
            let mn = req.max_new.max(2).min(self.cache_rows);
            max_new = max_new.max(mn);
            states.push(ReqState {
                b,
                row_off,
                forced_bos: tgt_id,
                enc_len: kv.enc_len,
                max_new: mn,
                early_stopping: req.early_stopping,
                beams: Vec::new(),
                hyps: BeamHyps::new(b, req.length_penalty),
                active: true,
            });
            kvs.push(kv);
            row_off += b;
        }
        let rows = row_off; // Σ beams over all requests

        // 2) Batch scratch sized for `rows`; per-row cross length = its enc_len.
        let mut crosstk_host = vec![0u32; rows];
        for st in &states {
            for i in 0..st.b {
                crosstk_host[st.row_off + i] = st.enc_len as u32;
            }
        }
        let buf = DecBuf::new(self, rows, &crosstk_host, max_new)?;
        let alloc_set = |n: usize| {
            (0..n)
                .map(|_| gpu.alloc(rows * max_new * self.d * 2))
                .collect::<Result<Vec<_>>>()
        };
        let mut sk = alloc_set(self.dec_layers)?;
        let mut sv = alloc_set(self.dec_layers)?;
        let mut sk2 = alloc_set(self.dec_layers)?;
        let mut sv2 = alloc_set(self.dec_layers)?;
        let perm_dev = gpu.alloc(rows * 4)?;
        let cross: Vec<CrossGroup> = states
            .iter()
            .zip(&kvs)
            .map(|(st, kv)| CrossGroup {
                k: &kv.cross_k,
                v: &kv.cross_v,
                enc_len: st.enc_len,
                row_off: st.row_off,
                n_rows: st.b,
            })
            .collect();

        let res = self.beam_multi_loop(
            &mut states,
            &cross,
            rows,
            max_new,
            &buf,
            &mut sk,
            &mut sv,
            &mut sk2,
            &mut sv2,
            perm_dev,
        );

        drop(cross);
        for p in sk.into_iter().chain(sv).chain(sk2).chain(sv2) {
            gpu.free(p)?;
        }
        gpu.free(perm_dev)?;
        buf.free(gpu)?;
        for kv in kvs {
            kv.free(gpu)?;
        }
        res
    }

    #[allow(clippy::too_many_arguments)]
    fn beam_multi_loop(
        &self,
        states: &mut [ReqState],
        cross: &[CrossGroup],
        rows: usize,
        max_new: usize,
        buf: &DecBuf,
        sk: &mut Vec<spark_runtime::gpu::DevicePtr>,
        sv: &mut Vec<spark_runtime::gpu::DevicePtr>,
        sk2: &mut Vec<spark_runtime::gpu::DevicePtr>,
        sv2: &mut Vec<spark_runtime::gpu::DevicePtr>,
        perm_dev: spark_runtime::gpu::DevicePtr,
    ) -> Result<Vec<Vec<u32>>> {
        let (dec_start, eos) = (self.lang.decoder_start_id, self.lang.eos_id);
        // Candidate width: top-(2·max_beams) per row (each request slices its own
        // 2·b). The on-device top-k kernel handles k ≤ NLLB_TOPK_KMAX; above that
        // (num_beams > 16) fall back to the host full-vocab scan. Same math either
        // way, so per-request output is byte-identical.
        //
        // Device top-k only wins once the per-step full-logits D2H dominates —
        // measured break-even is ~64 rows (below it the kernel's sync+launch+
        // serial extract cost more than a small D2H + host `top_k`). So gate on
        // the batch row count (ATLAS_NLLB_DEVICE_TOPK_MIN_ROWS, default 64);
        // ATLAS_NLLB_HOST_TOPK=1 forces the host path (A/B + >16-beam requests).
        let k = 2 * states.iter().map(|s| s.b).max().unwrap_or(1);
        let host_forced = std::env::var("ATLAS_NLLB_HOST_TOPK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let min_rows = std::env::var("ATLAS_NLLB_DEVICE_TOPK_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64);
        let use_device = k <= NLLB_TOPK_KMAX && rows >= min_rows && !host_forced;

        // Seed: step 0 = decoder_start (all rows), step 1 = per-request forced_bos.
        self.beam_forward_step(&vec![dec_start; rows], 0, rows, sk, sv, cross, buf)?;
        let mut seed1 = vec![eos; rows];
        for st in states.iter() {
            for i in 0..st.b {
                seed1[st.row_off + i] = st.forced_bos;
            }
        }
        self.beam_forward_step(&seed1, 1, rows, sk, sv, cross, buf)?;
        let init = self.beam_cands(buf, rows, k, use_device)?;
        for st in states.iter_mut() {
            st.beams = (0..st.b)
                .map(|bi| MBeam {
                    tokens: vec![dec_start, st.forced_bos],
                    score: if bi == 0 { 0.0 } else { f32::NEG_INFINITY },
                    lse: init[st.row_off + bi].0,
                    top: init[st.row_off + bi].1.clone(),
                })
                .collect();
        }

        let mut cur_len = 2usize; // shared beam length of active requests
        for step in 1..max_new {
            // Freeze any request that has reached its own generation cap.
            for st in states.iter_mut() {
                if st.active && step >= st.max_new {
                    st.active = false;
                }
            }
            let mut cur = vec![eos; rows];
            let mut perm: Vec<u32> = (0..rows as u32).collect();
            let mut staged: Vec<Option<Pending>> = (0..states.len()).map(|_| None).collect();
            let mut any_active = false;

            for (ri, st) in states.iter_mut().enumerate() {
                if !st.active {
                    continue; // rows keep identity perm + eos (output ignored)
                }
                // Candidate expansion over this request's beams, using each
                // beam's precomputed (lse, top-k) — its own 2·b slice.
                let mut cands: Vec<(f32, usize, u32)> = Vec::new();
                for (bi, beam) in st.beams.iter().enumerate() {
                    if !beam.score.is_finite() {
                        continue;
                    }
                    for &(val, tok) in beam.top.iter().take(2 * st.b) {
                        cands.push((beam.score + (val - beam.lse), bi, tok));
                    }
                }
                cands.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());

                let (mut nt, mut ns) = (Vec::new(), Vec::new());
                for (score, parent, tok) in cands {
                    if nt.len() == st.b {
                        break;
                    }
                    if tok == eos {
                        st.hyps.add(st.beams[parent].tokens.clone(), score);
                    } else {
                        let row = st.row_off + nt.len();
                        perm[row] = (st.row_off + parent) as u32;
                        cur[row] = tok;
                        let mut t = st.beams[parent].tokens.clone();
                        t.push(tok);
                        nt.push(t);
                        ns.push(score);
                    }
                }
                if nt.is_empty() {
                    st.active = false;
                    continue;
                }
                let best_running = ns[0];
                staged[ri] = Some(Pending {
                    new_tokens: nt,
                    new_scores: ns,
                    best_running,
                });
                any_active = true;
            }
            if !any_active {
                break;
            }

            // Reorder every row's self-KV by the global parent map, then forward.
            self.gpu.copy_h2d(u32_bytes(&perm), perm_dev)?;
            for l in 0..self.dec_layers {
                self.gather(sk[l], sk2[l], perm_dev, rows, cur_len, buf.cache_rows)?;
                self.gather(sv[l], sv2[l], perm_dev, rows, cur_len, buf.cache_rows)?;
            }
            std::mem::swap(sk, sk2);
            std::mem::swap(sv, sv2);

            self.beam_forward_step(&cur, cur_len, rows, sk, sv, cross, buf)?;
            let lh = self.beam_cands(buf, rows, k, use_device)?;

            for (ri, st) in states.iter_mut().enumerate() {
                let Some(p) = staged[ri].take() else {
                    continue;
                };
                st.beams = (0..p.new_tokens.len())
                    .map(|i| MBeam {
                        tokens: p.new_tokens[i].clone(),
                        score: p.new_scores[i],
                        lse: lh[st.row_off + i].0,
                        top: lh[st.row_off + i].1.clone(),
                    })
                    .collect();
                if st.hyps.is_done(p.best_running, cur_len, st.early_stopping) {
                    st.active = false;
                }
            }
            cur_len += 1;
        }

        // Collect each request's winner (remaining finite beams count as hyps).
        let mut out = Vec::with_capacity(states.len());
        for st in states.iter_mut() {
            for beam in &st.beams {
                if beam.score.is_finite() {
                    st.hyps.add(beam.tokens.clone(), beam.score);
                }
            }
            let mut best = st
                .hyps
                .best()
                .context("nllb beam multi: no finished hypotheses")?;
            best.push(eos);
            out.push(best[1..].to_vec());
        }
        Ok(out)
    }

    /// Per-row `(lse, top-k (value, token))` for the multi-request beam loop:
    /// the on-device top-k kernel (Phase d, ~`rows*k*8`-byte D2H/step) or the
    /// host full-vocab scan (`rows*vocab*2`-byte D2H + host `logsumexp`/`top_k`).
    fn beam_cands(
        &self,
        buf: &DecBuf,
        rows: usize,
        k: usize,
        use_device: bool,
    ) -> Result<Vec<(f32, Vec<(f32, u32)>)>> {
        if use_device {
            return self.beam_cands_device(buf, rows, k);
        }
        let lh = self.beam_logits_host(buf, rows)?;
        Ok(lh
            .into_iter()
            .map(|row| {
                let lse = logsumexp(&row);
                let top = top_k(&row, k)
                    .into_iter()
                    .map(|(v, i)| (v, i as u32))
                    .collect();
                (lse, top)
            })
            .collect())
    }
}
