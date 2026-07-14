// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of nllb_cuda_bf16.rs for the 500-LoC cap.

use super::*;

impl BeamHyps {
    pub fn new(num_beams: usize, length_penalty: f32) -> Self {
        Self {
            num_beams,
            length_penalty,
            beams: Vec::new(),
        }
    }
    pub fn worst(&self) -> f32 {
        self.beams
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::INFINITY, f32::min)
    }
    pub fn add(&mut self, tokens: Vec<u32>, sum_logprob: f32) {
        let score = sum_logprob / (tokens.len() as f32).powf(self.length_penalty);
        if self.beams.len() < self.num_beams || score > self.worst() {
            self.beams.push((tokens, score));
            if self.beams.len() > self.num_beams {
                let (wi, _) = self
                    .beams
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
                    .unwrap();
                self.beams.swap_remove(wi);
            }
        }
    }
    pub fn is_done(&self, best_running_sum_logprob: f32, cur_len: usize) -> bool {
        if self.beams.len() < self.num_beams {
            return false;
        }
        let highest = best_running_sum_logprob / (cur_len as f32).powf(self.length_penalty);
        self.worst() >= highest
    }
    pub fn best(&self) -> Option<Vec<u32>> {
        self.beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(t, _)| t.clone())
    }
}

// ───────────────────────── host helpers ─────────────────────────

pub fn logsumexp(x: &[f32]) -> f32 {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m + x.iter().map(|&v| (v - m).exp()).sum::<f32>().ln()
}

pub fn top_k(x: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, &v) in x.iter().enumerate() {
        if best.len() < k {
            best.push((v, i));
            if best.len() == k {
                best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }
        } else if v > best[k - 1].0 {
            best[k - 1] = (v, i);
            let mut j = k - 1;
            while j > 0 && best[j].0 > best[j - 1].0 {
                best.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    best
}

pub fn sinusoid_row(pos: f32, d: usize, out: &mut [bf16]) {
    let half = d / 2;
    let emb_scale = 10000f32.ln() / (half as f32 - 1.0);
    for j in 0..half {
        let ang = pos * (-(j as f32) * emb_scale).exp();
        out[j] = bf16::from_f32(ang.sin());
        out[half + j] = bf16::from_f32(ang.cos());
    }
}

pub fn decoder_pos_table_bf16(max_len: usize, d: usize) -> Vec<bf16> {
    let mut t = vec![bf16::from_f32(0.0); max_len * d];
    for i in 0..max_len {
        sinusoid_row((i + 2) as f32, d, &mut t[i * d..i * d + d]);
    }
    t
}

pub fn encoder_pos_bf16(ids: &[u32], d: usize, pad: u32) -> Vec<bf16> {
    let seq = ids.len();
    let mut t = vec![bf16::from_f32(0.0); seq * d];
    let mut running = 0u32;
    for (i, &id) in ids.iter().enumerate() {
        let p = if id != pad {
            running += 1;
            running + pad
        } else {
            pad
        };
        if p != pad {
            sinusoid_row(p as f32, d, &mut t[i * d..i * d + d]);
        }
    }
    t
}

pub fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn bf16_bytes(v: &[bf16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
