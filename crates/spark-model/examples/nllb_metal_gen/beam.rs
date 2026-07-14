// SPDX-License-Identifier: AGPL-3.0-only

use super::KvCache;

pub(super) struct Beam {
    pub(super) tokens: Vec<u32>,
    pub(super) score: f32,
    pub(super) cache: KvCache,
    pub(super) logits: Vec<f32>,
}

pub(super) struct BeamHyps {
    num_beams: usize,
    length_penalty: f32,
    beams: Vec<(Vec<u32>, f32)>,
}

impl BeamHyps {
    pub(super) fn new(num_beams: usize, length_penalty: f32) -> Self {
        Self {
            num_beams,
            length_penalty,
            beams: Vec::new(),
        }
    }

    fn worst(&self) -> f32 {
        self.beams
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::INFINITY, f32::min)
    }

    pub(super) fn add(&mut self, tokens: Vec<u32>, sum_logprob: f32) {
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

    pub(super) fn is_done(&self, best_running_sum_logprob: f32, cur_len: usize) -> bool {
        self.beams.len() >= self.num_beams
            && self.worst() >= best_running_sum_logprob / (cur_len as f32).powf(self.length_penalty)
    }

    pub(super) fn best(&self) -> Option<Vec<u32>> {
        self.beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(t, _)| t.clone())
    }
}

pub(super) fn logsumexp(x: &[f32]) -> f32 {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m + x.iter().map(|&v| (v - m).exp()).sum::<f32>().ln()
}

pub(super) fn top_k(x: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut best = Vec::with_capacity(k + 1);
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

pub(super) fn argmax_f32(bytes: &[u8]) -> usize {
    f32_slice(bytes)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub(super) fn decoder_pos_table(max_len: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let emb_scale = 10000f32.ln() / (half as f32 - 1.0);
    let mut t = vec![0f32; max_len * d];
    for i in 0..max_len {
        let p = (i + 2) as f32;
        for j in 0..half {
            let ang = p * (-(j as f32) * emb_scale).exp();
            t[i * d + j] = ang.sin();
            t[i * d + half + j] = ang.cos();
        }
    }
    t
}

fn f32_slice(b: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len() / 4) }
}
