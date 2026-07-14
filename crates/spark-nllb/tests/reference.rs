// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end validation against HuggingFace `M2M100ForConditionalGeneration`.
//!
//! Requires a local safetensors NLLB checkpoint; set `NLLB_MODEL_DIR` to run
//! (e.g. a clone of `MonumentalSystems/nllb-200-3.3B`). Skips silently when
//! unset so CPU-only CI without the weights stays green.
//!
//! Ground truth captured from `transformers` 4.x on
//! `facebook/nllb-200-3.3B` (fp32), input "Hello, world. How are you today?"
//! eng_Latn → fra_Latn, greedy:
//!   input_ids = [256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259,
//!                30435, 248130, 2]
//!   forced_bos (fra_Latn) = 256057
//!   generated = [256057, 17994, 141190, 248079, 25358, 123732, 248105,
//!                30213, 248079, 1724, 25601, 385, 2]
//!   encoder last_hidden_state.sum() = -14.769035

use std::path::PathBuf;

use spark_nllb::NllbModel;

fn model_dir() -> Option<PathBuf> {
    std::env::var("NLLB_MODEL_DIR").ok().map(PathBuf::from)
}

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057;
const EXPECTED_GEN: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 123732, 248105, 30213, 248079, 1724, 25601, 385, 2,
];

#[test]
fn nllb_greedy_matches_reference() {
    let Some(dir) = model_dir() else {
        eprintln!("NLLB_MODEL_DIR not set — skipping");
        return;
    };
    let model = NllbModel::load_dir(&dir).expect("load model");

    // Encoder numerics: sum of the encoder hidden state.
    let enc = model.encode(INPUT_IDS);
    let sum: f32 = enc.iter().sum();
    assert!(
        (sum - (-14.769035)).abs() < 0.05,
        "encoder sum {sum} != reference -14.769035"
    );

    // Greedy generation exact-token match.
    let out = model.generate(INPUT_IDS, FORCED_BOS, 64);
    assert_eq!(out, EXPECTED_GEN, "generated ids diverged from reference");
}
