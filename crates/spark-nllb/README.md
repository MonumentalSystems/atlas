<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# spark-nllb

Self-contained **CPU** reference runtime for **NLLB-200 / M2M-100** — a
translation-focused *encoder-decoder* (seq2seq) transformer.

## Why a separate crate

Atlas's production engine is **decoder-only and GPU-only**: the
`TransformerLayer`/paged-KV/scheduler stack assumes causal autoregressive
generation, and `GpuBackend` has no CPU implementation. NLLB is a seq2seq model
(bidirectional encoder + decoder cross-attention + sinusoidal absolute
positions + ReLU FFN + biased LayerNorm), so the `spark-model` marker loader for
`model_type = "m2m_100" | "nllb"` deliberately fails fast.

This crate is the *"load it at least on CPU"* path: a dependency-light, fp32
port of HuggingFace `M2M100ForConditionalGeneration` that actually loads the
checkpoint and produces translations, with **no CUDA/Metal dependency**. It is
validated bit-faithfully against `transformers` (see `tests/reference.rs`).

## Weights

NLLB-200 ships as PyTorch `.bin` (pickle), which Atlas's safetensors-only
loader cannot read. A converted fp32 safetensors copy lives at:

- **`MonumentalSystems/nllb-200-3.3B`** (HuggingFace)

Download it (or any safetensors NLLB checkpoint) to a local directory.

## Usage

```bash
cargo run -p spark-nllb --release --bin nllb-translate -- \
    --model /path/to/nllb-200-3.3B-st \
    --src eng_Latn --tgt fra_Latn \
    "Hello, world. How are you today?"
# -> Bonjour, comment allez-vous, mon monde ?
```

`--src` / `--tgt` are NLLB FLORES-200 language codes (`eng_Latn`, `fra_Latn`,
`spa_Latn`, `deu_Latn`, …).

## Validation

```bash
NLLB_MODEL_DIR=/path/to/nllb-200-3.3B-st cargo test -p spark-nllb --release
```

Asserts the encoder hidden-state checksum and the exact greedy token sequence
against the HuggingFace reference. Skips silently when `NLLB_MODEL_DIR` is unset
(so CPU CI without the weights stays green).

## Status / next steps

- ✅ CPU fp32 encoder-decoder forward + greedy generation, exact-match with HF.
- ⏳ GPU path: the closest reusable asset is Atlas's ViT vision encoder
  (biased LayerNorm, bias-GEMM, dense non-causal SDPA) plus a new plain-ReLU
  kernel and decoder cross-attention orchestration.
- ⏳ Beam search (NLLB default is `num_beams=5`); this runtime is greedy.
