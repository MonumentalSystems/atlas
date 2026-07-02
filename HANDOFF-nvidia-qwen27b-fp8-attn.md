# Handoff — nvidia/Qwen3.6-27B-NVFP4 FP8 attention (SHELVED)

Branch: `feat/native-fp8-attn-modelopt-mixed` (off `test/220-figdn`)
Commit: `0d82b36` feat(qwen35): keep FP8 attention native for ModelOpt mixed-precision checkpoints
Date: 2026-07-02. Status: **fix committed, model shelved again** (degeneration is orthogonal to the fix).

## Goal
User flagged that ModelOpt mixed-precision NVFP4 checkpoints ship **attention in FP8**, and Atlas
was quantizing it further to NVFP4 — a suspected quality issue. Task: keep FP8 attention native FP8.

## Root cause
`nvidia/Qwen3.6-27B-NVFP4` is `model_type: "qwen3_5"`, modelopt `MIXED_PRECISION`, and **DENSE**
(`0 experts`, 48 SSM + 16 full-attn, **head_dim=256**). NVFP4 FFN + **per-tensor-FP8 attention**
(q/k/v/o are FP8E4M3 with a *scalar* `weight_scale` + `input_scale`).

Being dense, it loads via `weight_loader/qwen35_dense.rs` (NOT the MoE `qwen35/load_layers.rs`).
The attention was going FP8 -> BF16 -> NVFP4 (`quantize_to_nvfp4`) because:
- `proj_is_native_fp8` only accepted a **2D** block scale → scalar failed.
- the dense native-FP8 *overlay* also required `Nvfp4Variant::Fp8Dequanted` (this model is `Standard`).

## The fix (commit 0d82b36)
Two files. Route FP8 attention through the existing native-FP8 path
(`load_fp8_block_scaled_as_fp8weight` → `set_fp8_weights` → `w8a16` gemv/gemm; the loader already
broadcasts a scalar scale into the [N/128,K/128] block matrix, so no new kernel).
- `qwen35/load_layers.rs` (MoE): `proj_is_native_fp8` accepts scalar; native-FP8 attn arm fires on
  FP8E4M3 dtype (dropped `native_fp8 &&`).
- `qwen35_dense.rs` (dense; the path this model takes): same scalar fix; overlay dropped the
  `Fp8Dequanted` requirement. **Opt-in via `ATLAS_DENSE_FP8=1`** (dense FP8 W8A16 attn is ~lossless
  but ~25% slower than the NVFP4 W4A16 fallback — quality/speed tradeoff).
Fully-NVFP4 attn (q_proj = packed E2M1) is unaffected; `force_nvfp4_all`/`fp4_proj_decode` preserved.

## Validation (gx10-9959, cuda13.2 container + FI-GDN)
Build local (`target/release/spark`, `ATLAS_TARGET_MODEL=*`) → scp to `~/spark-fp8attn` → serve.
- Overlay **fires on all 16 attention layers** (`ATLAS_DENSE_FP8=1`; per-layer
  "Skipping attention FP8 prefill transposes" — note `ATLAS_CUTLASS_NVFP4_GEMM` was on, so only
  **decode** got FP8; prefill transpose skipped → prefill stays NVFP4 = a prefill/decode mismatch).
- **Numeric A/B (no-think, temp=0): byte-IDENTICAL** native-FP8 vs NVFP4-requant on Fibonacci /
  primes / 17×24 / powers-of-2. No regression, no visible win — greedy margins don't flip.
- **Long-context + thinking (watchdogs OFF): BOTH degenerate identically.** 48-record salary
  aggregation (~797 prompt toks, gt Engineering $957,144). Native-FP8 AND NVFP4-requant baseline both
  loop ("Wait!!… No wait… STOP GUESSING… Actually…"), `finish=length`, never compute the sums.
  ⇒ long-context degeneration is **not** FP8-related; it's the model + watchdogs-off on a hard task.

## nvidia 27B degeneration — the real story (orthogonal to the fix)
- Much of the early "collapse" was **my error**: `max_tokens=260` truncated mid-thinking (model needs
  ~630 tokens of `<think>` before answering).
- `/no_think` prompt switch is **ignored** by this template; must use request
  `chat_template_kwargs:{enable_thinking:false}`.
- **No-think** converges cleanly (no loops); **think** reasons better but loops on ambiguous prompts.
- The loops are what the **content-loop watchdog** catches (`ATLAS_DISABLE_WATCHDOGS=1` turns all off;
  the code comment says watchdogs exist to compensate for **FP8 token-margin flips**).
- **Working setup**: `ATLAS_DISABLE_TEMPLATE_OVERRIDES=1` + `enable_thinking` (or no-think for
  stability) + model-default sampling (temp=1/top_p .95/top_k 20/min_p .08/top_n_sigma 1) +
  `max_tokens ≥ ~1200` + **watchdogs ON** + `--tool-call-parser qwen3_coder --disable-tool-grammar true`.

## Verdict
Fix is correct + does what was asked (native FP8 attn for mixed-precision), but is **empirically
neutral on this model** and is **not** the degeneration lever → **shelved**. Kept as opt-in.

## Serve recipe (reproduce)
```
docker run -d --name qwen27 --gpus all --ipc host --network host \
  -v ~/spark-fp8attn:/usr/local/bin/spark:ro -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/figdn_libs:/figdn_libs:ro \
  -e HF_HUB_OFFLINE=1 -e RUST_LOG=info -e ATLAS_GDN_FLASHINFER=1 -e ATLAS_DISABLE_TEMPLATE_OVERRIDES=1 \
  -e ATLAS_KV_OVERCOMMIT=1 -e ATLAS_DENSE_FP8=1 \  # drop ATLAS_DENSE_FP8 for NVFP4-requant baseline
  -e LD_LIBRARY_PATH=/figdn_libs:/usr/local/cuda/lib64 \
  atlas-holo:cuda13.2-fp4test serve nvidia/Qwen3.6-27B-NVFP4 --model-name test --port 8888 \
  --gpu-memory-utilization 0.8 --max-seq-len 8192 --max-num-seqs 1 --max-batch-size 1 \
  --scheduling-policy slai --kv-cache-dtype bf16 --tool-call-parser qwen3_coder --disable-tool-grammar true \
  --default-chat-template-kwargs '{"enable_thinking":true}'
```
Build: `CUTLASS_HOME=/home/ms/cutlass CUDA_HOME=/usr/local/cuda LD_LIBRARY_PATH=/home/ms/nccl/build/lib
LIBRARY_PATH=/home/ms/nccl/build/lib ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*'
cargo build --release -p spark-server --bin spark` (local, ~3 min incremental), then scp to gx10.

## NOTE
gx10's `~/atlas` checkout (branch `feature/holo-port-pr177`) also had this source patch applied by
hand during debugging — harmless (we build the binary locally + copy), can be reverted.

## Next
Test the same FP8-native-attn fix on **Ornith-1.0-35B-FP8** (a whole-model FP8 checkpoint — different
shape from this NVFP4+FP8-attn mixed case; may be the better target for the fix).
