# Atlas LoRA v0 — Implementation Status (M0 + M1-attention)

Implements the [MVP proposal](lora-mvp-proposal.md) / [codebase brief](lora-codebase-brief.md).
This is a **working POC**: a served, fine-tuned tiny model on GB10, verified end-to-end.

## What ships

**Serve a single PEFT LoRA adapter, loaded at startup, applied to every request as a
runtime BF16 delta** (`y += scale·(x@Aᵀ)@Bᵀ`) — never merged into the (NVFP4) base
weights. Zero new CUDA kernels; the deltas reuse `dense_gemv_bf16` / `dense_gemm_tc` /
`bf16_scaled_add` and are captured inside the existing CUDA decode graphs.

```
spark serve Hcompany/Holo-3.1-0.8B \
  --lora-adapter my-ft=/path/to/peft-adapter-dir \
  --max-lora-rank 64
```

### M0 — load, validate, account
- `--lora-adapter NAME=PATH_OR_HF_ID`, `--max-lora-rank` (64), `--max-loras` (8).
  Repeated flag → named reject (multi-adapter is M2).
- PEFT `adapter_config.json` parser (`atlas-core`), hard-fail with `REJECT(...)` reasons
  for every unsupported feature: non-LORA `peft_type`, DoRA, bias, rank/alpha patterns,
  `modules_to_save`, `all-linear`, absent `use_rslora`, `r=0`.
- Dedicated adapter safetensors loader (`spark-runtime`) — host F16→BF16 (the base
  `WeightDtype` whitelist rejects F16), header-only OOM preflight, named pickle reject.
- Key remap + per-`LayerType` allow-list (`spark-model`): only the **full-attention**
  layers × {k,v,o,gate,up,down} are accepted. Named hard rejections (never silent skips):
  `gated-q-proj` (holo's `attn_output_gate` interleaves Q+gate), `gdn-target`,
  `non-full-attention-layer`, plus a bidirectional tensor↔target audit and A/B shape audit.
- A/B packed **rank-padded to `max_lora_rank`** in one fixed-address pool with per-module
  `[max_loras]` device pointer tables (the frozen M2 layout contract; v0 fills slot 0).
- Pool VRAM is allocated before the KV-budget snapshot, so it is **budgeted against the KV
  cache** (GB10 unified-memory OOM = freeze).
- Scaling `alpha/r` (or `alpha/√r` under rsLoRA), read per adapter, never defaulted.

### M1 — runtime delta at the attention insertion points
- `apply_lora_delta` wired at **prefill k/v/o** and **decode k/v/o** (decode k/v applied
  before norm/RoPE/kv-cache-write, so the KV cache stores the adapted values, matching
  HF `k_norm(k_proj(x)+Δ)`).
- Deltas contract at the **padded** `max_rank` (B is packed with `max_rank` row stride);
  a real-rank contraction would misread every B row past the first when `r < max_rank`.
- `ATLAS_LORA_EAGER=1` disables decode-graph capture (debugging hatch); deltas are
  otherwise graph-safe (pool weights, arena scratch, and the f32 scale are load-time-fixed).
- `/v1/models` advertises the adapter name first; base-name requests get adapted output
  with a one-line warn (v0 always-on wart; per-request routing is M2).

## Verification (holo-3.1-0.8b on GB10, CUDA 13)

- **Offline parity oracle** (`scripts/reference_deltas.py`): the loaded A/B/scale reproduce
  the PEFT-exported reference deltas — 36/36 modules, `scaling=2.0`, **0.0 rel-err**.
- **`atlas-core` parser**: 11 unit tests (accept + every named reject) green.
- **Served, live**:
  - startup logs `LoRA adapter 'holo-tiny' installed on 6 layers (pool=117.0 MiB)`;
  - base output `"Paris."` → adapter output **differs** (delta is applied in the live
    decode path);
  - **graph == eager** (`ATLAS_LORA_EAGER=1`) — byte-identical, so deltas capture
    correctly inside CUDA graphs.

The test fixture (`test_data/lora-holo-tiny/`) is a **generated** PEFT adapter, deliberately
strong so its effect is unambiguous — no community adapter exists for Atlas's custom
NVFP4-packed bases, so a controllable fixture also exercises the reject/parity paths exactly.

## Deferred (documented cut lines)
- **Dense-FFN delta** (gate/up/down): types + install are in place; the compute insertion in
  `dense_ffn.rs` is not wired. The fixture targets FFN too, so enabling it is additive.
- **Prefix-cache warm-hit path** (`cache_skip_qkv.rs`) and **multi-seq decode** — until
  wired, cache-hit prefills and concurrency ≥2 silently skip the deltas. **Run the POC at
  batch size 1.**
- **q_proj / GDN / MoE / MLA targets**, **per-request adapter routing + multi-adapter**
  (M2), **`lora_bgmv` kernel**, **TP>1** (startup-guarded to `world_size=1`).

## Base-branch build note
`research/lora` predates the switch to the `cu*` driver API, so `cuda_backend`'s
`cudaMemcpy2DAsync` needs `libcudart` linked; `spark-runtime/build.rs` now does so and adds
the CUDA-13 SBSA lib path. On `main` (driver-API) this is a harmless no-op.
