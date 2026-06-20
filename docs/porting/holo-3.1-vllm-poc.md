# Holo 3.1 Atlas POC Notes

Date: 2026-06-19

## Model

- HF model: `Hcompany/Holo-3.1-35B-A3B-NVFP4`
- Local path used for Atlas/vLLM POC: `/tank/holo-bf16kv-test`
- Atlas served name: `holo3.1-atlas-poc`
- Atlas endpoint used during validation: `http://127.0.0.1:8890`

Holo 3.1 identifies as `qwen3_5_moe` in HF config, but is a VLM with
`image_token_id = 248056`. Atlas rewrites that case to `holo3_1_moe` so it can
pick the Holo kernel target, template, behavior, and parser defaults without
changing generic Qwen3.5 dispatch.

## vLLM Reference

Production service:

- systemd unit: `/etc/systemd/system/holo.service`
- launch script: `/home/ms/spark-vllm-docker/deploy/run-holo-aeon.sh`
- container: `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix`
- port: `8008`

Relevant vLLM flags:

```bash
--attention-backend flashinfer
--gpu-memory-utilization 0.34
--max-model-len 131072
--max-num-seqs 32
--max-num-batched-tokens 16384
--mamba_ssm_cache_dtype float32
--enable-prefix-caching
--tool-call-parser qwen3_coder
--reasoning-parser qwen3
--limit-mm-per-prompt '{"image": 3, "video": 0}'
--speculative-config '{"method":"dflash", ...}'
```

Important distinction: vLLM pins the SSM state/cache dtype to FP32, but keeps
Holo's SSM projection weights as quantized linear layers. It does not require
Atlas' current BF16 expansion of SSM projection weights for correctness.

The current production vLLM script uses DFlash:

```text
model=/tank/hf/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash/snapshots/a6ab3a277f856d91c43f28711611e7929073d56d
num_speculative_tokens=4
attention_backend=flash_attn
```

Atlas Holo target now declares the matching `[dflash]` pairing so `--dflash`
can resolve the drafter without a CLI `--draft-model`.

## Atlas Demo Command

Stable 64K/C12 POC command:

```bash
env \
  RUST_BACKTRACE=1 \
  RUST_LOG=info \
  ATLAS_TARGET_HW=gb10 \
  ATLAS_TARGET_MODEL=holo-3.1-35b-a3b \
  ATLAS_TARGET_QUANT=nvfp4 \
  ATLAS_FAST_LOAD_PREFETCH_SHARDS=1 \
  ATLAS_HOLO_LOW_MEMORY_MOE=1 \
  /home/ms/atlas/target/release/spark serve \
    --model-from-path /tank/holo-bf16kv-test \
    --model-name holo3.1-atlas-poc \
    --port 8890 \
    --bind 127.0.0.1 \
    --max-seq-len 65536 \
    --max-num-seqs 12 \
    --max-batch-size 12 \
    --max-prefill-tokens 4096 \
    --kv-cache-dtype bf16 \
    --lm-head-dtype bf16 \
    --gpu-memory-utilization 0.60 \
    --oom-guard-mb 256 \
    --ssm-cache-slots 0 \
    --ssm-checkpoint-interval 0 \
    --enable-prefix-caching false \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --fast-load-prefetch-shards \
    --vision-max-pixels 262144
```

Do not run DGX Holo over `gpu-memory-utilization=0.80`; `0.88` was called out
as an absolute stability edge during the POC.

## Memory Results

Measured from Atlas startup logs on DGX:

| Mode | GPU util | Pre-KV | KV | Notes |
| --- | ---: | ---: | ---: | --- |
| Initial stable path | 0.75 | 70.2 GB | 20.1 GB | BF16 attention/SSM, MoE prefill copies enabled |
| Low-memory MoE | 0.50 | 53.0 GB | 6.9-7.0 GB | `ATLAS_HOLO_LOW_MEMORY_MOE=1` |
| 64K/C12 low-memory MoE | 0.60 | 53.1-53.4 GB | 16.0-16.3 GB | 837k-852k max KV tokens |
| Native FP8 SSM test | 0.50 | 50.5 GB | 9.4 GB | starts, but fails real prefill |

The big memory win is skipping Holo ModelOpt MoE transpose/predequant prefill
copies. This saved about 17 GB while keeping the stable BF16 SSM path.

At `gpu-memory-utilization=0.50`, 64K context only reserved enough KV for three
concurrent 64K sequences. At `0.60`, the allocator accepted `max_batch=12` with
64K context:

```text
KV cache: 121.6 GB total × 60% util = 73.0 GB budget;
53.4 GB pre-KV + 3.6 GB reserve -> 16.0 GB for KV ->
52338 blocks × 16 tok/block = 837408 max KV tokens
```

Detached demo server command should be launched with `setsid -f ... </dev/null`
in this runner. Plain `nohup ... &` was reaped before CUDA initialization, while
the same command stayed up when attached to a foreground session.

## Vision Results

Real test image: `/tank/ops_ai_agent.png` (`1254x1254` PNG).

Before adding `--vision-max-pixels 262144`:

- Vision tokens/patches encoded: `1521`
- Prompt tokens: `1546`
- TTFT: about `102.9s`

After adding `--vision-max-pixels 262144`:

- Vision tokens/patches encoded: `256`
- Prompt tokens: about `280-281`
- TTFT: about `4.4-4.7s`
- End-to-end image request: about `10s`

Same image on the 64K/C12 server:

- Prompt tokens: `284`
- TTFT: `4.69s`
- End-to-end image request: `8.86s`
- Decode: `53.6 tok/s`

This matches the vLLM-style `max_pixels` behavior and is required for usable
Holo vision latency.

## Concurrency Results

Short-prompt probes on the 64K/C12 server (`gpu-memory-utilization=0.60`,
thinking enabled, `max_tokens=96`):

| Concurrency | Failures | Wall | Completion tokens | Aggregate decode |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0 | 23.87s | 900 | 37.7 tok/s |
| 12 | 0 | 42.14s | 1969 | 46.7 tok/s |

Scheduler behavior: decode batches correctly, but short prefills are still
effectively serialized. Per-request throughput drops with concurrency while
aggregate decode stays near the single-stream ceiling.

Longer-output probes (`max_tokens=360`) on the same 64K/C12 server:

| Concurrency | Failures | Wall | Completion tokens | Reasoning tokens | Aggregate decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0 | 38.59s | 1904 | 1037 | 49.3 tok/s |
| 8 | 0 | 84.87s | 4072 | 2101 | 48.0 tok/s |

The C4/C8 ceiling is not a KV capacity issue. It is decode-path efficiency.

## Long Context Result

A synthetic prompt at `63,023` tokens completed on the stable BF16 SSM path with
`max-seq-len=65536`.

- TTFT: `145.4s`
- Decode: `16.1 tok/s` for the short completion
- Effective long-prefill rate: about `435 prompt tok/s`

This proves the 64K context capacity path, but the long-context prefill speed is
now the clearest optimization target. The slow section is the repeated 4096-token
GDN/SSM chunked prefill path.

## Optimization Leads

Measured against the vLLM reference, Atlas currently has three clear gaps:

1. Initial prefill admission is serialized. `phase_start_prefills` loops over
   new requests and calls `start_chunked_prefill` one request at a time. Short
   prompts become active immediately and never hit Q12 `prefill_batch_chunk`.
   In the C8 probe, eight 34-token prompts each paid about `310ms` prefill
   admission before batched decode started.
2. Multi-sequence SSM decode still loops per sequence for the recurrent mixer.
   `Qwen3SsmLayer::decode_multi_seq_inner` batches only part of the MoE work;
   for `N>=4` it deliberately falls back to per-token MoE based on older
   Qwen3.5 measurements. Holo needs a fresh benchmark here.
3. Multi-sequence decode disables CUDA graphs because SSM state pointers are
   embedded in graph args, and final LM head is still one GEMV per padded row.
   vLLM's C4/C8 numbers are also using DFlash, so raw Atlas decode should not
   be compared to vLLM+DFlash as if they were equivalent.

Priority order:

1. Enable and validate Holo+DFlash on Atlas using
   `z-lab/Qwen3.5-35B-A3B-DFlash`, starting with vLLM-equivalent
   `--dflash-gamma 5` for four speculative tokens.
2. Add a scheduler fast path to batch initial same-length short prefills into
   Q12 `prefill_batch_chunk` instead of serial `start_chunked_prefill`.
3. Re-benchmark SSM `N>=4` MoE batching for Holo and add an opt-in flag if
   grouped/prefill MoE wins on this model.
4. Fix native FP8 SSM prefill and evaluate whether it improves long-context
   prefill enough to justify replacing the stable BF16 SSM path.

## NFS Load Path

Atlas now has an explicit fast-loader prefetch/readahead knob:

- CLI: `--fast-load-prefetch-shards`
- env: `ATLAS_FAST_LOAD_PREFETCH_SHARDS=1`

With the Holo checkpoint on `/tank`, startup logs confirmed prefetch hints for
all three safetensor shards:

- `model-00001-of-00003.safetensors` - 9.32 GB
- `model-00002-of-00003.safetensors` - 9.32 GB
- `model-00003-of-00003.safetensors` - 3.43 GB

## Native FP8 SSM Status

`ATLAS_HOLO_NATIVE_FP8_SSM=1` routes Holo ModelOpt SSM projections through
Atlas' native FP8 SSM loader. This is closer to vLLM's weight handling and
reduces pre-KV from `53.0 GB` to `50.5 GB`, but it is not stable yet.

Observed failure on real image request:

```text
GDN prefill: FLA chunked path ACTIVE
Prefill chunk layer 36 failed:
CUDA_ERROR_ILLEGAL_ADDRESS (700)
grid=[4,5,1], block=[128,1,1]
```

Saved log from this run:

```text
/tmp/holo-atlas-lowmem-native-ssm-fail.log
```

Likely next investigation: compare Atlas' GDN FLA prefill dispatch against the
vLLM AEON path. vLLM selects between FlashInfer, CuteDSL, and Triton/FLA GDN
prefill backends; Atlas is currently taking its own FLA chunked GDN path.

## Known Gaps

- Tiny text prompts can still over-generate after the correct answer. Example:
  `2+2` returns `4` and then continues. This is a stop/template cleanup issue,
  separate from Holo memory and vision support.
- Native FP8 SSM is implemented as an opt-in diagnostic path only. Keep the
  stable demo on BF16 SSM until the layer-36 prefill failure is fixed.
- WeightStore source freeing may still be worth doing, but the largest measured
  memory issue was MoE prefill copies, not stale source tensors.

## Verification Run

Commands run during the POC:

```bash
cargo fmt --all
ATLAS_SKIP_BUILD=1 cargo test -p spark-model vision_preprocess --lib
ATLAS_SKIP_BUILD=1 cargo check -p spark-server --no-default-features --features cuda
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server --no-default-features --features cuda
```

Stable low-memory endpoint was restored and health-checked after the failed
native FP8 SSM experiment.
