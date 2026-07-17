# Atlas Marlin MoE AOT bridge

This is the exact kernel family selected by the vLLM 270.1 tok/s reference
run on GB10 (`NvFp4 MoE backend: MARLIN`). It is not the FlashInfer b12x path.

The bridge specializes vLLM's Marlin template to Qwen3.6-35B-A3B decode:
BF16 activations, NVFP4 weights, group size 16, block-M 8, and 128x128
thread tiles. It exposes raw CUDA pointers through a C ABI, so Atlas does not
link LibTorch. Weight repacking runs once at load time; route alignment and two
Marlin GEMMs are graph-capturable decode operations.

Build in the CUDA 13.2 Atlas container, with a vLLM source checkout mounted at
`/vllm`:

```bash
VLLM_SRC=/vllm OUT_DIR=/out \
  bash /atlas/3rdparty_patches/marlin_moe_aot/build.sh
```

The implementation included by the bridge originates from vLLM/Marlin under
Apache-2.0. Atlas's adapter code is AGPL-3.0-only.
