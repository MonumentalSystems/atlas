#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

: "${VLLM_SRC:?set VLLM_SRC to a vLLM checkout}"
: "${OUT_DIR:?set OUT_DIR to the output directory}"

mkdir -p "$OUT_DIR"
torch_include="$(python3 -c 'from torch.utils.cpp_extension import include_paths; print(include_paths()[0])')"

nvcc -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr -arch=sm_121 \
  -shared -Xcompiler=-fPIC,-static-libstdc++,-static-libgcc \
  -I"$VLLM_SRC/csrc" -I"$torch_include" \
  "$(dirname "$0")/atlas_marlin_moe.cu" \
  -o "$OUT_DIR/libatlas_marlin_moe.so" -lcudart

nm -D "$OUT_DIR/libatlas_marlin_moe.so" | grep ' atlas_marlin_moe_'
