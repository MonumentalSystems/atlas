# SPDX-License-Identifier: AGPL-3.0-only
# Re-export BOTH GDN AOT kernels in one deterministic run:
#   [bf16-out]  gdn_holo_0.{h,o}  — byte-compatible replacement for the original
#                                   (the original gdn_holo_0.o is NOT on disk anymore;
#                                   relinking libatlasgdn.so needs it re-exported)
#   [f32-out]   gdn_holo_1.{h,o}  — same kernel, o dtype = float32 (q/k/v stay bf16).
#                                   NAME IS LOAD-BEARING: gdn_shim.cpp (ATLAS_GDN_F32)
#                                   includes gdn_holo_1.h and calls
#                                   cute_dsl_gdn_holo_1_wrapper.
#
# PREREQ (gx10, /home/ms/flashinfer — a PRISTINE checkout; the June-30 local edits
# were reverted). Apply the in-repo patch stack IN ORDER (all four live next to
# this script):
#   1. patch flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py \
#        < delta_rule_sm120_aot_export.patch
#   2. patch -p1 < delta_rule_sm120_sm121a.patch    (GPUArch sm_120a -> sm_121a)
#   3. patch -p1 < delta_rule_sm120_f32_out.patch   (o_dtype plumbing + direct store)
#   ---- export VARIANT A here (f32-out, fp16 inverse) -> libatlasgdn_f32out.so ----
#   4. patch -p1 < delta_rule_sm120_f32_inverse.patch (fp32 WY inverse, f32 leg only)
#   ---- export VARIANT B here (f32-out + fp32 inverse) -> libatlasgdn_f32inv.so ----
# Without patch 3 the f32 leg raises "q/k/v dtypes must match..." and this script
# exports only the bf16 kernel.
#
# RUN (gx10, quiet GPU):
#   cd /home/ms/atlas/3rdparty_patches/gdn_aot
#   PYTHONPATH=/home/ms/flashinfer CUTE_DSL_ARCH=sm_121a \
#     /home/ms/spark-vllm-docker/.venv/bin/python gdn_export_f32.py
# Outputs land in /tmp/gdn_aot/.
import os
import torch

# Capture compiled CuTe kernels by wrapping cached_compile (same trick as gdn_export.py).
import flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache as cc
import flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 as dr

captured = []
_orig = cc.cached_compile


def _wrap(func, *a, **k):
    cf = _orig(func, *a, **k)
    captured.append(cf)
    return cf


cc.cached_compile = _wrap
dr.cached_compile = _wrap

from flashinfer.gdn_prefill import chunk_gated_delta_rule  # noqa: E402

T, Hqk, Hv, D = 2048, 16, 32, 128
dev = "cuda"
q = torch.randn(T, Hqk, D, dtype=torch.bfloat16, device=dev)
k = (
    torch.nn.functional.normalize(
        torch.randn(T, Hqk, D, dtype=torch.float32, device=dev), dim=-1
    ).to(torch.bfloat16)
)
v = torch.randn(T, Hv, D, dtype=torch.bfloat16, device=dev)
g = torch.rand(T, Hv, dtype=torch.float32, device=dev)
beta = torch.rand(T, Hv, dtype=torch.float32, device=dev).sigmoid()
cu = torch.tensor([0, T], dtype=torch.int64, device=dev)
h0 = torch.zeros(1, Hv, D, D, dtype=torch.float32, device=dev)
so = torch.zeros_like(h0)

os.makedirs("/tmp/gdn_aot", exist_ok=True)


def run_and_export(export_name: str, out_dtype: torch.dtype) -> torch.Tensor:
    """Run one compile+launch with the given output dtype, export the captured kernel.

    `export_name` is the EXACT header/symbol prefix (gdn_holo_0 / gdn_holo_1 —
    what gdn_shim.cpp includes and calls). The Holo shape compiles exactly one
    kernel; extras would collide with the shim's expectations, so fail loudly.
    """
    captured.clear()
    out = torch.zeros(T, Hv, D, dtype=out_dtype, device=dev)
    chunk_gated_delta_rule(q, k, v, g, beta, None, h0, True, cu, False, out, so)
    torch.cuda.synchronize()
    print(f"[{export_name}] ran at Holo shape (T={T} Hqk={Hqk} Hv={Hv} D={D}), "
          f"o dtype={out_dtype}; captured {len(captured)} kernel(s)")
    if len(captured) != 1:
        raise RuntimeError(
            f"{export_name}: expected exactly 1 compiled kernel, got {len(captured)} "
            "— the shim binds one wrapper symbol per export name"
        )
    captured[0].export_to_c("/tmp/gdn_aot", export_name, export_name)
    print(f"  EXPORTED -> /tmp/gdn_aot/{export_name}.{{h,o}}")
    return out


# Leg 1: bf16 output (regenerates gdn_holo_0.{h,o} for the relink).
o_bf16 = run_and_export("gdn_holo_0", torch.bfloat16)
state_bf16 = so.clone()

# Leg 2: f32 output (requires the f32-output flashinfer patch).
so.zero_()
try:
    o_f32 = run_and_export("gdn_holo_1", torch.float32)
    state_f32 = so.clone()
    # Sanity: f32 output must agree with the bf16 kernel to bf16 rounding.
    # Carried state (always f32):
    #   VARIANT A (f32_out patch only): the state path is untouched -> the two
    #     legs MUST be bit-identical (state max_abs == 0.0). Nonzero = the
    #     direct-store change leaked into the recurrence: STOP.
    #   VARIANT B (+ f32_inverse patch): the f32 leg computes the WY inverse in
    #     fp32 while the bf16 leg keeps fp16 -> state max_abs is legitimately
    #     NONZERO (that delta IS the fix); expect state cos ~ 1.
    cos = torch.nn.functional.cosine_similarity(
        o_bf16.float().flatten(), o_f32.flatten(), dim=0
    ).item()
    max_abs = (o_bf16.float() - o_f32).abs().max().item()
    state_max = (state_bf16 - state_f32).abs().max().item()
    state_cos = torch.nn.functional.cosine_similarity(
        state_bf16.flatten(), state_f32.flatten(), dim=0
    ).item()
    print(f"[check] o cos(bf16,f32)={cos:.6f} max_abs={max_abs:.6f} | "
          f"state max_abs={state_max:.6f} cos={state_cos:.9f} "
          f"(variant A: state max_abs MUST be 0.0; variant B: nonzero, cos~1)")
except RuntimeError as e:
    print("[f32 leg SKIPPED] flashinfer f32-output patch not applied:", str(e)[:200])
