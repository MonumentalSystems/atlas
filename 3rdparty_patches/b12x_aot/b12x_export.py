# SPDX-License-Identifier: AGPL-3.0-only
#
# AOT-export the FlashInfer b12x DYNAMIC MoE kernel at the Holo-3.1-35B-A3B shape
# (E=256, hidden=2048, moe_intermediate=512, top_k=8) to a C-ABI object the Atlas shim
# dlopens. Clone of 3rdparty_patches/gdn_aot/gdn_export.py, adapted for b12x.
#
# ONE export serves ALL prefill token counts: num_tokens / max_rows / rows_padded /
# max_tasks / max_phys_tiles are runtime Int32 in the dynamic kernel (see the patched
# _DynamicMoELaunch.__call__); `mac` (max_active_clusters) is a Constexpr baked from the
# GB10 probe. Prereqs (parent recipe, gx10):
#   1. venv: torch cu13 aarch64 + nvidia-cutlass-dsl{,-libs-base,-libs-cu13}==4.4.2
#      (4.5.x emits bad PTX on sm121). CUDA>=13.
#   2. cutlass-dsl 4.4.2 sm_121a seds (run.sh:57-83) + wipe __pycache__.
#   3. git apply 3rdparty_patches/b12x_aot/b12x_moe_aot_export.patch on /home/ms/flashinfer
#      (+ drop the stale sm120_moe_dispatch_context import from blackwell_sm12x/__init__.py).
# Run:
#   CUTE_DSL_ARCH=sm_121a PYTHONPATH=/home/ms/flashinfer python b12x_export.py
#
# HARD-ASSERTS E==256 (the brief's E=512 is WRONG — do NOT bake 512 into geometry).
import argparse
import os

import torch

# ── Strip --enable-tvm-ffi from the cute compile flags (keep --opt-level 2) and capture
#    the compiled executor, exactly as gdn_export.py wraps cached_compile. ────────────
import cutlass.cute as cute  # noqa: E402

_orig_compile = cute.compile
_captured = []


def _wrap_compile(*a, **k):
    # Drop the tvm-ffi flag the AOT path can't honour on these kernels (export_to_c
    # renders a plain C shim). Everything else (incl. --opt-level 2) is preserved.
    opts = k.get("options")
    if isinstance(opts, (list, tuple)):
        k["options"] = [o for o in opts if "tvm-ffi" not in str(o)]
    cf = _orig_compile(*a, **k)
    _captured.append(cf)
    return cf


cute.compile = _wrap_compile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/b12x_aot")
    ap.add_argument("--name", default="b12x_dyn_0")
    ap.add_argument("--max-tokens", type=int, default=1024, help="capacity to dump geometry for")
    ap.add_argument("--jit-smoke", action="store_true", help="P1: run the JIT kernel once, no export")
    # Design-B decode follow-up surface: left UNEXECUTED here so decode is a pure add later.
    ap.add_argument("--static-m", default="", help="(deferred) static m1,m2,m3 for decode")
    args = ap.parse_args()

    E, H, I, TOPK = 256, 2048, 512, 8
    assert E == 256, f"Holo-3.1-35B-A3B is E=256 (brief's E=512 is WRONG); got {E}"

    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import _get_dynamic_kernel

    # Drive the dynamic kernel at the Holo dims. share_input_across_experts=False,
    # input_scales_are_reciprocal=False, fast_math=True, activation="silu".
    launcher = _get_dynamic_kernel(
        E=E,
        m=args.max_tokens,  # compile hint; the kernel keeps num_tokens runtime
        k=H,
        n=I,
        num_topk=TOPK,
        max_rows=args.max_tokens * TOPK,
        input_scales_are_reciprocal=False,
        fast_math=True,
        activation="silu",
        activation_precision="fp4",
        share_input_across_experts=False,
    )
    print(f"b12x dynamic kernel built at E={E} H={H} I={I} top_k={TOPK}")

    if args.static_m:
        print(f"[deferred] --static-m={args.static_m} ignored (Design-B decode follow-up)")

    if args.jit_smoke:
        print("--jit-smoke: kernel compiled OK (P1 numeric smoke lives in proof scripts).")
        return

    if not _captured:
        raise SystemExit("no compiled executor captured — did the launcher compile? check patch")
    cf = _captured[-1]
    os.makedirs(args.out, exist_ok=True)
    print(f"exporting {type(cf).__name__} export_to_c={hasattr(cf, 'export_to_c')}")
    cf.export_to_c(args.out, args.name, args.name)
    print(f"EXPORTED -> {args.out}/{args.name}.{{h,o}}")

    # Dump the frozen task geometry ints for the chosen capacity + the memref/arg order
    # (gdn_dump_meta.py pattern) so b12x_shim.cpp can be FROZEN against the real .h at P3.
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _dynamic_task_geometry,
        )

        geo = _dynamic_task_geometry(E=E, k=H, n=I, num_topk=TOPK, max_tokens=args.max_tokens)
        with open(os.path.join(args.out, f"{args.name}.geom.txt"), "w") as fh:
            fh.write(f"E={E} H={H} I={I} top_k={TOPK} max_tokens={args.max_tokens}\n")
            fh.write(repr(geo) + "\n")
        print(f"geometry dumped -> {args.name}.geom.txt")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] geometry dump skipped: {str(e)[:200]}")


if __name__ == "__main__":
    main()
