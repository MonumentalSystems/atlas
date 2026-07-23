# SPDX-License-Identifier: AGPL-3.0-only
#
# AOT-export the FlashInfer b12x DYNAMIC MoE kernel at the Laguna-S-2.1 shape
# (E=256, hidden=3072, moe_intermediate=1024, top_k=10) to a C-ABI object the Atlas shim
# dlopens. Clone of 3rdparty_patches/gdn_aot/gdn_export.py, adapted for b12x.
# (Ported from the Holo E=256/H=2048/I=512/top_k=8 export — geometry retargeted.)
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
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402

_orig_compile = cute.compile
_captured = []


def _resolve_str_annotations(fn):
    """moe_dispatch.py declares `from __future__ import annotations`, so the kernel
    function's `__annotations__` are STRINGS. The CuTe C-header generator dispatches
    numeric scalar args via `isinstance(annotation, NumericMeta)` and the stream via
    `issubclass(annotation, cuda.CUstream)` — both fail on strings, so the exported
    wrapper would omit / reject the 5 runtime Int32s (num_tokens, max_rows, rows_padded,
    max_tasks, max_phys_tiles) and the CUstream. Rewrite those specific annotations back
    to real classes BEFORE compile (the C-header args are frozen during compile, not at
    export). Pointer/Tensor args dispatch by runtime value type, so leave them as strings.
    """
    import cuda.bindings.driver as _cuda_aot  # noqa: PLC0415

    resolve = {
        "cutlass.Int32": cutlass.Int32,
        "cutlass.Constexpr": cutlass.Constexpr,
        "_cuda_aot.CUstream": _cuda_aot.CUstream,
    }
    # The dynamic prefill kernel is a `_DynamicMoELaunch` INSTANCE (a callable object),
    # not a plain function — its runtime signature (what inspect.getfullargspec reads) is
    # `type(fn).__call__.__annotations__`. Resolve both the object's own annotations and
    # its `__call__` method's, whichever carries the string annotations.
    dicts = []
    own = getattr(fn, "__annotations__", None)
    if isinstance(own, dict):
        dicts.append(own)
    call = getattr(type(fn), "__call__", None)
    call_ann = getattr(call, "__annotations__", None)
    if isinstance(call_ann, dict):
        dicts.append(call_ann)
    for ann in dicts:
        for name, a in list(ann.items()):
            if isinstance(a, str) and a in resolve:
                ann[name] = resolve[a]


def _wrap_compile(*a, **k):
    # Drop the tvm-ffi flag the AOT path can't honour on these kernels (export_to_c
    # renders a plain C shim). Everything else (incl. --opt-level 2) is preserved.
    # moe_dispatch passes options as a SPACE-SEPARATED STRING ("--opt-level 2
    # --enable-tvm-ffi"), not a list — strip the flag from either form or the compile
    # yields a TVMFFIJitCompiledFunction whose export_to_c has the wrong signature.
    opts = k.get("options")
    if isinstance(opts, str):
        k["options"] = opts.replace("--enable-tvm-ffi", "").replace("  ", " ").strip()
    elif isinstance(opts, (list, tuple)):
        k["options"] = [o for o in opts if "tvm-ffi" not in str(o)]
    if a:
        _resolve_str_annotations(a[0])  # the kernel fn being compiled
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

    E, H, I, TOPK = 256, 3072, 1024, 10
    assert E == 256, f"Laguna-S-2.1 is E=256; got {E}"
    assert (H, I, TOPK) == (3072, 1024, 10), (
        f"Laguna-S-2.1 geometry is H=3072 I=1024 top_k=10; got H={H} I={I} top_k={TOPK}"
    )

    # Establish a live CUDA driver context BEFORE building the kernel so the baked
    # `mac` Constexpr comes from the real get_max_active_clusters(1) probe. Without a
    # current context the DSL silently falls back to sm_count (CUDA_ERROR_INVALID_CONTEXT);
    # on GB10 that equals the probe (48) but this persistent kernel has grid-wide barriers,
    # so an over-large mac elsewhere could deadlock — don't rely on the coincidence.
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import _get_dynamic_kernel

    # Drive the dynamic kernel at the Laguna dims. share_input_across_experts=False,
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

    # Dump the frozen workspace geometry for the chosen capacity so b12x_shim.cpp can size
    # its cached workspace to match allocate_sm120_dynamic_workspace exactly. The kernel
    # takes num_tokens/max_rows/rows_padded/max_tasks/max_phys_tiles as RUNTIME Int32; only
    # num_tokens varies per call — the other four are these capacity constants.
    # Signature: _dynamic_task_geometry(state_E, n, routed_rows, *, tile_m, tile_n).
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _NVFP4_BLOCK_SIZE,
            _align_up,
            _dynamic_task_geometry,
            _level_tile_m,
            _level_tile_n,
        )

        routed_rows = args.max_tokens * TOPK  # each token routes to top_k experts
        tile_m = _level_tile_m("fp4")
        tile_n = _level_tile_n("fp4")
        physical_tiles, gate_tile_cnt, max_tasks = _dynamic_task_geometry(
            E, I, routed_rows, tile_m=tile_m, tile_n=tile_n
        )
        rows_padded = physical_tiles * tile_m
        cols_pad_k = _align_up(H // _NVFP4_BLOCK_SIZE, 4)
        with open(os.path.join(args.out, f"{args.name}.geom.txt"), "w") as fh:
            fh.write(f"E={E} H={H} I={I} top_k={TOPK} max_tokens={args.max_tokens}\n")
            fh.write(
                f"routed_rows={routed_rows} tile_m={tile_m} tile_n={tile_n} "
                f"physical_tiles={physical_tiles} gate_tile_cnt={gate_tile_cnt} "
                f"max_tasks={max_tasks} rows_padded={rows_padded} cols_pad_k={cols_pad_k}\n"
            )
        print(
            f"geometry: physical_tiles={physical_tiles} max_tasks={max_tasks} "
            f"rows_padded={rows_padded} cols_pad_k={cols_pad_k} -> {args.name}.geom.txt"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] geometry dump skipped: {str(e)[:200]}")


if __name__ == "__main__":
    main()
