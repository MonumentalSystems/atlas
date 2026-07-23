# SPDX-License-Identifier: AGPL-3.0-only
#
# P0 (GO/NO-GO) — SFB atom byte-identity proof, run FIRST (no model, no export).
#
# THE #1 numeric risk: does Atlas's `pack_weight_sfb` (CUTLASS SM120 blockscaled SFB
# swizzle atom, ue4m3) produce byte-IDENTICAL output to FlashInfer's b12x scale swizzle
# (`convert_sf_to_mma_layout` over `fp4_quantize(is_sf_swizzled_layout=True)` storage)?
#
#   PASS  => keep `SfbStrategy::ConcatReuse` (default). The repack in b12x_scales.rs /
#            b12x_weights.rs ships AS WRITTEN — no Rust change.
#   FAIL  => the first-differing offset localizes the tile-order mismatch. Supply the
#            FI-matching swizzle into `swizzle_sfb`'s `RebuildFromRaw` arm
#            (crates/spark-model/src/layers/moe/b12x_scales.rs); Stage-(a) assembly/bake
#            is UNTOUCHED. Do NOT trust the concat repack until this passes.
#
# This is pure host/Python except the two GPU swizzle calls. Atlas side is driven through
# a tiny ctypes shim over pack_weight_sfb (build with pack_sfb_driver.cpp, or reuse the
# atlas_cutlass_pack_weight_sfb symbol from the Atlas cutlass lib).
import numpy as np
import torch


def ramp_e4m3(n, k16):
    # Ramp e4m3 bytes across the finite-normal range 0x08..0x7E in an [n, k16] tile.
    lo, hi = 0x08, 0x7E
    vals = (np.linspace(lo, hi, n * k16).astype(np.uint8)).reshape(n, k16)
    return vals


def fi_swizzle(scale_u8, n, k):
    # FlashInfer reference: interpret the same bytes as [n, k/16] float8_e4m3fn and run
    # its block-scale interleave / convert_sf_to_mma_layout. (Import path per the pinned
    # FlashInfer rev a671c02.)
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout  # real module path  # type: ignore

    sf = torch.from_numpy(scale_u8.view(np.uint8)).view(torch.float8_e4m3fn).cuda()
    return convert_sf_to_mma_layout(sf, m=n, k=k)  # sig is (sf, m, k, num_groups=1, sf_vec_size=16).cpu().view(torch.uint8).numpy().ravel()


def atlas_swizzle(scale_u8, n, k):
    # Atlas reference: pack_weight_sfb(scale_in, scale_out, n, k). Drives the same
    # atlas_cutlass_pack_weight_sfb the grouped path uses (n=N, k=K, [K/16,N] input).
    # ============================ P0 BLOCKER (read before running) ============================
# This proof is NOT yet runnable end-to-end. Two things must be supplied first:
#   1. `atlas_pack_sfb` (imported below) does NOT exist — it must be a thin ctypes/cdylib
#      wrapper exposing Atlas's COMPILED CUTLASS symbol `atlas_cutlass_pack_weight_sfb`
#      (crates/spark-runtime/src/cutlass/pack.rs -> the C fn). Build a tiny cdylib that
#      re-exports it, or link spark-runtime's cutlass staticlib into a ctypes-loadable .so.
#   2. The FI side (`fi_swizzle`) must first quantize a REAL weight tile via
#      flashinfer `fp4_quantize(w, is_sf_swizzled_layout=True)` and take THAT swizzled SF as
#      input to convert_sf_to_mma_layout (the docstring is explicit: convert_sf expects the
#      fp4_quantize-swizzled 2D SF, not raw ramp bytes). Comparing raw ramp bytes is wrong.
# Preferred P0 = FUNCTIONAL: feed one expert's Atlas-packed weights+SF through b12x vs Atlas
#   grouped GEMM, compare output cos (tolerance, atomic-scatter). That sidesteps atom-layout
#   derivation entirely but needs b12x running (cutlass-dsl 4.4.2 + sm_121a patches).
# ==========================================================================================
    import atlas_pack_sfb  # thin ctypes wrapper around atlas_cutlass_pack_weight_sfb

    return atlas_pack_sfb.pack(scale_u8.tobytes(), n, k)


def check(n, k):
    k16 = k // 16
    scale = ramp_e4m3(n, k16)
    a = atlas_swizzle(scale, n, k)
    f = fi_swizzle(scale, n, k)
    a = np.frombuffer(a, dtype=np.uint8)
    f = np.frombuffer(f, dtype=np.uint8)
    if a.shape != f.shape:
        print(f"  [{n}x{k}] SHAPE MISMATCH atlas={a.shape} fi={f.shape} -> NO-GO")
        return False
    diff = np.nonzero(a != f)[0]
    if diff.size == 0:
        print(f"  [{n}x{k}] byte-IDENTICAL ({a.size} bytes) -> PASS")
        return True
    off = int(diff[0])
    print(f"  [{n}x{k}] MISMATCH: {diff.size} bytes differ, first@offset {off} "
          f"(atlas=0x{a[off]:02x} fi=0x{f[off]:02x}) -> NO-GO; localize tile order")
    return False


if __name__ == "__main__":
    print("P0 SFB atom A/B (ramp e4m3 0x08..0x7E):")
    ok = True
    # Laguna-S-2.1 w13 orientation: n=2I=2048, k=H=3072; w2: n=H=3072, k=I=1024.
    ok &= check(2048, 3072)
    ok &= check(3072, 1024)
    print("=== P0", "PASS: ConcatReuse ships as written ===" if ok
          else "NO-GO: implement RebuildFromRaw ===")
