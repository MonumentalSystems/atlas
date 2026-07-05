// SPDX-License-Identifier: AGPL-3.0-only
//
// Byte-wise 2D transpose — the offline equivalent of the runtime's
// `QuantizedWeight::transpose_for_gemm`.
//
// NVFP4 prefill-resident layout stores both the packed weights and the FP8
// block scales column-major (transposed): packed `[N, K/2] -> [K/2, N]` and
// scale `[N, K/16] -> [K/16, N]`. The runtime does this on the GPU (D2H -> CPU
// loop -> H2D); we do the identical byte transpose on CPU so the on-disk record
// is already resident-form and nothing is transformed at fetch time (invariant
// D). Because both are plain byte transposes, a transpose of a transpose is the
// exact original — which is how the builder's round-trip self-check verifies it.

/// Transpose a `rows x cols` row-major byte matrix into a `cols x rows`
/// row-major byte matrix. `src.len()` must equal `rows * cols`.
///
/// Uses cache-blocking so multi-hundred-MB expert tensors don't thrash — a
/// naive element loop over `[2048, 256]` scatters writes across the whole
/// output on every source row.
pub fn transpose_bytes(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(src.len(), rows * cols, "transpose_bytes: size mismatch");
    let mut dst = vec![0u8; rows * cols];
    const B: usize = 64; // block edge; 64x64 tile fits comfortably in L1.
    let mut r0 = 0;
    while r0 < rows {
        let r1 = (r0 + B).min(rows);
        let mut c0 = 0;
        while c0 < cols {
            let c1 = (c0 + B).min(cols);
            for r in r0..r1 {
                let src_row = r * cols;
                for c in c0..c1 {
                    // dst[c, r] = src[r, c]
                    dst[c * rows + r] = src[src_row + c];
                }
            }
            c0 = c1;
        }
        r0 = r1;
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        let mut dst = vec![0u8; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                dst[c * rows + r] = src[r * cols + c];
            }
        }
        dst
    }

    #[test]
    fn known_2x3() {
        // [[0,1,2],[3,4,5]] -> [[0,3],[1,4],[2,5]]
        let src = [0u8, 1, 2, 3, 4, 5];
        let t = transpose_bytes(&src, 2, 3);
        assert_eq!(t, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn blocked_matches_naive_nonsquare() {
        // Odd dims that straddle the 64-block boundary in both axes.
        let rows = 130;
        let cols = 70;
        let src: Vec<u8> = (0..rows * cols).map(|i| (i * 31 + 7) as u8).collect();
        assert_eq!(transpose_bytes(&src, rows, cols), naive(&src, rows, cols));
    }

    #[test]
    fn transpose_is_involutive() {
        // A3B gate shape: [512, 1024] packed. Transpose twice == original.
        let rows = 512;
        let cols = 1024;
        let src: Vec<u8> = (0..rows * cols).map(|i| (i % 251) as u8).collect();
        let t = transpose_bytes(&src, rows, cols);
        assert_eq!(t.len(), src.len());
        let back = transpose_bytes(&t, cols, rows);
        assert_eq!(back, src, "double transpose must recover the original");
    }

    #[test]
    fn a3b_scale_shape() {
        // gate scale [512, 128] -> [128, 512]
        let rows = 512;
        let cols = 128;
        let src: Vec<u8> = (0..rows * cols).map(|i| (i * 13) as u8).collect();
        let t = transpose_bytes(&src, rows, cols);
        assert_eq!(t.len(), rows * cols);
        assert_eq!(transpose_bytes(&t, cols, rows), src);
    }
}
