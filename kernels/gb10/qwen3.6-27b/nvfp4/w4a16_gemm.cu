// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMM — 35B model shadow.
//
// Optimizations:
// - w4a16_gemm_t: cp.async 2-stage double-buffered pipeline (overlaps next tile
//   loads with current tile compute), prmt BF16 packing, BP_PAD bank conflict fix
// - Vectorized uint4 (128-bit) B_packed loads
// - Both-nibble extraction from packed bytes
// - N_TILE=128 for reduced A bandwidth

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Standard E4M3 (1-4-3, bias 7) decode via pure bit-math. On real NVIDIA this is
// byte-identical to (float)__nv_fp8_e4m3; on SCALE/gfx1151 the built-in
// __nv_fp8_e4m3->float decode is a NON-STANDARD narrow format (verified: 1.0->1.5,
// 0.5->1.0, 3.5->2.75) which mismatches the standard E4M3 scales written by
// quantize_bf16_to_nvfp4 -> corrupts every block scale. Use this to match the encoder.
// HIP/gfx1151 (hipcc, __HIP_PLATFORM_AMD__) shares the same software path: it has
// no `cvt.rn.satfinite.e4m3x2.f32` PTX codegen, so it also routes E4M3 encode/decode
// through these pure-bit-math helpers (same recipe the port's strix-hip-real uses).
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;            // subnormal m*2^-9
    else if (e == 15u && m == 7u) v = 0.0f;                            // NaN -> 0
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20)); // 2^(e-7)*(1+m/8)
    return s ? -v : v;
}

__device__ __forceinline__ unsigned char scl_enc_fp8(float v) {
    if (v != v) return 0x7F;
    unsigned int bb = __float_as_uint(v); unsigned int sign = (bb >> 31) & 1u;
    int e = (int)((bb >> 23) & 0xFF) - 127; unsigned int man = bb & 0x7FFFFFu;
    int ee = e + 7; unsigned int em;
    if (ee < 1) { ee = 0; em = 0; if (e >= -10) { float a = v < 0 ? -v : v; em = (unsigned int)(a / 0.001953125f + 0.5f); if (em > 7u) em = 7u; } }
    else if (ee > 15) { ee = 15; em = 6; }
    else { em = (man + (1u << 19)) >> 20; if (em > 7u) { em = 0; ee++; if (ee > 15) { ee = 15; em = 6; } } }
    return (unsigned char)((sign << 7) | ((unsigned)ee << 3) | em);
}
#endif

// SCALE/gfx1151: the `cvt.rn.satfinite.e4m3x2.f32` inline PTX has no codegen
// (SCALE lacks the internal __nv_cvt_floatraw_to_fp8 lowering helper).
// __nv_cvt_float_to_fp8 is NVIDIA's own documented intrinsic with identical
// SATFINITE+E4M3 semantics — numerically exact, not an approximation. The
// #else branch is the verbatim original PTX, so with __forceinline__ at -O3
// the NVIDIA codegen is byte-identical (zero NVFP4/FP8 regression). SCALE
// defines __SCALE__ (not __HIP_PLATFORM_AMD__) in the device pass.
// PTX `cvt.e4m3x2.f32 d,a,b`: d hi-byte = e4m3(a), lo-byte = e4m3(b).
__device__ __forceinline__ unsigned short atlas_cvt_e4m3x2_f32(float a_hi, float b_lo) {
#if defined(__SCALE__)
    unsigned a8 = (unsigned)scl_enc_fp8(a_hi);
    unsigned b8 = (unsigned)scl_enc_fp8(b_lo);
    return (unsigned short)((a8 << 8) | (b8 & 0xFFu));
#elif defined(__HIP_PLATFORM_AMD__)
    // gfx1151 has no e4m3x2.f32 PTX; same software bit-math as SCALE / the
    // port's strix-hip-real atlas_cvt_e4m3x2_f32 (numerically exact SATFINITE E4M3).
    unsigned a8 = (unsigned)scl_enc_fp8(a_hi);
    unsigned b8 = (unsigned)scl_enc_fp8(b_lo);
    return (unsigned short)((a8 << 8) | (b8 & 0xFFu));
#else
    unsigned short d;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(a_hi), "f"(b_lo));
    return d;
#endif
}

// SCALE/gfx1151 has no codegen for mma.sync.m16n8k32.e4m3. Proven
// bit-exact replacement (scripts/scale-probe/e4m3_mma_helper_equiv.cu,
// validated on GB10: max|ref-cand|=0.0000): intra-warp-group __shfl
// repack of the e4m3 m16n8k32 fragments -> dequant -> 2x
// mma.m16n8k16.bf16 (K split 0..15 / 16..31). #else is the verbatim
// original e4m3 PTX so __forceinline__ keeps NVIDIA codegen
// byte-identical (zero NVFP4/FP8 regression).
//
// HIP/gfx1151 NOTE: there is intentionally NO __HIP_PLATFORM_AMD__ branch here.
// The AMD WMMA port (`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`, port branch
// kernels/strix-hip-real/qwen3.6-27b/nvfp4/w4a16_gemm.cu) restructures the whole
// kernel body — n8→n16 fragment tiling, v8f accumulators, synchronous smem copies
// replacing cp.async — so the MMA cannot be confined behind this n8-shaped helper
// without rewriting the surrounding NVIDIA kernel bodies (which would break PTX
// bit-identity). The raw `mma.sync`/`cp.async` PTX still present in the GEMM
// bodies below also is not hipcc-compilable. This whole .cu therefore HIP-compiles
// only via the strix-hip-real WMMA rewrite (symlinked in the follow-up stage),
// NOT through this shared gb10 source. The HIP-portable helper above
// (atlas_cvt_e4m3x2_f32) IS guarded, since elementwise kernels that use it
// (predequant_nvfp4_to_fp8, bf16_to_fp8) have no mma.sync/cp.async.
#if defined(__SCALE__)
__device__ __forceinline__ float atlas_e4m3_to_f32(unsigned char b) {
    return scl_fp8(b);  // standard E4M3, matches quantizer (SCALE __NV_E4M3 is non-standard)
}
__device__ __forceinline__ unsigned atlas_bf2(float lo, float hi) {
    unsigned short l = __bfloat16_as_ushort(__float2bfloat16(lo));
    unsigned short h = __bfloat16_as_ushort(__float2bfloat16(hi));
    return ((unsigned)h << 16) | l;
}
#endif
__device__ __forceinline__ void atlas_mma_e4m3(float* acc,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
#if defined(__SCALE__)
    unsigned lane = threadIdx.x & 31u, tig = lane & 3u, base = lane & ~3u;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        unsigned A_g = half ? a2 : a0, A_g8 = half ? a3 : a1, B_g = half ? b1 : b0;
        #define ATLAS_GA(reg, j) atlas_e4m3_to_f32((unsigned char)( \
            __shfl_sync(0xffffffffu, (reg), base + ((unsigned)(j) >> 2)) \
            >> (8 * ((j) & 3))))
        int j0 = 2 * (int)tig, j1 = 8 + 2 * (int)tig;
        unsigned A0 = atlas_bf2(ATLAS_GA(A_g, j0),  ATLAS_GA(A_g, j0 + 1));
        unsigned A1 = atlas_bf2(ATLAS_GA(A_g8, j0), ATLAS_GA(A_g8, j0 + 1));
        unsigned A2 = atlas_bf2(ATLAS_GA(A_g, j1),  ATLAS_GA(A_g, j1 + 1));
        unsigned A3 = atlas_bf2(ATLAS_GA(A_g8, j1), ATLAS_GA(A_g8, j1 + 1));
        unsigned B0 = atlas_bf2(ATLAS_GA(B_g, j0),  ATLAS_GA(B_g, j0 + 1));
        unsigned B1 = atlas_bf2(ATLAS_GA(B_g, j1),  ATLAS_GA(B_g, j1 + 1));
        #undef ATLAS_GA
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1),
              "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
    }
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
#endif
}

#define M_TILE 64
#define N_TILE_SM 64
#define N_TILE_LG 128
#define K_STEP 16
#define K_STEP_T 32
#define PAD 2
#define PAD_T 8        // cp.async needs 16-byte aligned rows: (32+8)*2=80, 80%16=0
#define BP_PAD 16      // smem_Bp row padding: stride 144 is 16-byte aligned, eliminates 4-way bank conflict
#define B_PAD 2        // BF16 padding for bank-conflict-free smem_B_bf16 (stride 17 coprime with 32)
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Original layout w4a16_gemm: unchanged, N_TILE=64
extern "C" __global__ void w4a16_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE_SM + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE_SM + PAD;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        {
            const unsigned int ept = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                smem_A[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE_SM;
                unsigned int n = idx % N_TILE_SM;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_packed[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);
                    unsigned int sg = gk / GROUP_SIZE;
                    unsigned char sb = B_scale[(unsigned long long)gn * num_groups + sg];
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__)
                    smem_B[k][n] = __float2bfloat16(E2M1_LUT[nibble] * scl_fp8(sb) * scale2);
#else
                    smem_B[k][n] = __float2bfloat16(E2M1_LUT[nibble] * (float)fp8 * scale2);
#endif
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;
        unsigned int fr0 = warp_m_offset + group_id;
        unsigned int fr1 = fr0 + 8;
        unsigned int fc0 = tid * 2, fc1 = fc0 + 8;
        unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0];
        unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0];
        unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1];
        unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned int nc = nt * 8 + group_id;
            unsigned int k0 = tid * 2, k1 = k0 + 8;
            unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
            unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// cp.async 2-stage double-buffered transposed GEMM.
//
// Overlaps global→smem loads for tile N+1 with MMA compute on tile N.
// All loads (A, Bp, Bs) use cp.async.16 for register-free transfers.
//
// smem (double-buffered):
//   A:  2 × 64 × 40 × 2B = 10240B
//   Bp: 2 × 16 × 144     =  4608B
//   Bs: 2 × 2  × 144     =   576B
//   LUT: 64B
//   Total: ~15.5KB → register-limited at ~6 CTAs/SM (unchanged)
//
// B_packed[K/2, N], B_scale[K/GROUP_SIZE, N].
// ═══════════════════════════════════════════════════════════════════

// cp.async helpers (SM80+)
__device__ __forceinline__ void cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

// Wait until at most N cp.async groups remain in flight (staged drain).
// Lets a deeper pipeline keep multiple tiles outstanding instead of the
// full-drain wait_group 0. N is a compile-time immediate (PTX requires it).
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

__device__ __forceinline__ unsigned int pack_bf16_pair(float lo, float hi) {
    unsigned int result;
    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(result)
        : "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi)));
    return result;
}

// ═══════════════════════════════════════════════════════════════════
// FP8-MMA transposed dense GEMM.
//
// Dequant B to FP8 E4M3 (not BF16). Convert A from BF16→FP8 in
// registers. Use mma.sync.m16n8k32.e4m3.e4m3 — processes full K=32
// per instruction (2x fewer MMA instructions vs BF16 m16n8k16).
//
// Pipeline: load[nxt] || MMA[cur] → wait → dequant[nxt] → sync
//
// smem: A 2×64×40×2=10240B, Bp 2×16×144=4608B, Bs 2×2×144=576B,
//       B_fp8 128×32=4096B, LUT 64B = ~19.6KB
// ═══════════════════════════════════════════════════════════════════

// Convert 4 BF16 values from smem to packed uint32 of 4 E4M3 values
__device__ __forceinline__ unsigned int bf16x4_to_e4m3x4(const unsigned short* src) {
    unsigned int p0 = *(const unsigned int*)src;
    unsigned int p1 = *(const unsigned int*)(src + 2);
    unsigned short bf0 = (unsigned short)(p0 & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p0 >> 16);
    unsigned short bf2 = (unsigned short)(p1 & 0xFFFFu);
    unsigned short bf3 = (unsigned short)(p1 >> 16);
    float f0, f1, f2, f3;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f0) : "h"(bf0));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f1) : "h"(bf1));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f2) : "h"(bf2));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f3) : "h"(bf3));
    unsigned short h0, h1;
    h0 = atlas_cvt_e4m3x2_f32(f1, f0);
    h1 = atlas_cvt_e4m3x2_f32(f3, f2);
    return ((unsigned int)h1 << 16) | (unsigned int)h0;
}

extern "C" __global__ void w4a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
#if defined(__SCALE__)
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
#else
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T];
#endif
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    #define ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[gr * K + gc], (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

#if defined(__SCALE__)
    // Dequant B: NVFP4 -> BF16 directly (gfx1151: device float->E4M3 encode is
    // broken in SCALE, and SCALE's E4M3 is a narrow [0.125,31] format; BF16
    // carries the full range/precision. Mirrors the base w4a16_gemm path.)
    #define DEQUANT_T(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = scl_fp8(*(const unsigned char*)&f0) * scale2, sv1 = scl_fp8(*(const unsigned char*)&f1) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4] * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4] * sv1); \
        } \
    } while(0)

    // BF16 MMA: 2x m16n8k16 over the 32-wide K step (no FP8 round-trip).
    #define COMPUTE_MMA(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        _Pragma("unroll") \
        for (int h = 0; h < 2; h++) { \
            unsigned int fc0 = h * 16 + tid * 2, fc1 = fc0 + 8; \
            unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
            unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
            unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
            unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
            _Pragma("unroll") \
            for (int nt = 0; nt < 16; nt++) { \
                unsigned int nc = nt * 8 + group_id; \
                const __nv_bfloat16* sb = &smem_B_bf16[nc][0]; \
                unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                    : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3]) \
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                      "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3])); \
            } \
        } \
    } while(0)
#else
    // Dequant B: FP4 → FP8 E4M3 (cvt.rn.satfinite.e4m3x2.f32)
    #define DEQUANT_T(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // FP8 MMA: convert A BF16→E4M3 in registers, single m16n8k32 per N-tile
    #define COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)
#endif

    ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    DEQUANT_T(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        DEQUANT_T(nxt);
        __syncthreads();
        cur = nxt;
    }

    COMPUTE_MMA(cur);

    #undef ISSUE_LOADS
    #undef DEQUANT_T
    #undef COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pre-dequanted FP8 GEMM (prefill).
//
// B_fp8 is pre-dequanted at load time: NVFP4 → FP8 E4M3 once.
// Eliminates the per-inference DEQUANT phase entirely.
// B_fp8[N, K] layout — each row is one output neuron, K consecutive.
//
// Pipeline: LOAD(A+B_fp8) || COMPUTE_MMA — only 1 sync per K step.
//
// smem: A 2×64×40×2=10240B, B_fp8 2×128×32=8192B = ~18.4KB
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void fp8_gemm_t(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B_fp8,   // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_B[2][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (BF16) + B (FP8, pre-dequanted) via cp.async
    #define FP8_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8 MMA — identical to w4a16_gemm_t COMPUTE_MMA
    #define FP8_COMPUTE(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            atlas_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)

    // Prolog: load first tile, wait, no dequant needed
    FP8_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Main loop: LOAD(nxt) || COMPUTE(cur) → wait → sync
    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FP8_LOADS(nxt, k_base);
        cp_async_commit();
        FP8_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FP8_COMPUTE(cur, cur);

    #undef FP8_LOADS
    #undef FP8_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pre-dequant: NVFP4 [N, K/2] + scales [N, K/GROUP_SIZE] → FP8 [N, K]
//
// One-time conversion at model load. Each thread processes 1 packed
// byte (2 FP4 values) → 2 FP8 E4M3 values.
// Grid: (ceil(N * K/2 / 256), 1, 1)  Block: (256, 1, 1)
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void predequant_nvfp4_to_fp8(
    const unsigned char* __restrict__ B_packed,  // [N, K/2]
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE]
    float scale2,
    unsigned char* __restrict__ B_fp8,           // [N, K]
    unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_K = K / 2;
    unsigned int total = N * half_K;
    if (idx >= total) return;

    unsigned int n = idx / half_K;
    unsigned int k_pair = idx % half_K;
    unsigned int k_even = k_pair * 2;

    unsigned char packed = B_packed[(unsigned long long)n * half_K + k_pair];
    unsigned int group = k_even / GROUP_SIZE;
    unsigned char sb = B_scale[(unsigned long long)n * (K / GROUP_SIZE) + group];
    __nv_fp8_e4m3 fp8_scale;
    *(unsigned char*)&fp8_scale = sb;
#if defined(__SCALE__)
    float sv = scl_fp8(sb) * scale2;
#else
    float sv = (float)fp8_scale * scale2;
#endif

    float val_lo = E2M1_LUT[packed & 0xF] * sv;
    float val_hi = E2M1_LUT[packed >> 4] * sv;

    unsigned short fp8_pair;
    fp8_pair = atlas_cvt_e4m3x2_f32(val_hi, val_lo);

    *(unsigned short*)&B_fp8[(unsigned long long)n * K + k_even] = fp8_pair;
}

// ═══════════════════════════════════════════════════════════════════
// BF16 → FP8 E4M3 activation conversion.
// Converts [M, K] BF16 activations to [M, K] FP8 E4M3 in-place or
// out-of-place. Grid: (ceil(M*K/2 / 256), 1, 1)  Block: (256, 1, 1)
// Each thread converts 2 BF16 values → 2 FP8 values via cvt.e4m3x2.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void bf16_to_fp8(
    const __nv_bfloat16* __restrict__ src,   // [M, K] BF16
    unsigned char* __restrict__ dst,          // [M, K] FP8 E4M3
    unsigned int total_elements               // M * K (must be even)
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    unsigned int p = *(const unsigned int*)&src[idx];
    unsigned short bf0 = (unsigned short)(p & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p >> 16);
    float f0, f1;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f0) : "h"(bf0));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f1) : "h"(bf1));
    unsigned short fp8_pair;
    fp8_pair = atlas_cvt_e4m3x2_f32(f1, f0);
    *(unsigned short*)&dst[idx] = fp8_pair;
}

// ═══════════════════════════════════════════════════════════════════
// FP8×FP8 GEMM: A [M, K] FP8 E4M3 × B [N, K] FP8 E4M3 → C [M, N] BF16
//
// Both A and B are pre-converted to FP8. No BF16→FP8 conversion in
// the inner loop — pure cp.async loads + FP8 MMA.
// Same tiling as fp8_gemm_t: M_TILE=64, N_TILE=128, K_STEP=32.
// A smem is FP8 (half the size of BF16 variant), no PAD needed.
// Grid: (ceil(N/128), ceil(M/64))  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
#define A_FP8_STRIDE 32  // K_STEP_T = 32 bytes per row for FP8

extern "C" __global__ void fp8_fp8_gemm_t(
    const unsigned char* __restrict__ A_fp8,  // [M, K] FP8 E4M3
    const unsigned char* __restrict__ B_fp8,  // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,            // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // A smem: FP8 [64][32] = 2 KB per buffer (vs 5 KB BF16)
    __shared__ unsigned char smem_Af[2][M_TILE][A_FP8_STRIDE];
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    // Load A (FP8) + B (FP8) via cp.async — both 1 byte per element
    #define FF_LOADS(buf, kb) do { \
        { \
            /* 128 threads load 64 rows × 32 bytes: each thread loads 16 bytes */ \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            unsigned int row = a_row_base; \
            unsigned int gr = cta_m + row; \
            cp_async_pred_16(&smem_Af[(buf)][row][a_col], \
                &A_fp8[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 15 < K)); \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8×FP8 MMA — no conversion needed, read A directly as FP8
    #define FF_COMPUTE(a_buf, b_buf) do { \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        /* A fragments: 4 bytes = 4 FP8 elements per register, need 8 regs (m16×k32) */ \
        unsigned int a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        unsigned int a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        unsigned int a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        unsigned int a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            atlas_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)

    // Prolog: load first tile, wait
    FF_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Main loop: LOAD(nxt) || COMPUTE(cur) → wait → sync
    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FF_LOADS(nxt, k_base);
        cp_async_commit();
        FF_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FF_COMPUTE(cur, cur);

    #undef FF_LOADS
    #undef FF_COMPUTE

    // Write results
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// K64 FP8-MMA transposed dense GEMM — halves outer K-loop vs K32.
//
// Same algorithm as w4a16_gemm_t but K_STEP_T64=64: 32 outer iterations
// instead of 64 for K=2048. Two m16n8k32 MMAs per N-tile per step.
// Reduces loop overhead and better amortizes DMA startup cost.
//
// K must be divisible by 64.
//
// smem: A 2×64×72×2=18432B, Bp 2×32×144=9216B, Bs 2×4×144=1152B,
//       B_fp8 128×80=10240B, LUT 64B = ~38.4KB
// ═══════════════════════════════════════════════════════════════════
#define K_STEP_T64 64
#define PAD_T64    8   // (64+8)*2=144, 144%16=0 ✓

extern "C" __global__ void w4a16_gemm_t_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // B_fp8 row stride 80 = K64+16: avoids 4-way bank conflicts.
    __shared__ __nv_bfloat16 smem_A_k64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_k64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_k64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8_k64[N_TILE_LG][K_STEP_T64 + 16];
    __shared__ float smem_LUT_k64[16];

    if (threadIdx.x < 16) smem_LUT_k64[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast64 = K_STEP_T64 + PAD_T64;

    // A: 4 rounds × 16 rows = 64 rows (M_TILE); each thread: 8 BF16 = 16 bytes.
    // Bp: 2 rounds × 16 rows = 32 rows (K64/2); each thread: 16 bytes per ns chunk.
    // Bs: inline with Bp when kp_cur < K_STEP_T64/GROUP_SIZE (4 scale groups).
    #define K64_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A_k64[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                cp_async_pred_16(&smem_Bp_k64[(buf)][kp_cur][ns], \
                    &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 <= K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    cp_async_pred_16(&smem_Bs_k64[(buf)][kp_cur][ns], \
                        &B_scale[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    // 4 scale groups, 32 dequant iters: sv0→K{0..15}, sv1→K{16..31},
    // sv2→K{32..47}, sv3→K{48..63}.
#if defined(__SCALE__)
    #define K64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_k64[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_k64[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_k64[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_k64[(buf)][3][my_n]; \
        float sv0 = scl_fp8(*(const unsigned char*)&f0) * scale2, sv1 = scl_fp8(*(const unsigned char*)&f1) * scale2; \
        float sv2 = scl_fp8(*(const unsigned char*)&f2) * scale2, sv3 = scl_fp8(*(const unsigned char*)&f3) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv0; \
            float hi = smem_LUT_k64[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            fp8_pair = atlas_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv1; \
            float hi = smem_LUT_k64[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            fp8_pair = atlas_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv2; \
            float hi = smem_LUT_k64[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            fp8_pair = atlas_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv3; \
            float hi = smem_LUT_k64[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            fp8_pair = atlas_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)
#else
    #define K64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_k64[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_k64[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_k64[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_k64[(buf)][3][my_n]; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        float sv2 = (float)f2 * scale2, sv3 = (float)f3 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv0; \
            float hi = smem_LUT_k64[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv1; \
            float hi = smem_LUT_k64[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv2; \
            float hi = smem_LUT_k64[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv3; \
            float hi = smem_LUT_k64[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)
#endif

    // Two m16n8k32 MMA calls per N-tile: first covers K=0..31, second K=32..63.
    #define K64_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_k64[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][16 + 4 * tid]; \
            atlas_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
        unsigned int a4 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 32 + tid * 4]); \
        unsigned int a5 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 32 + tid * 4]); \
        unsigned int a6 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 48 + tid * 4]); \
        unsigned int a7 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][48 + 4 * tid]; \
            atlas_mma_e4m3(acc[nt], a4,a5,a6,a7, b0, b1); \
        } \
    } while(0)

    K64_ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    K64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        K64_ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        K64_COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        K64_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    K64_COMPUTE_MMA(cur);

    #undef K64_ISSUE_LOADS
    #undef K64_DEQUANT
    #undef K64_COMPUTE_MMA
    #undef K_STEP_T64
    #undef PAD_T64

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// M128 variant: 2 consecutive 64-row M-chunks per CTA.
//
// For large-M prefill (e.g. ISL=1016, N=12288):
//   M_TILE=64: grid=(96,16,1)=1536 blocks, 16 weight re-reads  → 227MB B DRAM
//   M_TILE2=128: grid=(96,8,1)=768 blocks, 8 weight re-reads   → 114MB B DRAM
//
// SMEM: A 2×128×40×2=20480B, Bp 2×16×144=4608B, Bs 2×2×144=576B,
//       B_fp8 128×32=4096B, LUT 64B ≈ 29.8KB → 3 blocks/SM.
//
// For qkvz (K=2048,N=12288): ~2× speedup at ISL>128 vs w4a16_gemm_t.
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__
__launch_bounds__(128, 3)
void w4a16_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * (2 * M_TILE);  // base row for this block
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // A is 2× larger (128 rows instead of 64); B/LUT/dequant identical to w4a16_gemm_t.
    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];   // 20480 B
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD]; // 4608 B
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD]; // 576 B
#if defined(__SCALE__)
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];          // BF16 (gfx1151)
#else
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T];             // 4096 B
#endif
    __shared__ float smem_LUT[16];                                         //   64 B
    // Total ≈ 29.8 KB → 3 blocks/SM

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    // Two sets of accumulators: chunk0 = rows [cta_m..cta_m+63],
    //                           chunk1 = rows [cta_m+64..cta_m+127].
    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (4 rounds → 128 rows) + B (same as w4a16_gemm_t).
    #define M128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

#if defined(__SCALE__)
    // Dequant B tile: NVFP4 -> BF16 directly (gfx1151).
    #define M128_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = scl_fp8(*(const unsigned char*)&f0) * scale2, sv1 = scl_fp8(*(const unsigned char*)&f1) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv1); \
        } \
    } while(0)

    // MMA for both M-chunks; B tile (smem_B_fp8) loaded once, reused by both.
    #define M128_COMPUTE(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A[(a_buf)]; \
        _Pragma("unroll") \
        for (int ch = 0; ch < 2; ch++) { \
            unsigned int fr0 = ch * M_TILE + warp_m_offset + group_id; \
            unsigned int fr1 = fr0 + 8; \
            _Pragma("unroll") \
            for (int h = 0; h < 2; h++) { \
                unsigned int fc0 = h * 16 + tid * 2, fc1 = fc0 + 8; \
                unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
                unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
                unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
                unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
                _Pragma("unroll") \
                for (int nt = 0; nt < 16; nt++) { \
                    unsigned int nc = nt * 8 + group_id; \
                    const __nv_bfloat16* sb = &smem_B_bf16[nc][0]; \
                    unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                    unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                    float* acc = ch ? acc1[nt] : acc0[nt]; \
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]) \
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])); \
                } \
            } \
        } \
    } while(0)
#else
    // Dequant B tile: identical to w4a16_gemm_t's DEQUANT_T.
    #define M128_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4]  * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4]  * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // MMA for both M-chunks; B tile (smem_B_fp8) loaded once, reused by both.
    #define M128_COMPUTE(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        /* Chunk 0: smem rows 0..63 */ \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc0[nt][0]),"=f"(acc0[nt][1]),"=f"(acc0[nt][2]),"=f"(acc0[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc0[nt][0]),"f"(acc0[nt][1]),"f"(acc0[nt][2]),"f"(acc0[nt][3])); \
        } \
        /* Chunk 1: smem rows 64..127 (offset M_TILE=64) */ \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc1[nt][0]),"=f"(acc1[nt][1]),"=f"(acc1[nt][2]),"=f"(acc1[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc1[nt][0]),"f"(acc1[nt][1]),"f"(acc1[nt][2]),"f"(acc1[nt][3])); \
        } \
    } while(0)
#endif

    // Pipeline: same double-buffer structure as w4a16_gemm_t.
    M128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    M128_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        M128_LOADS(nxt, k_base);
        cp_async_commit();
        M128_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        M128_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    M128_COMPUTE(cur);

    #undef M128_LOADS
    #undef M128_DEQUANT
    #undef M128_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// M128 variant — LOSSLESS BF16 prefill (`w4a16_gemm_t_m128_bf16`).
//
// Identical 128x128 cp.async double-buffered pipeline to `w4a16_gemm_t_m128`,
// but the NVIDIA build keeps the FP4→BF16 dequant + `m16n8k16.f32.bf16.bf16.f32`
// MMA (the same math the base `w4a16_gemm` uses) instead of crushing weights
// AND activations to FP8 E4M3. The FP8 path perturbs generation (measured
// length-truncations / accuracy risk on Qwen3.6-27B); this kernel preserves
// outputs bit-for-bit vs the base while keeping the fast 128x128 tiling.
//
// The dequant + MMA below are byte-for-byte the SCALE branch's BF16
// M128_DEQUANT/M128_COMPUTE from `w4a16_gemm_t_m128`, with ONE NVIDIA-correct
// substitution: the block-scale decode uses the standard `(float)f0` E4M3 cast
// (matching `w4a16_gemm`'s #else NVIDIA path) rather than `scl_fp8()` (which is
// only defined under __SCALE__/__HIP and is the standard-E4M3 software decode
// the SCALE/gfx1151 builds need). On real NVIDIA these are byte-identical
// (see header note lines 15-19). The smem layout, A/B load order, K-iteration
// order and FP32 accumulation order all match the existing BF16 128x128 path,
// so results are ~bit-equivalent to `w4a16_gemm`.
//
// SMEM: A 2×128×40×2=20480B, Bp 2×16×144=4608B, Bs 2×2×144=576B,
//       B_bf16 128×32×2=8192B, LUT 64B ≈ 33.9KB → 2-3 blocks/SM.
// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(128, 3)
void w4a16_gemm_t_m128_bf16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * (2 * M_TILE);  // base row for this block
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // A is 2× larger (128 rows instead of 64); B/LUT/dequant identical to w4a16_gemm_t.
    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];   // 20480 B
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD]; // 4608 B
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD]; // 576 B
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];           // 8192 B (BF16)
    __shared__ float smem_LUT[16];                                         //   64 B

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    // Two sets of accumulators: chunk0 = rows [cta_m..cta_m+63],
    //                           chunk1 = rows [cta_m+64..cta_m+127].
    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (4 rounds → 128 rows) + B (same as w4a16_gemm_t / w4a16_gemm_t_m128).
    #define M128B_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant B tile: NVFP4 -> BF16 directly (no FP8 crush). Mirrors the SCALE
    // branch's M128_DEQUANT, but with the standard NVIDIA `(float)f0` E4M3 scale
    // decode (byte-identical to `scl_fp8()` on real NVIDIA — header lines 15-19).
    #define M128B_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv1); \
        } \
    } while(0)

    // MMA for both M-chunks; B tile (smem_B_bf16) loaded once, reused by both.
    // BF16 m16n8k16 with FP32 accumulators — same instruction/order as base.
    #define M128B_COMPUTE(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A[(a_buf)]; \
        _Pragma("unroll") \
        for (int ch = 0; ch < 2; ch++) { \
            unsigned int fr0 = ch * M_TILE + warp_m_offset + group_id; \
            unsigned int fr1 = fr0 + 8; \
            _Pragma("unroll") \
            for (int h = 0; h < 2; h++) { \
                unsigned int fc0 = h * 16 + tid * 2, fc1 = fc0 + 8; \
                unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
                unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
                unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
                unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
                _Pragma("unroll") \
                for (int nt = 0; nt < 16; nt++) { \
                    unsigned int nc = nt * 8 + group_id; \
                    const __nv_bfloat16* sb = &smem_B_bf16[nc][0]; \
                    unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                    unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                    float* acc = ch ? acc1[nt] : acc0[nt]; \
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]) \
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])); \
                } \
            } \
        } \
    } while(0)

    // Pipeline: same double-buffer structure as w4a16_gemm_t_m128.
    M128B_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    M128B_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        M128B_LOADS(nxt, k_base);
        cp_async_commit();
        M128B_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        M128B_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    M128B_COMPUTE(cur);

    #undef M128B_LOADS
    #undef M128B_DEQUANT
    #undef M128B_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// M128 variant — LOSSLESS BF16 prefill, PIPELINED v2 (`w4a16_gemm_t_m128_bf16_v2`).
//
// Same math, same smem layout, same MMA instruction sequence (and therefore
// the SAME per-output FP32 accumulation order) as `w4a16_gemm_t_m128_bf16`,
// so it is BIT-IDENTICAL to it (and ~bit-equivalent to base `w4a16_gemm`).
// The ONLY changes are scheduling/footprint, attacking the PROFILED root cause:
// the kernel is LATENCY-BOUND at ~16% occupancy (ncu: sm__warps_active 16.7%).
// v1 uses 34.9 KB smem/block, and with a 100 KB/SM budget that caps it to
// 2 CTAs/SM (3×34.9=104.8 > 100) — only 8 of 48 warps resident, too few to
// hide the cp.async + dequant + MMA latency chain.
//
//   LEVER — OCCUPANCY via SMEM: drop the A-tile bank-conflict pad from
//     PAD_T=8 to PAD_T_V2=0 (A stride 32 instead of 40). A row is 32 bf16 =
//     64 B = exactly 16-byte aligned, so cp.async.16 stays legal. This shaves
//     A from 2×128×40×2=20480 B to 2×128×32×2=16384 B (−4 KB) → 30.9 KB/block
//     → 3 CTAs/SM (3×30.9=92.6 ≤ 100) → +50% resident warps (12 vs 8).
//
// Everything else — the 2-stage cp.async double-buffer, the single-buffered
// smem_B_bf16 dequant target, the LOAD||MMA→wait→dequant→sync schedule, the
// MMA instruction order, registers (__launch_bounds__(128,3), no spill) — is
// byte-for-byte v1. The A read in COMPUTE is now row-strided by a_stride=32 →
// some smem bank conflicts (vs conflict-free at 40); this is a deliberate
// trade — on a LATENCY-bound kernel the +50% occupancy is the dominant term.
// (Deeper pipelines / B double-buffering / forcing 4 CTAs were all measured
// neutral-to-slower: register spill or smem-bound. See report.)
//
// SMEM (STAGES=2): A 2×128×32×2=16384B, Bp 2×16×144=4608B, Bs 2×2×144=576B,
//                  B_bf16 128×32×2=8192B, LUT 64B ≈ 30.9KB → 3 CTAs/SM.
// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
#define M128B_STAGES 2
#define PAD_T_V2 0
extern "C" __global__
__launch_bounds__(128, 3)
void w4a16_gemm_t_m128_bf16_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // cp.async-staged raw tiles (STAGES deep). The dequant target smem_B_bf16
    // is SINGLE-buffered (like v1) to keep total smem at 30.9 KB → 3 CTAs/SM;
    // A uses the reduced PAD_T_V2 pad (LEVER A) instead of PAD_T.
    __shared__ __nv_bfloat16 smem_A[M128B_STAGES][2 * M_TILE][K_STEP_T + PAD_T_V2];
    __shared__ unsigned char smem_Bp[M128B_STAGES][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[M128B_STAGES][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T_V2;

    // Issue cp.async loads for stage `buf` covering K-tile starting at `kb`.
    // Byte-identical addressing to M128B_LOADS in the v1 kernel.
    #define V2_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant raw tile in stage `buf` → the single BF16 buffer.
    // Byte-identical math to v1's M128B_DEQUANT (LUT * (float)e4m3 * scale2).
    #define V2_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv1); \
        } \
    } while(0)

    // MMA over A-stage `a_buf` and the dequanted B buffer.
    // IDENTICAL instruction order to v1's M128B_COMPUTE → bit-identical accum.
    #define V2_COMPUTE(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A[(a_buf)]; \
        _Pragma("unroll") \
        for (int ch = 0; ch < 2; ch++) { \
            unsigned int fr0 = ch * M_TILE + warp_m_offset + group_id; \
            unsigned int fr1 = fr0 + 8; \
            _Pragma("unroll") \
            for (int hh = 0; hh < 2; hh++) { \
                unsigned int fc0 = hh * 16 + tid * 2, fc1 = fc0 + 8; \
                unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
                unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
                unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
                unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
                _Pragma("unroll") \
                for (int nt = 0; nt < 16; nt++) { \
                    unsigned int nc = nt * 8 + group_id; \
                    const __nv_bfloat16* sb = &smem_B_bf16[nc][0]; \
                    unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                    unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                    float* acc = ch ? acc1[nt] : acc0[nt]; \
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]) \
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])); \
                } \
            } \
        } \
    } while(0)

    // Pipeline: byte-identical schedule to v1 (LOAD[nxt] || MMA[cur] → wait →
    // dequant[nxt] → sync). The ONLY difference from v1 is the smaller A pad
    // (PAD_T_V2), which lifts occupancy to 3 CTAs/SM.
    V2_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    V2_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        V2_LOADS(nxt, k_base);
        cp_async_commit();
        V2_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        V2_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    V2_COMPUTE(cur);

    #undef V2_LOADS
    #undef V2_DEQUANT
    #undef V2_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}
#undef M128B_STAGES

// ═══════════════════════════════════════════════════════════════════
// M64 lossless BF16-TC prefill (`w4a16_gemm_t_m64_bf16`).
//
// BYTE-IDENTICAL math to `w4a16_gemm_t_m128_bf16_v2` (same (float)f0 E4M3
// decode, same E2M1 LUT, same m16n8k16 bf16 MMA, same per-output K/N order)
// but ONE M-chunk of 64 rows per CTA instead of two. Each output element is
// computed by the identical instruction sequence as v2's chunk-0, so the
// result is bit-for-bit v2 (and ~bit-equivalent to base w4a16_gemm).
//
// WHY: v2's 2-chunk/CTA tiling holds 128 FP32 accumulators/thread (168 regs)
// and ~30KB smem -> 3 CTAs/SM = 23% occupancy (ncu), and the prefill GEMM is
// LATENCY-bound, not MMA- or bandwidth-bound (measured: fp8 m16n8k32 gives no
// speedup; 4% DRAM BW). Halving to 64 acc/thread (one chunk) drops registers
// and smem (A 2x64x32x2=8192B) -> 4-5 CTAs/SM, lifting occupancy enough to
// hide the cp.async+sync latency chain. Microbench: ~44 vs ~30 TFLOP/s (~1.47x)
// on gate/up+down, LOSSLESS. (M128 was tuned to halve B DRAM reads — a
// non-lever in this latency-bound, BW-slack regime.)
//
// SMEM: A 2x64x32x2=8192B, Bp 2x16x144=4608B, Bs 2x2x144=576B,
//       B_bf16 128x32x2=8192B, LUT 64B = ~21.6KB -> 4 CTAs/SM.
// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
#define M64B_PAD 0
extern "C" __global__
__launch_bounds__(128, 4)
void w4a16_gemm_t_m64_bf16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * M_TILE;   // 64 rows/CTA
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + M64B_PAD];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    const unsigned int a_stride = K_STEP_T + M64B_PAD;

    #define M64_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    #define M64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv1); \
        } \
    } while(0)

    #define M64_COMPUTE(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id; \
        unsigned int fr1 = fr0 + 8; \
        _Pragma("unroll") \
        for (int h = 0; h < 2; h++) { \
            unsigned int fc0 = h * 16 + tid * 2, fc1 = fc0 + 8; \
            unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
            unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
            unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
            unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
            _Pragma("unroll") \
            for (int nt = 0; nt < 16; nt++) { \
                unsigned int nc = nt * 8 + group_id; \
                const __nv_bfloat16* sb = &smem_B_bf16[nc][0]; \
                unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                float* ac = acc[nt]; \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                    : "=f"(ac[0]), "=f"(ac[1]), "=f"(ac[2]), "=f"(ac[3]) \
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                      "f"(ac[0]), "f"(ac[1]), "f"(ac[2]), "f"(ac[3])); \
            } \
        } \
    } while(0)

    M64_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    M64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        M64_LOADS(nxt, k_base);
        cp_async_commit();
        M64_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        M64_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    M64_COMPUTE(cur);

    #undef M64_LOADS
    #undef M64_DEQUANT
    #undef M64_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3]);
    }
}
#undef M64B_PAD

// ═══════════════════════════════════════════════════════════════════
// M128 variant of fp8_gemm_t: BF16 A × FP8 B, 2 M-chunks per CTA.
//
// For out_proj (K=2048, N=2048) and paged Q/K/V: halves the number of
// times B is read from DRAM (8 m-tile groups vs 16 at M=1015).
//
// SMEM: A 2×128×40×2=20480B, B 2×128×32=8192B ≈ 28.7KB → 3 blocks/SM.
// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(128, 3)
void fp8_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B_fp8,   // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];  // 20480 B
    __shared__ unsigned char  smem_B[2][N_TILE_LG][K_STEP_T];            //  8192 B

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (BF16, 4 rounds → 128 rows) + B (FP8, same as fp8_gemm_t).
    #define FGM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8 MMA for both M-chunks; B tile loaded once and reused.
    #define FGM128_COMPUTE(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        /* Chunk 0: smem rows 0..63 */ \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            atlas_mma_e4m3(acc0[nt], a0,a1,a2,a3, b0, b1); \
        } \
        /* Chunk 1: smem rows 64..127 */ \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            atlas_mma_e4m3(acc1[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)

    FGM128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FGM128_LOADS(nxt, k_base);
        cp_async_commit();
        FGM128_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FGM128_COMPUTE(cur, cur);

    #undef FGM128_LOADS
    #undef FGM128_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// M128 variant of fp8_fp8_gemm_t: FP8 A × FP8 B, 2 M-chunks per CTA.
//
// For Q/K/V projections in cache-skip prefill path (FP8 activations):
// halves B re-reads. Uses 3 blocks/SM (not 6) to avoid register spilling:
// dual acc0+acc1 need ~145 regs/thread; 3 blocks allows 170 regs/thread.
//
// SMEM: Af 2×128×32=8192B, Bf 2×128×32=8192B ≈ 16KB, 3 blocks → 48KB/SM.
// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(128, 3)
void fp8_fp8_gemm_t_m128(
    const unsigned char* __restrict__ A_fp8,  // [M, K] FP8 E4M3
    const unsigned char* __restrict__ B_fp8,  // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,            // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ unsigned char smem_Af[2][2 * M_TILE][A_FP8_STRIDE];  //  8192 B
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];        //  8192 B

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    // Load A (FP8, 2 rounds → 128 rows) + B (FP8, same as fp8_fp8_gemm_t).
    #define FFM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_Af[(buf)][row][a_col], \
                    &A_fp8[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 15 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8×FP8 MMA for both M-chunks; B loaded once, reused by both.
    #define FFM128_COMPUTE(a_buf, b_buf) do { \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        /* Chunk 0: smem rows 0..63 */ \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            atlas_mma_e4m3(acc0[nt], a0,a1,a2,a3, b0, b1); \
        } \
        /* Chunk 1: smem rows 64..127 */ \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            atlas_mma_e4m3(acc1[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)

    FFM128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FFM128_LOADS(nxt, k_base);
        cp_async_commit();
        FFM128_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FFM128_COMPUTE(cur, cur);

    #undef FFM128_LOADS
    #undef FFM128_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8 prefill GEMM (`int8_gemm_t_m128`).
//
// C[M,N] = A_i8[M,K] · B_i8[N,K]^T with PER-32-K-BLOCK dequant:
//   C[m,n] = Σ_blk ( s32_dot32(A_i8,B_i8) · A_scale[m,blk] · B_scale[n,blk] )
// via mma.m16n8k32.s32.s8.s8.s32 — llama-MMQ's scheme. 1-byte packed int8
// operands (4/load) cut shared-memory load INSTRUCTIONS ~4x (BF16 v2 is L1/TEX
// 90% smem-bound); int8's 8-bit precision holds generation where FP8-E4M3's
// 3-bit mantissa breaks it. A_scale[M,K/32], B_scale[N,K/32] fp32. 2 M-chunks.
// m16n8k32 fragment: thread owns (r0=gid,r1=gid+8)×(c0=2·tid,c1=2·tid+1).
// SMEM: A_i8 2×128×32 + B_i8 2×128×32 ≈ 16KB. Grid (N/128,M/128), block 128.
// ═══════════════════════════════════════════════════════════════════
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(128, 3)
void int8_gemm_t_m128(
    const signed char* __restrict__ A_i8,    // [M, K]
    const signed char* __restrict__ B_i8,    // [N, K] (transposed)
    const float* __restrict__ A_scale,        // [M, K/32]
    const float* __restrict__ B_scale,        // [N, K/32]
    __nv_bfloat16* __restrict__ C,            // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;
    const unsigned int nb = K >> 5;   // K/32 blocks

    __shared__ signed char smem_Ai[2][2 * M_TILE][32];   // 8192 B
    __shared__ signed char smem_Bi[2][N_TILE_LG][32];     // 8192 B
    __shared__ float smem_As[2][2 * M_TILE];              // 1024 B (per-block row scales)
    __shared__ float smem_Bs[2][N_TILE_LG];               // 1024 B (per-block col scales)

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0]=0.f; acc0[i][1]=0.f; acc0[i][2]=0.f; acc0[i][3]=0.f;
        acc1[i][0]=0.f; acc1[i][1]=0.f; acc1[i][2]=0.f; acc1[i][3]=0.f;
    }

    #define I8_LOADS(buf, kb) do { \
        { unsigned ar = threadIdx.x >> 1; unsigned ac = (threadIdx.x & 1) << 4; unsigned gc = (kb) + ac; \
          _Pragma("unroll") for (int rnd = 0; rnd < 2; rnd++) { \
            unsigned row = (unsigned)(rnd * 64) + ar; unsigned gr = cta_m + row; \
            cp_async_pred_16(&smem_Ai[(buf)][row][ac], &A_i8[(unsigned long long)gr * K + gc], (gr < M) && (gc + 15 < K)); } } \
        { unsigned my_n = threadIdx.x; unsigned gn = cta_n + my_n; bool v = (gn < N) && ((kb) + 31 < K); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][0],  &B_i8[(unsigned long long)gn * K + (kb)],      v); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][16], &B_i8[(unsigned long long)gn * K + (kb) + 16], v); } \
        { unsigned blk = (kb) >> 5; unsigned gr = cta_m + threadIdx.x; unsigned gn = cta_n + threadIdx.x; \
          smem_As[(buf)][threadIdx.x] = (gr < M) ? A_scale[(unsigned long long)gr * nb + blk] : 0.f; \
          smem_Bs[(buf)][threadIdx.x] = (gn < N) ? B_scale[(unsigned long long)gn * nb + blk] : 0.f; } \
    } while(0)

    #define I8_COMPUTE(buf, kb) do { \
        float as00 = smem_As[(buf)][warp_m_offset + group_id]; \
        float as01 = smem_As[(buf)][warp_m_offset + group_id + 8]; \
        float as10 = smem_As[(buf)][M_TILE + warp_m_offset + group_id]; \
        float as11 = smem_As[(buf)][M_TILE + warp_m_offset + group_id + 8]; \
        unsigned fr00 = warp_m_offset + group_id, fr01 = fr00 + 8; \
        unsigned a0c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][4*tid]; \
        unsigned a1c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][4*tid]; \
        unsigned a2c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][16+4*tid]; \
        unsigned a3c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][16+4*tid]; \
        unsigned fr10 = M_TILE + warp_m_offset + group_id, fr11 = fr10 + 8; \
        unsigned a0c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][4*tid]; \
        unsigned a1c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][4*tid]; \
        unsigned a2c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][16+4*tid]; \
        unsigned a3c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][16+4*tid]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt * 8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*tid]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*tid]; \
            float bs0 = smem_Bs[(buf)][nt*8 + tid*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + tid*2 + 1]; \
            int s0[4] = {0,0,0,0}, s1[4] = {0,0,0,0}; \
            ATLAS_MMA_S8(s0, a0c0,a1c0,a2c0,a3c0, b0,b1); \
            ATLAS_MMA_S8(s1, a0c1,a1c1,a2c1,a3c1, b0,b1); \
            acc0[nt][0] += (float)s0[0]*as00*bs0; acc0[nt][1] += (float)s0[1]*as00*bs1; \
            acc0[nt][2] += (float)s0[2]*as01*bs0; acc0[nt][3] += (float)s0[3]*as01*bs1; \
            acc1[nt][0] += (float)s1[0]*as10*bs0; acc1[nt][1] += (float)s1[1]*as10*bs1; \
            acc1[nt][2] += (float)s1[2]*as11*bs0; acc1[nt][3] += (float)s1[3]*as11*bs1; \
        } \
    } while(0)

    I8_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = 32; k_base < K; k_base += 32) {
        int nxt = 1 - cur;
        I8_LOADS(nxt, k_base);
        cp_async_commit();
        I8_COMPUTE(cur, k_base - 32);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    I8_COMPUTE(cur, K - 32);

    #undef I8_LOADS
    #undef I8_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc0[nt][3]);
    }
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + M_TILE + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc1[nt][3]);
    }
}
#undef ATLAS_MMA_S8

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8 prefill, M64 single-chunk (`int8_gemm_t_m64`). Same per-block
// scale dequant as int8_gemm_t_m128 but ONE M-chunk of 64 rows → half the
// accumulators/registers → higher occupancy (the lever that took fp8 27→44).
// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128,1,1)
// ═══════════════════════════════════════════════════════════════════
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(128, 4)
void int8_gemm_t_m64(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;   // 64 rows
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;
    const unsigned int nb = K >> 5;

    __shared__ signed char smem_Ai[2][M_TILE][32];     // 4096 B
    __shared__ signed char smem_Bi[2][N_TILE_LG][32];   // 8192 B
    __shared__ float smem_As[2][M_TILE];                // 512 B
    __shared__ float smem_Bs[2][N_TILE_LG];             // 512 B

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define I8M64_LOADS(buf, kb) do { \
        { unsigned ar = threadIdx.x >> 1; unsigned ac = (threadIdx.x & 1) << 4; unsigned gc = (kb) + ac; \
          unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned my_n = threadIdx.x; unsigned gn = cta_n + my_n; bool v = (gn<N)&&((kb)+31<K); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][0],  &B_i8[(unsigned long long)gn*K+(kb)],    v); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][16], &B_i8[(unsigned long long)gn*K+(kb)+16], v); } \
        { unsigned blk=(kb)>>5; \
          if (threadIdx.x < M_TILE) { unsigned gr=cta_m+threadIdx.x; smem_As[(buf)][threadIdx.x]=(gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; } \
          unsigned gn=cta_n+threadIdx.x; smem_Bs[(buf)][threadIdx.x]=(gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define I8M64_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][warp_m_offset + group_id]; \
        float as1 = smem_As[(buf)][warp_m_offset + group_id + 8]; \
        unsigned fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned a0 = *(const unsigned*)&smem_Ai[(buf)][fr0][4*tid]; \
        unsigned a1 = *(const unsigned*)&smem_Ai[(buf)][fr1][4*tid]; \
        unsigned a2 = *(const unsigned*)&smem_Ai[(buf)][fr0][16+4*tid]; \
        unsigned a3 = *(const unsigned*)&smem_Ai[(buf)][fr1][16+4*tid]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*tid]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*tid]; \
            float bs0 = smem_Bs[(buf)][nt*8+tid*2]; \
            float bs1 = smem_Bs[(buf)][nt*8+tid*2+1]; \
            int s[4] = {0,0,0,0}; \
            ATLAS_MMA_S8(s, a0,a1,a2,a3, b0,b1); \
            acc[nt][0]+=(float)s[0]*as0*bs0; acc[nt][1]+=(float)s[1]*as0*bs1; \
            acc[nt][2]+=(float)s[2]*as1*bs0; acc[nt][3]+=(float)s[3]*as1*bs1; \
        } \
    } while(0)

    I8M64_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int k_base = 32; k_base < K; k_base += 32) {
        int nxt = 1 - cur;
        I8M64_LOADS(nxt, k_base); cp_async_commit();
        I8M64_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    I8M64_COMPUTE(cur);
    #undef I8M64_LOADS
    #undef I8M64_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef ATLAS_MMA_S8

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8 SPLIT-K prefill (`int8_gemm_splitk` + `int8_splitk_reduce`).
//
// The non-split int8 kernel is latency/barrier-bound at ~8% achieved occupancy
// (per-block dequant serial chain + 160 syncs, too few active warps). Split-K
// manufactures occupancy: each (m,n,z) CTA reduces only K/ksplits of the K
// dimension into an fp32 PARTIAL tile; a separate reduce kernel sums the
// ksplits partials. ksplits× more CTAs → far more resident warps to hide the
// barrier/dequant latency. Cp layout: [ksplits, M, N] fp32 (z-major).
// Grid: (ceil(N/128), ceil(M/128), ksplits)  Block: (128,1,1)
// K must be a multiple of 32*ksplits.
// ═══════════════════════════════════════════════════════════════════
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(128, 3)
void int8_gemm_splitk(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    float* __restrict__ Cp,                   // [ksplits, M, N] fp32 partials
    unsigned int M, unsigned int N, unsigned int K, unsigned int ksplits
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    const unsigned int z = blockIdx.z;
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;
    const unsigned int nb = K >> 5;
    const unsigned int k_per = K / ksplits;       // multiple of 32
    const unsigned int k_lo = z * k_per;
    const unsigned int k_hi = k_lo + k_per;

    __shared__ signed char smem_Ai[2][2 * M_TILE][32];
    __shared__ signed char smem_Bi[2][N_TILE_LG][32];
    __shared__ float smem_As[2][2 * M_TILE];
    __shared__ float smem_Bs[2][N_TILE_LG];

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0]=0.f; acc0[i][1]=0.f; acc0[i][2]=0.f; acc0[i][3]=0.f;
        acc1[i][0]=0.f; acc1[i][1]=0.f; acc1[i][2]=0.f; acc1[i][3]=0.f;
    }

    #define SK_LOADS(buf, kb) do { \
        { unsigned ar = threadIdx.x >> 1; unsigned ac = (threadIdx.x & 1) << 4; unsigned gc = (kb) + ac; \
          _Pragma("unroll") for (int rnd = 0; rnd < 2; rnd++) { \
            unsigned row = (unsigned)(rnd * 64) + ar; unsigned gr = cta_m + row; \
            cp_async_pred_16(&smem_Ai[(buf)][row][ac], &A_i8[(unsigned long long)gr * K + gc], (gr < M) && (gc + 15 < K)); } } \
        { unsigned my_n = threadIdx.x; unsigned gn = cta_n + my_n; bool v = (gn < N) && ((kb) + 31 < K); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][0],  &B_i8[(unsigned long long)gn * K + (kb)],      v); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][16], &B_i8[(unsigned long long)gn * K + (kb) + 16], v); } \
        { unsigned blk = (kb) >> 5; unsigned gr = cta_m + threadIdx.x; unsigned gn = cta_n + threadIdx.x; \
          smem_As[(buf)][threadIdx.x] = (gr < M) ? A_scale[(unsigned long long)gr * nb + blk] : 0.f; \
          smem_Bs[(buf)][threadIdx.x] = (gn < N) ? B_scale[(unsigned long long)gn * nb + blk] : 0.f; } \
    } while(0)

    #define SK_COMPUTE(buf) do { \
        float as00 = smem_As[(buf)][warp_m_offset + group_id]; \
        float as01 = smem_As[(buf)][warp_m_offset + group_id + 8]; \
        float as10 = smem_As[(buf)][M_TILE + warp_m_offset + group_id]; \
        float as11 = smem_As[(buf)][M_TILE + warp_m_offset + group_id + 8]; \
        unsigned fr00 = warp_m_offset + group_id, fr01 = fr00 + 8; \
        unsigned a0c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][4*tid]; \
        unsigned a1c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][4*tid]; \
        unsigned a2c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][16+4*tid]; \
        unsigned a3c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][16+4*tid]; \
        unsigned fr10 = M_TILE + warp_m_offset + group_id, fr11 = fr10 + 8; \
        unsigned a0c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][4*tid]; \
        unsigned a1c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][4*tid]; \
        unsigned a2c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][16+4*tid]; \
        unsigned a3c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][16+4*tid]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt * 8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*tid]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*tid]; \
            float bs0 = smem_Bs[(buf)][nt*8 + tid*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + tid*2 + 1]; \
            int s0[4] = {0,0,0,0}, s1[4] = {0,0,0,0}; \
            ATLAS_MMA_S8(s0, a0c0,a1c0,a2c0,a3c0, b0,b1); \
            ATLAS_MMA_S8(s1, a0c1,a1c1,a2c1,a3c1, b0,b1); \
            acc0[nt][0] += (float)s0[0]*as00*bs0; acc0[nt][1] += (float)s0[1]*as00*bs1; \
            acc0[nt][2] += (float)s0[2]*as01*bs0; acc0[nt][3] += (float)s0[3]*as01*bs1; \
            acc1[nt][0] += (float)s1[0]*as10*bs0; acc1[nt][1] += (float)s1[1]*as10*bs1; \
            acc1[nt][2] += (float)s1[2]*as11*bs0; acc1[nt][3] += (float)s1[3]*as11*bs1; \
        } \
    } while(0)

    SK_LOADS(0, k_lo);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    int cur = 0;
    for (unsigned int kb = k_lo + 32; kb < k_hi; kb += 32) {
        int nxt = 1 - cur;
        SK_LOADS(nxt, kb);
        cp_async_commit();
        SK_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    SK_COMPUTE(cur);
    #undef SK_LOADS
    #undef SK_COMPUTE

    unsigned long long zoff = (unsigned long long)z * M * N;
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) Cp[zoff + (unsigned long long)r0*N+c0] = acc0[nt][0];
        if (r0 < M && c1 < N) Cp[zoff + (unsigned long long)r0*N+c1] = acc0[nt][1];
        if (r1 < M && c0 < N) Cp[zoff + (unsigned long long)r1*N+c0] = acc0[nt][2];
        if (r1 < M && c1 < N) Cp[zoff + (unsigned long long)r1*N+c1] = acc0[nt][3];
    }
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + M_TILE + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) Cp[zoff + (unsigned long long)r0*N+c0] = acc1[nt][0];
        if (r0 < M && c1 < N) Cp[zoff + (unsigned long long)r0*N+c1] = acc1[nt][1];
        if (r1 < M && c0 < N) Cp[zoff + (unsigned long long)r1*N+c0] = acc1[nt][2];
        if (r1 < M && c1 < N) Cp[zoff + (unsigned long long)r1*N+c1] = acc1[nt][3];
    }
}
#undef ATLAS_MMA_S8

// Reduce ksplits fp32 partials [ksplits,M,N] → C [M,N] bf16.
extern "C" __global__ void int8_splitk_reduce(
    const float* __restrict__ Cp, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int ksplits
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long MN = (unsigned long long)M * N;
    if (idx >= MN) return;
    float s = 0.f;
    for (unsigned z = 0; z < ksplits; z++) s += Cp[(unsigned long long)z * MN + idx];
    C[idx] = __float2bfloat16(s);
}

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8 prefill, K_STEP=64 (`int8_gemm_t_m128_k64`). Same per-block
// (32) scale dequant but the K-loop advances 64 at a time → HALF the
// cp.async-wait + __syncthreads barriers vs the K=32 kernel (the gate/up
// bottleneck: 160 syncs at 8% occ). int8's 1-byte operands let the 64-wide
// A/B tiles fit smem (16KB each = 32KB) at 3 CTAs/SM — bf16 couldn't.
// Two m16n8k32 sub-MMAs per N-tile (K 0..32 with blk0 scale, 32..64 with blk1).
// Grid (ceil(N/128), ceil(M/128), 1)  Block 128.  K multiple of 64.
// ═══════════════════════════════════════════════════════════════════
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(128, 3)
void int8_gemm_t_m128_k64(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;
    const unsigned int nb = K >> 5;

    __shared__ signed char smem_Ai[2][2 * M_TILE][64];   // 16384 B
    __shared__ signed char smem_Bi[2][N_TILE_LG][64];     // 16384 B
    __shared__ float smem_As[2][2 * M_TILE][2];           // 2048 B (2 blocks)
    __shared__ float smem_Bs[2][N_TILE_LG][2];            // 2048 B

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0]=0.f; acc0[i][1]=0.f; acc0[i][2]=0.f; acc0[i][3]=0.f;
        acc1[i][0]=0.f; acc1[i][1]=0.f; acc1[i][2]=0.f; acc1[i][3]=0.f;
    }

    #define K64_LOADS(buf, kb) do { \
        { unsigned ar = threadIdx.x >> 2; unsigned ac = (threadIdx.x & 3) << 4; unsigned gc = (kb) + ac; \
          _Pragma("unroll") for (int rnd = 0; rnd < 4; rnd++) { \
            unsigned row = (unsigned)(rnd * 32) + ar; unsigned gr = cta_m + row; \
            cp_async_pred_16(&smem_Ai[(buf)][row][ac], &A_i8[(unsigned long long)gr * K + gc], (gr < M) && (gc + 15 < K)); } } \
        { unsigned my_n = threadIdx.x; unsigned gn = cta_n + my_n; bool v = (gn < N) && ((kb) + 63 < K); \
          _Pragma("unroll") for (int c = 0; c < 4; c++) \
            cp_async_pred_16(&smem_Bi[(buf)][my_n][c*16], &B_i8[(unsigned long long)gn * K + (kb) + c*16], v); } \
        { unsigned blk = (kb) >> 5; unsigned gr = cta_m + threadIdx.x; unsigned gn = cta_n + threadIdx.x; \
          smem_As[(buf)][threadIdx.x][0] = (gr < M) ? A_scale[(unsigned long long)gr * nb + blk]     : 0.f; \
          smem_As[(buf)][threadIdx.x][1] = (gr < M) ? A_scale[(unsigned long long)gr * nb + blk + 1] : 0.f; \
          smem_Bs[(buf)][threadIdx.x][0] = (gn < N) ? B_scale[(unsigned long long)gn * nb + blk]     : 0.f; \
          smem_Bs[(buf)][threadIdx.x][1] = (gn < N) ? B_scale[(unsigned long long)gn * nb + blk + 1] : 0.f; } \
    } while(0)

    // One sub-block (sb=0 → K bytes 0..32, sb=1 → 32..64), scale index = sb.
    #define K64_SUB(buf, sb) do { \
        float as00 = smem_As[(buf)][warp_m_offset + group_id][sb]; \
        float as01 = smem_As[(buf)][warp_m_offset + group_id + 8][sb]; \
        float as10 = smem_As[(buf)][M_TILE + warp_m_offset + group_id][sb]; \
        float as11 = smem_As[(buf)][M_TILE + warp_m_offset + group_id + 8][sb]; \
        unsigned off = (sb) * 32; \
        unsigned fr00 = warp_m_offset + group_id, fr01 = fr00 + 8; \
        unsigned a0c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][off+4*tid]; \
        unsigned a1c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][off+4*tid]; \
        unsigned a2c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][off+16+4*tid]; \
        unsigned a3c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][off+16+4*tid]; \
        unsigned fr10 = M_TILE + warp_m_offset + group_id, fr11 = fr10 + 8; \
        unsigned a0c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][off+4*tid]; \
        unsigned a1c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][off+4*tid]; \
        unsigned a2c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][off+16+4*tid]; \
        unsigned a3c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][off+16+4*tid]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt * 8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][off+4*tid]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][off+16+4*tid]; \
            float bs0 = smem_Bs[(buf)][nt*8 + tid*2][sb]; \
            float bs1 = smem_Bs[(buf)][nt*8 + tid*2 + 1][sb]; \
            int s0[4] = {0,0,0,0}, s1[4] = {0,0,0,0}; \
            ATLAS_MMA_S8(s0, a0c0,a1c0,a2c0,a3c0, b0,b1); \
            ATLAS_MMA_S8(s1, a0c1,a1c1,a2c1,a3c1, b0,b1); \
            acc0[nt][0] += (float)s0[0]*as00*bs0; acc0[nt][1] += (float)s0[1]*as00*bs1; \
            acc0[nt][2] += (float)s0[2]*as01*bs0; acc0[nt][3] += (float)s0[3]*as01*bs1; \
            acc1[nt][0] += (float)s1[0]*as10*bs0; acc1[nt][1] += (float)s1[1]*as10*bs1; \
            acc1[nt][2] += (float)s1[2]*as11*bs0; acc1[nt][3] += (float)s1[3]*as11*bs1; \
        } \
    } while(0)

    K64_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    int cur = 0;
    for (unsigned int kb = 64; kb < K; kb += 64) {
        int nxt = 1 - cur;
        K64_LOADS(nxt, kb);
        cp_async_commit();
        K64_SUB(cur, 0);
        K64_SUB(cur, 1);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    K64_SUB(cur, 0);
    K64_SUB(cur, 1);
    #undef K64_LOADS
    #undef K64_SUB

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc0[nt][3]);
    }
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + M_TILE + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc1[nt][3]);
    }
}
#undef ATLAS_MMA_S8

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8 prefill, 8-WARP (`int8_gemm_8w`). MMQ-class structural fix #1:
// 256 threads / 8 warps, each warp owns 16 M-rows of a 128(M)x128(N) tile
// (single chunk → 64 int32 acc/thread, half the 4-warp-2-chunk register
// pressure). Targets the measured 8.3% achieved occupancy (4-warp base):
// 8 warps/CTA x 2 CTAs/SM (launch_bounds 256,2) = 16 warps/SM = ~33% vs 8%.
// Pure int32 m16n8k32.s8.s8.s32 accumulate, per-32-block scale folded as a
// float FMA on the int32 partial (llama mmq.cuh:1212). Scales staged in smem.
// SMEM: A 2x128x32 + B 2x128x32 + scales ~17KB. Grid (N/128, M/128), block 256.
// ═══════════════════════════════════════════════════════════════════
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_8w(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;          // 0..255
    const unsigned int warp_id = t >> 5;         // 0..7
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;     // 0..7
    const unsigned int t4 = lane & 3;            // 0..3
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;      // this warp's M-row base in the 128 tile

    __shared__ signed char smem_Ai[2][128][32];  // 8192 B
    __shared__ signed char smem_Bi[2][128][32];  // 8192 B
    __shared__ float smem_As[2][128];            // 512 B
    __shared__ float smem_Bs[2][128];            // 512 B

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define W8_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define W8_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned fr0 = wrow + group_id, fr1 = fr0 + 8; \
        unsigned a0 = *(const unsigned*)&smem_Ai[(buf)][fr0][4*t4]; \
        unsigned a1 = *(const unsigned*)&smem_Ai[(buf)][fr1][4*t4]; \
        unsigned a2 = *(const unsigned*)&smem_Ai[(buf)][fr0][16+4*t4]; \
        unsigned a3 = *(const unsigned*)&smem_Ai[(buf)][fr1][16+4*t4]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*t4]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*t4]; \
            float bs0 = smem_Bs[(buf)][nt*8 + t4*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + t4*2 + 1]; \
            int s[4] = {0,0,0,0}; \
            ATLAS_MMA_S8(s, a0,a1,a2,a3, b0,b1); \
            acc[nt][0] += (float)s[0]*as0*bs0; acc[nt][1] += (float)s[1]*as0*bs1; \
            acc[nt][2] += (float)s[2]*as1*bs0; acc[nt][3] += (float)s[3]*as1*bs1; \
        } \
    } while(0)

    W8_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        W8_LOADS(nxt, kb); cp_async_commit();
        W8_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    W8_COMPUTE(cur);
    #undef W8_LOADS
    #undef W8_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef ATLAS_MMA_S8

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8, 8-warp + 3-STAGE staged-drain cp.async pipeline (int8_gemm_8w3).
// ncu showed int8_gemm_8w hit 33% occupancy but 84% no-eligible: every warp
// stalls at cp_async_wait_all (FULL drain) + __syncthreads every 32-K (160x).
// Fix: 3 buffers, keep 2 cp.async groups in flight, drain with wait_group<1>
// so load latency overlaps compute (llama's ~4-syncs-per-256K effect) instead
// of a full stall per K-step. Same int32 + per-block-scale math (correct).
// SMEM: 3x(A 128x32 + B 128x32) + 3x scales ~25.5KB. Grid (N/128,M/128) blk 256.
// ═══════════════════════════════════════════════════════════════════
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))
// wait until <=N cp.async groups remain in flight (N compile-time immediate)
#define CP_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;" :: "n"(N))

extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_8w3(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;
    const unsigned int nk = K >> 5;               // # of 32-K iterations

    __shared__ signed char smem_Ai[3][128][32];
    __shared__ signed char smem_Bi[3][128][32];
    __shared__ float smem_As[3][128];
    __shared__ float smem_Bs[3][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define W83_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define W83_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned fr0 = wrow + group_id, fr1 = fr0 + 8; \
        unsigned a0 = *(const unsigned*)&smem_Ai[(buf)][fr0][4*t4]; \
        unsigned a1 = *(const unsigned*)&smem_Ai[(buf)][fr1][4*t4]; \
        unsigned a2 = *(const unsigned*)&smem_Ai[(buf)][fr0][16+4*t4]; \
        unsigned a3 = *(const unsigned*)&smem_Ai[(buf)][fr1][16+4*t4]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*t4]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*t4]; \
            float bs0 = smem_Bs[(buf)][nt*8 + t4*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + t4*2 + 1]; \
            int s[4] = {0,0,0,0}; \
            ATLAS_MMA_S8(s, a0,a1,a2,a3, b0,b1); \
            acc[nt][0] += (float)s[0]*as0*bs0; acc[nt][1] += (float)s[1]*as0*bs1; \
            acc[nt][2] += (float)s[2]*as1*bs0; acc[nt][3] += (float)s[3]*as1*bs1; \
        } \
    } while(0)

    // prologue: issue stages 0,1 (2 in flight)
    W83_LOADS(0, 0);  cp_async_commit();
    if (nk > 1) { W83_LOADS(1, 32); cp_async_commit(); }
    CP_WAIT_GROUP(1);   // stage 0 landed (<=1 group remains)
    __syncthreads();

    int cur = 0;
    for (unsigned int ki = 0; ki < nk; ki++) {
        // prefetch stage ki+2
        unsigned kn = ki + 2;
        if (kn < nk) { int b = kn % 3; W83_LOADS(b, kn*32); cp_async_commit(); }
        W83_COMPUTE(cur);
        // ensure the stage we compute NEXT (ki+1) is landed: keep <=1 in flight
        if (ki + 1 < nk) { CP_WAIT_GROUP(1); __syncthreads(); }
        cur = (cur + 1) % 3;
    }
    #undef W83_LOADS
    #undef W83_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef ATLAS_MMA_S8
#undef CP_WAIT_GROUP

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8, 8-warp + ldmatrix.x4 A-fragment load (int8_gemm_8w_ldm).
// THE load-bearing MMQ lever: replace the manual scalar smem loads of the
// weight (A) fragment with ONE ldmatrix.sync.aligned.m8n8.x4.b16 (proven
// correct on GB10 by /workspace/ldmatrix_probe.cu). The int8 tile read as b16
// (2 int8 = 1 b16) puts the f16-fragment layout exactly on the m16n8k32.s8 A
// operand. Keep manual vectorized loads for B/activations (llama's asymmetry,
// mmq.cuh:1433 load_generic). Cuts the smem-load instruction count that pins
// the inner loop. Grid (N/128,M/128) block 256.
// ═══════════════════════════════════════════════════════════════════
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_8w_ldm(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ signed char smem_Ai[2][128][32];
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define L_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define L_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned a0,a1,a2,a3; \
        const int* xs = (const int*)&smem_Ai[(buf)][wrow][0] + (lane % 16)*8 + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*t4]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*t4]; \
            float bs0 = smem_Bs[(buf)][nt*8 + t4*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + t4*2 + 1]; \
            int s[4] = {0,0,0,0}; \
            ATLAS_MMA_S8(s, a0,a1,a2,a3, b0,b1); \
            acc[nt][0] += (float)s[0]*as0*bs0; acc[nt][1] += (float)s[1]*as0*bs1; \
            acc[nt][2] += (float)s[2]*as1*bs0; acc[nt][3] += (float)s[3]*as1*bs1; \
        } \
    } while(0)

    L_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        L_LOADS(nxt, kb); cp_async_commit();
        L_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    L_COMPUTE(cur);
    #undef L_LOADS
    #undef L_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8, 8-warp + ldmatrix.x4 for BOTH A AND B (int8_gemm_8w_ldmab).
// THE throughput fix: ncu pinned int8_gemm_8w_ldm at L1/TEX 56% busy because
// the B (.col n8k32) weight fragment was read with 32 scalar smem loads/K-step
// (16 nt x 2). Probe /workspace/ldmatrix_b_probe.cu proved (cosine 1.0, bit-exact)
// that ldmatrix.x2.b16 NON-trans yields that exact B-fragment with NO weight
// repack (row-major weights already match), and x4 loads TWO nt-minitiles per
// instruction (q0,q1=nt0 b0/b1 ; q2,q3=nt1 b0/b1). => 8 ldmatrix.x4 replace 32
// scalar loads (4x fewer B-load instrs), and the two paired MMAs (shared A, two
// B halves) add ILP to hide the smem-read latency. Same int32 + per-block scale
// fold (llama mmq.cuh:1206-1212). Grid (N/128,M/128) block 256.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_8w_ldmab(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ signed char smem_Ai[2][128][32];
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define LAB_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define LAB_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned a0,a1,a2,a3; \
        const int* xs = (const int*)&smem_Ai[(buf)][wrow][0] + (lane % 16)*8 + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        _Pragma("unroll") for (int p = 0; p < 8; p++) { \
            unsigned nt0 = 2*p, nt1 = 2*p+1; \
            unsigned brow = ((lane<16)?nt0:nt1)*8 + (lane&7); \
            const void* bxs = &smem_Bi[(buf)][brow][((lane>>3)&1)*16]; \
            unsigned q0,q1,q2,q3; \
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
                : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "l"(bxs)); \
            float b00 = smem_Bs[(buf)][nt0*8 + t4*2]; float b10 = smem_Bs[(buf)][nt0*8 + t4*2 + 1]; \
            float b01 = smem_Bs[(buf)][nt1*8 + t4*2]; float b11 = smem_Bs[(buf)][nt1*8 + t4*2 + 1]; \
            int s0[4]={0,0,0,0}, s1[4]={0,0,0,0}; \
            ATLAS_MMA_S8(s0, a0,a1,a2,a3, q0,q1); \
            ATLAS_MMA_S8(s1, a0,a1,a2,a3, q2,q3); \
            acc[nt0][0]+=(float)s0[0]*as0*b00; acc[nt0][1]+=(float)s0[1]*as0*b10; \
            acc[nt0][2]+=(float)s0[2]*as1*b00; acc[nt0][3]+=(float)s0[3]*as1*b10; \
            acc[nt1][0]+=(float)s1[0]*as0*b01; acc[nt1][1]+=(float)s1[1]*as0*b11; \
            acc[nt1][2]+=(float)s1[2]*as1*b01; acc[nt1][3]+=(float)s1[3]*as1*b11; \
        } \
    } while(0)

    LAB_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        LAB_LOADS(nxt, kb); cp_async_commit();
        LAB_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    LAB_COMPUTE(cur);
    #undef LAB_LOADS
    #undef LAB_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef ATLAS_MMA_S8

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8, 8-warp + ldmatrix.x4 A-fragment load (int8_gemm_8w_ilp).
// THE load-bearing MMQ lever: replace the manual scalar smem loads of the
// weight (A) fragment with ONE ldmatrix.sync.aligned.m8n8.x4.b16 (proven
// correct on GB10 by /workspace/ldmatrix_probe.cu). The int8 tile read as b16
// (2 int8 = 1 b16) puts the f16-fragment layout exactly on the m16n8k32.s8 A
// operand. Keep manual vectorized loads for B/activations (llama's asymmetry,
// mmq.cuh:1433 load_generic). Cuts the smem-load instruction count that pins
// the inner loop. Grid (N/128,M/128) block 256.
// ═══════════════════════════════════════════════════════════════════
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_8w_ilp(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ signed char smem_Ai[2][128][32];
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define L_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define L_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned a0,a1,a2,a3; \
        const int* xs = (const int*)&smem_Ai[(buf)][wrow][0] + (lane % 16)*8 + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        int sv[16][4]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*t4]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*t4]; \
            sv[nt][0]=0; sv[nt][1]=0; sv[nt][2]=0; sv[nt][3]=0; \
            ATLAS_MMA_S8(sv[nt], a0,a1,a2,a3, b0,b1); \
        } \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            float bs0 = smem_Bs[(buf)][nt*8 + t4*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + t4*2 + 1]; \
            acc[nt][0] += (float)sv[nt][0]*as0*bs0; acc[nt][1] += (float)sv[nt][1]*as0*bs1; \
            acc[nt][2] += (float)sv[nt][2]*as1*bs0; acc[nt][3] += (float)sv[nt][3]*as1*bs1; \
        } \
    } while(0)

    L_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        L_LOADS(nxt, kb); cp_async_commit();
        L_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    L_COMPUTE(cur);
    #undef L_LOADS
    #undef L_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef ATLAS_MMA_S8

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8 MMQ-tile (`int8_gemm_mmq`). The structural fix none of the 12
// incremental variants did: load a BIG 128-K tile (A+B+scales) into smem
// ONCE per outer step, then iterate the 4 sub-blocks of 32 WITHIN smem —
// inner loop has ZERO global loads + ZERO __syncthreads (vs ~160 before).
// Kills the SHORT_SCOREBOARD smem-read stall: the 4 sub-block MMAs + their
// ldmatrix loads pipeline freely. 8-warp, ldmatrix.x4 A (verified), manual B,
// per-32-block scale folded as float FMA on int32 (llama mmq.cuh:1206-1212).
// SMEM: A 128x128 + B 128x128 + scales 128x4x2 = ~36KB -> 2 CTAs/SM.
// Grid (N/128, M/128), block 256.  BK=128 (K multiple of 128).
// ═══════════════════════════════════════════════════════════════════
#define MMQ_BK 128
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_mmq(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ signed char sA[128][MMQ_BK];      // 16384 B
    __shared__ signed char sB[128][MMQ_BK];      // 16384 B
    __shared__ float sAs[128][MMQ_BK/32];        // 128*4*4 = 2048 B
    __shared__ float sBs[128][MMQ_BK/32];        // 2048 B

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    for (unsigned int kb0 = 0; kb0 < K; kb0 += MMQ_BK) {
        // --- load 128x128 of A and B (4 cp.async.16 each), + 4-block scales ---
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            unsigned lin = c*256 + t;          // 0..1023
            unsigned row = lin >> 3;           // 0..127
            unsigned col = (lin & 7) << 4;     // 0,16,..,112
            unsigned gcA = kb0 + col, gcB = kb0 + col;
            cp_async_pred_16(&sA[row][col], &A_i8[(unsigned long long)(cta_m+row)*K + gcA], (cta_m+row<M)&&(gcA+15<K));
            cp_async_pred_16(&sB[row][col], &B_i8[(unsigned long long)(cta_n+row)*K + gcB], (cta_n+row<N)&&(gcB+15<K));
        }
        if (t < 128) {
            unsigned blk0 = kb0 >> 5;
            #pragma unroll
            for (int b = 0; b < MMQ_BK/32; b++) {
                unsigned gr = cta_m + t, gn = cta_n + t;
                sAs[t][b] = (gr<M)?A_scale[(unsigned long long)gr*nb + blk0 + b]:0.f;
                sBs[t][b] = (gn<N)?B_scale[(unsigned long long)gn*nb + blk0 + b]:0.f;
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // --- inner loop over the 4 sub-blocks, NO sync, NO global ---
        #pragma unroll
        for (int sb = 0; sb < MMQ_BK/32; sb++) {
            float as0 = sAs[wrow + group_id][sb];
            float as1 = sAs[wrow + group_id + 8][sb];
            unsigned a0,a1,a2,a3;
            const int* xs = (const int*)&sA[wrow][0] + (lane % 16)*(MMQ_BK/4) + sb*8 + (lane / 16)*4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs));
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned nc = nt*8 + group_id;
                unsigned b0 = *(const unsigned*)&sB[nc][sb*32 + 4*t4];
                unsigned b1 = *(const unsigned*)&sB[nc][sb*32 + 16 + 4*t4];
                float bs0 = sBs[nt*8 + t4*2][sb];
                float bs1 = sBs[nt*8 + t4*2 + 1][sb];
                int s[4] = {0,0,0,0};
                ATLAS_MMA_S8(s, a0,a1,a2,a3, b0,b1);
                acc[nt][0] += (float)s[0]*as0*bs0; acc[nt][1] += (float)s[1]*as0*bs1;
                acc[nt][2] += (float)s[2]*as1*bs0; acc[nt][3] += (float)s[3]*as1*bs1;
            }
        }
        __syncthreads();  // protect smem reuse for next big tile
    }

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef ATLAS_MMA_S8
#undef MMQ_BK

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8, OCCUPANCY-first (int8_gemm_8w_pipe). Findings from 15 prior
// variants: the wall is ldmatrix/smem-read LATENCY (short_scoreboard 43%),
// NOT load count (ldmab proved 4x fewer B-loads = 0 speedup). ncu: acc[16][4]
// =64 regs caps occupancy at 2 CTAs / 21-33%. Fix: HALVE the per-warp output
// tile (16Mx64N = 8 nt, acc[8][4]=32 regs) and double the warps to 16 (block
// 512), 2 warp-cols over N. Goal: 2 CTAs x 512 = 32 warps/SM (~66% occ) to
// hide the latency. ldmatrix.x4 for A and B (both proven). Grid (N/128,M/128).
// ═══════════════════════════════════════════════════════════════════
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(512, 2)
void int8_gemm_8w_pipe(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;          // 0..511
    const unsigned int warp_id = t >> 5;         // 0..15
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wm = warp_id & 7;          // M-group 0..7  -> rows wm*16
    const unsigned int wn = warp_id >> 3;         // N-half  0..1  -> cols wn*64
    const unsigned int wrow = wm * 16;
    const unsigned int ncol0 = wn * 64;           // this warp's N base (8 nt of 8)

    __shared__ signed char smem_Ai[2][128][32];
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    // 512 threads load 128x32 A + 128x32 B (each thread one 16-byte chunk: 128*32/16=256)
    #define P_LOADS(buf, kb) do { \
        if (t < 256) { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        else { unsigned u = t - 256; unsigned an = u >> 1; unsigned ac = (u & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; } \
        else if (t < 256) { unsigned blk=(kb)>>5; unsigned gn=cta_n+(t-128); smem_Bs[(buf)][t-128] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define P_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned a0,a1,a2,a3; \
        const int* xs = (const int*)&smem_Ai[(buf)][wrow][0] + (lane % 16)*8 + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        _Pragma("unroll") for (int p = 0; p < 4; p++) { \
            unsigned nt0 = 2*p, nt1 = 2*p+1; \
            unsigned brow = ncol0 + ((lane<16)?nt0:nt1)*8 + (lane&7); \
            const void* bxs = &smem_Bi[(buf)][brow][((lane>>3)&1)*16]; \
            unsigned q0,q1,q2,q3; \
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
                : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "l"(bxs)); \
            float b00 = smem_Bs[(buf)][ncol0 + nt0*8 + t4*2]; float b10 = smem_Bs[(buf)][ncol0 + nt0*8 + t4*2 + 1]; \
            float b01 = smem_Bs[(buf)][ncol0 + nt1*8 + t4*2]; float b11 = smem_Bs[(buf)][ncol0 + nt1*8 + t4*2 + 1]; \
            int s0[4]={0,0,0,0}, s1[4]={0,0,0,0}; \
            ATLAS_MMA_S8(s0, a0,a1,a2,a3, q0,q1); \
            ATLAS_MMA_S8(s1, a0,a1,a2,a3, q2,q3); \
            acc[nt0][0]+=(float)s0[0]*as0*b00; acc[nt0][1]+=(float)s0[1]*as0*b10; \
            acc[nt0][2]+=(float)s0[2]*as1*b00; acc[nt0][3]+=(float)s0[3]*as1*b10; \
            acc[nt1][0]+=(float)s1[0]*as0*b01; acc[nt1][1]+=(float)s1[1]*as0*b11; \
            acc[nt1][2]+=(float)s1[2]*as1*b01; acc[nt1][3]+=(float)s1[3]*as1*b11; \
        } \
    } while(0)

    P_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        P_LOADS(nxt, kb); cp_async_commit();
        P_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    P_COMPUTE(cur);
    #undef P_LOADS
    #undef P_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned c0 = cta_n + ncol0 + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef ATLAS_MMA_S8

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8 BANK-CONFLICT FIX (int8_gemm_padA). llama-MMQ spec (mmq.cuh:222,
// 751-758): the weight smem row stride must make the 16 ldmatrix.x4 row-base
// addresses hit distinct banks. My prior int8 kernels used 32B rows (8 int32)
// => (lane%16)*8 mod 32 collides 4-way (rows 0,4,8,12 -> bank0) = the hidden
// short_scoreboard. Pad the A row to PADI int32 (>8) so row bases spread across
// banks. Identical math/structure to int8_gemm_8w_ldm; only smem_Ai stride
// changes. PADI swept via ncu (9,11,19 distinct in the naive model; real
// ldmatrix model is 16B-granular so ncu measures the truth). B stays manual.
// ═══════════════════════════════════════════════════════════════════
#define PADI 12                // int32 per A smem row (32B data + 16B pad); 16B-aligned for
                               // ldmatrix AND r*3 mod 8 = all-distinct 16B bank groups (llama's =12)
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_padA(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ int   smem_Ai[2][128][PADI];   // padded: PADI int32/row (>8 kills bank conflict)
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define PA_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(((signed char*)&smem_Ai[(buf)][ar][0]) + ac, &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define PA_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned a0,a1,a2,a3; \
        const int* xs = &smem_Ai[(buf)][wrow][0] + (lane % 16)*PADI + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*t4]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*t4]; \
            float bs0 = smem_Bs[(buf)][nt*8 + t4*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + t4*2 + 1]; \
            int s[4] = {0,0,0,0}; \
            ATLAS_MMA_S8(s, a0,a1,a2,a3, b0,b1); \
            acc[nt][0] += (float)s[0]*as0*bs0; acc[nt][1] += (float)s[1]*as0*bs1; \
            acc[nt][2] += (float)s[2]*as1*bs0; acc[nt][3] += (float)s[3]*as1*bs1; \
        } \
    } while(0)

    PA_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        PA_LOADS(nxt, kb); cp_async_commit();
        PA_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    PA_COMPUTE(cur);
    #undef PA_LOADS
    #undef PA_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef ATLAS_MMA_S8
#undef PADI

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8 FAITHFUL llama-MMQ port (int8_gemm_faith). Combines ALL levers the
// 17 prior variants applied in isolation (each did nothing alone because it only
// exposed the next stall): (1) big K-tile loaded ONCE per outer step (K_TILE=64,
// 2 sub-blocks) via cp.async; (2) BANK-FIXED weight smem stride 36 int32/row
// (36/4=9, r*9 mod 8 distinct => zero ldmatrix bank conflict, 16B-aligned);
// (3) REGISTER PRE-STAGE: all weight ldmatrix fragments + scales for the tile
// hoisted into regs BEFORE the j-MMA loop so the ldmatrix latencies overlap the
// mma issue (llama mmq.cuh:1399-1424); (4) activations via cheap scalar load
// (llama load_generic, mma.cuh:698). Tiling = llama: warp owns 32 weight-rows
// (ntx=2 minitiles) x 64 tokens; acc[2][8][4]=64 fp32. Weights = ldmatrix 16-row
// A-operand (N), tokens = 8-col B-operand (M). grid (N/128, M/128) block 256.
// ═══════════════════════════════════════════════════════════════════
#define FK_TILE 64
#define FK_SB   (FK_TILE/32)        // 2 sub-blocks of 32-K
#define FW_STRIDE 36                // int32/row of weight smem (64 data int? no: see below)
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith(
    const signed char* __restrict__ A_i8,   // activations [M,K]  (tokens, B-operand)
    const signed char* __restrict__ B_i8,   // weights     [N,K]  (features, A/ldmatrix-operand)
    const float* __restrict__ A_scale,       // [M, K/32]
    const float* __restrict__ B_scale,       // [N, K/32]
    __nv_bfloat16* __restrict__ C,           // [M, N]
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;   // weight-row (N) base
    const unsigned int cta_m = blockIdx.y * 128;   // token (M) base
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;          // N row-group 0..3 -> rows ng*32..+31
    const unsigned int mh = warp_id & 1;           // M half 0/1 -> cols mh*64..+63
    const unsigned int nb = K >> 5;

    // weight smem: 128 N-rows x FK_TILE-K (=64 int8 = 16 int32) padded to FW_STRIDE=36? No:
    // 64 int8 = 16 int32 data; pad to stride S with S/4 odd for distinct 16B groups & 16B-align.
    // S=20 (16 data + 4 pad): 20/4=5, r*5 mod 8 = {0,5,2,7,4,1,6,3} distinct. 16B-aligned.
    __shared__ int  sW[128][20];      // weights  (int32)
    __shared__ int  sA[128][20];      // activations (int32), scalar-loaded
    __shared__ float sWs[128][FK_SB]; // weight scales per N-row per sub-block
    __shared__ float sAs[128][FK_SB]; // act scales per M-col per sub-block

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += FK_TILE) {
        // --- load 128 x 64-K of weights + activations (each thread: 2 chunks of 16B) ---
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            unsigned lin = c*256 + t;            // 0..511
            unsigned row = lin >> 2;             // 0..127
            unsigned col = (lin & 3) << 4;       // 0,16,32,48 -> within 64-K
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<FK_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        // --- PRE-STAGE: all weight frags + scales into regs ---
        unsigned WA[2][FK_SB][4];
        float wsc[2][2][FK_SB];      // [minitile][rowhalf][sb]
        #pragma unroll
        for (int n=0;n<2;n++){
            unsigned wbase_row = ng*32 + n*16;
            #pragma unroll
            for (int sb=0; sb<FK_SB; sb++){
                const int* xs = &sW[wbase_row][0] + (lane%16)*20 + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][sb][0]),"=r"(WA[n][sb][1]),"=r"(WA[n][sb][2]),"=r"(WA[n][sb][3]) : "l"(xs));
                wsc[n][0][sb] = sWs[wbase_row + lane/4][sb];
                wsc[n][1][sb] = sWs[wbase_row + 8 + lane/4][sb];
            }
        }
        // --- j-loop over 8 token-cols of 8, MMA + fold ---
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol0 = mh*64 + j*8;          // this 8-token chunk base
            float asc[2][FK_SB];
            #pragma unroll
            for (int sb=0;sb<FK_SB;sb++){
                asc[0][sb] = sAs[mcol0 + (lane%4)*2][sb];
                asc[1][sb] = sAs[mcol0 + (lane%4)*2 + 1][sb];
            }
            #pragma unroll
            for (int sb=0;sb<FK_SB;sb++){
                // B (activation) frag via scalar load_generic: rows mcol0+lane/4, cols sb*8 + {lane%4, +4}
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    ATLAS_MMA_S8(s, WA[n][sb][0],WA[n][sb][1],WA[n][sb][2],WA[n][sb][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0][sb]*asc[0][sb];
                    acc[n][j][1]+=(float)s[1]*wsc[n][0][sb]*asc[1][sb];
                    acc[n][j][2]+=(float)s[2]*wsc[n][1][sb]*asc[0][sb];
                    acc[n][j][3]+=(float)s[3]*wsc[n][1][sb]*asc[1][sb];
                }
            }
        }
        __syncthreads();
    }

    // --- store: C[m, n]. tile_C row = N-feature, col = M-token => write C[mcol, nrow] ---
    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;     // N feature (output col)
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2; // M token (output row)
            unsigned r0=mcol, r1=mcol;            // rows (M)
            unsigned cN0=nrow0, cN1=nrow0+8;      // cols (N), from l/2
            // l=0:(r=mcol,   c=nrow0)  l=1:(r=mcol+1, c=nrow0)
            // l=2:(r=mcol,   c=nrow0+8)l=3:(r=mcol+1, c=nrow0+8)
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
            (void)r0;(void)r1;
        }
    }
}
#undef ATLAS_MMA_S8
#undef FK_TILE
#undef FK_SB
#undef FW_STRIDE

// int8 W4A8 FAITH2 — structural follow-up to int8_gemm_faith. Two changes only:
//  (1) BIG K-tile (F2_TILE=128, 4 sub-blocks) loaded ONCE per outer step so the
//      cp.async + 2 __syncthreads amortize over 4× the MMA work (K=5120 => 40
//      outer steps instead of 80; halves the sync/commit traffic that the ncu
//      SHORT_SCOREBOARD stall feeds on).
//  (2) ROLLING weight pre-stage: sb-loop is now OUTER, j-loop INNER, so only ONE
//      sub-block's weight ldmatrix frag (WA[2][4]=8 regs) is live at a time instead
//      of all F2_SB of them (would be 32 regs at SB=4, spilling atop acc[2][8][4]).
//      The hoisted ldmatrix above the 8-wide j-loop still overlaps its latency with
//      the first MMAs, and the 8 independent B scalar-loads give the ILP. This
//      decouples K-tile size from register pressure => F2_TILE can scale to 256.
// smem weight/act row stride F2W int32: data = F2_TILE/4 int32; pad so (F2W/4) is
// odd => r*(F2W/4) mod 8 distinct for the 8 ldmatrix rows (bank-conflict-free) and
// F2W multiple of 4 (16B-aligned rows). F2_TILE=128: data=32, F2W=36 (9 odd ✓).
// ═══════════════════════════════════════════════════════════════════
#define F2_TILE 128
#define F2_SB   (F2_TILE/32)
#define F2W     36                  // weight/act smem int32 row stride (128-K=32 data +4 pad)
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith2(
    const signed char* __restrict__ A_i8,   // activations [M,K] (tokens, B-operand)
    const signed char* __restrict__ B_i8,   // weights     [N,K] (features, A/ldmatrix-operand)
    const float* __restrict__ A_scale,       // [M, K/32]
    const float* __restrict__ B_scale,       // [N, K/32]
    __nv_bfloat16* __restrict__ C,           // [M, N]
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;          // N row-group 0..3
    const unsigned int mh = warp_id & 1;           // M half 0/1
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F2W];
    __shared__ int   sA[128][F2W];
    __shared__ float sWs[128][F2_SB];
    __shared__ float sAs[128][F2_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F2_TILE) {
        // load 128 x F2_TILE-K: 128 rows * (F2_TILE/16) 16B-chunks/row = 128*F2_TILE/16
        // total chunks / 256 threads = F2_TILE/32 chunks per thread. CPR = chunks/row.
        const unsigned F2_CPR = F2_TILE/16;  // constant -> div=shift, mod=mask
        #pragma unroll
        for (int c = 0; c < F2_TILE/32; c++) {
            unsigned lin = c*256 + t;            // 0..(128*F2_TILE/16 - 1)
            unsigned row = lin / F2_CPR;
            unsigned col = (lin % F2_CPR) << 4; // byte col within F2_TILE
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F2_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        // sb OUTER (rolling weight pre-stage), j INNER.
        #pragma unroll
        for (int sb=0; sb<F2_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*F2W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }
            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                float asc0 = sAs[mcol0 + (lane%4)*2][sb];
                float asc1 = sAs[mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    ATLAS_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*asc0;
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*asc1;
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*asc0;
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*asc1;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef ATLAS_MMA_S8
#undef F2_TILE
#undef F2_SB
#undef F2W

// int8 W4A8 FAITH3 — faith2 + B-fragment ILP. ncu on faith2 showed the dominant
// stall is SHORT_SCOREBOARD (smem-read latency) 47% of 7.2 cyc/instr; occupancy
// is dead (raising CTA/SM regressed). The remaining lever is ILP on the smem reads:
// faith2 loads the activation B-fragment (2 scalar smem loads) INSIDE the j-loop,
// serialized 1:1 with the MMA that consumes it. faith3 HOISTS all 8 j B-fragments
// + act-scales to the top of each sb-iteration (16 scalar loads issue back-to-back
// => the loads' latencies overlap each other and the first MMAs), exactly mirroring
// the weight pre-stage. Costs bb[8][2]=16 + aa[8][2]=16 regs atop acc[2][8][4].
// ═══════════════════════════════════════════════════════════════════
#define F3_TILE 128
#define F3_SB   (F3_TILE/32)
#define F3W     36
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith3(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F3W];
    __shared__ int   sA[128][F3W];
    __shared__ float sWs[128][F3_SB];
    __shared__ float sAs[128][F3_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F3_TILE) {
        const unsigned F3_CPR = F3_TILE/16;
        #pragma unroll
        for (int c = 0; c < F3_TILE/32; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin / F3_CPR;
            unsigned col = (lin % F3_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F3_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int sb=0; sb<F3_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*F3W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }
            // PRE-STAGE all 8 j B-fragments + act-scales (smem-read ILP)
            unsigned bb[8][2];
            float    aa[8][2];
            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                bb[j][0] = abase[lane%4];
                bb[j][1] = abase[lane%4 + 4];
                aa[j][0] = sAs[mcol0 + (lane%4)*2][sb];
                aa[j][1] = sAs[mcol0 + (lane%4)*2 + 1][sb];
            }
            #pragma unroll
            for (int j=0;j<8;j++){
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    ATLAS_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], bb[j][0],bb[j][1]);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*aa[j][0];
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*aa[j][1];
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*aa[j][0];
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*aa[j][1];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef ATLAS_MMA_S8
#undef F3_TILE
#undef F3_SB
#undef F3W

// int8 W4A8 FAITH4 — faith2 inner loop, 512-thread CTA for OCCUPANCY. ncu on
// faith2/3 (44.7 TFLOP/s) showed the kernel is latency-bound: SHORT_SCOREBOARD
// (smem-read) ~38%, Compute(SM) only 26%, Achieved Occupancy 16.6%. Raising CTAs
// via launch_bounds(256,2) spilled the 64-reg acc[2][8][4] tile and regressed.
// faith4 keeps the SAME 128x128 output tile but uses 512 threads (16 warps): each
// warp owns 32 N-rows x 32 M-tokens (4x4 warp grid) => acc[2][4][4]=32 regs/thread
// (half of faith2). Same total acc registers, but spread over 2x the warps => 2x
// the warps/SM at the same register budget, directly attacking the occupancy wall
// WITHOUT spill. Weight ldmatrix A-frag + scalar B-frag + per-block scale fold all
// identical to faith2. block 512. grid (N/128, M/128).
// ═══════════════════════════════════════════════════════════════════
#define F4_TILE 128
#define F4_SB   (F4_TILE/32)
#define F4W     36
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(512, 1)
void int8_gemm_faith4(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;            // 0..511
    const unsigned int warp_id = t >> 5;           // 0..15
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id & 3;           // N-group 0..3 -> rows ng*32..+31
    const unsigned int mg = warp_id >> 2;          // M-group 0..3 -> tokens mg*32..+31
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F4W];
    __shared__ int   sA[128][F4W];
    __shared__ float sWs[128][F4_SB];
    __shared__ float sAs[128][F4_SB];

    float acc[2][4][4];                            // 2 N-minitiles x 4 M-chunks x 4
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<4;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F4_TILE) {
        const unsigned F4_CPR = F4_TILE/16;
        // 128 rows * 8 chunks/row = 1024 chunks / 512 threads = 2 chunks/thread
        #pragma unroll
        for (int c = 0; c < (128*F4_TILE/16)/512; c++) {
            unsigned lin = c*512 + t;
            unsigned row = lin / F4_CPR;
            unsigned col = (lin % F4_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F4_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int sb=0; sb<F4_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*F4W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }
            #pragma unroll
            for (int j=0;j<4;j++){
                unsigned mcol0 = mg*32 + j*8;
                float asc0 = sAs[mcol0 + (lane%4)*2][sb];
                float asc1 = sAs[mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    ATLAS_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*asc0;
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*asc1;
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*asc0;
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*asc1;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<4;j++){
            unsigned mcol = cta_m + mg*32 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef ATLAS_MMA_S8
#undef F4_TILE
#undef F4_SB
#undef F4W

// ═══════════════════════════════════════════════════════════════════
// int8 W4A8 MMQ2 — faith2 + TRUE 2-stage double-buffered cp.async pipeline.
// faith2 plateaued at 44.7 because it does `cp.async.wait_group 0 + __syncthreads`
// then computes => compute STALLS for the full K-tile load every outer step (zero
// load/compute overlap). ncu: SHORT_SCOREBOARD latency, nothing saturated. MMQ2
// double-buffers sW/sA (2 smem buffers): issue the load for tile k+1, then
// `wait_group<1>` so the PREVIOUS tile (k) is ready while tile k+1's cp.async
// streams in the background, overlapping the load latency with the MMA work of
// tile k. Identical math/indexing/output to faith2 (cosine must match 0.999999).
// smem: sW/sA[2][128][36] int + scales[2][128][4] f32 = 80KB -> 1 CTA/SM (256,1).
// ═══════════════════════════════════════════════════════════════════
#define M2_TILE 128
#define M2_SB   (M2_TILE/32)
#define M2W     36
#define ATLAS_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_mmq2(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[2][128][M2W];
    __shared__ int   sA[2][128][M2W];
    __shared__ float sWs[2][128][M2_SB];
    __shared__ float sAs[2][128][M2_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    const unsigned int ntiles = (K + M2_TILE - 1) / M2_TILE;
    const unsigned int M2_CPR = M2_TILE/16;     // 16B-chunks per row

    #define M2_LOAD_TILE(buf, kb_) { \
        _Pragma("unroll") \
        for (int c = 0; c < M2_TILE/32; c++) { \
            unsigned lin = c*256 + t; \
            unsigned row = lin / M2_CPR; \
            unsigned col = (lin % M2_CPR) << 4; \
            unsigned gk  = (kb_) + col; \
            signed char* wdst = ((signed char*)&sW[buf][row][0]) + col; \
            signed char* adst = ((signed char*)&sA[buf][row][0]) + col; \
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K)); \
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K)); \
        } \
        if (t < 128) { \
            unsigned blk = (kb_) >> 5; \
            _Pragma("unroll") \
            for (int s=0;s<M2_SB;s++){ \
                sWs[buf][t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f; \
                sAs[buf][t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f; \
            } \
        } \
    }

    // prologue: kick off tile 0
    M2_LOAD_TILE(0, 0); cp_async_commit();

    for (unsigned int i = 0; i < ntiles; i++) {
        unsigned int cur = i & 1, nxt = (i + 1) & 1;
        // issue next tile's load (overlaps with this tile's compute)
        if (i + 1 < ntiles) { M2_LOAD_TILE(nxt, (i+1)*M2_TILE); cp_async_commit(); }
        // wait so the CURRENT tile is ready (leave the just-issued next in flight)
        if (i + 1 < ntiles) cp_async_wait_group<1>(); else cp_async_wait_all();
        __syncthreads();

        #pragma unroll
        for (int sb=0; sb<M2_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[cur][wbase_row][0] + (lane%16)*M2W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[cur][wbase_row + lane/4][sb];
                wsc[n][1] = sWs[cur][wbase_row + 8 + lane/4][sb];
            }
            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                float asc0 = sAs[cur][mcol0 + (lane%4)*2][sb];
                float asc1 = sAs[cur][mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[cur][mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    ATLAS_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*asc0;
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*asc1;
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*asc0;
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*asc1;
                }
            }
        }
        __syncthreads();  // current buffer fully read before it is reused at i+2
    }
    #undef M2_LOAD_TILE

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef ATLAS_MMA_S8
#undef M2_TILE
#undef M2_SB
#undef M2W

// ═══════════════════════════════════════════════════════════════════
// REQUANT kernels feeding the int8 W4A8 prefill GEMM (faith2).
//   requant_w_nvfp4_int8 : NVFP4 weights [N,K/2] packed E2M1 + [N,K/16] E4M3
//     scales + per-tensor scale2  ->  int8 [N,K] + per-32 float scale [N,K/32].
//     Run ONCE per weight at load. E2M1 levels {0,.5,1,1.5,2,3,4,6} map cleanly
//     into int8 (max level 6 -> 127), so this is near-lossless. One thread per
//     (n, 32-block): reads 32 nibbles (2 E4M3 sub-block scales), finds the block
//     max, writes 32 int8 + 1 scale = max/127.
//   requant_a_bf16_int8 : bf16 activations [M,K] -> int8 [M,K] + per-32 float
//     scale [M,K/32]. Run per-prefill (~1.4% of GEMM time). One thread per
//     (m, 32-block): block max-abs -> scale=max/127 -> round.
// Layout matches faith2's A_i8[M,K]/B_i8[N,K] + A_scale[M,K/32]/B_scale[N,K/32].
// ═══════════════════════════════════════════════════════════════════
// Portable E4M3 decode: standard on real NVIDIA (__nv_fp8_e4m3), software on SCALE/HIP.
__device__ __forceinline__ float atlas_e4m3_decode_any(unsigned char b) {
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
    return scl_fp8(b);
#else
    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = b; return (float)fp8;
#endif
}

extern "C" __global__
void requant_w_nvfp4_int8(
    const unsigned char* __restrict__ W_packed,  // [N, K/2] E2M1 nibbles
    const unsigned char* __restrict__ W_e4m3,    // [N, K/16] per-16 E4M3 scales
    const float scale2,                          // per-tensor
    signed char* __restrict__ W_i8,              // [N, K] out
    float* __restrict__ W_scale,                 // [N, K/32] out
    unsigned int N, unsigned int K
) {
    unsigned long long blk = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nblocks = (unsigned long long)N * (K >> 5);
    if (blk >= nblocks) return;
    unsigned int nb = K >> 5;
    unsigned int n  = (unsigned int)(blk / nb);
    unsigned int kb = (unsigned int)(blk % nb) * 32;   // K base of this 32-block

    // dequant 32 values, track max-abs
    float vals[32];
    float maxa = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        unsigned int k = kb + i;
        unsigned char pb = W_packed[(unsigned long long)n * (K>>1) + (k>>1)];
        unsigned int nib = (k & 1) ? (pb >> 4) : (pb & 0xF);
        float s16 = atlas_e4m3_decode_any(W_e4m3[(unsigned long long)n * (K>>4) + (k>>4)]) * scale2;
        float v = E2M1_LUT[nib] * s16;
        vals[i] = v;
        float a = fabsf(v);
        if (a > maxa) maxa = a;
    }
    float sc = (maxa > 0.f) ? (maxa / 127.0f) : 1.0f;
    float inv = 1.0f / sc;
    W_scale[blk] = sc;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int q = __float2int_rn(vals[i] * inv);
        q = max(-127, min(127, q));
        W_i8[(unsigned long long)n * K + kb + i] = (signed char)q;
    }
}

extern "C" __global__
void requant_a_bf16_int8(
    const __nv_bfloat16* __restrict__ A_bf16,    // [M, K]
    signed char* __restrict__ A_i8,              // [M, K] out
    float* __restrict__ A_scale,                 // [M, K/32] out
    unsigned int M, unsigned int K
) {
    unsigned long long blk = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nblocks = (unsigned long long)M * (K >> 5);
    if (blk >= nblocks) return;
    unsigned int nb = K >> 5;
    unsigned int m  = (unsigned int)(blk / nb);
    unsigned int kb = (unsigned int)(blk % nb) * 32;

    float vals[32];
    float maxa = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float v = __bfloat162float(A_bf16[(unsigned long long)m * K + kb + i]);
        vals[i] = v;
        float a = fabsf(v);
        if (a > maxa) maxa = a;
    }
    float sc = (maxa > 0.f) ? (maxa / 127.0f) : 1.0f;
    float inv = 1.0f / sc;
    A_scale[blk] = sc;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int q = __float2int_rn(vals[i] * inv);
        q = max(-127, min(127, q));
        A_i8[(unsigned long long)m * K + kb + i] = (signed char)q;
    }
}
