// SPDX-License-Identifier: AGPL-3.0-only
// D2 carry-isolation harness: proves/refutes the multi-chunk h_state carry in
// libatlasgdn.so (atlas_gdn_prefill_packed_managed) WITHOUT running the model.
//
// The SAME T tokens go through the FlashInfer GDN as (A) one call vs (B) N
// carried chunks. The carry round-trip (post-call FI S[v][k] -> Atlas S[k][v]
// transpose, pre-call Atlas -> FI transpose of the fp32 state) is an exact fp32
// permutation, and aligned chunk boundaries (multiples of BLK=64) preserve the
// kernel's 64-token block partitioning, so a CORRECT carry must reproduce the
// single-call result BIT-EXACTLY (the in-register fp32 state round-trips
// through fp32 global memory losslessly). Divergence that GROWS with chunk
// count = carry defect (D2 confirmed). A ragged split (boundaries not %64)
// legitimately regroups intra-block math (WY inverse over different 64-token
// blocks): expect rounding-level noise there (cos ~ 1), NOT bit-exactness —
// it exercises the masked-tail + carry-after-ragged-chunk path that prod hits
// on the LAST prefill chunk before decode.
//
// Build/run (gx10, quiet GPU; links the existing checked-in lib — NO re-export):
//   cd 3rdparty_patches/gdn_aot
//   CUTE=/home/ms/spark-vllm-docker/.venv/lib/python3.13/site-packages/nvidia_cutlass_dsl/lib
//   g++ -O2 gdn_harness_carry.cpp -o hcarry -I/usr/local/cuda/include \
//       ./libatlasgdn.so -lcudart -L/usr/local/cuda/lib64 -L$CUTE \
//       -lcute_dsl_runtime -Wl,-rpath,$CUTE
//   LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:.:$CUTE:/usr/local/cuda/lib64 \
//       CUTE_DSL_ARCH=sm_121a ./hcarry
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void atlas_gdn_load();
extern "C" int atlas_gdn_prefill_packed_managed(
    void* qkv, void* gate_beta, void* output, void* h_state,
    float scale, int total_seqlen, int nk, int nv, int kd, int vd,
    int conv_dim, int gb_stride, int num_seqs, void* stream);

static const int T = 16384, NK = 16, NV = 32, KD = 128, VD = 128;
static const int KEY_DIM = NK * KD, VALUE_DIM = NV * VD;
static const int CONV_DIM = 2 * KEY_DIM + VALUE_DIM; // packed [Q|K|V] row, bf16
static const int GB = 2 * NV;                        // interleaved [gate|beta] row, fp32
static const float SCALE = 0.08838834764831843f;     // 1/sqrt(128)
static const size_t STATE_N = (size_t)NV * KD * VD;  // fp32 elems, num_seqs=1

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static float frand() { // xorshift64* -> [0,1)
  g_rng ^= g_rng >> 12; g_rng ^= g_rng << 25; g_rng ^= g_rng >> 27;
  return (float)((g_rng * 0x2545F4914F6CDD1Dull) >> 40) * (1.0f / 16777216.0f);
}

struct Cmp { double max_abs; double cos; int nan; };
static Cmp cmp_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n) {
  Cmp c = {0, 0, 0}; double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < n; i++) {
    float x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
    if (isnan(x) || isnan(y)) c.nan++;
    double e = fabs((double)x - y); if (e > c.max_abs) c.max_abs = e;
    dot += (double)x * y; na += (double)x * x; nb += (double)y * y;
  }
  c.cos = dot / (sqrt(na) * sqrt(nb) + 1e-30); return c;
}
static Cmp cmp_f32(const float* a, const float* b, size_t n) {
  Cmp c = {0, 0, 0}; double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < n; i++) {
    if (isnan(a[i]) || isnan(b[i])) c.nan++;
    double e = fabs((double)a[i] - b[i]); if (e > c.max_abs) c.max_abs = e;
    dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
  }
  c.cos = dot / (sqrt(na) * sqrt(nb) + 1e-30); return c;
}

// Run T tokens as chunks of `chunk` tokens (last chunk ragged if T % chunk != 0),
// carrying h_state exactly as Atlas's prefill_chunk loop does. Returns #chunks.
static int run_chunked(void* dqkv, void* dgb, void* dout, void* dst,
                       int chunk, cudaStream_t s) {
  CK(cudaMemsetAsync(dst, 0, STATE_N * 4, s));
  CK(cudaMemsetAsync(dout, 0, (size_t)T * VALUE_DIM * 2, s));
  int nchunks = 0;
  for (int off = 0; off < T; off += chunk, nchunks++) {
    int len = (T - off < chunk) ? (T - off) : chunk;
    int ret = atlas_gdn_prefill_packed_managed(
        (char*)dqkv + (size_t)off * CONV_DIM * 2,
        (char*)dgb + (size_t)off * GB * 4,
        (char*)dout + (size_t)off * VALUE_DIM * 2,
        dst, SCALE, len, NK, NV, KD, VD, CONV_DIM, GB, 1, s);
    if (ret != 0) { printf("managed ret=%d at off=%d\n", ret, off); exit(1); }
  }
  CK(cudaStreamSynchronize(s));
  CK(cudaGetLastError());
  return nchunks;
}

int main() {
  atlas_gdn_load();
  // ---- deterministic inputs; alpha near 1 so early-chunk state SURVIVES to the
  // last chunk (per-head decay rates log-spaced 1e-5..1e-2: long- and short-memory
  // heads both covered — a vanishing state would make the carry test vacuous).
  __nv_bfloat16* hqkv = (__nv_bfloat16*)malloc((size_t)T * CONV_DIM * 2);
  float* hgb = (float*)malloc((size_t)T * GB * 4);
  float rate[NV];
  for (int h = 0; h < NV; h++) rate[h] = 1e-5f * powf(1e3f, (float)h / (NV - 1));
  for (int t = 0; t < T; t++) {
    __nv_bfloat16* row = hqkv + (size_t)t * CONV_DIM;
    for (int i = 0; i < KEY_DIM; i++) row[i] = __float2bfloat16(2.0f * frand() - 1.0f);
    for (int h = 0; h < NK; h++) { // k: l2-normalized per (token, head), as Atlas feeds it
      float kraw[KD]; double ss = 0;
      for (int i = 0; i < KD; i++) { kraw[i] = 2.0f * frand() - 1.0f; ss += (double)kraw[i] * kraw[i]; }
      float inv = (float)(1.0 / sqrt(ss + 1e-12));
      for (int i = 0; i < KD; i++) row[KEY_DIM + h * KD + i] = __float2bfloat16(kraw[i] * inv);
    }
    for (int i = 0; i < VALUE_DIM; i++)
      row[2 * KEY_DIM + i] = __float2bfloat16(2.0f * frand() - 1.0f);
    float* gbrow = hgb + (size_t)t * GB;
    for (int h = 0; h < NV; h++) {
      gbrow[h] = 1.0f - rate[h] * frand();      // linear alpha (Atlas gate space)
      gbrow[NV + h] = 0.1f + 0.8f * frand();    // beta
    }
  }
  void *dqkv, *dgb, *dout, *dst;
  CK(cudaMalloc(&dqkv, (size_t)T * CONV_DIM * 2));
  CK(cudaMalloc(&dgb, (size_t)T * GB * 4));
  CK(cudaMalloc(&dout, (size_t)T * VALUE_DIM * 2));
  CK(cudaMalloc(&dst, STATE_N * 4));
  CK(cudaMemcpy(dqkv, hqkv, (size_t)T * CONV_DIM * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dgb, hgb, (size_t)T * GB * 4, cudaMemcpyHostToDevice));
  cudaStream_t s; CK(cudaStreamCreate(&s));

  const size_t on = (size_t)T * VALUE_DIM, tail0 = (size_t)(T - 1024) * VALUE_DIM;
  __nv_bfloat16* oref = (__nv_bfloat16*)malloc(on * 2);
  __nv_bfloat16* ocur = (__nv_bfloat16*)malloc(on * 2);
  float* sref = (float*)malloc(STATE_N * 4);
  float* scur = (float*)malloc(STATE_N * 4);

  // ---- arm A: single call (no carry) = reference; run twice for determinism.
  run_chunked(dqkv, dgb, dout, dst, T, s);
  CK(cudaMemcpy(oref, dout, on * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sref, dst, STATE_N * 4, cudaMemcpyDeviceToHost));
  run_chunked(dqkv, dgb, dout, dst, T, s);
  CK(cudaMemcpy(ocur, dout, on * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(scur, dst, STATE_N * 4, cudaMemcpyDeviceToHost));
  Cmp d0 = cmp_bf16(oref, ocur, on), d0s = cmp_f32(sref, scur, STATE_N);
  printf("determinism  : out max_abs=%.3e  state max_abs=%.3e  %s\n", d0.max_abs,
         d0s.max_abs, (d0.max_abs == 0 && d0s.max_abs == 0) ? "OK" : "NONDETERMINISTIC (noise floor!)");
  double sabs = 0; for (size_t i = 0; i < STATE_N; i++) sabs += fabs(sref[i]);
  printf("ref |state|mean=%.4f (must be >>0 or the test is vacuous)\n", sabs / STATE_N);

  // ---- arm B: aligned splits (bit-exact expected) + ragged split (rounding-level).
  struct { int chunk; const char* note; } arms[] = {
    {8192, "2 chunks, aligned"}, {4096, "4 chunks, aligned"},
    {2048, "8 chunks, aligned"}, {1000, "17 chunks, RAGGED (%64!=0)"},
  };
  int n_arms = 4, exact = 0, aligned_fail = 0; double prev = -1; int growing = 0;
  for (int a = 0; a < n_arms; a++) {
    int nch = run_chunked(dqkv, dgb, dout, dst, arms[a].chunk, s);
    CK(cudaMemcpy(ocur, dout, on * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(scur, dst, STATE_N * 4, cudaMemcpyDeviceToHost));
    Cmp o = cmp_bf16(oref, ocur, on);
    Cmp t = cmp_bf16(oref + tail0, ocur + tail0, on - tail0); // last 1024 tokens: most carry-sensitive
    Cmp st = cmp_f32(sref, scur, STATE_N);
    printf("chunk=%5d (%2d calls, %s): out max=%.3e cos=%.9f | tail max=%.3e | "
           "state max=%.3e cos=%.9f nan=%d\n", arms[a].chunk, nch, arms[a].note,
           o.max_abs, o.cos, t.max_abs, st.max_abs, st.cos, o.nan + st.nan);
    bool aligned = (arms[a].chunk % 64 == 0);
    if (aligned) {
      if (o.max_abs == 0 && st.max_abs == 0) exact++;
      else {
        if (o.cos < 0.9999 || st.cos < 0.9999 || o.nan + st.nan > 0) aligned_fail++;
        if (prev >= 0 && st.max_abs > 2.0 * prev) growing++;
        prev = st.max_abs;
      }
    } else if (o.cos < 0.999 || st.cos < 0.999 || o.nan + st.nan > 0) {
      printf("  ^^ RAGGED-TAIL DEFECT: masked final block corrupts the carried state\n");
      aligned_fail++;
    }
  }
  const char* verdict = (exact == 3) ? "CARRY_EXACT - D2 excluded (carry is bit-exact); the needle bug is NOT the carry"
      : (aligned_fail == 0 && growing == 0) ? "CARRY_OK - rounding-level only, not growing with chunk count; D2 unlikely"
      : "CARRY_DEFECT - divergence grows with chunk count / ragged tail: D2 CONFIRMED, fix the shim carry";
  printf("RESULT: %s\n", verdict);
  return (exact == 3 || (aligned_fail == 0 && growing == 0)) ? 0 : 2;
}
