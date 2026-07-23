// SPDX-License-Identifier: AGPL-3.0-only
//
// Native C++ harness: replays the reference IO dumped by b12x_export.py (P3) through the
// linked libatlasb12x.so and TOLERANCE-compares against the fp32 reference. Mirrors
// 3rdparty_patches/gdn_aot/gdn_harness.cpp. b12x scatter is atomic-add + fast-math SwiGLU
// => NOT bit-exact; accept cos >= 0.999 AND rel-L2 <= 2e-3.
//
// Build (gx10):
//   g++ -O2 b12x_harness.cpp -o b12x_harness -I. -I/usr/local/cuda/include \
//       ./libatlasb12x.so -lcudart -L<cute_lib> -lcute_dsl_runtime -Wl,-rpath,<cute_lib>
// Run:
//   LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:<cute_lib>:/usr/local/cuda/lib64 \
//     CUTE_DSL_ARCH=sm_121a ATLAS_B12X_MAX_TOKENS=1024 ./b12x_harness /tmp/b12x_aot
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" void atlas_b12x_load();
extern "C" int atlas_b12x_max_tokens();
extern "C" int atlas_b12x_moe_prefill(void *, void *, void *, void *, void *, void *,
                                      void *, void *, void *, void *, void *, int, void *);

static std::vector<uint8_t> slurp(const std::string &p) {
  FILE *f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> b(n); if (fread(b.data(), 1, n, f) != (size_t)n) exit(2);
  fclose(f); return b;
}
static void *up(const std::vector<uint8_t> &h) {
  void *d; cudaMalloc(&d, h.size()); cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice);
  return d;
}
static float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char **argv) {
  std::string dir = argc > 1 ? argv[1] : "/tmp/b12x_aot";
  atlas_b12x_load();
  printf("max_tokens=%d\n", atlas_b12x_max_tokens());
  // Reference IO dumped as raw little-endian tensors (names per b12x_export.py P3 dump).
  auto x = slurp(dir + "/ref_x_bf16.bin");
  auto ids = slurp(dir + "/ref_topk_ids_i32.bin");
  auto w = slurp(dir + "/ref_topk_w_f32.bin");
  auto w13 = slurp(dir + "/ref_w13_fp4.bin");
  auto w13sf = slurp(dir + "/ref_w13_sf.bin");
  auto w2 = slurp(dir + "/ref_w2_fp4.bin");
  auto w2sf = slurp(dir + "/ref_w2_sf.bin");
  auto a1 = slurp(dir + "/ref_w1_alpha.bin");
  auto a2 = slurp(dir + "/ref_w2_alpha.bin");
  auto fg = slurp(dir + "/ref_fc2_gs.bin");
  auto oref = slurp(dir + "/ref_out_f32.bin"); // fp32 reference output
  int T = (int)(x.size() / (2 * 3072));         // bf16 [T, H=3072] (Laguna-S-2.1)
  std::vector<uint8_t> obuf(x.size(), 0);

  void *dx = up(x), *did = up(ids), *dw = up(w), *doo = up(obuf);
  void *dw13 = up(w13), *dw13sf = up(w13sf), *dw2 = up(w2), *dw2sf = up(w2sf);
  void *da1 = up(a1), *da2 = up(a2), *dfg = up(fg);

  int ret = atlas_b12x_moe_prefill(dx, did, dw, doo, dw13, dw13sf, dw2, dw2sf, da1, da2, dfg, T, 0);
  cudaDeviceSynchronize();
  printf("atlas_b12x_moe_prefill ret=%d (T=%d)\n", ret, T);
  if (ret != 0) { printf("kernel returned nonzero — not yet frozen / capacity\n"); return ret; }

  cudaMemcpy(obuf.data(), doo, obuf.size(), cudaMemcpyDeviceToHost);
  const uint16_t *ob = (const uint16_t *)obuf.data();
  const float *rf = (const float *)oref.data();
  double dot = 0, na = 0, nb = 0, l2 = 0, rn = 0;
  size_t n = obuf.size() / 2;
  for (size_t i = 0; i < n; i++) {
    float a = bf16_to_f32(ob[i]), b = rf[i];
    dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
    l2 += (double)(a - b) * (a - b); rn += (double)b * b;
  }
  double cos = dot / (sqrt(na) * sqrt(nb) + 1e-12);
  double rel = sqrt(l2) / (sqrt(rn) + 1e-12);
  printf("cos=%.6f rel_l2=%.6f -> %s\n", cos, rel,
         (cos >= 0.999 && rel <= 2e-3) ? "PASS (tolerance)" : "FAIL");
  return (cos >= 0.999 && rel <= 2e-3) ? 0 : 1;
}
