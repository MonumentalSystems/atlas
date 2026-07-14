// SPDX-License-Identifier: AGPL-3.0-only
//
// Self-contained fp32 CUDA kernels for the NLLB-200 / M2M-100 encoder forward
// pass (milestone-1 GPU PoC). Deliberately naive (correctness-first, tiny
// sequence lengths) and kept in `common/` so the module `"nllb_encoder"` is
// present in every model's PTX set. All math is fp32 to match the
// bit-faithful CPU reference (spark-nllb) so the encoder checksum validates
// tightly rather than "approximately".
//
// Weight layout is HuggingFace `nn.Linear`: weight is [out=N, in=K] row-major,
// consumed transposed (C[m,n] = bias[n] + Σ_k A[m,k]·W[n,k]).

#include <cuda_runtime.h>
#include <math.h>

// out[tok, :] = table[ids[tok], :]   (embedding gather, fp32)
extern "C" __global__ void nllb_embed(
    const unsigned int* __restrict__ ids,
    const float* __restrict__ table,
    float* __restrict__ out,
    unsigned int d) {
    unsigned int tok = blockIdx.x;
    unsigned long long id = ids[tok];
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) {
        out[(unsigned long long)tok * d + i] = table[id * d + i];
    }
}

// x[i] *= s
extern "C" __global__ void nllb_scale_inplace(float* __restrict__ x, unsigned int n, float s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// dst[i] += src[i]
extern "C" __global__ void nllb_add_inplace(
    float* __restrict__ dst, const float* __restrict__ src, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// x[i] = max(x[i], 0)
extern "C" __global__ void nllb_relu_inplace(float* __restrict__ x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(x[i], 0.0f);
}

// LayerNorm over the last dim, affine (weight/bias), in place.
// One block per row; blockDim.x MUST be a power of two.
extern "C" __global__ void nllb_layernorm(
    float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    unsigned int rows, unsigned int dim, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sm[];
    unsigned int tid = threadIdx.x;
    float* rowp = x + (unsigned long long)row * dim;

    float local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) local += rowp[i];
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float mean = sm[0] / dim;
    __syncthreads();

    local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float dv = rowp[i] - mean;
        local += dv * dv;
    }
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sm[0] / dim + eps);
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        rowp[i] = (rowp[i] - mean) * inv * w[i] + b[i];
    }
}

// C[M,N] = A[M,K] @ W[N,K]^T + bias[N]   (bias may be null)
extern "C" __global__ void nllb_linear(
    const float* __restrict__ a, const float* __restrict__ w, const float* __restrict__ bias,
    float* __restrict__ c, unsigned int M, unsigned int N, unsigned int K) {
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    const float* arow = a + (unsigned long long)m * K;
    const float* wrow = w + (unsigned long long)n * K;
    float acc = bias ? bias[n] : 0.0f;
    for (unsigned int k = 0; k < K; k++) acc += arow[k] * wrow[k];
    c[(unsigned long long)m * N + n] = acc;
}

// Dense non-causal multi-head SDPA. q/k/v/out are [seq, H*D] row-major
// (heads interleaved on the feature axis). One block per (query, head);
// blockDim.x == D (power of two). scale applied to the logits.
extern "C" __global__ void nllb_attention(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, unsigned int seq, unsigned int H, unsigned int D, float scale) {
    unsigned int qh = blockIdx.x;
    unsigned int query = qh / H;
    unsigned int head = qh % H;
    unsigned int tid = threadIdx.x; // 0..D-1
    extern __shared__ float sh[];   // scores[seq] then red[D]
    float* scores = sh;
    float* red = sh + seq;
    unsigned int dmodel = H * D;
    unsigned long long bq = (unsigned long long)query * dmodel + (unsigned long long)head * D;

    for (unsigned int j = 0; j < seq; j++) {
        unsigned long long bk = (unsigned long long)j * dmodel + (unsigned long long)head * D;
        red[tid] = q[bq + tid] * k[bk + tid];
        __syncthreads();
        for (unsigned int s = D / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[j] = red[0] * scale;
        __syncthreads();
    }
    if (tid == 0) {
        float m = -1e30f;
        for (unsigned int j = 0; j < seq; j++) m = fmaxf(m, scores[j]);
        float su = 0.0f;
        for (unsigned int j = 0; j < seq; j++) {
            scores[j] = expf(scores[j] - m);
            su += scores[j];
        }
        for (unsigned int j = 0; j < seq; j++) scores[j] /= su;
    }
    __syncthreads();
    float acc = 0.0f;
    for (unsigned int j = 0; j < seq; j++) {
        acc += scores[j] * v[(unsigned long long)j * dmodel + (unsigned long long)head * D + tid];
    }
    out[bq + tid] = acc;
}
