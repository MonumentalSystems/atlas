// Atlas <-> FlashInfer GDN bridge: wraps the static-inline C-ABI header into real
// extern "C" symbols Rust can link. Shape-generic (head_dim D=128 fixed for Holo).
#include "gdn_holo_0.h"
#include <cuda_runtime.h>

static gdn_holo_0_Kernel_Module_t g_module;
static int g_loaded = 0;

extern "C" void atlas_gdn_load() {
  if (!g_loaded) { gdn_holo_0_Kernel_Module_Load(&g_module); g_loaded = 1; }
}

// q,k,v,o: fp16 device ptrs; alpha,beta,state,init_state: fp32; tensormaps: scratch; cu_seqlens: int64.
extern "C" int atlas_gdn_prefill(
    void* q, void* k, void* v, void* o,
    void* alpha, void* beta, void* state, void* init_state,
    void* tensormaps, void* cu_seqlens,
    float scale, int total_seqlen,
    int num_q_heads, int num_k_heads, int num_v_heads, int num_sab_heads,
    int num_seqs, int grid_x, void* stream)
{
  const int D = 128;
  gdn_holo_0_Tensor_g_q_t   g_q ={q,    {total_seqlen, D, num_q_heads}, {(int64_t)num_q_heads*D, D}};
  gdn_holo_0_Tensor_g_k_t   g_k ={k,    {D, total_seqlen, num_k_heads}, {(int64_t)num_k_heads*D, D}};
  gdn_holo_0_Tensor_g_v_t   g_v ={v,    {D, total_seqlen, num_v_heads}, {(int64_t)num_v_heads*D, D}};
  gdn_holo_0_Tensor_g_o_t   g_o ={o,    {D, total_seqlen, num_v_heads}, {(int64_t)num_v_heads*D, D}};
  gdn_holo_0_Tensor_g_alpha_t g_al={alpha, {total_seqlen*num_sab_heads}};
  gdn_holo_0_Tensor_g_beta_t  g_be={beta,  {total_seqlen*num_sab_heads}};
  gdn_holo_0_Tensor_g_state_t g_st={state, {num_seqs*num_sab_heads*D*D}};
  gdn_holo_0_Tensor_g_init_state_t g_in={init_state, {num_seqs*num_sab_heads*D*D}};
  gdn_holo_0_Tensor_g_tensormaps_t g_tm={tensormaps, {6144}};
  gdn_holo_0_Tensor_cu_seqlens_t   g_cu={cu_seqlens, {num_seqs+1}};
  return cute_dsl_gdn_holo_0_wrapper(&g_module, &g_q,&g_k,&g_v,&g_o,&g_al,&g_be,&g_st,&g_in,&g_tm,&g_cu,
      scale, num_q_heads, num_k_heads, num_v_heads, num_sab_heads, num_seqs, 1, 0, grid_x, (cudaStream_t)stream);
}

// Atlas-NATIVE entry: takes Atlas's packed QKV ([Q(key_dim)|K(key_dim)|V(value_dim)] bf16,
// row stride = conv_dim) + interleaved gate_beta ([gate(nv)|beta(nv)] fp32, row stride gb_stride),
// + contiguous output [T,value_dim]. Deinterleaves gate/beta internally; q/k/v passed via
// conv_dim strides (no copy). This is what Atlas's prefill_gdn_full_inner will call directly.
static void* s_alpha=nullptr; static void* s_beta=nullptr; static size_t s_cap=0;
extern "C" int atlas_gdn_prefill_packed(
    void* qkv, void* gate_beta, void* output, void* h_state, void* init_state,
    void* tensormaps, void* cu_seqlens,
    float scale, int total_seqlen, int nk, int nv, int kd, int vd,
    int conv_dim, int gb_stride, int num_seqs, void* stream)
{
  cudaStream_t st=(cudaStream_t)stream;
  size_t need=(size_t)total_seqlen*nv*4;
  if(need>s_cap){ if(s_alpha)cudaFree(s_alpha); if(s_beta)cudaFree(s_beta);
    cudaMalloc(&s_alpha,need); cudaMalloc(&s_beta,need); s_cap=need; }
  // deinterleave gate_beta[T,gb_stride] fp32 -> contiguous alpha,beta[T,nv]
  cudaMemcpy2DAsync(s_alpha,(size_t)nv*4, gate_beta,(size_t)gb_stride*4,(size_t)nv*4,total_seqlen,cudaMemcpyDeviceToDevice,st);
  cudaMemcpy2DAsync(s_beta,(size_t)nv*4,(char*)gate_beta+(size_t)nv*4,(size_t)gb_stride*4,(size_t)nv*4,total_seqlen,cudaMemcpyDeviceToDevice,st);
  int key_dim=nk*kd; int num_sab=nv; int grid_x=num_seqs*num_sab;
  void* q=qkv; void* k=(char*)qkv+(size_t)key_dim*2; void* v=(char*)qkv+(size_t)key_dim*2*2; // bf16
  gdn_holo_0_Tensor_g_q_t   g_q ={q, {total_seqlen, kd, nk}, {(int64_t)conv_dim, kd}};
  gdn_holo_0_Tensor_g_k_t   g_k ={k, {kd, total_seqlen, nk}, {(int64_t)conv_dim, kd}};
  gdn_holo_0_Tensor_g_v_t   g_v ={v, {vd, total_seqlen, nv}, {(int64_t)conv_dim, vd}};
  gdn_holo_0_Tensor_g_o_t   g_o ={output, {vd, total_seqlen, nv}, {(int64_t)nv*vd, vd}};
  gdn_holo_0_Tensor_g_alpha_t g_al={s_alpha,{total_seqlen*nv}};
  gdn_holo_0_Tensor_g_beta_t  g_be={s_beta, {total_seqlen*nv}};
  gdn_holo_0_Tensor_g_state_t g_st={h_state,{num_seqs*nv*128*128}};
  gdn_holo_0_Tensor_g_init_state_t g_in={init_state,{num_seqs*nv*128*128}};
  gdn_holo_0_Tensor_g_tensormaps_t g_tm={tensormaps,{6144}};
  gdn_holo_0_Tensor_cu_seqlens_t   g_cu={cu_seqlens,{num_seqs+1}};
  return cute_dsl_gdn_holo_0_wrapper(&g_module,&g_q,&g_k,&g_v,&g_o,&g_al,&g_be,&g_st,&g_in,&g_tm,&g_cu,
      scale, nk, nk, nv, num_sab, num_seqs, 1, 0, grid_x, st);
}
