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
