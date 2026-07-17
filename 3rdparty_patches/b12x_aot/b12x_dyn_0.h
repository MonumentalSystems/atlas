
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} b12x_dyn_0_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_b12x_dyn_0_cuda_init(void **);
void _mlir_b12x_dyn_0_cuda_load_to_device(void **);
static inline void b12x_dyn_0_Kernel_Module_Load(b12x_dyn_0_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_b12x_dyn_0_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_b12x_dyn_0_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void b12x_dyn_0_Kernel_Module_Unload(b12x_dyn_0_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
} b12x_dyn_0_Tensor_barrier_count_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_barrier_epoch_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_pair_head_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_producers_done_count_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_all_work_published_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_task_head_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_task_tail_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_b_w13_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_b_down_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_row_counts_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_expert_write_rows_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_expert_tile_base_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_input_global_scale_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_alpha_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_down_alpha_t;


typedef struct {
    void *data;
} b12x_dyn_0_Tensor_global_scale_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_b12x_dyn_0__mlir_ciface_cutlass___call___flashinferfused_moecute_dslblackwell_sm12xmoe_dispatch_DynamicMoELaunch_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_FakeTensorInt32_1_1_Fake(void **args, int32_t num_args);

static inline int32_t cute_dsl_b12x_dyn_0_wrapper(b12x_dyn_0_Kernel_Module_t *module, void *a_ptr, void *topk_ids_ptr, void *topk_weights_ptr, void *packed_a_ptr, void *sfa_ptr, void *packed_a_storage_ptr, void *scale_storage_ptr, b12x_dyn_0_Tensor_barrier_count_t *barrier_count, b12x_dyn_0_Tensor_barrier_epoch_t *barrier_epoch, b12x_dyn_0_Tensor_pair_head_t *pair_head, b12x_dyn_0_Tensor_producers_done_count_t *producers_done_count, b12x_dyn_0_Tensor_all_work_published_t *all_work_published, b12x_dyn_0_Tensor_task_head_t *task_head, b12x_dyn_0_Tensor_task_tail_t *task_tail, void *task_ready_ptr, void *task_expert_ptr, void *task_m_tile_ptr, void *task_slice_begin_ptr, void *task_slice_count_ptr, void *task_valid_rows_ptr, void *tile_write_count_ptr, b12x_dyn_0_Tensor_b_w13_t *b_w13, void *sfb_w13_ptr, b12x_dyn_0_Tensor_b_down_t *b_down, void *sfb_down_ptr, b12x_dyn_0_Tensor_row_counts_t *row_counts, b12x_dyn_0_Tensor_expert_write_rows_t *expert_write_rows, b12x_dyn_0_Tensor_expert_tile_base_t *expert_tile_base, b12x_dyn_0_Tensor_input_global_scale_t *input_global_scale, b12x_dyn_0_Tensor_alpha_t *alpha, b12x_dyn_0_Tensor_down_alpha_t *down_alpha, b12x_dyn_0_Tensor_global_scale_t *global_scale, void *scatter_ptr, void *token_map_ptr, void *token_weights_ptr, int32_t num_tokens, int32_t max_rows, int32_t rows_padded, int32_t max_tasks, int32_t max_phys_tiles, cudaStream_t stream) {
    int32_t ret;
    void *args[42] = {
        &a_ptr, &topk_ids_ptr, &topk_weights_ptr, &packed_a_ptr, &sfa_ptr, &packed_a_storage_ptr, &scale_storage_ptr, barrier_count, barrier_epoch, pair_head, producers_done_count, all_work_published, task_head, task_tail, &task_ready_ptr, &task_expert_ptr, &task_m_tile_ptr, &task_slice_begin_ptr, &task_slice_count_ptr, &task_valid_rows_ptr, &tile_write_count_ptr, b_w13, &sfb_w13_ptr, b_down, &sfb_down_ptr, row_counts, expert_write_rows, expert_tile_base, input_global_scale, alpha, down_alpha, global_scale, &scatter_ptr, &token_map_ptr, &token_weights_ptr, &num_tokens, &max_rows, &rows_padded, &max_tasks, &max_phys_tiles, &stream,
        &ret
    };
    _mlir_b12x_dyn_0__mlir_ciface_cutlass___call___flashinferfused_moecute_dslblackwell_sm12xmoe_dispatch_DynamicMoELaunch_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_FakeTensorInt32_1_1_Fake(args, 42);
    return ret;
}

