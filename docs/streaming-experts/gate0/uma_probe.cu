#include <cuda_runtime.h>
#include <cstdio>
__global__ void sum_kernel(const unsigned char* p, size_t n, unsigned long long* out){
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned long long s=0; for(size_t j=i;j<n;j+=gridDim.x*blockDim.x) s+=p[j];
  atomicAdd(out, s);
}
int main(){
  cudaSetDevice(0);
  int uva=0,maph=0,cma=0,pageable=0;
  cudaDeviceGetAttribute(&uva,cudaDevAttrUnifiedAddressing,0);
  cudaDeviceGetAttribute(&maph,cudaDevAttrCanMapHostMemory,0);
  cudaDeviceGetAttribute(&cma,cudaDevAttrConcurrentManagedAccess,0);
  cudaDeviceGetAttribute(&pageable,cudaDevAttrPageableMemoryAccess,0);
  printf("unified_addressing=%d can_map_host=%d concurrent_managed=%d pageable_mem_access=%d\n",uva,maph,cma,pageable);
  size_t n = 64ull<<20; // 64MB
  void* h=nullptr; cudaHostAlloc(&h,n,cudaHostAllocMapped);
  void* dp=nullptr; cudaError_t r=cudaHostGetDevicePointer(&dp,h,0);
  printf("host_ptr=%p dev_ptr=%p get_devptr_rc=%s same_addr=%d\n",h,dp,cudaGetErrorString(r),h==dp);
  // fill host, have GPU read the pinned buffer DIRECTLY (zero-copy) and verify + time
  for(size_t i=0;i<n;i++) ((unsigned char*)h)[i]=1;
  unsigned long long* out; cudaMalloc(&out,8); cudaMemset(out,0,8);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  sum_kernel<<<256,256>>>((unsigned char*)dp,n,out); cudaDeviceSynchronize(); // warm
  cudaMemset(out,0,8);
  cudaEventRecord(a); sum_kernel<<<256,256>>>((unsigned char*)dp,n,out); cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms=0; cudaEventElapsedTime(&ms,a,b);
  unsigned long long hsum=0; cudaMemcpy(&hsum,out,8,cudaMemcpyDeviceToHost);
  printf("zero-copy GPU read of pinned host buf: %.3f ms, %.1f GB/s, checksum=%llu (expect %llu) OK=%d\n",
         ms, (n/1e9)/(ms/1e3), hsum, (unsigned long long)n, hsum==(unsigned long long)n);
  return 0;
}
