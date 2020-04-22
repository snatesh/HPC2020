#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <iostream>

#define BLOCK_SIZE 1024

// error checking wrapper (taken from stackexchange)
#define CUDA_Error_Check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// threaded vec-dot
void vec_dot(double* c, const double* a, const double* b, long N)
{
  double sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i]*b[i];
  *c = sum;
}

__global__ void vec_dot_kernel(double* c, const double* a, const double* b, long N)
{
  __shared__ double smem[BLOCK_SIZE];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // each thread reads data from global into shared memory
  if (idx < N) smem[threadIdx.x] = a[idx] * b[idx];
  else smem[threadIdx.x] = 0;
  __syncthreads();

  unsigned int s;
  for (s=blockDim.x/2; s>32; s>>=1) 
  {
    if (threadIdx.x < s) 
    {
      smem[threadIdx.x] += smem[threadIdx.x + s];
    }
    __syncthreads();
  }
  // last 32 threads belong to same warp
  if (threadIdx.x < 32)
  {
    volatile double* s_ptr = smem;
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 32]; 
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 16]; 
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 8]; 
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 4]; 
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 2]; 
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 1]; 
  }
  

  // write to global memory
  if (threadIdx.x == 0) c[blockIdx.x] = smem[threadIdx.x];
}

__global__ void reduction(double* sum, const double* a, long N)
{
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  // each thread reads data from global into shared memory
  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>32; s>>=1) 
  {
    if (threadIdx.x < s) 
    {
    smem[threadIdx.x] += smem[threadIdx.x + s];
    }
    __syncthreads();
  }

  // last 32 threads belong to same warp
  if (threadIdx.x < 32)
  {
    volatile double* s_ptr = smem;
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 32]; 
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 16]; 
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 8]; 
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 4]; 
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 2]; 
    s_ptr[threadIdx.x] += s_ptr[threadIdx.x + 1]; 

  }
  // write to global memory
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}

int main() 
{
  const long N = (1UL<<25);
  
  // host copies and initialization
  double *a, *b, c;
  CUDA_Error_Check(cudaMallocHost(&a, N * sizeof(double)));
  CUDA_Error_Check(cudaMallocHost(&b, N * sizeof(double)));
  #pragma omp parallel for 
  for (long i = 0; i < N; ++i) a[i] = b[i] = drand48();
  
  // get reference val and time
  double c_ref; double tt = omp_get_wtime();
  vec_dot(&c_ref, a, b, N);
  printf("CPU Bandwidth = %f GB/s\n", N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  
  // device copies
  double *a_d, *b_d, *c_d; long N_work = 1, Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  CUDA_Error_Check(cudaMalloc(&a_d, N * sizeof(double)));
  CUDA_Error_Check(cudaMalloc(&b_d, N * sizeof(double)));
  // extra memory buffer for reduction across thread-blocks
  for (long i = Nb; i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  CUDA_Error_Check(cudaMalloc(&c_d, N_work*sizeof(double))); 
  // copy host inputs to device
  CUDA_Error_Check(cudaMemcpyAsync(a_d, a, N * sizeof(double), cudaMemcpyHostToDevice)); 
  CUDA_Error_Check(cudaMemcpyAsync(b_d, b, N * sizeof(double), cudaMemcpyHostToDevice)); 
  CUDA_Error_Check(cudaDeviceSynchronize());
  
  // call kernel recursively
  tt = omp_get_wtime();
  double* sum_d = c_d;
  vec_dot_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, a_d, b_d, N);
  while (Nb > 1) 
  {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction<<<Nb,BLOCK_SIZE>>>(sum_d + N, sum_d, N);
    sum_d += N;
  }
  // copy device result back to host
  CUDA_Error_Check(cudaMemcpyAsync(&c, sum_d, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_Error_Check(cudaDeviceSynchronize());
  printf("GPU Bandwidth = %f GB/s\n", N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("Error = %lf\n", fabs(c-c_ref));
  printf("%lf\t%lf\n",c,c_ref);
  // free mem
  CUDA_Error_Check(cudaFree(a_d)); 
  CUDA_Error_Check(cudaFree(b_d)); 
  CUDA_Error_Check(cudaFree(c_d));
  CUDA_Error_Check(cudaFreeHost(a)); 
  CUDA_Error_Check(cudaFreeHost(b));
  return 0; 
}
