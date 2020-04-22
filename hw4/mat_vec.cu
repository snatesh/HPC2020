#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <math.h>

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

// threaded mat_vec
void mat_vec(double* A, double* x, double* b, long N)
{
#pragma omp parallel for
  for (long i = 0; i < N; ++i)
  {
    double sum = 0.0;
    for (long j = 0; j < N; ++j)
    {
      sum += A[i * N + j] * x[j];
    }
    b[i] = sum;
  }
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


__global__ void mat_vec_kernel(double* c, double* c_d, const double* A, const double* b, long N, long N_work)
{
    long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    double* sum_d = &c_d[blockIdx.x * N_work];
    vec_dot_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, &A[blockIdx.x * N], b, N);
    while (Nb > 1) 
    {
      long N = Nb;
      Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
      reduction<<<Nb,BLOCK_SIZE>>>(sum_d + N, sum_d, N);
      sum_d += N;
    }
    // copy nested device result back to this one
    cudaMemcpyAsync(&c[blockIdx.x], sum_d, sizeof(double), cudaMemcpyDeviceToDevice);
}

int main() 
{
  //const long N = BLOCK_SIZE;
  const long N = (1UL<<11);
  
  // host copies and initialization
  double *A, *b, *c;
  CUDA_Error_Check(cudaMallocHost(&A, N * N * sizeof(double)));
  CUDA_Error_Check(cudaMallocHost(&b, N * sizeof(double)));
  CUDA_Error_Check(cudaMallocHost(&c, N * sizeof(double)));
  #pragma omp parallel
  {
    #pragma omp for 
    for (long i = 0; i < N * N; ++i) A[i] = drand48();
    #pragma omp for
    for (long i = 0; i < N; ++i) b[i] = drand48();
  }

  // get reference val and time
  double *c_ref, tt = omp_get_wtime();
  CUDA_Error_Check(cudaMallocHost(&c_ref, N * sizeof(double)));
  mat_vec(A, b, c_ref, N);
  printf("CPU Bandwidth = %f GB/s\n", N * N * sizeof(double) / (omp_get_wtime()-tt)/1e9);
  
  // device copies
  double *A_d, *b_d, *c_d; long N_work = 1;
  CUDA_Error_Check(cudaMalloc(&A_d, N * N * sizeof(double)));
  CUDA_Error_Check(cudaMalloc(&b_d, N * sizeof(double)));
  CUDA_Error_Check(cudaMalloc(&c_d, N * sizeof(double)));
  
  // extra memory buffer for reduction across thread-blocks
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; \
       i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  double* c_dd; cudaMalloc(&c_dd, N*N_work*sizeof(double)); 

  // copy host inputs to device
  CUDA_Error_Check(cudaMemcpyAsync(A_d, A, N * N * sizeof(double), cudaMemcpyHostToDevice)); 
  CUDA_Error_Check(cudaMemcpyAsync(b_d, b, N * sizeof(double), cudaMemcpyHostToDevice)); 
  CUDA_Error_Check(cudaDeviceSynchronize());
  
  // call kernel recursively
  tt = omp_get_wtime();

  mat_vec_kernel<<<N,1>>>(c_d, c_dd, A_d, b_d, N, N_work);
  //double* sum_d = c_dd;
  //long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  //mat_vec_kernel1<<<N*Nb,BLOCK_SIZE>>>(sum_d, A_d, b_d, N);
  //for (long i = 0; i < N; ++i)
  //{
  //  Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  //  double* sum_dd = &c_dd[i];
  //  while (Nb > 1) 
  //  {
  //    long N1 = Nb;
  //    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
  //    reduction<<<Nb,BLOCK_SIZE>>>(sum_dd + N1, sum_dd, N1);
  //    sum_dd += N1;
  //  }
  //}
  //// copy device result back to host
  //CUDA_Error_Check(cudaMemcpyAsync(c, sum_d, N * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_Error_Check(cudaMemcpyAsync(c, c_d, N * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_Error_Check(cudaDeviceSynchronize());

  printf("GPU Bandwidth = %f GB/s\n", N * N * sizeof(double) / (omp_get_wtime()-tt)/1e9);
  double err = 0;
  #pragma omp parallel for reduction(+:err)
  for (long i = 0; i < N; ++i) err += fabs(c[i]-c_ref[i]);
  printf("Error = %f\n", err/N);
  // free mem
  CUDA_Error_Check(cudaFree(A_d)); 
  CUDA_Error_Check(cudaFree(b_d)); 
  CUDA_Error_Check(cudaFree(c_d));
  CUDA_Error_Check(cudaFreeHost(A)); 
  CUDA_Error_Check(cudaFreeHost(b));
  CUDA_Error_Check(cudaFreeHost(c));
  CUDA_Error_Check(cudaFreeHost(c_ref));
  return 0; 
}



