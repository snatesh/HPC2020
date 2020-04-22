#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<iostream>
#include<iomanip>
#include <omp.h>

using std::setw;

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

// matrix vector product, Ax = b
void matvec(double* A, double* x, double* b, int N)
{
  #pragma omp parallel for
  for (int i = 0; i < N; ++i)
  {
    double sum = 0.0;
    for (int j = 0; j < N; ++j)
    {
      sum += A[i + j*N]*x[j];
    }
    b[i] = sum;
  }
}

// computes ||Au-b||_2
double residual(double* b, double* Au, int N)
{
  double resid = 0;
  #pragma omp parallel for reduction(+:resid)
  for (int i = 0; i < N; ++i)
  {
    resid += (Au[i]-b[i])*(Au[i]-b[i]);
  }
  return sqrt(resid);
}


// construct 2D centered second difference matrix w/ 0-Dirichlet BCs
void SecondDerivMat(int N, double* A)
{ 
  int Nsq = N * N;
  A[0] = 4; A[Nsq] = A[Nsq*N] = -1.0; // row 0 
  A[Nsq - 1 + Nsq*(Nsq-N-1)] = A[Nsq - 1 + Nsq*(Nsq-2)] = -1.0; // row N^2-1
  A[Nsq - 1 + Nsq*(Nsq-1)] = 4.0; // row N^2-1
  #pragma omp for
  for (int k = 1; k < N; ++k) // rows 1..N-1
  {
    A[k + Nsq*(k-1)] = -1.0;
    A[k + Nsq*k] = 4.0; 
    if ((k+1) % N) A[k + Nsq*(k+1)] = -1.0;
    A[k + Nsq*(k+N)] = -1.0; 
  
  } 
  #pragma omp for
  for (int k = Nsq-N; k < Nsq-1; ++k) // rows N^2-N .. N^2-1
  {
    A[k + Nsq*(k-N)] = -1.0;
    if ((k-1) % (Nsq-N-1)) A[k + Nsq*(k-1)] = -1.0;
    A[k + Nsq*k] = 4.0; 
    A[k + Nsq*(k+1)] = -1.0; 
  }
  #pragma omp for
  for (int k = N; k <= Nsq-N-1; ++k) // rows N..N^2-N-1
  {
    A[k + Nsq*(k-N)] = -1.0;
    if (k % N) A[k + Nsq*(k-1)] = -1.0;
    A[k + Nsq*k] = 4.0;
    if ((k+1) % N) A[k + Nsq*(k+1)] = -1.0; 
    A[k + Nsq*(k+N)] = -1.0;
  } 
}

/* Jacobi iteration
 - u : initial guess (and void return)
 - A : second deriv matrix
 - f : RHS vector of BVP
 - N : number of grid points
 - maxit : maximum iterations
 returns the final residual
*/
void Jacobi_omp(double* u, double* u1, const double* A, const double* f, const long N, const long maxit)
{
  for (int it = 0; it < maxit; ++it)
  { 
    #pragma omp parallel for 
    for (long i = 0; i < N; ++i)
    {
      double sum = 0; 
      for (long j = 0; j < N; ++j)
      {
        if (j != i)
          sum += A[i + j * N] * u[j];
      }
      u1[i] = (f[i] - sum) / 4;
    }
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) u[i] = u1[i];
  }
}

__global__ void Jacobi_kernel(double* u, double* u1, const double* A, const double* f, const long N)
{
  double sum = 0;
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  for (long j = 0; j < N; ++j)
  {
    if (j != i)
      sum += A[i + j * N] * u[j];
  }
  u1[i] = (f[i] - sum) / 4; 

  __syncthreads();
  u[i] = u1[i];
} 

int main(int argc, char* argv[])
{
  int Ns[8] = {30, 40, 50, 60, 70, 80, 90, 100};
  std::cout << setw(3) << "N" << setw(16) << "CPU (s)" << setw(16) << "GPU (s)" \
            << setw(16) << "Speedup" << setw(16) << "Res CPU" << setw(16) << "Res GPU" \
            << setw(16) << "Error\n";
  for (int iN = 0; iN < 8; ++iN)
  {
    // num dof, grid spacing
    long N = Ns[iN], Nsq = N * N, Nsqsq = Nsq * Nsq, maxit=1000;
    double h = 1.0/(N+1); h *= h;
    // Au = f
    double *A, *Au, *u, *u1, *f;
    CUDA_Error_Check(cudaMallocHost(&A, Nsqsq * sizeof(double))); 
    CUDA_Error_Check(cudaMallocHost(&u, Nsq * sizeof(double)));
    CUDA_Error_Check(cudaMallocHost(&Au, Nsq * sizeof(double)));
    CUDA_Error_Check(cudaMallocHost(&u1, Nsq * sizeof(double)));
    CUDA_Error_Check(cudaMallocHost(&f, Nsq * sizeof(double)));
    
    // initialize arrays
    #pragma omp parallel
    {
      #pragma omp for
      for (int k = 0; k < Nsqsq; ++k) A[k] = 0.0;
      #pragma omp for
      for (int k = 0; k < Nsq; ++k) u[k] = u1[k] = 0.0;
      #pragma omp for
      for (int k = 0; k < Nsq; ++k) f[k] = h * 1.0;
      // compute 2D second deriv matrix
      SecondDerivMat(N,A);
    }

    // run jacobi with openmp 
    double tt = omp_get_wtime(), jresOmp;
    Jacobi_omp(u, u1, A, f, Nsq, maxit);
    double timeOmp = omp_get_wtime()-tt;
    matvec(A,u,Au,Nsq); jresOmp = residual(Au,f,Nsq); 
    // save u as reference
    double* u_ref; 
    CUDA_Error_Check(cudaMallocHost(&u_ref, Nsq * sizeof(double)));  
    CUDA_Error_Check(cudaMemcpy(u_ref, u, Nsq * sizeof(double), cudaMemcpyHostToHost));
    // reinitialize sol for cuda run
    #pragma omp parallel for
    for (int k = 0; k < Nsq; ++k) u[k] = u1[k] = 0.0;
    
    // allocate and copy host inputs to device
    double *A_d, *u_d, *u1_d, *f_d;
    CUDA_Error_Check(cudaMalloc(&A_d, Nsqsq * sizeof(double))); 
    CUDA_Error_Check(cudaMalloc(&u_d, Nsq * sizeof(double)));
    CUDA_Error_Check(cudaMalloc(&u1_d, Nsq * sizeof(double)));
    CUDA_Error_Check(cudaMalloc(&f_d, Nsq * sizeof(double)));
    CUDA_Error_Check(cudaMemcpyAsync(A_d, A, Nsqsq * sizeof(double), cudaMemcpyHostToDevice)); 
    CUDA_Error_Check(cudaMemcpyAsync(u_d, u, Nsq * sizeof(double), cudaMemcpyHostToDevice)); 
    CUDA_Error_Check(cudaMemcpyAsync(u1_d, u1, Nsq * sizeof(double), cudaMemcpyHostToDevice)); 
    CUDA_Error_Check(cudaMemcpyAsync(f_d, f, Nsq * sizeof(double), cudaMemcpyHostToDevice)); 
    CUDA_Error_Check(cudaDeviceSynchronize());
    
    // run jacobi in cuda during every iteration to maxit
    tt = omp_get_wtime();
    for (int it = 0; it < maxit; ++it)
    {
      Jacobi_kernel<<< N, N >>>(u_d, u1_d, A_d, f_d, Nsq);
    }

    // copy device sol to host
    CUDA_Error_Check(cudaMemcpyAsync(u, u_d, Nsq * sizeof(double), cudaMemcpyDeviceToHost)); 
    CUDA_Error_Check(cudaDeviceSynchronize());
    
    double timeCuda = omp_get_wtime()-tt, jresCuda;
    matvec(A,u,Au,Nsq); jresCuda = residual(Au,f,Nsq); 
    double err = 0;
    #pragma omp parallel for reduction(+:err)
    for (long i = 0; i < Nsq; ++i) err += fabs(u_ref[i] - u[i]); 
    
    std::cout << setw(3) << N << setw(16) << timeOmp << setw(16) << timeCuda \
              << setw(16) << timeOmp/timeCuda << setw(16) << jresOmp << setw(16) << jresCuda \
              << setw(16) << err << "\n";

    CUDA_Error_Check(cudaFreeHost(A));
    CUDA_Error_Check(cudaFreeHost(u));
    CUDA_Error_Check(cudaFreeHost(Au));
    CUDA_Error_Check(cudaFreeHost(u1));
    CUDA_Error_Check(cudaFreeHost(f));
    CUDA_Error_Check(cudaFreeHost(u_ref));
    CUDA_Error_Check(cudaFree(A_d));
    CUDA_Error_Check(cudaFree(u_d));
    CUDA_Error_Check(cudaFree(u1_d));
    CUDA_Error_Check(cudaFree(f_d));
  }
  return 0;
}
