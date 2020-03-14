#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utils.h"
#include "common.h"

/* Gauss-Seidel iteration
 - u : initial guess (and void return)
 - A : second deriv matrix
 - f : RHS vector of BVP
 - N : number of grid points
 - maxit : maximum iterations
 returns the final residual
*/
double Gauss_Seidel(double* u, double* A, double* f, int N, int maxit);


double Gauss_Seidel(double* u, double* A, double* f, int N, int maxit)
{
  double reltol = 1e-6; int it = 0;  
  double* Au = (double*) aligned_malloc(N * sizeof(double));
  matvec(A,u,Au,N);
  double resid = residual(f,Au,N);
  #ifndef NOPRINT
  printf("Starting Gauss_Seidel iteration ...\n");
  printf(" Iteration       residual\n");
  #endif
  while(residual(f,Au,N)/resid > reltol && it < maxit)
  {
    #ifndef NOPRINT 
    printf("%10d %10f\n",it,residual(f,Au,N));
    #endif
    #pragma omp parallel for 
    for (int i = 0; i < N; i+=2) // first update red pts
    {
      double sum = 0; 
      for (int j = 0; j < N; ++j)
      {
        if (j != i)
          sum += A[i + j * N] * u[j];
      }
      u[i] = (f[i] - sum) / A[i + i * N];
    }
    #pragma omp parallel for 
    for (int i = 1; i < N; i+=2) // now update black pts
    {
      double sum = 0; 
      for (int j = 0; j < N; ++j)
      {
        if (j != i)
          sum += A[i + j * N] * u[j];
      }
      u[i] = (f[i] - sum) / A[i + i * N];
    }
    matvec(A,u,Au,N);
    it += 1;
  }
  aligned_free(Au);
  return residual(f,Au,N);
}

int main(int argc, char* argv[])
{
  int numthreads[4] = {3,4,5,6}; 
  int Ns[7] = {5,10,15,20,25,30,35};
  int N,Nsq,Nsqsq,maxit=2000;
  printf("N    Threads\tTime\t\tResidual\n");
  printf("__   _______\t____\t\t________\n");

  for (int iN = 0; iN < 7; ++iN)
  {
    N = Ns[iN];
    Nsq = N * N; Nsqsq = Nsq * Nsq;
    double h = 1.0/(N+1); h *= h;
    // Au = f
    double* A = (double*) aligned_malloc(Nsqsq * sizeof(double)); 
    double* u = (double*) aligned_malloc(Nsq * sizeof(double));
    double* f = (double*) aligned_malloc(Nsq * sizeof(double));
    for (int nt = 0; nt < 4; ++nt)
    { 
      #ifdef _OPENMP
        omp_set_num_threads(numthreads[nt]);
      #else
        numthreads[nt] = 1;
      #endif

      #pragma omp parallel
      {
        #pragma omp for
        for (int k = 0; k < Nsqsq; ++k) A[k] = 0.0;
        #pragma omp for
        for (int k = 0; k < Nsq; ++k) u[k] = 0.0;
        #pragma omp for
        for (int k = 0; k < Nsq; ++k) f[k] = h * 1.0;
        // compute 2D second deriv matrix
        SecondDerivMat(N,A);
      }


      // nrepeat runs for timing 
      int nrepeat = 5; 
      // last residual 
      double gres;
      // timer init 
      Timer T; double timeGS = 0.0;
      for (int rep = 0; rep < nrepeat; ++rep)
      {
        T.tic();
        gres = Gauss_Seidel(u, A, f, Nsq, maxit);
        timeGS += T.toc();
        for (int i = 0; i < Nsq; ++i) u[i] = 0.0;
      }
      timeGS = timeGS/nrepeat;
      printf("%d\t%d\t%lf\t%e\n",N,numthreads[nt],timeGS,gres);
    }
    aligned_free(A);
    aligned_free(u);
    aligned_free(f);
  }
  return 0;
}
