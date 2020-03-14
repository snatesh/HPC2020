#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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

// helper to visualize matrix A
void printMat(double* A, int N)
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      printf("%d ", (int) A[i+j*N]);
    }
    printf("\n");
  }
}
