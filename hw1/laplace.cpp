#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "utils.h"

// naive matrix vector product, Ax = b
void matvec(double* A, double* x, double* b, int N);

// computes ||Au-b||_2
double residual(double* b, double* Au, int N);

// helper to visualize matrix A
void printMat(double* A, int N);

// construct centered second difference matrix w/ 0-Dirichlet BCs
void SecondDerivMat(int N, double* A);

/* Jacobi iteration
 - u : initial guess (and void return)
 - A : second deriv matrix
 - f : RHS vector of BVP
 - N : number of grid points
 - maxit : maximum iterations
 returns the last iteration number
*/
int Jacobi(double* u, double* A, double* f, int N, int maxit);


/* Gauss-Seidel iteration
 - u : initial guess (and void return)
 - A : second deriv matrix
 - f : RHS vector of BVP
 - N : number of grid points
 - maxit : maximum iterations
 returns the last iteration number
*/
int Gauss_Seidel(double* u, double* A, double* f, int N, int maxit);

// naive matrix vector product
void matvec(double* A, double* x, double* b, int N)
{
  for (int i = 0; i < N; ++i)
  {
    b[i] = 0.0;
    for (int j = 0; j < N; ++j)
    {
      b[i] += A[i + j*N]*x[j];
    }
  }
}

// computes ||Au-b||_2
double residual(double* b, double* Au, int N)
{
  double resid = 0;
  for (int i = 0; i < N; ++i)
  {
    resid += (Au[i]-b[i])*(Au[i]-b[i]);
  }
  return sqrt(resid);
}

void printMat(double* A, int N)
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      std::cout << A[i+j*N] << " ";
    }
    std::cout << std::endl;
  }
}

void SecondDerivMat(int N, double* A)
{
  // endpoint entries
  A[0]           =  2.0; 
  A[1]           = -1.0; 
  A[N-2+(N-1) * N] = -1.0;
  A[N-1+(N-1) * N] =  2.0;
  // populate tridiagonals 
  for (int i = 1; i < N-1; ++i)
  {
    A[i-1 + i * N] = -1.0;
    A[i   + i * N] =  2.0;
    A[i+1 + i * N] = -1.0;
  }
}

int Jacobi(double* u, double* A, double* f, int N, int maxit)
{
  double reltol = 1e-6; int it = 0;  
  double* Au = (double*) calloc(N,sizeof(double));
  double* u1 = (double*) malloc(N * sizeof(double));
  matvec(A,u,Au,N);
  double resid = residual(f,Au,N);
  #ifndef NOPRINT
  printf(" Iteration       residual\n");
  #endif
  while(residual(f,Au,N)/resid > reltol && it < maxit)
  { 
    #ifndef NOPRINT 
    printf("%10ld %10f\n",it,residual(f,Au,N));
    #endif
    for (int i = 0; i < N; ++i)
    {
      double sum = 0; 
      for (int j = 0; j < N; ++j)
      {
        if (j != i)
          sum += A[i + j * N] * u[j];
      }
      u1[i] = (f[i] - sum) / A[i + i * N];
    }
    for (int i = 0; i < N; ++i) u[i] = u1[i];
    matvec(A,u,Au,N);
    it += 1;
  }
  free(Au);
  free(u1);
  return it;
}

int Gauss_Seidel(double* u, double* A, double* f, int N, int maxit)
{
  double reltol = 1e-6; int it = 0;  
  double* Au = (double*) calloc(N,sizeof(double));
  matvec(A,u,Au,N);
  double resid = residual(f,Au,N);
  #ifndef NOPRINT
  printf(" Iteration       residual\n");
  #endif
  while(residual(f,Au,N)/resid > reltol && it < maxit)
  {
    #ifndef NOPRINT 
    printf("%10ld %10f\n",it,residual(f,Au,N));
    #endif
    for (int i = 0; i < N; ++i)
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
  free(Au);
  return it;
}


int main(int argc, char* argv[])
{
  // grid and max iters
  int N = 100;
  if (argc == 2)
    N = atoi(argv[1]);
  std::cout << N << std::endl; 
  double h = 1.0/(N+1); h *= h;
  double maxit = 100;
  // Au = f
  double* A = (double*) calloc(N * N, sizeof(double)); 
  double* u = (double*) calloc(N, sizeof(double));
  double* f = (double*) malloc(N * sizeof(double));
  SecondDerivMat(N,A);
  for (int i = 0; i < N; ++i) f[i] = h * 1.0;
  // nrepeat runs for timing 
  int nrepeat = 5; 
  // last iter number 
  int jIt, gsIt; 
  Timer T; double timeJ = 0.0; double timeGS = 0.0;

  for (int rep = 0; rep < nrepeat; ++rep)
  {
    T.tic();
    jIt = Jacobi(u, A, f, N, maxit);
    timeJ += T.toc();
    for (int i = 0; i < N; ++i) u[i] = 0.0;
    std::cout << rep << std::endl;
  }
  timeJ = timeJ/nrepeat;

  for (int rep = 0; rep < nrepeat; ++rep)
  {  
    T.tic();
    gsIt = Gauss_Seidel(u, A, f, N,maxit);
    timeGS += T.toc();
    for (int i = 0; i < N; ++i) u[i] = 0.0;
    std::cout << rep << std::endl;
  }
  timeGS = timeGS/nrepeat;
  std::cout << "Jacobi : " << jIt << " , Gauss-Seidel : " << gsIt << std::endl;
  std::cout << "Jacobi Time (s) : " << timeJ << std::endl;
  std::cout << "Gauss-Seidel Time (s) : " << timeGS<< std::endl;
  free(A);
  free(u);
  free(f);
  return 0;
}
