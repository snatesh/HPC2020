#ifndef _COMMON_H_
#define _COMMON_H_



// matrix vector product, Ax = b
void matvec(double* A, double* x, double* b, int N);

// computes ||Au-b||_2
double residual(double* b, double* Au, int N);

// helper to visualize matrix A
void printMat(double* A, int N);

// construct 2D centered second difference matrix w/ 0-Dirichlet BCs
void SecondDerivMat(int N, double* A);

#endif
