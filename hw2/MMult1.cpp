// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utils.h"

#define BLOCK_SIZE 64

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  #pragma omp parallel for 
  for (long j = 0; j < n; j+=BLOCK_SIZE) {
    for (long p = 0; p < k; p+=BLOCK_SIZE) {
      for (long i = 0; i < m; i+=BLOCK_SIZE) {
        for (long jb = j; jb < j+BLOCK_SIZE; ++jb) {
          for (long pb = p; pb < p+BLOCK_SIZE; ++pb) {
            for (long ib = i; ib < i+BLOCK_SIZE; ++ib) {
              double A_ip = a[ib+pb*m];
              double B_pj = b[pb+jb*k];
              double C_ij = c[ib+jb*m];
              C_ij = C_ij + A_ip * B_pj;
              c[ib+jb*m] = C_ij;
            }
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  const long PFIRST = 10*BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = 1.0e-9*(2.0*m*n*k*NREPEATS/time); 
    double bandwidth = 1.0e-9*2.0*(n * n * n / BLOCK_SIZE + n * n)*NREPEATS/time; 
    printf("%10ld %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }

  return 0;
}


