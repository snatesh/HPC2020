#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include<iostream>
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) 
{
  #ifndef _OPENMP
    scan_seq(prefix_sum,A,n);
  #else
  if (n == 0) return;
  prefix_sum[0] = 0;
  int numthreads;
  long p;
  #pragma omp parallel
  {
    numthreads = omp_get_num_threads();
    p = (long) n/numthreads;
    int tid = omp_get_thread_num();
    long beg = (long) p*tid + 1;
    long end = p*(tid+1)+1;
    end = n - end + 1 >= p ? end : n;
    for (long i = beg; i < end; ++i)
    {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }
  }
  for (int tid = 1; tid < numthreads; ++tid)
  {
    long beg = (long) p*tid + 1;
    long end = p*(tid+1)+1;
    end = n - end + 1 >= p ? end : n;
    for (long i = beg; i < end; ++i)
    {
      prefix_sum[i] += prefix_sum[beg-1];
    }
  }
  #endif 
}


int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
