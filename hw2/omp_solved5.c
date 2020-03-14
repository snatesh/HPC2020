/******************************************************************************
* FILE: omp_bug5.c
* DESCRIPTION:
*   Using SECTIONS, two threads initialize their own array and then add
*   it to the other's array, however a deadlock occurs.
* AUTHOR: Blaise Barney  01/29/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1000000
#define PI 3.1415926535
#define DELTA .01415926535

int main (int argc, char *argv[]) 
{
int nthreads, tid, i;
float a[N], b[N];
omp_lock_t locka, lockb;

/* Initialize the locks */
omp_init_lock(&locka);
omp_init_lock(&lockb);

/* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel shared(a, b, nthreads, locka, lockb) private(tid)
  {

  /* Obtain thread number and number of threads */
  tid = omp_get_thread_num();
  #pragma omp master
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);
  #pragma omp barrier

  #pragma omp sections nowait
    {
    #pragma omp section
      {
      printf("Thread %d initializing a[]\n",tid);
      omp_set_lock(&locka); // set locka
      for (i=0; i<N; i++)
        a[i] = i * DELTA;
      omp_unset_lock(&locka); // free locka
      omp_set_lock(&lockb);  // set lockb
      printf("Thread %d adding a[] to b[]\n",tid);
      for (i=0; i<N; i++)
        b[i] += a[i];
      omp_unset_lock(&lockb); // free lockb

      }

    #pragma omp section
      {
      printf("Thread %d initializing b[]\n",tid);
      omp_set_lock(&lockb); // set lockb
      for (i=0; i<N; i++)
        b[i] = i * PI;
      omp_unset_lock(&lockb); // free lockb
      omp_set_lock(&locka); // set locka
      printf("Thread %d adding b[] to a[]\n",tid);
      for (i=0; i<N; i++)
        a[i] += b[i];
      omp_unset_lock(&locka); // free locka

      }
    }  /* end of sections */
  }  /* end of parallel region */

}

/* What went wrong, and how did we fix it?
    * 1) The use of locks here is wrong. A lock enforces that no other thread can
    *    get the locked resource until the thread that has the lock gets it AND
    *    releases the lock. No one can take a cookie from the cookie jar while
    *    someone else is reaching in for a cookie.
    *    - FIX: Make sure the thread in section1 releases locka after setting it, and
    *           do the same for lockb. Similarly, the thread in section2 must release
    *           lockb after setting it, and do the same for locka. 
*/
