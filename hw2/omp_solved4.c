/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1000 // decreased so it fits on thread stacks

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
double a[N][N];
/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a)
  {
  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i][j] = tid + i + j;

  /* For confirmation */
  printf("Thread %d done. Last element= %f\n",tid,a[N-1][N-1]);

  }  /* All threads join master thread and disband */

}

/* What went wrong, and how did we fix it?
		* 1) The memory required for the 2D array <a>, which is private to each thread,
		*	   exceeds the memory limit for the thread stack.  
    *   - FIX: Decrease the macro variable <N> (done here), or increase OMP_STACKSIZE. 
	  *		-			 The latter solution might also require one to mess with ulimit (on linux).
		
*/

