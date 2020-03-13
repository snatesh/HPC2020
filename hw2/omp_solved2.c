/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
double total;

/*** Spawn parallel region ***/
#pragma omp parallel private(tid) reduction(+:total) 
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  #pragma omp for schedule(dynamic,10)
  for (i=0; i<1000000; i++) 
     total = total + i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
	printf("total = %e\n",total); // print value of shared <total>
}

/* What went wrong, and how did we fix it?
		* 1) The variable <total> and <tid> were shared by all threads. So, Line 34 was a
		*    race condition, and the printed values corresponded to the 
		*    last thread to store <tid> and accumulate into <total>.
    *   - FIX: Make <tid> a private variable and <total> a reduction+ variable.
		*   -      We don't need to initialize <total> to 0, as reduction with +
		*   -      creates a private copy of <total> initialized to 0 automatically.
		*   -      The original shared <total> variable is accumulated into with the
		*   -      private ones. Also, to account for non-associativity of floating point 
		*   -      addition, we increase the precision of <total> to double.
*/
