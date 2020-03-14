/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

void dotprod (float& sum)
{
int i,tid;
//float sum;

tid = omp_get_thread_num();
#pragma omp for reduction(+:sum)
  for (i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
}


int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

#pragma omp parallel shared(sum)
  dotprod(sum);

printf("Sum = %f\n",sum);

}

/* What went wrong, and how did we fix it?
  * 1) The program did not compile because we tried using a reduction on the variable
  *    <sum>, which is private given its declaration in the function <dotprod>. Reduction
  *    variables must be shared (so the reduction can occur at the end of the parallel region)
  *   - FIX: I changed the prototype of <dotprod> to use reference semantics. Then,
  *          I can pass in the shared variable <sum> to the function <dotprod> at Line 40 
  *          as a reference, enabling the use of a reduction on it at Line 22. Note,
  *          I also removed the private declaration of <sum> at Line 19.
  *   - NOTE: This solution will not work with older C++ compilers.
*/
