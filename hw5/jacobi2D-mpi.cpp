/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i,j;
  double tmp, gres = 0.0, lres = 0.0;

  for (i = 1; i <= lN; i++)
  {
    for (j = 1; j <= lN; ++j)
    {
      tmp = (4.0 * lu[j + i * lN] - lu[j + (i - 1) * lN] - lu[j - 1 + i * lN]
                                  - lu[j + (i + 1) * lN] - lu[j + 1 + i * lN]) * invhsq - 1;
      lres += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[])
{
  int mpirank, i, j, p, N, lN, iter, max_iters;
  MPI_Status status, status1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  //printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);  
  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  lN = N / p;
  //printf("Nl = %d\n", lN);
  if ((N % p != 0) && mpirank == 0 ) 
  {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();
  int sqrtP = (int) sqrt(p); 
  //printf("sqrtP = %d\n",sqrtP);
  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2) * (lN + 2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2) * (lN + 2));
  double * lutemp;

  // allocate send and recieve buffers
  double* sendl = (double*) calloc(sizeof(double), lN);
  double* sendr = (double*) calloc(sizeof(double), lN);
  double* sendu = (double*) calloc(sizeof(double), lN);
  double* sendd = (double*) calloc(sizeof(double), lN);
  double* recl = (double*) calloc(sizeof(double), lN);
  double* recr = (double*) calloc(sizeof(double), lN);
  double* recu = (double*) calloc(sizeof(double), lN);
  double* recd = (double*) calloc(sizeof(double), lN);

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) 
  {

    /* 2D Jacobi step for local points */
    for (i = 1; i <= lN; i++)
    {
      for (j = 1; j <= lN; ++j)
      {
        lunew[j + i * lN]  = 0.25 * (hsq + lu[j + (i - 1) * lN] + lu[j + (i + 1) * lN] 
                                                + lu[j -  1 + i * lN] + lu[j + 1 + i * lN]);
      }
    }

    // collect ghost values
    for (i = 0; i < lN; ++i)
    {
      // right col is sent right
      sendr[i] = lunew[lN - 1 + i * lN];
      // left col is sent left
      sendl[i] = lunew[i * lN];
      // top row is sent up
      sendu[i] = lunew[i];
      // bottom row is sent down
      sendd[i] = lunew[i + (lN - 1) * lN];
    }

    /* communicate ghost values */
    
    // first do below top row of procs
    if (mpirank < p - sqrtP) 
    {
      //double_exchange(sendu, recu, lN, sqrtP, 124, MPI_COMM_WORLD, mpirank, &status);
      MPI_Send(sendu, lN, MPI_DOUBLE, mpirank + sqrtP, 124, MPI_COMM_WORLD);
      MPI_Recv(recu, lN, MPI_DOUBLE, mpirank + sqrtP, 123, MPI_COMM_WORLD, &status);
    }
    // second do above bottom row of procs - make sure tags match
    if (mpirank >= sqrtP)
    {
      MPI_Send(sendd, lN, MPI_DOUBLE, mpirank - sqrtP, 123, MPI_COMM_WORLD);
      MPI_Recv(recd, lN, MPI_DOUBLE, mpirank - sqrtP, 124, MPI_COMM_WORLD, &status);
    }
    // third do right of left col of procs
    if (mpirank % sqrtP)
    {
      MPI_Send(sendl, lN, MPI_DOUBLE, mpirank - 1, 126, MPI_COMM_WORLD);
      MPI_Recv(recl, lN, MPI_DOUBLE, mpirank - 1, 125, MPI_COMM_WORLD, &status);
    }
    // fourth do left of right col of procs
    if ((mpirank + 1) % sqrtP) 
    {
      MPI_Send(sendr, lN, MPI_DOUBLE, mpirank + 1, 125, MPI_COMM_WORLD);
      MPI_Recv(recr, lN, MPI_DOUBLE, mpirank + 1, 126, MPI_COMM_WORLD, &status);
    }
    ///* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) 
    {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) 
      {   
        printf("Iter %d: Residual: %g\n", iter, gres);
      }   
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);
  free(sendu);
  free(sendd);
  free(sendr);
  free(sendl);
  free(recu);
  free(recd);
  free(recr);
  free(recl);
  
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) 
  {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
