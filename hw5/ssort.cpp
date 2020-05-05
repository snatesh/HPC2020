// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 1000000;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  double time = MPI_Wtime();
  // sort locally

  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* local_split = (int*) malloc((p - 1) * sizeof(int));
  for (unsigned int i = 0; i < p - 1; ++i)
  {
    local_split[i] = vec[i * N / (p - 1) + N / p];
  } 

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* cand_split = NULL;
  if (rank == 0)
  {
    cand_split = (int *) malloc(p * (p - 1) * sizeof(int));
  }
  MPI_Gather(local_split, p - 1, MPI_INT, cand_split, p - 1, MPI_INT, 0, comm);
  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int* root_picks = (int*) malloc((p - 1) * sizeof(int));
  if (rank == 0)
  {
    // sort
    std::sort(cand_split, cand_split + p * (p - 1));
    // pick p-1
    for (unsigned int i = 0; i < p - 1; ++i) root_picks[i] = cand_split[p * (i + 1) - 1];
  }
  // root process broadcasts splitters to all other processes
  MPI_Bcast(root_picks, p - 1, MPI_INT, 0, comm);
  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  int* sdispls = (int*) malloc(p * sizeof(int)); sdispls[0] = 0;
  int* rdispls = (int*) malloc(p * sizeof(int)); rdispls[0] = 0; 
  for (unsigned int i = 0; i < p-1; ++i)
  {
    sdispls[i+1] = std::lower_bound(vec, vec+N, root_picks[i]) - vec;
  }
  int* num_send = (int*) malloc(p * sizeof(int));
  int* num_rec = (int*) malloc(p * sizeof(int));
  num_send[p-1] = N - sdispls[p-1];
  for (unsigned int i = 0; i < p-1; ++i)
  {
    num_send[i] = sdispls[i+1] - sdispls[i];
  }
   
  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, 
  MPI_Alltoall(num_send, 1, MPI_INT, num_rec, 1, MPI_INT, comm);
  //and then use MPI_Alltoallv to exchange the data
  for (unsigned int i = 0; i < p-1; ++i) rdispls[i+1] = rdispls[i] + num_rec[i];  
  int sort_size = rdispls[p-1]+num_rec[p-1]; 
  int* proc_sort = (int*) malloc(sort_size * sizeof(int));
  MPI_Alltoallv(vec, num_send, sdispls, MPI_INT, proc_sort, num_rec, rdispls, MPI_INT, comm);
  // do a local sort of the received data
  std::sort(proc_sort, proc_sort + sort_size);
  MPI_Barrier(comm);
  time = MPI_Wtime()-time;
  if (rank == 0) printf("Elapsed time (s) : %lf\n", time);
  // every process writes its result to a file
  FILE* file; char fname[256]; snprintf(fname, 256, "output%02d.txt", rank);
  file = fopen(fname,"w+"); if(!file) {printf("Could not open file\n");exit(1);}  
  for (unsigned int i = 0; i < sort_size; ++i) {fprintf(file,"%d\n",proc_sort[i]);}
  fclose(file); 
  free(vec);
  free(local_split);
  free(cand_split);
  free(root_picks);
  free(sdispls);
  free(rdispls);
  free(proc_sort);
  free(num_rec);
  free(num_send);
  MPI_Finalize();
  return 0;
}
