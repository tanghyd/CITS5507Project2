/* experiments_sort.h */
#ifndef EXPERIMENTS_SORT
#define EXPERIMENTS_SORT

#include "mpi.h"
#include <omp.h>

#include "mpi_utils.h"
#include "random_array.h"
#include "merge_sort.h"
#include "quick_sort.h"
#include "enumeration_sort.h"

// sorting experiment functions
double run_parallel_enumeration_sort(MPI_Comm comm, int world_size, int rank, char *file_name, int size, int save, char *construct);
double run_parallel_merge_sort(MPI_Comm comm, int world_size, int rank, char *file_name, int size, int save, char *construct);

#endif