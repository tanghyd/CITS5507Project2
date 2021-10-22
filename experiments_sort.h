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

/* Function Declarations */

double run_merge_merge_sort(MPI_Comm comm, int world_size, int rank, int size, char *construct, char *in_file, char *out_file);
// double run_merge_quick_sort(MPI_Comm comm, int world_size, int rank, int size, char *construct, char *in_file, char *out_file);
double run_merge_enumeration_sort(MPI_Comm comm, int world_size, int rank, int size, char *construct, char *in_file, char *out_file);
// double run_partition_merge_sort(MPI_Comm comm, int world_size, int rank, int size, char *construct, char *in_file, char *out_file);
double run_partition_quick_sort(MPI_Comm comm, int world_size, int rank, int size, char *construct, char *in_file, char *out_file);
// double run_partition_enumeration_sort(MPI_Comm comm, int world_size, int rank, int size, char *construct, char *in_file, char *out_file);
#endif