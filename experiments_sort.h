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

// project 2

double run_mpi_merge(MPI_Comm comm, int world_size, int rank, int size, char *algorithm, char *construct, char *in_file, char *out_file);
double run_mpi_partition(MPI_Comm comm, int world_size, int rank, int size, char *algorithm, char *construct, char *in_file, char *out_file);

// project 1

// quick sort
double time_quick_sort_serial(double *arr, int size);
double time_quick_sort_tasks(double *arr, int size, int cutoff);
double time_quick_sort_sections(double *arr, int size, int cutoff);

// merge sort
double time_merge_sort_serial(double *arr, double *temp, int size);
double time_merge_sort_tasks(double *arr, double *temp, int size, int cutoff);
double time_merge_sort_sections(double *arr, double *temp, int size, int cutoff);

// enumeration sort
double time_enumeration_sort_serial(double *arr, double *temp, int size);
double time_enumeration_sort_parallel(double *arr, double *temp, int size);

#endif 