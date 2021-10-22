/* merge_sort.h */
#ifndef MERGE_SORT
#define MERGE_SORT

#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "mpi_utils.h"
#include "random_array.h"

/* the main function to implement mergesort */ 
void merge(double *arr, double *temp, int size);

// serial
void merge_sort(double *arr, double *temp, int size);

// parallel (with hybrid cutoff)
void merge_sort_tasks(double *arr, double *temp, int size, int cutoff);
void merge_sort_sections(double *arr, double *temp, int size, int cutoff);

// mpi
void merge_halves(double *half1, double *half2, double *result, int half1_size, int half2_size);
double* mpi_merge(int depth, int rank, double *chunk, int size, MPI_Comm comm, double *data);


#endif