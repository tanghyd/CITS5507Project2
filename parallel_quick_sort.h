/* parallel_quick_sort.h */
#ifndef PARALLEL_QUICK_SORT
#define PARALLEL_QUICK_SORT

#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "mpi_utils.h"
#include "random_array.h"
#include "quick_sort.h"

/* the main function to implement enumeration_sort */ 
void enumeration_sort(double arr[], double sorted_arr[], int size);
void enumeration_sort_parallel(double arr[], double sorted_arr[], int size);

#endif