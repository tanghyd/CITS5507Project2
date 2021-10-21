/* mpi_utils.h */
#ifndef MPI_UTILS
#define MPI_UTILS

#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// checking array size and process configurations
void validate_equal_chunks(int world_size, int n);
void validate_log2_procs(int world_size, int n);

#endif