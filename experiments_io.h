/* experiments_io.h */
#ifndef EXPERIMENTS_IO
#define EXPERIMENTS_IO

#include "mpi.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

double time_parallel_read(MPI_Comm comm, int world_size, int rank, char *file_name, int size, double *chunk, int chunk_size);
double time_parallel_write(MPI_Comm comm, int world_size, int rank, char *file_name, int size, double *chunk, int chunk_size);
double time_serial_read(char *file_name, double *array, int size);
double time_serial_write(char *file_name, double *array, int size);

int run_all_serial_io();
int run_all_parallel_io(MPI_Comm comm, int world_size, int rank);

#endif