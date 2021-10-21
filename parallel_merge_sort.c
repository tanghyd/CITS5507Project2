#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "mpi_utils.h"
#include "random_array.h"
#include "merge_sort.h"

#define N 10000

int main(int argc, char* argv[])
{
    //  set up MPi
    MPI_File file;
    MPI_Status status;

    int rank, world_size;
    int rc = MPI_Init(&argc, &argv); // return the status from MPI_Init
    if (rc != MPI_SUCCESS) {
        printf("Error in creating MPI program.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // set up array sizes
    int n = N; // total number of array elements
    validate_equal_chunks(world_size, n);  // all chunk sizes match
    int chunk_size = n / world_size;  // assumes no remainder
    double *chunk = malloc(chunk_size * sizeof(double));

    // sync processes and start timer
    MPI_Barrier(MPI_COMM_WORLD);
    double duration, max_duration;  // time elapsed for sorting algorithm
    duration = MPI_Wtime();

    // read in file chunks with MPI independent parallel (rather than read in one process and scatter)
    MPI_File_open(MPI_COMM_WORLD, "data/unsorted_array.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_read_at(file, rank*chunk_size*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&file);

    // apply merge sort over each array
    int depth = log2(world_size);
    double *data;  // final aggregated array will be saved on rank 0 only
    if (rank == 0)
    {
        data = malloc(n*sizeof(double));
        data = mpi_merge_sort(depth, rank, chunk, chunk_size, MPI_COMM_WORLD, data);
    } else {
        mpi_merge_sort(depth, rank, chunk, chunk_size, MPI_COMM_WORLD, NULL);
    }

    // send chunks of full data array to each process
    MPI_Scatter(data, chunk_size, MPI_DOUBLE, chunk, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_File_open(MPI_COMM_WORLD, "data/sorted_array.bin", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    MPI_File_write_at(file, rank*chunk_size*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&file);

    duration = MPI_Wtime() - duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        // ordered == 1 if sorted, else 0.
        int ordered = check_array_order(data, n);
        printf("ordered: %d\n", ordered);
		printf("Sorting array of size %d took %fs.\n", n, max_duration);
        free(data);
    }
    free(chunk);   
    MPI_Finalize();
    return 0;
}