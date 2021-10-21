#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "mpi_utils.h"
#include "random_array.h"
#include "enumeration_sort.h"
#include "merge_sort.h"

#define N 10001

int main(int argc, char* argv[])
{
    // setup MPI 
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

    int n = N; // total number of array elements
    // validate_equal_chunks(world_size, n);
    int chunk_size = n / world_size;  // assumes no remainder
    int remainder = n % chunk_size;
    if (remainder > 0 && (rank+1) == world_size)
        chunk_size += remainder;  // add remainder to final rank
    

    printf("rank %d chunk_size %d\n", rank, chunk_size);
    double *chunk = malloc(chunk_size * sizeof(double));
    double *temp = malloc(chunk_size * sizeof(double));

    // sync processes and start timer
    MPI_Barrier(MPI_COMM_WORLD);
    double duration, max_duration;  // time elapsed for sorting algorithm
    duration = MPI_Wtime();

    // read in file chunks with MPI independent parallel (rather than read in one process and scatter)
    char in_file[64];
    char out_file[64];
    snprintf(in_file, sizeof(in_file), "data/unsorted_%d.bin", n);
    snprintf(out_file, sizeof(out_file), "data/sorted_%d.bin", n);
    
    MPI_File_open(MPI_COMM_WORLD, in_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_read_at(file, rank*(n / world_size)*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&file);
    
    // sort chunks on each process separately with enumeration_sort
    enumeration_sort(chunk, temp, chunk_size);

    // merge sort should be faster now that sub-arrays are sorted
    // we should not run enumeration_sort on large array due to O(n^2)
    double *data, *array;  // pointers for final sort (only allocated on rank 0)
    if (rank == 0)
    {
        // free(temp);  // temp was used for enumeration sort on chunk
        data = malloc(n * sizeof(double));  // full array to gather from chunks
        array = malloc(n * sizeof(double));  // temp for sort on full array
    }

    // communicate chunk size to master process
    int recvcounts[world_size];
    int displacements[world_size];
    
    MPI_Gather(&chunk_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        displacements[0] = 0;
        for (int i = 0; i < world_size-1; i++)
            displacements[i+1] = recvcounts[i];

        for (int i = 0; i < world_size; i++) {
            printf("displacement[%d]=%d | recvcounts[%d]=%d\n", i, displacements[i], i, recvcounts[i]);
        }
    }

    // gather variable length chunks to main data array on rank 0
    MPI_Gatherv(chunk, chunk_size, MPI_DOUBLE, data, recvcounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // final sort
    if (rank == 0)
    {
        // enumeration_sort(data, array, n);
        merge_sort(data, array, n);

        // returns 1 if array is correctly sorted, else 0.
        int ordered = check_array_order(data, n);
		printf("Sorting array of size %d took %fs.\n", n, max_duration);
        printf("Ordered: %d\n", ordered);
    }

    // scatter variable length chunks back to processes for MPI Parallel IO
    // Warning: likely high communication cost!
    MPI_Scatterv(data, recvcounts, displacements, MPI_DOUBLE, chunk, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // write results with MPI
    MPI_File_open(MPI_COMM_WORLD, out_file, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    MPI_File_write_at(file, rank*chunk_size*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&file);

    // different processes may take different times - we want the maximum time (bottleneck)
    duration = MPI_Wtime() - duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // deallocate memory
    if (rank == 0)
    {
        free(data);
        free(array);
    }
    free(temp);
    free(chunk);
    MPI_Finalize();
    return 0;
}