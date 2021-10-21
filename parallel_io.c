#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "mpi_utils.h"
#include "random_array.h"

#define N 100000

int main(int argc, char *argv[])
{
    int rank, world_size;
    MPI_File fh;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // random seed init -- must be different on each rank
    srand(rank);

    // TO DO: Generate random arrays on multiple processes simultaneously
    // TO DO: Compare single process generation + writing vs. multi-process generation and saving
    // TO DO: Writing on one process and broadcasting before saving should be slower, but we can test
    
    // specify data sizes for arrays on each process
    int n = N;  // total size
    int chunk_size = n / world_size;  // this makes assumptions about even division
    validate_equal_chunks(world_size, n);  // all chunk sizes match

    // file name for unsorted random array
    // char file_name[] = "data/unsorted_array.bin";
    char file_name[64];
    snprintf(file_name, sizeof(file_name), "data/unsorted_%d.bin", n);

    // instantiate array to store unsorted values
    double *random_array = malloc(chunk_size * sizeof(double));

    // generate random doubles manually
    fill_double_array(random_array, chunk_size);

    // read pre-generated doubles
    // MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    // MPI_File_read_at(fh, rank*bufsize_byte, random_array, chunk_size, MPI_DOUBLE, &status);

    // write file with MPI
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at(fh, rank*chunk_size*sizeof(random_array), random_array, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&fh);

    // read file with MPI
    double *array = malloc(chunk_size * sizeof(double));
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, rank*chunk_size*sizeof(array), array, chunk_size, MPI_DOUBLE, &status);

    // validate arrays
    printf(
        "Rank: %d, random_array[%d]: %f, random_array[%d]: %f\n",
        rank, rank*chunk_size, random_array[0],
        (rank+1)*chunk_size-1, random_array[chunk_size-1]
    );

    printf(
        "Rank: %d, array[%d]: %f, array[%d]: %f\n",
        rank, rank*chunk_size, array[0],
        (rank+1)*chunk_size-1, array[chunk_size-1]
    );

    if (rank == 0)
    {
        // array equality function returns 1 (True) if equal
        int equal = check_array_equality(random_array, array, chunk_size);
        if (equal == 1)
            printf("Rank %d arrays are equal.\n", rank);
        else
            printf("Rank %d arrays are not equal.\n", rank);
    }

    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}