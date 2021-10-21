/* parallel_quick_sort.c */
// #include "parallel_quick_sort.h"

#include <omp.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "mpi_utils.h"
#include "random_array.h"
#include "quick_sort.h"


#define N 1000000
// #define N 1000000
#define MAX_THREADS 0  // 2**n; i.e. 2**4 = 16 threads

#define VERBOSE 1

int main(int argc, char* argv[])
{
    // check command line arguments
    if (argc != 2)
    {
        printf("One file path must be provided in command line input.\n");
        exit(1);
    }

    // set up MPI
    MPI_File file;
    MPI_Status status;
    int rc, provided;
    // rc = MPI_Init(&argc, &argv);
    rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided); // for openmp
    if (rc != MPI_SUCCESS) {
        printf("Error in creating MPI program.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // OpenMP: set number of threads
    omp_set_dynamic(0);  // disable dynamic thread teams
    int thread_exp = MAX_THREADS;
    int num_threads = (int)pow(2, thread_exp);  // 2, 4, 8, 16
    omp_set_num_threads(num_threads);
    if (rank == 0 && VERBOSE)
        printf("OpenMP Threads: %d\n", omp_get_max_threads());


    int n = N; // total number of array elements
    validate_log2_procs(world_size, n);  // num processes is power of 2

    // validate_equal_chunks(world_size, n);  // all chunk sizes match
    int chunk_size = n / world_size;  // assumes no remainder
    int remainder = n % chunk_size;  // handle variable length chunk sizes
    if (remainder > 0 && (rank+1) == world_size)
        chunk_size += remainder;  // add remainder to final rank

    // instantiate data rrays
    double *chunk = malloc(chunk_size * sizeof(double));
    double* data;  // pointer to array data of length n  
    if (rank == 0)
        data = malloc(n * sizeof(double));

    // sync processes and start timer
    MPI_Barrier(MPI_COMM_WORLD);
    double duration, max_duration;  // time elapsed for sorting algorithm
    duration = MPI_Wtime();
    
    // read in file chunks with MPI independent parallel (rather than read in one process and scatter)
    MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_read_at(file, rank*(n/world_size)*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&file);

    // for (int r=0; r<world_size; r++)
    // {
    //     if(rank==r){
    //         printf("Starting chunk rank %d\n", rank);
    //         for(int i=0; i<chunk_size; i++){
    //             printf("%lf ", chunk[i]);
    //         }
    //         printf("\n");
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    // fflush(stdout);

    // double unsorted_chunk_total = 0;
    // double unsorted_data_total = 0;
    // for (int i = 0; i < chunk_size; i++)
    //     unsorted_chunk_total += chunk[i];
    // MPI_Reduce(&unsorted_chunk_total, &unsorted_data_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // printf("unsorted_chunk_total=%lf\n", unsorted_chunk_total);
    
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);

    // final aggregated array will be saved on rank 0
    if (rank == 0)
        data = mpi_quick_sort(MPI_COMM_WORLD, rank, world_size, chunk, chunk_size, data);
    else
        mpi_quick_sort(MPI_COMM_WORLD, rank, world_size, chunk, chunk_size, NULL);

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // double sorted_data_total = 0;
    // if (rank == 0)
    // {
    //     for (int i = 0; i < n; i++)
    //     {
    //         sorted_data_total += data[i];
    //     }

    //     printf(
    //         "sorted_data_total=%f | unsorted_data_total=%f \n",
    //         sorted_data_total, unsorted_data_total
    //     );
    // }
    // fflush(stdout);


    duration = MPI_Wtime() - duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        // ordered == 1 if sorted, else 0.
        int ordered = check_array_order(data, n);
        if (ordered == 0)
            printf("Warning! Full array from %d file %s is not correctly sorted!\n", rank, argv[1]);
		printf("Quick sort array of size %d took %fs.\n", n, max_duration);
    }

    MPI_Finalize();
    return 0;
}