#include "random_array.h"
#include "experiments_io.h"

// #define TRIALS 5

#define VERBOSE 1
#define RUN_SERIAL 1
#define RUN_PARALLEL 1


/* Function Definitions */


/*
 * Function:  run_all_parallel_io
 * --------------------
 *  Main driver code for all parallel read/write experiments.
 *
 *  Arguments:
 *      comm: MPI_COMM_WORLD object from MPI.
 *      world_size: total number of processes.
 *      rank: process id.
 *
 *  Returns: 1 if success
 */
int run_all_parallel_io(MPI_Comm comm, int world_size, int rank)
{
    // random seed init -- call only once
    srand(rank);

    // MPI duration timings require MPI_Reduce
    double duration, runtime;
    double *reference_chunk, *chunk;
    int size, chunk_size, remainder, equal;

    // run specifications
    int n_arrays = 7;
    int ARRAY_SIZES[7] = {1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
    char parallel_file[64];

    // output .csv to save experiment runtime results
    FILE *fptr;
    char results_file[] = "results/parallel_io_results.csv";
    
    // only need to write performance results on rank 0
    if (rank == 0)
    {
        fptr = fopen(results_file, "a");
        if (!fptr)
        {  
            printf("Failed to open file\n");
            MPI_Finalize;
            return 0;
        }
    }

    // loop over multiple array sizes
    for (int s = 0; s < n_arrays; s++)
    {
        size = ARRAY_SIZES[s];

        snprintf(parallel_file, sizeof(parallel_file), "data/parallel/unsorted_%d.bin", size);

        // specify data sizes for arrays on each process
        chunk_size = size / world_size;
        // validate_equal_chunks(world_size, size);  // all chunk sizes must match
        remainder = size % chunk_size;  // handle variable length chunk sizes
        if (remainder > 0 && (rank+1) == world_size)
            chunk_size += remainder;  // add remainder to final rank

        // instantiate array to store unsorted values
        reference_chunk = malloc(chunk_size * sizeof(double));
        chunk = malloc(chunk_size * sizeof(double));

        // generate random doubles between 0 and 1
        fill_double_array(reference_chunk, chunk_size);

        // write array to disk and also save max runtime over all processes
        duration = time_parallel_write(comm, world_size, rank, parallel_file, size, reference_chunk, chunk_size);
        if (rank == 0)
            fprintf(fptr, "write,parallel,%d,%d,%.17f\n", world_size, size, duration);

        // read array to disk and also save max runtime over all processes
        duration = time_parallel_read(comm, world_size, rank, parallel_file, size, chunk, chunk_size);
        if (rank == 0)
            fprintf(fptr, "read,parallel,%d,%d,%.17f\n", world_size, size, duration);

        // validate arrays - 1 ("True") if arrays are equal, else 0.
        equal = check_array_equality(reference_chunk, chunk, chunk_size);
        if (equal != 1)
            printf("Warning! Rank %d %s does not match reference array.\n", rank, parallel_file);

        free(reference_chunk);
        reference_chunk = NULL;
        free(chunk);
        chunk = NULL;
    }

    if (rank == 0)
        fclose(fptr);

    return 0;
}

/*
 * Function:  run__all_serial_io
 * --------------------
 *  Main driver code for all serial read/write experiments.
 *
 *  Returns: 1 if success
 */
int run_all_serial_io(void)
{
    // random seed init -- call only once
    // srand(time(NULL));
    srand(1);

    // output .csv to save experiment runtime results
    FILE *fptr;
    char results_file[] = "results/serial_io_results.csv";

    // full run specifications
    double duration, runtime;
    int size, equal;
    int n_arrays = 7;
    int ARRAY_SIZES[7] = {1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

    char serial_file[64];

    fptr = fopen(results_file, "a");
    if (fptr)
    {  
        for (int s = 0; s < n_arrays; s++)
        {
            // total array size
            size = ARRAY_SIZES[s];
            
            // specify output file
            snprintf(serial_file, sizeof(serial_file), "data/serial/unsorted_%d.bin", size);

            // generate random doubles between 0 and 1
            double *reference_array = malloc(size*sizeof(double));
            double *array = malloc(size*sizeof(double));
            fill_double_array(reference_array, size);

            // write array to disk and also save runtime
            duration = time_serial_write(serial_file, reference_array, size);
            fprintf(fptr, "write,serial,%d,%d,%.17f\n", 1, size, duration);

            // read array to disk and also save runtime
            duration = time_serial_read(serial_file, array, size);
            fprintf(fptr, "read,serial,%d,%d,%.17f\n", 1, size, duration);

            // validate arrays - 1 ("True") if arrays are equal, else 0.
            equal = check_array_equality(reference_array, array, size);
            if (equal != 1)
                printf("Warning! %s does not match reference array.\n", serial_file);

            free(reference_array);
            reference_array = NULL;
            free(array);
            array = NULL;
        }
    } else {
        printf("Failed to open file\n");
    }
    fclose(fptr);  

    return 0;
}

/*
 * Function:  time_parallel_read 
 * --------------------
 *  Simultaenously reads a file from disk using MPI_File_read_at with
 *  chunked arrays, while also timing the maximum duration across processes.
 *  The file type is assumed to be binary.
 *
 *  Note: We pass world_size and size as the array chunk on the final rank may
 *  be different to others if the total array size is not divisible by world size.
 *  
 *  For this reason we offset file writing at rank*(size/world_size)*sizeof(double).
 *
 *  Arguments:
 *      comm: MPI_COMM_WORLD object from MPI.
 *      world_size: total number of processes.
 *      rank: process id.
 *      *file_name: a file_name to save the binary file.
 *      size: The total size of the array.
 *      *chunk: double array to read data from.
 *      chunk_size: the count of array elements to save in a given chunk.
 *
 *  Returns:
 *      Maximum runtime across all processes in MPI_COMM_WORLD.
 */
double time_parallel_read(MPI_Comm comm, int world_size, int rank, char *file_name, int size, double *chunk, int chunk_size)
{   
    MPI_File fh;
    MPI_Status status;
    double duration, max_duration;
    duration = MPI_Wtime();
    
    // read file with MPI
    MPI_File_open(comm, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, rank*(size/world_size)*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&fh);

    // get max run-time across all processes
    duration = MPI_Wtime() - duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Bcast(&max_duration, 1, MPI_DOUBLE, 0, comm);
    if (rank == 0 && VERBOSE)
        printf("Max read duration for array size %d = %.7fs\n", size, max_duration);
    return duration;
}

/*
 * Function:  time_parallel_write
 * --------------------
 *  Simultaenously writes a file to disk using MPI_File_write_at with
 *  chunked arrays, while also timing the maximum duration across processes.
 *  The file type is assumed to be binary.
 *
 *  Note: We pass world_size and size as the array chunk on the final rank may
 *  be different to others if the total array size is not divisible by world size.
 *  
 *  For this reason we offset file writing at rank*(size/world_size)*sizeof(double).
 *
 *  Arguments:
 *      comm: MPI_COMM_WORLD object from MPI.
 *      world_size: total number of processes.
 *      rank: process id.
 *      *file_name: a file_name to save the binary file.
 *      size: The total size of the array.
 *      *chunk: double array to write data from.
 *      chunk_size: the count of array elements to save in a given chunk.
 *
 *  Returns:
 *      Maximum runtime across all processes in MPI_COMM_WORLD.
 */
double time_parallel_write(MPI_Comm comm, int world_size, int rank, char *file_name, int size, double *chunk, int chunk_size)
{
    MPI_File fh;
    MPI_Status status;
    double duration, max_duration;
    duration = MPI_Wtime();

    // write file with MPI
    MPI_File_open(comm, file_name, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at(fh, rank*(int)(size/world_size)*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&fh);

    // get max run-time across all processes
    duration = MPI_Wtime() - duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Bcast(&max_duration, 1, MPI_DOUBLE, 0, comm);
    if (rank == 0 && VERBOSE)
        printf("Max write duration for array size %d = %.7fs\n", size, max_duration);
    return max_duration;
}


/*
 * Function:  time_serial_read
 * --------------------
 *  Reads a full array of doubles from disk using with a single process.
 *  The file type is assumed to be binary.
 *
 *  Note: This function has only been written for one process.

 *  Arguments:
 *      *file_name: a file_name to save the binary file.
 *      *array: double array to read data from.
 *      size: The total size of the array.
 *
 *  Returns:
 *      Runtime duration.
 */
double time_serial_read(char *file_name, double *array, int size)
{   
    double duration;
    duration = MPI_Wtime();

    read_array(file_name, array, size);

    duration = MPI_Wtime() - duration;
    if (VERBOSE)
        printf("Read duration for array size %d = %.7fs\n", size, duration);
    return duration;
}

/*
 * Function:  time_serial_write
 * --------------------
 *  Writes a full array of doubles to disk using with a single process.
 *  The file type is assumed to be binary.
 *
 *  Note: This should only be run with one process otherwise duplicates will be saved.

 *  Arguments:
 *      *file_name: a file_name to save the binary file.
 *      *array: double array to write data to.
 *      size: The total size of the array.
 *
 *  Returns:
 *      Runtime duration.
 */
double time_serial_write(char *file_name, double *array, int size)
{
    double duration;
    duration = MPI_Wtime();

    write_array(file_name, array, size);

    duration = MPI_Wtime() - duration;
    if (VERBOSE)
       printf("Write duration for array size %d = %.7fs\n", size, duration);
    return duration;
}


int main(int argc, char* argv[])
{

    int rank, world_size;
    int rc = MPI_Init(&argc, &argv);  // return init status
    if (rc != MPI_SUCCESS)
    {
        printf("Error in creating MPI program.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double runtime, max_runtime;

    // run serial experiment on rank 0 only
    if (rank == 0 && RUN_SERIAL == 1)
    {
        runtime = MPI_Wtime();

        run_all_serial_io();

        runtime = MPI_Wtime() - runtime;
        // if (VERBOSE)
        printf("Serial IO experiments completed in %.7gs.\n\n", runtime);
    }

    // all processes wait    
    MPI_Barrier(MPI_COMM_WORLD);

    // run parallel experiments on all ranks
    if (RUN_PARALLEL == 1)
    {
        runtime = MPI_Wtime();

        run_all_parallel_io(MPI_COMM_WORLD, world_size, rank);

        runtime = MPI_Wtime() - runtime;
        MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            // if (VERBOSE)
            printf("Parallel IO experiments with %d processes completed in %.7gs.\n\n", world_size, max_runtime);
        }
    }

    MPI_Finalize();
    return 0;
}