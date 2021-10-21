/* experiments_sort.c */
#include "experiments_sort.h"

#define MAX_THREADS 1  // 2**4 = 16 threads
#define TRIALS 5

#define SAVE 0
#define VERBOSE 1
#define RUN_SERIAL 0
#define RUN_PARALLEL 1


/*-------------------------------------------------------------------
 * Function:    run_parallel_merge_sort
 *      
 *      Runs one parallel merge sort trial.
 *
 *      If the final array is not sorted a warning message will print.
 *
 *  Arguments:
 *      comm: MPI_COMM_WORLD object from MPI.
 *      world_size: total number of processes.
 *      rank: process id.
 *      *file_name: a file_name to save the binary file.
 *      size: The total size of the array.
 *      save: Whether to save to disk ( Save if != 0 ).
 *
 *  Returns:
 *      Maximum runtime duration over all processes.
 */

double run_parallel_merge_sort(MPI_Comm comm, int world_size, int rank, char *file_name, int size, int save, char *construct)
{
    if (!((strcmp("serial", construct)==0) || (strcmp("tasks", construct)==0) || (strcmp("sections", construct)==0)))
    {
        printf("Warning! Only 'serial', 'tasks', and sections' are valid for merge_sort. Received: %s. Aborting!\n", construct);
        return 0.;
    }

    // set up MPI variables
    MPI_File file;
    MPI_Status status;

    // time elapsed for sorting algorithm
    double duration, max_duration;
    
    // set up array sizes
    validate_equal_chunks(world_size, size);  // all chunk sizes match
    int chunk_size = size / world_size;  // assumes no remainder
    double *chunk = malloc(chunk_size * sizeof(double));
    double *temp = malloc(chunk_size * sizeof(double));  // workspace array for merge sort
    double *data;

    // sync processes and start timer
    MPI_Barrier(comm);
    duration = MPI_Wtime();

    // read in file chunks with MPI independent parallel (rather than read in one process and scatter)
    MPI_File_open(comm, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_read_at(file, rank*chunk_size*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&file);

    // separately merge chunks using a merge sort algorithm
    if (strcmp("serial", construct)==0)
        merge_sort(chunk, temp, chunk_size);
    else if (strcmp("tasks", construct)==0) 
        merge_sort_tasks(chunk, temp, chunk_size, 100);
    else if (strcmp("sections", construct)==0) 
        merge_sort_sections(chunk, temp, chunk_size, 100);

    // apply merge sort recursively down tree of processes
    int depth = log2(world_size);
    if (rank == 0)
    {
        data = malloc(size*sizeof(double));  // aggregate final sorted array on rank 0
        data = mpi_merge(depth, rank, chunk, chunk_size, MPI_COMM_WORLD, data);
    }
    else
    {
        mpi_merge(depth, rank, chunk, chunk_size, MPI_COMM_WORLD, NULL);
    }

    // if save == 0, then we write sorted arrays to disk
    // note: this will obviously effect runtimes
    if (save != 0)
    {
        // send chunks of full data array to each process
        MPI_Scatter(data, chunk_size, MPI_DOUBLE, chunk, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_File_open(comm, file_name, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        MPI_File_write_at(file, rank*chunk_size*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
        MPI_File_close(&file);
    }

    duration = MPI_Wtime() - duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_duration, 1, MPI_DOUBLE, 0, comm);
    if (rank == 0)
    {
        // ordered == 1 if sorted, else 0.
        int ordered = check_array_order(data, size);
        if (ordered == 0)
            printf("Warning! rank %d file %s is not correctly sorted!\n", rank, file_name);
        free(data);
    }
    free(chunk);
    free(temp);
    return max_duration;
}



/*-------------------------------------------------------------------
 * Function:    run_parallel_enumeration_sort
 *      
 *      Runs one parallel enumeration sort trial.
 *      
 *      Enumeration sort is run on each process separately and then
 *      sorted array chunks are merged using the mpi_merge function.
 *
 *      If the final array is not sorted a warning message will print.
 *
 *  Arguments:
 *      comm: MPI_COMM_WORLD object from MPI.
 *      world_size: total number of processes.
 *      rank: process id.
 *      *file_name: a file_name to save the binary file.
 *      size: The total size of the array.
 *      save: Whether to save to disk ( Save if != 0 ).
 *
 *  Returns:
 *      Maximum runtime duration over all processes.
 */

double run_parallel_enumeration_sort(MPI_Comm comm, int world_size, int rank, char *file_name, int size, int save, char *construct)
{
    if (!((strcmp("serial", construct)==0) || (strcmp("parallel", construct)==0)))
    {
        printf("Warning! Only 'serial' and parallel' are valid constructs for enumeration_sort. Received: %s. Aborting!\n", construct);
        return 0.;
    }

    // set up MPI variables
    MPI_File file;
    MPI_Status status;

    // time elapsed for sorting algorithm
    double duration, max_duration;
    
    // set up array sizes
    validate_equal_chunks(world_size, size);  // all chunk sizes match
    int chunk_size = size / world_size;  // assumes no remainder
    double *chunk = malloc(chunk_size * sizeof(double));
    double *temp = malloc(chunk_size * sizeof(double));  // workspace array for merge sort
    double *data;

    // sync processes and start timer
    MPI_Barrier(comm);
    duration = MPI_Wtime();

    // read in file chunks with MPI independent parallel (rather than read in one process and scatter)
    MPI_File_open(comm, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_read_at(file, rank*chunk_size*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&file);

    // separately merge chunks using a enumeration sort algorithm
    if (strcmp("serial", construct) == 0)
        enumeration_sort(chunk, temp, chunk_size);
    else if (strcmp("parallel", construct) == 0)
        enumeration_sort_parallel(chunk, temp, chunk_size);

    // apply merge sort recursively down tree of processes
    int depth = log2(world_size);
    if (rank == 0)
    {
        data = malloc(size*sizeof(double));  // aggregate final sorted array on rank 0
        data = mpi_merge(depth, rank, chunk, chunk_size, MPI_COMM_WORLD, data);
    }
    else
    {
        mpi_merge(depth, rank, chunk, chunk_size, MPI_COMM_WORLD, NULL);
    }

    // if save == 0, then we write sorted arrays to disk
    // note: this will obviously effect runtimes
    if (save != 0)
    {
        // send chunks of full data array to each process
        MPI_Scatter(data, chunk_size, MPI_DOUBLE, chunk, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_File_open(comm, file_name, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        MPI_File_write_at(file, rank*chunk_size*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
        MPI_File_close(&file);
    }

    duration = MPI_Wtime() - duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_duration, 1, MPI_DOUBLE, 0, comm);
    if (rank == 0)
    {
        // ordered == 1 if sorted, else 0.
        int ordered = check_array_order(data, size);
        if (ordered == 0)
            printf("Warning! rank %d file %s is not correctly sorted!\n", rank, file_name);
        free(data);
    }
    free(chunk);
    free(temp);
    return max_duration;
}


/*-------------------------------------------------------------------
 * Function:    run_parallel_quick_sort
 *      
 *      Runs one parallel quick sort trial with "Smart Partition approach".
 *      
 *      Each process swaps halves of their array until all values in rank i
 *      are less than all values rank i+1 for all ranks. After partitioning
 *      data across MPI processes we can sort independently.
 *
 *      To write file to disk, we can either write data from each process in
 *      chunks or use MPI_Gatherv to aggregate data onto one rank then save.
 *
 *      If the final array is not sorted a warning message will print.
 *
 *  Arguments:
 *      comm: MPI_COMM_WORLD object from MPI.
 *      world_size: total number of processes.
 *      rank: process id.
 *      *file_name: a file_name to save the binary file.
 *      size: The total size of the array.
 *      save: Whether to save to disk ( Save if != 0 ).
 *
 *  Returns:
 *      Maximum runtime duration over all processes.
 */

double run_parallel_quick_sort(MPI_COMM comm, int int world_size, int rank, char *file_name, int size, int save, char *construct)
{
    validate_log2_procs(world_size, size);  // num processes is power of 2
    // validate_equal_chunks(world_size, n);  // all chunk sizes match

    // calculate chunk size per process
    int chunk_size = size / world_size;
    int remainder = size % chunk_size;  // handle variable length chunk sizes
    if (remainder > 0 && (rank+1) == world_size)
        chunk_size += remainder;  // add remainder elements to final rank

    // instantiate data arrays
    double *chunk = malloc(chunk_size * sizeof(double));
    double* data;  // pointer to array data of length n  
    if (rank == 0)
        data = malloc(size * sizeof(double));

    // sync processes and start timer
    MPI_Barrier(MPI_COMM_WORLD);
    double duration, max_duration;  // time elapsed for sorting algorithm
    duration = MPI_Wtime();
    
    // read in file chunks with MPI independent parallel (rather than read in one process and scatter)
    MPI_File_open(comm, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_read_at(file, rank*(n/world_size)*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&file);

    // final aggregated array will be saved on rank 0
    if (rank == 0)
        data = mpi_quick_sort(comm, rank, world_size, chunk, chunk_size, data);
    else
        mpi_quick_sort(comm, rank, world_size, chunk, chunk_size, NULL);


    // separately merge chunks using a merge sort algorithm
    // if (strcmp("serial", construct)==0)
    //     merge_sort(chunk, temp, chunk_size);
    // else if (strcmp("tasks", construct)==0) 
    //     merge_sort_tasks(chunk, temp, chunk_size, 100);
    // else if (strcmp("sections", construct)==0) 
    //     merge_sort_sections(chunk, temp, chunk_size, 100);
    
    duration = MPI_Wtime() - duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0)
    {
        // ordered == 1 if sorted, else 0.
        int ordered = check_array_order(data, n);
        if (ordered == 0)
            printf("Warning! Full array from %d file %s is not correctly sorted!\n", rank, argv[1]);
    }

    MPI_Finalize();
    return max_duration;
}


int main(int argc, char* argv[])
{
    // MPI
    int rank, world_size, provided;
    // int rc = MPI_Init(&argc, &argv);  // return init status
    int rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (rc != MPI_SUCCESS)
    {
        printf("Error in creating MPI program.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // OpenMP    
    omp_set_dynamic(0);  // disable dynamic thread teams

    // set number of threads
    int thread_exp = MAX_THREADS;
    int num_threads = (int)pow(2, thread_exp);  // 2, 4, 8, 16
    omp_set_num_threads(num_threads);
    if (rank == 0 && VERBOSE)
    {
        printf("SETTINGS: SAVE=%d | VERBOSE=%d | RUN_SERIAL=%d | RUN_PARALLEL=%d\n", SAVE, VERBOSE, RUN_SERIAL, RUN_PARALLEL);  
        printf("OpenMP Threads: %d\n", omp_get_max_threads());
    }


    // experiment timer
    double runtime, max_runtime;

    // run experiments with single process on rank 0 only
    // if (rank == 0 && RUN_SERIAL == 1)
    // {
    //     runtime = MPI_Wtime();

    //     run_serial_merge_sort();

    //     runtime = MPI_Wtime() - runtime;
    //     // if (VERBOSE)
    //     printf("Serial IO experiments completed in %.7gs.\n\n", runtime);
    // }

    // all processes wait    
    MPI_Barrier(MPI_COMM_WORLD);

    // run parallel experiments on all ranks
    if (RUN_PARALLEL == 1)
    {
        // run specifications
        int size;
        int n_arrays = 5;
        int ARRAY_SIZES[5] = {10000, 100000, 1000000, 10000000, 100000000};

        double duration;  // sorting algorithm runtime
        runtime = MPI_Wtime();  // runtime for all experiments

        // output .csv to save experiment runtime results
        FILE *fptr;
        char results_file[] = "results/parallel_merge_sort_results.csv";

        char file_name[64];  // array to read in for sorting
        char construct[32]; // OpenMP construct to use for sorting algorithm
        
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
            // get file of specific array length
            size = ARRAY_SIZES[s];
            snprintf(file_name, sizeof(file_name), "data/serial/unsorted_%d.bin", size);

            // sort array with (single threaded) parallel merge sort
            // duration = run_parallel_merge_sort(MPI_COMM_WORLD, world_size, rank, file_name, size);

            // record performance results to file
            // if (rank == 0)
            // {
            //     // columns=[algorithm,mpi:world_size, openmp:construct, openmp:threads, size, duration]
            //     fprintf(fptr, "merge,%d,serial,%d,%d,%.17f\n", world_size, 1, size, duration);

            //     if (VERBOSE)
            // 		printf("Sorting array of size %d took %.7fs.\n", size, duration);
            // }

            // sort array with (openmp multi threaded) parallel enumeration sort (if size is below 100,000)
            if (size <= 100000)
            {
                snprintf(construct, sizeof(construct), "parallel");
                duration = run_parallel_enumeration_sort(MPI_COMM_WORLD, world_size, rank, file_name, size, SAVE, construct);

            if (rank == 0 && (strcmp("serial", construct) || strcmp("parallel", construct)))
                {
                    // columns=[algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                    fprintf(fptr, "enumeration,%d,%s,%d,%d,%.17f\n", world_size, construct, num_threads, size, duration);

                    if (VERBOSE)
                        printf("Enumeration sort (%s) for array of size %d took %.7fs.\n", construct, size, duration);
                }
            }

            // sort array with (openmp multi threaded) parallel merge sort
            snprintf(construct, sizeof(construct), "tasks");

            duration = run_parallel_merge_sort(MPI_COMM_WORLD, world_size, rank, file_name, size, SAVE, construct);

            if (rank == 0 && (strcmp("serial", construct) || strcmp("tasks", construct) || strcmp("sections", construct)))
            {
                // columns=[algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                fprintf(fptr, "merge,%d,%s,%d,%d,%.17f\n", world_size, construct, num_threads, size, duration);

                if (VERBOSE)
            		printf("Merge sort (%s) for array of size %d took %.7fs.\n", construct, size, duration);
            }

            // sort array with (openmp multi threaded) parallel quick sort
            snprintf(construct, sizeof(construct), "tasks");
            duration = run_parallel_quick_sort(MPI_COMM_WORLD, world_size, rank, file_name, size, SAVE, construct);

            if (rank == 0 && (strcmp("serial", construct) || strcmp("tasks", construct) || strcmp("sections", construct)))
            {
                // columns=[algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                fprintf(fptr, "quick,%d,%s,%d,%d,%.17f\n", world_size, construct, num_threads, size, duration);

                if (VERBOSE)
            		printf("Quick sort (%s) for array of size %d took %.7fs.\n", construct, size, duration);
            }

        }

        // end timer and close file
        runtime = MPI_Wtime() - runtime;
        MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            fclose(fptr);
            if (VERBOSE)
                printf("Parallel IO experiments with %d processes completed in %.7gs.\n\n", world_size, max_runtime);
        }
    }

    MPI_Finalize();
    return 0;
}