/* experiments_sort.c */
#include "experiments_sort.h"

/* Function Definitions */



/*-------------------------------------------------------------------
 * Function:    validate_algorithm_and_construct
 *      
 *      Checks whether an algorithm is one of "quick", "merge", or "enumeration".
 *      If quick or merge, we ensure the construct is either "single", "tasks", or "sections".
 *      If enumeration, we ensure that the construct is either "single" or "parallel".
 *
 *  Arguments:
 *      *algorithm: a string label for a sorting algorithm (quick, merge or eumeration).
 *      *construct: The OpenMP construct to use (or serial - no parallelisation).
 *
 *  Returns:
 *      1 if algorithm+construction pairing is valid, else 0.
 */

int validate_algorithm_and_construct(char* algorithm, char *construct)
{
    // check if valid sorting algorithm is provided
    if (!((strcmp("quick", algorithm)==0) || (strcmp("merge", algorithm)==0) || (strcmp("enumeration", algorithm)==0)))
    {
        printf("Warning! Only 'quick', 'merge', and enumeration' are valid sorting algorithms. Received: %s. Aborting!\n", algorithm);
        return 0;
    }

    if (strcmp("enumeration", algorithm)==0)
    {
        if (!((strcmp("single", construct)==0) || (strcmp("parallel", construct)==0)))
        {
            printf("Warning! Only 'single' and parallel' are valid constructs for %s. Received: %s. Aborting!\n", algorithm, construct);
            return 0.;
        }
    }
    else
    {
        // check if valid OpenMP construct is provided
        if (!((strcmp("single", construct)==0) || (strcmp("tasks", construct)==0) || (strcmp("sections", construct)==0)))
        {
            printf("Warning! Only 'single', 'tasks', and sections' are valid for %s. Received: %s. Aborting!\n", algorithm, construct);
            return 0;
        }
    }
    return 1;
}



/*-------------------------------------------------------------------
 * Function:    run_mpi_merge
 *      
 *      Runs one MPI merge sort trial. Array chunks are sorted first
 *      with one of quick sort, merge sort, or enumeration sort, before
 *      data is then agrgegated across processes with a merge sort.
 *
 *      If the final array is not sorted a warning message will print.
 *
 *  Arguments:
 *      comm: MPI_COMM_WORLD object from MPI.
 *      world_size: total number of processes.
 *      rank: process id.
 *      size: The total size of the array.
 *      *construct: The OpenMP construct to use (or serial - no parallelisation).
 *      *in_file: a file_name to read the binary file.
 *      *out_file: a file_name to write the binary file.
 *          If NULL, no file will be saved.
 *
 *  Returns:
 *      Maximum runtime duration over all processes.
 */
double run_mpi_merge(MPI_Comm comm, int world_size, int rank, int size, char *algorithm, char *construct, char *in_file, char *out_file)
{
    // checks valid pairings of sort and openmp construct
    // if 1 we have a valid string, else 0 will abort.
    if (validate_algorithm_and_construct(algorithm, construct)==0)
        return 0.;  // returns run-time of 0s

    // set up MPI variables
    MPI_File file;
    MPI_Status status;

    // time elapsed for sorting algorithm
    double duration, max_duration;

    // calculate chunk size per process0
    int chunk_size = chunk_size = size / world_size;
    int remainder = size % chunk_size;  // handle variable length chunk sizes
    if (remainder > 0 && (rank+1) == world_size)
        chunk_size += remainder;  // add remainder elements to final rank

    double *chunk = malloc(chunk_size * sizeof(double));
    double *data;
    double *temp;  // for merge or enumeration sort

    // sync processes and start timer
    MPI_Barrier(comm);
    duration = MPI_Wtime();

    // read in file chunks with MPI independent parallel (rather than read in one process and scatter)
    MPI_File_open(comm, in_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_read_at(file, rank*(size/world_size)*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&file);

    // separately merge chunks using a sort algorithm
    if (strcmp("quick", algorithm)==0)
    {
        // sort using a quick sort algorithm
        if (strcmp("single", construct)==0)
            quick_sort(chunk, 0, chunk_size-1);
        else if (strcmp("tasks", construct)==0) 
            quick_sort_tasks(chunk, 0, chunk_size-1, 100);
        else if (strcmp("sections", construct)==0) 
            quick_sort_sections(chunk, 0, chunk_size-1, 100);
    }
    else if (strcmp("merge", algorithm)==0)
    {
        // sort using a merge sort algorithm
        temp = malloc(chunk_size * sizeof(double));  // workspace array
        if (strcmp("single", construct)==0)
            merge_sort(chunk, temp, chunk_size);
        else if (strcmp("tasks", construct)==0) 
            merge_sort_tasks(chunk, temp, chunk_size, 100);
        else if (strcmp("sections", construct)==0) 
            merge_sort_sections(chunk, temp, chunk_size, 100);
    }
    else if (strcmp("enumeration", algorithm)==0)
    {   
        // sort using enumeration sort
        temp = malloc(chunk_size * sizeof(double));  // workspace array
        if (strcmp("single", construct) == 0)
            enumeration_sort(chunk, temp, chunk_size);
        else if (strcmp("parallel", construct) == 0)
            enumeration_sort_parallel(chunk, temp, chunk_size);
    }
    else
    {
        printf("WARNING! INVALID ALGORITHM IN run_mpi_merge!!\n");
    }

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

    // if out_file is not NULL we write to disk (assumes valid file)
    if (out_file)
    {
        write_array(out_file, data, size);
        // send chunks of full data array to each process
        // MPI_Scatter(data, chunk_size, MPI_DOUBLE, chunk, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_File_open(comm, out_file, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        // MPI_File_write_at(file, rank*chunk_size*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
        // MPI_File_close(&file);
    }

    duration = MPI_Wtime() - duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_duration, 1, MPI_DOUBLE, 0, comm);
    if (rank == 0)
    {
        // ordered == 1 if sorted, else 0.
        int ordered = check_array_order(data, size);
        if (ordered == 0)
            printf("Warning! rank %d file %s is not correctly sorted!\n", rank, in_file);
        free(data);
    }
    free(chunk);

    if (strcmp("quick", algorithm)!=0)
    {
        free(temp);  // quick sort does not use extra memory
    }

    return max_duration;
}


/*-------------------------------------------------------------------
 * Function:    run_mpi_partition
 *      
 *      Runs one parallel sort trial with "Smart Partition approach".
 *      
 *      Each process swaps halves of their array until all values in rank i
 *      are less than all values rank i+1 for all ranks. After partitioning
 *      data across MPI processes we can sort independently using one of
 *      merge sort, quick sort, or enumeration sort with an OpenMP construct.
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
 *      size: The total size of the array.
 *      *construct: The OpenMP construct to use (or serial - no parallelisation).
 *      *in_file: a file_name to read the binary file.
 *      *out_file: a file_name to write the binary file.
 *          If NULL, no file will be saved.
 *
 *  Returns:
 *      Maximum runtime duration over all processes.
 */
double run_mpi_partition(MPI_Comm comm, int world_size, int rank, int size, char *algorithm, char *construct, char *in_file, char *out_file)
{
   
    // checks valid pairings of sort and openmp construct
    // if 1 we have a valid string, else 0 will abort.
    if (validate_algorithm_and_construct(algorithm, construct)==0)
        return 0.;  // returns run-time of 0s
    
    validate_log2_procs(world_size, size);  // num processes is power of 2
    // validate_equal_chunks(world_size, n);  // all chunk sizes match

    // set up MPI variables
    MPI_File file;
    MPI_Status status;

    // calculate chunk size per process0
    int chunk_size = chunk_size = size / world_size;
    int remainder = size % chunk_size;  // handle variable length chunk sizes
    if (remainder > 0 && (rank+1) == world_size)
        chunk_size += remainder;  // add remainder elements to final rank
    double *chunk = malloc(chunk_size * sizeof(double));
    double *temp;  // for merge or enumeration sort

    // variables for error checking
    double* data;  // pointer to array data of length n  
    int recvcounts[world_size];
    int displacements[world_size];
    int ordered;

    // sync processes and start timer
    MPI_Barrier(MPI_COMM_WORLD);
    double duration, max_duration;  // time elapsed for sorting algorithm
    duration = MPI_Wtime();
    
    // read in file chunks with MPI independent parallel (rather than read in one process and scatter)
    MPI_File_open(comm, in_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_read_at(file, rank*(size/world_size)*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
    MPI_File_close(&file);

    // mpi partition will re-organise chunks between processes such that they are ordered
    chunk = mpi_partition(comm, world_size, rank, chunk, &chunk_size);  // chunk_size edited in function

    // separately merge chunks using a sort algorithm
    if (strcmp("quick", algorithm)==0)
    {
        // sort using a quick sort algorithm
        if (strcmp("single", construct)==0)
            quick_sort(chunk, 0, chunk_size-1);
        else if (strcmp("tasks", construct)==0) 
            quick_sort_tasks(chunk, 0, chunk_size-1, 100);
        else if (strcmp("sections", construct)==0) 
            quick_sort_sections(chunk, 0, chunk_size-1, 100);
    }
    else if (strcmp("merge", algorithm)==0)
    {
        // sort using a merge sort algorithm
        temp = malloc(chunk_size * sizeof(double));  // workspace array
        if (strcmp("single", construct)==0)
            merge_sort(chunk, temp, chunk_size);
        else if (strcmp("tasks", construct)==0) 
            merge_sort_tasks(chunk, temp, chunk_size, 100);
        else if (strcmp("sections", construct)==0) 
            merge_sort_sections(chunk, temp, chunk_size, 100);
    }
    else if (strcmp("enumeration", algorithm)==0)
    {   
        // sort using enumeration sort
        temp = malloc(chunk_size * sizeof(double));  // workspace array
        if (strcmp("single", construct) == 0)
            enumeration_sort(chunk, temp, chunk_size);
        else if (strcmp("parallel", construct) == 0)
            enumeration_sort_parallel(chunk, temp, chunk_size);
    }
    else
    {
        printf("WARNING! INVALID ALGORITHM IN run_mpi_merge!!\n");
    }

    // compute aggregated array size and displacement buffers for file writing and/or gatherv
    MPI_Allgather(&chunk_size, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    displacements[0] = 0;
    for (int i = 0; i < world_size-1; i++)
    {
        displacements[i+1] = displacements[i] + recvcounts[i];
    }

    // if out_file is not None we save
    if (out_file)
    {
        // write chunks (of variable length) to file from each process
        MPI_File_open(comm, out_file, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        MPI_File_write_at(file, displacements[rank]*sizeof(chunk), chunk, chunk_size, MPI_DOUBLE, &status);
        MPI_File_close(&file);
    }
    
    duration = MPI_Wtime() - duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    // validation:
    // check ordering on each individual process
    ordered = check_array_order(chunk, chunk_size);
    if (ordered == 0)  // ordered == 1 if sorted, else 0.
        printf("Warning! Array from %d file %s is not correctly sorted!\n", rank, in_file);

    // confirm that the aggregated data across all chunks is in fact equal
    if (rank == 0)
        data = malloc(size * sizeof(double));

    // gather variable length arrays to data - data must be allocated memory equal to correct total size
    MPI_Gatherv(chunk, chunk_size, MPI_DOUBLE, data, recvcounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        // ordered == 1 if sorted, else 0.
        ordered = check_array_order(data, size);
        if (ordered == 0)
            printf("Warning! Full array from %d file %s is not correctly sorted!\n", rank, out_file);
        free(data);
    }

    if (strcmp("quick", algorithm)!=0) {
        free(temp);  // quick sort does not use extra memory
    }

    free(chunk);
    

    return max_duration;
}

double time_quick_sort_serial(double *arr, int size)
{
    double start = omp_get_wtime();
    quick_sort(arr, 0, size-1);  // size-1 is final idx pos
    double end = omp_get_wtime();

    printf("Serial quick sort in %12.10fs. \n", end - start);
    if (check_array_order(arr, size) == 0)
        printf("### Above array is not sorted. ### \n");

    return end - start;
}
 
double time_quick_sort_tasks(double *arr, int size, int cutoff)
{
    double start = omp_get_wtime();
    quick_sort_tasks(arr, 0, size-1, cutoff);
    double end = omp_get_wtime();

    printf("Tasking quick sort in %12.10fs. \n", end - start);
    if (check_array_order(arr, size) == 0)
        printf("### Above array is not sorted. ### \n");
    
    return end - start;
}


double time_quick_sort_sections(double *arr, int size, int cutoff)
{
    double start = omp_get_wtime();
    quick_sort_sections(arr, 0, size-1, cutoff);
    double end = omp_get_wtime();

    printf("Sections quick sort in %12.10fs. \n", end - start);
    if (check_array_order(arr, size) == 0)
        printf("### Above array is not sorted. ### \n");
    
    return end - start;
}


// Merge Sort

double time_merge_sort_serial(double *arr, double *temp, int size)
{
    double start = omp_get_wtime();
    merge_sort(arr, temp, size);  // size-1 is final idx pos
    double end = omp_get_wtime();

    printf("Serial merge sort in %12.10fs. \n", end - start);
    if (check_array_order(arr, size) == 0)
        printf("### Above array is not sorted. ### \n");
    
    return end - start;
}

double time_merge_sort_tasks(double *arr, double *temp, int size, int cutoff)
{
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        merge_sort_tasks(arr, temp, size, cutoff);
    }   
    double end = omp_get_wtime();

    printf("Tasking merge sort in %12.10fs. \n", end - start);
    if (check_array_order(arr, size) == 0)
        printf("### Above array is not sorted. ### \n");
    
    return end - start;
}

double time_merge_sort_sections(double *arr, double *temp, int size, int cutoff)
{
    double start = omp_get_wtime();
    merge_sort_sections(arr, temp, size, cutoff);
    double end = omp_get_wtime();

    printf("Sections merge sort in %12.10fs. \n", end - start);
    if (check_array_order(arr, size) == 0)
        printf("### Above array is not sorted. ### \n");

    return end - start;
}

double time_enumeration_sort_serial(double *arr, double *temp, int size)
{
    double start = omp_get_wtime();
    enumeration_sort(arr, temp, size);
    double end = omp_get_wtime();

    printf("\nEnumeration sort in %12.10fs. \n", end - start);
    if (check_array_order(arr, size) == 0)
        printf("### Above array is not sorted. ### \n");

    return end - start;
}

double time_enumeration_sort_parallel(double *arr, double *temp, int size)
{
    double start = omp_get_wtime();
    enumeration_sort_parallel(arr, temp, size);
    double end = omp_get_wtime();

    printf("Parallel enumeration sort in %12.10fs. \n", end - start);
    if (check_array_order(arr, size) == 0)
        printf("### Above array is not sorted. ### \n");

    return end - start;
}