/* quick_sort */

#include "quick_sort.h"
#include "random_array.h"

/* Function Definitions */


/*-------------------------------------------------------------------
 * Function:    mpi_partition
 *       
 *       Re-organises arrays from all processes until we have a
 *       each one has an array where values in rank i are less than
 *       all values in rank i+1 (the array values themselves are not sorted).
 *           
 *       Pivots are sent according to the following (consider 8 workers with max depth of log2(8)):
 *       depth: 0 | source: 0 | destination: [0, 1, 2, 3, 4, 5, 6, 7]
 *       depth: 1 | source: 0 | destination: [0, 1, 2, 3]  # lower half
 *       depth: 1 | source: 4 | destination: [4, 5, 6, 7]  # upper half
 *       depth: 2 | source: 0 | destination: [0, 1]   # lowest quarter
 *       depth: 2 | source: 2 | destination: [2, 3]   # middle lower
 *       depth: 2 | source: 4 | destination: [4, 5]   # middle upper
 *       depth: 2 | source: 6 | destination: [6, 7]   # upper quarter
 *
 *       Likewise for 4 workers:
 *       depth: 0 | source: 0 | destination: [0, 1, 2, 3]
 *       depth: 1 | source: 0 | destination: [0, 1]
 *       depth: 1 | source: 2 | destination: [2, 3]
 *    
 *       Assumes number of processes is 2^n; adapted from:
 *       https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Ramkumar-Spring-2014-CSE633.pdf
 *
 *
 *  Arguments:
 *      comm: MPI_COMM_WORLD object from MPI.
 *      world_size: total number of processes.
 *      rank: process id.
 *      *array: an array chunk of doubles.
 *      size: The total size of the array.
 *
 *      This function will alter *array and size in replace - returning the pointer
 *      to the newly ordered chunk and its corresponding chunk size.

 */

double* mpi_partition(MPI_Comm comm, int world_size, int rank, double *array, int *size)
{
    MPI_Request request;
    MPI_Status status;

    double pivot;  // value in array used as pivot
    int pivot_source, pivot_destination;  // ranks to send pivots to
    int partition_partner, partition_size, partition_idx;  // ranks to share partitions
    int lower_size, upper_size, new_size;
    int i, j, k;  // loop iterators
    
    // int size = chunk_size;
    // double *array;  // array pointer we'll use between tree recursions
    // array = chunk;
    double *new_array; // temp array for transfering data between processes

    // recurse through tree of depth log2(n_processes)
    int depth = (int)log2(world_size);
    for (int current_depth = 0; current_depth < depth; current_depth++)
    {
        // work out which processes to send data to/fr
        partition_size = world_size / (int)pow(2, current_depth);  // process groups
        pivot_source = (int)(rank / partition_size) * partition_size;  // rank sends to its group

        // if rank matches, we get partition from data array on this rank
        if (rank == pivot_source)
        {
            // 1. Get pivot from one of the processes and send to every process in group
            pivot = array[*size-1];  // we select pivot by last element

            // loop through processes to check pivot destinations
            for (k=0; k<partition_size; k++)
            {
                pivot_destination = k + rank;  // apply offset per partition group
                if (rank != pivot_destination)
                {
                    // printf("depth %d | source %d --pivot--> destination %d\n", current_depth, rank, pivot_destination);
                    MPI_Send(&pivot, 1, MPI_DOUBLE, pivot_destination, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            // handle receiving from the source process
  		    MPI_Recv(&pivot, 1, MPI_DOUBLE, pivot_source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /*  2. Each process divides its unsorted list into two lists: those smaller
            than (or equal to) the pivot, and those greater than the pivot */
            
        // partition arrays into [ x < pivot , x >= pivot].
        i = -1; // number of elements less than pivot - 1

        // #pragma omp parallel for shared(array, i)
        for (j = 0; j < *size; j++)
        {
            // if current element is smaller than the pivot
            if (array[j] < pivot)
            {
                i++; // increment index of smaller element
                swap(&array[i], &array[j]);  // pass mem address of array
            }
        }

        /* swap final value & pivot line of code is present in quick sort but
        here other ranks won't have the exact pivot value - this can cause bugs */
        // swap(&array[i + 1], &array[size-1]);

        // this index is the first element position of the upper array
        partition_idx = i + 1;

        /*  3. Each process in the upper half of the process list sends its "lower list" 
            in the lower half of the process list and receives a "high list" in return. */

        // compute swapping partner process index to send half of array to
        partition_partner = (rank + (int)(partition_size/2)) % partition_size;
        partition_partner += ((int)(rank/partition_size) * partition_size);  // take floor of partition group

        // communicate array sizes by integers
        lower_size = partition_idx;
        upper_size = *size - partition_idx;
        if (rank > partition_partner)
        {
            // send lower half and new size will be [new + upper]
            MPI_Isend(&lower_size, 1, MPI_INT, partition_partner, 0, MPI_COMM_WORLD, &request);
            MPI_Irecv(&new_size, 1, MPI_INT, partition_partner, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
            *size = new_size + upper_size;
            
        } else {
            // send upper half and new size will be [lower + new]
            MPI_Isend(&upper_size, 1, MPI_INT, partition_partner, 0, MPI_COMM_WORLD, &request);
            MPI_Irecv(&new_size, 1, MPI_INT, partition_partner, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
            *size = lower_size + new_size;
        }

        // allocate new memory to store incoming chunks
        new_array = malloc(*size*sizeof(double));

        // transfer array data
        if (rank > partition_partner)
        {
            // send lower half data y1 from [y1, y2] and receive external upper half data x2 for [x2, y2]
            MPI_Isend(array, lower_size, MPI_DOUBLE, partition_partner, 0, MPI_COMM_WORLD, &request);
            MPI_Irecv(new_array, new_size, MPI_DOUBLE, partition_partner, 1, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);

            // #pragma omp parallel for shared(new_array, array)  // slower!
            for (i = 0; i < upper_size; i++)
                new_array[new_size+i] = array[lower_size+i];


        } else {
            // send upper half data x2 from [x1, x2] and receive external lower half data y1 for [x1, y1]
            MPI_Isend(&array[lower_size], upper_size, MPI_DOUBLE, partition_partner, 1, MPI_COMM_WORLD, &request);
            MPI_Irecv(&new_array[lower_size], new_size, MPI_DOUBLE, partition_partner, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);

            // #pragma omp parallel for shared(new_array, array)  // slower!
            for (i = 0; i < lower_size; i++)
                new_array[i] = array[i];
        }
        
        /*  4. In the next step of the tree, the processes will divide themselves
            into two groups and the algorithm is recursive. */
        /*  5. Here the upper-half processes have only values greater than the pivot,
            and the lower-half processes have only values smaller than the pivot. */

        // re-assign pointer to new array and clear old one
        MPI_Barrier(MPI_COMM_WORLD);  // sync processes in case we free memory in use
        free(array);
        array = new_array;  // do i need to do *array = new_array?
    }

    /*  6. After log(P) recursions(!), every process has an unsorted list of values
        completely disjoint from the values held by other processes. */
    /*  7. The largest value on process i will be smaller than the smallest value
        held by process i + 1. */
    // 8. Each process now can sort its list using sequential quicksort.
    return array;
}



/*-------------------------------------------------------------------
 * Function:    mpi_gather_quick_sort
 *       
 *      Takes a collection of chunks where each rank is ordered according
 *      to the output of mpi_partition and runs a quick sort algorithm on
 *      each process independently. The results are then gathered into the
 *      data pointer provided in the arguments.
 *
 *      If the array chunks on each process are not ordered according to
 *      mpi_partition this will not yield a fully sorted data array.
 *
 *  Arguments:
 *      comm: MPI_COMM_WORLD object from MPI.
 *      world_size: total number of processes.
 *      rank: process id.
 *      *data: a double array pointer whose size is the sum of all chunk sizes
 *             (memory must be allocated on rank 0).
 *      *chunk: a double array pointer to an ordered chunked array.
 *      chunk_size: the size of each array chunk according to each process.
 *      *construct: determines which OpenMP construct is used when calling sort.
 *
 *  Returns:
 *      Maximum runtime duration over all processes.
 */
double* mpi_gatherv_quick_sort(MPI_Comm comm, int world_size, int rank, double *data, double *chunk, int chunk_size, char *construct)
{
    // check if valid OpenMP consrtuct is provided
    if (!((strcmp("serial", construct)==0) || (strcmp("tasks", construct)==0) || (strcmp("sections", construct)==0)))
    {
        printf("Warning! Only 'serial', 'tasks', and sections' are valid for quick_sort. Received: %s. Aborting!\n", construct);
        return data;
    }

    // separately sort chunks using a quick sort algorithm
    if (strcmp("serial", construct)==0)
        quick_sort(chunk, 0, chunk_size-1);
    else if (strcmp("tasks", construct)==0) 
        quick_sort_tasks(chunk, 0, chunk_size-1, 100);
    else if (strcmp("sections", construct)==0) 
        quick_sort_sections(chunk, 0, chunk_size-1, 100);

    // pack arrays together and communicate to rank 0 expected chunk size
    // double *data;
    // int total_size;
    int recvcounts[world_size];
    int displacements[world_size];
    MPI_Gather(&chunk_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // compute aggregated array size and displacement buffers for MPI_Gatherv
    if (rank == 0)
    {
        displacements[0] = 0;
        for (int i = 0; i < world_size-1; i++)
        {
            displacements[i+1] = displacements[i] + recvcounts[i];
        }

        // allocate memory on rank 0
        // total_size = 0;
        // for (i = 0; i < world_size; i++)
        //     total_size += recv_counts[i];
        // data = malloc(total_size*sizeof(double));
    }

    // gather variable length arrays to data - data must be allocated memory equal to correct total size
    MPI_Gatherv(chunk, chunk_size, MPI_DOUBLE, data, recvcounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // free(chunk);  // should this be done outside function?
    return data;
}

/* Utility function to swap two elements by memory address. */
void swap(double *a, double *b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}


/* Standard last-element partition function for quick sort. */
int partition(double *arr, int low, int high)
{
    double pivot = arr[high];
    int i = low - 1;

    // loop through partitioned array indices
    for (int j = low; j < high; j++)
    {
        // if current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);  // pass mem address of array
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}


/* Serial recursive quick sort */
void quick_sort(double *arr, int low, int high)
{
    {
        if (low >= 0 && high >= 0)
        {        
            if (low < high)
            {
                int pi = partition(arr, low, high);

                // separately sort elements before and after pi
                quick_sort(arr, low, pi - 1);
                quick_sort(arr, pi + 1, high);
            }
        }
    }
}


/* Parallel recursive quicksort - tasking */
void quick_sort_tasks(double *arr, int low, int high, int cutoff)
{
    {
        if (low >= 0 && high >= 0)
        {        
            if (low < high)
            {
                int pi = partition(arr, low, high);

                // separately sort elements before and after pivot
                // worker takes task itself if array bigger than cutoff
                #pragma omp task shared(arr) firstprivate(low, pi) if (high - low > cutoff)
                {
                    quick_sort_tasks(arr, low, pi - 1, cutoff);
                }
                #pragma omp task shared(arr) firstprivate(pi, high) if (high - low > cutoff)
                {
                    quick_sort_tasks(arr, pi + 1, high, cutoff);
                }
            }
        }
    }
}


/* Parallel recursive quicksort - sections */
void quick_sort_sections(double *arr, int low, int high, int cutoff)
{
    if (low >= 0 && high >= 0 && low < high)
    {        
        /* pi is the "partitioning index, where
        arr[pi] is swapped to correct location*/
        int pi = partition(arr, low, high);

        if (high - low > cutoff)
        {
            // array large enough to warrant parallelism
            #pragma omp parallel shared(arr)
            #pragma omp sections
            {
                #pragma omp section
                {
                    quick_sort_sections(arr, low, pi - 1, cutoff);
                }
                #pragma omp section
                {
                    quick_sort_sections(arr, pi + 1, high, cutoff);
                }
            }
        }
        else
        {
            // serial implementation
            quick_sort(arr, low, pi - 1);
            quick_sort(arr, pi + 1, high);
        }
    }
}


/*-------------------------------------------------------------------
 * Function:    mpi_quick_sort
 *       
 *       Merges sorted arrays from processes until we have a
 *       single array containing all integers in sorted order
 *           
 *       Pivots are sent according to the following (consider 8 workers with max depth of log2(8)):
 *       depth: 0 | source: 0 | destination: [0, 1, 2, 3, 4, 5, 6, 7]
 *       depth: 1 | source: 0 | destination: [0, 1, 2, 3]  # lower half
 *       depth: 1 | source: 4 | destination: [4, 5, 6, 7]  # upper half
 *       depth: 2 | source: 0 | destination: [0, 1]   # lowest quarter
 *       depth: 2 | source: 2 | destination: [2, 3]   # middle lower
 *       depth: 2 | source: 4 | destination: [4, 5]   # middle upper
 *       depth: 2 | source: 6 | destination: [6, 7]   # upper quarter
 *
 *       Likewise for 4 workers:
 *       depth: 0 | source: 0 | destination: [0, 1, 2, 3]
 *       depth: 1 | source: 0 | destination: [0, 1]
 *       depth: 1 | source: 2 | destination: [2, 3]
 *    
 *       Assumes number of processes is 2^n; adapted from:
 *       https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Ramkumar-Spring-2014-CSE633.pdf
 */

// double* mpi_partitioned_quick_sort(MPI_Comm comm, int rank, int world_size, double *chunk, int chunk_size, double *data)
// {
//     /*  3. Each process in the upper half of the process list sends its "lower list" 
//         in the lower half of the process list and receives a "high list" in return. 

//         */

//     MPI_Request request;
//     MPI_Status status;

//     double pivot;  // value in array used as pivot
//     int pivot_source, pivot_destination;
//     int partition_partner, partition_size;
//     int lower_size, upper_size, new_size;
//     int i, j, k;  // loop iterators
    
//     int size = chunk_size;
//     double *new_array; // temp array for transfering data between processes
//     double *array;  // array pointer we'll use between tree recursions
//     array = chunk;

//     // DEBUG
//     // int max_idx = get_maximum_idx(array, size);
//     // int min_idx = get_mininimum_idx(array, size);
//     // printf(
//     //     "START   | rank %d | size: %4d |          \
//     //     min: array[%4d]=%lf | max: array[%4d]=%lf     \
//     //     array[0]=%lf | array[1]=%lf | array[2] = %lf \
//     //     \n",
//     //     rank, size, min_idx, array[min_idx], max_idx, array[max_idx],
//     //     array[0], array[1], array[2]
//     // );
//     // fflush(stdout);
//     // MPI_Barrier(MPI_COMM_WORLD);

//     // recurse through tree of depth log2(n_processes)
//     int depth = (int)log2(world_size);
//     for (int current_depth = 0; current_depth < depth; current_depth++)
//     {
//         // DEBUG
//         // double sorted_total = 0; // reduction
//         // double sorted_array_total=0;
//         // for (i = 0; i < size; i++)
//         //     sorted_array_total += array[i];
//         // MPI_Reduce(&sorted_array_total, &sorted_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//         // if (rank == 0)  
//         //     printf("depth %d sorted_total=%lf\n", current_depth, sorted_total);
//         // fflush(stdout);

//         // work out which processes to send data to/fr
//         partition_size = world_size / (int)pow(2, current_depth);  // process groups
//         pivot_source = (int)(rank / partition_size) * partition_size;  // rank sends to its group

//         // if rank matches, we get partition from data array on this rank
//         if (rank == pivot_source)
//         {
//             // 1. Get pivot from one of the processes and send to every process in group
//             pivot = array[size-1];  // we select pivot by last element

//             // loop through processes to check pivot destinations
//             for (k=0; k<partition_size; k++)
//             {
//                 pivot_destination = k + rank;  // apply offset per partition group
//                 if (rank != pivot_destination)
//                 {
//                     // printf("depth %d | source %d --pivot--> destination %d\n", current_depth, rank, pivot_destination);
//                     MPI_Send(&pivot, 1, MPI_DOUBLE, pivot_destination, 0, MPI_COMM_WORLD);
//                 }
//             }
//         } else {
//             // handle receiving from the source process
//   		    MPI_Recv(&pivot, 1, MPI_DOUBLE, pivot_source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         }

//         /*  2. Each process divides its unsorted list into two lists: those smaller
//             than (or equal to) the pivot, and those greater than the pivot */
            
//         // partition arrays into [ x < pivot , x >= pivot].
//         i = -1; // number of elements less than pivot - 1

//         // #pragma omp parallel for shared(array, i)
//         for (j = 0; j < size; j++)
//         {
//             // if current element is smaller than the pivot
//             if (array[j] < pivot)
//             {
//                 i++; // increment index of smaller element
//                 swap(&array[i], &array[j]);  // pass mem address of array
//             }
//         }

//         /* swap final value & pivot is present in quick sort but in this case
//         other ranks won't have the exact pivot value - this can cause bugs */
//         // swap(&array[i + 1], &array[size-1]);

//         // this index is the first element position of the upper array
//         int partition_idx = i + 1;

//         // DEBUG
//         // check ordering of intermediate arrays (adds extra time)
//         // for (i=0; i < partition_idx; i++)
//         // {
//         //     if (pivot < array[i]) {
//         //         printf(
//         //             "ERROR rank %d | array[partition_idx]=%lf | array[%d]=%lf\n",
//         //             rank, array[partition_idx], i, array[i]
//         //         );
//         //     }
//         // }

//         // for (i=partition_idx; i < size; i++)
//         // {
//         //     // printf("array[partition_idx]=%lf\n", array[partition_idx]);
//         //     if (pivot > array[i]) {
//         //         printf(
//         //             "ERROR rank %d | array[partition_idx]=%lf | array[%d]=%lf\n",
//         //             rank, array[partition_idx], i, array[i]
//         //         );
//         //     }
//         // }

//         // compute swapping partner process index to send half of array to
//         partition_partner = (rank + (int)(partition_size/2)) % partition_size;
//         partition_partner += ((int)(rank/partition_size) * partition_size);  // take floor of partition group

//         // communicate array sizes by integers
//         lower_size = partition_idx;
//         upper_size = size - partition_idx;
//         if (rank > partition_partner)
//         {
//             // send lower half and new size will be [new + upper]
//             MPI_Isend(&lower_size, 1, MPI_INT, partition_partner, 0, MPI_COMM_WORLD, &request);
//             MPI_Irecv(&new_size, 1, MPI_INT, partition_partner, 0, MPI_COMM_WORLD, &request);
//             MPI_Wait(&request, &status);
//             size = new_size + upper_size;
            
//         } else {
//             // send upper half and new size will be [lower + new]
//             MPI_Isend(&upper_size, 1, MPI_INT, partition_partner, 0, MPI_COMM_WORLD, &request);
//             MPI_Irecv(&new_size, 1, MPI_INT, partition_partner, 0, MPI_COMM_WORLD, &request);
//             MPI_Wait(&request, &status);
//             size = lower_size + new_size;
//         }

//         // allocate new memory to store incoming chunks
//         new_array = malloc(size*sizeof(double));

//         // transfer array data
//         if (rank > partition_partner)
//         {
//             // send lower half data y1 from [y1, y2] and receive external upper half data x2 for [x2, y2]
//             MPI_Isend(array, lower_size, MPI_DOUBLE, partition_partner, 0, MPI_COMM_WORLD, &request);
//             MPI_Irecv(new_array, new_size, MPI_DOUBLE, partition_partner, 1, MPI_COMM_WORLD, &request);
//             MPI_Wait(&request, &status);  // non-blocking and wait prevents deadlock

//             // #pragma omp parallel for shared(new_array, array)  // slower!
//             for (i = 0; i < upper_size; i++)
//                 new_array[new_size+i] = array[lower_size+i];


//         } else {
//             // send upper half data x2 from [x1, x2] and receive external lower half data y1 for [x1, y1]
//             MPI_Isend(&array[lower_size], upper_size, MPI_DOUBLE, partition_partner, 1, MPI_COMM_WORLD, &request);
//             MPI_Irecv(&new_array[lower_size], new_size, MPI_DOUBLE, partition_partner, 0, MPI_COMM_WORLD, &request);
//             MPI_Wait(&request, &status);  // non-blocking and wait prevents deadlock

//             // #pragma omp parallel for shared(new_array, array)  // slower!
//             for (i = 0; i < lower_size; i++)
//                 new_array[i] = array[i];
//         }

//         /* Re-assign pointer to new array and clear old one
//         Race condition: we must sync processes to prevent 
//         freeing memory in use during point to point communcation */
//         MPI_Barrier(MPI_COMM_WORLD);
//         free(array);
//         array = new_array;

//         // DEBUG
//         // write file to disk for debug
//         // char file_name[50];
//         // snprintf(file_name, sizeof(file_name), "tests/test_%d_rank_%d_%d.out", size, rank, current_depth);
//         // printf("%s\n", file_name);
//         // write_array(file_name, array, size);

            
//         // DEBUG
//         // int max_idx = get_maximum_idx(array, size);
//         // int min_idx = get_mininimum_idx(array, size);
//         // printf(
//         //     "depth %d | rank %d | size: %4d | pivot @ %lf \
//         //     min: array[%4d]=%lf  | max: array[%4d]=%lf     \
//         //     array[0]=%lf | array[1]=%lf | array[2] = %lf\n",
//         //     current_depth, rank, size, pivot,
//         //     min_idx, array[min_idx], max_idx, array[max_idx],
//         //     array[0], array[1], array[2]
//         // );
//         // fflush(stdout);
//     }

//     /*  5. Now the upper-half processes have only values greater than the pivot,
//         and the lower-half processes have only values smaller than the pivot. */
//     /*  6. After log(P) recursions(!), every process has an unsorted list of values
//         completely disjoint from the values held by other processes. */
//     /*  7. The largest value on process i will be smaller than the smallest value
//         held by process i + 1. */
//     // 8. Each process now can sort its list using sequential quicksort.

//     // quick_sort(array, 0, size-1);
//     quick_sort_tasks(array, 0, size-1, 100);

//     // DEBUG
//     // int ordered = check_array_order(array, size);
//     // if (ordered == 0)
//     //     printf("Warning! rank %d is not correctly sorted!\n", rank);
//     // else
//     //     printf("rank %d successfully sorted.\n", rank);
        
//     // pack arrays together
//     int recvcounts[world_size];
//     int displacements[world_size];
    
//     MPI_Gather(&size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     if (rank == 0)
//     {
//         displacements[0] = 0;
//         for (i = 0; i < world_size-1; i++){
//             displacements[i+1] = displacements[i] + recvcounts[i];
        
//             // DEBUG
//             // printf(
//             //     "displacements[%d]=%d | recvcounts[%d]=%d\n",
//             //     i, displacements[i], i, recvcounts[i]
//             // ); 
//         }
//     }

//     MPI_Gatherv(array, size, MPI_DOUBLE, data, recvcounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     free(array);
//     return data;
// }