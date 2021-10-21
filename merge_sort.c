/* merge_sort */

#include "merge_sort.h"

/* Function Definitions */

/*-------------------------------------------------------------------
 * Function:    mpi_merge
 *      
 *      Merges sorted array chunks from processes until we have a
 *      single array containing all integers in sorted order.
 *
 *      References implementation from:
 *      http://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html
 */

double* mpi_merge(int depth, int rank, double *chunk, int size, MPI_Comm comm, double *data)
{
    int parent, right_child;
    double *half, *right_half, *result;

    int current_depth = 0;
    half = chunk;  // assign chunk to lower half ?

    // recurse through tree until we hit max depth
    while (current_depth < depth)
    {
        parent = (rank & (~(1 << current_depth)));

        if (parent == rank)
        {
            // array data is merged to left child in tree
		    right_child = (rank | (1 << current_depth));

  		    // allocate memory and receive array of right child
  		    right_half = malloc(size * sizeof(double));
  		    MPI_Recv(right_half, size, MPI_DOUBLE, right_child, 0, comm, MPI_STATUS_IGNORE);

  		    // allocate memory for result of merge
  		    result = malloc(2*size*sizeof(double));

            // merge sort into result (size = length of half)
  		    merge_halves(half, right_half, result, size);

  		    // reassign half to merge result
            half = result;
			size = size * 2;

			free(right_half); 
			result = NULL;

            current_depth++;

        } else {
			  // right child sends local chunk to parent and frees memory
              MPI_Send(half, size, MPI_DOUBLE, parent, 0, comm);
              if(current_depth != 0) free(half);  
              current_depth = depth;
        }
    }

    if (rank == 0)
	{
        data = half;
        return data;
    }
}

// void merge_sort_halves(double *half1, double *half2, double *result, int size)
// {
//     if (size < 2)
//         return;

//     // recursively sort subarrays
//     merge_sort_halves(arr, temp, size/2);
//     merge_sort_halves(arr + (size/2), temp + (size/2), size - (size/2));
//     merge_halves(arr, temp, size);
// }


/*-------------------------------------------------------------------
 * Function:    merge_halves
 *      
 *      Merges and sorts two arrays (half1 and half2) into a larger
 *      result array where half1 and half2 are of length size.
 */
 
void merge_halves(double *half1, double *half2, double *result, int size)
{
    // begin loop to copy from temp to main array in sorted order
    int i, j, n;
    i = j = n = 0;

    // loop if both subarrays have elements
    while (i < size && j < size)
    {
        // add smaller element to temp array
        if (half2[j] > half1[i])
        {
            result[n] = half1[i];
            i++;
        }
        else 
        {
            result[n] = half2[j];
            j++;
        }
        n++;
    }

    // if both conditions above are not satisfied
    // then one subarray has been emptied
    // loop through remaining (sorted) elements
    while (i < size)
    {
        result[n] = half1[i];
        i++; n++;
    }

    while (j < size)
    {
        result[n] = half2[j];
        j++; n++;
    }
}


/*-------------------------------------------------------------------
 * Function:    merge_sort
 *      
 *      Computes merge sort algorithm on array defined defined
 *      on the interval arr[low:high] using a temp copy array.
 */
 
void merge_sort(double *arr, double *temp, int size)
{
    if (size < 2)
        return;

    // recursively sort subarrays
    // select sub-arrays by incrementing pointer positions
    merge_sort(arr, temp, size/2);
    merge_sort(arr + (size/2), temp + (size/2), size - (size/2));
    merge(arr, temp, size);
}

/*-------------------------------------------------------------------
 * Function:    merge
 *      
 *      Merges array using array pointers and temp array.
 */
void merge(double *arr, double *temp, int size)
{
    // begin loop to copy from temp
    // to main array in sorted order
    int i = 0;
    int j = size/2;
    int n = 0;

    // loop if both subarrays have elements
    while (i<(size/2) && j<size)
    {
        // add smaller element to temp array
        if (arr[j] > arr[i])
        {
            temp[n] = arr[i];
            i++;
        }
        else 
        {
            temp[n] = arr[j];
            j++;
        }
        n++;
    }

    // if both conditions above not satisfied
    // then one subarray has been emptied
    // loop through remaining (sorted) elements
    while (i < (size/2))
    {
        temp[n] = arr[i];
        i++;
        n++;
    }

    while (j < size)
    {
        temp[n] = arr[j];
        j++;
        n++;
    }

    // copy temp array into main array
    memcpy(arr, temp, size*sizeof(double));
}

/*-------------------------------------------------------------------
 * Function:    merge_sort_tasks
 *      
 *      Computes merge sort algorithm on array defined defined
 *      on the interval arr[low:high] using a temp copy array.
 *      
 *      This approach uses a tasks implementation with OpenMP.
 */
void merge_sort_tasks(double *arr, double *temp, int size, int cutoff)
{
    if (size < 2)
        return;

    // recursively sort subarrays
    // select sub-arrays by incrementing pointer positions
    #pragma omp task shared(arr, temp) if (size > cutoff)
    merge_sort_tasks(arr, temp, size/2, cutoff);

    #pragma omp task shared(arr, temp) if (size > cutoff)
    merge_sort_tasks(arr + (size/2), temp + (size/2), size - (size/2), cutoff);

    #pragma omp taskwait
    merge(arr, temp, size);

}

/*-------------------------------------------------------------------
 * Function:    merge_sort_sections
 *      
 *      Computes merge sort algorithm on array defined defined
 *      on the interval arr[low:high] using a temp copy array.
 *      
 *      This approach uses a sections implementation with OpenMP.
 */
void merge_sort_sections(double *arr, double *temp, int size, int cutoff)
{
    if (size < 2)
        return;

    if (size > cutoff)
    {
        // recursively sort subarrays
        #pragma omp parallel shared(arr, temp)
        #pragma omp sections
        {
            #pragma omp section
            {
                // merge sort lower half
                merge_sort_sections(arr, temp, size/2, cutoff);
            }
            #pragma omp section
            {
                // merge sort upper half
                merge_sort_sections(arr + (size/2), temp + (size/2), size - (size/2), cutoff);
            }
        }
    }
    else
    {
        // else serial implementation
        merge_sort(arr, temp, size/2);
        merge_sort(arr + (size/2), temp + (size/2), size - (size/2));
    }
    merge(arr, temp, size);
}