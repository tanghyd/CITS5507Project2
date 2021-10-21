#include "enumeration_sort.h"

/* Function Definitions */

void enumeration_sort(double arr[], double temp[], int size)
{
    // initialize enumerated array of ranks with 0s
    int ranks[size];
    for (int i = 0; i < size; i++)
        ranks[i] = 0;

    for (int i = 0; i < size; i++)
    {
        for (int j = i+1; j < size; j++)
        {
            if (arr[i] > arr[j])
                ranks[i]++;
            else
                ranks[j]++;
        }

        temp[ranks[i]] = arr[i];
    }

    // copy temp array into main array
    memcpy(arr, temp, size*sizeof(double));
}

void enumeration_sort_parallel(double arr[], double temp[], int size)
{
    // compare each element against other elements in parallel
    // rank is how many other elements it is greater than
    #pragma omp parallel for shared(arr, temp) schedule(guided)
    for (int i = 0; i < size; i++)
    {
        int rank = 0;
        for (int j = 0; j < size; j++)
        {
            if ((arr[i] > arr[j]) || (i > j && arr[i] == arr[j]))
                rank++;
        }
        temp[rank] = arr[i];
    }

    // copy temp array into main array
    memcpy(arr, temp, size*sizeof(double));
}