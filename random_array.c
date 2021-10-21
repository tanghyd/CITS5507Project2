/* random_array.h */
#include "random_array.h"

/* Function Definitions */

// fills an array with random doubles
void fill_double_array(double arr[], int size)
{
    // #pragma omp parallel for firstprivate(arr, size, max)
    for (int i = 0; i < size; i++)
    {
        arr[i] = (double)rand()/RAND_MAX; // range 0. to 1.
    }
}

void copy_double_array(double arr[], double arr_copy[], int size)
    {   
        for (int i = 0; i < size; i++)
        {
            arr_copy[i] = arr[i];
        }
    }

void reset_double_array(double arr[], int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = 0;
    }

}

void print_double_array(double arr[], int start, int end)
{
    for (int i = start; i < end; i++)
    {
        printf("%.15f ", arr[i]);
    }
    printf("\n");
}

// returns 1 if array is correctly sorted, else 0.
int check_array_order(double arr[], int size)
{
    int sorted = 1;
    for (int i=0; i < size-1; i++)
    {
        if (arr[i] > arr[i+1])
        {
            sorted = 0;
            break;
        }
    }
    return sorted;
}

int get_maximum_idx(double arr[], int size)
{
    int idx = 0;
    for (int i = 1; i < size-1; i++)
    {
        if (arr[i] > arr[idx])
            idx = i;
    }
    return idx;
}

int get_mininimum_idx(double arr[], int size)
{
    int idx = 0;
    for (int i = 1; i < size-1; i++)
    {
        if (arr[i] < arr[idx])
            idx = i;
    }
    return idx;
}


/*
 * Function:  check_array_equality 
 * --------------------
 * Compares two double arrays of equal size by comparing each element.
 * Assumes both arrays are of the same size.
 *
 *  arr1: a double array to compare.
 *  arr2: another double array to compare.
 *  size: the count of array elements.
 *
 *  Returns:
 *      1 ("True") if arrays are equal, else 0. (Different to write/read functions!)
 */
int check_array_equality(double arr1[], double arr2[], int size)
{
    int equal = 1;
    for (int i=0; i < size-1; i++)
        if (arr1[i] != arr2[i])
        {
            equal = 0;
            break;
        }
    return equal;
}


/*
 * Function:  write_array 
 * --------------------
 *  Writes a double array of length size to disk as "file_name".
 *  The file type is assumed to be binary.
 *
 *  Arguments:
 *      *file_name: a file_name to save the binary file.
 *      *array: double array to write data from.
 *      size: the count of array elements.
 *
 *  Returns:
 *      0 if success, else 1.
 */
int write_array(char *file_name, double *array, int size)
{
    int status = 0;
    FILE *fptr;
    fptr = fopen(file_name, "wb");
    if (fptr)
        fwrite(array, sizeof(array), size, fptr);
    else
    {
        printf("Failed to open file!\n");
        status = 1;
    }
    fclose(fptr);
    return status;  // 0 if success
}

/*
 * Function:  read_array 
 * --------------------
 *  Reads a double array with "size" elements to disk from "file_name".
 *  The file type is assumed to be binary and is saved to the specified array..
 *
 *  Arguments:
 *      *file_name: a file_name to save the binary file.
 *      *array: double array to write data to.
 *      size: the count of array elements.
 *
 *  Returns:
 *      0 if success, else 1.
 */
int read_array(char *file_name, double *array, int size)
{
    int status = 0;
    FILE *fptr;
    fptr = fopen(file_name, "rb");
    if (fptr)
        fread(array, sizeof(array), size, fptr);
    else
    {
        printf("Failed to open file!\n");
        status = 1;
    }
    fclose(fptr);
    return status;  // 0 if success

}
