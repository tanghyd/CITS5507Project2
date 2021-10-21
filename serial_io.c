#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "random_array.h"

#define SIZE 10001

int main(void)
{
    clock_t start, end; // timer variables for serial io
 
    // build array of random doubles in [0, 1]
    int size = SIZE;  // random array size
    double *random_array = malloc(size * sizeof(double));
    double *array = malloc(size * sizeof(double));
    fill_double_array(random_array, size);

    // write array as binary file to disk
    // char file_name[] = "data/unsorted_array.bin";
    char file_name[64];
    snprintf(file_name, sizeof(file_name), "data/unsorted_%d.bin", size);
    printf("%s\n", file_name);

    // timer writing file
    start = clock();
    write_array(file_name, random_array, size);
    end = clock();
    printf("Random array of size %d written in %fs.\n\n", size, (double)(end - start) / CLOCKS_PER_SEC);


    // timer for reading file
    start = clock();
    read_array(file_name, array, size);
    end = clock();
    printf("Loading array of size %d read in %fs.\n", size, (double)(end - start) / CLOCKS_PER_SEC);
    

    // validate arrays
    equal = check_array_equality(random_array, array, size);
    if (equal == 1)
        printf("Errays are equal.\n");
    else
        printf("Errays are not equal.\n");
        
    return 0;
}