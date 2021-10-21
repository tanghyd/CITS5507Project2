/* random_array.h */
#ifndef RANDOM_ARRAY
#define RANDOM_ARRAY

#include <stdio.h>
#include <stdlib.h>

// Functions to print an array
void fill_double_array(double arr[], int size);
void print_double_array(double arr[], int start, int end);
void copy_double_array(double arr[], double arr_copy[], int size);
void reset_double_array(double arr[], int size);

// array validation
int check_array_order(double arr[], int size);
int check_array_equality(double arr1[], double arr2[], int size);
int get_maximum_idx(double arr[], int size);
int get_mininimum_idx(double arr[], int size);

// IO utils
int write_array(char *file_name, double *array, int size);
int read_array(char *file_name, double *array, int size);

#endif