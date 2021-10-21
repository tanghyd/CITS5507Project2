/* mpi_utils */

#include "mpi_utils.h"

/* Function Definitions */

/*-------------------------------------------------------------------
 * Function:    validate_equal_chunks
 *
 *              Checks whether array is equally divisible by world size.
 *              Useful if sorting implementation doesn't account for
 *              corner cases where array length may not be divisible equally.
 */
void validate_equal_chunks(int world_size, int n)
{
    if ((n % world_size) != 0) 
    {
        printf("Algorithm only implemented for array sizes divisible by world size. size=%d | world_size=%d\n", n, world_size); 
        MPI_Finalize();
        exit(1);
    }
}

/*-------------------------------------------------------------------
 * Function:    validate_log2_procs
 *              
 *              Checks if number of processes is a power of 2.
 *              Useful for "binary tree" implementations (mpi_merge_sort).
 */
void validate_log2_procs(int world_size, int n)
{
    if ((ceil(log2(world_size)) != floor(log2(world_size))))
    {
        printf("Number of processes (world_size) must be a power of 2. size=%d | world_size=%d\n", n, world_size);
        MPI_Finalize();
        exit(1);
    }
}