/* main.c */
#include "experiments_sort.h"

#define MAX_THREADS 3  // 2**4 = 16 threads
#define TRIALS 3

// #define SAVE 0  // save to file
#define VERBOSE 1
#define RUN_SERIAL 1
#define RUN_PARALLEL 1

int main(int argc, char* argv[])
{
    // MPI
    int rank, world_size, provided;
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
    int num_threads;

    // run configuration
    if (rank == 0)
    {
        printf("SETTINGS: VERBOSE=%d | RUN_SERIAL=%d | RUN_PARALLEL=%d\n", VERBOSE, RUN_SERIAL, RUN_PARALLEL);  
        if (VERBOSE)
        {
            printf("OpenMP Max Threads: %d | ", (int)pow(2, MAX_THREADS));
            printf("MPI Processes: %d\n", world_size);
        }
    }

    // run specifications
    int size;
    int n_arrays = 5;
    int ARRAY_SIZES[5] = {10000, 100000, 1000000, 10000000, 100000000};

    // experiment timers
    int trial;
    double runtime, max_runtime;
    double duration;  // sorting algorithm runtime
    runtime = MPI_Wtime();  // runtime for all experiments

    // output .csv to save experiment runtime results
    FILE *fptr;
    char results_file[] = "results/mpi_sort_results.csv";

    char file_name[64];  // array to read in for sorting
    char algorithm[32]; // sorting algorithm
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
    
    // experiments that do not use MPI - single process on rank 0 only
    if (rank == 0 && RUN_SERIAL != 0)
    {
        omp_set_num_threads(1);
        if (rank == 0)
        {
            printf("Running parallel experiments with %d processes\n", world_size);
            printf("OpenMP Threads: %d\n", omp_get_max_threads());
        }

        for (int s = 0; s < n_arrays; s++)
        {
            size = ARRAY_SIZES[s];
            snprintf(file_name, sizeof(file_name), "data/parallel/unsorted_%d.bin", size);
            
            // dynamic memory array
            double *random_arr = malloc(size * sizeof(double));
            double *arr = malloc(size * sizeof(double));
            double *temp = malloc(size * sizeof(double));
            
            // build array of random doubles in [0, 1]
            // fill_double_array(random_arr, size);
            read_array(file_name, random_arr, size);

            for (int i=0; i < TRIALS; i++)
            {
                // copy array and write quick sort serial results
                copy_double_array(random_arr, arr, size);

                // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                fprintf(
                    fptr, "serial,quick_sort,%d,single,%d,%d,%.17g\n",
                    1, 1, size, time_quick_sort_serial(arr, size)
                );
                // merge sort serial
                reset_double_array(temp, size);  // reset to 0
                copy_double_array(random_arr, arr, size);
                
                // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                fprintf(
                    fptr, "serial,merge_sort,%d,single,%d,%d,%.17g\n",
                    1, 1, size, time_merge_sort_serial(arr, temp, size)
                );

                // Enumeration sort not viable for extremely large arrays
                if (size <= 100000)
                {   
                    // enumeration sort serial
                    reset_double_array(temp, size);  // reset to 0
                    copy_double_array(random_arr, arr, size);
                    
                    // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                    fprintf(
                        fptr, "serial,enumeration_sort,%d,single,%d,%d,%.17g\n",
                        1, 1, size, time_enumeration_sort_serial(arr, temp, size)
                    );
                }
            }

            // loop through each thread
            for (int n = 1; n < MAX_THREADS+1; n++)
            {
                num_threads = (int)pow(2,n);  // 2, 4, 8, 16
                omp_set_num_threads(num_threads);
                printf("\nOpenMP Threads: %d\n", omp_get_max_threads());

                // Note: all OpenMP implementations would use cutoff=100

                // loop through experiments n=TRIALS times
                for (int i=0; i < TRIALS; i++)
                {
                    printf("\nTrial %d:\n", i);

                    // QUICK SORT
                    
                    // openmp quick sort - tasks
                    copy_double_array(random_arr, arr, size);

                    // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                    fprintf(
                        fptr, "serial,quick_sort,%d,tasks,%d,%d,%.17g\n",
                        1, num_threads, size, time_quick_sort_tasks(arr, size, 100)
                    );

                    // openmp quick sort - sections
                    // copy_double_array(random_arr, arr, size);
                    // fprintf(
                    //     fptr, "quick,sections,%d,%d,%.17g\n",
                    //     num_threads, size, time_quick_sort_sections(arr, size, cutoff)
                    // );

                    // MERGE SORT

                    // openmp merge sort - tasks
                    reset_double_array(temp, size);  // reset to 0
                    copy_double_array(random_arr, arr, size);

                    // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                    fprintf(
                        fptr, "serial,merge_sort,%d,tasks,%d,%d,%.17g\n",
                        1, num_threads, size, time_merge_sort_tasks(arr, temp, size, 100)
                    );

                    // openmp sections merge sort
                    // reset_double_array(temp, size);  // reset to 0
                    // copy_double_array(random_arr, arr, size);
                    // fprintf(
                    //     fptr, "merge,sections,%d,%d,%.17g\n",
                    //     num_threads, size, time_merge_sort_sections(arr, temp, size, 100)
                    // );

                    // ENUMERATION SORT
                    
                    // Enumeration sort not viable for arrays bigger than the stack
                    if (size <= 100000)
                    {   
                        // openmp enumeration sort parallel for
                        reset_double_array(temp, size);  // reset to 0
                        copy_double_array(random_arr, arr, size);

                        // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                        fprintf(
                            fptr, "serial,enumeration_sort,%d,parallel,%d,%d,%.17g\n",
                            1, num_threads, size, time_enumeration_sort_parallel(arr, temp, size)
                        );
                    }
                }
            }

            // free memory as we remake arrays with new size
            free(temp);
            free(arr);
            free(random_arr);
        }
    }

    // all processes wait    
    MPI_Barrier(MPI_COMM_WORLD);

    // all experiments that do use MPI
    if (RUN_PARALLEL != 0)
    {
        // loop through each thread size (dont want to switch threads frequently)
        for (int n = 0; n < MAX_THREADS+1; n++)
        {
            num_threads = (int)pow(2, n);  // 1, 2, 4, 8... 16
            omp_set_num_threads(num_threads);
            if (rank == 0)
            {
                printf("Running MPI Parallel experiments with %d processes\n", world_size);
                printf("\nOpenMP Threads: %d\n", omp_get_max_threads());
            }

            // loop over multiple array sizes
            for (int s = 0; s < n_arrays; s++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                size = ARRAY_SIZES[s];

                snprintf(file_name, sizeof(file_name), "data/parallel/unsorted_%d.bin", size);

                // loop through trials
                for (trial=0; trial < TRIALS; trial++)
                {
                    // ENUMERATION SORT - ONE THREAD WITH MPI MERGE AND MPI PARTITION

                    // sort array with (openmp single threaded) enumeration sort (if size is below 100,000)
                    if (size <= 100000)
                    {
                        // select sorting algorithm implementation based on num_threads
                        snprintf(algorithm, sizeof(algorithm), "enumeration");
                        if (num_threads == 1)
                            snprintf(construct, sizeof(construct), "single");
                        else
                            snprintf(construct, sizeof(construct), "parallel");

                        // run mpi_merge experiment
                        duration = run_mpi_merge(MPI_COMM_WORLD, world_size, rank, size, algorithm, construct, file_name, NULL);
                        if (rank == 0)
                        {
                            // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                            fprintf(
                                fptr, "merge,%s_sort,%d,%s,%d,%d,%.17f\n",
                                algorithm, world_size, construct, num_threads, size, duration
                            );
                            if (VERBOSE)
                                printf("%s sort (%s) mpi_merge for array of size %d trial %d took %.7fs.\n", algorithm, construct, size, trial, duration);
                        }

                        // run mpi_partition experiment
                        duration = run_mpi_partition(MPI_COMM_WORLD, world_size, rank, size, algorithm, construct, file_name, NULL);
                        if (rank == 0)
                        {
                            // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                            fprintf(
                                fptr, "partition,%s_sort,%d,%s,%d,%d,%.17f\n",
                                algorithm, world_size, construct, num_threads, size, duration
                            );
                            if (VERBOSE)
                                printf("%s sort (%s) mpi_partition for array of size %d trial %d took %.7fs.\n", algorithm, construct, size, trial, duration);
                        }
                    }

                    // MERGE SORT - ONE THREAD WITH MPI MERGE AND MPI PARTITION

                    // select sorting algorithm implementation based on num_threads
                    snprintf(algorithm, sizeof(algorithm), "merge");
                    if (num_threads == 1)
                        snprintf(construct, sizeof(construct), "single");
                    else
                        snprintf(construct, sizeof(construct), "tasks");

                    // run mpi_merge experiment
                    duration = run_mpi_merge(MPI_COMM_WORLD, world_size, rank, size, algorithm, construct, file_name, NULL);
                    if (rank == 0)
                    {
                        // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                        fprintf(
                            fptr, "merge,%s_sort,%d,%s,%d,%d,%.17f\n",
                            algorithm, world_size, construct, num_threads, size, duration
                        );
                        if (VERBOSE)
                            printf("%s sort (%s) mpi_merge for array of size %d trial %d took %.7fs.\n", algorithm, construct, size, trial, duration);
                    }

                    // run mpi_partition experiment
                    duration = run_mpi_partition(MPI_COMM_WORLD, world_size, rank, size, algorithm, construct, file_name, NULL);
                    if (rank == 0)
                    {
                        // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                        fprintf(
                            fptr, "partition,%s_sort,%d,%s,%d,%d,%.17f\n",
                            algorithm, world_size, construct, num_threads, size, duration
                        );
                        if (VERBOSE)
                            printf("%s sort (%s) mpi_partition for for array of size %d trial %d took %.7fs.\n", algorithm, construct, size, trial, duration);
                    }

                    // QUICK SORT - ONE THREAD WITH MPI MERGE AND MPI PARTITION

                    // select sorting algorithm implementation based on num_threads
                    snprintf(algorithm, sizeof(algorithm), "quick");
                    if (num_threads == 1)
                        snprintf(construct, sizeof(construct), "single");
                    else
                        snprintf(construct, sizeof(construct), "tasks");

                    // run mpi_merge experiment
                    duration = run_mpi_merge(MPI_COMM_WORLD, world_size, rank, size, algorithm, construct, file_name, NULL);
                    if (rank == 0)
                    {
                        // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                        fprintf(
                            fptr, "merge,%s_sort,%d,%s,%d,%d,%.17f\n",
                            algorithm, world_size, construct, num_threads, size, duration
                        );
                        if (VERBOSE)
                            printf("%s sort (%s) mpi_merge for array of size %d trial %d took %.7fs.\n", algorithm, construct, size, trial, duration);
                    }

                    // run mpi_partition experiment
                    duration = run_mpi_partition(MPI_COMM_WORLD, world_size, rank, size, algorithm, construct, file_name, NULL);
                    if (rank == 0)
                    {
                        // columns=[mpi:approach, sort_algorithm, mpi:world_size, openmp:construct, openmp:threads, size, duration]
                        fprintf(
                            fptr, "partition,%s_sort,%d,%s,%d,%d,%.17f\n",
                            algorithm, world_size, construct, num_threads, size, duration
                        );
                        if (VERBOSE)
                            printf("%s sort (%s) mpi_partition for array of size %d trial %d took %.7fs.\n", algorithm, construct, size, trial, duration);
                    }
                }
            }
        }
    }

    // end timer and close file
    runtime = MPI_Wtime() - runtime;
    MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        fclose(fptr);
        if (VERBOSE)
            printf("All sorting experiments with %d processes completed in %.7gs.\n\n", world_size, max_runtime);
    }

    MPI_Finalize();
    return 0;
}
