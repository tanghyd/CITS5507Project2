# CITS5507 Project 2

This repository stores the code used to implement serial and parallel implementations of quick sort, merge sort, and enumeration sorting using C, OpenMP, and MPI.

## Setup

Code was run on a machine with the following specifications:

- Operating System: Ubuntu 18.04.5 LTS
- CPU: AMD Ryzen Threadripper 2920X 12-Core Processor
- RAM: 62.7GB

For example, code was compiled (mpicc) and then run (mpiexec) with 4 processes with commands:

`mpicc -o experiments_io experiments_io.c random_array.c mpi_utils.c -lm`

`mpicc -fopenmp -o experiments_sort experiments_sort.c random_array.c mpi_utils.c merge_sort.c quick_sort.c enumeration_sort.c -lm`

`mpiexec -n 4 experiments_io`

`mpiexec -n 4 experiments_sort`


Data analysis and visualisations were done in Jupyter notebook in the `/notebook` directory, with performance data being saved during our experiments to the `/results` directory.
