#include "omp.h" /* OpenMP compiler directives, APIs are declared here */
#include<stdio.h>
void main()
{
    //Serial code
    double start1 = omp_get_wtime();
    printHello_Serial();
    double end1 = omp_get_wtime();
    printf("Time taken by serial code: %f\n",end1-start1);

    //Parallel code
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf("Hello World\n");
    }
    
}