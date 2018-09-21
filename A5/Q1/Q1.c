//16CO145 Sumukha PK
//16CO234 Prajval.M
#include "omp.h" /* OpenMP compiler directives, APIs are declared here */
#include<stdio.h>
void main()
{
    //Serial code
    double start1 = omp_get_wtime();
    printf("Hello World!\n");
    double end1 = omp_get_wtime();
    printf("Time taken by serial code: %f\n",end1-start1);

    //Parallel code
    start1 = omp_get_wtime();
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf("Hello World\n");
    }
    end1 = omp_get_wtime();
    printf("Time taken by parallel code: %f\n",end1-start1);
}
