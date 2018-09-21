#include "omp.h" 
#include<stdio.h>
#include<time.h>
void printHello_parallel(int k);
void printHello_Serial();
void main()
{
   
    //Serial code
    double start1 = omp_get_wtime();
    printHello_Serial();
    double end1 = omp_get_wtime();
    printf("Time taken by serial code: %f\n",end1-start1);

    //Parallel code
    omp_set_num_threads(4);
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printHello_parallel(ID);
    }
    double end = omp_get_wtime();
    printf("Time taken by parallel code: %f\n",end-start);
}
void printHello_parallel(int k)
{
    printf("Hello World %d\n",k);
}
void printHello_Serial()
{
    printf("Hello World!\n");
}