#include "omp.h" /* OpenMP compiler directives, APIs are declared here */
#include<stdio.h>
void printHello(int k);
void main()
{
    /* Parallel region begins here */
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printHello(ID);
    }
}
void printHello(int k)
{
    printf("Hello World %d\n",k);
}