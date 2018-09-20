#include "omp.h" /* OpenMP compiler directives, APIs are declared here */
#include<stdio.h>
void main()
{
    /* Parallel region begins here */
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf("Hello World\n");
    }
}