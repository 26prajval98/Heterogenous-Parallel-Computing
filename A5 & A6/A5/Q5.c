#include "omp.h"
#include<stdio.h>
#include<time.h>
static long num_steps = 100000;
double step;
double pi_s();
void main()
{
    //Calculate serial code's time and output
    double start = omp_get_wtime();
    double pi_serial  = pi_s();
    double end = omp_get_wtime();
    printf("Time taken by serial code is %f\n",end-start);
 
    printf("Serial output value: %f\n",pi_serial);

    //Calculate parallel code's output and answer
    int i,nthrds;
    double x, pi, sum = 0.0;
    step = 1.0/(double)num_steps;
    omp_set_num_threads(10000);
    start = omp_get_wtime();
    #pragma omp parallel
    {
        float partial_sum;
        int i, id;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        step = 1.0/(double)num_steps;
        for(i=id;i<num_steps;i=i+nthrds)
        {
            double x = (i+0.5)*step;
            partial_sum= 4.0/(1.0+x*x);
            #pragma omp atomic                    //To ensure atomicity while adding
                sum = sum + partial_sum;
        }
    }
    end = omp_get_wtime();

    printf("Time taken by parallel code: %f\n",end-start);
    printf("Total threads = %d\n",nthrds);
    pi = step * sum;
    printf("Parallel Value: %f\n",pi);
}

//Serial code
double pi_s()
{
    int i;
    double x, pi, sum = 0.0;
    step = 1.0/(double)num_steps;
    for (i=0; i<num_steps; i++)
    {
        x = (i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }
    pi = step * sum;
    return pi;
}