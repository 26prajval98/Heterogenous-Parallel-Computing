#include "omp.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
double rng_doub(double range);
int rng_int(void);
double serial();
double parallel();
int state,state1;
#define RNG_MOD 0x80000000
void main()
{
    //Serial
    double start = omp_get_wtime();
    double pi_serial = serial();
    double end = omp_get_wtime();
    printf("The value of pi from serial calculations is: %f\nTime taken is: %lf\n",pi_serial,end-start);

    //Parallel
    start = omp_get_wtime();
    double pi_parallel = parallel();
    end = omp_get_wtime();
    printf("The value of pi from serial calculations is: %f\nTime taken is: %lf\n",pi_parallel,end-start);
}
double serial()
{
    int i, numIn, n;
    double x, y, pi;

    n = 1<<30;
    numIn = 0;
        
    state1 = 25234 + 17;
    for (i = 0; i <= n; i++)
    {
        x = (double)rng_doub(1.0);
        y = (double)rng_doub(1.0);
        if (x*x + y*y <= 1) numIn++;
    }
    pi = 4.*numIn / n;
    return pi;
}

double parallel()
{
    int i, numIn, n;
    double x, y, pi;

    n = 1<<30;
    numIn = 0;

    #pragma omp threadprivate(state)
    #pragma omp parallel private(x, y) reduction(+:numIn) 
    {
        
        state = 25234 + 17 * omp_get_thread_num();
        #pragma omp for
        for (i = 0; i <= n; i++) {
            x = (double)rng_doub(1.0);
            y = (double)rng_doub(1.0);
            if (x*x + y*y <= 1) numIn++;
        }
    }
    pi = 4.*numIn / n;
    return pi;
}

int rng_int(void) {
   return (state = (state * 1103515245 + 12345) & 0x7fffffff);
}

double rng_doub(double range) {
    return ((double)rng_int()) / (((double)RNG_MOD)/range);
}
