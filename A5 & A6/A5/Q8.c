#include "omp.h"
#include<stdio.h>
#include<stdlib.h>
int Sum_array(double *A);
void fill_rand(double *A);
double get();
void producer(int tid);
void consumer(int tid);
void put(double item);
#define N 10
int i,j;
int nextin = 0;
int nextout = 0;
int count = 0;
int empty = 1;
int full = 0;
double *A;
double *B;
double S =0;
int main()
{
    double sum, runtime;
    int flag = 0;
    A = (double *)malloc(N*sizeof(double));
    B = (double *)malloc(N*sizeof(double));

    //Serial code
    runtime = omp_get_wtime();
    fill_rand(A);
    
    for(int h=0;h<N;h++)
        sum+=A[h];
    
    runtime = omp_get_wtime() - runtime;
    printf("In %lf seconds, The sum is %lf \n",runtime,sum);
 
    //Parallel code
    runtime = omp_get_wtime();
    int tid;
    int NUM_THREADS;
    i=j=0;
    omp_set_num_threads(100);
    #pragma omp parallel firstprivate(i,j) private(tid) 
    {
       tid=omp_get_thread_num();

       if(tid%2==1)
       {
           producer(tid);
       }
       else
       {
           consumer(tid);
       }
       NUM_THREADS = omp_get_num_threads();

    }
    runtime = omp_get_wtime() - runtime;
    printf("In %lf seconds,with %d threads The sum is %lf \n",runtime,NUM_THREADS,S);
    
    if(S==sum)
        printf("Sequential and parallel resuts match!\n");

}
//Serial code
void fill_rand(double *A)
{
    srand(time(NULL));
    for(int i=0;i<N;i++)
        A[i] = rand();
}

//Parallel code
void consumer(int tid)
{
    double item;
    while(j < N )
    {
        #pragma omp critical
        {
            j++;
            item = get();
            S+=item;
            //printf("%d ...Consuming %f\n",tid, item);
        }
    sleep(1);
    }
}

void producer(int tid)
{
    double item;
    while( i < N)
    {
        #pragma omp critical
        {
            item = A[i];
            put(item);
            i++;
            //printf("%d Producing %f ...\n",tid, item);
        }
        sleep(1);
    }
}

double get()
{
    double item;

    item = A[nextout];
    nextout = (nextout + 1) % N;
    count--;
    if (count == 0) // buffer is empty
        empty = 1;
    if (count == (N-1))
        // buffer was full
        full = 0;
    return item;
}

void put(double item)
{
    B[nextin] = item;
    nextin = (nextin + 1) % N;

    count++;
    if (count == N)
        full = 1;
    if (count == 1) // buffer was empty
        empty = 0;
}