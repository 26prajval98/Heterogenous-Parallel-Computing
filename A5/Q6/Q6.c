//16CO145 Sumukha PK
//16CO234 Prajval.M
#include "omp.h"
#include <stdio.h>
#include <time.h>

#define MAX_THREADS 4

static long num_steps = 100000;

double step;
double pi_s();

void parallel()
{
	int  j;
	double sum = 0.0;
	double start_time, run_time;

	step = 1.0 / (double)num_steps;

	for (j = 1; j <= MAX_THREADS; j++)
	{
		omp_set_num_threads(j);
		sum = 0.0;
		double x = 0;
		start_time = omp_get_wtime();
#pragma omp parallel reduction(+: sum)
		{
#pragma omp for
			for (int i = 0; i < num_steps; i++) {
				x = (i - 0.5) * step;
				sum = sum + 4.0 / (1.0 + x * x);
			}
		}
		run_time = omp_get_wtime() - start_time;
		printf("\n pi is %f in %f seconds %d threads \n ", sum/num_steps, run_time, j);
	}
}

void main()
{
	//Calculate serial code's time and output
	double start = omp_get_wtime();
	double pi_serial = pi_s();
	double end = omp_get_wtime();
	printf("Time taken by serial code is %f\n", end - start);

	printf("Serial output value: %f\n", pi_serial);

	parallel();
}

//Serial code
double pi_s()
{
	int i;
	double x, pi, sum = 0.0;
	step = 1.0 / (double)num_steps;
	for (i = 0; i < num_steps; i++)
	{
		x = (i + 0.5) * step;
		sum = sum + 4.0 / (1.0 + x * x);
	}
	pi = step * sum;
	return pi;
}