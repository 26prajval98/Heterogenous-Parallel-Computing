
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

static char *base_dir;
const size_t NUM_BINS = 32;
const unsigned int BIN_CAP = 127;

unsigned int getMax(unsigned int *A, int l)
{
	unsigned int max = A[0];

	for(int i=0; i < l; i++)
		if(A[i] > max)
			max = A[i];
	
	return max;
}

static void compute(unsigned int *bins, unsigned int *input, int num)
{
	for (int i = 0; i < num; ++i)
	{
		int idx = input[i];
		if (bins[idx] < BIN_CAP)
			(char *)++bins[idx];
	}
}

static unsigned int *generate_data(size_t n, unsigned int num_bins)
{
	unsigned int *data = (unsigned int *)malloc(sizeof(unsigned int) * n);
	for (unsigned int i = 0; i < n; i++)
	{
		data[i] = rand() % num_bins;
	}
	return data;
}

static void write_data(char *file_name, unsigned int *data, int num)
{
	FILE *handle = fopen(file_name, "w");
	fprintf(handle, "%d", num);
	for (int ii = 0; ii < num; ii++)
	{
		fprintf(handle, "\n%d", *data++);
	}
	fflush(handle);
	fclose(handle);
}

static void create_dataset(int datasetNum, size_t input_length,
						   size_t num_bins)
{

	const char *dir_name = base_dir;

	char *input_file_name = (char *)"input.raw";
	char *output_file_name = (char *)"output.raw";

	unsigned int *input_data = generate_data(input_length, num_bins);
	unsigned int *output_data =
		(unsigned int *)calloc(sizeof(unsigned int), num_bins);

	compute(output_data, input_data, input_length);

	write_data(input_file_name, input_data, input_length);
	unsigned int max = getMax(input_data, input_length);
	write_data(output_file_name, output_data, max+1);

	free(input_data);
	free(output_data);
}

int main()
{
	base_dir = (char *)"";

	create_dataset(0, 16, NUM_BINS);
	create_dataset(1, 1024, NUM_BINS);
	create_dataset(2, 513, NUM_BINS);
	create_dataset(3, 511, NUM_BINS);
	create_dataset(4, 1, NUM_BINS);
	create_dataset(5, 500000, NUM_BINS);
	return 0;
}
