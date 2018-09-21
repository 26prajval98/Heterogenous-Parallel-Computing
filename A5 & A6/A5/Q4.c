#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>

typedef double TYPE;
#define MAX_DIM 2000*2000
#define MAX_VAL 10
#define MIN_VAL 1

// Method signatures
TYPE** randomSquareMatrix(int dimension);
TYPE** zeroSquareMatrix(int dimension);
void displaySquareMatrix(TYPE** matrix, int dimension);
void convert(TYPE** matrixA, TYPE** matrixB, int dimension);

// Matrix multiplication methods
double sequentialMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);
double parallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);
double optimizedParallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);

// Test cases
void sequentialMultiplyTest(int dimension, int iterations);
void parallelMultiplyTest(int dimension, int iterations);
void optimizedParallelMultiplyTest(int dimension, int iterations);

// 1 Dimensional matrix on stack
TYPE flatA[MAX_DIM];
TYPE flatB[MAX_DIM];

// Verify multiplication
void verifyMultiplication(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension);

int main(int argc, char* argv[]) {
	int iterations = strtol(argv[1], NULL, 10);

	// Generate Necessary files
	// Create Sequential Multiply test log
	FILE* fp;
	fp = fopen("SequentialMultiplyTest.txt", "w+");
	fclose(fp);

	// Create Parallel Multiply test log
	fp = fopen("ParallelMultiplyTest.txt", "w+");
	fclose(fp);

	// Create Optimized Parallel Multiply test log
	fp = fopen("OptimizedParallelMultiplyTest.txt", "w+");
	fclose(fp);

	for (int dimension = 200; dimension <= 2000; dimension += 200) {
		optimizedParallelMultiplyTest(dimension, iterations);
	}

	for (int dimension = 200; dimension <= 2000; dimension += 200) {
		parallelMultiplyTest(dimension, iterations);
	}

	for (int dimension = 200; dimension <= 2000; dimension += 200) {
		sequentialMultiplyTest(dimension, iterations);
	}

	return 0;
}

TYPE** randomSquareMatrix(int dimension) {
	/*
	Generate 2 dimensional random TYPE matrix.
	*/

	TYPE** matrix = (TYPE **)malloc(dimension * sizeof(TYPE*));

	for (int i = 0; i<dimension; i++) {
		matrix[i] = (TYPE *)malloc(dimension * sizeof(TYPE));
	}

	//Random seed
	srand(time(0) + clock() + rand());

#pragma omp parallel for
	for (int i = 0; i<dimension; i++) {
		for (int j = 0; j<dimension; j++) {
			matrix[i][j] = rand() % MAX_VAL + MIN_VAL;
		}
	}

	return matrix;
}

TYPE** zeroSquareMatrix(int dimension) {
	/*
	Generate 2 dimensional zero TYPE matrix.
	*/

	TYPE** matrix = (TYPE **)malloc(dimension * sizeof(TYPE*));

	for (int i = 0; i<dimension; i++) {
		matrix[i] = (TYPE *)malloc(dimension * sizeof(TYPE));
	}

	//Random seed
	srand(time(0) + clock() + rand());
	for (int i = 0; i<dimension; i++) {
		for (int j = 0; j<dimension; j++) {
			matrix[i][j] = 0;
		}
	}

	return matrix;
}

void displaySquareMatrix(TYPE** matrix, int dimension) {
	for (int i = 0; i<dimension; i++) {
		for (int j = 0; j<dimension; j++) {
			printf("%f\t", matrix[i][j]);
		}
		printf("\n");
	}
}

double sequentialMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension) {
	/*
	Sequentiall multiply given input matrices and return resultant matrix
	*/

	double t = omp_get_wtime();

	/* Head */
	for (int i = 0; i<dimension; i++) {
		for (int j = 0; j<dimension; j++) {
			for (int k = 0; k<dimension; k++) {
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}
	/* Tail */

	return t - omp_get_wtime();
}

double parallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension) {
	/*
	Parallel multiply given input matrices and return resultant matrix
	*/
	
	double t = omp_get_wtime();

	/* Head */
#pragma omp parallel for
	for (int i = 0; i<dimension; i++) {
		for (int j = 0; j<dimension; j++) {
			for (int k = 0; k<dimension; k++) {
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}
	/* Tail */

	return t - omp_get_wtime();
}

double optimizedParallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension) {
	/*
	Parallel multiply given input matrices using optimal methods and return resultant matrix
	*/

	int i, j, k, iOff, jOff;
	TYPE tot;
	
	double t = omp_get_wtime();

	/* Head */
	convert(matrixA, matrixB, dimension);
#pragma omp parallel shared(matrixC) private(i, j, k, iOff, jOff, tot) num_threads(40)
	{
#pragma omp for schedule(static)
		for (i = 0; i<dimension; i++) {
			iOff = i * dimension;
			for (j = 0; j<dimension; j++) {
				jOff = j * dimension;
				tot = 0;
				for (k = 0; k<dimension; k++) {
					tot += flatA[iOff + k] * flatB[jOff + k];
				}
				matrixC[i][j] = tot;
			}
		}
	}
	/* Tail */

	return t - omp_get_wtime();
}

void convert(TYPE** matrixA, TYPE** matrixB, int dimension) {
#pragma omp parallel for
	for (int i = 0; i<dimension; i++) {
		for (int j = 0; j<dimension; j++) {
			flatA[i * dimension + j] = matrixA[i][j];
			flatB[j * dimension + i] = matrixB[i][j];
		}
	}
}
