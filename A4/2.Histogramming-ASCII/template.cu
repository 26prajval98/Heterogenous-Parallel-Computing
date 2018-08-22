#include "wb.h"

#define NUM_BINS 128
#define SIZE 128


#define CUDA_CHECK(ans)                       \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}


inline void gpuAssert(cudaError_t code, const char *file, int line,
					  bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
				file, line);
		if (abort)
			exit(code);
	}
}


__global__ void hist(int * d_ip, int * d_b, int l) 
{	

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int x = threadIdx.x;
	__shared__ int buff[NUM_BINS];

	for(int i=0; i<NUM_BINS;i++) 
		buff[i] = 0;
	__syncthreads();

	if(idx < l)
		atomicAdd(&buff[d_ip[idx]], 1);
	__syncthreads();

	atomicAdd(&d_b[x], buff[x]);
}


int main(int argc, char *argv[])
{
	wbArg_t args;
	int inputLength;
	int *hostInput;
	int *hostBins;
	int *deviceInput;
	int *deviceBins;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	
	hostInput = (int *)wbImport(wbArg_getInputFile(args, 3), &inputLength, 0);
	hostBins = (int *)malloc(NUM_BINS * sizeof(int));

	for(int i=0; i < NUM_BINS; i++)
		hostBins[i] = 0;

	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);

	wbTime_start(GPU, "Allocating GPU memory.");

	//@@ Allocate GPU memory here
	cudaMalloc((void **)&deviceInput, inputLength * sizeof(int));
	cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(int));

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");

	//@@ Copy memory to the GPU here
	cudaMemcpy(deviceInput, hostInput, inputLength*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBins, hostBins, NUM_BINS*sizeof(int), cudaMemcpyHostToDevice);

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// Launch kernel
	// ----------------------------------------------------------
	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");
	
	//@@ Perform kernel computation here

	dim3 DimBlock(SIZE);
	dim3 DimGrid(max((int)ceil((float)inputLength/SIZE), 1));

	hist<<<DimGrid,DimBlock>>>(deviceInput, deviceBins, inputLength);

	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");

	//@@ Copy the GPU memory back to the CPU here
	cudaMemcpy(hostInput, deviceInput, inputLength*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostBins, deviceBins, NUM_BINS*sizeof(int), cudaMemcpyDeviceToHost);

	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	
	//@@ Free the GPU memory here
	cudaFree(deviceBins);
	cudaFree(deviceInput);

	wbTime_stop(GPU, "Freeing GPU Memory");

	// Verify correctness
	// -----------------------------------------------------

	wbSolution(args, hostBins, NUM_BINS);

	free(hostBins);
	free(hostInput);
	return 0;
}
