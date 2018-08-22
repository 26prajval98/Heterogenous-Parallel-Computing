#include "wb.h"

#define THREAD_PER_DIM 3
#define wbCheck(stmt)                                                      \
	do                                                                     \
	{                                                                      \
		cudaError_t err = stmt;                                            \
		if (err != cudaSuccess)                                            \
		{                                                                  \
			wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
			wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
			return -1;                                                     \
		}                                                                  \
	} while (0)

__host__ __device__ int val(float *arr, int i, int j, int k, int width, int depth)
{
	return arr[(i * width + j) * depth + k];
}

__global__ void stencil(float *output, float *input, int width, int height, int depth)
{
	__shared__ float cache[(THREAD_PER_DIM + 2) * (THREAD_PER_DIM + 2) * (THREAD_PER_DIM + 2)];
	int gx = threadIdx.x + blockDim.x * blockIdx.x;
	int gy = threadIdx.y + blockDim.y * blockIdx.y;
	int gz = threadIdx.z + blockDim.z * blockIdx.z;

	int lx = threadIdx.x + 1;
	int ly = threadIdx.y + 1;
	int lz = threadIdx.z + 1;

	cache[val(lx, ly, lz, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)] = input[val(gx, gy, gz, width, depth)];

	if (threadIdx.x == 0)
	{
		if (gx > 0)
		{
			cache[val(0, ly, lz, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)] =
				input[val(gx - 1, gy, gz, width, depth)];
		}
		if (gx + THREAD_PER_DIM < height)
		{
			cache[val(THREAD_PER_DIM + 1, ly, lz, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)] =
				input[val(gx + THREAD_PER_DIM, gy, gz, width, depth)];
		}
	}
	if (threadIdx.y == 0)
	{
		if (gy > 0)
		{
			cache[val(lx, 0, lz, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)] =
				input[val(gx, gy - 1, gz, width, depth)];
		}
		if (gy + THREAD_PER_DIM < width)
		{
			cache[val(lx, THREAD_PER_DIM + 1, lz, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)] =
				input[val(gx, gy + THREAD_PER_DIM, gz, width, depth)];
		}
	}
	if (threadIdx.z == 0)
	{
		if (gz > 0)
		{
			cache[val(lx, ly, 0, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)] =
				input[val(gx, gy, gz - 1, width, depth)];
		}
		if (gz + THREAD_PER_DIM < depth)
		{
			cache[val(lx, ly, THREAD_PER_DIM + 1, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)] =
				input[val(gx, gy, gz + THREAD_PER_DIM, width, depth)];
		}
	}
	__syncthreads();

	if (gx > 0 && gy > 0 && gz > 0 && gx < height - 1 && gz < depth - 1 && gy < width - 1)
	{
		unsigned char cur = cache[val(lx, ly, lz, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)];
		unsigned char down = cache[val(lx, ly + 1, lz, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)];
		unsigned char up = cache[val(lx, ly - 1, lz, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)];
		unsigned char left = cache[val(lx, ly, lz - 1, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)];
		unsigned char right = cache[val(lx, ly, lz + 1, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)];
		unsigned char front = cache[val(lx - 1, ly, lz, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)];
		unsigned char back = cache[val(lx + 1, ly, lz, THREAD_PER_DIM + 2, THREAD_PER_DIM + 2)];

		int res = right + left + up + down + front + back - 6 * cur;
		res = Clamp(res, 0, 255);

		output[val(gx, gy, gz, width, depth)] = res;
	}
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData, int width, int height, int depth)
{
	dim3 blockDim(THREAD_PER_DIM,THREAD_PER_DIM,THREAD_PER_DIM);
	dim3 gridDim((height-1)/THREAD_PER_DIM+1,(width-1)/THREAD_PER_DIM+1,(depth-1)/THREAD_PER_DIM+1);

	stencil<<<gridDim, blockDim>>>(deviceOutputData, deviceInputData, width, height, depth);
}

int main(int argc, char *argv[])
{
	wbArg_t arg;
	int width;
	int height;
	int depth;
	char *inputFile;
	wbImage_t input;
	wbImage_t output;
	float *hostInputData;
	float *hostOutputData;
	float *deviceInputData;
	float *deviceOutputData;

	arg = wbArg_read(argc, argv);

	inputFile = wbArg_getInputFile(arg, 3);

	input = wbImport(inputFile);

	width = wbImage_getWidth(input);
	height = wbImage_getHeight(input);
	depth = wbImage_getChannels(input);

	output = wbImage_new(width, height, depth);

	hostInputData = wbImage_getData(input);
	hostOutputData = wbImage_getData(output);

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputData,
			   width * height * depth * sizeof(float));
	cudaMalloc((void **)&deviceOutputData,
			   width * height * depth * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputData, hostInputData,
			   width * height * depth * sizeof(float),
			   cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputData, deviceOutputData,
			   width * height * depth * sizeof(float),
			   cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbSolution(arg, output);

	cudaFree(deviceInputData);
	cudaFree(deviceOutputData);

	wbImage_delete(output);
	wbImage_delete(input);

	return 0;
}