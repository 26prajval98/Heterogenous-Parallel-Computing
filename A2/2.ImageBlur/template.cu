#include <cuda.h>
#include "wb.h"
#include <cuda_runtime_api.h>

#define BLUR_SIZE 3
#define THREADS 16

//@@ INSERT CODE HERE

__global__ void Gaussian(float *input, float *output, int width, int height)
{

	__shared__ float temp[THREADS + 2][THREADS + 2];
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float conv_res = 0;

	if (tx == 0 || ty == 0 || tx == THREADS - 1 || ty == THREADS - 1)
	{

		temp[ty][tx] = (y > 0 && x > 0) ? input[(y - 1) * width + x - 1] : 0;
		temp[ty][tx + 1] = (y > 0) ? input[(y - 1) * width + x] : 0;
		temp[ty][tx + 2] = (y > 0 && x < width) ? input[(y - 1) * width + x + 1] : 0;
		temp[ty + 1][tx] = (x > 0) ? input[y * width + x - 1] : 0;
		temp[ty + 1][tx + 1] = input[y * width + x];
		temp[ty + 1][tx + 2] = (x < width) ? input[y * width + x + 1] : 0;
		temp[ty + 2][tx] = (y < height && x > 0) ? input[(y + 1) * width + x - 1] : 0;
		temp[ty + 2][tx + 1] = (y < height) ? input[(y + 1) * width + x] : 0;
		temp[ty + 2][tx + 2] = (y < height && x < width) ? input[(y + 1) * width + x + 1] : 0;
	}
	else
	{
		temp[ty + 1][tx + 1] = input[(y * width) + x];
	}

	__syncthreads();

	for (int i = 0; i < BLUR_SIZE; i++)
	{
		for (int j = 0; j < BLUR_SIZE; j++)
		{
			conv_res += temp[ty + i][tx + j];
		}
	}
	conv_res = (float)conv_res / (float)(BLUR_SIZE * BLUR_SIZE);

	output[y * width + x] = conv_res;
}

int main(int argc, char *argv[])
{

	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;

	/* parse the input arguments */
	//@@ Insert code here

	if (argc != 9)
	{
		printf("use:./ImageBlur_Template ­e <expected.ppm> ­i <input.ppm> ­o <output.ppm> ­t image");
		exit(0);
	}

	wbArg_t args = {argc, argv};

	inputImageFile = wbArg_getInputFile(args, 3);

	inputImage = wbImport(inputImageFile);

	// The input image is in grayscale, so the number of channels
	// is 1
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);

	// Since the image is monochromatic, it only contains only one channel
	outputImage = wbImage_new(imageWidth, imageHeight, 1);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData,
			   imageWidth * imageHeight * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData,
			   imageWidth * imageHeight * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData,
			   imageWidth * imageHeight * sizeof(float),
			   cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");

	dim3 blockDim(16, 16, 1);
	dim3 gridDim((int)ceil((float)(imageWidth) / blockDim.x), (int)ceil((float)(imageHeight) / blockDim.y), 1);

	Gaussian<<<gridDim, blockDim>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");

	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
			   imageWidth * imageHeight * sizeof(float),
			   cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbImage_save(outputImage, "convoluted.ppm");

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}