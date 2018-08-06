#include <cuda.h>
#include "wb.h"

#define BLUR_SIZE 5

//@@ INSERT CODE HERE
__global__ void Gaussian(float *input, float *output, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
}

int main(int argc, char *argv[]) {

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
	wbArg_t args = wbArg_read(argc, argv);
	inputImageFile = wbArg_getInputFile(args, 0);
	printf("%s\n",inputImageFile);

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
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData,	imageWidth * imageHeight * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");
	dim3 blockDim(5, 5, 1);
	dim3 gridDim((int)ceil((float)(imageWidth) / blockDim.x), (int)ceil((float)(imageHeight) / blockDim.y), 1);

	Gaussian << <gridDim, blockDim >> >(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}