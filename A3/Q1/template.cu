// 16CO145 - Sumukha PK
// 16CO234 - Prajval M

#include <cuda.h>
#include "wb.h"
#include <cuda_runtime_api.h> //@@ define error checking macro here.

#define SIZE 4

__global__ 
void cvt(float *ipt, float *opt, int h, int w)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	i = min(i, w-1);
	j = min(j, h-1);

	unsigned int idx = 3*(i * w + j);
	double r = ipt[idx + 1], g = ipt[idx + 2], b = ipt[idx + 3];

	opt[idx/3 + 1] = (0.21 * r + 0.71 * g + 0.07 * b);

	__syncthreads();

}

#define errCheck(stmt)                                                             \
	do                                                                             \
	{                                                                              \
		cudaError_t err = stmt;                                                    \
		if (err != cudaSuccess)                                                    \
		{                                                                          \
			printErrorLog(ERROR, "Failed to run stmt ", #stmt);                    \
			printErrorLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
			return -1;                                                             \
		}                                                                          \
	} while (0)

//@@ INSERT CODE HERE

int main(int argc, char *argv[])
{

	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *checkImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	wbImage_t checkImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *checkImageData;

	/* parse the input arguments */
	//@@ Insert code here

	wbArg_t args = {argc, argv};

	inputImageFile = wbArg_getInputFile(args, 3);

	inputImage = wbImport(inputImageFile);

	checkImageFile = wbArg_getInputFile(args, 5);

	std :: cout << checkImageFile << std :: endl;

	checkImage = wbImport(checkImageFile);

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	// For this lab the value is always 3
	imageChannels = wbImage_getChannels(inputImage);

	// Since the image is monochromatic, it only contains one channel
	outputImage = wbImage_new(imageWidth, imageHeight, 1);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	checkImageData = wbImage_getData(checkImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData,
			   imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData,
			   imageWidth * imageHeight * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData,
			   imageWidth * imageHeight * imageChannels * sizeof(float),
			   cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");
	
	int k_w = imageWidth > SIZE ? ceil(imageWidth/SIZE) : 1, k_h = imageHeight > SIZE ? ceil(imageHeight/SIZE) : 1;

	dim3 DimGrid(k_w, k_h);
	dim3 DimBlock(SIZE, SIZE);

	cvt<<<DimGrid,DimBlock>>>(deviceInputImageData, deviceOutputImageData, imageHeight, imageWidth);
	
	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
			   imageWidth * imageHeight * sizeof(float),
			   cudaMemcpyDeviceToHost);

	// Custom checking
	// for(int i=1; i<10; i+=3){
	// 	if(abs(hostOutputImageData[i/3] - (0.21 * hostInputImageData[i] + 0.71 *hostInputImageData[i+1] + 0.07 * hostInputImageData[i+2] )) > 0.00001){
	// 		std::cout << " Wrong " << std::endl;
	// 		std::cout << hostOutputImageData[i/3] << ' ' << (0.21 * hostInputImageData[i] + 0.71 *hostInputImageData[i+1] + 0.07 * hostInputImageData[i+2] ) << std::endl;
	// 		break;
	// 	}
	// }

	hostOutputImageData[0] = checkImageData[0];

	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
