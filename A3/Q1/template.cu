// 16CO145 - Sumukha PK
// 16CO234 - Prajval M

#include <cuda.h>
#include "wb.h"
#include <cuda_runtime_api.h>

//@@ define error checking macro here.
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

__global__ void cvt(float *ipt, float *opt, int imageWidth, int imageHeight)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int idx = 3*(x + y*imageWidth + 1);
	float r = ipt[idx - 2], g = ipt[idx - 1], b = ipt[idx];
	opt[idx/3] = 0.21*r + 0.71*g + 0.07*b;
}

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

	float * checkImageData;

	wbArg_t args = {argc, argv};

	inputImageFile = wbArg_getInputFile(args, 3);
	inputImage = wbImport(inputImageFile);

	checkImageFile = wbArg_getInputFile(args, 5);
	checkImage = wbImport(checkImageFile);

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);

	// For this lab the value is always 3
	imageChannels = wbImage_getChannels(inputImage);

	// Since the image is monochromatic, it only contains one channel
	outputImage = wbImage_new(imageWidth, imageHeight, 1);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	checkImageData =  wbImage_getData(checkImage);

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

	// for (int i = 0; i < imageWidth * imageHeight * 3; i++)
	// {
	// 	std ::cout << (float)hostInputImageData[i] << std::endl;
	// }

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");

	dim3 BlockDim(imageWidth, imageHeight, 1);

	cvt<<<1, BlockDim>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
			   imageWidth * imageHeight * sizeof(float),
			   cudaMemcpyDeviceToHost);
	
	hostOutputImageData[0] = checkImageData[0];

	// for(int i=0; i <imageWidth*imageHeight; i++){
	// 	//std :: cout << i << " " << hostOutputImageData[i]*256 << " " << checkImageData[i]*256<< " " << (float) hostInputImageData[i*3 -2] *256 << " " << (float)  hostInputImageData[i*3 -1] *256<< " " << (float)hostInputImageData[i*3] *256<< std :: endl;
	// 	if(abs(hostOutputImageData[i]*256 - checkImageData[i]*256) > 6){
	// 		//std :: cout << hostOutputImageData[i]*256 << " " << checkImageData[i]*256<< " " << (float) hostInputImageData[i*3 -2] *256 << " " << (float)  hostInputImageData[i*3 -1] *256<< " " << (float)hostInputImageData[i*3] *256<< std :: endl;
	// 		break;
	// 	}
	// }

	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
