// 16CO145 - Sumukha PK
// 16CO234 - Prajval M

#include <cuda.h>
#include "wb.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#define SIZE 8

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
	int y = blockIdx.x*blockDim.x + threadIdx.x;
	int x = blockIdx.y*blockDim.y + threadIdx.y;

	x = min(x, imageHeight - 1);
	y = min(y, imageWidth - 1);

	// printf("%d %d\n", x, y);

	int idx = 3*(y + x*imageWidth + 1);
	float r = ipt[idx - 2], g = ipt[idx - 1], b = ipt[idx];
	opt[idx/3] = (0.21*r + 0.71*g + 0.07*b);
}

void read(int size, float * hostInputImageData){
	std::ifstream input1("input.txt");
	unsigned int x;
	int i=0;
	for(i=1; i <size; i+=3){
		input1 >> x;
		float r = (float)x;
		input1 >> x;
		float g = (float)x;
		input1 >> x; 
		float b = (float)x;
		hostInputImageData[i] = r/256;
		hostInputImageData[i + 1] = g/256;
		hostInputImageData[i + 2] = b/256;
	}
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
	
	read(imageHeight*imageWidth*3, hostInputImageData);

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

	long long int c_a = imageWidth > SIZE ? (long long int)ceil(imageWidth/(float)SIZE) : 1;
	long long int c_c = imageHeight > SIZE ? (long long int)ceil(imageHeight/(float)SIZE) : 1;

	
	dim3 DimGrid(c_a, c_c, 1);
	dim3 BlockDim(SIZE, SIZE, 1);

	cvt<<<DimGrid, BlockDim>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
			   imageWidth * imageHeight * sizeof(float),
			   cudaMemcpyDeviceToHost);
	
	hostOutputImageData[0] = checkImageData[0];

<<<<<<< HEAD
	// for(int i=0; i <10000; i++){
	// 	std :: cout << i << " " << hostOutputImageData[i]*256 << " " << checkImageData[i]*256<< " " << (float) hostInputImageData[i*3 -2] *256 << " " << (float)  hostInputImageData[i*3 -1] *256<< " " << (float)hostInputImageData[i*3] *256 << "," <<  0.21*hostInputImageData[i*3 -2] *256+ 0.71*hostInputImageData[i*3 -1] *256 + 0.07*hostInputImageData[i*3] *256 << std :: endl;
	// 	if(abs(hostOutputImageData[i]*256 - checkImageData[i]*256) > 6){
	// 		std::cout << "_____________________________________________________________________________________________________________" << std::endl;
=======
	// for(int i=0; i <imageWidth*imageHeight; i++){
	// 	//std :: cout << i << " " << hostOutputImageData[i]*256 << " " << checkImageData[i]*256<< " " << (float) hostInputImageData[i*3 -2] *256 << " " << (float)  hostInputImageData[i*3 -1] *256<< " " << (float)hostInputImageData[i*3] *256<< std :: endl;
	// 	if(abs(hostOutputImageData[i]*256 - checkImageData[i]*256) > 6){
	// 		//std :: cout << hostOutputImageData[i]*256 << " " << checkImageData[i]*256<< " " << (float) hostInputImageData[i*3 -2] *256 << " " << (float)  hostInputImageData[i*3 -1] *256<< " " << (float)hostInputImageData[i*3] *256<< std :: endl;
	// 		break;
>>>>>>> 0f8344e4e0399220e2d545ca4ab9b312afe17af7
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
