#include "wb.h"


#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define SIZE 16
#define w (SIZE + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

__global__ void convolution(float* deviceInputImageData,float* deviceMaskData,float* deviceOutputImageData,int imageChannels,int imageWidth,int imageHeight)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y; 
  int k ;
  for(k=0;k<imageChannels;k++)
  {
    float accum  = 0;
    for(int y = -Mask_radius; y< Mask_radius ; y++)
    {
      for(int x = -Mask_radius; x< Mask_radius ; x++)
      {
        int xOff = j+x;
        int yOff = i+y;
        if(xOff >=0 && xOff< imageWidth && yOff >=0 && yOff <imageHeight)
        {
          int temp = deviceInputImageData[(yOff*imageWidth + xOff)*imageChannels +  k];
          int temp1 = deviceMaskData[(y+Mask_radius)*Mask_width+x+Mask_radius];
          accum += temp*temp1;
        }
      }
    }
    deviceOutputImageData[(i*imageWidth+j)*imageChannels + k] = clamp(accum);
  }
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  
  cudaMalloc((void **) deviceInputImageData,imageWidth * imageHeight * imageChannels * sizeof(float));                    //1.Allocate device memory
  cudaMalloc((void **) deviceMaskData,sizeof(hostMaskData));
  cudaMalloc((void **) deviceOutputImageData,sizeof(hostOutputImageData));
  // cudaMalloc((void **) imageChannels,sizeof(int));
  // cudaMalloc((void **) imageHeight, sizeof(int));
  // cudaMalloc((void **) imageWidth,sizeof(int));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");

  cudaMemcpy(deviceInputImageData,hostInputImageData,imageWidth * imageHeight * imageChannels * sizeof(float),cudaMemcpyHostToDevice);   // 2.Copt host to device
  cudaMemcpy(deviceMaskData,hostMaskData,sizeof(hostMaskData),cudaMemcpyHostToDevice);

  // cudaMemcpy(,cudaMemcpyHostToDevice);
  // cudaMemcpy(,cudaMemcpyHostToDevice);
  // cudaMemcpy(,cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  
  dim3 dimBlock(SIZE);                                                                  //3.inititalise thread block and grid dimensions
  dim3 dimGrid(ceil(imageHeight/SIZE),ceil(imageWidth/SIZE),1);

  convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,              //4.Invoke CUDA kernel
                                     deviceOutputImageData, imageChannels,
                                     imageWidth, imageHeight);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
 
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,imageWidth * imageHeight * imageChannels * sizeof(float),cudaMemcpyDeviceToHost); //5. Copy results
        
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ Insert code here

  free(hostMaskData);                    //6. Free device memory
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
