// 16CO145 Sumukha PK
// 16CO234 Prajval M

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

__global__ void Stencil(int*,int*,int,int,int);
bool CheckAnswer(int*,int*,int,int,int);

#define BW 8
#define value(array, i, j, k) array[((i)*width + (j)) * depth + (k)]
#define OutputArray(i, j, k) value(OutputArray, i, j, k)
#define InputArray(i, j, k) value(InputArray, i, j, k)


int main()
{	
	clock_t start_overhead,end_overhead;
	start_overhead=clock();
	
    /* No. of rows and column in the image*/
	int height=64;
	int width=64;
	int depth=64;
	
	/* Dynamically allocate memory to the input image, output image and the mask. Each matrix is represented as a 1D array*/
	int* InputArray=new int[height*width*depth];
	int* OutputArray=new int[height*width*depth];
	
	srand(time(NULL));
	
	/* Populate the input image and the mask with random numbers*/
	for(int i=0;i<height;i++)
	for(int j=0;j<width;j++)
	for(int k=0;k<depth;k++)
	InputArray(i,j,k)=rand()%4;

  
	int InputArraySize=height*width*depth*sizeof(int);
	
	
	int* InputArray_d, *OutputArray_d;
	
	/* Allocate memory on the device*/
	
	cudaMalloc((void**)&InputArray_d,InputArraySize);
	cudaMalloc((void**)&OutputArray_d,InputArraySize);
	
	/* Copy data from host to device*/
	cudaMemcpy(InputArray_d,InputArray,InputArraySize,cudaMemcpyHostToDevice);

	/* Invoke the Stencil kernel*/
	dim3 dimBlock(BW,BW,BW);
	dim3 dimGrid(ceil((float)width/BW),ceil((float)height/BW),ceil((float)depth/BW));
	
	clock_t start,end,total;
	start=clock();
	
	Stencil<<<dimGrid,dimBlock>>>(InputArray_d,OutputArray_d,height,width,depth);
	
	end=clock();
	total=(double)(end - start) / CLOCKS_PER_SEC;
	printf("Time taken on device: %ld\n",total);
	
	/* Copy the convoluted image to host*/
	cudaMemcpy(OutputArray,OutputArray_d,InputArraySize,cudaMemcpyDeviceToHost);
	
	end_overhead=clock();
	
	if(CheckAnswer(InputArray,OutputArray,height,width,depth))
	printf("Solution is right!\n");
    else
	printf("Solution is wrong!\n");
	
	printf("Time spent on overhead calculations: %lf\n",(double)(end_overhead - start_overhead) / CLOCKS_PER_SEC);
	
	cudaFree(InputArray);
	cudaFree(OutputArray);
}

__global__ void Stencil(int* InputArray_d,int* OutputArray_d, int height, int width,int depth)
{

#define InputArray_d(i, j, k) value(InputArray_d, i, j, k)
#define OutputArray_d(i, j, k) value(OutputArray_d, i, j, k)

  	/* Store the thread dimensions on registers*/
	int bx=blockIdx.x,by=blockIdx.y,bz=blockIdx.z;
	int tx=threadIdx.x,ty=threadIdx.y,tz=threadIdx.z;
	
	/* Declare a shared memory region for each block of threads*/
	__shared__ int SharedMemBlock[BW][BW][BW];
	
	/* Find out the row and column for each thread*/
	int XIdx=bx*blockDim.x+tx;
	int YIdx=by*blockDim.y+ty;
	int ZIdx=bz*blockDim.z+tz;
	
	SharedMemBlock[tx][ty][tz]=InputArray_d(XIdx,YIdx,ZIdx);
	__syncthreads();
	
	if(XIdx>0 && XIdx<width-1 && YIdx>0 && YIdx<height-1 && ZIdx>0 && ZIdx<depth-1)
	{		

		int Pvalue=0;
		
		if(ZIdx+1>=(bz+1)*BW)
		Pvalue+=InputArray_d(XIdx,YIdx,ZIdx+1);
		else
		Pvalue+=SharedMemBlock[tx][ty][tz+1];

		if(ZIdx-1<bz*BW)
		Pvalue+=InputArray_d(XIdx,YIdx,ZIdx-1);
		else
		Pvalue+=SharedMemBlock[tx][ty][tz-1];

		if(YIdx+1>=(by+1)*BW)
		Pvalue+=InputArray_d(XIdx,YIdx+1,ZIdx);
		else
		Pvalue+=SharedMemBlock[tx][ty+1][tz];

		if(YIdx-1<by*BW)
		Pvalue+=InputArray_d(XIdx,YIdx-1,ZIdx);
		else
		Pvalue+=SharedMemBlock[tx][ty-1][tz];

		if(XIdx+1>=(bx+1)*BW)
		Pvalue+=InputArray_d(XIdx+1,YIdx,ZIdx);
		else
		Pvalue+=SharedMemBlock[tx+1][ty][tz];

		if(XIdx-1<(bx)*BW)
		Pvalue+=InputArray_d(XIdx-1,YIdx,ZIdx);
		else
		Pvalue+=SharedMemBlock[tx-1][ty][tz];

		Pvalue-=6*InputArray_d(XIdx,YIdx,ZIdx);
		OutputArray_d(XIdx,YIdx,ZIdx)=Pvalue;
	}

#undef OutputArray_d
#undef InputArray_d
}

/* A function to verify correctness of solution*/
bool CheckAnswer(int* InputArray, int* OutputArray,int height, int width,int depth)
{
	clock_t start,end;
	start=clock();
	for(int i=1;i<height-1;i++)
	{
		for(int j=1;j<width-1;j++)
		{
			for(int k=1;k<depth-1;k++)
			{
				int Pvalue=0;
				Pvalue=InputArray(i,j,k+1)+InputArray(i,j,k-1)+
				       InputArray(i,j+1,k)+InputArray(i,j-1,k)+
					   InputArray(i+1,j,k)+InputArray(i-1,j,k)-
					   6*InputArray(i,j,k);				  
				       
				if(Pvalue!=OutputArray(i,j,k))
				return false;
			}

		}
	}
	end=clock();
	printf("Time taken on host: %lf\n",(double)(end - start) / CLOCKS_PER_SEC);
	return true;

}