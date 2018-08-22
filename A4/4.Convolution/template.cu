// 16CO145 Sumukha PK
// 16CO234 Prajval M

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

__global__ void Convolution(int*,const int*,int*,int,int);
bool CheckAnswer(int*,int*,int*,int,int);

#define TW 32
#define MaskWidth 5
#define MaskRadius 2

int main()
{	
	clock_t start_overhead,end_overhead;
	start_overhead=clock();
	
    /* No. of rows and column in the image*/
	int nImageROW=800;
	int nImageCOL=800;
	
	/* Dynamically allocate memory to the input image, output image and the mask. Each matrix is represented as a 1D array*/
	int* ImageIn=new int[nImageROW*nImageCOL];
	int* ImageOut=new int[nImageROW*nImageCOL];
	int* Mask=new int[MaskWidth*MaskWidth];
	
	srand(time(NULL));
	
	/* Populate the input image and the mask with random numbers*/
	for(int i=0;i<nImageROW;i++)
	for(int j=0;j<nImageCOL;j++)
	ImageIn[i*nImageCOL+j]=rand()%4;

	for(int i=0;i<MaskWidth;i++)
	for(int j=0;j<MaskWidth;j++)
	Mask[i*MaskWidth+j]=rand()%5;
  
	int ImageSize=nImageROW*nImageCOL*sizeof(int);
	int MaskSize=MaskWidth*MaskWidth*sizeof(int);
	
	
	int* ImageIn_d, *ImageOut_d,*Mask_d;
	
	/* Allocate memory on the device*/
	
	cudaMalloc((void**)&ImageIn_d,ImageSize);
	cudaMalloc((void**)&ImageOut_d,ImageSize);
	cudaMalloc((void**)&Mask_d,MaskSize);
	
	/* Copy data from host to device*/
	cudaMemcpy(ImageIn_d,ImageIn,ImageSize,cudaMemcpyHostToDevice);
	cudaMemcpy(Mask_d,Mask,MaskSize,cudaMemcpyHostToDevice);
	
	/* Invoke the convolution kernel*/
	dim3 dimBlock(TW,TW);
	dim3 dimGrid(ceil((float)nImageCOL/TW),ceil((float)nImageROW/TW));
	
	clock_t start,end,total;
	start=clock();
	
	Convolution<<<dimGrid,dimBlock>>>(ImageIn_d,Mask_d,ImageOut_d,nImageROW,nImageCOL);
	
	end=clock();
	total=(double)(end - start) / CLOCKS_PER_SEC;
	printf("Time taken on device: %ld\n",total);
	
	/* Copy the convoluted image to host*/
	cudaMemcpy(ImageOut,ImageOut_d,ImageSize,cudaMemcpyDeviceToHost);
	
	end_overhead=clock();
	
	if(CheckAnswer(ImageIn,Mask,ImageOut,nImageROW,nImageCOL))
	printf("Solution is right!\n");
    else
	printf("Solution is wrong!\n");
	
	printf("Time spent on overhead calculations: %lf\n",(double)(end_overhead - start_overhead) / CLOCKS_PER_SEC);
	
	cudaFree(ImageIn);
	cudaFree(ImageOut);
	cudaFree(Mask);	
}

__global__ void Convolution(int* ImageIn_d, const int* __restrict__ Mask_d,int* ImageOut_d, int nImageROW, int nImageCOL)
{
	/* Store the thread dimensions on registers*/
	int bx=blockIdx.x,by=blockIdx.y;
	int tx=threadIdx.x,ty=threadIdx.y;
	
	/* Declare a shared memory region for each block of threads*/
	__shared__ int SharedMemBlock[TW][TW];
	
	/* Find out the row and column for each thread*/
	int row=by*blockDim.y+ty;
	int col=bx*blockDim.x+tx;

	SharedMemBlock[ty][tx]=ImageIn_d[row*nImageCOL+col];
    __syncthreads();

	int Pvalue=0;
	for(int i=-MaskRadius;i<=MaskRadius;i++)
	for(int j=-MaskRadius;j<=MaskRadius;j++)
	{
		int GloabalMemRow=row+i;
		int GloabalMemCol=col+j;
		
		int SharedMemRow=ty+i;
		int SharedMemCol=tx+j;
		
		int MaskRow=MaskRadius+i;
		int MaskCol=MaskRadius+j;
		
		if(GloabalMemRow<by*TW || GloabalMemRow>=(by+1)*TW||GloabalMemCol<bx*TW || GloabalMemCol>=(bx+1)*TW)
		{
			if(GloabalMemRow>=0 && GloabalMemRow<nImageROW && GloabalMemCol>=0 && GloabalMemCol<nImageCOL)
			Pvalue+=ImageIn_d[GloabalMemRow*nImageCOL+GloabalMemCol]*Mask_d[MaskRow*MaskWidth+MaskCol];
			
		}
		else
		Pvalue+=SharedMemBlock[SharedMemRow][SharedMemCol]*Mask_d[MaskRow*MaskWidth+MaskCol];
	}
	ImageOut_d[row*nImageCOL+col]=Pvalue;
}

/* A function to verify correctness of solution*/
bool CheckAnswer(int* ImageIn, int* Mask, int* ImageOut,int nImageROW, int nImageCOL)
{
	clock_t start,end;
	start=clock();
	for(int row=0;row<nImageROW;row++)
	{
		for(int col=0;col<nImageCOL;col++)
		{
			int Pvalue=0;
			for(int MaskRow=-MaskRadius;MaskRow<=MaskRadius;MaskRow++)
			{
				for(int MaskCol=-MaskRadius;MaskCol<=MaskRadius;MaskCol++)
				{
					int imageRow=row+MaskRow;
					int imageCol=col+MaskCol;
					
					if(imageRow>=0 && imageRow<nImageROW && imageCol>=0 && imageCol<nImageCOL)
					Pvalue+=ImageIn[imageRow*nImageCOL+imageCol]*Mask[(MaskRow+MaskRadius)*MaskWidth+(MaskCol+MaskRadius)];
				}
			}
			if(Pvalue!=ImageOut[row*nImageCOL+col])
			return false;
		}
	}
	end=clock();
	printf("Time taken on host: %lf\n",(double)(end - start) / CLOCKS_PER_SEC);
	return true;

}