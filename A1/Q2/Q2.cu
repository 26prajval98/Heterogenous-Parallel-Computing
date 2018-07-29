#include<stdio.h>
#include<cuda.h>
#include<math.h>
__global__ void Matrix_Add(float *d_m, float *d_n, float *d_s, long long int a, long long int b)
{
	long long int i = blockIdx.y*blockDim.y + threadIdx.y;
	long long int j = blockIdx.x*blockDim.x + threadIdx.x;
	
	if((i<a) && (j<b))
	{
		*(d_s + i*b + j) = *(d_m + i*b +j) + *(d_n + i*b + j);
	}
	
}
int main()
{
	long long int a,b,i,j;
	printf("Enter the dimensions of the 2 matrices: ");
	scanf("%lld %lld",&a,&b);                             //Matrices are aXb
	float *h_m = (float *)malloc(a*b*sizeof(float));
	float *h_n = (float *)malloc(b*a*sizeof(float));
	float *h_s = (float *)malloc(a*b*sizeof(float));
	srand((unsigned int)time(NULL));                      //Seeding the random function
	for(i=0;i<a;i++)
	{
		for(j=0;j<b;j++)
			*(h_m + i*b +j) = ((float)rand());
	}
	for(i=0;i<a;i++)
	{
		for(j=0;j<b;j++)
			*(h_n + i*b +j) = ((float)rand());
	}
	
	dim3 DimGrid(ceil(a/16),ceil(b/16),1);
	dim3 DimBlock(16,16,1);
	
	float *d_m, *d_n, *d_s;
	cudaMalloc((float**)&d_m,a*b*sizeof(float));
	cudaMalloc((float**)&d_n,b*a*sizeof(float));
	cudaMalloc((float**)&d_s,a*b*sizeof(float));

	cudaMemcpy(d_m,h_m,a*b*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_n,h_n,b*a*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_s,h_s,a*b*sizeof(float),cudaMemcpyHostToDevice);
	 
	Matrix_Add<<<DimGrid,DimBlock>>>(d_m,d_n,d_s,a,b);
	
	cudaMemcpy(h_m,d_m,a*b*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_n,d_n,b*a*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s,d_s,a*b*sizeof(float),cudaMemcpyDeviceToHost);
	
	for(i=0;i<a;i++)
        {
                for(j=0;j<b;j++)
                        printf("%f ",*(h_s + i*b + j));
                printf("\n");
        }printf("\n");
        for(i=0;i<a;i++)
        {
                for(j=0;j<b;j++)
                        printf("%f ",*(h_m + i*b + j));
                printf("\n");
        }printf("\n");
	for(i=0;i<a;i++)
	{
		for(j=0;j<b;j++)
			printf("%f ",*(h_n + i*b + j));
		printf("\n");
	}

	cudaFree(d_m); cudaFree(d_n); cudaFree(d_s);
	
	free(h_m); free(h_n); free(h_s);
	printf("1. How many floating operations are being performed in the matrix addition kernel? \n Ans: The number of floating point operations is %lld\n",a*b);
	printf("2. How many global memory reads are being performed by your kernel?\n Ans: The number of global memory reads is 2 per thread operation = %lld\n",2*a*b);
	printf("3. How many global memory writes are being performed by your kernel?\n Ans: The number of global memory writes is oen per thread operation = %lld\n",a*b);
	return 0;
}
