#include<stdio.h>
#include<cuda.h>
#include<math.h>
__global__ void Matrix_Add(float *d_m, float *d_n, float *d_s, long long int a, long long int b)
{
	long long int i = blockIdx.y*blockDim.y + threadIdx.y;
	long long int j = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i<a)
	{
		if(j<b)
			*(d_s + i*a + j) = *(d_m + i*a +j) + *(d_n + i*a + j);
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
			*(h_m + i*a +j) = ((float)rand());
	}
	for(i=0;i<a;i++)
	{
		for(j=0;j<b;j++)
			*(h_n + i*b +j) = ((float)rand());
	}
	
	dim3 DimBlock(16,16,1);
	dim3 DimGrid(ceil(n/16),ceil(n/16),1);

	cudaMalloc((void**)&d_m,a*b*sizeof(float));
	cudaMalloc((void**)&d_n,b*a*sizeof(float));
	cudaMalloc((void**)&d_s,a*b*sizeof(float));

	cudaMemcpy(d_m,h_m,a*b*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_n,h_n,b*a*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_s,h_s,a*b*sizeof(float),cudaMemcpyHostToDevice);
	 
	Matrix_Add<<<DimGrid,DimBlock>>(*d_m,*d_n,*d_s,a,b);
	
	cudaMemcpy(h_m,d_m,a*b*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_n,d_n,b*a*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s,d_s,a*b*sizeof(float),cudaMemcpyDeviceToHost);
	
	cudaFree(d_m); cudaFree(d_n); cudaFree(d_s); 
	
	for(i=0;i<a;i++)
	{
		for(j=0;j<b;j++)
			printf("%f ",*(h_s + i*a + j));
		printf("\n");
	}
	
	free(h_m); free(h_n); free(h_s);
	return 0;
}
