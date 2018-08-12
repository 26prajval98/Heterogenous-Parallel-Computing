// 16CO145 Sumukha PK
// 16CO234 Prajval M

#include<stdio.h>
#include<cuda.h>
#include<math.h>
__global__ void Matrix_Mul(long long int *d_m, long long int *d_n, long long int *d_p, long long int a, long long int b,long long int c)
{
	long long int i = blockIdx.y*blockDim.y + threadIdx.y;
	long long int j = blockIdx.x*blockDim.x + threadIdx.x;
	
	if((i<a) && (j<b))
	{
        for(long long int k=0;k<b;k++)
            *(d_p + b*i + j ) += (*(d_m + c*i + k))*(*(d_n +b*k +j )); 
	}
}
int main()
{
	long long int a,b,c,i,j;
	printf("Enter the dimensions of the matrices: ");
	scanf("%lld %lld %lld",&a,&b,&c);                                                //Matrices are aXb and bXc
	long long int *h_m = (long long int *)malloc(a*b*sizeof(long long int));
	long long int *h_n = (long long int *)malloc(b*c*sizeof(long long int));
	long long int *h_p = (long long int *)malloc(a*c*sizeof(long long int));
	srand((unsigned int)time(NULL));                                                 //Seeding the random function
	for(i=0;i<a;i++)
	{
		for(j=0;j<b;j++)
			*(h_m + i*b +j) = ((long long int)rand());
	}
	for(i=0;i<b;i++)
	{
		for(j=0;j<c;j++)
			*(h_n + i*c +j) = ((long long int)rand());
	}
	
	dim3 DimGrid(ceil(a/16),ceil(c/16),1);
	dim3 DimBlock(16,16,1);
	
	printf("rrg");	
	long long int *d_m, *d_n, *d_p;
	cudaMalloc((long long int**)&d_m,a*b*sizeof(long long int));
	cudaMalloc((long long int**)&d_n,b*c*sizeof(long long int));
	cudaMalloc((long long int**)&d_p,a*c*sizeof(long long int));

	cudaMemcpy(d_m,h_m,a*b*sizeof(long long int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_n,h_n,b*c*sizeof(long long int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_p,h_p,a*c*sizeof(long long int),cudaMemcpyHostToDevice);
	 
	Matrix_Mul<<<DimGrid,DimBlock>>>(d_m,d_n,d_p,a,b,c);
	
	printf("EF");
	cudaMemcpy(h_m,d_m,a*b*sizeof(long long int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n,d_n,b*c*sizeof(long long int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p,d_p,a*c*sizeof(long long int),cudaMemcpyDeviceToHost);
	
	for(i=0;i<a;i++)
    {
        for(j=0;j<b;j++)
            printf("%lld ",*(h_p + i*b + j));
        printf("\n");
    }
    printf("\n");
    for(i=0;i<b;i++)
    {
        for(j=0;j<c;j++)
            printf("%lld ",*(h_m + i*c + j));
        printf("\n");
    }
    printf("\n");
	for(i=0;i<a;i++)
	{
		for(j=0;j<c;j++)
			printf("%lld ",*(h_n + i*c + j));
		printf("\n");
	}

	cudaFree(d_m); cudaFree(d_n); cudaFree(d_p);
	
	free(h_m); free(h_n); free(h_p);
	return 0;
}
