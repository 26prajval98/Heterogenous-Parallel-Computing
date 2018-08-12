// 16CO145 Sumukha PK
// 16CO234 Prajval M

#include<stdio.h>
#include<cuda.h>
#include<math.h>

#define SIZE 4
#define TILE_WIDTH 4

__global__ void Matrix_Mul(long long int *d_m, long long int *d_n, long long int *d_p, long long int a, long long int b,long long int c)
{
    long long int bx = blockIdx.x; 
    long long int by = blockIdx.y;
    long long int tx = threadIdx.x; 
    long long int ty = threadIdx.y;
    long long int Row = by * blockDim.y + ty;
    long long int Col = bx * blockDim.x + tx;
	long long int temp = 0;

    __shared__ long long int ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ long long int ds_B[TILE_WIDTH][TILE_WIDTH];
    
	long long int i = min(Col, a-1);
	long long int j = min(Row, c-1);

    for(long long int p=0;p<k;p++)
    {
        ds_A[ty][tx] = A[Row*n + t*TILE_WIDTH+tx];
        ds_B[ty][tx] = B[(t*TILE_WIDTH+ty)*k + Col];
        __syncthreads();

        for(long long int k=0; k<TILE_WIDTH; k++)
        {
            temp += ds_A[ty][i] * ds_B[i][tx];
            __syncthreads();
        }
    
        d_p[Row*k + Col]  =  temp;
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
			h_m[i*b + j] = ((long long int)rand());
	}

	for(i=0;i<b;i++)
	{
		for(j=0;j<c;j++)
			h_n[i*c + j] = ((long long int)rand());
	}
	long long int c_a = a > SIZE ? (long long int)ceil(a/(float)SIZE) : 1;
	long long int c_c = c > SIZE ? (long long int)ceil(c/(float)SIZE) : 1;

	dim3 DimGrid(c_a, c_c, 1);
	dim3 DimBlock(SIZE,SIZE,1);
	
	long long int *d_m, *d_n, *d_p;
	cudaMalloc((long long int**)&d_m,a*b*sizeof(long long int));
	cudaMalloc((long long int**)&d_n,b*c*sizeof(long long int));
	cudaMalloc((long long int**)&d_p,a*c*sizeof(long long int));

	cudaMemcpy(d_m,h_m,a*b*sizeof(long long int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_n,h_n,b*c*sizeof(long long int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_p,h_p,a*c*sizeof(long long int),cudaMemcpyHostToDevice);
	 
	Matrix_Mul<<<DimGrid, DimBlock>>>(d_m,d_n,d_p,a,b,c);
		
	cudaMemcpy(h_m,d_m,a*b*sizeof(long long int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n,d_n,b*c*sizeof(long long int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p,d_p,a*c*sizeof(long long int),cudaMemcpyDeviceToHost);
	
	for(i=0;i<a;i++)
    {
        for(j=0;j<c;j++){
			printf("%lld ", h_p[i*c + j]);
		}
        printf("\n");
    }
    printf("\n");
    for(i=0;i<a;i++)
    {
        for(j=0;j<b;j++)
            printf("%lld ", h_m[i*b + j] );
        printf("\n");
    }
	printf("\n");
	
	for(i=0;i<b;i++)
	{
		for(j=0;j<c;j++)
			printf("%lld ", h_n[i*c + j]);
		printf("\n");
	}

	cudaFree(d_m); cudaFree(d_n); cudaFree(d_p);
	
	free(h_m); free(h_n); free(h_p);
	return 0;
}
