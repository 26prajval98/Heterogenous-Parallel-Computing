#include<stdio.h>
#include<cuda.h>
#include<math.h>

int main()
{
	long long int a,b,c,i,j;
	printf("Enter the dimensions of the 2 matrices: ");
	scanf("%lld %lld %lld",&a,&b,&c);                     //Matrices are aXb and bXc
	float *h_m = (float *)malloc(a*b*sizeof(float));
	float *h_n = (float *)malloc(b*c*sizeof(float));
	float *h_s = (float *)malloc(a*c*sizeof(float));      //aXb * bXc = aXc
	srand((unsigned int)time(NULL));                      //Seeding the random function
	for(i=0;i<a;i++)
	{
		for(j=0;j<b;j++)
			*(h_m + i*a +j) = ((float)rand());
	}
	for(i=0;i<b;i++)
	{
		for(j=0;j<c;j++)
			*(h_n + i*b +j) = ((float)rand());
	}
	dim3 DimBlock(16,16,1);
	dim3 DimGrid(ceil(n/16),ceil(n/16),1);

	cudaMalloc((void**)&d_m,a*b*sizeof(float));
	cudaMalloc((void**)&d_n,b*c*sizeof(float));
	cudaMallic((void**)&d_s,a*c*sizeof(float));

	cudaMemcpy(d_m,h_m,a*b*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_n,h_n,b*c*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_s,h_s,a*c*sizeof(float),cudaMemcpyHostToDevice);
	 
	Matrix_Mul<<<DimGrid,DimBlock>>(*d_m,*d_n,*d_s,a,b,c);
	
	cudaMemcpy(h_m,d_m,a*b*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_n,d_n,b*c*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s,d_s,a*c*sizeof(float),cudaMemcpyDeviceToHost);
	
	cudaFree(d_m); cudaFree(d_n); cudaFree(d_s); 
	
	for(i=0;i<a;i++)	
	{
		for(j=0;j<c;j++)
			printf("%f ",*(h_s + i*a + j));
		printf("\n");
	}
	
	free(h_m); free(h_n); free(h_s);
	return 0;
}
