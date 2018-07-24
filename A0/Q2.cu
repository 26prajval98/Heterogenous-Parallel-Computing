#include<stdio.h>
#include<cuda.h>
__global__ void add_vec(float *d_a, int n)
{
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n) d_a[n-1]+=d_a[id];	
}
int main()
{
	int i,n=10;
	float h_a[n+1];
	h_a[n]=0;
	for(i=0;i<n;i++)
		scanf("%f",&h_a[i]);
	n++;
	float *d_a;
        cudaMalloc((void**)&d_a,n*sizeof(float));
        cudaMemcpy(d_a,h_a,n*sizeof(float),cudaMemcpyHostToDevice);
        add_vec<<<2,5>>>(d_a,n);
	cudaMemcpy(h_a,d_a,n*sizeof(float),cudaMemcpyDeviceToHost);
	printf("The sum is : %f \n",h_a[n-1]);
	cudaFree(d_a);
	return 0;
}
