#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void generate(float * A, int size, int num, int MAX_THREAD){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < size)
        A[idx] = num;
}

__global__ void sum(float * A, float * B, float * C, int size, int MAX_THREAD){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < size)
        C[idx] = A[idx] + B[idx];
}

int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if(deviceCount > 0){

		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);		                     // Use default GPU
		
		#define MAX_THREAD devProp.maxThreadsDim[0]
        #define MAX_BLOCK devProp.maxGridSize[0]

        int itr = rand() % (1024 - 0 + 1) + 1024;
        
        float * h_A, * h_B, * h_C;
        float * d_A, * d_B, * d_C;

        h_A = (float *)malloc(itr * sizeof(float));
        h_B = (float *)malloc(itr * sizeof(float));
        h_C = (float *)malloc(itr * sizeof(float));

        cudaMalloc((void **)&d_A, itr*sizeof(float));
        cudaMalloc((void **)&d_B, itr*sizeof(float));
        cudaMalloc((void **)&d_C, itr*sizeof(float));

        int blocks = itr / MAX_THREAD;

        if(blocks < MAX_BLOCK){
            generate<<<blocks, MAX_THREAD>>>(d_A, itr, rand(), MAX_THREAD);
            generate<<<blocks, MAX_THREAD>>>(d_B, itr, rand(), MAX_THREAD);
            sum<<<blocks, MAX_THREAD>>>(d_A, d_B, d_C, itr, MAX_THREAD);

            cudaMemcpy( h_A, d_A, itr * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy( h_B, d_B, itr * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy( h_C, d_C, itr * sizeof(float), cudaMemcpyDeviceToHost);


            for(int i =0 ; i<itr; i++){
                printf("%f %f %f\n",h_A[i], h_B[i], h_C[i]);
            }
        }
	}
	else{
		printf("Nvidia GPU not not found");
	}
}
