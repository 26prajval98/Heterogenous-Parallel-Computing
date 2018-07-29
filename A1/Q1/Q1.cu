#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

__global__ void generate(float * A, int size, int num, int MAX_THREAD){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    A[idx] = num;
}

void generate_in_cpu(float *A, int size){
    for(int i = 0; i< size; i++){
        A[i] = rand();
    }
}

__global__ void sum(float * A, float * B, float * C, int size, int MAX_THREAD){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
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

        int h_itr = rand() % (16001) + 16000;
        int d_itr = ceil((float)h_itr/MAX_THREAD) * MAX_THREAD;

        float * h_A, * h_B, * h_C;
        float * d_A, * d_B, * d_C;

        // printf("%d %d %d\n", h_itr, d_itr);

        h_A = (float *)malloc(h_itr * sizeof(float));
        h_B = (float *)malloc(h_itr * sizeof(float));
        h_C = (float *)malloc(h_itr * sizeof(float));

        cudaMalloc((void **)&d_A, d_itr*sizeof(float));
        cudaMalloc((void **)&d_B, d_itr*sizeof(float));
        cudaMalloc((void **)&d_C, d_itr*sizeof(float));

        int blocks = d_itr / MAX_THREAD;

        if(blocks < MAX_BLOCK){
            generate_in_cpu(h_A, h_itr);
            generate_in_cpu(h_B, h_itr);
            cudaMemcpy( d_A, h_A, d_itr * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy( d_B, h_B, d_itr * sizeof(float), cudaMemcpyHostToDevice);

            // generate<<<blocks, MAX_THREAD>>>(d_A, d_itr, 1, MAX_THREAD);
            // generate<<<blocks, MAX_THREAD>>>(d_B, d_itr, 3, MAX_THREAD);
            
            sum<<<blocks, MAX_THREAD>>>(d_A, d_B, d_C, d_itr, MAX_THREAD);

            cudaMemcpy( h_A, d_A, h_itr * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy( h_B, d_B, h_itr * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy( h_C, d_C, h_itr * sizeof(float), cudaMemcpyDeviceToHost);


            for(int i = 0 ; i< h_itr; i++){
                printf("%f %f %f\n",h_A[i], h_B[i], h_C[i]);
            }
        }
	}
	else{
		printf("Nvidia GPU not not found");
	}
}
