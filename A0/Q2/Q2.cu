// 16CO234 	Prajval M
// 16CO145 	Sumukha PK

#include<stdio.h>
#include<cuda.h>

__global__ void add_vec(float *d_a, int n){		                         //7.CUDA Kernel that computes sum
	int i = threadIdx.x;
	if((n-i-1)!=i)	{
		d_a[i]+=d_a[n-i-1];
	}
}


int main(){

	int i, n, deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if(deviceCount > 0){

		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);							// Use default GPU
		
		#define MAX_THREAD devProp.maxThreadsPerBlock

		printf("Enter the size of the array: ");
		scanf("%d",&n);
		
		if(n < MAX_THREAD){											   // Do not sum if array size is greater than thread-size per block 

			float *h_a;
			h_a = (float *)malloc(sizeof(float)*n);

			printf("Enter the values of the array: ");
			for(i=0;i<n;i++)
				scanf("%f",&h_a[i]);

			float *d_a; 
			cudaMalloc((void**)&d_a,n*sizeof(float));                    //1.Allocate device memory
			cudaMemcpy(d_a,h_a,n*sizeof(float),cudaMemcpyHostToDevice);  //2.Copy host memory to device
			int Block_size = 1, threads_used;                            //3.Initialise thread block and kernel grid dimensions

			for(i=n;i>1;i= i%2?i/2 + 1:i/2){
				threads_used = i/2;                                 
				add_vec<<<Block_size,threads_used>>>(d_a,i);         	 //4.Invoke kernel
			}

			float z;
			
			cudaMemcpy(&z,&d_a[0],sizeof(float),cudaMemcpyDeviceToHost); //5.Copy results from device to host 
			cudaFree(d_a);                                               //6.Free device memory
			free(h_a);                         			     			 // free host memory
			printf("The sum is : %f \n",z);    

			return 0;
		}
		else{
			printf("Not Supported :( \n");
		}
		
	}
	else{
		printf("Nvidia GPU not not found");
	}
	return -1;
}
