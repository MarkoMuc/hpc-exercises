#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cuda.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"


/*
nvcc sort.cu -o sortGPU
        100         1000         10000        100000
CPU     0,76        7,41200     702,13500    72 357,685 [ms]
GPU    0.0954        0.2253      1.6485         85.5603 [ms]
        x7,96       x32,89       x425,92        x845,69
*/



#define THREADS_PER_BLOCK   (128)

void fillVector(int *V, int N);


__global__ void sort(int* IN,int* OUT,int length)
{
    //local memory
    __shared__ int part[THREADS_PER_BLOCK];

    //position in global array
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = 0;
    int stevec=0;

    int value = IN[i];

    int threadID = threadIdx.x;

    for(int count = 0; count < gridDim.x; count++){      

        part[threadIdx.x] = IN[threadID];

        //sync
        __syncthreads();
        
        if(i < length){
        //calc on current local memory
            for(int partID = 0; partID < THREADS_PER_BLOCK ; partID++){

                if(j >= length){ 
                    break;
                } 

                int val2 = part[partID];
                if(value > val2 || (value==val2 && (j < i))){
                stevec++;
                }

                j++;
            }
        }

        //sync
        __syncthreads();
        threadID += THREADS_PER_BLOCK;
    }

    if(i < length) OUT[stevec] = value;
}

int main(int argc, char* argv[])
{
    int length = atoi(argv[1]);

    //HOST
    int* arrayIn;
    int* arraySorted;

    //DEVICE
    int* arrayOut;
    int* arrayCopy;

    int blockspergrid = (length+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

    arrayIn = (int*)malloc(sizeof(int)*length);
    arraySorted = (int*)malloc(sizeof(int)*length);

    fillVector(arrayIn,length);

    cudaMalloc(&arrayOut,length*sizeof(int));
    cudaMalloc(&arrayCopy,length*sizeof(int));

    getLastCudaError("Memory allocation failed\n");
    
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy(arrayCopy,arrayIn,length * sizeof(int), cudaMemcpyHostToDevice);
    getLastCudaError("Memory copy failed\n");

    sort<<<blockspergrid,THREADS_PER_BLOCK>>>(arrayCopy,arrayOut,length);
    getLastCudaError("sort() exec fail\n");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("cudaGetDeviceCount error %d\n-> %s\n", err, cudaGetErrorString(err));
        exit(EXIT_FAILURE); 
    }

    cudaMemcpy(arraySorted, arrayOut, length*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("%0.3f\n",elapsedTime);

    cudaFree(arrayOut);
    cudaFree(arrayCopy);
    getLastCudaError("Error Freeing CUDA memory");
	
    free(arrayIn);
    free(arraySorted);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    getLastCudaError("Error Destroying events");
	
	return 0;
}

void fillVector(int *V, int N)
{
    for(int i = 0; i < N; i++){
        V[i] = rand() % 100;
    }
}
