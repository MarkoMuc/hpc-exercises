#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

#define BLOCK_SIZE 16

__global__ void printGPU(const unsigned char *text)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y == 0 && x == 0)
    {
        printf("%s", text);
    }
}

int main(void)
{
    char h_text[] = "Hello from GPU!\n";

    unsigned char *d_text;
    checkCudaErrors(cudaMalloc(&d_text, sizeof(h_text)));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(ceil(1.0 / blockSize.x), ceil(1.0 / blockSize.y));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    checkCudaErrors(cudaMemcpy(d_text, h_text, sizeof(h_text), cudaMemcpyHostToDevice));

    printGPU<<<gridSize, blockSize>>>(d_text);
    getLastCudaError("printGPU() execution failed\n");

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel & Memcpy Execution time is: %0.3f milliseconds \n", milliseconds);

    checkCudaErrors(cudaFree(d_text));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
