#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

#define THREADS_PER_BLOCK 16

/*
module load CUDA/10.1.243-GCC-8.3.0
nvcc mandelbrot.cu -o mandelbrotGPU
srun -n1 -G1 --reservation=fri mandelbrotGPU

	640x480 800x600, 1600x900, 1920x1080, 3840x2160
CPU	 1.01s  1.698	 5.087		7.313	   	29.423
GPU  1.2681 1.9165   3.9398     5.4191      18.4954 ms
    x796    x885     x1291       x1349      x1590

*/

__global__ void mandelbrotGPU(unsigned char *image, int height, int width){

   float x0, y0, x, y, xtemp;
	int j = blockIdx.x*blockDim.x+threadIdx.x;
    int i = blockIdx.y*blockDim.y+threadIdx.y;
	int color;
	int iter;
	int max_iteration = 1000;
	unsigned char max = 255;

    if(1){
    x0 = (float)j / width * (float)3.5 - (float)2.5;
    y0 = (float)i / height * (float)2.0 - (float)1.0;
    x = 0;
    y = 0;
    iter = 0;
    while ((x*x + y * y <= 4) && (iter < max_iteration))
    {
        xtemp = x * x - y * y + x0;
        y = 2 * x*y + y0;
        x = xtemp;
        iter++;
    }
    color = 1.0 + iter - log(log(sqrt(x*x + y * y))) / log(2.0);
    color = (8 * max * color) / max_iteration;
    if (color > max)
        color = max;
    image[4 * i*width + 4 * j + 0] = color; //Red
    image[4 * i*width + 4 * j + 1] = 0; // Green
    image[4 * i*width + 4 * j + 2] = 0; // Blue
    image[4 * i*width + 4 * j + 3] = 255;   // Alpha
    }
}

int main(int argc, char* argv[])
{
    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    int cpp = 4;
    int size = height * width * sizeof(unsigned char) * cpp;

    unsigned char *imageDevice = (unsigned char *)malloc(size);
    unsigned char *imageGPU;
    
    checkCudaErrors(cudaMalloc(&imageGPU, size));
    getLastCudaError("Memory assignment failed\n");

    
    dim3 blockSize(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
    dim3 gridSize((width-1)/THREADS_PER_BLOCK + 1, (height-1)/THREADS_PER_BLOCK + 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    mandelbrotGPU<<<gridSize, blockSize>>>(imageGPU,height,width);
    getLastCudaError("mandelbrotGPU() execution failed\n");

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%0.3f\n", milliseconds);
    
    cudaMemcpy(imageDevice,imageGPU,size, cudaMemcpyDeviceToHost);
    getLastCudaError("Error getting image from device\n");
    
    stbi_write_png("mandelbrot.png", width, height, cpp, imageDevice, width * cpp);

    free(imageDevice);
    cudaFree(imageGPU);
    getLastCudaError("Error Freeing imageGPU");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    getLastCudaError("Error Destroying events");


    return 0;
}
