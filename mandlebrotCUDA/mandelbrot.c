#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


void mandelbrotCPU(unsigned char *image, int height, int width) {
	float x0, y0, x, y, xtemp;
	int i, j;
	int color;
	int iter;
	int max_iteration = 1000;
	unsigned char max = 255;

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
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
	int height = atoi(argv[1]);//768
	int width = atoi(argv[2]);//1024
	int cpp=4; //color channels
	
	unsigned char *image = (unsigned char *)malloc(height * width * sizeof(unsigned char) * cpp);

	mandelbrotCPU(image, height, width);

	stbi_write_png("mandelbrot.png", width, height, cpp, image, width * cpp);

	free(image);
	return 0;
}
