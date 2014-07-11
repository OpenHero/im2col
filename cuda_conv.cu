
#include <stdio.h>
#include <cstdlib>

#include "im2col.hpp"

void init_data(float * data, int size)
{
	for(int i = 0; i < size; i++)
	{
		data[i] = (float)rand()/(float)size;
	}
}

int main()
{
	const int height = 256;
	const int width = 256;
	const int channels = 3;
	const int batch_size = 128;//128;
	const int ksize = 5; // 5-11
	const int pad = 2; // 0-2
	const int stride = 1; // 1

	const int arraySize = height * width * channels * batch_size; //each bacth have 128 image, each image have 256*256 size and 3 channels
	float *a = new float[arraySize];// = { 1, 2, 3, 4, 5 };
	float *b = new float[arraySize];// = { 10, 20, 30, 40, 50 };
	float *c = new float[arraySize];// = { 0 };

	//checkCudaErrors(cudaMallocManaged(&a, sizeof(float) *arraySize));
	//checkCudaErrors(cudaMallocManaged(&b, sizeof(float) *arraySize));
	//checkCudaErrors(cudaMallocManaged(&c, sizeof(float) *arraySize));
	srand(2014);
	init_data(a, arraySize);
	init_data(b, arraySize);

	// Add vectors in parallel.
	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//    fprintf(stderr, "addWithCuda failed!");
	//    return 1;
	//}

	// image to col
	cudaError_t cudaStatus = im2colWithCuda(a, batch_size, channels, height, width, ksize, pad, stride, b);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "im2colWithCuda failed!");
		return 1;
	}

	cudaStatus = bu_im2colWithCuda(a, batch_size, channels, height, width, ksize, pad, stride, c);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "im2colWithCuda failed!");
		return 1;
	}
		
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int col_size = height_col * width_col * channels * batch_size;
	int ret = check_result(b,c, col_size);

	printf("Error at %d.\n", ret);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//cudaFree(a);
	//cudaFree(b);
	//cudaFree(c);
	delete [] a;
	delete [] b;
	delete [] c;

	return 0;
}

