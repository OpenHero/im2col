
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
	const int batch_size = 32;//128;
	const int ksize = 5; // 5-11
	const int pad = 2; // 0-2
	const int stride = 1; // 1
	const int num_kernels = 64;


	const int arraySize = height * width * channels * batch_size; //each bacth have 128 image, each image have 256*256 size and 3 channels
	float *image = new float[arraySize];// = { 1, 2, 3, 4, 5 };

	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int colArraySize = height_col * width_col * channels *ksize*ksize* batch_size;
	float *col1 = new float[colArraySize]();// = { 10, 20, 30, 40, 50 };
	float *col2 = new float[colArraySize]();// = { 0 };
	
	const int kernelArraySize = num_kernels*ksize*ksize*channels;
	float *data_kernel = new float[kernelArraySize];

	const int resultArraySize = num_kernels * height_col * width_col*batch_size;
	float *r1 = new float[resultArraySize]();
	float *r2 = new float[resultArraySize]();

	//checkCudaErrors(cudaMallocManaged(&a, sizeof(float) *arraySize));
	//checkCudaErrors(cudaMallocManaged(&b, sizeof(float) *arraySize));
	//checkCudaErrors(cudaMallocManaged(&c, sizeof(float) *arraySize));
	srand(2014);
	init_data(image, arraySize);
	init_data(data_kernel, kernelArraySize);


		// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	// Add vectors in parallel.
	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//    fprintf(stderr, "addWithCuda failed!");
	//    return 1;
	//}

	// image to col
	 cudaStatus = im2colWithCuda(image, batch_size, channels, height, width, ksize, pad, stride, col1, num_kernels, data_kernel, r1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "im2colWithCuda failed!");
		return 1;
	}

	cudaStatus = bu_im2colWithCuda(image, batch_size, channels, height, width, ksize, pad, stride, col2, num_kernels, data_kernel,r2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "im2colWithCuda failed!");
		return 1;
	}

	int ret = -1;
	ret = check_result(col1,col2, resultArraySize);
	printf("Im2col error at %d.\n", ret);

	ret = check_result(r1,r2, resultArraySize);

	printf("Error at %d.\n", ret);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	delete [] image;
	delete [] col1;
	delete [] col2;
	delete [] data_kernel;
	delete [] r1;
	delete [] r2;

	return 0;
}

