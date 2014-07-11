/********************************************************************
created:	2014/07/11
created:	11:7:2014   16:22
file base:	bu_im2col
file ext:	cu
author:		Zhao Kaiyong

zhao.kaiyong(at)gmail.com
kyzhao(at)comp.hkbu.edu.hk
http://www.comp.hkbu.edu.hk/~kyzhao/
http://blog.csdn.net/openhero

purpose:
Based on caffe im2col. Merge the loop into one kernel.
On GTX640:
image 256*256 with 3 channels
batch size is 128
The time show as below:
caffe is 106.883766ms
bu_im2col is 22.095470ms	
*********************************************************************/
#include "common.h"

template <typename Dtype>
__global__ void bu_im2col_gpu_kernel(
	const int n, const Dtype* data_im,
	const int height, const int width, const int ksize, const int pad,
	const int stride, const int height_col, const int width_col,
	Dtype* data_col,
	const int data_im_size,
	const int data_col_size,
	const int batch_size) 
{
	for(int batch_index = 0; batch_index < batch_size; batch_index++)
	{
		for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x){
			int w_out = index % width_col;
			int h_index = index / width_col;
			int h_out = h_index % height_col;
			int channel_in = h_index / height_col;
			int channel_out = channel_in * ksize * ksize;
			int h_in = h_out * stride - pad;
			int w_in = w_out * stride - pad;
			Dtype* data_col_ptr = data_col;
			data_col_ptr += batch_index* data_col_size + (channel_out * height_col + h_out) * width_col + w_out;
			const Dtype* data_im_ptr = data_im;
			data_im_ptr += batch_index* data_im_size + (channel_in * height + h_in) * width + w_in;

			Dtype temp_ret = 0.0f;
			for (int i = 0; i < ksize; ++i) {
				for (int j = 0; j < ksize; ++j) {
					int h = h_in + i;
					int w = w_in + j;
					temp_ret += (h >= 0 && w >= 0 && h < height && w < width) ?
						data_im_ptr[i * width + j]  : 0;
					data_col_ptr += height_col * width_col;
				}
			}

		}
	}
}

template <typename Dtype>
void bu_im2col_gpu(const Dtype* data_im, const int channels,
				   const int height, const int width, const int ksize, const int pad,
				   const int stride, Dtype* data_col, const int batch_size)
{
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int num_kernels = channels * height_col * width_col;

	int data_im_size = height*width*channels;
	int data_col_size = num_kernels*ksize*ksize;
	// NOLINT_NEXT_LINE(whitespace/operators)
	bu_im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), // num_kernels/16, means each thread process 16 elements
		CAFFE_CUDA_NUM_THREADS>>>(
		num_kernels, data_im, height, width, ksize, pad, stride, height_col,
		width_col, data_col, data_im_size, data_col_size, batch_size);
	CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void bu_im2col_gpu<float>(
	const float* data_im, const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride, float* data_col,
	const int batch_size);
template void bu_im2col_gpu<double>(
	const double* data_im, const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride, double* data_col,
	const int batch_size);


// Helper function for using CUDA to add vectors in parallel.
//const float* data_im // raw data,
//const int channels // image channels
//const int height //image height
//const int width // image width
//const int ksize // kernel size
//const int pad // pad size
//const int stride // stride size
//const int height_col // output column height
//const int width_col // output column width
//float* data_col // outpu data

cudaError_t bu_im2colWithCuda(
	const float* data_im,
	const int batch_size,
	const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride,
	float* data_col,
	const int num_kernels,
	float* data_kernel,
	float* data_ret)
{
	float *dev_a = 0;
	float *dev_c = 0;
	float *dev_k = 0;
	float *dev_ret = 0;
	cudaError_t cudaStatus;
	StopWatchInterface *timer = NULL;

	cublasHandle_t handle;
	cublasStatus_t ret;

	ret = cublasCreate(&handle);
	if (ret != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
		goto Error;
	}

	sdkCreateTimer(&timer);

	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;

	int K = ksize*ksize*channels;
	int M = num_kernels;
	int N = height_col*width_col*batch_size;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, N * K * batch_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, height * width * channels * batch_size* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, data_im, height * width * channels * batch_size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_k, num_kernels*ksize*ksize*channels * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_k, data_kernel, num_kernels*ksize*ksize*channels * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_ret, num_kernels*height_col *width_col *batch_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	const float alpha = 1.0f;
	const float beta  = 0.0f;
	int Batch_N = N * batch_size;
	sdkStartTimer(&timer);
	// Launch a kernel on the GPU with one thread for each element.
	bu_im2col_gpu<float>(dev_a, channels, height, width, ksize, pad, stride, dev_c, batch_size);
	//Perform warmup operation with cublas

	ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
		Batch_N , M,  K, &alpha,
		dev_c, Batch_N, dev_k, K, &beta, dev_ret, Batch_N);

	if (ret != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
		goto Error;
	}
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching im2col Kernel!\n", cudaStatus);
		goto Error;
	}

	sdkStopTimer(&timer);
	double elapsedTimeInMs = sdkGetTimerValue(&timer);
	printf("bu_im2col is %fms\n", elapsedTimeInMs);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(data_col, dev_c, N * K *batch_size* sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(data_ret, dev_ret, num_kernels*height_col *width_col *batch_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cublasDestroy(handle);
Error:

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_k);
	cudaFree(dev_ret);
	sdkDeleteTimer(&timer);

	return cudaStatus;
}