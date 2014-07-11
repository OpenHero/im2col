/********************************************************************
	created:	2014/07/11
	created:	11:7:2014   16:22
	file base:	common
	file ext:	h
	author:		Zhao Kaiyong

	zhao.kaiyong(at)gmail.com
	kyzhao(at)comp.hkbu.edu.hk
	http://www.comp.hkbu.edu.hk/~kyzhao/
	http://blog.csdn.net/openhero
	
	purpose:	
*********************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>
#include <cublas_v2.h>

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
	/* Code block avoids redefinition of cudaError_t error */ \
	do { \
	cudaError_t error = condition; \
	if (error != cudaSuccess) \
	std::cout << " " << cudaGetErrorString(error); \
	} while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
	i < (n); \
	i += blockDim.x * gridDim.x)
// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
	return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
#endif