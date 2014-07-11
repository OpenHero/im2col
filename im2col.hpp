#ifndef __IM2COL_H__
#define __IM2COL_H__

#include "common.h"

int check_result(float* a, float* b, int size);

cudaError_t addWithCuda(float *c, float *a, float *b, unsigned int size);

cudaError_t im2colWithCuda(
	const float* data_im,
	const int batch_size,
	const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride,
	float* data_col);

cudaError_t bu_im2colWithCuda(
	const float* data_im,
	const int batch_size,
	const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride,
	float* data_col);

#endif // __IM2COL_H__
