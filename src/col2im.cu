#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include "common.h"

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels, const int ksize,
    const int pad, const int stride, const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
    int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
    int coeff_w_col = (1 - stride * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

//__global__ void col2im_gpu_kernel<float>(const int n, const float* data_col,
//    const int height, const int width, const int channels, const int ksize,
//    const int pad, const int stride, const int height_col, const int width_col,
//    float* data_im) {
//  CUDA_KERNEL_LOOP(index, n) {
//    float val = 0;
//    int w = index % width + pad;
//    int h = (index / width) % height + pad;
//    int c = index / (width * height);
//    // compute the start and end of the output
//    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
//    int w_col_end = min(w / stride + 1, width_col);
//    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
//    int h_col_end = min(h / stride + 1, height_col);
//    /*
//    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
//      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
//        // the col location: [c * width * height + h_out, w_out]
//        int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
//        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
//      }
//    }
//    */
//    // equivalent implementation
//    int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
//    int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
//    int coeff_w_col = (1 - stride * height_col * width_col);
//    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
//      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
//        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
//      }
//    }
//    data_im[index] = val;
//  }
//}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im) {
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, ksize, pad, stride,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im);

// Helper function for using CUDA to add vectors in parallel.
cudaError_t col2imWithCuda(float *c, float *a, float *b, unsigned int size)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	StopWatchInterface *timer = NULL;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	sdkCreateTimer(&timer);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	sdkStartTimer(&timer);

    // Launch a kernel on the GPU with one thread for each element.
	dim3 block(32,16);
	dim3 grid(size/(32*16));
    //addKernel <<< grid, block>>>(dev_c, dev_a, dev_b, size);

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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	sdkStopTimer(&timer);
	double elapsedTimeInMs = sdkGetTimerValue(&timer);
	printf("time is %fms\n", elapsedTimeInMs);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    sdkDeleteTimer(&timer);

    return cudaStatus;
}