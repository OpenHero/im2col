
#include "common.h"

cudaError_t addWithCuda(float *c, float *a, float *b, unsigned int size);

__global__ void addKernel(float *c, float *a, float *b, int size)
{
    int i = blockIdx.x * blockDim.x *blockDim.y  + blockDim.x * threadIdx.y * threadIdx.x;
	while(i < size)
	{
		c[i] = a[i] + b[i];
		i += gridDim.x * blockDim.x * blockDim.y;
	}

}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *c, float *a, float *b, unsigned int size)
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
    addKernel <<< grid, block>>>(dev_c, dev_a, dev_b, size);

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

int check_result(float* a, float* b, int size)
{
	int ret = -1; 
	for(int i = 0; i < size; i++)
	{
		if(a[i] != b[i])
		{
			ret = i;
			break;
		}
	}
	return ret;
}