#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <windows.h>
#include <math.h>

cudaError_t addWithCuda(int *c, const int *a, unsigned int size);

__global__ void addKernel(int *c, const int *a)
{
	int i =threadIdx.x;
	int j = threadIdx.y;
	c[i] = a[i] + 1;
	//printf("%d\n", c[i]);
}

int main()
{
	const int arraySize = 100;
	const int a[arraySize][arraySize] = { 0 };
	int c[arraySize][arraySize] = { 0 };
	int i, j; 
	/*for (i = 0; i < arraySize; i++){
		for (j = 0; j < arraySize; j++){
			printf("%d", c[i][j]);
		}
		printf("\n");
	}*/
	system("PAUSE");
	DWORD dw1 = GetTickCount();
	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(*c, *a, arraySize*arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	DWORD dw2 = GetTickCount();
	for (i = 0; i < arraySize; i++){
		for (j = 0; j < arraySize; j++){
			printf("%d", c[i][j]);
		}
		printf("\n");
	printf("Time difference is %d miliseconds\n", (dw2 - dw1));
	system("PAUSE");
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	for (i = 0; i < arraySize; i++){
		for (j = 0; j < arraySize; j++){
			printf("%d",c[i][j]);
		}
		printf("\n");
	}
	system("PAUSE");
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, unsigned int size)
{
	int *dev_a = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int blocks, threads;
	if (size <= 512){
		blocks = 1;
		threads = size;
	}
	else{
		threads = 512;
		blocks = (int)ceil((float)size / 512);
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <blocks, threads >> >(dev_c, dev_a);

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

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);

	return cudaStatus;
}