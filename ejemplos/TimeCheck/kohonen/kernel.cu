
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

cudaError_t addWithCuda(float *map,float *in,float *weight, unsigned int size, int numColumns,int inSize);

__global__ void addKernel(float *map, float *weight, float *in, int numColumns, int inSize)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);
	
	const int thread_1D_pos = thread_2D_pos.y * numColumns + thread_2D_pos.x;
	map[thread_1D_pos] = map[thread_1D_pos] + 1;
	int i;
	for (i = 0; i < inSize;i++){
		weight[thread_1D_pos*inSize + i] = weight[thread_1D_pos*inSize + i] + i + 2;
	}
}

int main()
{
	const int numColumns = 10;
	const int numLines = 10;
	const int inSize = 7;
    const int mapSize = numLines * numColumns;
	float weight[numLines * numColumns * inSize] = { 0 };
	float in[inSize] = { 0 };
	float map[numLines*numColumns] = { 0 };
	
    // Add vectors in parallel.
	DWORD dw1 = GetTickCount();
    cudaError_t cudaStatus = addWithCuda(map,in,weight, mapSize, numColumns,inSize);
	DWORD dw2 = GetTickCount();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	
	printf("Time difference is %d miliseconds\n", (dw2 - dw1));
	int i;
	printf("**************************************************************\n");
	for (i = 0; i < sizeof(weight) / sizeof(weight[0]); i++){
		printf("%f ", weight[i]);
	}
	printf("\n**************************************************************\n");
	for (i = 0; i < sizeof(map) / sizeof(map[0]); i++){
		printf("%f ", map[i]);
	}
	printf("\n**************************************************************\n");
	system("pause");
   // printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
   //     map[0], map[1], map[2], map[3], map[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *map,float *in,float *weight, unsigned int size, int numColumns, int inSize)
{
    float *dev_map = 0;
	float *dev_in = 0;
	float *dev_weight = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_map, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc((void**)&dev_in, inSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_weight, inSize*size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}



	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_map, map, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_in, in, inSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_weight, weight, inSize*size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}



    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_map,dev_weight, dev_in,numColumns,inSize);

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
    cudaStatus = cudaMemcpy(map, dev_map, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(weight, dev_weight, size*inSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
    cudaFree(dev_map);
    
    return cudaStatus;
}
