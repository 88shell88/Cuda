
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kohonen.h"
#include "device_functions.h"

#include <stdio.h>
#include <math.h>

__shared__ float map_shared[kohonen::mapSize * kohonen::dimension];
__shared__ float input_shared[kohonen::inputSize*kohonen::numInput];
__shared__ float weight_shared[kohonen::inputSize*kohonen::mapSize];

__global__ void learnApuntes(int mapSize, int inputSize, float maxInputX, float minInputX, float maxInputY, float minInputY, float *dev_input, float *dev_map, float *dev_weight){
	int minMapLeft1, minMapRight1, minMapLeft2, minMapRight2;
	float eta = 0.1f;
	int i = threadIdx.x;
	float hI = 0.0f;
	int minMap = 0;
	int nodo;
	float hR;
	int epoch;

	//start data
	if (i < kohonen::numInput){
		input_shared[i * 2] = (dev_input[i * 2] - (minInputX + maxInputX) / 2) / (maxInputX - minInputX);
		input_shared[i * 2 + 1] = (dev_input[i * 2 + 1] - (minInputY + maxInputY) / 2) / (maxInputY - minInputY);
		
	}
	map_shared[i * 2] = dev_map[i * 2];
	map_shared[i * 2 + 1] = dev_map[i * 2 + 1];
	weight_shared[i*inputSize] = dev_weight[i*inputSize];
	weight_shared[i*inputSize + 1] = dev_weight[i*inputSize +1];

	__syncthreads();

	printf("weight = %f    input= %f\n", weight_shared[0 * inputSize], input_shared[i*inputSize]);
	for (epoch = 0; epoch < 1000; epoch++){
		//hR = 0.0f;
		//sacar la neurona ganadora y sus vecinos
		hI = sqrt(pow(weight_shared[0*inputSize] * (input_shared[i*inputSize] - map_shared[0*inputSize]), 2) + pow(weight_shared[0*inputSize + 1] * (input_shared[i*inputSize + 1] - map_shared[0*inputSize + 1]), 2));
		for (nodo= 1; nodo < mapSize; nodo++){
			hR = sqrt(pow(weight_shared[nodo*inputSize] * (input_shared[i*inputSize] - map_shared[nodo*inputSize]), 2) + pow(weight_shared[nodo*inputSize + 1] * (input_shared[i*inputSize + 1]-map_shared[nodo*inputSize+1]), 2));
			if (hR < hI){
				printf("i= %d nodo = %d      hI = %f         hR = %f\n", i, nodo, hI, hR);
				hI = hR;
				minMap = nodo;

			}
		}
		minMapRight1 = minMap+1;
		if (minMapRight1 == mapSize) {
			minMapRight1 = 0;
			minMapRight2 = 1;
		}

		minMapLeft1 = minMap - 1;
		if (minMapLeft1 == -1)  {
			minMapLeft1 == mapSize - 1;
			minMapLeft2 == mapSize - 2;
		}
		minMapRight2 = minMap + 2;
		if (minMapRight2 == mapSize) minMapRight2 = 0;
		minMapLeft2 = minMap -2;
		if (minMapLeft2 == -1) minMapLeft2 = mapSize - 1;
		
		weight_shared[minMap*inputSize] = weight_shared[minMap*inputSize] + eta*(input_shared[i*inputSize] - weight_shared[minMap*inputSize]);
		weight_shared[minMap*inputSize + 1] = weight_shared[minMap*inputSize+1] + eta*(input_shared[i*inputSize+1] - weight_shared[minMap*inputSize+1]);
		__syncthreads();

		weight_shared[minMapLeft1*inputSize] = weight_shared[minMapLeft1*inputSize] + eta*0.5*(input_shared[i*inputSize] - weight_shared[minMapLeft1*inputSize]);
		weight_shared[minMapLeft1*inputSize + 1] = weight_shared[minMapLeft1*inputSize + 1] + eta*0.5*(input_shared[i*inputSize + 1] - weight_shared[minMapLeft1*inputSize + 1]);
		__syncthreads();

		weight_shared[minMapLeft2*inputSize]=weight_shared[minMapLeft2*inputSize] + eta*0.25*(input_shared[i*inputSize] - weight_shared[minMapLeft2*inputSize]);
		weight_shared[minMapLeft1*inputSize + 1] = weight_shared[minMapLeft2*inputSize + 1] + eta*0.25*(input_shared[i*inputSize + 1] - weight_shared[minMapLeft2*inputSize + 1]);
		__syncthreads();

		weight_shared[minMapRight1*inputSize] = weight_shared[minMapRight1*inputSize] + eta*0.5*(input_shared[i*inputSize] - weight_shared[minMapRight1*inputSize]);
		weight_shared[minMapLeft1*inputSize + 1] = weight_shared[minMapRight1*inputSize + 1] + eta*0.5*(input_shared[i*inputSize + 1] - weight_shared[minMapRight1*inputSize + 1]);
		__syncthreads();

		weight_shared[minMapRight2*inputSize] = weight_shared[minMapRight2*inputSize] + eta*0.25*(input_shared[i*inputSize] - weight_shared[minMapRight2*inputSize]);
		weight_shared[minMapLeft1*inputSize + 1] = weight_shared[minMapRight2*inputSize + 1] + eta*0.25*(input_shared[i*inputSize + 1] - weight_shared[minMapRight2*inputSize + 1]);
	}
	printf("***********%d    %d\n",i, minMap);
	//stop data
	if (i < kohonen::numInput){
		dev_weight[i*inputSize] = weight_shared[i*inputSize];
		dev_weight[i*inputSize + 1] = weight_shared[i*inputSize + 1];
	}
}


// Helper function for using CUDA to add vectors in parallel.
void kohonen::train(int inputSize, int mapSize, int numInput, float *input, float *map, float *weight, float maxInputX, float minInputX, float maxInputY, float minInputY)
{
	float *dev_input = 0;
	float *dev_map = 0;
	float *dev_weight = 0;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_weight, inputSize*mapSize* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_input, inputSize*numInput* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_map, mapSize * dimension * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_weight, weight, inputSize*mapSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_input, input, inputSize*numInput * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_map, map, mapSize * 2 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    // Launch a kernel on the GPU with one thread for each element.
	learnApuntes << <1, mapSize >> >(mapSize, inputSize, maxInputX, minInputX, maxInputY, minInputY, dev_input, dev_map, dev_weight);
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
	cudaStatus = cudaMemcpy(weight, dev_weight, mapSize * inputSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(map, dev_map, mapSize * 2 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
    cudaFree(dev_input);
	cudaFree(dev_map);
	cudaFree(dev_weight);
    
    //return cudaStatus;
}
