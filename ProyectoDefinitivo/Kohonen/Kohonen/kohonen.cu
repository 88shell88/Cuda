#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kohonen.h"
#include "device_functions.h"

__global__ void learnFirstIteration(int mapSize, int inputSize, int numInput, float maxInputX, float minInputX, float maxInputY, float minInputY, float *dev_input, float *dev_map){
	extern __shared__ float shared[];
	float *map_shared = shared;
	float *input_shared = (float*)&map_shared[mapSize * 2];
	float *hits_shared = (float*)&input_shared[numInput * 2];
	
	float eta = 0.1f;
	int i = threadIdx.x;
	float hI;
	int minMap = 0;
	int minNodo = 0;
	int nodo;
	float hR;
	int epoch;
	//start data
	
	
	if (i < kohonen::numInput){
		input_shared[i * 2] = dev_input[i * 2];
		input_shared[i * 2 + 1] = dev_input[i * 2 + 1] ;
		hits_shared[i] = 0;
	}

	map_shared[i * 2] = dev_map[i * 2];
	map_shared[i * 2 + 1] = dev_map[i * 2 + 1];
	__syncthreads();

	for (epoch = 0; epoch < 50; epoch++){
		hR = 0.0f;
		//sacar la neurona ganadora y sus vecinos
		hI = sqrt(pow((input_shared[0 * inputSize] - map_shared[i*inputSize]), 2) + pow((input_shared[0 * inputSize + 1] - map_shared[i*inputSize + 1]), 2));
		for (nodo = 1; nodo < numInput; nodo++){
			hR = sqrt(pow((input_shared[nodo*inputSize] - map_shared[i*inputSize]), 2) + pow((input_shared[nodo*inputSize + 1] - map_shared[i*inputSize + 1]), 2)) *((hits_shared[nodo]+1) / (epoch + 1));
			if (hR < hI){

				hI = hR;
				minNodo = nodo;

			}

		}
		hits_shared[minNodo] = hits_shared[minNodo] + 1;
		minMap = i;
		
		map_shared[minMap*inputSize] = map_shared[minMap*inputSize] + eta*(input_shared[minNodo*inputSize] - map_shared[minMap*inputSize]);
		map_shared[minMap*inputSize + 1] = map_shared[minMap*inputSize + 1] + eta*(input_shared[minNodo*inputSize + 1] - map_shared[minMap*inputSize + 1]);

	}
	//stop data
	dev_map[i*inputSize] = map_shared[i*inputSize];
	dev_map[i*inputSize + 1] = map_shared[i*inputSize + 1];
}



__global__ void learnSecondIteration(int mapSize, int inputSize, int numInput, float maxInputX, float minInputX, float maxInputY, float minInputY, float *dev_input, float *dev_map){
	extern __shared__ float shared[];
	float *map_shared = shared;
	float *input_shared = (float*)&map_shared[mapSize * 2];

	int minMapLeft1, minMapRight1, minMapLeft2, minMapRight2;
	float eta = 0.1f;
	int i = threadIdx.x;
	float hI;
	int minMap = 0;
	int minNodo = 0;
	int nodo;
	float hR;
	int epoch;

	//start data
	input_shared[i * 2] = dev_input[i * 2];
	input_shared[i * 2 + 1] = dev_input[i * 2 + 1];


	map_shared[i * 6] = dev_map[i * 6];
	map_shared[i * 6 + 1] = dev_map[i * 6 + 1];
	map_shared[i * 6 + 2] = dev_map[i * 6 + 2];
	map_shared[i * 6 + 3] = dev_map[i * 6 + 3];
	map_shared[i * 6 + 4] = dev_map[i * 6 + 4];
	map_shared[i * 6 + 5] = dev_map[i * 6 + 5];


	__syncthreads();

	
	//Iteraciones desde 0 hasta X
	
	for (epoch = 0; epoch < 500; epoch++){
		// hR es la distancia a comprobar, empieza a 0
		hR = 0.0f;

		//sacar la neurona ganadora y sus vecinos
		//hI es el minimo local
		hI = sqrt(pow((input_shared[0] - map_shared[i*inputSize]), 2) + pow((input_shared[1] - map_shared[i*inputSize + 1]), 2));
		//Por cada nodo que no sea el primero
		for (nodo = 1; nodo < mapSize; nodo++){
			//calcula la distancia de ese nodo

			hR = sqrt(pow((input_shared[i*inputSize] - map_shared[nodo*inputSize]), 2) + pow((input_shared[i*inputSize + 1] - map_shared[nodo*inputSize + 1]), 2));
			if ((hR < hI)){
				hI = hR;
				minMap = nodo;

			}

		}

		minNodo = i;
		//necesito almacenar el indice de los nodos siguientes
		minMapRight1 = minMap + 1;
		minMapRight2 = minMap + 2;
		if (minMapRight1 == mapSize) {
			minMapRight1 = 0;
			minMapRight2 = 1;
		}
		if (minMapRight2 == mapSize) minMapRight2 = 0;

		//y los dos nodos anteriores
		minMapLeft1 = minMap - 1;
		minMapLeft2 = minMap - 2;
		if (minMapLeft1 == -1)  {
			
			minMapLeft1 = mapSize - 1;
			minMapLeft2 = mapSize - 2;
		}
		if (minMapLeft2 == -1) minMapLeft2 = mapSize - 1;
		eta = eta - eta / 100;
		if (epoch >= 100) eta = 0.5f;
		map_shared[minMap*inputSize] = map_shared[minMap*inputSize] + eta*(input_shared[minNodo*inputSize] - map_shared[minMap*inputSize]);
		map_shared[minMap*inputSize + 1] = map_shared[minMap*inputSize + 1] + eta*(input_shared[minNodo*inputSize + 1] - map_shared[minMap*inputSize + 1]);

		if (epoch < 100){
			map_shared[minMapLeft1*inputSize] = map_shared[minMapLeft1*inputSize] + eta*0.5*(input_shared[minNodo*inputSize] - map_shared[minMapLeft1*inputSize]);
			map_shared[minMapLeft1*inputSize + 1] = map_shared[minMapLeft1*inputSize + 1] + eta*0.5*(input_shared[minNodo*inputSize + 1] - map_shared[minMapLeft1*inputSize + 1]);

			map_shared[minMapLeft2*inputSize] = map_shared[minMapLeft2*inputSize] + eta*0.25*(input_shared[minNodo*inputSize] - map_shared[minMapLeft2*inputSize]);
			map_shared[minMapLeft2*inputSize + 1] = map_shared[minMapLeft2*inputSize + 1] + eta*0.25*(input_shared[minNodo*inputSize + 1] - map_shared[minMapLeft2*inputSize + 1]);

			map_shared[minMapRight1*inputSize] = map_shared[minMapRight1*inputSize] + eta*0.5*(input_shared[minNodo*inputSize] - map_shared[minMapRight1*inputSize]);
			map_shared[minMapRight1*inputSize + 1] = map_shared[minMapRight1*inputSize + 1] + eta*0.5*(input_shared[minNodo*inputSize + 1] - map_shared[minMapRight1*inputSize + 1]);

			map_shared[minMapRight2*inputSize] = map_shared[minMapRight2*inputSize] + eta*0.25*(input_shared[minNodo*inputSize] - map_shared[minMapRight2*inputSize]);
			map_shared[minMapRight2*inputSize + 1] = map_shared[minMapRight2*inputSize + 1] + eta*0.25*(input_shared[minNodo*inputSize + 1] - map_shared[minMapRight2*inputSize + 1]);
		}
	}
	__syncthreads();
	//stop data
	dev_map[i * 6] = map_shared[i * 6];
	dev_map[i * 6 + 1] = map_shared[i * 6 + 1];
	dev_map[i * 6 + 2] = map_shared[i * 6 + 2];
	dev_map[i * 6 + 3] = map_shared[i * 6 + 3];
	dev_map[i * 6 + 4] = map_shared[i * 6 + 4];
	dev_map[i * 6 + 5] = map_shared[i * 6 + 5];
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t kohonen::train(int inputSize, int mapSize, int numInput, float *input, float *map, float maxInputX, float minInputX, float maxInputY, float minInputY)
{
	
	float *dev_input = 0;
	float *dev_map = 0;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
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
	learnFirstIteration << <1, mapSize , sizeof(float)*(mapSize*2+numInput*2 + numInput)>> >(mapSize, inputSize, numInput, maxInputX, minInputX, maxInputY, minInputY, dev_input, dev_map);
	learnSecondIteration << <1, numInput, sizeof(float)*(mapSize * 2 + numInput * 2) >> >(mapSize, inputSize, numInput, maxInputX, minInputX, maxInputY, minInputY, dev_input, dev_map);

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
	cudaStatus = cudaMemcpy(map, dev_map, mapSize * 2 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
    cudaFree(dev_input);
	cudaFree(dev_map);
    
    return cudaStatus;
}
