#ifndef KOHONEN_H_
#define KOHONEN_H_

#include "cuda_runtime.h"
class kohonen{
public: 
	cudaError_t train(int inputSize, int mapSize, int numInput, float *input, float *map, float maxInputX, float minInputX, float maxInputY, float minInputY);

public:
	static const int inputSize = 2;
	static const int dimension = 2;
	
	static const int numInput=20;
	static const int mapSize=60;
	static const int N;

};
#endif