#ifndef KOHONEN_H_
#define KOHONEN_H_

#include "cuda_runtime.h"
class kohonen{
public: 
	cudaError_t train(int inputSize, int mapSize, int numInput, float *input, float *map, float maxInputX, float minInputX, float maxInputY, float minInputY, int timesMap, int numEpoch1, int numEpoch2, int numEpoch3, float eta1, float eta2);

public:
	static const int inputSize = 2;
	
};
#endif