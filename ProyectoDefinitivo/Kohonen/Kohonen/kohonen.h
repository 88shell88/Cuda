#ifndef KOHONEN_H_
#define KOHONEN_H_
class kohonen{
public: 
	void train(int inputSize, int mapSize, int numInput, float *input, float *map, float *weight, float maxInputX, float minInputX, float maxInputY, float minInputY);

public:
	static const int inputSize = 2;
	static const int dimension = 2;
	static const int mapSize = 24;
	static const int numInput = 20;
	
};
#endif