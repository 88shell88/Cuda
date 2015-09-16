#include "kohonen.h"
#include <cuda.h>
#include <time.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <exception>
#include "kohonen.h"
#include <algorithm>
#include <fstream>
#include <windows.h>
using namespace std;

int main(const int argc, const char *const *const argv){
	try{
		int i = 0;
		kohonen koh;
		float input[kohonen::numInput * kohonen::inputSize] = { 53.214, 19.155, 52.349, 18.924,
			53.560, 19.221, 49.763, 19.076,
			53.280, 19.167, 52.758, 18.790,
			53.111, 19.220, 52.988, 18.920,
			54.200, 19.210, 53.719, 19.194,
			54.390, 19.185, 53.684, 19.031,
			53.848, 19.079, 54.058, 19.269,
			53.721, 19.170, 54.076, 19.399,
			53.603, 19.027, 54.390, 19.399,
			53.494, 18.969, 49.763, 18.790 };
		float inputX[kohonen::numInput * kohonen::inputSize / 2];
		float inputY[kohonen::numInput * kohonen::inputSize / 2];
		for (i = 0; i < kohonen::numInput * kohonen::inputSize;i++){
			if ((i % 2) == 0){
				inputX[i / 2] = input[i];
			}
			else{
				inputY[i / 2] = input[i];
			}
		}
		float maxInputX = *std::max_element(inputX, inputX + kohonen::numInput * kohonen::inputSize/2);
		float minInputX = *std::min_element(inputX, inputX + kohonen::numInput * kohonen::inputSize/2);
		float maxInputY = *std::max_element(inputY, inputY + kohonen::numInput * kohonen::inputSize/2);
		float minInputY = *std::min_element(inputY, inputY + kohonen::numInput * kohonen::inputSize/2);
		//360 entre numero de puntos distancia entre angulos. 1º cos angulo y 2º sen angulo
		float map[kohonen::mapSize][kohonen::dimension] = { { 1, 0 }, { 0.966, 0.259 }, { 0.866, 0.5 }, { 0.707, 0.707 }, { 0.5, 0.866 }, { 0.259, 0.966 },
		{ 0, 1 }, { -0.259, 0.966 }, { -0.5, 0.866 }, { -0.707, 0.707 }, { -0.866, 0.5 }, { -0.966, 0.259 },
		{ -1, 0 }, { -0.966, -0.259 }, { -0.866, -0.5 }, { -0.707, -0.707 }, { -0.5, -0.866 }, { -0.259, -0.966 },
		{ 0, -1 }, { 0.259, -0.966 }, { 0.5, -0.866 }, { 0.707, -0.707 }, { 0.866, -0.5 }, { 0.966, -0.259 } };
		float weight[kohonen::mapSize*kohonen::inputSize] = { 0 };//rand() / ((double) RAND_MAX);
		for (i = 0; i < kohonen::mapSize*kohonen::inputSize; i++){
			weight[i] = rand() / ((float)RAND_MAX);
			printf("%f\n", weight[i]);
		}
		DWORD dw1 = GetTickCount();
		koh.train(kohonen::inputSize, kohonen::mapSize, kohonen::numInput, input, *map, weight, maxInputX, minInputX, maxInputY, minInputY);
		DWORD dw2 = GetTickCount();
		printf("Time difference is %d miliseconds\n", (dw2 - dw1));
		system("PAUSE");
		
		for (i = 0; i < kohonen::mapSize*kohonen::inputSize; i++){
			printf("%f\n",weight[i]);
		}
		system("PAUSE");
		/*printf("*****************************************************************************************");
		for (i = 0; i < kohonen::mapSize*kohonen::inputSize; i++){
			weight[i] = rand() / ((double)RAND_MAX);
			printf("%f\n", weight[i]);
		}
		system("PAUSE");*/
	}
	catch (std::exception &e){
		printf("error");
	}
	return 0;
}