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
#include <iostream>
#include "cuda_runtime.h"
using namespace std;

#define PI 3.14159265

int main(const int argc, const char *const *const argv){
	try{
		int i = 0;
		kohonen koh;
		float input[kohonen::numInput * kohonen::inputSize] = { 53.214, 19.155, 52.349, 18.924, //53.214, 19.105, 52.349, 18.924,//
			53.560, 19.221, 49.763, 19.076,
			53.280, 19.167, 52.758, 18.790,
			53.111, 19.220, 52.988, 18.920,
			54.200, 19.210, 53.719, 19.194,
			54.390, 19.185, 53.684, 19.031,
			53.848, 19.079, 54.058, 19.269, //53.948, 19.079, 54.058, 19.269, //
			53.721, 19.170, 54.076, 19.399, //53.711, 19.070, 54.076, 19.399,//
			53.603, 19.027, 54.390, 19.399, //53.503, 18.927, 54.390, 19.399,//
			53.494, 18.969, 49.763, 18.790 };
		
		float inputX[kohonen::numInput * kohonen::inputSize / 2];
		float iX = 0, iY = 0;
		float inputY[kohonen::numInput * kohonen::inputSize / 2];
		for (i = 0; i < kohonen::numInput * kohonen::inputSize;i++){
			if ((i % 2) == 0){
				inputX[i / 2] = input[i];
				iX = iX + input[i];
			}
			else{
				inputY[i / 2] = input[i];
				iY = iY + input[i];
			}
		}
		iX = iX / 20;
		iY = iY / 20;
		float maxInputX = *std::max_element(inputX, inputX + kohonen::numInput * kohonen::inputSize/2);
		float minInputX = *std::min_element(inputX, inputX + kohonen::numInput * kohonen::inputSize/2);
		float maxInputY = *std::max_element(inputY, inputY + kohonen::numInput * kohonen::inputSize/2);
		float minInputY = *std::min_element(inputY, inputY + kohonen::numInput * kohonen::inputSize/2);
		//printf("minX : %f   . minY : %f    . maxX : %f    . maxy : %f", minInputX,minInputY,maxInputX,maxInputY);
		float inputaux[kohonen::numInput * kohonen::inputSize] = { 0 };
		for (i = 0; i < kohonen::numInput*kohonen::inputSize;i++){
			if ((i % 2) == 0){
				//input[i] = (input[i] - minInputX) / (maxInputX - minInputX);
				input[i] = (input[i] - (minInputX + maxInputX)/2)*1.5 / (maxInputX - minInputX);
				//input[i] = (input[i] - iX/2) / (maxInputX - minInputX);
				//printf("------>%f    ,      ", input[i]);
			}
			else{
				//input[i] = (input[i] - minInputY) / (maxInputY - minInputY);
				input[i] = (input[i] - (minInputY + maxInputY)/2)*1.5 / (maxInputY - minInputY);
				//input[i] = (input[i] - iY/2) / (maxInputY - minInputY);
				//printf("%f\n", input[i]);
			}
		}
		ofstream fout("input.txt");
		if (fout.is_open())
		{
			for (i = 0; i < kohonen::numInput; i++)
			{
				fout << input[i*kohonen::dimension] << "   " << input[i*kohonen::dimension + 1] << endl; //writing ith character of array in the file
			}
		}
		fout.close();
		//360 entre numero de puntos distancia entre angulos. 1º cos angulo y 2º sen angulo
		/*float map[kohonen::mapSize][kohonen::dimension] = { { 1, 0 }, { 0.966, 0.259 }, { 0.866, 0.5 }, { 0.707, 0.707 }, { 0.5, 0.866 }, { 0.259, 0.966 },
		{ 0, 1 }, { -0.259, 0.966 }, { -0.5, 0.866 }, { -0.707, 0.707 }, { -0.866, 0.5 }, { -0.966, 0.259 },
		{ -1, 0 }, { -0.966, -0.259 }, { -0.866, -0.5 }, { -0.707, -0.707 }, { -0.5, -0.866 }, { -0.259, -0.966 },
		{ 0, -1 }, { 0.259, -0.966 }, { 0.5, -0.866 }, { 0.707, -0.707 }, { 0.866, -0.5 }, { 0.966, -0.259 } };
		float weight[kohonen::mapSize*kohonen::inputSize] = { 0 };*///rand() / ((double) RAND_MAX);

		float map[kohonen::numInput*kohonen::dimension * 3];
		float angulo = 360 / (kohonen::numInput * 3);
		printf("%f", angulo);
		
		
		for (i = 0; i < kohonen::numInput * 3; i++){
			map[i*kohonen::dimension] = cos((angulo*i)* PI / 180.0);
			printf("nodo = %d  angulo = %f   x = %f   ,   ", i, i*angulo, map[i*kohonen::dimension]);
			map[i*kohonen::dimension + 1] = sin((angulo*i)* PI / 180);
			printf("y = %f\n", map[i*kohonen::dimension+1]);
			
		}
		ofstream fout2("mapOrig.txt");
		if (fout2.is_open())
		{
			for (i = 0; i < kohonen::numInput * 3; i++)
			{
				fout2 << map[i*kohonen::dimension] << "   " << map[i*kohonen::dimension + 1]<<endl; //writing ith character of array in the file
			}
		}
		fout2.close();
		system("PAUSE");
		//float weight[kohonen::numInput*kohonen::dimension * 3]={0};
		/*for (i = 0; i < kohonen::mapSize*kohonen::inputSize; i++){
			weight[i] = rand() / ((float)RAND_MAX);
			printf("%f\n", weight[i]);
		}*/
		DWORD dw1 = GetTickCount();
		cudaError_t cudaStatus = koh.train(kohonen::inputSize, kohonen::mapSize, kohonen::numInput, input, map, maxInputX, minInputX, maxInputY, minInputY);
		DWORD dw2 = GetTickCount();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
		printf("Time difference is %d miliseconds\n", (dw2 - dw1));
		system("PAUSE");
		
		for (i = 0; i < kohonen::numInput * 3; i++){
			printf("nodo = %d     x = %f   ,",i, map[i*kohonen::dimension]);
			printf("    y = %f\n", map[i*kohonen::dimension+1]);
		}
		ofstream fout3("mapAfter.txt");
		if (fout3.is_open())
		{
			for (i = 0; i < kohonen::numInput * 3; i++)
			{
				fout3 << map[i*kohonen::dimension] << "   " << map[i*kohonen::dimension + 1] << endl; //writing ith character of array in the file
			}
		}
		fout3.close();
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