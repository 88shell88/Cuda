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
#include <sstream>
#include "cuda_runtime.h"
#include <vector>
using namespace std;

#define PI 3.14159265

int main(const int argc, const char *const *const argv){
	bool input = false;
	string st;
	while (!input){
		printf("Introduzca el nombre del archivo que contiene el mapa : \n");
		getline(cin,st);
		std::ifstream infile0(st);
		if (infile0){
			input = true;
		}
	}
	try{
		int i = 0;
		kohonen koh;
		//int kohonen::numInput = 20;
		const int numIn = kohonen::numInput;
		int num = 0;
		std::ifstream infile(st);
		float a, b;
		while (infile >> a >> b)
		{
			num++;
			// process pair (a,b)
		}
		float* input = new float[num*2];
		int index = 0;
		printf("numInputs = %d\n", num);
		std::ifstream infile2(st);
		
		while (infile2 >> a >> b)
		{
			printf("inputs = %f   %f\n", a, b);
			// process pair (a,b)
			input[index*2] = a;
			input[index*2+1] = b;
			index++;
		}
		printf("**index = %d   inputs = %f   %f\n",index, input[36], input[37]);

		system("PAUSE");
		float *inputX = new float[num];
		float iX = 0, iY = 0;
		float *inputY = new float[num];
		for (i = 0; i < num * kohonen::inputSize;i++){
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
		float inputaux[kohonen::numInput * kohonen::inputSize] = { 0 };
		for (i = 0; i < kohonen::numInput*kohonen::inputSize;i++){
			if ((i % 2) == 0){
				input[i] = (input[i] - (minInputX + maxInputX)/2)*1.5 / (maxInputX - minInputX);
			}
			else{
				input[i] = (input[i] - (minInputY + maxInputY)/2)*1.5 / (maxInputY - minInputY);
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

		DWORD dw1 = GetTickCount();
		cudaError_t cudaStatus = koh.train(kohonen::inputSize, num*3, kohonen::numInput, input, map, maxInputX, minInputX, maxInputY, minInputY);
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

	}
	catch (std::exception &e){
		printf("error");
	}
	return 0;
}