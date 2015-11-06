
#include <cuda.h>
#include <time.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <exception>
#include <algorithm>
#include <fstream>
#include <windows.h>
#include <iostream>
#include <sstream>

#include "kohonen.h"
#include "cuda_runtime.h"


using namespace std;

#define PI 3.14159265

int main(const int argc, const char *const *const argv){
	bool inputGood = false;
	int numEpoch1=0, numEpoch2=0, numEpoch3=0, timesMap=0;
	string fileName;
	string pathName;
	std::ifstream pathIn("path");
	getline(pathIn, pathName);
	float eta1, eta2;

	while (!inputGood){
		printf("Introduzca el nombre del archivo que contiene el mapa : \n");
		getline(cin, fileName);
		std::ifstream infile0(pathName+"\\"+fileName);
		if (infile0){
			inputGood = true;
		}
		
	}
	inputGood = false;
	while (!inputGood){
		printf("Introduzca el numero de epocas para la primera fase de aprendizaje : \n");
		if (cin >> numEpoch1){
			inputGood = true;
		}
		else{
			cin.clear();
			std::cin.ignore(256, '\n');
		}
	}
	inputGood = false;
	while (!inputGood){
		printf("Introduzca el numero de epocas para la segunda fase de aprendizaje : \n");
		if (cin >> numEpoch2){
			inputGood = true;
		}
		else{
			cin.clear();
			std::cin.ignore(256, '\n');
		}
	}
	inputGood = false;
	while (!inputGood){
		printf("Introduzca el numero a partir del cual la segunda fase se convierte en la tercera : \n");
		printf("Para no lanzar segunda fase poner a 0\n");
		if (cin >> numEpoch3){
			inputGood = true;
		}
		else{
			cin.clear();
			std::cin.ignore(256, '\n');
		}
	}
	inputGood = false;
	while (!inputGood && timesMap<1){
		printf("Introduzca el numero de veces mas grande que tiene que ser el mapa con respecto a la entrada : \n");
		printf("Necesariamnte mayor que 0.\n");
		if (cin >> timesMap){
			inputGood = true;
		}
		else{
			cin.clear();
			std::cin.ignore(256, '\n');
		}
	}
	inputGood = false;
	while (!inputGood){
		printf("Introduzca el valor de aprendizaje para las dos primeras fases de aprendizaje : \n");
		if (cin >> eta1){
			inputGood = true;
		}
		else{
			cin.clear();
			std::cin.ignore(256, '\n');
		}
	}
	inputGood = false;
	while (!inputGood){
		printf("Introduzca el valor de aprendizaje para la tercera fase de aprendizaje : \n");
		if (cin >> eta2){
			inputGood = true;
		}
		else{
			cin.clear();
			std::cin.ignore(256, '\n');
		}
	}
	DWORD time = GetTickCount();
	string path = pathName + "\\" + std::to_string(time) + "-" + fileName + "-" + std::to_string(numEpoch1) + "-" + std::to_string(numEpoch2) + "-" + std::to_string(numEpoch3) + "-" + std::to_string(timesMap) + "-" + std::to_string(eta1) + "-" + std::to_string(eta2);
	LPCSTR lpMyString = path.c_str();
	
	CreateDirectory(lpMyString ,NULL);
	try{
		int i = 0;
		kohonen koh;
		//int kohonen::numInput = 20;
		int numberOfInputs = 0;
		std::ifstream infile(pathName + "\\" + fileName);
		
		float xCoordinate, yCoordinate;
		while (infile >> xCoordinate >> yCoordinate)
		{
			numberOfInputs++;
			// process pair (a,b)
		}
		float* input = new float[numberOfInputs * kohonen::inputSize];
		int index = 0;
		std::ifstream infile2(pathName + "\\" + fileName);
		
		while (infile2 >> xCoordinate >> yCoordinate)
		{
			// process pair (a,b)
			input[index * kohonen::inputSize] = xCoordinate;
			input[index * kohonen::inputSize + 1] = yCoordinate;
			index++;
		}

		float *inputX = new float[numberOfInputs];
		float iX = 0, iY = 0;
		float *inputY = new float[numberOfInputs];
		for (i = 0; i < numberOfInputs * kohonen::inputSize; i++){
			if ((i % 2) == 0){
				inputX[i / 2] = input[i];
				iX = iX + input[i];
			}
			else{
				inputY[i / 2] = input[i];
				iY = iY + input[i];
			}
		}
		iX = iX / numberOfInputs;
		iY = iY / numberOfInputs;
		float maxInputX = *std::max_element(inputX, inputX + numberOfInputs);
		float minInputX = *std::min_element(inputX, inputX + numberOfInputs);
		float maxInputY = *std::max_element(inputY, inputY + numberOfInputs);
		float minInputY = *std::min_element(inputY, inputY + numberOfInputs);
		float *inputaux = new float[numberOfInputs * kohonen::inputSize] { 0 };
		for (i = 0; i < numberOfInputs*kohonen::inputSize; i++){
			if ((i % 2) == 0){
				input[i] = (input[i] - (minInputX + maxInputX)/2)*1.5 / (maxInputX - minInputX);
			}
			else{
				input[i] = (input[i] - (minInputY + maxInputY)/2)*1.5 / (maxInputY - minInputY);
			}
		}
		
		ofstream fout(path+"\\"+ "input.txt");
		if (fout.is_open())
		{
			for (i = 0; i < numberOfInputs; i++)
			{
				fout << input[i*kohonen::inputSize] << "   " << input[i*kohonen::inputSize + 1] << endl; //writing ith character of array in the file
			}
		}
		fout.close();


		float *map = new float[numberOfInputs*kohonen::inputSize * timesMap];
		float angulo = 360 / (numberOfInputs * timesMap);
		
		
		for (i = 0; i < numberOfInputs * timesMap; i++){
			map[i*kohonen::inputSize] = cos((angulo*i)* PI / 180.0);
			map[i*kohonen::inputSize + 1] = sin((angulo*i)* PI / 180);
			
		}
		ofstream fout2(path + "\\" + "mapOrig.txt");
		if (fout2.is_open())
		{
			for (i = 0; i < numberOfInputs * timesMap; i++)
			{
				fout2 << map[i*kohonen::inputSize] << "   " << map[i*kohonen::inputSize + 1] << endl; //writing ith character of array in the file
			}
		}
		fout2.close();

		DWORD dw1 = GetTickCount();
		cudaError_t cudaStatus = koh.train(kohonen::inputSize, numberOfInputs * timesMap, numberOfInputs, input, map, maxInputX, minInputX, maxInputY, minInputY,timesMap,numEpoch1,numEpoch2,numEpoch3,eta1,eta2);
		DWORD dw2 = GetTickCount();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
		printf("Time difference is %d miliseconds\n", (dw2 - dw1));

		ofstream fout3(path + "\\" + "mapAfter.txt");
		if (fout3.is_open())
		{
			for (i = 0; i < numberOfInputs * 3; i++)
			{
				fout3 << map[i*kohonen::inputSize] << "   " << map[i*kohonen::inputSize + 1] << endl; //writing ith character of array in the file
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