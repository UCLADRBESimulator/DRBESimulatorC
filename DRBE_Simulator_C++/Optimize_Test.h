#include <iostream> 
#include <vector> 
#include <chrono> 

using namespace std;
#pragma once
class Optimize_Test
{
public:

	vector<double*> Matrix_generator(int m1, int n1, int m2, int n2);
	double* Matrix_Multiplication_V1(int m1, int n1, double* M1, int m2, int n2, double* M2);
	double* Matrix_Multiplication_V2(int m1, int n1, double* M1, int m2, int n2, double* M2);
	double* V1_test_flow(int m1, int n1, int m2, int n2);

	//void Write_txt_file()
};