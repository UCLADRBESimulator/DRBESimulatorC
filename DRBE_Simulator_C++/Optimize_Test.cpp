#include "Optimize_Test.h"
#include <iostream> 
#include <vector> 
#include <chrono> 
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std::chrono;
using namespace std;


vector<double*> Optimize_Test::Matrix_generator(int m1, int n1, int m2, int n2)
{
	int i = 0;

	/* initialize random seed: */
	srand(time(NULL));
	auto start = high_resolution_clock::now();

	int sizeA = m1 * n1;
	int sizeB = m2 * n2;
	int sizeC = n1 * n2;

	double* matrixA = new double[sizeA];
	double* matrixB = new double[sizeB];


	while (i < sizeA)
	{
		matrixA[i] = rand();
		i++;
	}
	
	i = 0;
	while (i < sizeB)
	{
		matrixB[i] = rand();
		i++;
	}




	vector<double*> a = { matrixA, matrixB };



	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Finished: " <<duration.count() << endl;
	return a;
}

double* Matrix_Multiplication_V1(int m1, int n1, double* M1, int m2, int n2, double* M2)
{
	/* initialize random seed: */
	srand(time(NULL));
	auto start = high_resolution_clock::now();



	int size1 = m1 * n1;
	int size2 = m2 * n2;
	int size3 = n1 * n2;

	double* matrixC = new double[size3];




	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Finished: " << duration.count() << endl;
	return matrixC;
}