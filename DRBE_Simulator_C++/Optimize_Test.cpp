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

	double* matrixA = new double[sizeA];
	double* matrixB = new double[sizeB];


	while (i < sizeA)
	{
		//matrixA[i] = rand();
		matrixA[i] = i;
		i++;
	}

	i = 0;
	while (i < sizeB)
	{
		//matrixB[i] = rand();
		matrixB[i] = i;
		i++;
	}




	vector<double*> a = { matrixA, matrixB };



	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Finished: " << duration.count() << endl;
	return a;
}

double* Optimize_Test::Matrix_Multiplication_V1(int m1, int n1, double* Ma1, int m2, int n2, double* Ma2)
{
	/* initialize random seed: */
	srand(time(NULL));
	auto start = high_resolution_clock::now();



	int size1 = m1 * n1;
	int size2 = m2 * n2;
	int size3 = n1 * m2;
	int i = 0;
	int ii = 0;
	int iii = 0;
	int iiii = 0;
	double* matrixC = new double[size3];
	double temp = 0;
	i = 0;
	while (i < n1)
	{
		ii = 0;
		while (ii < m2)
		{
			iii = 0;
			iiii = 0;
			while (iii < n2)
			{
				//cout << (*(Ma1 + i * m1 + iiii)) << " ,  " << (*(Ma2 + ii + iii * m2)) << endl;
				temp += (*(Ma1 + i * m1 + iiii)) * (*(Ma2 + ii + iii * m2));


				iii++;
				iiii++;
			}

			matrixC[i * m2 + ii] = temp;
			temp = 0;
			ii++;
		}
		i++;
	}


	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Finished: " << duration.count() << endl;
	i = 0;
	ii = 0;
	while (i < m2)
	{
		ii = 0;
		while (ii < n1)
		{
			//cout << matrixC[i * n1 + ii] << " , ";
			ii++;
		}
		//cout << endl;
		i++;
	}
	return matrixC;
}

double* Optimize_Test::Matrix_Multiplication_V2(int m1, int n1, double* Ma1, int m2, int n2, double* Ma2)
{
	/* initialize random seed: */
	srand(time(NULL));
	auto start = high_resolution_clock::now();



	int size1 = m1 * n1;
	int size2 = m2 * n2;
	int size3 = n1 * m2;

	int sizel1 = 4;
	int sizejump = m1 / sizel1;
	int sizemod = m1 % sizel1;
	double m1array[4] = { 0,0,0,0 };




	int i = 0;
	int ii = 0;
	int iii = 0;
	int iiii = 0;
	double* matrixC = new double[size3];
	double temp = 0;
	i = 0;
	while (i < size1)
	{
		ii = 0;
		while (ii < m2)
		{
			iii = 0;
			iiii = 0;
			while (iii < n2)
			{
				//cout << (*(Ma1 + i * m1 + iiii)) << " ,  " << (*(Ma2 + ii + iii * m2)) << endl;
				temp += (*(Ma1 + i * m1 + iiii)) * (*(Ma2 + ii + iii * m2));


				iii++;
				iiii++;
			}

			matrixC[i * m2 + ii] = temp;
			temp = 0;
			ii++;
		}
		i++;
	}


	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Finished: " << duration.count() << endl;
	i = 0;
	ii = 0;
	while (i < m2)
	{
		ii = 0;
		while (ii < n1)
		{
			//cout << matrixC[i * n1 + ii] << " , ";
			ii++;
		}
		//cout << endl;
		i++;
	}
	return matrixC;
}

double* Optimize_Test::V1_test_flow(int m1, int n1, int m2, int n2)
{
	double* result;
	vector<double*> temp = { 0,0 };
	temp = Matrix_generator(m1, n1, m2, n2);

	result = Matrix_Multiplication_V1(m1, n1, temp[0], m2, n2, temp[1]);
	return result;
}