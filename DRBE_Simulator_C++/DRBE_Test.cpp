#include "DRBE_Test.h"
#include <iostream>
using namespace std;





int DRBE_Test::Test_return(int &b)
{
	int a = b;

	cout << a;
	
	return 2;
}