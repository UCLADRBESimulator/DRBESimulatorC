#pragma once

#define Debug 1

#if  Debug == 1
#define De
#elif defined(PR)
#else
#define De
#endif //  PR_DEBUG


class GeometryEngine
{
};

class RCS : GeometryEngine {
public:
	double Wavelength;
	int Order;
	double K; //number of point

//Point
	double No;

	double NoAngTheta;
	double NoAngPhi;

	double NoBw1;
	double NoBw2;

	double Sigma1;
	double Sigma2;


};

