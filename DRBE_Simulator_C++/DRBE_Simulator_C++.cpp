// DRBE_Simulator_C++.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream> 
#include <vector> 
#include <chrono> 
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>
#include "DRBE_Test.h"
#include "Optimize_Test.h"
#include <Eigen/Dense>
#include <iomanip>
#include <complex>
#include <cmath>
#include <valarray>
#include <unsupported/Eigen/MatrixFunctions>


#define PI 3.14159265358979323846
#define PI2 3.14159265358979323846*2

using namespace std;
using namespace Eigen;
DRBE_Test drbe;

Optimize_Test ot;



typedef complex<double> Complex;
typedef valarray<Complex> CArray;

// From http://rosettacode.org/wiki/Fast_Fourier_transform

// Cooley-Tukey FFT (in-place, breadth-first, decimation-in-frequency)
// Better optimized but less intuitive
// !!! Warning : in some cases this code make result different from not optimased version above (need to fix bug)
// The bug is now fixed @2017/05/30 

void four1(double* data, unsigned long nn)
{
    unsigned long n, mmax, m, j, istep, i;
    double wtemp, wr, wpr, wpi, wi, theta;
    double tempr, tempi;

    // reverse-binary reindexing
    n = nn << 1;
    j = 1;
    for (i = 1; i < n; i += 2) {
        if (j > i) {
            swap(data[j - 1], data[i - 1]);
            swap(data[j], data[i]);
        }
        m = nn;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    };

    // here begins the Danielson-Lanczos section
    mmax = 2;
    while (n > mmax) {
        istep = mmax << 1;
        theta = -(2 * PI / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        for (m = 1; m < mmax; m += 2) {
            for (i = m; i <= n; i += istep) {
                j = i + mmax;
                tempr = wr * data[j - 1] - wi * data[j];
                tempi = wr * data[j] + wi * data[j - 1];

                data[j - 1] = data[i - 1] - tempr;
                data[j] = data[i] - tempi;
                data[i - 1] += tempr;
                data[i] += tempi;
            }
            wtemp = wr;
            wr += wr * wpr - wi * wpi;
            wi += wi * wpr + wtemp * wpi;
        }
        mmax = istep;
    }
}

void fft(CArray& x)
{
    // DFT
    unsigned int N = x.size(), k = N, n;

    double thetaT = 3.14159265358979323846264338328L / N;
    Complex phiT = Complex(cos(thetaT), -sin(thetaT)), T;

    while (k > 1)
    {
        n = k;
        k >>= 1;
        phiT = phiT * phiT;
        T = 1.0L;
        for (unsigned int l = 0; l < k; l++)
        {
            for (unsigned int a = l; a < N; a += n)
            {
                unsigned int b = a + k;
                cout << a << " , " << b << "|" << x.size() << endl;
                Complex t = x[a] - x[b];
                x[a] += x[b];
                x[b] = t * T;
            }
            T *= phiT;
        }
    }
    return;
    // Decimate
    unsigned int m = (unsigned int)log2(N);
    for (unsigned int a = 0; a < N; a++)
    {
        unsigned int b = a;
        // Reverse bits
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        if (b > a)
        {
            Complex t = x[a];
            x[a] = x[b];
            x[b] = t;
        }
    }
    //// Normalize (This section make it not working correctly)
    //Complex f = 1.0 / sqrt(N);
    //for (unsigned int i = 0; i < N; i++)
    //    x[i] *= f;
}

// inverse fft (in-place)
void ifft(CArray& x)
{
    // conjugate the complex numbers
    x = x.apply(std::conj);

    // forward fft
    fft(x);

    // conjugate the complex numbers again
    x = x.apply(std::conj);

    // scale the numbers
    x /= x.size();
}

CArray mul_poly(CArray& x, CArray& y) {
    int n = 1;
    int deg = x.size() + y.size();
    while (n < deg) {
        n = n << 1;
    }
    CArray padX(n);
    CArray padY(n);
    CArray z(n);
    for (int i = 0; i < x.size(); i++) {
        padX[i] = x[i];
    }
    for (int i = 0; i < y.size(); i++) {
        padY[i] = y[i];
    }
    fft(padX);
    fft(padY);
    z = padX * padY;
    ifft(z);
    return z;
}

CArray sdp(CArray& x, CArray& y) {
    int n = x.size();
    int m = y.size();
    CArray revY(m);
    for (int i = 0; i < m; i++) {
        revY[i] = y[m - i - 1];
    }
    CArray z = mul_poly(x, revY);
    return CArray(z[slice(m - 1, n - m + 1, 1)]);
}

vector<int> int_sdp(vector<int> x, vector<int> y) {
    CArray cx(x.size());
    CArray cy(y.size());
    for (int i = 0; i < x.size(); i++) {
        cx[i] = Complex(x[i], 0);
    }
    for (int i = 0; i < y.size(); i++) {
        cy[i] = Complex(y[i], 0);
    }
    CArray cz = sdp(cx, cy);
    vector<int> z(cz.size());
    for (int i = 0; i < z.size(); i++) {
        z[i] = int(cz[i].real() + 0.5);
    }
    return z;
}


vector<complex<double>> dftc(vector<complex<double>> X)
{
    int N = X.size();
    int K = N;
    double PI_N = PI2 / N;

    complex<double> sum;
    vector<complex<double>> output;

    for (int k = 0; k < K; k++)
    {
        sum = complex<double>(0, 0);
        for (int n = 0; n < N; n++)
        {
            double real = cos(PI_N * k * n);
            double imag = -sin(PI_N * k * n);
            complex<double> w(real, imag);
            sum += X[n] * w;
        }
        output.push_back(sum);
    }
    return output;
}

vector<complex<double>> idftc(vector<complex<double>> X)
{
    int N = X.size();
    int K = N;
    double PI_N = PI2 / N;

    complex<double> CN = complex<double>(N, 0);
    complex<double> sum;
    vector<complex<double>> output;

    for (int k = 0; k < K; k++)
    {
        sum = complex<double>(0, 0);
        for (int n = 0; n < N; n++)
        {
            double real = cos(PI_N * k * n);
            double imag = sin(PI_N * k * n);
            complex<double> w(real, imag);
            sum += X[n] * w;
        }
        sum = sum / CN;
        output.push_back(sum);
    }
    return output;
}


struct RCS_out
{
    vector<VectorXd> Sigma;
    vector<double> range;
};

class RCS_parameter {
public:
    double Wavelength;
    int Fidelity_order;
    int Number_of_points;
    int Number_of_plate;
    double Incident_elevation_angle;
    double Incident_azimuth_angle;
    double Reflection_elevation_angle;
    double Reflection_azimuth_angle;
    double Reflection_old_theta;
    double Reflection_old_phi;
    VectorXd Old_reflection_vector;
    VectorXd Scatter_PDF;
    int Scatter_PDF_index;
    double Length;
    double Delta_width;

    
    RCS_parameter() 
        : Wavelength( 30e-3 ), Fidelity_order( 0 ), Number_of_points( 4 ), Number_of_plate( 1 ), Incident_elevation_angle( 90.0 ),
        Incident_azimuth_angle( 315.0 ), Reflection_elevation_angle( 90.0 ), Reflection_azimuth_angle( 45.0 )
        , Reflection_old_theta(80), Reflection_old_phi(45), Scatter_PDF_index(1), Length(0.1)
    {
        
        Old_reflection_vector = VectorXd(3);
        Old_reflection_vector << sin(Reflection_old_theta / 180 * PI) * cos(Reflection_old_phi / 180 * PI), sin(Reflection_old_theta / 180 * PI)* sin(Reflection_old_phi / 180 * PI), cos(Reflection_old_theta / 180 * PI);
        Scatter_PDF = VectorXd(10);
        Scatter_PDF << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1;
        Delta_width = cos(Wavelength / (2 * Length));

    }

    void RCS_default()
    {
        Old_reflection_vector = VectorXd(3);
        Old_reflection_vector << sin(Reflection_old_theta / 180 * PI) * cos(Reflection_old_phi / 180 * PI), sin(Reflection_old_theta / 180 * PI)* sin(Reflection_old_phi / 180 * PI), cos(Reflection_old_theta / 180 * PI);
    
    }

    RCS_parameter(double _wavelength, int _fidelity_order, int _number_of_point, int _number_of_plate)
        :Wavelength{ _wavelength }, Fidelity_order{ _fidelity_order }, Number_of_points{ _number_of_point }, Number_of_plate{ _number_of_plate }
    {
    }



};


class RCS_plate : public RCS_parameter
{
public:
    vector<VectorXd> pPosition;
    vector<VectorXd> pOrientation;
    vector<VectorXd> pBandwidth;
    vector<VectorXd> pSigma;
    vector<VectorXd> pNorm;
    int Numberofplate;

    RCS_plate()
    {
    }
};

class RCS_calculation : public RCS_parameter
{
public:
    vector<VectorXd> Position;
    vector<VectorXd> Orientation;
    vector<VectorXd> Bandwidth;
    vector<VectorXd> Sigma;
    vector<VectorXd> Norm;
    vector<VectorXd> BWArray;

    RCS_out rcs_out;


    vector<VectorXd> pPosition;
    vector<VectorXd> pOrientation;
    vector<VectorXd> pBandwidth;
    vector<VectorXd> pSigma;
    vector<VectorXd> pNorm;


    VectorXd temp;
    RCS_calculation()
    {

        temp = VectorXd(3);
        temp << 0, 0, 0;
        Position.push_back(temp);
        temp << 1, 0, 0.5;
        Position.push_back(temp);
        temp << 0, 1, 1;
        Position.push_back(temp);
        temp << -1, 0, 0;
        Position.push_back(temp);

        temp = VectorXd(2);
        temp << 90, 0;
        Orientation.push_back(temp);
        temp << 90, 90;
        Orientation.push_back(temp);
        temp << 90, 0;
        Orientation.push_back(temp);
        temp << 0, 45;
        Orientation.push_back(temp);

        temp << 45, 135;
        Bandwidth.push_back(temp);
        temp << 60, 45;
        Bandwidth.push_back(temp);
        temp << 60, 25;
        Bandwidth.push_back(temp);
        temp << 40, 40;
        Bandwidth.push_back(temp);


        temp << 0.2, 0.1;
        Sigma.push_back(temp);
        temp << 0.3, 0.2;
        Sigma.push_back(temp);
        temp << 0.3, 0.1;
        Sigma.push_back(temp);
        temp << 0.2, 0.1;
        Sigma.push_back(temp);
        
        int i = 0;
        while (i < Number_of_points)
        {
            temp = VectorXd(3);
            temp << sin(Orientation[i](0) / 180 * PI) * cos(Orientation[i](1) / 180 * PI), sin(Orientation[i](0) / 180 * PI)* sin(Orientation[i](1) / 180 * PI), cos(Orientation[i](0) / 180 * PI);
            Norm.push_back(temp);

            temp = VectorXd(2);
            temp << cos(Bandwidth[i](0) / 180 * PI), cos(Bandwidth[i](0) / 180 * PI + Bandwidth[i](1) / 180 * PI);
            BWArray.push_back(temp);

            i++;
        }

        temp = VectorXd(3);
        temp << 1, 1, 1;
        pPosition.push_back(temp);

        temp = VectorXd(2);
        temp << 90, 10;
        pOrientation.push_back(temp);

        temp << 10, 0;
        pBandwidth.push_back(temp);


        temp << 0.3, 0;
        pSigma.push_back(temp);

        i = 0;
        while (i < Number_of_plate)
        {
            temp = VectorXd(3);
            temp << sin(pOrientation[i](0) / 180 * PI) * cos(pOrientation[i](1) / 180 * PI), sin(pOrientation[i](0) / 180 * PI)* sin(pOrientation[i](1) / 180 * PI), cos(pOrientation[i](0) / 180 * PI);
            pNorm.push_back(temp);

            i++;
        }
    }

    void Set_parameter()
    {
        int i = 0;

        Old_reflection_vector = VectorXd(3);
        Old_reflection_vector << sin(Reflection_old_theta / 180 * PI) * cos(Reflection_old_phi / 180 * PI), sin(Reflection_old_theta / 180 * PI)* sin(Reflection_old_phi / 180 * PI), cos(Reflection_old_theta / 180 * PI);
        Delta_width = cos(Wavelength / (2 * Length));

        while (i < Number_of_points)
        {
            temp = VectorXd(3);
            temp << sin(Orientation[i](0) / 180 * PI) * cos(Orientation[i](1) / 180 * PI), sin(Orientation[i](0) / 180 * PI)* sin(Orientation[i](1) / 180 * PI), cos(Orientation[i](0) / 180 * PI);
            Norm.push_back(temp);

            temp = VectorXd(2);
            temp << cos(Bandwidth[i](0) / 180 * PI), cos(Bandwidth[i](0) / 180 * PI + Bandwidth[i](1) / 180 * PI);
            BWArray.push_back(temp);

            i++;
        }

        i = 0;
        while (i < Number_of_plate)
        {
            temp = VectorXd(3);
            temp << sin(pOrientation[i](0) / 180 * PI) * cos(pOrientation[i](1) / 180 * PI), sin(pOrientation[i](0) / 180 * PI)* sin(pOrientation[i](1) / 180 * PI), cos(pOrientation[i](0) / 180 * PI);
            pNorm.push_back(temp);

            i++;
        }

    }

    RCS_out& Order0()
    {
        VectorXd temp(1);
        rcs_out.Sigma.clear();

        temp << Sigma[0](0);
        rcs_out.Sigma.push_back(temp);
        rcs_out.range.clear();
        rcs_out.range.push_back(0);

        return rcs_out;

    }

    RCS_out& Order1()
    {
        VectorXd nr(3);
        VectorXd temp(1);
        nr << sin(Reflection_elevation_angle / 180 * PI) * cos(Reflection_azimuth_angle / 180 * PI), sin(Reflection_elevation_angle / 180 * PI)* sin(Reflection_azimuth_angle / 180 * PI), cos(Reflection_elevation_angle / 180 * PI);
        rcs_out.Sigma.clear();
        rcs_out.range.clear();
        rcs_out.range.push_back(0);
        Scatter_PDF_index = 0;
        if (nr.dot(Old_reflection_vector) < Delta_width)
        {

            Scatter_PDF_index++;
            temp << Scatter_PDF[Scatter_PDF_index];
            rcs_out.Sigma.push_back(temp);
            Old_reflection_vector = nr;
        }
        else
        {
            temp << Scatter_PDF[Scatter_PDF_index];
            rcs_out.Sigma.push_back(temp);
        }

        return rcs_out;
    }

    RCS_out& Order2()
    {
        int i = 0;
        rcs_out.Sigma.clear();
        rcs_out.range.clear();
        VectorXd nt(3); nt << sin(Reflection_elevation_angle / 180 * PI) * cos(Reflection_azimuth_angle / 180 * PI), sin(Reflection_elevation_angle / 180 * PI)* sin(Reflection_azimuth_angle / 180 * PI), cos(Reflection_elevation_angle / 180 * PI);
        VectorXd nr(3); nr << sin(Incident_elevation_angle / 180 * PI) * cos(Incident_azimuth_angle / 180 * PI), sin(Incident_elevation_angle / 180 * PI)* sin(Incident_azimuth_angle / 180 * PI), cos(Incident_elevation_angle / 180 * PI);
        VectorXd nn(3); nn = nt + nr;
        VectorXd temp(1);

        while (i < Number_of_points)
        {
            rcs_out.range.push_back(-Position[i].dot(nn));

            temp << Sigma[i](0);
            cout << temp << endl;
            rcs_out.Sigma.push_back(temp);
            i++;
        }
        
        return rcs_out;

    }

    RCS_out& Order3()
    {
        int i = 0;
        rcs_out.Sigma.clear();
        rcs_out.range.clear();
        VectorXd nr(3); nr << sin(Reflection_elevation_angle / 180 * PI) * cos(Reflection_azimuth_angle / 180 * PI), sin(Reflection_elevation_angle / 180 * PI)* sin(Reflection_azimuth_angle / 180 * PI), cos(Reflection_elevation_angle / 180 * PI);
        VectorXd nt(3); nt << sin(Incident_elevation_angle / 180 * PI) * cos(Incident_azimuth_angle / 180 * PI), sin(Incident_elevation_angle / 180 * PI)* sin(Incident_azimuth_angle / 180 * PI), cos(Incident_elevation_angle / 180 * PI);
        VectorXd nn(3); nn = nt + nr;

        VectorXd Sigmat(Number_of_points);
        VectorXd Sigmar(Number_of_points);
        i = 0;
        double dtemp = 0;

        while (i < Number_of_points)
        {
            rcs_out.range.push_back(-Position[i].dot(nn));

            dtemp = nt.dot(Norm[i]);
            if (dtemp > BWArray[i](0))
            {
                Sigmat(i) = Sigma[i](0);
            }
            else
            {
                Sigmat(i) = 0;
            }
            cout << "T: " << nt.dot(Norm[i]) << endl;
            dtemp = nr.dot(Norm[i]);

            if (dtemp > BWArray[i](0))
            {
                Sigmar(i) = Sigma[i](0);
            }
            else
            {
                Sigmar(i) = 0;
            }
            cout << "R: " << Sigmar[i] << endl;
            i++;
        }
        rcs_out.Sigma.push_back(Sigmat.cwiseProduct(Sigmar));

        return rcs_out;

    }

    RCS_out& Order4()
    {
        int i = 0;
        rcs_out.Sigma.clear();
        rcs_out.range.clear();
        VectorXd nr(3); nr << sin(Reflection_elevation_angle / 180 * PI) * cos(Reflection_azimuth_angle / 180 * PI), sin(Reflection_elevation_angle / 180 * PI)* sin(Reflection_azimuth_angle / 180 * PI), cos(Reflection_elevation_angle / 180 * PI);
        VectorXd nt(3); nt << sin(Incident_elevation_angle / 180 * PI) * cos(Incident_azimuth_angle / 180 * PI), sin(Incident_elevation_angle / 180 * PI)* sin(Incident_azimuth_angle / 180 * PI), cos(Incident_elevation_angle / 180 * PI);
        VectorXd nn(3); nn = nt + nr;

        cout << "Check 1";

        VectorXd Sigmat(Number_of_points);
        VectorXd Sigmar(Number_of_points);
        i = 0;
        double dtemp;
        while (i < Number_of_points)
        {
            cout << "Check 2";
            rcs_out.range.push_back( -Position[i].dot(nn));

            dtemp = nt.dot(Norm[i]);

            if (dtemp > BWArray[i](0))
            {
                Sigmat(i) = Sigma[i](0);
            }
            else if (dtemp > BWArray[i](1))
            {
                Sigmat(i) = Sigma[i](1);
            }
            else
            {
                Sigmat(i) = 0;
            }

            cout << "Check 3";
            dtemp = nr.dot(Norm[i]);

            if (dtemp > BWArray[i](0))
            {
                Sigmar(i) = Sigma[i](0);
            }
            else if (dtemp > BWArray[i](1))
            {
                Sigmar(i) = Sigma[i](1);
            }
            else
            {
                Sigmar(i) = 0;
            }
            i++;
        }
        rcs_out.Sigma.push_back(Sigmat.cwiseProduct(Sigmar));

        return rcs_out;

    }

    RCS_out& Order5()
    {
        int i = 0;
        rcs_out.Sigma.clear();
        rcs_out.range.clear();
        VectorXd nr(3); nr << sin(Reflection_elevation_angle / 180 * PI) * cos(Reflection_azimuth_angle / 180 * PI), sin(Reflection_elevation_angle / 180 * PI)* sin(Reflection_azimuth_angle / 180 * PI), cos(Reflection_elevation_angle / 180 * PI);
        VectorXd nt(3); nt << sin(Incident_elevation_angle / 180 * PI) * cos(Incident_azimuth_angle / 180 * PI), sin(Incident_elevation_angle / 180 * PI)* sin(Incident_azimuth_angle / 180 * PI), cos(Incident_elevation_angle / 180 * PI);
        VectorXd nn(3); nn = nt + nr;

        VectorXd Sigmat(Number_of_points);
        VectorXd Sigmar(Number_of_points);
        i = 0;
        double dtemp;
        while (i < Number_of_points)
        {
            rcs_out.range.push_back(-Position[i].dot(nn));

            dtemp = nt.dot(Norm[i]);

            if (dtemp > BWArray[i](0))
            {
                Sigmat[i] = Sigma[i](0);
            }
            else if (dtemp > BWArray[i](1))
            {
                Sigmat[i] = Sigma[i](1);
            }
            else
            {
                Sigmat[i] = 0;
            }
            dtemp = nr.dot(Norm[i]);

            if (dtemp > BWArray[i](0))
            {
                Sigmar[i] = Sigma[i](0);
            }
            else if (dtemp > BWArray[i](1))
            {
                Sigmar[i] = Sigma[i](1);
            }
            else
            {
                Sigmar[i] = 0;
            }
            i++;
        }
        rcs_out.Sigma.push_back(Sigmat.cwiseProduct(Sigmar));

        VectorXd Sigmap(Number_of_plate);
        VectorXd ns(Number_of_plate);
        double pdtemp;
        
        i = 0;
        while (i < Number_of_plate)
        {
            rcs_out.range.push_back(-pPosition[i].dot(nn));
            ns = (pNorm[i] * (2 * pNorm[i].dot(nt))) - nt;
            pdtemp = pow((ns.dot(nr)), pBandwidth[i](0));
            if (pdtemp > 0)
            {
                Sigmap[i] = pdtemp * pSigma[i](0);
            }
            else
            {
                Sigmap[i] = 0;
            }



            i++;
        }
        rcs_out.Sigma.push_back(Sigmap);
        return rcs_out;

    }

    void RCS_scatter_default()
    {


    }
};

struct Antenna_out
{
    double Gain;
    vector<double> Gain_vec;
};

class Antenna_parameter {
public:
    double unity_gain;
    int no_antennas; // for 4th order
    int angle_res;
    int no_bins; // following values are arbitraryand they actually depend on antenna characteristics.Some values are to be taken from a look - up table
    int secondary_fidelity;
    double tx_mainbeam_gain_value;
    double rx_mainbeam_gain_value;
    double tx_sidelobe_gain_value;
    double rx_sidelobe_gain_value;

    double tx_azimuth_beamwidth;
    double tx_elevation_beamwidth;
    double rx_azimuth_beamwidth;
    double rx_elevation_beamwidth;

    double tx_azimuth_steering_angle;
    double tx_elevation_steering_angle;
    double rx_azimuth_steering_angle;
    double rx_elevation_steering_angle;

    double tx_angle_azimuth;
    double tx_angle_elevation;
    double rx_angle_azimuth;
    double rx_angle_elevation;

    Antenna_out antenna_out;

    Antenna_parameter()
        : unity_gain(1), no_antennas(64),angle_res(1), no_bins(180),secondary_fidelity(1),
        tx_mainbeam_gain_value(10),rx_mainbeam_gain_value(20),tx_sidelobe_gain_value(1),rx_sidelobe_gain_value(2),
        tx_azimuth_beamwidth(10),tx_elevation_beamwidth(10),rx_azimuth_beamwidth(10),rx_elevation_beamwidth(10),
        tx_azimuth_steering_angle(0),tx_elevation_steering_angle(0),rx_azimuth_steering_angle(0),rx_elevation_steering_angle(0),
        tx_angle_azimuth(50), tx_angle_elevation(50), rx_angle_azimuth(30), rx_angle_elevation(30)
    {

    }

    Antenna_out& Order0(vector<int>& IDv)
    {
        double TxGain = unity_gain;
        double RxGain = unity_gain;
        antenna_out.Gain_vec.clear();
        antenna_out.Gain = TxGain * RxGain;

        int i = 0;
        while (i < IDv.size())
        {
            antenna_out.Gain_vec.push_back(TxGain * RxGain);
            i++;
        }
        return antenna_out;
    }

    Antenna_out& Order1(vector<int>& IDv)
    {
        double TxGain = tx_sidelobe_gain_value;
        double RxGain = rx_sidelobe_gain_value;
        antenna_out.Gain_vec.clear();
        antenna_out.Gain = TxGain * RxGain;

        int i = 0;
        while (i < IDv.size())
        {
            antenna_out.Gain_vec.push_back(TxGain * RxGain);
            i++;
        }
        return antenna_out;
    }

    Antenna_out& Order2(vector<int>& IDv, vector<double>& TxAziAng, vector<double>& TxEleAng, vector<double>& RxAziAng, vector<double>& RxEleAng)
    {
        double TxGain = 0;
        double RxGain = 0;
        antenna_out.Gain_vec.clear();
        antenna_out.Gain =0;

        if (((tx_angle_azimuth - tx_azimuth_steering_angle) < tx_azimuth_beamwidth / 2) && ((tx_angle_elevation - tx_elevation_steering_angle) < tx_elevation_beamwidth / 2))
        {
            TxGain = tx_mainbeam_gain_value;
        }
        else
        {
            TxGain = unity_gain;
        }

        if (((rx_angle_azimuth - rx_azimuth_steering_angle) < rx_azimuth_beamwidth / 2) && ((rx_angle_elevation - rx_elevation_steering_angle) < rx_elevation_beamwidth / 2))
        {
            RxGain = rx_mainbeam_gain_value;
        }
        else
        {
            RxGain = unity_gain;
        }
        antenna_out.Gain = TxGain * RxGain;

        int i = 0;
        while (i < IDv.size())
        {
            if (((TxAziAng[i] - tx_azimuth_steering_angle) < tx_azimuth_beamwidth / 2) && ((TxEleAng[i] - tx_elevation_steering_angle) < tx_elevation_beamwidth / 2))
            {
                TxGain = tx_mainbeam_gain_value;
            }
            else
            {
                TxGain = unity_gain;
            }
            if (((RxAziAng[i] - rx_azimuth_steering_angle) < rx_azimuth_beamwidth / 2) && ((RxEleAng[i] - rx_elevation_steering_angle) < rx_elevation_beamwidth / 2))
            {
                RxGain = rx_mainbeam_gain_value;
            }
            else
            {
                RxGain = unity_gain;
            }
            i++;
            antenna_out.Gain_vec.push_back(TxGain * RxGain);
        }
        return antenna_out;
    }

    Antenna_out& Order3(vector<int>& IDv, vector<double>& TxAziAng, vector<double>& TxEleAng, vector<double>& RxAziAng, vector<double>& RxEleAng)
    {
        double TxGain = 0;
        double RxGain = 0;
        antenna_out.Gain_vec.clear();
        antenna_out.Gain = 0;

        if (((tx_angle_azimuth - tx_azimuth_steering_angle) < tx_azimuth_beamwidth / 2) && ((tx_angle_elevation - tx_elevation_steering_angle) < tx_elevation_beamwidth / 2))
        {
            TxGain = tx_mainbeam_gain_value;
        }
        else
        {
            TxGain = tx_sidelobe_gain_value;
        }

        if (((rx_angle_azimuth - rx_azimuth_steering_angle) < rx_azimuth_beamwidth / 2) && ((rx_angle_elevation - rx_elevation_steering_angle) < rx_elevation_beamwidth / 2))
        {
            RxGain = rx_mainbeam_gain_value;
        }
        else
        {
            RxGain = rx_sidelobe_gain_value;
        }
        antenna_out.Gain = TxGain * RxGain;

        int i = 0;
        while (i < IDv.size())
        {
            if (((TxAziAng[i] - tx_azimuth_steering_angle) < tx_azimuth_beamwidth / 2) && ((TxEleAng[i] - tx_elevation_steering_angle) < tx_elevation_beamwidth / 2))
            {
                TxGain = tx_mainbeam_gain_value;
            }
            else
            {
                TxGain = tx_sidelobe_gain_value;
            }
            if (((RxAziAng[i] - rx_azimuth_steering_angle) < rx_azimuth_beamwidth / 2) && ((RxEleAng[i] - rx_elevation_steering_angle) < rx_elevation_beamwidth / 2))
            {
                RxGain = rx_mainbeam_gain_value;
            }
            else
            {
                RxGain = rx_sidelobe_gain_value;
            }
            i++;
            antenna_out.Gain_vec.push_back(TxGain * RxGain);
        }
        return antenna_out;
    }

    Antenna_out& Order4(vector<int>& IDv, vector<double>& TxAziAng, vector<double>& TxEleAng, vector<double>& RxAziAng, vector<double>& RxEleAng, int SecFidel)
    {
        int i = 0;
        int i1 = 0;
        int i2 = 0;
        double TxGain = 0;
        double RxGain = 0;
        antenna_out.Gain_vec.clear();
        antenna_out.Gain = 0;
        vector<double> angle_vec;
        vector<double> window_vec;

        vector<double> tx_sectored_beam_azimuth(no_antennas, 0);
        vector<double> tx_sectored_beam_elevation(no_antennas, 0);
        vector<double> rx_sectored_beam_azimuth(no_antennas, 0);
        vector<double> rx_sectored_beam_elevation(no_antennas, 0);
        
        vector<double> k;
        vector<double> psi;
        vector<double> n;
        
        VectorXd mk = VectorXd(no_antennas);
        VectorXd mpsi = VectorXd(no_antennas);
        VectorXd mn = VectorXd(no_antennas);

        int zeropad_size = pow(2, nextpow2n(no_bins)) - no_antennas + 2;
        int a_zero_totalsize = zeropad_size + no_antennas - 1;


        vector<double> zeroadd(pow(2,nextpow2n(no_bins))-no_antennas + 2);

        VectorXcd mtx_sectored_beam_azimuth = VectorXcd(no_antennas).setZero();
        VectorXcd mtx_sectored_beam_elevation = VectorXcd(no_antennas).setZero();
        VectorXcd mrx_sectored_beam_azimuth = VectorXcd(no_antennas).setZero();
        VectorXcd mrx_sectored_beam_elevation = VectorXcd(no_antennas).setZero();






        VectorXcd mtx_azimuth_awv_domain = VectorXcd(no_antennas);
        VectorXcd mtx_elevation_awv_domain = VectorXcd(no_antennas);
        VectorXcd mrx_azimuth_awv_domain = VectorXcd(no_antennas);
        VectorXcd mrx_elevation_awv_domain = VectorXcd(no_antennas);

        vector<complex<double>> tx_azimuth_awv_domain;
        vector<complex<double>> tx_elevation_awv_domain;
        vector<complex<double>> rx_azimuth_awv_domain;
        vector<complex<double>> rx_elevation_awv_domain;

        VectorXcd mawn_tx_azimuth = VectorXcd(no_antennas);
        VectorXcd mawn_tx_elevation = VectorXcd(no_antennas);
        VectorXcd mawn_rx_azimuth = VectorXcd(no_antennas);
        VectorXcd mawn_rx_elevation = VectorXcd(no_antennas);

        vector<complex<double>> awn_tx_azimuth;
        vector<complex<double>> awn_tx_elevation;
        vector<complex<double>> awn_rx_azimuth;
        vector<complex<double>> awn_rx_elevation;

        VectorXcd max_tx_azimuth = VectorXcd(no_antennas-1);
        VectorXcd max_tx_elevation = VectorXcd(no_antennas-1);
        VectorXcd max_rx_azimuth = VectorXcd(no_antennas-1);
        VectorXcd max_rx_elevation = VectorXcd(no_antennas-1);

        vector<complex<double>> ax_tx_azimuth;
        vector<complex<double>> ax_tx_elevation;
        vector<complex<double>> ax_rx_azimuth;
        vector<complex<double>> ax_rx_elevation;

        VectorXcd ma_tx_azimuth = VectorXcd(a_zero_totalsize);
        VectorXcd ma_tx_elevation = VectorXcd(a_zero_totalsize);
        VectorXcd ma_rx_azimuth = VectorXcd(a_zero_totalsize);
        VectorXcd ma_rx_elevation = VectorXcd(a_zero_totalsize);

        MatrixXcd ma_temp = MatrixXcd(no_antennas, no_antennas);


        vector<complex<double>> awk_tx_azimuth;
        vector<complex<double>> awk_tx_elevation;
        vector<complex<double>> awk_rx_azimuth;
        vector<complex<double>> awk_rx_elevation;

        double ATest = pow(2, nextpow2n(no_bins));
        double new_angleres = 181 / pow(2, nextpow2n(no_bins));
        

        double angle_incre = 180.0 / (no_antennas - 1);
        double tx_thetamin = tx_azimuth_steering_angle - tx_azimuth_beamwidth / 2; // tx azimuth
        double tx_thetamax = tx_azimuth_steering_angle + tx_azimuth_beamwidth / 2;
        double tx_phimin = tx_elevation_steering_angle - tx_elevation_beamwidth / 2; // tx elevation
        double tx_phimax = tx_elevation_steering_angle + tx_elevation_beamwidth / 2;

        double rx_thetamin = rx_azimuth_steering_angle - rx_azimuth_beamwidth / 2; // rx azimuth
        double rx_thetamax = rx_azimuth_steering_angle + rx_azimuth_beamwidth / 2;
        double rx_phimin = rx_elevation_steering_angle - rx_elevation_beamwidth / 2; // rx elevation
        double rx_phimax = rx_elevation_steering_angle + rx_elevation_beamwidth / 2;
        int alt_az = 1 - no_antennas % 2;
        




        i = 0;
        while (i < no_antennas)
        {
            angle_vec.push_back(angle_incre * i);
            window_vec.push_back(0.42 - 0.5 * cos(2 * PI * i / no_antennas) + 0.08 * cos(4 * PI * i / no_antennas));

            if ((angle_vec[i] < tx_thetamax) && (angle_vec[i] > tx_thetamin))
            {
                tx_sectored_beam_azimuth[i] = tx_mainbeam_gain_value;
                mtx_sectored_beam_azimuth(i) = tx_mainbeam_gain_value;
            }
            else
            {
                if (SecFidel == 2)
                {
                    mtx_sectored_beam_azimuth(i) = unity_gain;
                    tx_sectored_beam_azimuth[i] = unity_gain;
                }
                else if (SecFidel == 3)
                {
                    mtx_sectored_beam_azimuth(i) = tx_sidelobe_gain_value;
                    tx_sectored_beam_azimuth[i] = tx_sidelobe_gain_value;
                }

            }
            if ((angle_vec[i] < tx_phimax) && (angle_vec[i] > tx_phimin))
            {
                tx_sectored_beam_elevation[i] = tx_mainbeam_gain_value;
                mtx_sectored_beam_elevation(i) = tx_mainbeam_gain_value;
            }
            else
            {
                if (SecFidel == 2)
                {
                    mtx_sectored_beam_elevation(i) = unity_gain;
                    tx_sectored_beam_elevation[i] = unity_gain;
                }
                else if (SecFidel == 3)
                {
                    mtx_sectored_beam_elevation(i) = tx_sidelobe_gain_value;
                    tx_sectored_beam_elevation[i] = tx_sidelobe_gain_value;
                }

            }
            if ((angle_vec[i] < rx_thetamax) && (angle_vec[i] > rx_thetamin))
            {
                rx_sectored_beam_azimuth[i] = rx_mainbeam_gain_value;
                mrx_sectored_beam_azimuth(i) = rx_mainbeam_gain_value;
            }
            else
            {
                if (SecFidel == 2)
                {
                    mrx_sectored_beam_azimuth(i) = unity_gain;
                    rx_sectored_beam_azimuth[i] = unity_gain;
                }
                else if (SecFidel == 3)
                {
                    mrx_sectored_beam_azimuth(i) = rx_sidelobe_gain_value;
                    rx_sectored_beam_azimuth[i] = rx_sidelobe_gain_value;
                }

            }
            if ((angle_vec[i] < rx_phimax) && (angle_vec[i] > rx_phimin))
            {
                rx_sectored_beam_elevation[i] = rx_mainbeam_gain_value;
                mrx_sectored_beam_elevation(i) = rx_mainbeam_gain_value;
            }
            else
            {
                if (SecFidel == 2)
                {
                    mrx_sectored_beam_elevation(i) = unity_gain;
                    rx_sectored_beam_elevation[i] = unity_gain;
                }
                else if (SecFidel == 3)
                {
                    mrx_sectored_beam_elevation(i) = rx_sidelobe_gain_value;
                    rx_sectored_beam_elevation[i] = rx_sidelobe_gain_value;
                }

            }




            k.push_back(i - alt_az * (no_antennas - 1) / 2.0); //DFT index
            psi.push_back(2 * PI * k[i] / no_antennas);
            n.push_back(i - (no_antennas - 1) / 2.0);

            mk(i) = i - alt_az * (no_antennas - 1) / 2.0;
            mpsi(i) = 2 * PI * mk(i) / no_antennas;
            mn(i) = i - (no_antennas - 1) / 2.0;



            tx_azimuth_awv_domain.push_back(tx_sectored_beam_azimuth[i] * exp(-1i * conj(psi[i]) * n[i]));
            tx_elevation_awv_domain.push_back(tx_sectored_beam_elevation[i] * exp(-1i * conj(psi[i]) * n[i]));
            rx_azimuth_awv_domain.push_back(rx_sectored_beam_azimuth[i] * exp(-1i * conj(psi[i]) * n[i]));
            rx_elevation_awv_domain.push_back(rx_sectored_beam_elevation[i] * exp(-1i * conj(psi[i]) * n[i]));

            awn_tx_azimuth.push_back(tx_azimuth_awv_domain[i] * window_vec[i]);
            awn_tx_elevation.push_back(tx_elevation_awv_domain[i] * window_vec[i]);
            awn_rx_azimuth.push_back(rx_azimuth_awv_domain[i] * window_vec[i]);
            awn_rx_elevation.push_back(rx_elevation_awv_domain[i] * window_vec[i]);

            i++;




        }


        ma_temp = -1i * (mpsi * mn.transpose());
        i1 = 0;
        while (i1 < no_antennas)
        {
            i2 = 0;
            while (i2 < no_antennas)
            {
                ma_temp(i1, i2) = exp(ma_temp(i1, i2));
                i2++;
            }
            i1++;
        }




        mtx_azimuth_awv_domain = mtx_sectored_beam_azimuth.transpose() * ma_temp;
        mtx_elevation_awv_domain = mtx_sectored_beam_elevation.transpose() * ma_temp;
        mrx_azimuth_awv_domain = mrx_sectored_beam_azimuth.transpose() * ma_temp;
        mrx_elevation_awv_domain = mrx_sectored_beam_elevation.transpose() * ma_temp;

        i = 0;
        while (i < no_antennas)
        {
            mawn_tx_azimuth(i) = window_vec[i] * mtx_azimuth_awv_domain(i);
            mawn_tx_elevation(i) = window_vec[i] * mtx_elevation_awv_domain(i);
            mawn_rx_azimuth(i) = window_vec[i] * mrx_azimuth_awv_domain(i);
            mawn_rx_elevation(i) = window_vec[i] * mrx_elevation_awv_domain(i);
            i++;
        }

        i = 0;
        while (i < no_antennas - 1)
        {
            ax_tx_azimuth.push_back(exp(-1i * PI * complex<double>(i,0))* awn_tx_azimuth[i]);
            ax_tx_elevation.push_back(exp(-1i * PI * complex<double>(i, 0))* awn_tx_elevation[i]);
            ax_rx_azimuth.push_back(exp(-1i * PI * complex<double>(i, 0))* awn_rx_azimuth[i]);
            ax_rx_elevation.push_back(exp(-1i * PI * complex<double>(i, 0))* awn_rx_elevation[i]);


            max_tx_azimuth(i) = (exp(-1i * PI * complex<double>(i, 0))* mawn_tx_azimuth(i));
            max_tx_elevation(i) = (exp(-1i * PI * complex<double>(i, 0))* mawn_tx_elevation(i));
            max_rx_azimuth(i) = (exp(-1i * PI * complex<double>(i, 0))* mawn_rx_azimuth(i));
            max_rx_elevation(i) = (exp(-1i * PI * complex<double>(i, 0))* mawn_rx_elevation(i));


           

            i++;
        }



        complex<double> maxte = complex<double>(0, 0);
        complex<double> maxta = complex<double>(0, 0);
        complex<double> maxre = complex<double>(0, 0);
        complex<double> maxra = complex<double>(0, 0);
        int tempsize = no_antennas - 1;
        i = 0;
        while (i < tempsize)
        {
            awk_tx_azimuth.push_back(max_tx_azimuth(i));
            awk_tx_elevation.push_back(max_tx_elevation(i));
            awk_rx_azimuth.push_back(max_rx_azimuth(i));
            awk_rx_elevation.push_back(max_rx_elevation(i));

            i++;
        }

        while (i < a_zero_totalsize - 1)
        {
            awk_tx_azimuth.push_back(complex<double>(0.0, 0.0));
            awk_tx_elevation.push_back(complex<double>(0.0, 0.0));
            awk_rx_azimuth.push_back(complex<double>(0.0, 0.0));
            awk_rx_elevation.push_back(complex<double>(0.0, 0.0));
            i++;
        }


        complex<double> ta0 = awk_tx_azimuth[0];
        complex<double> te0 = awk_tx_elevation[0];
        complex<double> ra0 = awk_rx_azimuth[0];
        complex<double> re0 = awk_rx_elevation[0];
        awk_tx_azimuth = idftc(awk_tx_azimuth);
        awk_tx_elevation = idftc(awk_tx_elevation);
        awk_rx_azimuth = idftc(awk_rx_azimuth);
        awk_rx_elevation = idftc(awk_rx_elevation);


        awk_tx_azimuth.push_back(awk_tx_azimuth[0]);
        awk_tx_elevation.push_back(awk_tx_elevation[0]);
        awk_rx_azimuth.push_back(awk_rx_azimuth[0]);
        awk_rx_elevation.push_back(awk_rx_elevation[0]);



        i = 0;
        while (i < a_zero_totalsize)
        {
            if (abs(maxte) < abs(awk_tx_elevation[i]))
            {
                maxte = awk_tx_elevation[i];
            }
            awk_tx_elevation[i] += te0;

            if (abs(maxta) < abs(awk_tx_azimuth[i]))
            {
                maxta = awk_tx_azimuth[i];
            }
            awk_tx_azimuth[i] += ta0;

            if (abs(maxre) < abs(awk_rx_elevation[i]))
            {
                maxre = awk_rx_elevation[i];
            }
            awk_rx_elevation[i] += re0;

            if (abs(maxra) < abs(awk_rx_azimuth[i]))
            {
                maxra = awk_rx_azimuth[i];
            }
            awk_rx_azimuth[i] += ra0;
            i++;
        }

      

       

        i = 0;
        while (i < a_zero_totalsize)
        {
            awk_tx_azimuth[i] = tx_mainbeam_gain_value * awk_tx_azimuth[i] / maxta;
            awk_tx_elevation[i] = tx_mainbeam_gain_value * awk_tx_elevation[i] / maxte;
            awk_rx_azimuth[i] = rx_mainbeam_gain_value * awk_rx_azimuth[i] / maxra;
            awk_rx_elevation[i] = rx_mainbeam_gain_value * awk_rx_elevation[i] / maxre;

            i++;
        }

        

        antenna_out.Gain = abs(awk_tx_azimuth[ceil(tx_angle_azimuth/new_angleres)-1]) * abs(awk_tx_elevation[ceil(tx_angle_elevation / new_angleres)-1])
            * abs(awk_rx_azimuth[ceil(rx_angle_azimuth / new_angleres)-1]) * abs(awk_rx_elevation[ceil(rx_angle_elevation / new_angleres)-1]);


        antenna_out.Gain_vec.clear();
        i = 0;
        while (i < IDv.size())
        {
            antenna_out.Gain_vec.push_back(abs(awk_tx_azimuth[ceil(TxAziAng[i] / new_angleres) - 1]) * abs(awk_tx_elevation[ceil(TxEleAng[i] / new_angleres) - 1])
                * abs(awk_rx_azimuth[ceil(RxAziAng[i] / new_angleres) - 1]) * abs(awk_rx_elevation[ceil(RxEleAng[i] / new_angleres) - 1]));
            i++;
        }
        return antenna_out;
    }

    vector<double>& nextpow2v(vector<double>& darray)
    {
        int i = 0;
        int presult = 0;
        vector<double> result;
        while (i < darray.size())
        {
            while (pow(2, presult) < darray[i])
            {
                presult++;
            }
            result.push_back(presult);
            i++;
        }
        return result;
    }
    double nextpow2n(double darray)
    {
        int presult = 0;
        
        while (pow(2, presult) < darray)
        {
            presult++;
        }

        return presult;
    }
};



int main()
{

    // mode - fidelity - payload

    // rcs - (1-5) - wavelength - num_point - num_plat - p_pos - p_orin - p_bw - p_sig - pl_pos - pl_orin - pl_bw - pl_sig - inci_ele - inci_azi - ref_ele - ref_azi - O_theta - O_phi - O_ref_vec - PDF - PDFi - s_L - del_W



    RCS_calculation rcs_scatter;
    Antenna_parameter ant_parameter;
    RCS_out rcs_result;
    Antenna_out ant_result;
    int i = 0;
    int ii = 0;
    int mode = 0;
    int fidelity = 0;
    double wavelength = 0;
    int num_point = 0;
    int num_plate = 0;
    int PDF_size = 0;
    double dtemp1 = 0;
    double dtemp2 = 0;
    double dtemp3 = 0;

    int a_fidelity = 0;
    int a_numberofobj = 0;
    int at_ids = 0;
    double at_dtemp = 0;
    vector<int> obj_ids;
    vector<double> rxAzAngle;
    vector<double> rxElAngle;
    vector<double> txAzAngle;
    vector<double> txElAngle;
    int a_sec_fidel = 1;

    VectorXd temp;


    obj_ids.push_back(2);
    obj_ids.push_back(10);
    obj_ids.push_back(8);
    obj_ids.push_back(9);
    obj_ids.push_back(1);
    obj_ids.push_back(5);
    obj_ids.push_back(7);
    obj_ids.push_back(6);
    obj_ids.push_back(3);
    obj_ids.push_back(4);

    i = 0;
    while (i < 64)
    {
        rxAzAngle.push_back(50 + i * 80/9);
        i++;
    }

    i = 0;
    while (i < 64)
    {
        rxElAngle.push_back(130 - i * 80 / 9);
        i++;
    }

    i = 0;
    while (i < 64)
    {
        txAzAngle.push_back(10 + i * 140 / 9);
        i++;
    }

    i = 0;
    while (i < 64)
    {
        txElAngle.push_back(150 - i * 140 / 9);
        i++;
    }

    //    Antenna_out& Order4(vector<int>& IDv, vector<double>& TxAziAng, vector<double>& TxEleAng, vector<double>& RxAziAng, vector<double>& RxEleAng, vector<int>& SecFidel)


    ant_result = ant_parameter.Order4(obj_ids, txAzAngle, txElAngle, rxAzAngle, rxElAngle, a_sec_fidel);
    cout << "Result out: " << endl;
    cout << "Antenna direct gain:  ";
    cout << fixed << "{" << ant_result.Gain << "}";
    cout << endl;
    cout << "Antenna gain vector [s" << ant_result.Gain_vec.size() << "] :  ";

    cout << endl;
    cout << endl;
    i = 0;
    while (i < ant_result.Gain_vec.size())
    {
        cout << fixed << "{" << ant_result.Gain_vec[i] << "}";
        i++;
    }
    cout << endl;


    return 0;


    i = 0;
    while (true)
    {
        cin >> mode;
        if (mode == 1)
        {
            cin >> fidelity;
            cin >> wavelength;
            cin >> num_point;
            cin >> num_plate;
            rcs_scatter.Number_of_plate = num_plate;
            rcs_scatter.Number_of_points = num_point;
            rcs_scatter.Wavelength = wavelength;
            rcs_scatter.Position.clear();
            i = 0;
            //
            while (i < num_point)
            {
                temp = VectorXd(3);
                cin >> dtemp1;
                cin >> dtemp2;
                cin >> dtemp3;
                temp << dtemp1, dtemp2, dtemp3;
                rcs_scatter.Position.push_back(temp);
                i++;
            }

            i = 0;
            while (i < num_point)
            {
                temp = VectorXd(2);
                cin >> dtemp1;
                cin >> dtemp2;
                temp << dtemp1, dtemp2;
                rcs_scatter.Orientation.push_back(temp);
                i++;
            }

            i = 0;
            while (i < num_point)
            {
                temp = VectorXd(2);
                cin >> dtemp1;
                cin >> dtemp2;
                temp << dtemp1, dtemp2;
                rcs_scatter.Bandwidth.push_back(temp);
                i++;
            }

            i = 0;
            while (i < num_point)
            {
                temp = VectorXd(2);
                cin >> dtemp1;
                cin >> dtemp2;
                temp << dtemp1, dtemp2;
                rcs_scatter.Sigma.push_back(temp);
                i++;
            }

            i = 0;
            while (i < num_plate)
            {
                temp = VectorXd(3);
                cin >> dtemp1;
                cin >> dtemp2;
                cin >> dtemp3;
                temp << dtemp1, dtemp2, dtemp3;
                rcs_scatter.pPosition.push_back(temp);
                i++;
            }

            i = 0;
            while (i < num_plate)
            {
                temp = VectorXd(2);
                cin >> dtemp1;
                cin >> dtemp2;
                temp << dtemp1, dtemp2;
                rcs_scatter.pOrientation.push_back(temp);
                i++;
            }

            i = 0;
            while (i < num_plate)
            {
                temp = VectorXd(2);
                cin >> dtemp1;
                cin >> dtemp2;
                temp << dtemp1, dtemp2;
                rcs_scatter.pBandwidth.push_back(temp);
                i++;
            }

            i = 0;
            while (i < num_plate)
            {
                temp = VectorXd(2);
                cin >> dtemp1;
                cin >> dtemp2;
                temp << dtemp1, dtemp2;
                rcs_scatter.pSigma.push_back(temp);
                i++;
            }

            cout << "finish load1";
            // rcs - (1-5) - wavelength - num_point - num_plat - p_pos - p_orin - p_bw - p_sig - pl_pos - pl_orin - pl_bw - pl_sig - inci_ele - inci_azi - ref_ele - ref_azi - O_theta - O_phi - PDF_size - PDF - PDFi - s_L

            cin >> rcs_scatter.Incident_elevation_angle;
            cin >> rcs_scatter.Incident_azimuth_angle;
            cin >> rcs_scatter.Reflection_elevation_angle;
            cin >> rcs_scatter.Reflection_azimuth_angle;
            cin >> rcs_scatter.Reflection_old_theta;
            cin >> rcs_scatter.Reflection_old_phi;
            cin >> PDF_size;

            rcs_scatter.Scatter_PDF = VectorXd(PDF_size);
            cout << "finish load2";
            cout << PDF_size;
            i = 0;
            while (i < PDF_size)
            {
                cin >> dtemp1;
                cout << dtemp1;
                rcs_scatter.Scatter_PDF(i) = dtemp1;

                i++;
            }
            cin >> rcs_scatter.Scatter_PDF_index;
            cin >> rcs_scatter.Length;
            cout << "finish load3";
            rcs_scatter.Set_parameter();

            std::stringstream toprint;


            switch (fidelity) {
            case 0:
                rcs_result = rcs_scatter.Order0();
                cout << "Result out: " << endl;
                cout << "RCS Range [r" << rcs_result.range.size() << "] :  ";
                i = 0;
                while (i < rcs_result.range.size())
                {
                    cout << fixed << "{" << rcs_result.range[i] << "}";
                    i++;
                }
                cout << endl;
                cout << "RCS Sigma [s" << rcs_result.Sigma.size() << "] :  ";

                i = 0;
                while (i < rcs_result.Sigma.size())
                {
                    ii = 0;
                    while (ii < rcs_result.Sigma[i].size())
                    {
                        cout << fixed << "{" << rcs_result.Sigma[i](ii) << "}";
                        ii++;
                    }
                    i++;
                }
                cout << endl;
                break;
            case 1:
                rcs_result = rcs_scatter.Order1();
                cout << "Result out: " << endl;
                cout << "RCS Range [r" << rcs_result.range.size() << "] :  ";
                i = 0;
                while (i < rcs_result.range.size())
                {
                    cout << fixed << "{" << rcs_result.range[i] << "}";
                    i++;
                }
                cout << endl;
                cout << "RCS Sigma [s" << rcs_result.Sigma.size() << "] :  ";

                i = 0;
                while (i < rcs_result.Sigma.size())
                {
                    ii = 0;
                    while (ii < rcs_result.Sigma[i].size())
                    {
                        cout << fixed << "{" << rcs_result.Sigma[i](ii) << "}";
                        ii++;
                    }
                    i++;
                }
                cout << endl;
                break;
            case 2:
                rcs_result = rcs_scatter.Order2();
                cout << "Result out: " << endl;
                cout << "RCS Range [r" << rcs_result.range.size() << "] :  ";
                i = 0;
                while (i < rcs_result.range.size())
                {
                    cout << fixed << "{" << rcs_result.range[i] << "}";
                    i++;
                }
                cout << endl;
                cout << "RCS Sigma [s" << rcs_result.Sigma.size() << "] :  ";

                i = 0;
                while (i < rcs_result.Sigma.size())
                {
                    ii = 0;
                    while (ii < rcs_result.Sigma[i].size())
                    {
                        cout << fixed << "{" << rcs_result.Sigma[i](ii) << "}";
                        ii++;
                    }
                    i++;
                }
                cout << endl;
                break;
            case 3:
                rcs_result = rcs_scatter.Order3();
                cout << "Result out: " << endl;
                cout << "RCS Range [r" << rcs_result.range.size() << "] :  ";
                i = 0;
                while (i < rcs_result.range.size())
                {
                    cout << fixed << "{" << rcs_result.range[i] << "}";
                    i++;
                }
                cout << endl;
                cout << "RCS Sigma [s" << rcs_result.Sigma.size() << "] :  ";

                i = 0;
                while (i < rcs_result.Sigma.size())
                {
                    ii = 0;
                    while (ii < rcs_result.Sigma[i].size())
                    {
                        cout << fixed << "{" << rcs_result.Sigma[i](ii) << "}";
                        ii++;
                    }
                    i++;
                }
                cout << endl;
                break;
            case 4:
                rcs_result = rcs_scatter.Order4();
                cout << "Result out: " << endl;
                cout << "RCS Range [r" << rcs_result.range.size() << "] :  ";
                i = 0;
                while (i < rcs_result.range.size())
                {
                    cout << fixed << "{" << rcs_result.range[i] << "}";
                    i++;
                }
                cout << endl;
                cout << "RCS Sigma [s" << rcs_result.Sigma.size() << "] :  ";

                i = 0;
                while (i < rcs_result.Sigma.size())
                {
                    ii = 0;
                    while (ii < rcs_result.Sigma[i].size())
                    {
                        cout << fixed << "{" << rcs_result.Sigma[i](ii) << "}";
                        ii++;
                    }
                    i++;
                }
                cout << endl;
                break;
            case 5:
                rcs_result = rcs_scatter.Order5();
                cout << "Result out: " << endl;
                cout << "RCS Range [r" << rcs_result.range.size() << "] :  ";
                i = 0;
                while (i < rcs_result.range.size())
                {
                    cout << fixed << "{" << rcs_result.range[i] << "}";
                    i++;
                }
                cout << endl;
                cout << "RCS Sigma [s" << rcs_result.Sigma.size() << "] :  ";

                i = 0;
                while (i < rcs_result.Sigma.size())
                {
                    ii = 0;
                    while (ii < rcs_result.Sigma[i].size())
                    {
                        cout << fixed << "{" << rcs_result.Sigma[i](ii) << "}";
                        ii++;
                    }
                    i++;
                }
                cout << endl;
                break;
            default:
                cout << "Invalid grade" << endl;
            }
        }
        else if (mode == 2)
        {
            cin >> fidelity;
            cin >> a_numberofobj;
            i = 0;
            while (i < a_numberofobj)
            {
                cin >> at_ids;
                obj_ids.push_back(at_ids);
                i++;
            }
            i = 0;
            while (i < a_numberofobj)
            {
                cin >> at_dtemp;
                txAzAngle.push_back(at_dtemp);
                i++;
            }
            i = 0;
            while (i < a_numberofobj)
            {
                cin >> at_dtemp;
                txElAngle.push_back(at_dtemp);
                i++;
            }
            i = 0;
            while (i < a_numberofobj)
            {
                cin >> at_dtemp;
                rxAzAngle.push_back(at_dtemp);
                i++;
            }
            i = 0;
            while (i < a_numberofobj)
            {
                cin >> at_dtemp;
                rxElAngle.push_back(at_dtemp);
                i++;
            }
            cin >> at_ids;
            switch (fidelity) {
            case 0:
                ant_result = ant_parameter.Order0(obj_ids);
                cout << "Result out: " << endl;
                cout << "Antenna direct gain:  ";
                cout << fixed << "{" << ant_result.Gain << "}";
                cout << endl;
                cout << "Antenna gain vector [s" << ant_result.Gain_vec.size() << "] :  ";

                i = 0;
                while (i < ant_result.Gain_vec.size())
                {
                    cout << fixed << "{" << ant_result.Gain_vec[i] << "}";
                    i++;
                }
                cout << endl;
                break;
            case 1:
                ant_result = ant_parameter.Order1(obj_ids);
                cout << "Result out: " << endl;
                cout << "Antenna direct gain:  ";
                cout << fixed << "{" << ant_result.Gain << "}";
                cout << endl;
                cout << "Antenna gain vector [s" << ant_result.Gain_vec.size() << "] :  ";

                i = 0;
                while (i < ant_result.Gain_vec.size())
                {
                    cout << fixed << "{" << ant_result.Gain_vec[i] << "}";
                    i++;
                }
                cout << endl;
                break;
            case 2:
                ant_result = ant_parameter.Order2(obj_ids, txAzAngle, txElAngle, rxAzAngle, rxElAngle);
                cout << "Result out: " << endl;
                cout << "Antenna direct gain:  ";
                cout << fixed << "{" << ant_result.Gain << "}";
                cout << endl;
                cout << "Antenna gain vector [s" << ant_result.Gain_vec.size() << "] :  ";

                i = 0;
                while (i < ant_result.Gain_vec.size())
                {
                    cout << fixed << "{" << ant_result.Gain_vec[i] << "}";
                    i++;
                }
                cout << endl;
                break;
            case 3:
                ant_result = ant_parameter.Order3(obj_ids, txAzAngle, txElAngle, rxAzAngle, rxElAngle);
                cout << "Result out: " << endl;
                cout << "Antenna direct gain:  ";
                cout << fixed << "{" << ant_result.Gain << "}";
                cout << endl;
                cout << "Antenna gain vector [s" << ant_result.Gain_vec.size() << "] :  ";

                i = 0;
                while (i < ant_result.Gain_vec.size())
                {
                    cout << fixed << "{" << ant_result.Gain_vec[i] << "}";
                    i++;
                }
                cout << endl;
                break;
            case 4:
                ant_result = ant_parameter.Order4(obj_ids, txAzAngle, txElAngle, rxAzAngle, rxElAngle, a_sec_fidel);
                cout << "Result out: " << endl;
                cout << "Antenna direct gain:  ";
                cout << fixed << "{" << ant_result.Gain << "}";
                cout << endl;
                cout << "Antenna gain vector [s" << ant_result.Gain_vec.size() << "] :  ";

                i = 0;
                while (i < ant_result.Gain_vec.size())
                {
                    cout << fixed << "{" << ant_result.Gain_vec[i] << "}";
                    i++;
                }
                cout << endl;
                break;
            default:
                cout << "Invalid grade" << endl;
            }
        }
        

    }



    
    //rcs_scatter.RCS_default();








    //RCS_out result;

    //result = rcs_scatter.Order5();
    //i = 0;
    //cout << "Range : " << endl;
    //while (i < result.range.size())
    //{
    //    cout << result.range[i] << " , ";
    //    i++;
    //}
    //cout << endl;
    //i = 0;
    //cout << "Sigma : " << result.Sigma.size() << endl;
    //while (i < result.Sigma.size())
    //{
    //    cout << result.Sigma[i] << " , ";
    //    i++;
    //}
    //cout << endl;

}

