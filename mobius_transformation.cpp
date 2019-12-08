#pragma once

#ifndef HALFEDGE_DS_HEADER
#define HALFEDGE_DS_HEADER
#include "HalfedgeDS.cpp"
#endif

using namespace Eigen;
using namespace std;


class Mobius
{

public:
    
    Mobius(VectorXcd &points, int i, int j, int k)
    {


        pointsBefore = &points;

        MatrixXcd A = MatrixXcd::Zero(2,2);


        std::complex<double> y1  (-0.5, 0.5*sqrt(3));
        std::complex<double> y2  (-0.5, -0.5*sqrt(3));
        std::complex<double> y3  (1, 0);


        A(0,0) = y2 - y3;
        A(0,1) = y1*y3 - y1*y2;
        A(1,0) = y2 - y1;
        A(1,1) = y1*y3 - y3*y2;

        A = A.inverse();


        MatrixXcd B = MatrixXcd::Zero(2,2);


        std::complex<double> z1;
        std::complex<double> z2;
        std::complex<double> z3;  


        z1 = points(i);
        z2 = points(j);
        z3 = points(k); 




        B(0,0) = z2 - z3;
        B(0,1) = z1*z3 - z1*z2;
        B(1,0) = z2 - z1;
        B(1,1) = z1*z3 - z3*z2;

        mobius_transformation = A*B;        

        pointsAfter = VectorXcd::Zero(pointsBefore->rows());
        
        for(int pt = 0; pt < pointsBefore->rows(); pt++) {

            pointsAfter(pt) = (mobius_transformation(0,0)*(*pointsBefore)(pt) + mobius_transformation(0,1)) / (mobius_transformation(1,0)*(*pointsBefore)(pt) + mobius_transformation(1,1)) ;

        } 
        
    }

    VectorXcd getNewCoordinates()
    {
        return pointsAfter;
    }

private:

    VectorXcd *pointsBefore;
    VectorXcd pointsAfter;

    MatrixXcd mobius_transformation;

};
