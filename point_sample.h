#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#pragma once

using namespace Eigen;
#include <igl/octree.h>
#include <igl/knn.h>
#include <igl/gaussian_curvature.h>
#include <igl/exact_geodesic.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/jet.h>

#include "HalfedgeBuilder.cpp"


void gaussian_curv();
void localMaxima(VectorXd& K, MatrixXd& V, MatrixXi& F, VectorXi& maxima);
void fpsSampling(MatrixXd& V, MatrixXi& F, VectorXi& sampled);