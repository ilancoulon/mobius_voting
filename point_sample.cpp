#include "point_sample.h"
#include <iostream>

void gaussian_curv()
{
	using namespace std;
	MatrixXd V;
	MatrixXi F;
	igl::readOFF("../../../data/bunny.off", V, F);

	VectorXd K;
	// Compute integral of Gaussian curvature
	igl::gaussian_curvature(V, F, K);
	std::cout << "Rows : " << V.rows() << ", Cols : " << V.cols() << "; V(10, 2): " << V(10, 2) << std::endl;
	std::cout << "Rows : " << K.rows() << ", Cols : " << K.cols() << "; K(10): " << K(10) << std::endl;
	// Compute mass matrix
	SparseMatrix<double> M, Minv;
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, M);
	igl::invert_diag(M, Minv);
	// Divide by area to get integral average
	//K = (Minv * K).eval();

	// Compute pseudocolor
	MatrixXd C;
	igl::jet(K, true, C);

	// Plot the mesh with pseudocolors
	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(V, F);
	viewer.data().set_colors(C);
	viewer.launch();
}
