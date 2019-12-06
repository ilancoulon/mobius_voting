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


	VectorXi maxima;
	localMaxima(K, V, F, maxima);
	fpsSampling(V, F, maxima);

	// Compute pseudocolor
	MatrixXd C;

	igl::jet(maxima, true, C);



	// Plot the mesh with pseudocolors
	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(V, F);
	viewer.data().set_colors(C);
	viewer.launch();
}

void localMaxima(VectorXd &K, MatrixXd &V, MatrixXi &F, VectorXi &maxima) {
	// We use the half-edge data structure to easily get the neighbors
	HalfedgeBuilder* builder = new HalfedgeBuilder();
	HalfedgeDS heDs = builder->createMeshWithFaces(V.rows(), F);


	maxima = VectorXi::Zero(K.rows());

	// Iterate over the vertices
	for (size_t i = 0; i < V.rows(); i++)
	{
		bool isMax = true;
		int startHe = heDs.getEdge(i);

		// Iterate over the neighbors
		int currentHe = heDs.getOpposite(heDs.getNext(startHe));
		if (K(i) < 2. * K(heDs.getTarget(heDs.getOpposite(startHe)))) {
			isMax = false;
		}
		while (isMax && currentHe != startHe) {
			if (K(i) < 2. * K(heDs.getTarget(heDs.getOpposite(currentHe)))) {
				isMax = false;
			}
			currentHe = heDs.getOpposite(heDs.getNext(currentHe));
		}
		if (isMax) {
			maxima(i) = 1;
		}
	}
}

void fpsSampling(MatrixXd& V, MatrixXi& F, VectorXi& sampled) {
	Eigen::VectorXi VS, FS, VT, FT;
	int n = 20;
	int nSampled = 0;
	for (size_t i = 0; i < sampled.rows(); i++)
	{
		if (sampled(i) == 1) {
			nSampled++;
		}
	}

	// In case nothing sampled at the beginning
	if (nSampled == 0) {
		nSampled++;
		sampled(0) = 1;
	}


	VT.setLinSpaced(V.rows(), 0, V.rows() - 1);

	while (nSampled < n) {
		VS = VectorXi::Zero(nSampled);
		int j = 0;
		for (size_t i = 0; i < sampled.rows(); i++)
		{
			if (sampled(i) == 1) {
				VS(j) = i;
				j++;
			}
		}
		Eigen::VectorXd d;
		igl::exact_geodesic(V, F, VS, FS, VT, FT, d);

		double maxDist = d(0);
		int maxIndex = VT(0);
		for (size_t j = 0; j < d.rows(); j++)
		{
			if (d(j) > maxDist) {
				maxDist = d(j);
				maxIndex = VT(j);
			}
		}
		nSampled++;
		sampled(maxIndex) = 1;
	}
}
