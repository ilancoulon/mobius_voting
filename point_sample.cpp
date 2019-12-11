#include "point_sample.h"
#include <iostream>

VectorXi point_sampling(MatrixXd V, MatrixXi F, int numberToSample) {
	using namespace std;
	VectorXd K;

	igl::gaussian_curvature(V, F, K);

	VectorXi maxima;
	localMaxima(K, V, F, maxima, numberToSample);
	fpsSampling(V, F, maxima, numberToSample);
	return maxima;

}

void localMaxima(VectorXd &K, MatrixXd &V, MatrixXi &F, VectorXi &maxima, int numberToSample) {
	// We use the half-edge data structure to easily get the neighbors
	HalfedgeBuilder* builder = new HalfedgeBuilder();
	HalfedgeDS heDs = builder->createMeshWithFaces(V.rows(), F);


	maxima = VectorXi::Zero(K.rows());

	int maximaFound = 0;

	// Iterate over the vertices
	for (size_t i = 0; i < V.rows() && maximaFound < numberToSample; i++)
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
			maximaFound++;
		}
	}
}

void fpsSampling(MatrixXd& V, MatrixXi& F, VectorXi& sampled, int numberToSample) {
	Eigen::VectorXi VS, FS, VT, FT;
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

	while (nSampled < numberToSample) {
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
