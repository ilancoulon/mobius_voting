#pragma once

#ifndef HALFEDGE_DS_HEADER
#define HALFEDGE_DS_HEADER
#include "HalfedgeDS.cpp"
#endif

#include <Eigen/Sparse>

using namespace Eigen;
using namespace std;


/*
This class computes the planar embedding
You have to compute u, and u_star using the corresponding methods
Then you use embedding() to put the results in Vplain
You can then display the embedding using getVertexCoordinates() and getFaces()
Or get the complex coordinates using getComplexCoordinates()
*/
class PlanarEmbedding
{

public:
	PlanarEmbedding(MatrixXd& V_original, MatrixXi& F_original, HalfedgeDS& mesh, MatrixXd& Vmid_original, MatrixXi& Fmid_original, HalfedgeDS& meshmid, int* _halfpoints);

	double cot(Vector3d v, Vector3d w);

	VectorXd u();

	VectorXd u_star(VectorXd u);

	void embedding(VectorXd u, VectorXd u_star);

	VectorXcd getComplexCoordinates();

	MatrixXd getVertexCoordinates();

	MatrixXi getFaces();

private:
	int vertexDegree(int v);

	HalfedgeDS* he;
	MatrixXd* V;
	MatrixXi* F;

	HalfedgeDS* hemid;
	MatrixXd* Vmid;
	MatrixXi* Fmid;

	int nVertices, nFaces;
	int nVerticesmid, nFacesmid;
	int* halfpoints;

	MatrixXd* Vplain;
};