#pragma once

#ifndef HALFEDGE_DS_HEADER
#define HALFEDGE_DS_HEADER
#include "HalfedgeDS.cpp"
#endif
#include "planar_embedding.h"

using namespace Eigen;
using namespace std;

PlanarEmbedding::PlanarEmbedding(MatrixXd &V_original, MatrixXi &F_original, HalfedgeDS &mesh, MatrixXd &Vmid_original, MatrixXi &Fmid_original, HalfedgeDS &meshmid, int *_halfpoints)
{
    he = &mesh;
    V = &V_original;
    F = &F_original;
    hemid = &meshmid;
    Vmid = &Vmid_original;
    Fmid = &Fmid_original;
    int e = he->sizeOfHalfedges() / 2;
    int n = V_original.rows();

    halfpoints = _halfpoints;

    int emid = hemid->sizeOfHalfedges() / 2;
    int nmid = Vmid_original.rows();

    nVertices = n;
    nFaces = F->rows();

    nVerticesmid = nmid;
    nFacesmid = Fmid->rows();

    Vplain = new MatrixXd(nVerticesmid, V->cols());
}

double PlanarEmbedding::cot(Vector3d v, Vector3d w)
{
    return (v.dot(w) / (v.cross(w).norm()));
}

VectorXd PlanarEmbedding::u()
{

    MatrixXd System = MatrixXd::Zero(nVertices, nVertices);
    VectorXd u = VectorXd::Zero(nVertices);
    VectorXd rightMember = VectorXd::Zero(nVertices);
	size_t nonZeroInSystem = 0;

    for (int v = 0; v < nVertices; v++)
    {

        int e = he->getEdge(v);
        int e0 = e;
        int e1 = he->getOpposite(he->getNext(e0));
        int e2 = he->getOpposite(he->getNext(e1));

        int v0 = he->getTarget(he->getOpposite(e0));
        int v1 = he->getTarget(he->getOpposite(e1));
        int v2 = he->getTarget(he->getOpposite(e2));

        //std::cout << v1 << std::endl;
        //std::cout << v2 << std::endl;

        Vector3d vec1a = V->row(v1) - V->row(v0);
        Vector3d vec1b = V->row(v) - V->row(v0);
        Vector3d vec2a = V->row(v1) - V->row(v2);
        Vector3d vec2b = V->row(v) - V->row(v2);

        //std::cout << vec0 << std::endl;
        //std::cout << vec1 << std::endl;
        //std::cout << vec2 << std::endl;

        double cota = cot(vec1a, vec1b);
        double cotb = cot(vec2a, vec2b);

        //std::cout << cota << std::endl;
        //std::cout << cotb << std::endl;

        System(v, v) = System(v, v) + cota + cotb;
        System(v, v1) = System(v, v1) - cota - cotb;
		nonZeroInSystem += 2;


        int currentEdge = he->getOpposite(he->getNext(e));

        while (currentEdge != e)
        {

            e0 = currentEdge;
            e1 = he->getOpposite(he->getNext(e0));
            e2 = he->getOpposite(he->getNext(e1));

            v0 = he->getTarget(he->getOpposite(e0));
            v1 = he->getTarget(he->getOpposite(e1));
            v2 = he->getTarget(he->getOpposite(e2));

            //std::cout << v0 << std::endl;
            //std::cout << v1 << std::endl;
            //std::cout << v2 << std::endl;

            vec1a = V->row(v1) - V->row(v0);
            vec1b = V->row(v) - V->row(v0);
            vec2a = V->row(v1) - V->row(v2);
            vec2b = V->row(v) - V->row(v2);

            //std::cout << vec0 << std::endl;
            //std::cout << vec1 << std::endl;
            //std::cout << vec2 << std::endl;

            cota = cot(vec1a, vec1b);
            cotb = cot(vec2a, vec2b);

            //std::cout << cota << std::endl;
            //std::cout << cotb << std::endl;

            System(v, v) = System(v, v) + cota + cotb;
            System(v, v1) = System(v, v1) - cota - cotb;
			nonZeroInSystem++;


            currentEdge = he->getOpposite(he->getNext(currentEdge));
        }
    }
		


    int vic = (*F)(0, 0);
    int vjc = (*F)(0, 1);

    for (int v = 0; v < nVertices; v++)
    {

        System(vic, v) = 0;
        System(vjc, v) = 0;
    }

    System(vic, vic) = 1;
    System(vjc, vjc) = 1;

    rightMember(vic) = -1;
    rightMember(vjc) = 1;
	nonZeroInSystem += 4;

	Eigen::SparseMatrix<double> sparseSystem(System.rows(), System.cols());
	std::vector<Eigen::Triplet<double>> tripletList;
	tripletList.reserve(nonZeroInSystem+50);
	for (size_t i = 0; i < System.rows(); i++)
	{
		for (size_t j = 0; j < System.cols(); j++)
		{
			if (System(i, j) != 0.) {
				tripletList.push_back(Eigen::Triplet<double>(i, j, System(i, j)));
			}

		}
	}
	sparseSystem.setFromTriplets(tripletList.begin(), tripletList.end());
	Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;
	sparseSystem.makeCompressed();
	solver.analyzePattern(sparseSystem);
	solver.factorize(sparseSystem);

	u = solver.solve(rightMember);

    return u;
}

VectorXd PlanarEmbedding::u_star(VectorXd u)
{
	Eigen::SparseMatrix<double> sparseSystem(nVerticesmid, nVerticesmid);
	std::vector<Eigen::Triplet<double>> tripletList;
	tripletList.reserve(nFaces*3 + 50);
    VectorXd u_star = VectorXd::Zero(nVerticesmid);
    VectorXd rightMember = VectorXd::Zero(nVerticesmid);


	tripletList.push_back(Eigen::Triplet<double>(0,0, 1));
    rightMember(0) = 0;

    bool *alreadyDone;

    alreadyDone = new bool[nVerticesmid];
    for (int i = 0; i < nVerticesmid; i++)
    {

        alreadyDone[i] = false;
    }

    alreadyDone[0] = true;

    for (int f = 0; f < nFaces; f++)
    {

        if (f == 0)
            continue;

        int e = he->getEdgeInFace(f);

        for (int i = 0; i < 3; i++)
        {

            int he0 = e;
            int he1 = he->getNext(he0);
            int he2 = he->getNext(he1);

            int vi = he->getTarget(he0);
            int vj = he->getTarget(he1);
            int vk = he->getTarget(he2);

            int vr = halfpoints[he1];

            if (true)
            {

                int vs = halfpoints[he2];

                Vector3d vik = V->row(vk) - V->row(vi);
                Vector3d vij = V->row(vj) - V->row(vi);

                Vector3d vki = V->row(vi) - V->row(vk);
                Vector3d vkj = V->row(vj) - V->row(vk);

                double coti = cot(vik, vij);
                double cotk = cot(vki, vkj);

                double ui = u(vi);
                double uj = u(vj);
                double uk = u(vk);

				if (vr != 0) {
					// Triplet list is summed up at the end
					tripletList.push_back(Eigen::Triplet<double>(vr, vr, 1));
					tripletList.push_back(Eigen::Triplet<double>(vr, vs, -1));
				}
				

                rightMember(vr) += 0.5 * ((ui - uj) * cotk + (uk - uj) * coti);

                alreadyDone[vr] = true;
            }


            e = he->getNext(e);
        }
    }

    rightMember(0) = 0;

	
	/*for (size_t i = 0; i < System.rows(); i++)
	{
		for (size_t j = 0; j < System.cols(); j++)
		{
			if (System(i, j) != 0.) {
				tripletList.push_back(Eigen::Triplet<double>(i, j, System(i, j)));
			}

		}
	}*/
	sparseSystem.setFromTriplets(tripletList.begin(), tripletList.end());
	Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;
	sparseSystem.makeCompressed();
	solver.analyzePattern(sparseSystem);
	solver.factorize(sparseSystem);
	u_star = solver.solve(rightMember);

    return u_star;
}

void PlanarEmbedding::embedding(VectorXd u, VectorXd u_star)
{

    for (int v = 0; v < nVerticesmid; v++)
    {

        int index = 0;

        for (int e = 0; e < 2 * nVerticesmid; e++)
        {

            if (halfpoints[e] == v)
            {
                index = e;
                break;
            }
        }

        //std::cout << v << std::endl;

        int v1 = he->getTarget(index);
        int v2 = he->getTarget(he->getOpposite(index));

        Vplain->row(v) << (0.5 * u(v1) + 0.5 * u(v2)) * 0.1, u_star(v) * 0.1, 0.;
    }

    Fmid->row(0) = Fmid->row(1);

    /*std::cout << *Vplain << std::endl;
    std::cout << *Fmid << std::endl;*/
}

VectorXcd PlanarEmbedding::getComplexCoordinates() {

    VectorXcd res = VectorXcd::Zero(Vplain->rows());

    for(int v = 0; v < Vplain->rows(); v++) {

        std::complex<double> point ((*Vplain)(v,0),(*Vplain)(v,1));
        res(v) = point;

    }
    return res;
}

MatrixXd PlanarEmbedding::getVertexCoordinates()
{
	//std::cout << "Vertices :" << *Vplain << std::endl;
	return *Vplain;
}

MatrixXi PlanarEmbedding::getFaces()
{
    MatrixXi testF = MatrixXi::Zero(Fmid->rows() - 1, 3);
    for (size_t i = 1; i < Fmid->rows(); i++)
    {
        testF.row(i - 1) = Fmid->row(i);
    }
    //std::cout << "Faces :" << testF << std::endl;
    return testF;
}

int PlanarEmbedding::vertexDegree(int v)
{
    int result = 0;
    int e = he->getEdge(v);
    int pEdge = he->getOpposite(he->getNext(e));
    while (pEdge != e)
    {
        pEdge = he->getOpposite(he->getNext(pEdge));
        result++;
    }
    return result + 1;
}