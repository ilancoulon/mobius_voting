#pragma once

#include <igl/opengl/glfw/Viewer.h>

#ifndef HALFEDGE_DS_HEADER
#define HALFEDGE_DS_HEADER
#include "HalfedgeDS.cpp"
#endif

using namespace Eigen;
using namespace std;


class MidEdge
{

public:
    /** 
	 * Initialize the data structures
	 **/
    MidEdge(MatrixXd &V_original, MatrixXi &F_original, HalfedgeDS &mesh)
    {
        he = &mesh;
        V = &V_original;
        F = &F_original;          
        int e = he->sizeOfHalfedges() / 2; 
        int n = V_original.rows();         

        nVertices = e;
		nFaces = F->rows();

		V1 = new MatrixXd(nVertices, 3);
		F1 = new MatrixXi(nFaces, 3);
        halfpoints = new int[2*e];
    }

    void subdivide()
    {
        std::cout << "Performing one round subdivision" << endl;
        int e = he->sizeOfHalfedges() / 2; // number of edges in the original mesh
        int n = he->sizeOfVertices();      // number of vertices in the original mesh
        int F = he->sizeOfFaces();         // number of vertices in the original mesh


        for (int i = 0; i < 2 * e; i++)
            halfpoints[i] = -1;

        int id_point = 0;

        std::cout << e << std::endl;

        for (int i = 0; i < 2 * e; i++)
        {
            if (halfpoints[i] == -1)
            {
                halfpoints[i] = id_point;
                halfpoints[he->getOpposite(i)] = id_point;
                V1->row(id_point) = computeEdgePoint(i);
                id_point++;
            }
        }

        for (int i = 0; i < F; i++)
        {

            int v0 = halfpoints[he->getEdgeInFace(i)];
            int v1 = halfpoints[he->getNext(he->getEdgeInFace(i))];
            int v2 = halfpoints[he->getNext(he->getNext(he->getEdgeInFace(i)))];

            F1->row(i) << v0, v1, v2;
        }
    }

    /** 
	 * Return the number of half-edges
	 **/
    MatrixXd getVertexCoordinates()
    {
        return *V1;
    }

    /** 
	 * Return the number of faces
	 **/
    MatrixXi getFaces()
    {
        return *F1;
    }

    int* getHalfPoints()
    {
        return halfpoints;
    }

private:
    /**
	 * Compute the midpoint of the given half-edge 'h=(u,v)'
	 */
    MatrixXd computeEdgePoint(int h)
    {

        MatrixXd v0 = V->row(he->getTarget(h));
        MatrixXd v1 = V->row(he->getTarget(he->getOpposite(h)));

        return 0.5 * (v0 + v1);
    }


    int vertexDegree(int v)
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

    HalfedgeDS *he;
    MatrixXd *V;

    MatrixXi *F; 

    int nVertices, nFaces; 
    int *halfpoints;
    MatrixXd *V1;          
    MatrixXi *F1;         
};
