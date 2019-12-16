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


    MidEdge(MatrixXd &V_original, MatrixXi &F_original, HalfedgeDS &mesh)
    {
        he = &mesh;
        V = &V_original;
        F = &F_original;    

        nVertices = he->sizeOfHalfedges() / 2;
		nFaces = F->rows();

		V1 = new MatrixXd(nVertices, 3);
		F1 = new MatrixXi(nFaces, 3);
        halfpoints = new int[he->sizeOfHalfedges()];
    }

    void makeMidEdge()
    {

        int e = he->sizeOfHalfedges() / 2; 
        int n = he->sizeOfVertices();     
        int F = he->sizeOfFaces();       


        for (int i = 0; i < 2 * e; i++)
            halfpoints[i] = -1;

        int id_point = 0;

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

 
    MatrixXd getVertexCoordinates()
    {
        return *V1;
    }

  
    MatrixXi getFaces()
    {
        return *F1;
    }

    int* getHalfPoints()
    {
        return halfpoints;
    }

private:
  
  
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
