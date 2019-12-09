#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <igl/octree.h>
#include <igl/knn.h>
#include <igl/writeOBJ.h>
#include <iostream>
#include <ostream>
#include <list>

#include "point_sample.h"

#include "mid_edge_uniformisation.cpp"
#include "planar_embedding.cpp"
#include "mobius_transformation.cpp"

//using namespace Eigen; // to use the classes provided by Eigen library
MatrixXd V1; // matrix storing vertex coordinates of the input curve
MatrixXi F1;
MatrixXd V2; // matrix storing vertex coordinates of the input curve
MatrixXi F2;

MatrixXd V;
MatrixXi F;

VectorXd u;
VectorXd u_star;

MatrixXd Vmid;
MatrixXi Fmid;

VectorXcd planarPoints;
VectorXcd planarPointsAfterMobius;

int *halfpoints;

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier)
{
	std::cout << "pressed Key: " << key << " " << (unsigned int)key << std::endl;

	if (key == '2')
	{
		HalfedgeBuilder *builder = new HalfedgeBuilder();
		HalfedgeDS he = (builder->createMeshWithFaces(V.rows(), F)); // create the half-edge representation
		MidEdge *midedge = new MidEdge(V, F, he);					 //
		midedge->subdivide();

		// update the current mesh
		V = midedge->getVertexCoordinates(); // update vertex coordinates
		F = midedge->getFaces();
		viewer.data().clear();
		viewer.data().set_mesh(V, F);
		return true;
	}

	if (key == '3')
	{
		HalfedgeBuilder *builder = new HalfedgeBuilder();
		HalfedgeDS he = (builder->createMeshWithFaces(V.rows(), F)); // create the half-edge representation
		MidEdge *midedge = new MidEdge(V, F, he);
		midedge->subdivide();

		Vmid = midedge->getVertexCoordinates();
		Fmid = midedge->getFaces();
		halfpoints = midedge->getHalfPoints();

		HalfedgeBuilder *builder2 = new HalfedgeBuilder();
		HalfedgeDS hemid = (builder2->createMeshWithFaces(Vmid.rows(), Fmid));

		PlanarEmbedding *planar = new PlanarEmbedding(V, F, he, Vmid, Fmid, hemid, halfpoints);

		u = planar->u();
		u_star = planar->u_star(u);

		planar->embedding(u, u_star);

		V = planar->getVertexCoordinates(); // update vertex coordinates
		F = planar->getFaces();

		planarPoints = planar->getComplexCoordinates();

		//std::cout << planarPoints << std::endl << std::endl;

		Mobius *mobius = new Mobius(planarPoints, 0, 1, 2);

		planarPointsAfterMobius = mobius->getNewCoordinates();

		//std::cout << planarPointsAfterMobius << std::endl << std::endl;

		viewer.data().clear();
		viewer.data().set_mesh(V, F);
		return true;
	}

	return false;
}

void draw_normals(igl::opengl::glfw::Viewer &viewer, const MatrixXd &V, const MatrixXd &n)
{
	MatrixXd current_edge(V.rows(), 3);
	for (unsigned i = 1; i < V.rows() - 1; ++i)
	{
		current_edge(i, 0) = V(i, 0) + n(i, 0);
		current_edge(i, 1) = V(i, 1) + n(i, 1);
		current_edge(i, 2) = V(i, 2) + n(i, 2);
		viewer.data().add_edges(
			V.row(i),
			current_edge.row(i),
			Eigen::RowVector3d(1, 1, 1));
	}
}

void set_meshes(igl::opengl::glfw::Viewer &viewer)
{
	viewer.callback_key_down = &key_down; // for dealing with keyboard events
	viewer.data().set_mesh(V1, F1);
	viewer.append_mesh();
	viewer.data().set_mesh(V2, F2);
	viewer.data(0).set_colors(Eigen::RowVector3d(0.3, 0.8, 0.3));
	viewer.data(1).set_colors(Eigen::RowVector3d(0.8, 0.3, 0.3));
}

void set_pc(igl::opengl::glfw::Viewer &viewer)
{
	viewer.callback_key_down = &key_down; // for dealing with keyboard events
	viewer.data().add_points(V1, Eigen::RowVector3d(0.3, 0.8, 0.3));
}

void createOctagon(MatrixXd &Vertices, MatrixXi &Faces)
{
	Vertices = MatrixXd(6, 3);
	Faces = MatrixXi(8, 3);

	Vertices << 0.0, 0.0, 1.0,
		1.000000, 0.000000, 0.000000,
		0.000000, 1.000000, 0.000000,
		-1.000000, 0.000000, 0.000000,
		0.000000, -1.000000, 0.000000,
		0.000000, 0.000000, -1.000000;

	Faces << 0, 1, 2,
		0, 2, 3,
		0, 3, 4,
		0, 4, 1,
		5, 2, 1,
		5, 3, 2,
		5, 4, 3,
		5, 1, 4;
}

MatrixXd mobius_voting(MatrixXd V1, MatrixXi F1, MatrixXd V2, MatrixXi F2, int I, int K, double epsilon)
{

	VectorXcd planarPoints1;
	VectorXcd planarPoints2;

	////////////////////////////////////////////////
	///////// Planar Embedding for Shape 1 /////////
	////////////////////////////////////////////////

	HalfedgeBuilder *builder1 = new HalfedgeBuilder();
	HalfedgeDS he1 = (builder1->createMeshWithFaces(V1.rows(), F1)); // create the half-edge representation
	MidEdge *midedge1 = new MidEdge(V1, F1, he1);
	midedge1->subdivide();

	MatrixXd Vmid1;
	MatrixXi Fmid1;
	int *halfpoints1;

	Vmid1 = midedge1->getVertexCoordinates();
	Fmid1 = midedge1->getFaces();
	halfpoints1 = midedge1->getHalfPoints();

	HalfedgeBuilder *buildermid1 = new HalfedgeBuilder();
	HalfedgeDS hemid1 = (buildermid1->createMeshWithFaces(Vmid1.rows(), Fmid1));

	PlanarEmbedding *planar1 = new PlanarEmbedding(V1, F1, he1, Vmid1, Fmid1, hemid1, halfpoints1);

	VectorXd u1;
	VectorXd u_star1;

	u1 = planar1->u();
	u_star1 = planar1->u_star(u1);

	planar1->embedding(u1, u_star1);

	planarPoints1 = planar1->getComplexCoordinates();

	/////////////////////////////////////////////////
	///////// Planar Embedding for Shape 2 //////////
	/////////////////////////////////////////////////

	HalfedgeBuilder *builder2 = new HalfedgeBuilder();
	HalfedgeDS he2 = (builder2->createMeshWithFaces(V2.rows(), F2)); // create the half-edge representation
	MidEdge *midedge2 = new MidEdge(V2, F2, he2);
	midedge2->subdivide();

	MatrixXd Vmid2;
	MatrixXi Fmid2;
	int *halfpoints2;

	Vmid2 = midedge2->getVertexCoordinates();
	Fmid2 = midedge2->getFaces();
	halfpoints2 = midedge2->getHalfPoints();

	HalfedgeBuilder *buildermid2 = new HalfedgeBuilder();
	HalfedgeDS hemid2 = (buildermid2->createMeshWithFaces(Vmid2.rows(), Fmid2));

	PlanarEmbedding *planar2 = new PlanarEmbedding(V2, F2, he2, Vmid2, Fmid2, hemid2, halfpoints2);

	VectorXd u2;
	VectorXd u_star2;

	u2 = planar2->u();
	u_star2 = planar2->u_star(u2);

	planar2->embedding(u2, u_star2);

	planarPoints2 = planar2->getComplexCoordinates();

	//////////////////////////////////////////////
	//////////// TAKE SAMPLE POINTS //////////////
	//////////////////////////////////////////////

	VectorXi sampled1 = gaussian_curv(V1, F1);
	VectorXi sampled2 = gaussian_curv(V2, F2);

	//TO DO : DO THE SAMPLING

	VectorXi points1 = VectorXi::Zero(20); 
	VectorXi points2 = VectorXi::Zero(20); 

	VectorXcd sigma1 = VectorXcd::Zero(20); //Contains the indexes of sample points in Mid-Edge Mesh 1
	VectorXcd sigma2 = VectorXcd::Zero(20); //Contains the indexes of sample points in Mid-Edge Mesh 2

	int count1 = 0;
	int count2 = 0;

	for (int i = 0; i < sampled1.rows(); i++)
	{
		if (sampled1(i) == 1)
		{
			points1(count1) = i;
			sigma1(count1) = halfpoints1[he1.getEdge(i)]; //FOR THE MOMENT : Some half-edge. Can do better ! TO IMPROVE
			count1++;
		}
		if (sampled2(i) == 1)
		{
			points2(count2) = i;
			sigma2(count2) = halfpoints2[he2.getEdge(i)]; //FOR THE MOMENT : Some half-edge. Can do better !  TO IMPROVE
			count2++;
		}
	}

	int nbSampled = sigma1.rows();
	MatrixXd C = MatrixXd::Zero(nbSampled, nbSampled);

	///////////////////////////////////////////////
	////////   MOBIUS VOTING ALGORITHM  ///////////
	///////////////////////////////////////////////

	for (int i = 0; i < I; i++)
	{

		//Pick 3 random points in both meshes

		int i1;
		int j1;
		int k1;
		int i2;
		int j2;
		int k2;

		i1 = rand() % nbSampled;
		do
		{
			j1 = rand() % nbSampled;
		} while (j1 == i1);
		do
		{
			k1 = rand() % nbSampled;
		} while (k1 == j1 || k1 == i1);

		i2 = rand() % nbSampled;
		do
		{
			j2 = rand() % nbSampled;
		} while (j2 == i2);
		do
		{
			k2 = rand() % nbSampled;
		} while (k2 == j2 || k2 == i2);

		//Compute Möbius Transformation

		Mobius *mobius1 = new Mobius(sigma1, i1, j1, k1);
		Mobius *mobius2 = new Mobius(sigma2, i2, j2, k2);

		VectorXcd planarPointsAfterMobius1;
		VectorXcd planarPointsAfterMobius2;

		planarPointsAfterMobius1 = mobius1->getNewCoordinates();
		planarPointsAfterMobius2 = mobius2->getNewCoordinates();

		//FIND MUTUALLY CLOSEST NEIGHBORS AND FILL THEM IN "neigh1", "neigh2"

		int nb_neigh = 0;
		list<int> neigh1;
		list<int> neigh2;

		MatrixXd dist = MatrixXd::Zero(nbSampled, nbSampled);
		VectorXi nn_1 = VectorXi::Zero(nbSampled);
		VectorXi nn_2 = VectorXi::Zero(nbSampled);

		for (int i = 0; i < nbSampled; i++)
		{
			for (int j = 0; j < nbSampled; j++)
			{
				dist(i, j) = abs(planarPointsAfterMobius1[i] - planarPointsAfterMobius2[j]);
			}
		}

		for (int i = 0; i < nbSampled; i++)
		{
			double dist_min = dist(i, 0);
			int nn = 0;
			for (int j = 1; j < nbSampled; j++)
			{
				if (dist(i, j) < dist_min)
				{
					dist_min = dist(i, j);
					nn = j;
				}
				nn_1(i) = nn;
			}

			for (int i = 0; i < nbSampled; i++)
			{
				double dist_min = dist(0, i);
				int nn = 0;
				for (int j = 1; j < nbSampled; j++)
				{
					if (dist(j, i) < dist_min)
					{
						dist_min = dist(j, i);
						nn = j;
					}
					nn_2(i) = nn;
				}
			}

			for (int i = 0; i < nbSampled; i++)
			{
				if (nn_2(nn_1(i)) == i)
				{
					nb_neigh++;
					neigh1.push_front(i);
					neigh2.push_front(nn_1(i));
				}
			}
		}

		//Vote if the number of mutually closest neighbors > K 

		if (nb_neigh > K)
		{

			double energy = 1.;
			//TO-DO : CALCULATE THE ENERGY

			for (int neigh = 0; neigh < nb_neigh; neigh++)
			{

				int k = neigh1.front();
				int l = neigh2.front();

				neigh1.pop_front();
				neigh2.pop_front();

				C(k, l) += 1 / (epsilon + energy / (double)nb_neigh);
			}
		}
	}

	std::cout << points1 << std::endl;
	std::cout << points2 << std::endl;
	std::cout << C << std::endl;

	return C;
}

int main(int argc, char *argv[])
{

	igl::readOFF("../data/bunny.off", V1, F1);
	igl::readOFF("../data/bunny_rotated.off", V2, F2);

	MatrixXd C = mobius_voting(V1, F1, V2, F2, 1000, 20, 0.1);

	igl::readOFF("../data/star.off", V, F);

	igl::opengl::glfw::Viewer viewer; // create the 3d viewer

	viewer.callback_key_down = &key_down;
	viewer.data().set_mesh(V, F);

	viewer.core(0).align_camera_center(V, F);
	viewer.launch(); // run the viewer
}
