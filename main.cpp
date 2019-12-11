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
#include "planar_embedding.h"
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

VectorXi mobius_voting(MatrixXd V1, MatrixXi F1, MatrixXd V2, MatrixXi F2, int I, int K, double epsilon, int numberToSample, VectorXi &points1, VectorXi &points2, MatrixXd &C)
{

	//Modifies sampled1, sampled2, and C

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

	cout << "Planar Embedding for Shape 1...";
	PlanarEmbedding *planar1 = new PlanarEmbedding(V1, F1, he1, Vmid1, Fmid1, hemid1, halfpoints1);

	VectorXd u1;
	VectorXd u_star1;
	cout << " u...";

	u1 = planar1->u();
	cout << " u*...";
	u_star1 = planar1->u_star(u1);

	planar1->embedding(u1, u_star1);

	planarPoints1 = planar1->getComplexCoordinates();
	cout << "Done !" << endl;

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
	cout << "Planar Embedding for Shape 2...";
	PlanarEmbedding *planar2 = new PlanarEmbedding(V2, F2, he2, Vmid2, Fmid2, hemid2, halfpoints2);

	VectorXd u2;
	VectorXd u_star2;

	cout << " u...";
	u2 = planar2->u();
	cout << " u*...";
	u_star2 = planar2->u_star(u2);

	planar2->embedding(u2, u_star2);

	planarPoints2 = planar2->getComplexCoordinates();
	cout << "Done !" << endl;

	//////////////////////////////////////////////////////////
	//////////// TAKE SAMPLE POINTS IN THE MESH //////////////
	//////////////////////////////////////////////////////////

	cout << "Sampling...";
	VectorXi sampled1 = point_sampling(V1, F1, numberToSample);
	VectorXi sampled2 = point_sampling(V2, F2, numberToSample);
	cout << "Done!" << endl;
	points1 = VectorXi::Zero(numberToSample);
	points2 = VectorXi::Zero(numberToSample);

	VectorXcd sigma1 = VectorXcd::Zero(numberToSample); //Contains the sample points in Mid-Edge Mesh 1
	VectorXcd sigma2 = VectorXcd::Zero(numberToSample); //Contains the sample points in Mid-Edge Mesh 2



    /////////////////////////////////////////////////////////////////
	////// MATCH THE SAMPLED POINTS IN THE PLANAR EMBEDDING /////////
	/////////////////////////////////////////////////////////////////

	int count1 = 0;
	int count2 = 0;

	for (int i = 0; i < sampled1.rows(); i++)
	{
		if (sampled1(i) == 1)
		{
			points1(count1) = i;
			int e = he1.getEdge(i);
			int bestEdge = e;

			int currentEdge = he1.getOpposite(he1.getNext(e));
			while (currentEdge != e) {

				if ((V1.row(he1.getTarget(he1.getOpposite(currentEdge))) - V1.row(i)).norm() < (V1.row(he1.getTarget(he1.getOpposite(bestEdge))) - V1.row(i)).norm()) {
					bestEdge = currentEdge;
				}
				currentEdge = he1.getOpposite(he1.getNext(currentEdge));
			}
			sigma1(count1) = planarPoints1(halfpoints1[bestEdge]);
			count1++;
		}
		if (sampled2(i) == 1)
		{
			points2(count2) = i;
			int e = he2.getEdge(i);
			int bestEdge = e;

			int currentEdge = he2.getOpposite(he2.getNext(e));
			while (currentEdge != e) {

				if ((V2.row(he2.getTarget(he2.getOpposite(currentEdge))) - V2.row(i)).norm() < (V2.row(he2.getTarget(he2.getOpposite(bestEdge))) - V2.row(i)).norm()) {
					bestEdge = currentEdge;
				}
				currentEdge = he2.getOpposite(he2.getNext(currentEdge));
			}
			sigma2(count2) = planarPoints2(halfpoints2[bestEdge]);
			count2++;
		}
	}


	///////////////////////////////////////////////
	////////   MOBIUS VOTING ALGORITHM  ///////////
	///////////////////////////////////////////////


	int nbSampled = sigma1.rows();
	C = MatrixXd::Zero(nbSampled,nbSampled);

	for (int iter = 0; iter < I; iter++)
	{

		if (iter % (I/100) == 0) 
		std::cout << (float)iter / (float)I * 100 << " %" << std::endl;

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

		//Compute distance matrix

		for (int i = 0; i < nbSampled; i++)
		{
			for (int j = 0; j < nbSampled; j++)
			{
				dist(i, j) = abs(planarPointsAfterMobius1[i] - planarPointsAfterMobius2[j]);
			}
		}

		//Compute nn_1

		for (int i = 0; i < nbSampled; i++)
		{
			nn_1(i) = 0;
			double min_dist = dist(i,0);
			for (int j = 0; j < nbSampled; j++)
			{
				if (dist(i,j) < min_dist) {
					nn_1(i) = j;
					min_dist = dist(i,j);
				}
			}
		}

		//Compute nn_2

		for (int j = 0; j < nbSampled; j++)
		{
			nn_2(j) = 0;
			double min_dist = dist(0,j);
			for (int i = 0; i < nbSampled; i++)
			{
				if (dist(i,j) < min_dist) {
					nn_2(j) = i;
					min_dist = dist(i,j);
				}
			}
		}

		//Add mutual nn to the lists

		for (int i = 0; i < nbSampled; i++)
		{
			if (nn_2(nn_1(i)) == i)
			{
				nb_neigh++;
				neigh1.push_front(i);
				neigh2.push_front(nn_1(i));
			}
		}


		//Vote if the number of mutually closest neighbors > K

		if (nb_neigh > K)
		{
			// Computing the energy...
			double energy = 0.;
			list<int>::iterator it1 = neigh1.begin();
			list<int>::iterator it2 = neigh2.begin();
			while (it1 != neigh1.end())
			{
				energy += dist(*it1, *it2);
				it1++;
				it2++;
			}

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

	//std::cout << points1 << std::endl;
	//std::cout << points2 << std::endl;
	//std::cout << C << std::endl;

	VectorXi correspondances = VectorXi::Zero(nbSampled);
	int foundCorrespondances = 0;
	while (foundCorrespondances < nbSampled) {
		double maxFound = -1.;
		int rowMax = 0;
		int colMax = 0;
		for (size_t i = 0; i < C.rows(); i++)
		{
			for (size_t j = 0; j < C.cols(); j++)
			{
				if (C(i, j) > maxFound) {
					maxFound = C(i, j);
					rowMax = i;
					colMax = j;
				}
			}
		}
		//std::cout << C << std::endl;
		//std::cout << maxFound << std::endl;
		//std::cout << rowMax << std::endl;
		//std::cout << colMax << std::endl;
		//std::cout << "= = = = = = = = = = = = = = = = = = = = " << std::endl;
		for (size_t i = 0; i < C.rows(); i++)
		{
			C(i, colMax) = -1.;
		}
		for (size_t j = 0; j < C.cols(); j++)
		{
			C(rowMax, j) = -1.;
		}
		if (maxFound != -1)
			correspondances[rowMax] = colMax;
		foundCorrespondances++;
	}

	return correspondances;

}



int main(int argc, char *argv[])
{
	string figure1 = "gargoyle_tri.off";
	string figure2 = "gargoyle_tri_rotated.off";
	igl::readOFF("../data/star.off", V, F);
	igl::readOFF("../data/"+ figure1, V1, F1);
	igl::readOFF("../data/"+ figure2, V2, F2);

	VectorXi sampled1;
	VectorXi sampled2;

	VectorXi points1 = VectorXi::Zero(20);
	VectorXi points2 = VectorXi::Zero(20);

	MatrixXd C; 
	
	VectorXi correspondances = mobius_voting(V1, F1, V2, F2, 1000000, 15, 0.01, 150, sampled1, sampled2, C);

	std::cout << sampled1 << std::endl;
	std::cout << sampled2 << std::endl;
	std::cout << C << std::endl;
	std::cout << correspondances << std::endl;



	igl::opengl::glfw::Viewer viewer; // create the 3d viewer

	viewer.callback_key_down = &key_down;

	viewer.load_mesh_from_file("../data/"+figure1);
	viewer.load_mesh_from_file("../data/"+figure2);

	unsigned int left_view, right_view;
	int fig1_id = viewer.data_list[0].id;
	int fig2_id = viewer.data_list[1].id;
	viewer.callback_init = [&](igl::opengl::glfw::Viewer&)
	{
		viewer.core().viewport = Eigen::Vector4f(0, 0, 640, 800);
		left_view = viewer.core_list[0].id;
		right_view = viewer.append_core(Eigen::Vector4f(640, 0, 640, 800));
		viewer.core(left_view).align_camera_center(V1, F1);
		viewer.core(right_view).align_camera_center(V2, F2);
		viewer.data(fig1_id).set_visible(false, right_view);
		viewer.data(fig2_id).set_visible(false, left_view);
		return false;
	};

	for (size_t i = 0; i < correspondances.rows(); i++)
	{
		float r = (float)std::rand() / (float)RAND_MAX;
		float g = (float)std::rand() / (float)RAND_MAX;
		float b = (float) std::rand() / (float) RAND_MAX;

		viewer.data(fig1_id).add_points(V1.row(sampled1[i]), Eigen::RowVector3d(r, g, b));
		viewer.data(fig2_id).add_points(V2.row(sampled2[correspondances[i]]), Eigen::RowVector3d(r, g, b));
	}

	//viewer.data().set_mesh(V2, F2);

	
	viewer.launch(); // run the viewer

	

}
