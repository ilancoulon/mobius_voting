#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <igl/octree.h>
#include <igl/knn.h>
#include <igl/writeOBJ.h>
#include <igl/file_exists.h>
#include <igl/pathinfo.h>
#include <iostream>
#include <ostream>
#include <list>
#include <nanoflann.ĥpp>

#include "point_sample.h"
//#include "nanoflann/include/nanoflann.hpp"
#include "mid_edge_uniformisation.cpp"
#include "planar_embedding.h"
#include "mobius_transformation.cpp"


MatrixXd V1; 
MatrixXi F1;
MatrixXd V2;
MatrixXi F2;

string rootPath = "../data/";



void nearest_neighbour(const MatrixXd &V1, const MatrixXd &V2, VectorXi &nn){

	nanoflann::KDTreeEigenMatrixAdaptor<Matrix<double, Dynamic, Dynamic>> mat_index(2, std::cref(V2), 10);
	mat_index.index->buildIndex();

	nn = VectorXi::Zero(V1.rows());

	for(int i = 0; i < V1.rows(); i++) {
		std::vector<double> query_pt{ V1(i,0), V1(i,1) };
		vector<size_t> ret_indexes(1);
  		vector<double> out_dists_sqr(1);
		nanoflann::KNNResultSet<double> resultSet(1);
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
		mat_index.index->findNeighbors(resultSet, &query_pt[0],
                                 nanoflann::SearchParams(10));
		nn(i) = ret_indexes[0];
	}
}

VectorXi mobius_voting(MatrixXd V1, MatrixXi F1, MatrixXd V2, MatrixXi F2, int I, int numberToSample, int K, double confidenceLevel, VectorXi &points1, VectorXi &points2, MatrixXd &C)
{

	//Modifies sampled1, sampled2, and C

	VectorXcd planarPoints1;
	VectorXcd planarPoints2;

	////////////////////////////////////////////////
	///////// Planar Embedding for Shape 1 /////////
	////////////////////////////////////////////////

	HalfedgeBuilder *builder1 = new HalfedgeBuilder();
	HalfedgeDS he1 = (builder1->createMeshWithFaces(V1.rows(), F1)); 
	MidEdge *midedge1 = new MidEdge(V1, F1, he1);
	midedge1->makeMidEdge();

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
	cout << " u..." << endl;

	u1 = planar1->u();
	cout << " u*..." << endl;
	u_star1 = planar1->u_star(u1);

	planar1->embedding(u1, u_star1);

	planarPoints1 = planar1->getComplexCoordinates();
	cout << "Done !" << endl;

	/////////////////////////////////////////////////
	///////// Planar Embedding for Shape 2 //////////
	/////////////////////////////////////////////////

	HalfedgeBuilder *builder2 = new HalfedgeBuilder();
	HalfedgeDS he2 = (builder2->createMeshWithFaces(V2.rows(), F2));
	MidEdge *midedge2 = new MidEdge(V2, F2, he2);
	midedge2->makeMidEdge();

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

	cout << " u..." << endl;
	u2 = planar2->u();
	cout << " u*..." << endl;
	u_star2 = planar2->u_star(u2);

	planar2->embedding(u2, u_star2);

	planarPoints2 = planar2->getComplexCoordinates();
	cout << "Done !" << endl;

	//////////////////////////////////////////////////////////
	//////////// TAKE SAMPLE POINTS IN THE MESH //////////////
	//////////////////////////////////////////////////////////

	cout << "Sampling..." << endl;
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

	cout << "Matching sample points in planar embedding..." << endl;

	int count1 = 0;
	int count2 = 0;

	//std::cout << sampled1.rows() << std::endl;
	//std::cout << sampled2.rows() << std::endl;

	int max = sampled2.rows();
	if (sampled1.rows() > sampled2.rows()) max = sampled1.rows();

	for (int i = 0; i < max; i++)
	{
		if (i < sampled1.rows() && sampled1(i) == 1)
		{
			points1(count1) = i;
			int e = he1.getEdge(i);
			int bestEdge = e;

			int currentEdge = he1.getOpposite(he1.getNext(e));
			while (currentEdge != e)
			{

				if ((V1.row(he1.getTarget(he1.getOpposite(currentEdge))) - V1.row(i)).norm() < (V1.row(he1.getTarget(he1.getOpposite(bestEdge))) - V1.row(i)).norm())
				{
					bestEdge = currentEdge;
				}
				currentEdge = he1.getOpposite(he1.getNext(currentEdge));
			}
			sigma1(count1) = planarPoints1(halfpoints1[bestEdge]);
			count1++;
		}
		if (i < sampled2.rows() && sampled2(i) == 1)
		{
			points2(count2) = i;
			int e = he2.getEdge(i);
			int bestEdge = e;

			int currentEdge = he2.getOpposite(he2.getNext(e));
			while (currentEdge != e)
			{

				if ((V2.row(he2.getTarget(he2.getOpposite(currentEdge))) - V2.row(i)).norm() < (V2.row(he2.getTarget(he2.getOpposite(bestEdge))) - V2.row(i)).norm())
				{
					bestEdge = currentEdge;
				}
				currentEdge = he2.getOpposite(he2.getNext(currentEdge));
			}
			sigma2(count2) = planarPoints2(halfpoints2[bestEdge]);
			count2++;
		}
	}

	cout << "Done !";

	///////////////////////////////////////////////
	////////   MOBIUS VOTING ALGORITHM  ///////////
	///////////////////////////////////////////////

	int nbSampled = sigma1.rows();
	C = MatrixXd::Zero(nbSampled, nbSampled);

	for (int iter = 0; iter < I; iter++)
	{

		if (iter % (I / 100) == 0)
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

		VectorXcd complexPointsAfterMobius1;
		VectorXcd complexPointsAfterMobius2;

		complexPointsAfterMobius1 = mobius1->getNewCoordinates();
		complexPointsAfterMobius2 = mobius2->getNewCoordinates();

		MatrixXd pts1 = MatrixXd::Zero(nbSampled, 2);
		MatrixXd pts2 = MatrixXd::Zero(nbSampled, 2);

		for(int i = 0; i<nbSampled; ++i) {
			pts1(i,0) = complexPointsAfterMobius1(i).real();
			pts1(i,1) = complexPointsAfterMobius1(i).imag();
			pts2(i,0) = complexPointsAfterMobius2(i).real();
			pts2(i,1) = complexPointsAfterMobius2(i).imag();
		}

		//std::cout << pts2;
		//std::cout << complexPointsAfterMobius1;
		

		//FIND MUTUALLY CLOSEST NEIGHBORS AND FILL THEM IN "neigh1", "neigh2"

		int nb_neigh = 0;
		list<int> neigh1;
		list<int> neigh2;
		VectorXi nn_1;
		VectorXi nn_2;

		nearest_neighbour(pts1,pts2,nn_1);
		nearest_neighbour(pts2,pts1,nn_2);
		
		//std::cout << nn_1 << std::endl;
		//std::cout << nn_2 << std::endl;

		

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

		double epsilon = 0.1;

		if (nb_neigh > K)
		{
			// Computing the energy...
			double energy = 0.;
			list<int>::iterator it1 = neigh1.begin();
			list<int>::iterator it2 = neigh2.begin();
			while (it1 != neigh1.end())
			{
				energy += abs(complexPointsAfterMobius1[*it1] - complexPointsAfterMobius2[*it2]);
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

	for (size_t i = 0; i < correspondances.rows(); i++)
	{
		correspondances(i) = -1;
	} 

	int foundCorrespondances = 0;

	double realMax = -1.;
	for (size_t i = 0; i < C.rows(); i++)
	{
		for (size_t j = 0; j < C.cols(); j++)
		{
			if (C(i, j) > realMax)
				realMax = C(i, j);
		}
	} 

	//cout << realMax << endl;

	while (foundCorrespondances < nbSampled)
	{
		double maxFound = -1.;
		int rowMax = 0;
		int colMax = 0;
		for (size_t i = 0; i < C.rows(); i++)
		{
			for (size_t j = 0; j < C.cols(); j++)
			{
				if (C(i, j) > maxFound)
				{
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

		for (size_t i = 0; i < C.rows(); i++)
		{
			C(i, colMax) = -1.;
		}
		for (size_t j = 0; j < C.cols(); j++)
		{
			C(rowMax, j) = -1.;
		}

		//cout << maxFound << endl;

		if ((maxFound != -1) && maxFound >= confidenceLevel * realMax)
			correspondances[rowMax] = colMax;
		foundCorrespondances++;
	}

	return correspondances;
}

string getExtensionFromFilename(string mesh_filename) {
	string dirname, basename, extension, filename;
	igl::pathinfo(mesh_filename, dirname, basename, extension, filename);
	std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
	return extension;
}

void doPlanarEmbedding(string figure1) {
	string extension1 = getExtensionFromFilename(rootPath + figure1);
	if (extension1 == "obj") {
		igl::readOBJ(rootPath + figure1, V1, F1);
	}
	else {
		igl::readOFF(rootPath + figure1, V1, F1);
	}


	HalfedgeBuilder* builder1 = new HalfedgeBuilder();
	HalfedgeDS he1 = (builder1->createMeshWithFaces(V1.rows(), F1));
	MidEdge* midedge1 = new MidEdge(V1, F1, he1);
	midedge1->makeMidEdge();

	MatrixXd Vmid1;
	MatrixXi Fmid1;
	int* halfpoints1;

	Vmid1 = midedge1->getVertexCoordinates();
	Fmid1 = midedge1->getFaces();
	halfpoints1 = midedge1->getHalfPoints();

	HalfedgeBuilder* buildermid1 = new HalfedgeBuilder();
	HalfedgeDS hemid1 = (buildermid1->createMeshWithFaces(Vmid1.rows(), Fmid1));

	cout << "Planar Embedding for Shape ...";
	PlanarEmbedding* planar1 = new PlanarEmbedding(V1, F1, he1, Vmid1, Fmid1, hemid1, halfpoints1);

	VectorXd u1;
	VectorXd u_star1;
	cout << " u...";

	u1 = planar1->u();
	cout << " u*...";
	u_star1 = planar1->u_star(u1);

	planar1->embedding(u1, u_star1);

	cout << "Done !" << endl;

	MatrixXd vertexCoordinates = planar1->getVertexCoordinates();
	MatrixXd V_embedded = MatrixXd::Zero(vertexCoordinates.rows(), 3);
	for (size_t i = 0; i < V_embedded.rows(); i++)
	{
		V_embedded.row(i) << vertexCoordinates(i, 0), vertexCoordinates(i, 1), 0.;
	}

	igl::opengl::glfw::Viewer viewer;
	viewer.load_mesh_from_file(rootPath + "star.off");
	viewer.load_mesh_from_file(rootPath + "star.off");

	unsigned int left_view, right_view;
	int fig1_id = viewer.data_list[0].id;
	int fig2_id = viewer.data_list[1].id;
	viewer.data(fig1_id).clear();
	viewer.data(fig2_id).clear();
	viewer.data(fig1_id).set_mesh(V1, F1);
	viewer.data(fig2_id).set_mesh(V_embedded, Fmid1);
	viewer.callback_init = [&](igl::opengl::glfw::Viewer&) {
		viewer.core().viewport = Eigen::Vector4f(0, 0, 640, 800);
		left_view = viewer.core_list[0].id;
		right_view = viewer.append_core(Eigen::Vector4f(640, 0, 640, 800));
		viewer.core(left_view).align_camera_center(V1, F1);
		viewer.core(right_view).align_camera_center(V2, F2);
		viewer.data(fig1_id).set_visible(false, right_view);
		viewer.data(fig2_id).set_visible(false, left_view);
		return false;
	};
	viewer.launch();
}

void doMobiusVoting(string figure1, string figure2) {
	// Reading 1st file
	string extension1 = getExtensionFromFilename(rootPath + figure1);
	if (extension1 == "obj") {
		igl::readOBJ(rootPath + figure1, V1, F1);
	}
	else {
		igl::readOFF(rootPath + figure1, V1, F1);
	}

	// Reading 2nd file
	string extension2 = getExtensionFromFilename(rootPath + figure2);
	if (extension1 == "obj") {
		igl::readOBJ(rootPath + figure2, V2, F2);
	}
	else {
		igl::readOFF(rootPath + figure2, V2, F2);
	}


	VectorXi points1;
	VectorXi points2;

	MatrixXd C;

	VectorXi correspondances;

	correspondances = mobius_voting(V1, F1, V2, F2, 12000, 20, 25, 0.3, points1, points2, C);

	//std::cout << points1 << std::endl;
	//std::cout << points2 << std::endl;
	//std::cout << C << std::endl;
	//std::cout << correspondances << std::endl;

	igl::opengl::glfw::Viewer viewer; 


	viewer.load_mesh_from_file(rootPath+"star.off");
	viewer.load_mesh_from_file(rootPath+"star.off");

	unsigned int left_view, right_view;
	int fig1_id = viewer.data_list[0].id;
	int fig2_id = viewer.data_list[1].id;
	viewer.data(fig1_id).clear();
	viewer.data(fig2_id).clear();
	viewer.data(fig1_id).set_mesh(V1,F1);
	viewer.data(fig2_id).set_mesh(V2,F2);
	viewer.callback_init = [&](igl::opengl::glfw::Viewer &) {
		viewer.core().viewport = Eigen::Vector4f(0, 0, 1280, 1600);
		left_view = viewer.core_list[0].id;
		right_view = viewer.append_core(Eigen::Vector4f(1280, 0, 1280, 1600));
		viewer.core(left_view).align_camera_center(V1, F1);
		viewer.core(right_view).align_camera_center(V2, F2);
		viewer.data(fig1_id).set_visible(false, right_view);
		viewer.data(fig2_id).set_visible(false, left_view);
		return false;
	};

	for (size_t i = 0; i < correspondances.rows(); i++)
	{
		if (correspondances(i) != -1) {
		float r = (float)std::rand() / (float)RAND_MAX;
		float g = (float)std::rand() / (float)RAND_MAX;
		float b = (float)std::rand() / (float)RAND_MAX;

		viewer.data(fig1_id).add_points(V1.row(points1[i]), Eigen::RowVector3d(r, g, b));
		viewer.data(fig2_id).add_points(V2.row(points2[correspondances[i]]), Eigen::RowVector3d(r, g, b));
		}
	}

	viewer.launch(); 

}

int main(int argc, char *argv[])
{
	

	//string figure1 = "bunny.off";
	//string figure2 = "bunny_rotated.off";

	int toCall;

	std::cout << "Please type 1 to try Planar Embedding, type 2 to try Möbius Voting"  << std::endl;
	std::cin >> toCall;
	std::cin.clear();
	std::cin.ignore(10, '\n');

	if (toCall == 1) {
		string figure1;
		string defaultFigure = "bunny.off";

		do
		{
			std::cout << "Enter figure path... [" << defaultFigure << "] ";

			
			getline(std::cin, figure1);
			if (figure1.empty()) {
				figure1 = defaultFigure;
			}

			if (!igl::file_exists(rootPath + figure1)) {
				std::cout << figure1 << " does not exist!" << std::endl;
			}
		} while (!igl::file_exists(rootPath + figure1));

		doPlanarEmbedding(figure1);

	}
	else  {

		string figure1;
		string defaultFigure1 = "SHREC15/test/0.obj";
		string figure2;
		string defaultFigure2 = "SHREC15/test/1.obj";

		do
		{
			std::cout << "Enter figure 1 path... [" << defaultFigure1 << "] ";

			getline(std::cin, figure1);
			if (figure1.empty()) {
				figure1 = defaultFigure1;
			}

			if (!igl::file_exists(rootPath + figure1)) {
				std::cout << figure1 << " does not exist!" << std::endl;
			}
		} while (!igl::file_exists(rootPath+figure1));
		

		do
		{
			std::cout << "Enter figure 2 path... [" << defaultFigure2 << "] ";

			getline(std::cin, figure2);
			if (figure2.empty()) {
				figure2 = defaultFigure2;
			}


			if (!igl::file_exists(rootPath + figure2)) {
				std::cout << figure2 << " does not exist!" << std::endl;
			}
		} while (!igl::file_exists(rootPath + figure2));



		doMobiusVoting(figure1, figure2);
	}

}
