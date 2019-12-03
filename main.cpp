#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <igl/octree.h>
#include <igl/knn.h>
#include <igl/writeOBJ.h>
#include <iostream>
#include <ostream>
#include "pca.h"
#include "ICP.h"

//using namespace Eigen; // to use the classes provided by Eigen library
MatrixXd V1; // matrix storing vertex coordinates of the input curve
MatrixXi F1;
MatrixXd V2; // matrix storing vertex coordinates of the input curve
MatrixXi F2;


bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier) {
  std::cout << "pressed Key: " << key << " " << (unsigned int)key << std::endl;

  if (key == '1')
  {
    transform(V1,V2);
    viewer.data(0).clear(); // Clear should be called before drawing the mesh
    viewer.data(0).set_mesh(V1, F1);
    viewer.data(0).set_colors(Eigen::RowVector3d(0.3, 0.8, 0.3));// update the mesh (both coordinates and faces)
  }

  return false;
}

void draw_normals(igl::opengl::glfw::Viewer &viewer, const MatrixXd &V, const MatrixXd &n){
  MatrixXd current_edge(V.rows(), 3);
  for (unsigned i = 1; i < V.rows()-1; ++i){
    current_edge(i,0) =  V(i, 0) + n(i, 0);
    current_edge(i,1) =  V(i, 1)+ n(i, 1);
    current_edge(i,2) =  V(i, 2)+ n(i, 2);
    viewer.data().add_edges(
        V.row(i),
        current_edge.row(i),
        Eigen::RowVector3d(1, 1, 1));
    }
}

void set_meshes(igl::opengl::glfw::Viewer &viewer) {
  viewer.callback_key_down = &key_down; // for dealing with keyboard events
  viewer.data().set_mesh(V1, F1);
  viewer.append_mesh();
  viewer.data().set_mesh(V2, F2);
  viewer.data(0).set_colors(Eigen::RowVector3d(0.3, 0.8, 0.3));
  viewer.data(1).set_colors(Eigen::RowVector3d(0.8, 0.3, 0.3));
}

void set_pc(igl::opengl::glfw::Viewer &viewer) {
  viewer.callback_key_down = &key_down; // for dealing with keyboard events
  viewer.data().add_points(V1,Eigen::RowVector3d(0.3, 0.8, 0.3));
}

void ex1() {
  igl::readOFF("../data/gargoyle_tri.off", V1, F1);
  igl::readOFF("../data/gargoyle_tri_rotated.off", V2, F2);
  igl::opengl::glfw::Viewer viewer;
  set_meshes(viewer);
  viewer.launch();
}

void ex2() {
  igl::readOFF("../data/egea.off", V1, F1);
  igl::readOFF("../data/egea_rotated.off", V2, F2);
  igl::opengl::glfw::Viewer viewer;
  set_meshes(viewer);
    viewer.core().is_animating = true;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & )->bool // run animation
    {
      Eigen::MatrixXd nn_V2(V1.rows(), V1.cols());
      nearest_neighbour(V1, V2, nn_V2);

	  double sum = 0;
	  for (size_t i = 0; i < V1.rows(); i++)
	  {
		  sum += (V1.row(i) - nn_V2.row(i)).squaredNorm();
	  }
	  std::cout << sum << std::endl;

      //  complete here by displaying the pair wise distances
      transform(V1,nn_V2);
      viewer.data(0).clear(); // Clear should be called before drawing the mesh
      viewer.data(0).set_mesh(V1, F1);
      viewer.data(0).set_colors(Eigen::RowVector3d(0.3, 0.8, 0.3));// update the mesh (both coordinates and faces)}
      return false; };
      viewer.launch(); // run the editor
}

void ex3(){
  igl::readOFF("../data/sphere.off", V1, F1);
  igl::opengl::glfw::Viewer viewer;
  set_pc(viewer);
  Eigen::MatrixXi I;
  k_nearest_neighbour(V1,I,12);
  Eigen::MatrixXd normals(V1.rows(), 3);
  Eigen::MatrixXd A(V1.rows(), 3);
  Eigen::MatrixXd B(V1.rows(), 3);
  compute_normals(V1,I, 12, normals);
  draw_normals(viewer, V1, normals);
  viewer.launch();
}

int main(int argc, char *argv[])
{
  //ex1();
  //ex2();
  ex3();
}
