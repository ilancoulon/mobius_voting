#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
using namespace Eigen;
#include <igl/octree.h>
#include <igl/knn.h>
#include <igl/gaussian_curvature.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/jet.h>


void gaussian_curv();
