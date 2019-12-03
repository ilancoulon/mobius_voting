
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
using namespace Eigen;
#include <igl/octree.h>
#include <igl/knn.h>

void nearest_neighbour(const MatrixXd &V1, const MatrixXd &V2, MatrixXd &nn_V2);
void transform(MatrixXd &V1,const MatrixXd &V2);
