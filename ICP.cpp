#include "ICP.h"
#include <iostream>

void nearest_neighbour(const MatrixXd &V1, const MatrixXd &V2, MatrixXd &nn_V2){
	nn_V2 = MatrixXd::Zero(V1.rows(), V1.cols());

	std::vector<std::vector<int > > O_PI;
	Eigen::MatrixXi O_CH;
	Eigen::MatrixXd O_CN;
	Eigen::VectorXd O_W;
	igl::octree(V2, O_PI, O_CH, O_CN, O_W);

	Eigen::MatrixXi I;

	igl::knn(V1, 1, O_PI, O_CH, O_CN, O_W, I);

	for (size_t i = 0; i < I.rows(); i++)
	{
		nn_V2.row(i) = V2.row(I(i, 0));
	}

	// Bruteforce
	/*for (int i = 0; i < V1.rows(); i++) {
		int nearestIndex = 0;
		double minNorm = (V2.row(nearestIndex) - V1.row(i)).squaredNorm();
		for (int j = 0; j < V2.rows(); j++) {
			double norm = (V2.row(j) - V1.row(i)).squaredNorm();
			if (norm < minNorm) {
				nearestIndex = j;
				minNorm = norm;
			}
		}
		nn_V2.row(i) = V2.row(nearestIndex);
	}*/
}


void transform(MatrixXd &V1,const MatrixXd &V2){
	int n = V1.rows();

	Vector3d mean1 = V1.colwise().mean();

	Vector3d mean2 = V2.colwise().mean();

	//Rotation

	MatrixXd C = MatrixXd::Zero(3, 3);
	for (int i = 0; i < n; i++) {
		C = C + (V2.row(i) - mean2.transpose()).transpose() * (V1.row(i) - mean1.transpose());
	}

	JacobiSVD<MatrixXd> svd(C, ComputeThinU | ComputeThinV);
	MatrixXd R(3, 3);

	float det = (svd.matrixU() * svd.matrixV().transpose()).determinant();
	if (det == 1) {
		R = svd.matrixU() * svd.matrixV().transpose();
	}
	else {
		MatrixXd diag = MatrixXd::Identity(3, 3);
		diag(2, 2) = -1;
		R = svd.matrixU() * diag * svd.matrixV().transpose();
	}

	for (int i = 0; i < n; i++) {
		V1.row(i) = (R * V1.row(i).transpose()).transpose();
	}

	// Translation

	Vector3d translation = mean2 - mean1;

	for (int i = 0; i < n; i++) {
		V1.row(i) = V1.row(i) + translation.transpose();
	}
}
