#include "pca.h"


void k_nearest_neighbour(const MatrixXd& V1, Eigen::MatrixXi& I, int k) {
	std::vector<std::vector<int > > O_PI;
	Eigen::MatrixXi O_CH;
	Eigen::MatrixXd O_CN;
	Eigen::VectorXd O_W;
	igl::octree(V1, O_PI, O_CH, O_CN, O_W);

	igl::knn(V1, k, O_PI, O_CH, O_CN, O_W, I);
}



void compute_normals(const MatrixXd& V1, const Eigen::MatrixXi& I, int k, MatrixXd& normals) {
	MatrixXd vectsToKnn = MatrixXd::Zero(k, 3);

	for (int i = 0; i < V1.rows(); i++) {
		for (int j = 0; j < k; j++) {
			vectsToKnn.row(j) = V1.row(I(i, j)) - V1.row(i);
		}

		MatrixXd cov = vectsToKnn.transpose() * vectsToKnn;

		// Eigenvector of the smallest eigenvalue

		EigenSolver<MatrixXd> solv(cov);
		int indexMax = 0;
		for (int j = 0; j < solv.eigenvalues().size(); j++) {
			if (solv.eigenvalues()[j].real() <= solv.eigenvalues()[indexMax].real()) {
				indexMax = j;
			}
		}

		VectorXcd eigenv = solv.eigenvectors().col(indexMax);
		VectorXd result_i = VectorXd::Zero(3);
		result_i(0) = eigenv(0).real();
		result_i(1) = eigenv(1).real();
		result_i(2) = eigenv(2).real();
		result_i.normalize();

		normals.row(i) = result_i;
	}
}
